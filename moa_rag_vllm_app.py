"""
Mixture-of-Agents (MoA) RAG server using two Qwen2.5 models behind vLLM.

- Agents:
  * Qwen2.5-7B-Instruct (generalist)
  * Qwen2.5-Coder-7B-Instruct (code specialist)
- Retrieval: FAISS + sentence-transformers (BAAI/bge-m3) by default
- Server: FastAPI
- Backend LLMs: vLLM / llama.cpp OpenAI-compatible servers (one per model)

Runbook (quick):
1) Start two OpenAI-compatible servers (vLLM or llama.cpp) on ports 8001 and 8002.
2) pip install: fastapi uvicorn httpx sentence-transformers faiss-cpu
3) python moa_rag_vllm_app.py  → API at http://127.0.0.1:9000
4) POST /ingest with paths to index docs.
5) POST /chat with {"query": "..."} to talk (Discord bridge calls this).
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional server runtime
    import uvicorn
except ImportError:  # pragma: no cover - not needed for tests
    uvicorn = None  # type: ignore

try:  # pragma: no cover
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore

    class HTTPException(RuntimeError):
        def __init__(self, status_code: int, detail: str | None = None):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail or f"HTTP {status_code}")

from pydantic import BaseModel, Field

import httpx
import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# -----------------------------
# Config
# -----------------------------

OCR_AVAILABLE = False
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    pass


class Settings(BaseModel):
    general_base_url: str = Field(
        default=os.environ.get("GENERAL_VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    )
    coder_base_url: str = Field(
        default=os.environ.get("CODER_VLLM_BASE_URL", "http://127.0.0.1:8002/v1")
    )
    api_key: str = Field(default=os.environ.get("OPENAI_API_KEY", ""))  # only sent if non-empty
    embed_model_name: str = Field(default=os.environ.get("EMBED_MODEL", "BAAI/bge-m3"))
    index_dir: str = Field(default=os.environ.get("INDEX_DIR", "./index_store"))
    chunk_size: int = Field(default=900)  # chars heuristic (token-ish)
    chunk_overlap: int = Field(default=200)
    top_k: int = Field(default=5)
    max_context_chars: int = Field(default=8000)
    router_code_keywords: List[str] = Field(default_factory=lambda: [
        'python','javascript','typescript','rust','c++','c#','java','go','bash','shell','regex','stack trace',
        'error:','exception','traceback','compile','package.json','requirements.txt','pip','npm','yarn','pnpm',
        'gradle','maven','cargo','poetry','pytest','unittest','tsconfig','webpack','vite','docker','dockerfile',
        'kubernetes','yaml','json','toml','ini','makefile','cmake','llvm','linker','gcc','clang'
    ])
    router_threshold: float = Field(default=0.35)  # 0..1 scale
    judge_merge: bool = Field(default=False)       # set True to ensemble both agents


SET = Settings()
Path(SET.index_dir).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""


def simple_chunk(text: str, size: int, overlap: int) -> List[str]:
    out = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i:i+size])
        i += max(1, size - overlap)
    return out

# -----------------------------
# Vector store
# -----------------------------

class VectorStore:
    """Embeddings-backed retrieval with optional FAISS acceleration."""

    def __init__(self, index_dir: str, model_name: str, model: SentenceTransformer | None = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.index_dir / "meta.jsonl"
        self.index_path = self.index_dir / ("faiss.index" if FAISS_AVAILABLE else "vectors.npy")
        self.model_name = model_name
        self._model = model
        self._dim: int | None = None
        self.index = None  # type: ignore[assignment]
        self._vectors: np.ndarray | None = None
        self.metas: List[Dict[str, Any]] = []
        if self.meta_path.exists():
            self._load()

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("Install sentence-transformers to use the default embedder or pass a custom model.")
            self._model = SentenceTransformer(self.model_name)
        if self._dim is None:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _ensure_index(self):
        if self._dim is None:
            raise RuntimeError("Vector dimension unknown; call _ensure_model first")
        if FAISS_AVAILABLE:
            if self.index is None:
                self.index = faiss.IndexFlatIP(self._dim)
        else:
            if self._vectors is None:
                self._vectors = np.empty((0, self._dim), dtype=np.float32)

    def _load(self):
        if FAISS_AVAILABLE and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self._dim = int(self.index.d)
        elif not FAISS_AVAILABLE and self.index_path.exists():
            arr = np.load(self.index_path, allow_pickle=False)
            if arr.ndim == 1:
                arr = np.reshape(arr, (0, 0))
            self._vectors = arr.astype(np.float32)
            if self._vectors.size:
                self._dim = int(self._vectors.shape[1])
        with contextlib.suppress(FileNotFoundError):
            with self.meta_path.open('r', encoding='utf-8') as f:
                self.metas = [json.loads(line) for line in f]

    def _save(self):
        if FAISS_AVAILABLE:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
        else:
            if self._vectors is not None:
                np.save(str(self.index_path), self._vectors)
        with self.meta_path.open('w', encoding='utf-8') as f:
            for m in self.metas:
                f.write(json.dumps(m) + "\n")

    def add_texts(self, docs: List[Tuple[str, Dict[str, Any]]]):
        if not docs:
            return
        model = self._ensure_model()
        self._ensure_index()
        texts = [d[0] for d in docs]
        embeds = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        if hasattr(embeds, 'size') and embeds.size == 0:
            return
        embeds = np.asarray(embeds, dtype=np.float32)
        if FAISS_AVAILABLE:
            self.index.add(embeds)
        else:
            if self._vectors is None or self._vectors.size == 0:
                self._vectors = embeds
            else:
                self._vectors = np.vstack([self._vectors, embeds])
        self.metas.extend([{**d[1]} for d in docs])
        self._save()

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if len(self.metas) == 0:
            return []
        model = self._ensure_model()
        q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        if hasattr(q, 'size') and q.size == 0:
            return []
        q = np.asarray(q, dtype=np.float32)
        k = min(k, len(self.metas))
        if k <= 0:
            return []
        if FAISS_AVAILABLE:
            self._ensure_index()
            distances, neighbors = self.index.search(q, k)
            indices = neighbors[0].tolist()
            scores = distances[0].tolist()
        else:
            if self._vectors is None or self._vectors.size == 0:
                return []
            sims = self._vectors @ q[0]
            top = np.argsort(sims)[::-1][:k]
            indices = top.tolist()
            scores = sims[top].astype(float).tolist()
        results = []
        for idx, score in zip(indices, scores, strict=False):
            if idx < 0 or idx >= len(self.metas):
                continue
            meta = self.metas[idx].copy()
            meta['score'] = float(score)
            results.append(meta)
        return results



VECTOR = VectorStore(SET.index_dir, SET.embed_model_name)

# -----------------------------
# DeepSeek OCR wrapper (optional)
# -----------------------------

class DeepSeekOCR:
    """Lightweight wrapper around deepseek-ai/DeepSeek-OCR via Transformers.
    Converts images/PDFs to Markdown/plain text and returns text chunks that we can ingest.
    """
    def __init__(self, device: str = 'cuda', dtype: str = 'bfloat16'):
        if not OCR_AVAILABLE:
            raise RuntimeError("Install transformers, torch, pillow to enable OCR")
        self.model_name = os.environ.get('DEEPSEEK_OCR_MODEL', 'deepseek-ai/DeepSeek-OCR')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().to(device)
        if dtype == 'bfloat16':
            self.model = self.model.to(torch.bfloat16)
        elif dtype == 'float16':
            self.model = self.model.to(torch.float16)
        self.base_size = int(os.environ.get('DEEPSEEK_OCR_BASE_SIZE', '1024'))
        self.image_size = int(os.environ.get('DEEPSEEK_OCR_IMAGE_SIZE', '640'))

    def ocr_image(self, img_path: str, mode: str = 'markdown') -> str:
        prompt = "<image>\n<|grounding|>Convert the document to markdown." if mode == 'markdown' else "<image>\nFree OCR."
        res = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=img_path,
            output_path=None,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=True,
            save_results=False,
            test_compress=True,
        )
        text = res.get('md', None) or res.get('text', '') or res
        if isinstance(text, dict):
            text = text.get('text', '')
        return str(text)

    def _pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[str]:
        tmp_dir = Path(SET.index_dir) / 'pdf_pages'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_paths: List[str] = []
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=dpi)
            for i, page in enumerate(pages):
                out = tmp_dir / f"{sha1(pdf_path)}_p{i+1}.png"
                page.save(out)
                out_paths.append(str(out))
            return out_paths
        except Exception:
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(pdf_path)
                for i in range(len(pdf)):
                    page = pdf[i]
                    pil = page.render(scale=dpi/72).to_pil()
                    out = tmp_dir / f"{sha1(pdf_path)}_p{i+1}.png"
                    pil.save(out)
                    out_paths.append(str(out))
                return out_paths
            except Exception as e:
                raise RuntimeError("Install pdf2image or pypdfium2 to OCR PDFs") from e

    def ocr_pdf(self, pdf_path: str, mode: str = 'markdown') -> str:
        imgs = self._pdf_to_images(pdf_path)
        texts = []
        for img in imgs:
            texts.append(self.ocr_image(img, mode=mode))
        return "\n\n".join(texts)


# Lazy singleton for OCR
@lru_cache(maxsize=1)
def get_ocr() -> Optional[DeepSeekOCR]:
    try:
        return DeepSeekOCR()
    except Exception:
        return None

# -----------------------------
# vLLM / llama.cpp Client (OpenAI-compatible)
# -----------------------------

class VLLMClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or ""
        self._client = httpx.Client(timeout=60.0)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        headers: Dict[str, str] = {}
        if self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": "placeholder",  # ignored by most local servers
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 0.9),
        }
        r = self._client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"vLLM error {r.status_code}: {r.text}")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data, indent=2)


GENERAL = VLLMClient(SET.general_base_url, SET.api_key)
CODER = VLLMClient(SET.coder_base_url, SET.api_key)

# -----------------------------
# Router / MoA logic
# -----------------------------

def code_score(prompt: str, keywords: List[str]) -> float:
    p = prompt.lower()
    hits = sum(1 for kw in keywords if kw in p)
    return min(1.0, hits / 10.0)


def build_context(docs: List[Dict[str, Any]], max_chars: int) -> str:
    buf = []
    used = 0
    for d in docs:
        chunk = d.get('text', '')
        src = d.get('source', 'unknown')
        snippet = f"[Source: {src}]\n{chunk}\n\n"
        if used + len(snippet) > max_chars:
            remain = max_chars - used
            if remain <= 0:
                break
            snippet = snippet[:remain]
        buf.append(snippet)
        used += len(snippet)
        if used >= max_chars:
            break
    return "".join(buf)


def moa_generate(query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    cscore = code_score(query, SET.router_code_keywords)
    ctx = build_context(retrieved, SET.max_context_chars)

    system = (
        "You are part of a Mixture-of-Agents system. Another agent may also answer. "
        "Use the provided CONTEXT to ground your answer. If the answer is in the context, cite short source names inline like [source]. "
        "If not in context, be honest and reason concisely."
    )
    user = f"CONTEXT:\n{ctx}\n---\nUSER QUESTION: {query}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    primary = CODER if cscore >= SET.router_threshold else GENERAL
    secondary = GENERAL if primary is CODER else CODER

    primary_out = primary.chat(messages)

    if not SET.judge_merge:
        return {
            "route": "coder" if primary is CODER else "general",
            "code_score": cscore,
            "answer": primary_out,
            "retrieved": retrieved,
        }

    secondary_out = secondary.chat(messages)

    def overlap_score(ans: str, query: str, docs: List[Dict[str, Any]]) -> float:
        ans_l = ans.lower()
        q_terms = set(t for t in query.lower().split() if len(t) > 3)
        d_terms = set()
        for d in docs:
            for t in d.get('text', '').lower().split():
                if len(t) > 5:
                    d_terms.add(t)
        score = sum(1 for t in q_terms if t in ans_l) + 0.25 * sum(1 for t in list(d_terms)[:200] if t in ans_l)
        return float(score)

    pscore = overlap_score(primary_out, query, retrieved)
    sscore = overlap_score(secondary_out, query, retrieved)

    if pscore >= sscore:
        picked = primary_out
        route = "coder" if primary is CODER else "general"
    else:
        picked = secondary_out
        route = "general" if primary is CODER else "coder"

    return {
        "route": route,
        "code_score": cscore,
        "answer": picked,
        "alt": {
            "primary": primary_out,
            "secondary": secondary_out,
            "primary_score": pscore,
            "secondary_score": sscore,
        },
        "retrieved": retrieved,
    }

# -----------------------------
# FastAPI app
# -----------------------------

class IngestReq(BaseModel):
    paths: List[str] = Field(..., description="Files or globs (txt, md, rst, json, py, etc.)")
    glob_recursive: bool = Field(default=True)
    min_chars: int = Field(default=50)

class OCRIngestReq(BaseModel):
    paths: List[str] = Field(..., description="Images (png/jpg) and/or PDFs; globs allowed")
    glob_recursive: bool = Field(default=True)
    mode: str = Field(default="markdown", description="'markdown' or 'text'")
    chunk_size: int = Field(default=SET.chunk_size)
    chunk_overlap: int = Field(default=SET.chunk_overlap)

class ChatReq(BaseModel):
    query: str
    k: int = Field(default=SET.top_k)
    temperature: float = 0.7
    max_tokens: int = 512

def ingest(req: IngestReq):
    files: List[str] = []
    for p in req.paths:
        if any(ch in p for ch in ["*", "?", "["]):
            files.extend(glob.glob(p, recursive=req.glob_recursive))
        else:
            files.append(p)
    files = [f for f in files if Path(f).is_file()]
    if not files:
        raise HTTPException(status_code=400, detail="No files matched.")

    added = 0
    for f in files:
        path = Path(f)
        raw = read_text_file(path)
        if len(raw) < req.min_chars:
            continue
        chunks = simple_chunk(raw, SET.chunk_size, SET.chunk_overlap)
        docs = []
        for i, ch in enumerate(chunks):
            meta = {
                "id": sha1(f"{path}::{i}"),
                "source": path.name,
                "path": str(path.resolve()),
                "text": ch,
            }
            docs.append((ch, meta))
        VECTOR.add_texts(docs)
        added += len(chunks)

    return {"status": "ok", "files": len(files), "chunks_added": added}

def ocr_ingest(req: OCRIngestReq):
    ocr = get_ocr()
    if ocr is None:
        raise HTTPException(status_code=500, detail="DeepSeek OCR not available. Install transformers, torch (CUDA), pillow, and optionally pdf2image/pypdfium2.")

    files: List[str] = []
    for p in req.paths:
        if any(ch in p for ch in ["*", "?", "["]):
            files.extend(glob.glob(p, recursive=req.glob_recursive))
        else:
            files.append(p)
    files = [f for f in files if Path(f).is_file() and Path(f).suffix.lower() in {'.png','.jpg','.jpeg','.pdf'}]
    if not files:
        raise HTTPException(status_code=400, detail="No matching images or PDFs.")

    added = 0
    for f in files:
        path = Path(f)
        if path.suffix.lower() == '.pdf':
            text = ocr.ocr_pdf(str(path), mode=req.mode)
        else:
            text = ocr.ocr_image(str(path), mode=req.mode)
        chunks = simple_chunk(text, req.chunk_size, req.chunk_overlap)
        docs = []
        for i, ch in enumerate(chunks):
            meta = {
                "id": sha1(f"{path}::ocr::{i}"),
                "source": path.name,
                "path": str(path.resolve()),
                "text": ch,
                "ocr": True,
            }
            docs.append((ch, meta))
        VECTOR.add_texts(docs)
        added += len(chunks)

    return {"status": "ok", "files": len(files), "chunks_added": added}

# -------- Project Echo entrypoint --------

def _emotion_tag(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["error", "failed", "cannot", "bug", "exception"]):
        return "frustrated"
    if "!" in text and len(text) < 200:
        return "excited"
    if any(w in t for w in ["let’s try", "we can", "here’s how", "works like", "you can"]):
        return "confident"
    return "calm"

def chat(req: ChatReq):
    hits = VECTOR.search(req.query, req.k)
    out = moa_generate(req.query, hits)
    answer = out["answer"]
    return {
        "route": out["route"],
        "code_score": out["code_score"],
        "answer": answer,
        "emotion": _emotion_tag(answer),
        "retrieved": [{"source": r.get("source"), "score": r.get("score")} for r in hits],
    }
if FastAPI is not None:
    app = FastAPI(title="MoA RAG with Qwen2.5 (vLLM/llama.cpp)")
    app.post("/ingest")(ingest)
    app.post("/ocr_ingest")(ocr_ingest)
    app.post("/chat")(chat)
else:
    app = None  # type: ignore



# -----------------------------
# Local Runner
# -----------------------------

if __name__ == "__main__":
    if uvicorn is None or FastAPI is None or app is None:
        raise SystemExit("Install fastapi and uvicorn to run the API server.")
    print("MoA RAG server starting on http://127.0.0.1:9000 ...")
    print("DeepSeek OCR:", "enabled" if get_ocr() else "unavailable (install extras)")
    uvicorn.run(app, host="127.0.0.1", port=9000)
