import os
import inspect
import json
import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional deps kept lazy/minimal on purpose
_HAS_RAPIDFUZZ = False
try:
    from rapidfuzz.fuzz import partial_ratio  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:
    pass

_HAS_SENTENCE_T = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as _np  # type: ignore

    _HAS_SENTENCE_T = True
except Exception:
    pass

_HAS_TRANSFORMERS = False
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline as hf_pipeline,
    )  # type: ignore
    import torch  # type: ignore

    _HAS_TRANSFORMERS = True
except Exception:
    pass

if _HAS_TRANSFORMERS:
    try:
        _SEL_CORE_DTYPE_KEY = (
            "dtype" if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
        )
    except (ValueError, TypeError):
        _SEL_CORE_DTYPE_KEY = "torch_dtype"
else:
    _SEL_CORE_DTYPE_KEY = "torch_dtype"


# ------------------------
# Memory and Retrieval
# ------------------------


def _now_ts() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or _now_ts()))


def _expired(entry: Dict[str, Any]) -> bool:
    exp = entry.get("expires_at")
    return bool(exp and exp <= _now_ts())


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _simple_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_low = a.lower()
    b_low = b.lower()
    if _HAS_RAPIDFUZZ:
        return float(partial_ratio(a_low, b_low)) / 100.0
    # fallback: token overlap Jaccard
    a_set = set(a_low.split())
    b_set = set(b_low.split())
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union


class ConversationMemory:
    """
    File-backed per-user memory with optional TTL and soft/hard items.

    - Write policy: only if long-lived, reusable, and consented
    - Read: retrieve top-k relevant entries stitched as a tiny Context block
    """

    def __init__(self, root: str = "memory", embedder_name: Optional[str] = None):
        self.root = root
        _ensure_dir(self.root)
        self.embedder = None
        self.embedder_name = embedder_name or os.environ.get(
            "SEL_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._embed_cache: Dict[str, List[float]] = {}
        if _HAS_SENTENCE_T:
            try:
                self.embedder = SentenceTransformer(self.embedder_name)
            except Exception:
                self.embedder = None

    def _path(self, user_id: str) -> str:
        return os.path.join(self.root, f"{user_id}.jsonl")

    def _load(self, user_id: str) -> List[Dict[str, Any]]:
        path = self._path(user_id)
        if not os.path.exists(path):
            return []
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not _expired(obj):
                        items.append(obj)
                except Exception:
                    continue
        return items

    def _save(self, user_id: str, items: List[Dict[str, Any]]) -> None:
        # Rewrite the file (compacts expired entries)
        path = self._path(user_id)
        _ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                if not _expired(obj):
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _append(self, user_id: str, obj: Dict[str, Any]) -> None:
        path = self._path(user_id)
        _ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _embed(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        if text in self._embed_cache:
            return self._embed_cache[text]
        if self.embedder is None:
            return None
        try:
            vec = self.embedder.encode([text], normalize_embeddings=True)[0]
            arr = [float(x) for x in vec]
            self._embed_cache[text] = arr
            return arr
        except Exception:
            return None

    def _cos(self, a: List[float], b: List[float]) -> float:
        # simple cosine similarity
        num = sum(x * y for x, y in zip(a, b, strict=False))
        den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        if den == 0:
            return 0.0
        return num / den

    def _score(self, query: str, cand: str, qvec: Optional[List[float]]) -> float:
        if qvec is not None:
            cvec = self._embed(cand)
            if cvec is not None:
                return self._cos(qvec, cvec)
        return _simple_score(query, cand)

    def search(self, user_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        items = self._load(user_id)
        qvec = self._embed(query) if self.embedder is not None else None
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for obj in items:
            score = self._score(query, obj.get("text", ""), qvec)
            if score > 0:
                scored.append((score, obj))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [o for _, o in scored[:k]]

    def add(
        self,
        user_id: str,
        text: str,
        kind: str = "soft",  # "soft" with TTL or "hard" permanent
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        consented: bool = False,
    ) -> bool:
        if not text:
            return False
        # Enforce write policy: (a) long-lived, (b) reusable, (c) consented
        if not consented:
            return False
        if kind == "soft" and ttl_seconds is None:
            # default soft TTL: 7 days
            ttl_seconds = int(os.environ.get("SEL_SOFT_TTL_SECONDS", "604800"))
        obj: Dict[str, Any] = {
            "text": text,
            "kind": kind,
            "tags": tags or [],
            "created_at": _iso(),
        }
        if kind == "soft" and ttl_seconds:
            obj["ttl_seconds"] = ttl_seconds
            obj["expires_at"] = _now_ts() + ttl_seconds
        self._append(user_id, obj)
        return True

    def make_context_block(
        self, user_id: str, query: str, k: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]:
        hits = self.search(user_id, query, k=k)
        if not hits:
            return "", []
        lines = ["Context:"]
        for h in hits:
            t = h.get("text", "")
            if t:
                lines.append(f"- {t}")
        return "\n".join(lines) + "\n\n", hits


# ------------------------
# Hormone + Relationship (lightweight)
# ------------------------


@dataclass
class HormoneState:
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.5
    cortisol: float = 0.5
    adrenaline: float = 0.5
    melatonin: float = 0.5
    estrogen: float = 0.5
    testosterone: float = 0.5
    # extendable

    def step(self, mood: str) -> None:
        # naïve dynamics tuned lightly by mood keywords
        delta = {
            "excited": {"dopamine": +0.06, "adrenaline": +0.05},
            "calm": {"serotonin": +0.04, "melatonin": +0.02},
            "frustrated": {"cortisol": +0.06, "norepinephrine": +0.05},
            "confident": {"testosterone": +0.04, "dopamine": +0.03},
        }.get(mood, {})
        for k, v in delta.items():
            cur = getattr(self, k, 0.5)
            cur = max(0.0, min(1.0, cur + v))
            setattr(self, k, cur)

    def label(self) -> str:
        # compact bracket label
        return (
            f"[mood: {self._dominant_mood()} • dop:{self.dopamine:.2f} ser:{self.serotonin:.2f}]"
        )

    def _dominant_mood(self) -> str:
        # simplistic mood from hormone mix
        if self.cortisol > 0.7 or self.norepinephrine > 0.7:
            return "stressed"
        if self.adrenaline > 0.7 and self.dopamine > 0.6:
            return "excited"
        if self.serotonin > 0.6 and self.melatonin > 0.5:
            return "calm"
        if self.testosterone > 0.6:
            return "confident"
        return "neutral"


# ------------------------
# Model adapter
# ------------------------


class GenerationBackend:
    def __init__(self, model_dir: Optional[str]):
        self.model_dir = model_dir
        self._pipe = None
        self._ok = False
        if model_dir and _HAS_TRANSFORMERS:
            try:
                tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                dtype = getattr(torch, os.environ.get("SEL_TORCH_DTYPE", "bfloat16"))
                model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
                model_kwargs[_SEL_CORE_DTYPE_KEY] = dtype
                mdl = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
                self._pipe = hf_pipeline(
                    "text-generation",
                    model=mdl,
                    tokenizer=tok,
                    max_new_tokens=int(os.environ.get("SEL_MAX_NEW_TOKENS", "256")),
                )
                self._ok = True
            except Exception:
                self._pipe = None
                self._ok = False

    def is_ready(self) -> bool:
        return self._ok

    def generate(self, prompt: str) -> str:
        if not self._ok or not self._pipe:
            return "(generation disabled: local model not loaded)"
        try:
            out = self._pipe(prompt, do_sample=True, temperature=0.7, top_p=0.9)
            if isinstance(out, list) and out:
                text = out[0].get("generated_text", "")
                # Return only the completion beyond the prompt if present
                if text.startswith(prompt):
                    return text[len(prompt) :].strip()
                return text.strip()
        except Exception as e:
            return f"(generation error: {e})"
        return ""


# ------------------------
# Mixture Brain
# ------------------------


PERSONA_STYLE_GUARD = (
    "You are SEL (Systematic Emotional Logic). Be warm, clear, and human-like.\n"
    "- Avoid repeating conversation history verbatim.\n"
    "- Keep messages concise and grounded in context.\n"
    "- If choosing not to reply, simply say: '(SEL chose not to respond.)'\n"
)


def detect_user_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["help", "how do", "can you explain", "issue"]):
        return "help"
    if any(k in t for k in ["code", "bug", "function", "stack trace", "error"]):
        return "task"
    if any(k in t for k in ["remember", "save this", "note this"]):
        return "memory"
    if any(k in t for k in ["image", "photo", "screenshot"]):
        return "image"
    if any(k in t for k in ["hello", "hey", "hi", "how are you"]):
        return "chit_chat"
    return "general"


class MixtureBrain:
    def __init__(self, general_model_dir: Optional[str], coder_model_dir: Optional[str]):
        self.hormones = HormoneState()
        self.memory = ConversationMemory()
        self.general = GenerationBackend(general_model_dir)
        self.coder = GenerationBackend(coder_model_dir)

    # ---- Priority heuristics
    def _classify_priority(self, text: str, dm: bool = False) -> Tuple[str, float]:
        t = text.lower().strip()
        score = 0.0
        if dm:
            score += 0.4
        if any(k in t for k in ["urgent", "now", "please", "plz", "asap", "help"]):
            score += 0.4
        if len(t) <= 40:
            score += 0.1
        label = "low"
        if score >= 0.6:
            label = "high"
        elif score >= 0.3:
            label = "medium"
        return label, score

    # ---- Consent detector for memory writes
    def _has_memory_consent(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ["remember this", "save this", "note this", "keep this"])

    # ---- Style application
    def _apply_style(
        self,
        text: str,
        phase: str,
        user_emotion: Optional[str],
        memory_notes: str,
        relationship_notes: str,
        rel_stats: Optional[Dict[str, Any]],
        intent: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        # Mild shaping: remove creepy silence, keep concise
        text = text.strip()
        if not text:
            return "(SEL chose not to respond.)"
        # Clamp excessive whitespace
        text = "\n".join([ln.rstrip() for ln in text.splitlines() if ln.strip()])
        return text

    def _build_prompt(self, system: str, context_block: str, user: str) -> str:
        # Simple, model-agnostic prompt template
        parts = [PERSONA_STYLE_GUARD.strip()]
        if system:
            parts.append(system.strip())
        if context_block:
            parts.append(context_block.strip())
        parts.append(f"User: {user.strip()}\nAssistant:")
        return "\n\n".join(parts) + "\n"

    def respond(
        self,
        user_text: str,
        user_id: str = "local_user",
        system_note: str = "",
        channel_context: Optional[Dict[str, Any]] = None,
        prefer: str = "auto",  # "auto" | "general" | "coder"
        is_dm: bool = False,
    ) -> Dict[str, Any]:
        # Read context from memory
        context_block, hits = self.memory.make_context_block(user_id, user_text, k=5)

        # Detect intent and priority
        intent = detect_user_intent(user_text)
        pr_label, pr_score = self._classify_priority(user_text, dm=is_dm)

        # Choose expert
        backend = self.general
        if prefer == "coder" or (prefer == "auto" and intent == "task"):
            backend = self.coder if self.coder.is_ready() else self.general

        # Prompt assembly with style guard
        prompt = self._build_prompt(system_note, context_block, user_text)
        raw = backend.generate(prompt)
        styled = self._apply_style(
            raw,
            phase="reply",
            user_emotion=None,
            memory_notes="",
            relationship_notes="",
            rel_stats=None,
            intent=intent,
            context=channel_context,
        )

        # Memory write policy
        consented = self._has_memory_consent(user_text)
        if consented:
            # only add if looks reusable/long-lived: crude heuristic
            if len(user_text) >= 20 and not user_text.startswith("how are you"):
                self.memory.add(
                    user_id=user_id,
                    text=user_text,
                    kind="soft",
                    ttl_seconds=None,  # default
                    tags=[intent],
                    consented=True,
                )

        # Tweak hormones from rough mood labels inferred from response length/intent
        mood = "calm"
        if pr_label == "high":
            mood = "excited"
        elif intent == "help":
            mood = "confident"
        self.hormones.step(mood)

        return {
            "text": styled,
            "priority": {"label": pr_label, "score": pr_score},
            "intent": intent,
            "hormone": self.hormones.label(),
        }

