import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import moa_rag_vllm_app as app


class DummySentenceTransformer:
    def __init__(self, dim: int = 4):
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, **_: object):
        vectors = []
        for text in texts:
            if not text:
                vec = np.zeros(self._dim, dtype=np.float32)
            else:
                base = float((sum(ord(c) for c in text) % 97) + 1)
                vec = np.array([(base + i) for i in range(self._dim)], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm:
                    vec /= norm
            vectors.append(vec)
        if not vectors:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack(vectors)


@pytest.fixture(autouse=True)
def reset_vector_store_globals(monkeypatch):
    """Ensure tests always exercise the NumPy fallback backend."""
    monkeypatch.setattr(app, "FAISS_AVAILABLE", False, raising=False)
    monkeypatch.setattr(app, "faiss", None, raising=False)
    importlib.reload(app)
    monkeypatch.setattr(app, "FAISS_AVAILABLE", False, raising=False)
    monkeypatch.setattr(app, "faiss", None, raising=False)
    yield
    importlib.reload(app)


def test_vector_store_numpy_backend_add_search_and_persist(tmp_path):
    store = app.VectorStore(index_dir=str(tmp_path), model_name="dummy", model=DummySentenceTransformer())
    docs = [
        ("alpha beta", {"id": "1", "source": "alpha.txt", "text": "alpha beta"}),
        ("gamma delta", {"id": "2", "source": "gamma.txt", "text": "gamma delta"}),
    ]
    store.add_texts(docs)

    results = store.search("alpha question", k=2)
    assert results, "Expected at least one search hit"
    assert results[0]["source"] == "alpha.txt"
    assert "score" in results[0]

    store2 = app.VectorStore(index_dir=str(tmp_path), model_name="dummy", model=DummySentenceTransformer())
    results2 = store2.search("gamma delta", k=1)
    assert results2 and results2[0]["source"] == "gamma.txt"


def test_vector_store_ignores_empty_docs(tmp_path):
    store = app.VectorStore(index_dir=str(tmp_path), model_name="dummy", model=DummySentenceTransformer())
    store.add_texts([])
    assert store.metas == []


def test_simple_chunk_and_build_context():
    chunks = app.simple_chunk("abcdefghij", size=4, overlap=1)
    assert chunks[0] == "abcd"
    assert len(chunks) >= 3

    ctx = app.build_context([
        {"text": "foo", "source": "a"},
        {"text": "bar", "source": "b"},
    ], max_chars=40)
    assert ctx.count("[Source:") == 2
    assert len(ctx) <= 40


def test_code_score_detects_code_keywords():
    score = app.code_score("Python traceback shows exception", app.SET.router_code_keywords)
    assert score > 0
    assert app.code_score("Just chatting about weather", app.SET.router_code_keywords) == 0
