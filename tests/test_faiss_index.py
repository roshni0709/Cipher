"""FAISS retrieval tests. Skipped when faiss/sentence-transformers are not
installed (base install runs TF-IDF only)."""
import numpy as np
import pytest

pytest.importorskip("faiss")

from pricematch.rag.corpus import Document
from pricematch.rag.index import FaissIndex
from pricematch.rag.retriever import Retriever


class StubEmbedder:
    """Deterministic bag-of-words embedder: no model download, and the
    FAISS-vs-linear equivalence is exactly what we want to verify."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for token in text.lower().split():
                v[hash(token) % self.dim] += 1.0
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n:
                    v /= n
            rows.append(v)
        return np.asarray(rows)


DOCS = [
    Document("a", "boAt airdopes earbuds price drop wait", {"cluster_id": "boat"}),
    Document("b", "redmi phone declining price buy now", {"cluster_id": "redmi"}),
    Document("c", "electric kettle landed cost shipping", {"cluster_id": "kettle"}),
    Document("d", "maroon kurti apparel cotton", {"cluster_id": "kurti"}),
]


def test_faiss_index_matches_linear_scan():
    emb = StubEmbedder()
    matrix = emb.encode([d.text for d in DOCS])
    index = FaissIndex(matrix)
    q = emb.encode(["earbuds price drop"])[0]
    scores, idx = index.search(q, 2)
    linear = matrix @ q
    assert list(idx) == list(np.argsort(-linear)[:2])
    assert idx[0] == 0  # the boAt doc


def test_retriever_faiss_backend_picks_relevant():
    r = Retriever(DOCS, backend="faiss", embed_model=StubEmbedder())
    assert r.backend == "faiss"
    top, score = r.retrieve("kurti cotton apparel", k=2, embed_model=StubEmbedder())[0]
    assert top.meta["cluster_id"] == "kurti"
    assert score > 0.05


def test_retriever_auto_prefers_faiss_when_available():
    r = Retriever(DOCS, backend="auto", embed_model=StubEmbedder())
    assert r.backend == "faiss"
