"""Similarity index: FAISS when installed, exact NumPy fallback otherwise.

The fallback is fine to ~100k vectors on a laptop, so the demo never
needs FAISS; FAISS matters when she scales the corpus and wants the
resume bullet about it.
"""
from __future__ import annotations

import numpy as np


class NumpyIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = _normalize(vectors)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise RuntimeError("index is empty")
        q = _normalize(query)
        sims = q @ self.vectors.T
        top = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, top, axis=1)
        return scores, top


class FaissIndex:
    def __init__(self, dim: int):
        import faiss

        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(_normalize(vectors).astype(np.float32))

    def search(self, query: np.ndarray, k: int = 5):
        return self.index.search(_normalize(query).astype(np.float32), k)


def build_index(dim: int):
    try:
        import faiss  # noqa: F401

        return FaissIndex(dim)
    except ImportError:
        return NumpyIndex(dim)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms
