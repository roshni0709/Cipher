"""FAISS vector index for retrieval.

Wraps faiss.IndexFlatIP over L2-normalized embeddings (cosine similarity
as inner product). IndexFlatIP is exact brute-force search — the right
choice at demo scale (hundreds of chunks) and the same interface holds
when the catalog grows: swap in IndexIVFFlat or HNSW and retrieval
becomes approximate nearest neighbour over lakhs of vectors without
touching the retriever. That swap story is the interview point.
"""
from __future__ import annotations

import numpy as np


class FaissIndex:
    def __init__(self, embeddings: np.ndarray):
        import faiss  # local import: requirements-ml.txt only

        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2-D array")
        self.dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) of the top-k nearest rows."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        scores, idx = self.index.search(q, k)
        return scores[0], idx[0]
