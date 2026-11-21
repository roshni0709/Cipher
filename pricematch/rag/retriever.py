"""Vector retrieval over the catalog corpus, with interchangeable
backends behind one interface:

- ``faiss``: neural embeddings (MiniLM) served from a FAISS index
  (pricematch.rag.index). Needs requirements-ml.txt.
- ``minilm``: same embeddings, plain linear scan. The fallback when
  FAISS is not installed.
- ``tfidf``: scikit-learn TF-IDF + cosine similarity. Always installed,
  fully offline, zero downloads. This is also what the tests use, so
  retrieval in CI is deterministic.

``backend="auto"`` picks the best available, so the demo can never
break on stage.
"""
from __future__ import annotations

import numpy as np

from .corpus import Document


class Retriever:
    def __init__(self, docs: list[Document], backend: str = "auto", embed_model=None):
        self.docs = docs
        self.backend = self._resolve(backend, embed_model)
        if self.backend in ("minilm", "faiss"):
            matrix = np.asarray(embed_model.encode(
                [d.text for d in docs], normalize_embeddings=True))
            if self.backend == "faiss":
                from .index import FaissIndex

                self._faiss = FaissIndex(matrix)
                self._matrix = None
            else:
                self._faiss = None
                self._matrix = matrix
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self._matrix = self._vec.fit_transform([d.text for d in docs])
            self._faiss = None

    def _resolve(self, backend: str, embed_model) -> str:
        if backend in ("tfidf", "minilm", "faiss"):
            if backend in ("minilm", "faiss") and embed_model is None:
                raise ValueError(f"{backend} backend requires a loaded embed_model")
            if backend == "faiss":
                import faiss  # noqa: F401 — raises ImportError when absent
            return backend
        if embed_model is None:
            return "tfidf"
        try:
            import faiss  # noqa: F401

            return "faiss"
        except ImportError:
            return "minilm"

    def retrieve(self, query: str, k: int = 4, embed_model=None,
                 min_score: float = 0.05) -> list[tuple[Document, float]]:
        if self.backend in ("minilm", "faiss"):
            q = np.asarray(embed_model.encode([query], normalize_embeddings=True))[0]
            if self._faiss is not None:
                scores, idx = self._faiss.search(q, k)
                return [(self.docs[i], float(s)) for i, s in zip(idx, scores)
                        if 0 <= i < len(self.docs) and s > min_score]
            scores = self._matrix @ q
        else:
            q = self._vec.transform([query])
            scores = (self._matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > min_score]
