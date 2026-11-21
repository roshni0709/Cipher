"""Phase 1: sentence-transformers text matcher (needs requirements-ml.txt)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TextEmbeddingMatcher:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements-ml.txt"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def score_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        left = self.model.encode(pairs["left_text"].tolist(), normalize_embeddings=True)
        right = self.model.encode(pairs["right_text"].tolist(), normalize_embeddings=True)
        return np.sum(left * right, axis=1)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
