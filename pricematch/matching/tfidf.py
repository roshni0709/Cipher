"""Phase 0 baseline: TF-IDF cosine similarity over pair texts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfMatcher:
    name = "tfidf"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )

    def fit(self, pairs: pd.DataFrame) -> "TfidfMatcher":
        self.vectorizer.fit(pd.concat([pairs["left_text"], pairs["right_text"]]))
        return self

    def score_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        left = self.vectorizer.transform(pairs["left_text"])
        right = self.vectorizer.transform(pairs["right_text"])
        return np.array([
            cosine_similarity(left[i], right[i])[0, 0] for i in range(left.shape[0])
        ])
