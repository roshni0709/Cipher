"""Phase 1 centerpiece: CLIP image + text embeddings with late fusion.

score = w * text_sim + (1 - w) * image_sim, where w is learned by grid
search on labelled training pairs. Falls back to text-only similarity
when a pair has no images, so it never hard-fails on missing photos.
Needs requirements-ml.txt (open-clip-torch, Pillow).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class MultimodalMatcher:
    name = "clip-multimodal"

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is not installed. Run: pip install -r requirements-ml.txt"
            ) from exc
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.weight = 0.5  # learned in fit()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            tokens = self.tokenizer(texts)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def encode_images(self, paths: list[str | Path | None]) -> np.ndarray:
        """One row per path; missing images get an all-zero (masked) row."""
        from PIL import Image

        batch, idx = [], []
        for i, p in enumerate(paths):
            if p and Path(p).exists():
                batch.append(self.preprocess(Image.open(p).convert("RGB")))
                idx.append(i)
        if not batch:
            return np.zeros((len(paths), 1), dtype=np.float32)
        with self.torch.no_grad():
            feats = self.model.encode_image(self.torch.stack(batch))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu().numpy()
        out = np.zeros((len(paths), feats.shape[1]), dtype=np.float32)
        out[np.array(idx)] = feats
        return out

    def score_pairs(self, pairs: pd.DataFrame, image_weight: float | None = None) -> np.ndarray:
        w = self.weight if image_weight is None else image_weight
        left_t = self.encode_texts(pairs["left_text"].tolist())
        right_t = self.encode_texts(pairs["right_text"].tolist())
        text_sim = np.sum(left_t * right_t, axis=1)
        if "left_image" not in pairs.columns:
            return text_sim
        left_i = self.encode_images(pairs["left_image"].tolist())
        right_i = self.encode_images(pairs["right_image"].tolist())
        has_both = (left_i.shape[1] > 1) & (right_i.shape[1] > 1) & \
                   (left_i.any(axis=1)) & (right_i.any(axis=1))
        img_sim = np.sum(left_i * right_i, axis=1)
        return np.where(has_both, w * text_sim + (1 - w) * img_sim, text_sim)

    def fit(self, train_pairs: pd.DataFrame, labels: np.ndarray) -> "MultimodalMatcher":
        """Grid-search the fusion weight for best F1 on labelled pairs."""
        from ..eval.metrics import best_f1

        best_w, best_score = 1.0, -1.0
        for w in np.arange(0.0, 1.01, 0.1):
            scores = self.score_pairs(train_pairs, image_weight=w)
            _, f1 = best_f1(labels, scores)
            if f1 > best_score:
                best_w, best_score = float(w), f1
        self.weight = best_w
        return self
