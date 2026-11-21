"""Evaluation: precision/recall/F1 with threshold tuning, top-1 accuracy.

top-1 accuracy treats each left product as an anchor: among its scored
candidate pairs, the highest-scoring one must be a true match. This is
the number that maps to the real product ("we show the right twin").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prf_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": round(float(threshold), 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) maximising F1 over observed score values."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    best_t, best = 0.5, -1.0
    for t in np.unique(np.round(scores, 3)):
        f1 = prf_at_threshold(labels, scores, t)["f1"]
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t, best


def top1_accuracy(pairs: pd.DataFrame, scores: np.ndarray) -> float:
    """For each anchor with at least one true match, did the top-scoring
    candidate match? Anchors with no positive candidate are skipped."""
    df = pairs.assign(_score=scores)
    correct, total = 0, 0
    for _, group in df.groupby("left_id"):
        if group["label"].sum() == 0:
            continue
        top = group.loc[group["_score"].idxmax()]
        correct += int(top["label"] == 1)
        total += 1
    return correct / total if total else 0.0


def evaluate(pairs: pd.DataFrame, labels: np.ndarray, scores: np.ndarray,
             threshold: float | None = None) -> dict:
    if threshold is None:
        threshold, _ = best_f1(labels, scores)
    report = prf_at_threshold(labels, scores, threshold)
    report["top1_accuracy"] = round(top1_accuracy(pairs, scores), 4)
    report["n_pairs"] = len(pairs)
    report["n_positive"] = int(np.sum(np.asarray(labels) == 1))
    return report


def image_coverage(pairs: pd.DataFrame) -> float:
    """Fraction of pairs where BOTH sides carry a usable image. Report this
    with every multimodal number: a multimodal F1 over pairs with no images
    is a text-only result wearing a costume."""
    if "left_image" not in pairs.columns or pairs.empty:
        return 0.0
    def _ok(v) -> bool:
        return isinstance(v, str) and len(v) > 0
    both = pairs.apply(lambda r: _ok(r["left_image"]) and _ok(r["right_image"]), axis=1)
    return float(both.mean())
