import numpy as np
import pandas as pd

from pricematch.eval import metrics


def test_perfect_scores():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    r = metrics.prf_at_threshold(labels, scores, 0.5)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0


def test_best_f1_finds_threshold():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.7, 0.6, 0.4, 0.3])
    t, f1 = metrics.best_f1(labels, scores)
    assert f1 == 1.0 and 0.4 < t <= 0.6


def test_top1_accuracy():
    pairs = pd.DataFrame({
        "left_id": ["a", "a", "b", "b"],
        "right_id": ["x", "y", "x", "z"],
        "label": [1, 0, 0, 1],
    })
    scores = np.array([0.9, 0.1, 0.2, 0.95])
    assert metrics.top1_accuracy(pairs, scores) == 1.0
    scores_bad = np.array([0.1, 0.9, 0.2, 0.95])
    assert metrics.top1_accuracy(pairs, scores_bad) == 0.5
