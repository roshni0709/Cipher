"""CLIP matcher on the bundled image smoke set. Downloads ~350 MB of model
weights on first run (cached afterwards). Skipped when the ML deps are
not installed."""
import pytest

pytest.importorskip("open_clip")

from pricematch.data import image_eval
from pricematch.eval import metrics
from pricematch.matching.multimodal import MultimodalMatcher


def test_image_eval_set_is_fully_covered():
    df = image_eval.load_pairs()
    assert metrics.image_coverage(df) == 1.0
    assert set(df["label"].unique()) == {0, 1}


def test_clip_scores_positives_higher():
    df = image_eval.load_pairs()
    matcher = MultimodalMatcher()
    scores = matcher.score_pairs(df)
    pos = scores[df["label"] == 1]
    neg = scores[df["label"] == 0]
    assert pos.mean() > neg.mean()
    t, f1 = metrics.best_f1(df["label"].to_numpy(), scores)
    assert f1 >= 0.5
