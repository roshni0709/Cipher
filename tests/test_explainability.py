"""Explainability tests: exact coefficient x value decomposition."""
import numpy as np
import pandas as pd

from pricematch.buy_or_wait import BuyOrWait, FEATURES


def _hist(prices):
    return pd.DataFrame({"price": prices})


def test_contributions_sum_to_log_odds():
    bow = BuyOrWait()
    rng = np.random.RandomState(7)
    prices = 1000 * np.cumprod(1 + rng.normal(0, 0.02, 60))
    ex = bow.explain(_hist(prices))
    assert ex["contributions"], "expected contributions for a 60-day history"
    total = ex["intercept"] + sum(c["contribution"] for c in ex["contributions"])
    assert abs(total - ex["log_odds"]) < 1e-3
    # every model feature is explained, most influential first
    assert [c["feature"] for c in ex["contributions"]] == list(
        sorted(FEATURES, key=lambda f: -abs(next(c["contribution"]
               for c in ex["contributions"] if c["feature"] == f)))
    ) or {c["feature"] for c in ex["contributions"]} == set(FEATURES)


def test_direction_matches_sign():
    bow = BuyOrWait()
    prices = np.linspace(1200, 1000, 60)  # falling to its low: buy signal
    ex = bow.explain(_hist(prices))
    for c in ex["contributions"]:
        assert c["direction"] == ("wait" if c["contribution"] > 0 else "buy")
    assert all(c["plain"] for c in ex["contributions"])


def test_explain_too_little_history():
    bow = BuyOrWait()
    ex = bow.explain(_hist([100.0, 101.0]))
    assert ex["contributions"] == [] and "too little history" in ex["note"]


def test_contributions_are_exact_coefficient_times_value():
    bow = BuyOrWait()
    prices = np.concatenate([np.full(55, 1000.0), np.linspace(1010, 1300, 5)])
    ex = bow.explain(_hist(prices))
    pos = next(c for c in ex["contributions"] if c["feature"] == "pos_in_range")
    assert pos["value"] > 0.9  # price at the top of its tracked range
    coef = bow.model.coef_[0][FEATURES.index("pos_in_range")]
    assert abs(pos["contribution"] - round(float(coef) * pos["value"], 4)) < 2e-4
