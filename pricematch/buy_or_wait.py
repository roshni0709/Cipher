"""Buy-now-or-wait: probability the price drops within the horizon.

Features come from the price history (position vs min/max/median, recent
slope, volatility, days since last drop). A logistic regression is fitted
on synthetic labelled histories at first use so the model runs out of the
box; retrain on your real accumulated history once you have weeks of it.
With almost no history, a conservative heuristic answers instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_DAYS = 14

FEATURES = ["pos_in_range", "vs_median", "slope_14d", "volatility", "days_since_drop"]


def features(history: pd.DataFrame) -> np.ndarray:
    prices = history["price"].to_numpy(dtype=float)
    current = prices[-1]
    lo, hi = prices.min(), prices.max()
    pos = (current - lo) / (hi - lo) if hi > lo else 0.5
    vs_median = current / np.median(prices) - 1.0
    window = prices[-14:] if len(prices) >= 14 else prices
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0] / window.mean() if len(window) > 1 else 0.0
    volatility = prices.std() / prices.mean() if prices.mean() else 0.0
    drops = np.where(np.diff(prices) < -0.005 * prices[:-1])[0]
    days_since = len(prices) - 1 - drops[-1] if len(drops) else len(prices)
    return np.array([pos, vs_median, slope, volatility, min(days_since, 60) / 60.0])


class BuyOrWait:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(0)
        x, y = _synthetic_training(rng, n=4000)
        self.model = LogisticRegression(max_iter=1000).fit(x, y)

    def predict(self, history: pd.DataFrame) -> dict:
        if history.empty or len(history) < 7:
            return {"recommendation": "BUY", "drop_probability": 0.0,
                    "reason": "not enough price history to predict; buy if the landed cost is good"}
        p_drop = float(self.model.predict_proba(features(history).reshape(1, -1))[0, 1])
        current = history["price"].iloc[-1]
        near_min = current <= history["price"].min() * 1.02
        if p_drop > 0.55 and not near_min:
            rec, reason = "WAIT", f"model sees a {p_drop:.0%} chance of a lower price within {HORIZON_DAYS} days"
        elif near_min:
            rec, reason = "BUY", "price is at its tracked low"
        else:
            rec, reason = "BUY", f"only a {p_drop:.0%} chance of a drop soon"
        return {"recommendation": rec, "drop_probability": round(p_drop, 3), "reason": reason}

    def explain(self, history: pd.DataFrame) -> dict:
        """Why the model said what it said: exact log-odds contributions.

        For a linear model, SHAP values reduce to coefficient x value, so
        this decomposition is exact, not an approximation, and needs no
        shap dependency. Positive contribution pushes toward WAIT (a drop
        is likely); negative pushes toward BUY.
        """
        if history.empty or len(history) < 7:
            return {"contributions": [], "note": "heuristic path: too little history to explain"}
        x = features(history)
        coefs = self.model.coef_[0]
        contribs = coefs * x
        order = np.argsort(-np.abs(contribs), kind="stable")
        rows = []
        for i in order:
            rows.append({"feature": FEATURES[i],
                         "value": round(float(x[i]), 4),
                         "contribution": round(float(contribs[i]), 4),
                         "direction": "wait" if contribs[i] > 0 else "buy",
                         "plain": _plain_english(FEATURES[i], float(x[i]), float(contribs[i]))})
        return {"intercept": round(float(self.model.intercept_[0]), 4),
                "log_odds": round(float(self.model.intercept_[0] + contribs.sum()), 4),
                "contributions": rows}


def _plain_english(name: str, value: float, contrib: float) -> str:
    direction = "pushes toward waiting" if contrib > 0 else "pushes toward buying now"
    phrases = {
        "pos_in_range": f"price sits at {value:.0%} of its tracked range",
        "vs_median": f"price is {value:+.1%} vs its median",
        "slope_14d": f"14-day trend is {value:+.2%} per day",
        "volatility": f"price volatility is {value:.1%}",
        "days_since_drop": f"{value * 60:.0f} days since the last meaningful drop",
    }
    return f"{phrases.get(name, name)}, which {direction}"


def _synthetic_training(rng: np.random.RandomState, n: int = 4000):
    """Generate synthetic price paths; label = did price drop >3% in horizon."""
    x, y = [], []
    for _ in range(n):
        days = 60
        drift = rng.uniform(-0.004, 0.004)
        noise = rng.uniform(0.005, 0.05)
        prices = 1000 * np.cumprod(1 + drift + rng.normal(0, noise, days))
        hist = pd.DataFrame({"price": prices[:-HORIZON_DAYS]})
        future = prices[-HORIZON_DAYS:]
        label = int(future.min() < prices[-HORIZON_DAYS - 1] * 0.97)
        x.append(features(hist))
        y.append(label)
    return np.array(x), np.array(y)
