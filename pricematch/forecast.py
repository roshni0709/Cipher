"""14-day price forecasting with confidence bands.

Method (deliberately simple and defensible in an interview):
- Ordinary least squares on features: linear trend (days since first
  observation) + weekly seasonality (sin/cos of a 7-day cycle). Weekly
  cycles matter in Indian e-commerce: sales and weekend promos repeat.
- Confidence bands from the residual standard deviation, widened by
  sqrt(h) for h days ahead — the classic random-walk uncertainty growth,
  so honesty about uncertainty is built in rather than tuned away.
- numpy only, deterministic, no model weights. Not a crystal ball: the
  chart labels the bands as a projection of recent trend, not a promise.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np


def _design(t: np.ndarray) -> np.ndarray:
    """Columns: intercept, trend, weekly sin, weekly cos."""
    return np.column_stack([
        np.ones_like(t),
        t,
        np.sin(2 * np.pi * t / 7.0),
        np.cos(2 * np.pi * t / 7.0),
    ])


def forecast(points: list[dict], horizon: int = 14, band_width: float = 1.28) -> dict:
    """points: [{"ts": "YYYY-MM-DD", "price": float}, ...] oldest first.

    band_width 1.28 ≈ 80% interval under a normal-residual assumption.
    Returns {"points": [{"ts", "predicted", "lower", "upper"}...], "residual_std",
    "trend_per_week"} or {"points": []} when there is too little history.
    """
    if len(points) < 10:
        return {"points": [], "residual_std": None, "trend_per_week": None,
                "note": "need at least 10 daily prices to forecast"}

    first = date.fromisoformat(points[0]["ts"])
    t = np.array([(date.fromisoformat(p["ts"]) - first).days for p in points], dtype=float)
    y = np.array([p["price"] for p in points], dtype=float)

    X = _design(t)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    sigma = float(np.sqrt(resid @ resid / dof))

    last = date.fromisoformat(points[-1]["ts"])
    out = []
    for h in range(1, horizon + 1):
        day = last + timedelta(days=h)
        th = float((day - first).days)
        row = _design(np.array([th]))[0]  # shape (4,): keep matmul 0-d
        pred = float(row @ coef)  # 0-d scalar: safe on numpy 1.x and 2.x
        half = band_width * sigma * math.sqrt(h)
        out.append({"ts": day.isoformat(),
                    "predicted": round(pred, 2),
                    "lower": round(max(pred - half, 0.0), 2),
                    "upper": round(pred + half, 2)})
    return {"points": out,
            "residual_std": round(sigma, 2),
            "trend_per_week": round(float(coef[1]) * 7, 2),
            "note": "projection of recent trend + weekly seasonality; bands widen with horizon"}
