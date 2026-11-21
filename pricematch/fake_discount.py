"""Fake-discount detection from price history.

Flags:
- inflated_before_sale: price ramped up shortly before a 'sale' dropped it,
  so the discount percentage is computed off an inflated MRP.
- genuine_drop: current price is meaningfully below the period median.
- always_discounted: 'sale' price is just the normal price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def analyze(history: pd.DataFrame, sale_window: int = 7, inflate_ratio: float = 1.15,
            drop_ratio: float = 0.90) -> dict:
    """history: DataFrame with ts + price, oldest first."""
    if history.empty or len(history) < 5:
        return {"flags": ["insufficient_history"], "detail": "need more price snapshots"}
    prices = history["price"].to_numpy(dtype=float)
    current = prices[-1]
    median = float(np.median(prices))
    recent = prices[-sale_window:]
    before = prices[:-sale_window] if len(prices) > sale_window else prices[:1]

    flags = []
    detail = []
    pre_max = float(np.max(before))
    pre_median = float(np.median(before))
    ramp = pre_max / pre_median if pre_median else 1.0
    drop = current / pre_max if pre_max else 1.0

    if ramp >= inflate_ratio and drop <= 1.0 and current >= pre_median * 0.9:
        flags.append("inflated_before_sale")
        detail.append(
            f"price rose {ramp - 1:.0%} above normal in the weeks before the sale; "
            f"the 'discount' is measured from that inflated price"
        )
    if current <= median * drop_ratio:
        flags.append("genuine_drop")
        detail.append(f"current price is {1 - current / median:.0%} below the {len(prices)}-day median")
    if not flags:
        flags.append("normal")
        detail.append("no discount games detected in the tracked window")
    return {
        "flags": flags,
        "detail": " ".join(detail),
        "stats": {
            "current": round(current, 2),
            "median": round(median, 2),
            "min": round(float(np.min(prices)), 2),
            "max": round(float(np.max(prices)), 2),
            "days": int(len(prices)),
        },
    }
