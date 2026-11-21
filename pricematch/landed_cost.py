"""True landed cost: the price you actually pay, not the listed price.

landed = price + shipping - coupon_value
Return policy is surfaced as risk, because a cheap listing with paid/no
returns can cost more than a pricier listing with free returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Offer:
    site: str
    title: str
    price: float
    shipping: float = 0.0
    coupon: dict | None = None       # {"type": "flat"|"percent", "value": X}
    return_days: int = 0
    return_free: bool = False

    def coupon_value(self) -> float:
        if not self.coupon:
            return 0.0
        if self.coupon.get("type") == "percent":
            return round(self.price * self.coupon["value"] / 100.0, 2)
        return float(self.coupon.get("value", 0.0))

    def landed(self) -> float:
        return round(self.price + self.shipping - self.coupon_value(), 2)

    def risk_notes(self) -> list[str]:
        notes = []
        if self.return_days == 0:
            notes.append("no returns accepted")
        elif not self.return_free:
            notes.append(f"{self.return_days}-day returns, but you pay return shipping")
        return notes


def compare(offers: list[Offer]) -> list[dict]:
    """Offers ranked by landed cost, cheapest first."""
    rows = [
        {
            "site": o.site,
            "price": o.price,
            "shipping": o.shipping,
            "coupon_value": o.coupon_value(),
            "landed_cost": o.landed(),
            "risk": "; ".join(o.risk_notes()) or "free returns",
        }
        for o in offers
    ]
    return sorted(rows, key=lambda r: r["landed_cost"])
