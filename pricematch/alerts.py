"""Price-drop alerts: scan tracked offers, flag meaningful drops, and
render an email-ready digest. Offline and deterministic; a cron job or
scheduled runner calls scan() and sends the digest through any mailer.

A drop counts as meaningful when the latest price is both:
- at least threshold_pct below the trailing 30-day average (not noise), and
- at or below the tracked minimum of the lookback window (a real new low),
or when it crosses a caller-supplied target price.
"""
from __future__ import annotations


def scan(price_histories: dict[str, list[dict]], threshold_pct: float = 3.0,
         targets: dict[str, float] | None = None,
         current_prices: dict[str, float] | None = None) -> list[dict]:
    """price_histories: offer_id -> [{"ts", "price"}...] oldest first.
    targets: optional offer_id -> target price the user set.
    Returns alerts sorted by biggest percentage drop first."""
    targets = targets or {}
    current_prices = current_prices or {}
    alerts = []
    for offer_id, points in price_histories.items():
        if len(points) < 7:
            continue
        prices = [p["price"] for p in points]
        # A current offer quote wins over the history endpoint. Earlier
        # points remain the comparison baseline; caller data is untouched.
        latest = current_prices.get(offer_id, prices[-1])
        trailing = prices[-31:-1]  # up to 30 days before today
        avg = sum(trailing) / len(trailing)
        prior_min = min(prices[:-1])
        pct_below_avg = (avg - latest) / avg * 100.0
        target = targets.get(offer_id)
        hit_target = target is not None and latest <= target
        new_low = latest <= prior_min
        if (pct_below_avg >= threshold_pct and new_low) or hit_target:
            alerts.append({
                "offer_id": offer_id,
                "latest": round(latest, 2),
                "trailing_avg": round(avg, 2),
                "pct_below_avg": round(pct_below_avg, 1),
                "new_low": new_low,
                "target": target,
                "hit_target": hit_target,
            })
    alerts.sort(key=lambda a: -a["pct_below_avg"])
    return alerts


def format_digest(alerts: list[dict], titles: dict[str, str] | None = None) -> dict:
    """Email-ready digest. titles maps offer_id -> product title."""
    titles = titles or {}
    if not alerts:
        return {"subject": "Price watch: no drops today",
                "body": "No tracked product dropped meaningfully today."}
    lines = []
    for a in alerts:
        name = titles.get(a["offer_id"], a["offer_id"])
        line = (f"- {name}: now Rs {a['latest']:,.0f} "
                f"({a['pct_below_avg']}% below its 30-day average of Rs {a['trailing_avg']:,.0f})")
        if a["hit_target"]:
            line += f", hitting your target of Rs {a['target']:,.0f}"
        if a["new_low"]:
            line += " [new tracked low]"
        lines.append(line)
    return {"subject": f"Price watch: {len(alerts)} drop(s) today",
            "body": "Today's price drops:\n" + "\n".join(lines)}
