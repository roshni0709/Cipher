"""Forecasting tests: shape, determinism, band widening, sanity on trend."""
from datetime import date, timedelta

from pricematch.forecast import forecast


def _mk_points(n=60, start_price=1000.0, daily_drop=2.0, weekly_amp=50.0):
    """Deterministic synthetic series: downtrend + weekly wave."""
    import math
    first = date(2026, 9, 1)
    pts = []
    for i in range(n):
        price = (start_price - daily_drop * i + weekly_amp * math.sin(2 * math.pi * i / 7)
                 + 9.0 * math.sin(i * 1.7))  # incommensurate wobble: nonzero residuals
        pts.append({"ts": (first + timedelta(days=i)).isoformat(), "price": round(price, 2)})
    return pts


def test_forecast_shape_and_horizon():
    out = forecast(_mk_points())
    assert len(out["points"]) == 14
    p0 = out["points"][0]
    assert set(p0) == {"ts", "predicted", "lower", "upper"}
    last_ts = date.fromisoformat(out["points"][-1]["ts"])
    assert (last_ts - date.fromisoformat(_mk_points()[-1]["ts"])).days == 14


def test_forecast_recovers_trend():
    out = forecast(_mk_points(daily_drop=3.0))
    # fitted weekly trend should be close to -21 (3/day x 7)
    assert abs(out["trend_per_week"] - (-21.0)) < 3.0


def test_bands_widen_with_horizon():
    pts = forecast(_mk_points())["points"]
    widths = [p["upper"] - p["lower"] for p in pts]
    assert all(b >= a for a, b in zip(widths, widths[1:]))
    assert widths[-1] > widths[0]


def test_bands_bracket_prediction():
    for p in forecast(_mk_points())["points"]:
        assert p["lower"] <= p["predicted"] <= p["upper"]


def test_deterministic():
    assert forecast(_mk_points()) == forecast(_mk_points())


def test_too_little_history():
    out = forecast(_mk_points()[:5])
    assert out["points"] == [] and "10 daily prices" in out["note"]


def test_forecast_on_seeded_db(tmp_path):
    """End-to-end against real sample history from the store."""
    from pricematch.pipeline import seed
    from pricematch.agent.tools import Toolset
    db = str(tmp_path / "f.db")
    seed(db)
    ts = Toolset(db)
    hits = ts.search_products("boAt Airdopes")
    offer_id = hits[0]["offer_id"]
    hist = ts.price_history(offer_id)
    ts.close()
    out = forecast(hist["points"])
    assert len(out["points"]) == 14
    latest = hist["points"][-1]["price"]
    # first-day prediction should be in a sane range around the latest price
    assert abs(out["points"][0]["predicted"] - latest) < latest * 0.3
