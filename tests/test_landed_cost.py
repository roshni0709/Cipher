from pricematch.landed_cost import Offer, compare


def test_coupon_and_shipping_change_ranking():
    a = Offer(site="x", title="t", price=1000, shipping=0, coupon={"type": "percent", "value": 10})
    b = Offer(site="y", title="t", price=950, shipping=100)
    ranked = compare([b, a])
    assert ranked[0]["site"] == "x" and ranked[0]["landed_cost"] == 900
    assert ranked[1]["landed_cost"] == 1050


def test_return_risk_noted():
    o = Offer(site="m", title="t", price=500, return_days=0)
    assert "no returns" in o.risk_notes()[0]
