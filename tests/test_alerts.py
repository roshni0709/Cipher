"""Price-drop alert tests: scan logic, targets, digest format, planner tool."""
from pricematch.alerts import scan, format_digest


def _hist(prices):
    return [{"ts": f"2026-09-{i + 1:02d}", "price": p} for i, p in enumerate(prices)]


def test_scan_flags_meaningful_new_low():
    stable = [1000.0] * 29
    dropped = stable + [920.0]  # 8% below avg and a new low
    alerts = scan({"offer-a": _hist(dropped)})
    assert len(alerts) == 1
    a = alerts[0]
    assert a["offer_id"] == "offer-a" and a["latest"] == 920.0
    assert a["new_low"] and a["pct_below_avg"] >= 3.0


def test_scan_ignores_small_noise():
    wobble = [1000.0, 1005.0, 995.0] * 10 + [990.0]  # 1% off avg: no alert
    assert scan({"offer-a": _hist(wobble)}) == []


def test_scan_ignores_drop_that_is_not_new_low():
    # price already fell to 900 long ago; 920 today is below avg but not a new low
    hist = [1000.0] * 10 + [900.0] + [1000.0] * 18 + [920.0]
    assert scan({"offer-a": _hist(hist)}) == []


def test_scan_target_crossing_triggers_alert():
    stable = [1000.0] * 30
    alerts = scan({"offer-a": _hist(stable)}, targets={"offer-a": 1000.0})
    assert len(alerts) == 1 and alerts[0]["hit_target"]


def test_scan_sorts_biggest_drop_first():
    hists = {
        "small": _hist([1000.0] * 29 + [940.0]),
        "big": _hist([1000.0] * 29 + [800.0]),
    }
    alerts = scan(hists)
    assert [a["offer_id"] for a in alerts] == ["big", "small"]


def test_scan_needs_history():
    assert scan({"offer-a": _hist([100.0] * 3)}) == []


def test_format_digest_email_ready():
    alerts = scan({"offer-a": _hist([1000.0] * 29 + [900.0])},
                  targets={"offer-a": 950.0})
    d = format_digest(alerts, {"offer-a": "boAt Airdopes 141"})
    assert "1 drop" in d["subject"]
    assert "boAt Airdopes 141" in d["body"] and "Rs 900" in d["body"]
    assert "target" in d["body"] and "new tracked low" in d["body"]


def test_format_digest_empty():
    d = format_digest([])
    assert "no drops" in d["subject"]


def test_planner_price_alerts_tool(tmp_path):
    from pricematch.pipeline import seed
    from pricematch.agent.tools import Toolset
    from pricematch.agent.planner import RulePlanner, default_registry
    db = str(tmp_path / "a.db")
    seed(db)
    ts = Toolset(db)
    planner = RulePlanner(default_registry(ts))
    result = planner.run("any price drop alerts for boAt airdopes")
    ts.close()
    assert any(s.tool == "price_alerts" for s in result.steps)
    doc = next((d for d in result.tool_docs if d["meta"]["tool"] == "price_alerts"), None)
    assert doc is not None and "Price watch" in doc["text"]


def test_current_quote_wins_over_stale_history_endpoint():
    from pricematch.alerts import scan
    points = [{'ts': str(i), 'price': 100.0} for i in range(8)]
    points[-1]['price'] = 50.0
    histories = {'one': points}
    assert scan(histories)[0]['latest'] == 50.0
    assert scan(histories, current_prices={'one': 110.0}) == []
    out = scan(histories, targets={'one': 90.0}, current_prices={'one': 80.0})
    assert out[0]['latest'] == 80.0 and out[0]['hit_target']
    assert histories['one'][-1]['price'] == 50.0
