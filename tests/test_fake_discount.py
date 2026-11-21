from pricematch.data import sample
from pricematch.fake_discount import analyze


def _offer(pattern):
    for o in sample.offers():
        if o["pattern"] == pattern:
            return o
    raise AssertionError(pattern)


def test_fake_sale_flagged():
    hist = sample.make_history(_offer("fake_sale"))
    r = analyze(hist)
    assert "inflated_before_sale" in r["flags"], r


def test_stable_not_flagged():
    hist = sample.make_history(_offer("stable"))
    r = analyze(hist)
    assert "inflated_before_sale" not in r["flags"], r


def test_insufficient_history():
    r = analyze(sample.make_history(_offer("stable"), days=3))
    assert r["flags"] == ["insufficient_history"]
