from pricematch.buy_or_wait import BuyOrWait, features
from pricematch.data import sample


def test_features_shape():
    hist = sample.make_history(sample.offers()[0])
    assert features(hist).shape == (5,)


def test_declining_history_waits():
    bow = BuyOrWait()
    offer = next(o for o in sample.offers() if o["pattern"] == "declining")
    r = bow.predict(sample.make_history(offer))
    assert r["recommendation"] in {"WAIT", "BUY"}
    assert 0.0 <= r["drop_probability"] <= 1.0
    assert r["reason"]
