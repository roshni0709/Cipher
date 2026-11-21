from pricematch.data import sample
from pricematch.db.store import Store


def test_seed_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    offer = sample.offers()[0]
    store.upsert_offer(offer)
    for row in sample.make_history(offer).itertuples():
        store.add_snapshot(offer["offer_id"], row.ts.date().isoformat(), row.price)
    offers = store.offers()
    assert len(offers) == 1
    assert offers.iloc[0]["offer_id"] == offer["offer_id"]
    hist = store.history(offer["offer_id"])
    assert len(hist) == 60
    assert hist["ts"].is_monotonic_increasing
    store.close()
