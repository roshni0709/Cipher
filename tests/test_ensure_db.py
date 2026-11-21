import os

from pricematch.data import sample
from pricematch.pipeline import ensure_db
from pricematch.db.store import Store


def test_ensure_db_seeds_only_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not os.path.exists("pricematch.db")
    assert ensure_db("pricematch.db") is True
    assert os.path.exists("pricematch.db")
    store = Store("pricematch.db")
    assert len(store.offers()) == len(sample.offers())
    store.close()
    # second call must not reseed
    assert ensure_db("pricematch.db") is False


def test_ensure_db_reseeds_stale_catalog(tmp_path, monkeypatch):
    """A DB from an older version of the catalog gets replaced automatically."""
    monkeypatch.chdir(tmp_path)
    ensure_db("pricematch.db")
    # simulate an old-version DB: fewer clusters than sample.py ships
    import sqlite3
    conn = sqlite3.connect("pricematch.db")
    conn.execute("DELETE FROM offers WHERE cluster_id LIKE 'boat%'")
    conn.commit(); conn.close()
    store = Store("pricematch.db")
    stale_n = int(store.offers()["cluster_id"].nunique())
    store.close()
    assert stale_n < len(sample.CLUSTERS)

    assert ensure_db("pricematch.db") is True  # stale: reseeded
    store = Store("pricematch.db")
    assert int(store.offers()["cluster_id"].nunique()) == len(sample.CLUSTERS)
    store.close()
    import glob
    assert glob.glob("pricematch.stale-*.db")  # old data kept, out of the way


def test_ensure_db_reseeds_unreadable_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with open("pricematch.db", "wb") as f:
        f.write(b"not a sqlite database")
    assert ensure_db("pricematch.db") is True
    store = Store("pricematch.db")
    assert len(store.offers()) == len(sample.offers())
    store.close()
