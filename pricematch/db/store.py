"""SQLite store: offers + daily price snapshots. Stdlib only."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS offers (
    offer_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    site TEXT NOT NULL,
    title TEXT NOT NULL,
    brand TEXT,
    category TEXT,
    price REAL,
    shipping REAL DEFAULT 0,
    coupon TEXT,
    return_days INTEGER DEFAULT 0,
    return_free INTEGER DEFAULT 0,
    url TEXT,
    image_url TEXT,
    last_seen TEXT
);
CREATE TABLE IF NOT EXISTS price_snapshots (
    offer_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    price REAL NOT NULL,
    shipping REAL DEFAULT 0,
    PRIMARY KEY (offer_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_snap_offer ON price_snapshots (offer_id);
"""


class Store:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def upsert_offer(self, offer: dict) -> None:
        self.conn.execute(
            """INSERT INTO offers (offer_id, cluster_id, site, title, brand, category,
                                   price, shipping, coupon, return_days, return_free, url, image_url, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
               ON CONFLICT(offer_id) DO UPDATE SET
                   price=excluded.price, shipping=excluded.shipping,
                   coupon=excluded.coupon, last_seen=excluded.last_seen""",
            (
                offer["offer_id"], offer["cluster_id"], offer["site"], offer["title"],
                offer.get("brand"), offer.get("category"), offer.get("price"),
                offer.get("shipping", 0), json.dumps(offer.get("coupon")),
                offer.get("return_days", 0), int(bool(offer.get("return_free"))),
                offer.get("url"), offer.get("image_url"),
            ),
        )
        self.conn.commit()

    def add_snapshot(self, offer_id: str, ts: str, price: float, shipping: float = 0.0) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO price_snapshots (offer_id, ts, price, shipping) VALUES (?,?,?,?)",
            (offer_id, ts, price, shipping),
        )
        self.conn.commit()

    def offers(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM offers", self.conn)

    def history(self, offer_id: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT ts, price, shipping FROM price_snapshots WHERE offer_id=? ORDER BY ts",
            self.conn, params=(offer_id,),
        )
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    def clusters(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT cluster_id, COUNT(*) AS n_offers, MIN(price) AS best_price FROM offers GROUP BY cluster_id",
            self.conn,
        )
