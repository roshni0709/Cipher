"""Tools the agent layer can call. Each wraps one project component and
returns plain dicts so an LLM (or the rule fallback) can narrate them."""
from __future__ import annotations

import json

from ..buy_or_wait import BuyOrWait
from ..db.store import Store
from ..fake_discount import analyze as fake_discount_analyze
from ..landed_cost import Offer, compare


class Toolset:
    def __init__(self, db_path: str):
        self.store = Store(db_path)
        self._bow: BuyOrWait | None = None

    def close(self) -> None:
        self.store.close()

    def search_products(self, query: str, limit: int = 10) -> list[dict]:
        """Keyword search over tracked offers (SQL LIKE; the vector index
        replaces this in Phase 1)."""
        words = [w for w in query.lower().split() if len(w) > 2]
        if not words:
            return []
        df = self.store.offers()
        mask = df["title"].str.lower().apply(
            lambda t: any(w in t for w in words)
        )
        cols = ["offer_id", "cluster_id", "site", "title", "price", "shipping"]
        return df[mask][cols].head(limit).to_dict("records")

    def cluster_offers(self, cluster_id: str) -> list[dict]:
        df = self.store.offers()
        return df[df["cluster_id"] == cluster_id].to_dict("records")

    def landed_costs(self, cluster_id: str) -> list[dict]:
        offers = []
        for row in self.cluster_offers(cluster_id):
            coupon = json.loads(row["coupon"]) if row.get("coupon") else None
            offers.append(Offer(
                site=row["site"], title=row["title"], price=row["price"],
                shipping=row.get("shipping") or 0.0, coupon=coupon,
                return_days=row.get("return_days") or 0,
                return_free=bool(row.get("return_free")),
            ))
        return compare(offers)

    def price_check(self, offer_id: str) -> dict:
        history = self.store.history(offer_id)
        return fake_discount_analyze(history)

    def buy_or_wait(self, offer_id: str) -> dict:
        if self._bow is None:
            self._bow = BuyOrWait()
        return self._bow.predict(self.store.history(offer_id))

    def price_history(self, offer_id: str) -> dict:
        hist = self.store.history(offer_id)
        return {"offer_id": offer_id,
                "points": [{"ts": row.ts.date().isoformat(), "price": float(row.price)}
                           for row in hist.itertuples()],
                "min": float(hist["price"].min()) if not hist.empty else None,
                "max": float(hist["price"].max()) if not hist.empty else None}

    # OpenAI function-calling schemas for the LLM path
    SCHEMAS = [
        {"type": "function", "function": {"name": "search_products",
         "description": "Search tracked product offers by keywords",
         "parameters": {"type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]}}},
        {"type": "function", "function": {"name": "landed_costs",
         "description": "Compare true landed costs for all offers of one matched product cluster",
         "parameters": {"type": "object",
                        "properties": {"cluster_id": {"type": "string"}},
                        "required": ["cluster_id"]}}},
        {"type": "function", "function": {"name": "price_check",
         "description": "Check an offer's price history for fake discounts",
         "parameters": {"type": "object",
                        "properties": {"offer_id": {"type": "string"}},
                        "required": ["offer_id"]}}},
        {"type": "function", "function": {"name": "buy_or_wait",
         "description": "Predict whether an offer's price will drop soon",
         "parameters": {"type": "object",
                        "properties": {"offer_id": {"type": "string"}},
                        "required": ["offer_id"]}}},
    ]

    def call(self, name: str, args: dict):
        return getattr(self, name)(**args)
