"""Turn the tracked catalog into retrieval documents.

Two document types:
- one per offer: what it costs to actually buy (landed cost, shipping,
  coupon, returns) at one store;
- one per product cluster: price intelligence on the best offer
  (60-day range, fake-discount flags, buy-or-wait verdict).

Every sentence a generator can later cite comes from these documents,
so grounding is checkable by eye.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..landed_cost import Offer


@dataclass
class Document:
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


def _money(v: float) -> str:
    return f"₹{v:,.0f}"


def _offer_doc(row) -> Document:
    coupon = json.loads(row.coupon) if isinstance(row.coupon, str) and row.coupon else None
    offer = Offer(site=row.site, title=row.title, price=float(row.price),
                  shipping=float(row.shipping or 0), coupon=coupon,
                  return_days=int(row.return_days or 0),
                  return_free=bool(row.return_free))
    ship = "free shipping" if not offer.shipping else f"shipping {_money(offer.shipping)}"
    if offer.coupon_value():
        deal = f"a coupon saves {_money(offer.coupon_value())}"
    else:
        deal = "no coupon"
    if offer.return_days and offer.return_free:
        ret = f"free {offer.return_days}-day returns"
    elif offer.return_days:
        ret = f"{offer.return_days}-day returns but you pay return shipping"
    else:
        ret = "no returns accepted"
    text = (f"{row.title} at {row.site}: listed {_money(offer.price)} with {ship}, "
            f"{deal}; true landed cost {_money(offer.landed())}; {ret}.")
    return Document(
        doc_id=f"offer:{row.offer_id}",
        text=text,
        meta={"type": "offer", "cluster_id": row.cluster_id, "site": row.site,
              "landed_cost": offer.landed(), "price": offer.price},
    )


def _intel_doc(cluster_id: str, rows, store, bow) -> Document:
    import numpy as np

    best_row = min(rows, key=lambda r: Offer(
        site=r.site, title=r.title, price=float(r.price),
        shipping=float(r.shipping or 0),
        coupon=json.loads(r.coupon) if isinstance(r.coupon, str) and r.coupon else None,
    ).landed())
    history = store.history(best_row.offer_id)
    from ..fake_discount import analyze

    check = analyze(history)
    if bow is not None:
        verdict = bow.predict(history)
    else:
        verdict = {"recommendation": "unavailable", "reason": "buy-or-wait model not loaded"}
    if history.empty:
        span = "no price history yet"
    else:
        span = (f"60-day range {_money(history['price'].min())}–{_money(history['price'].max())}, "
                f"median {_money(float(np.median(history['price'])))}")
    flags = ", ".join(check["flags"])
    title = min((r.title for r in rows), key=len)
    text = (f"{cluster_id} ({title}): tracked at {len(rows)} store(s). "
            f"Best true landed cost {_money(Offer(site=best_row.site, title=best_row.title, price=float(best_row.price), shipping=float(best_row.shipping or 0), coupon=json.loads(best_row.coupon) if isinstance(best_row.coupon, str) and best_row.coupon else None).landed())} "
            f"at {best_row.site}. {span}. "
            f"Price check: {flags} — {check['detail']}. "
            f"Buy-or-wait verdict: {verdict['recommendation']} — {verdict['reason']}.")
    return Document(
        doc_id=f"intel:{cluster_id}",
        text=text,
        meta={"type": "intel", "cluster_id": cluster_id,
              "best_site": best_row.site, "flags": check["flags"],
              "recommendation": verdict["recommendation"], "reason": verdict["reason"]},
    )


def build_corpus(store, bow=None) -> list[Document]:
    """All retrieval documents for the current database contents."""
    df = store.offers()
    docs: list[Document] = []
    for row in df.itertuples():
        docs.append(_offer_doc(row))
    for cluster_id, grp in df.groupby("cluster_id"):
        docs.append(_intel_doc(cluster_id, list(grp.itertuples()), store, bow))
    return docs
