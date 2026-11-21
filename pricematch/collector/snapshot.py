"""Daily price snapshot collector.

Politics of this script: it fetches ONLY public product pages you list,
respects robots.txt, waits between requests, and identifies itself.
Run it once a day (cron/scheduler) to build price history. Do not point
it at Amazon/Flipkart at any real rate: their anti-bot will block you
and aggressive scraping violates their terms. For the demo, seed history
from `pricematch.data.sample` instead.

Usage:
  python -m pricematch.collector.snapshot urls.txt --db pricematch.db --delay 5
"""
from __future__ import annotations

import argparse
import hashlib
import time

from ..config import USER_AGENT
from ..db.store import Store
from .extract import extract_product
from .robots import RobotsCache


def fetch(url: str) -> str:
    try:
        import httpx

        r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30,
                      follow_redirects=True)
        r.raise_for_status()
        return r.text
    except ImportError:
        from urllib.request import Request, urlopen

        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")


def snapshot_urls(urls: list[str], db_path: str, delay: float = 5.0,
                  respect_robots: bool = True) -> list[dict]:
    store = Store(db_path)
    robots = RobotsCache(USER_AGENT)
    saved = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        if respect_robots and not robots.allowed(url):
            print(f"SKIP (robots.txt): {url}")
            continue
        try:
            html = fetch(url)
        except Exception as exc:
            print(f"ERROR {url}: {exc}")
            continue
        product = extract_product(html, url)
        if not product:
            print(f"NO DATA {url} (no JSON-LD Product or OG price found)")
            continue
        offer_id = hashlib.sha1(url.encode()).hexdigest()[:16]
        product["offer_id"] = offer_id
        product.setdefault("cluster_id", offer_id)  # matching phase assigns clusters
        product.setdefault("site", url.split("/")[2])
        store.upsert_offer(product)
        store.add_snapshot(offer_id, _today(), product["price"])
        saved.append(product)
        print(f"OK {product['price']:>10.2f}  {product['title'][:70]}")
        time.sleep(delay)
    store.close()
    return saved


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls_file")
    parser.add_argument("--db", default="pricematch.db")
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--ignore-robots", action="store_true",
                        help="Fetch even when robots.txt disallows. Don't.")
    args = parser.parse_args(argv)
    with open(args.urls_file) as fh:
        urls = fh.read().splitlines()
    snapshot_urls(urls, args.db, args.delay, not args.ignore_robots)


if __name__ == "__main__":
    main()
