"""Product extraction from a product page's HTML.

Strategy 1: schema.org Product JSON-LD (most large e-commerce sites embed
it: name, offers.price, image). Strategy 2: OpenGraph meta tags. CSS
selectors per site are a last resort because they break constantly.
"""
from __future__ import annotations

import json
import re
from html import unescape

JSONLD_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
OG_RE = re.compile(
    r'<meta[^>]*property=["\']og:(title|image|price:amount)["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _walk_products(node) -> list[dict]:
    found = []
    if isinstance(node, dict):
        types = node.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if "Product" in types:
            found.append(node)
        for value in node.values():
            found.extend(_walk_products(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_products(item))
    return found


def _loads_tolerant(text: str):
    """Parse a JSON-LD block; fall back to the first complete value when the
    publisher appends trailing junk (some Indian retailers emit an extra `}`)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            obj, _end = json.JSONDecoder().raw_decode(text)
            return obj
        except json.JSONDecodeError:
            return None


def extract_product(html: str, url: str) -> dict | None:
    for block in JSONLD_RE.findall(html):
        data = _loads_tolerant(unescape(block.strip()))
        if data is None:
            continue
        for product in _walk_products(data):
            offers = product.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            price = offers.get("price") or offers.get("lowPrice")
            if price is None:
                continue
            image = product.get("image")
            if isinstance(image, list):
                image = image[0] if image else None
            if isinstance(image, dict):
                image = image.get("url")
            return {
                "title": (product.get("name") or "").strip(),
                "price": float(price),
                "currency": offers.get("priceCurrency", "INR"),
                "image_url": image,
                "url": url,
            }
    og = dict((m.group(1).lower(), m.group(2)) for m in OG_RE.finditer(html))
    if "title" in og and "price:amount" in og:
        return {
            "title": unescape(og["title"]).strip(),
            "price": float(og["price:amount"]),
            "currency": "INR",
            "image_url": og.get("image"),
            "url": url,
        }
    return None
