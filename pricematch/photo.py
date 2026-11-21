"""Photo-to-price: identify a product from an uploaded photo, then hand the
matched cluster to the normal landed-cost + verdict pipeline.

Uses CLIP (clip-ViT-B-32 via sentence-transformers) to embed both the
uploaded photo and the bundled catalog photos, then ranks clusters by
cosine similarity. Everything runs offline once the model is cached;
without the ML extras installed, PhotoSearcher.unavailable_reason explains
what to install instead of crashing, and the UI hides the tab.

Two design points that make the demo trustworthy:

- Multiple photos per product. Catalog photos live in
  pricematch/data/photos/ as <cluster_id>.jpg, <cluster_id>-2.jpg, ... A
  cluster's score is the max cosine over its views, so any one
  representative angle is enough - one unrepresentative shot can't drag
  the whole cluster down.

- Calibrated confidence. Raw CLIP cosines on everyday photos sit in a
  narrow band (~0.6-0.75), so "0.69 vs 0.66" looks like a near-tie even
  when the model is sure. match() therefore also reports a softmax
  confidence over the catalog (temperature T): the top pick's share of
  probability mass. A photo that matches everything equally lands near
  1/N and is rejected; a decisive match stands out. The raw cosine is
  kept in the output for transparency.

- Two gates before naming a product. A cluster is only a confident
  match when (a) it wins a decisive share of softmax probability mass
  AND (b) its best view lands in the same-object zone. Gate (a) stops
  near-ties between lookalikes (two earbuds at 0.90/0.90 split the mass
  and abstain). Gate (b) stops same-colour stand-ins: a tan PUMA sneaker
  scores ~0.80-0.86 against tan catalog shoes - high, sometimes
  unopposed - but measured true matches of the actual SKU land at 0.89+,
  so the floor sits between the two bands. Below it the photo still gets
  "visually similar" suggestions (shoes like shoes), never a wrong name.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

PHOTOS_DIR = Path(__file__).resolve().parent / "data" / "photos"
TEMPERATURE = 0.02       # softmax temperature over catalog cosines
MIN_CONFIDENCE = 0.65    # top-pick probability mass below this: say "not sure"
MIN_SIMILARITY = 0.87    # same-object zone: measured true matches land >=0.89,
                         # same-colour lookalikes (wrong brand shoe/sneaker) ~0.80-0.86
SUGGESTION_FLOOR = 0.72  # suggestions below this are unrelated noise, not "similar"


def catalog_photos(photos_dir: Path | str = PHOTOS_DIR) -> dict[str, list[Path]]:
    """cluster_id -> all its catalog photos (supports several per product)."""
    d = Path(photos_dir)
    if not d.is_dir():
        return {}
    out: dict[str, list[Path]] = {}
    for p in sorted(d.glob("*.jpg")):
        # base file is the bare cluster id; extra views are "<id>-2.jpg" .. "-9".
        # (Cluster ids may end in numbers like airdopes-141 or rockerz-450, so
        # only a single digit 2-9 counts as a view suffix.)
        m = re.match(r"^(.+)-([2-9])$", p.stem)
        # ...unless the base has no primary photo: then the digit is part of
        # the cluster id itself (e.g. boat-wave-call-2, where no
        # boat-wave-call.jpg exists).
        if m and not (d / f"{m.group(1)}.jpg").exists():
            m = None
        cid = m.group(1) if m else p.stem
        out.setdefault(cid, []).append(p)
    # base photo first, extra views after (glob order puts "-2" before ".jpg")
    for cid in out:
        out[cid].sort(key=lambda p: (p.stem != cid, p.stem))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class PhotoSearcher:
    """Ranks catalog clusters against an uploaded photo."""

    def __init__(self, embed_image=None, photos_dir: Path | str = PHOTOS_DIR):
        """embed_image: fn(PIL.Image) -> list[float]. None loads CLIP lazily."""
        self._embed_image = embed_image
        self._model = None
        self.photos = catalog_photos(photos_dir)
        self._catalog_vecs: dict[str, list[float]] | None = None

    @property
    def unavailable_reason(self) -> str | None:
        if not self.photos:
            return "no catalog photos bundled yet"
        if self._embed_image is None:
            try:
                import sentence_transformers  # noqa: F401
                from PIL import Image  # noqa: F401
            except ImportError:
                return "install requirements-ml.txt for photo search"
        return None

    def _embed(self, image) -> list[float]:
        if self._embed_image is not None:
            return list(self._embed_image(image))
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("clip-ViT-B-32")
        return list(self._model.encode(image))

    def _catalog_embeddings(self) -> dict[str, list[list[float]]]:
        """cluster_id -> one normalized vector per catalog view."""
        if self._catalog_vecs is None:
            from PIL import Image
            self._catalog_vecs = {
                cid: [_normalize(self._embed(Image.open(p).convert("RGB")))
                      for p in paths]
                for cid, paths in self.photos.items()
            }
        return self._catalog_vecs

    def match(self, image, top: int = 3,
              temperature: float = TEMPERATURE,
              min_confidence: float = MIN_CONFIDENCE,
              min_similarity: float = MIN_SIMILARITY) -> dict:
        """image: PIL.Image (or path). Returns ranked clusters, each with the
        raw cosine and calibrated confidence; the top pick is confident only
        when it holds enough probability mass AND clears the absolute
        similarity floor. A cluster's score is the max cosine over its
        catalog views, so one good angle is enough."""
        reason = self.unavailable_reason
        if reason:
            return {"matches": [], "reason": reason}
        if isinstance(image, (str, Path)):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        query_vec = _normalize(self._embed(image))
        sims = {cid: max((_cosine(query_vec, v) for v in vecs), default=0.0)
                for cid, vecs in self._catalog_embeddings().items()}
        if not sims:
            return {"matches": [], "reason": "catalog is empty"}
        peak = max(sims.values())
        weights = {cid: math.exp((s - peak) / temperature) for cid, s in sims.items()}
        total = sum(weights.values())
        ranked = sorted(sims, key=lambda cid: -sims[cid])[:top]
        matches = [{"cluster_id": cid,
                    "score": round(sims[cid], 4),
                    "confidence": round(weights[cid] / total, 4)}
                   for cid in ranked]
        top1 = matches[0]
        top1["confident"] = (top1["confidence"] >= min_confidence
                             and top1["score"] >= min_similarity)
        for m in matches[1:]:
            m["confident"] = False
        # Gate the suggestion list too: CLIP cosines compress, so an unrelated
        # photo still scores ~0.6-0.71 against the whole catalog. Showing
        # those as "visually similar" reads as nonsense (a jeans photo
        # listing headphones), so anything below SUGGESTION_FLOOR is dropped
        # and an empty list means "no suggestions", not "top-N whatever".
        matches = [m for m in matches if m["score"] >= SUGGESTION_FLOOR]
        out = {"matches": matches}
        if not top1["confident"]:
            out["reason"] = ("no confident match: photo is too different from the "
                             "tracked catalog (try a clearer, front-facing shot)")
        return out
