"""Photo-to-price tests with a stub embedder (no model weights needed)."""
import pytest
from PIL import Image

from pricematch.photo import (PhotoSearcher, catalog_photos, MIN_CONFIDENCE,
                              TEMPERATURE)


@pytest.fixture()
def stub_dir(tmp_path):
    """Catalog 'photos' whose stub embedding is their mean color; red-phone
    gets TWO photos to exercise multi-photo averaging."""
    d = tmp_path / "photos"
    d.mkdir()
    photos = {"red-phone": (220, 30, 30), "red-phone-2": (210, 40, 35),
              "blue-buds": (30, 30, 220), "green-flask": (30, 200, 30)}
    for name, rgb in photos.items():
        Image.new("RGB", (64, 64), rgb).save(d / f"{name}.jpg")
    return d


def _mean_color_embed(image) -> list[float]:
    px = list(image.convert("RGB").resize((8, 8)).getdata())
    n = len(px)
    return [sum(c[i] for c in px) / n / 255.0 for i in range(3)]


def test_catalog_photos_groups_multi_shots(stub_dir):
    photos = catalog_photos(stub_dir)
    assert photos["red-phone"] == [stub_dir / "red-phone.jpg",
                                   stub_dir / "red-phone-2.jpg"]
    assert photos["blue-buds"] == [stub_dir / "blue-buds.jpg"]


def test_match_ranks_correct_cluster(stub_dir):
    ps = PhotoSearcher(embed_image=_mean_color_embed, photos_dir=stub_dir)
    query = Image.new("RGB", (64, 64), (212, 38, 33))  # red-ish
    out = ps.match(query)
    top = out["matches"][0]
    assert top["cluster_id"] == "red-phone"
    assert top["confident"]
    assert top["confidence"] > 0.5
    assert all(top["score"] > m["score"] for m in out["matches"][1:])


def test_calibrated_confidence_separates(stub_dir):
    """Decisive matches get high probability mass; compressed raw cosines
    must not look like a near-tie in the calibrated output."""
    ps = PhotoSearcher(embed_image=_mean_color_embed, photos_dir=stub_dir)
    out = ps.match(Image.new("RGB", (64, 64), (215, 35, 32)))
    top = out["matches"][0]
    assert top["cluster_id"] == "red-phone"
    assert top["confidence"] > 0.9  # decisive match, calibrated high


def test_match_reports_unconfident(stub_dir):
    ps = PhotoSearcher(embed_image=_mean_color_embed, photos_dir=stub_dir)
    # black embeds to the zero vector: every similarity 0, softmax uniform
    out = ps.match(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert out["matches"] == []  # all-zero similarities sit under the floor
    assert "reason" in out


def test_suggestions_below_floor_are_dropped(tmp_path):
    """CLIP cosines compress, so an unrelated photo still scores ~0.6-0.71
    against the catalog. Those must not surface as 'visually similar': a
    jeans upload must never list headphones. Signed-vector stub models that
    geometry: red-phone at cosine 1.0, blue-buds at 0.71 (just junk)."""
    from pricematch.photo import SUGGESTION_FLOOR
    d = tmp_path / "photos"
    d.mkdir()
    vecs = {"red-phone": [1.0, 0.0, 0.0], "blue-buds": [0.71, 0.7041, 0.0],
            "green-flask": [0.0, 0.0, 1.0]}
    colors = {"red-phone": (200, 10, 10), "blue-buds": (10, 10, 200),
              "green-flask": (10, 200, 10)}
    for name, rgb in colors.items():
        Image.new("RGB", (64, 64), rgb).save(d / f"{name}.jpg")
    by_color = {v: k for k, v in colors.items()}

    def _embed(image):
        px = image.convert("RGB").getpixel((0, 0))
        # JPEG nudges pixels a little; snap to the nearest known stub color
        name = min(by_color, key=lambda c: sum((a - b) ** 2
                                               for a, b in zip(px, c)))
        return vecs[by_color[name]]

    ps = PhotoSearcher(embed_image=_embed, photos_dir=d)
    q = Image.new("RGB", (64, 64), colors["red-phone"])
    out = ps.match(q)
    assert [m["cluster_id"] for m in out["matches"]] == ["red-phone"]
    assert out["matches"][0]["confident"]
    assert all(m["score"] >= SUGGESTION_FLOOR for m in out["matches"])


def test_near_tie_at_high_similarity_abstains(tmp_path):
    """Regression for the wrong-label incident: two near-identical earbuds
    at 0.90/0.90, or a same-colour wrong-brand shoe against one catalog
    shoe - high cosine, split probability mass. The old 0.45 mass bar
    rendered these as "Matched"; the gate must now abstain while still
    listing both as visually similar."""
    d = tmp_path / "photos"
    d.mkdir()
    vecs = {"tan-shoe-a": [1.0, 0.0, 0.0],
            "tan-shoe-b": [0.9995, 0.0316, 0.0],
            "green-flask": [0.0, 0.0, 1.0]}
    colors = {"tan-shoe-a": (200, 150, 90), "tan-shoe-b": (198, 152, 92),
              "green-flask": (10, 200, 10)}
    for name, rgb in colors.items():
        Image.new("RGB", (64, 64), rgb).save(d / f"{name}.jpg")
    by_color = {v: k for k, v in colors.items()}

    def _embed(image):
        px = image.convert("RGB").getpixel((0, 0))
        name = min(by_color, key=lambda c: sum((a - b) ** 2
                                               for a, b in zip(px, c)))
        return vecs[by_color[name]]

    ps = PhotoSearcher(embed_image=_embed, photos_dir=d)
    q = Image.new("RGB", (64, 64), (199, 151, 91))  # same tan family
    out = ps.match(q)
    assert len(out["matches"]) == 2  # both tan shoes stay visible as similar
    assert not any(m["confident"] for m in out["matches"])
    assert "reason" in out


def test_no_photos_bundled(tmp_path):
    ps = PhotoSearcher(embed_image=_mean_color_embed, photos_dir=tmp_path / "empty")
    out = ps.match(Image.new("RGB", (16, 16)))
    assert out["matches"] == [] and "no catalog photos" in out["reason"]


def test_bundled_catalog_photos_exist():
    photos = catalog_photos()
    assert len(photos) >= 11, "expected bundled photos for the demo clusters"
    # all five audio products must be represented for the demo story
    for cid in ("boat-airdopes-141", "noise-buds-vs104", "realme-buds-t110",
                "jbl-wave-flex", "boat-rockerz-450"):
        assert cid in photos, cid
    # every photographed cluster must exist in the sample catalog (no orphans
    # and no stand-ins under a wrong name)
    from pricematch.data.sample import CLUSTERS
    known = {c["cluster_id"] for c in CLUSTERS}
    for cid in photos:
        assert cid in known, f"photo cluster {cid!r} has no catalog cluster"
    for paths in photos.values():
        for p in paths:
            Image.open(p).verify()  # valid JPEGs


def test_confidence_constants_sane():
    assert 0.0 < MIN_CONFIDENCE < 1.0
    assert 0.0 < TEMPERATURE < 0.2


def test_end_to_end_match_feeds_cluster_into_toolset(stub_dir, tmp_path):
    """The matched cluster id plugs straight into the existing pipeline."""
    from pricematch.pipeline import seed
    from pricematch.agent.tools import Toolset
    db = str(tmp_path / "p.db")
    seed(db)
    ps = PhotoSearcher(embed_image=_mean_color_embed, photos_dir=stub_dir)
    hit = ps.match(Image.new("RGB", (32, 32), (213, 37, 34)))["matches"][0]
    assert hit["cluster_id"] == "red-phone"
    ts = Toolset(db)
    ranked = ts.landed_costs("boat-airdopes-141")
    ts.close()
    assert ranked and ranked[0]["landed_cost"] > 0
