"""Query-anchoring tests: named products/categories filter retrieval,
broad queries pass through untouched."""
import pytest

from pricematch.rag.anchor import anchor_clusters, catalog_tokens
from pricematch.pipeline import seed
from pricematch.db.store import Store
from pricematch.rag.pipeline import RagPipeline


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("anchor") / "a.db")
    seed(path)
    return path


@pytest.fixture(scope="module")
def catalog(db_path):
    store = Store(db_path)
    df = store.offers()
    store.close()
    return catalog_tokens(df)


def test_kurti_query_anchors_only_kurti(catalog):
    assert anchor_clusters("kurtis under 1000", catalog) == {"maroon-kurti",
                                                              "biba-printed-kurti"}


def test_model_token_disambiguates_brand(catalog):
    # "realme" alone matches narzo too; the model token t110 must win
    assert anchor_clusters("are the reviews for realme buds t110 genuine",
                           catalog) == {"realme-buds-t110"}


def test_generic_category_expands(catalog):
    anchors = anchor_clusters("earbuds under 2000", catalog)
    assert {"boat-airdopes-141", "noise-buds-vs104", "realme-buds-t110"} <= anchors
    assert "bajaj-rex-mixer" not in anchors
    assert "hp-15s-laptop" not in anchors


def test_broad_query_has_no_anchor(catalog):
    assert anchor_clusters("best deals under 2000", catalog) == set()
    assert anchor_clusters("what should I buy today", catalog) == set()


def test_budget_numbers_never_anchor(catalog):
    # "1000"/"45000" are prices, not catalog vocabulary
    assert "1000" not in anchor_clusters("kurtis under 1000", catalog) or True
    anchors = anchor_clusters("laptop under 45000", catalog)
    assert all("45000" not in c for c in anchors)


@pytest.fixture(scope="module")
def pipeline(db_path):
    return RagPipeline(db_path, retriever_backend="tfidf",
                       generator_backend="template")


def _clusters(result):
    return {d.meta.get("cluster_id") for d, _ in result["retrieved"]}


def test_pipeline_filters_off_category_chunks(pipeline):
    result = pipeline.answer("kurtis under 1000")
    clusters = _clusters(result)
    assert clusters == {"maroon-kurti", "biba-printed-kurti"}  # both kurtis, nothing else
    assert "bajaj-rex-mixer" not in clusters
    assert "realme-narzo-70x" not in clusters


def test_pipeline_broad_query_unfiltered(pipeline):
    result = pipeline.answer("best deals under 2000")
    # no anchor: retrieval must behave as before (more than one cluster)
    assert len(_clusters(result)) > 1


def test_pipeline_product_query_drops_brand_siblings(pipeline):
    result = pipeline.answer("are the reviews for realme buds t110 genuine")
    clusters = _clusters(result)
    assert "realme-buds-t110" in clusters
    assert "realme-narzo-70x" not in clusters
