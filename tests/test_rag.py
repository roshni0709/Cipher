"""RAG pipeline tests. All run fully offline with deterministic backends
(TF-IDF retrieval + template generation): no model downloads, no network.
"""
import pytest

from pricematch.buy_or_wait import BuyOrWait
from pricematch.db.store import Store
from pricematch.pipeline import seed
from pricematch.rag.corpus import build_corpus
from pricematch.rag.generator import generate
from pricematch.rag.pipeline import RagPipeline
from pricematch.rag.retriever import Retriever


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rag") / "rag.db")
    seed(path)
    return path


@pytest.fixture(scope="module")
def docs(db_path):
    store = Store(db_path)
    bow = BuyOrWait()
    try:
        return build_corpus(store, bow=bow)
    finally:
        store.close()


def test_corpus_has_offer_and_intel_docs(db_path, docs):
    store = Store(db_path)
    try:
        n_offers = len(store.offers())
        n_clusters = store.offers()["cluster_id"].nunique()
    finally:
        store.close()
    types = [d.meta["type"] for d in docs]
    assert types.count("offer") == n_offers
    assert types.count("intel") == n_clusters
    offer = next(d for d in docs if d.meta["type"] == "offer")
    assert "landed cost" in offer.text and "₹" in offer.text
    intel = next(d for d in docs if d.meta["type"] == "intel")
    assert "verdict" in intel.text.lower() and "Price check" in intel.text


def test_tfidf_retriever_ranks_relevant_first(docs):
    r = Retriever(docs, backend="tfidf")
    top_doc, score = r.retrieve("boAt airdopes earbuds price", k=4)[0]
    assert "boat" in top_doc.meta.get("cluster_id", "") or "boAt" in top_doc.text
    assert score > 0.05
    top_doc2, _ = r.retrieve("ryzen laptop", k=4)[0]
    assert "hp" in top_doc2.meta.get("cluster_id", "")


def test_template_generator_is_grounded_and_cites(docs):
    r = Retriever(docs, backend="tfidf")
    # k=8: with a 100-product catalog the intel chunk competes with many more
    # offer chunks; the production pipeline narrows the pool via anchoring
    retrieved = r.retrieve("boAt airdopes under 1500", k=8)
    out = generate("boAt airdopes under 1500", retrieved, backend="template")
    assert out["generator"] == "template"
    assert "[1]" in out["answer"]
    # the boAt intel chunk says WAIT (fake-sale pattern) and must be quotable
    intel = next(d for d, _ in retrieved
                 if d.meta.get("type") == "intel" and d.meta.get("cluster_id") == "boat-airdopes-141")
    assert intel.meta["recommendation"] in out["answer"]


def test_generator_no_match(docs):
    r = Retriever(docs, backend="tfidf")
    retrieved = r.retrieve("washing machine", k=4)
    out = generate("washing machine", retrieved, backend="template")
    assert "retrieved nothing" in out["answer"]


def test_pipeline_end_to_end_offline(db_path):
    pipe = RagPipeline(db_path, retriever_backend="tfidf",
                       generator_backend="template", bow=BuyOrWait())
    result = pipe.answer("boAt airdopes under 1500")
    assert result["retriever"] == "tfidf"
    assert result["generator"] == "template"
    assert result["retrieved"], "expected retrieved chunks"
    assert "[1]" in result["answer"] and "₹" in result["answer"]


def test_corpus_reflects_collected_data(db_path, docs):
    # collector-written offers get their own cluster ids; corpus must cover them
    cluster_ids = {d.meta["cluster_id"] for d in docs}
    assert "maroon-kurti" in cluster_ids  # apparel stretch-goal cluster
