"""Review summarizer + fake-review detector tests (deterministic sample data)."""
import pytest

from pricematch.review_intel import summarize, trust
from pricematch.agent.planner import RulePlanner, default_registry
from pricematch.agent.tools import Toolset
from pricematch.pipeline import seed


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("reviews") / "r.db")
    seed(path)
    return path


@pytest.fixture()
def planner(db_path):
    toolset = Toolset(db_path)
    yield RulePlanner(default_registry(toolset))
    toolset.close()


def test_summarize_returns_pros_and_cons():
    s = summarize("boat-airdopes-141")
    assert s["review_count"] == 7
    assert 1 <= len(s["pros"]) <= 3
    assert 1 <= len(s["cons"]) <= 3
    # pros come from high-rated reviews, cons from low-rated ones
    assert any("battery" in p.lower() or "bass" in p.lower() for p in s["pros"])
    assert any("mic" in c.lower() or "charging" in c.lower() for c in s["cons"])


def test_summarize_is_deterministic():
    assert summarize("milton-thermosteel-1l") == summarize("milton-thermosteel-1l")


def test_summarize_unknown_cluster_is_empty():
    s = summarize("no-such-cluster")
    assert s["review_count"] == 0 and s["pros"] == [] and s["cons"] == []


def test_trust_flags_burst_pattern():
    t = trust("realme-buds-t110")
    flags = {f["flag"] for f in t["flags"]}
    assert "review_burst" in flags
    assert "copy_paste_text" in flags
    assert t["label"] == "low"
    assert t["score"] < 50


def test_trust_flags_copy_paste():
    t = trust("jbl-wave-flex")
    flags = {f["flag"] for f in t["flags"]}
    assert "copy_paste_text" in flags
    assert t["score"] < 100


def test_trust_clean_cluster_scores_high():
    t = trust("milton-thermosteel-1l")
    assert t["flags"] == []
    assert t["score"] == 100
    assert t["label"] == "high"


def test_trust_unknown_cluster():
    t = trust("no-such-cluster")
    assert t["score"] is None and t["label"] == "no reviews"


def test_planner_review_summary_tool(planner):
    result = planner.run("what do reviewers say about the boAt Airdopes 141")
    tools = [d["meta"]["tool"] for d in result.tool_docs]
    assert "review_summary" in tools
    doc = next(d for d in result.tool_docs if d["meta"]["tool"] == "review_summary")
    assert "review trust" in doc["text"]
    assert "Pros:" in doc["text"]


def test_planner_surfaces_manipulation_flags(planner):
    result = planner.run("are the reviews for realme buds t110 genuine", k_docs=5)
    doc = next((d for d in result.tool_docs
                if d["meta"]["tool"] == "review_summary"
                and d["meta"]["cluster_id"] == "realme-buds-t110"), None)
    assert doc is not None
    assert "Manipulation flags" in doc["text"]
