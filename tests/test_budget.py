"""Budget-constraint answers: in-budget names the cheapest fit, over-budget
says so plainly and names the cheapest available."""
import pytest

from pricematch.pipeline import seed
from pricematch.agent.tools import Toolset
from pricematch.rag.pipeline import RagPipeline


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("budget") / "b.db")
    seed(path)
    return path


@pytest.fixture(scope="module")
def pipeline(db_path):
    return RagPipeline(db_path, retriever_backend="tfidf",
                       generator_backend="template")


def test_over_budget_says_nothing_fits_and_names_cheapest(db_path, pipeline):
    ts = Toolset(db_path)
    cheapest = min(ts.landed_costs("maroon-kurti"), key=lambda r: r["landed_cost"])
    ts.close()
    assert cheapest["landed_cost"] > 700, "fixture assumption: kurti costs over 700"

    result = pipeline.answer("kurtis under 700")
    assert "nothing here fits the ₹700 budget" in result["answer"]
    assert f"₹{cheapest['landed_cost']:,.0f}" in result["answer"]
    assert cheapest["site"] in result["answer"]
    # the budget verdict is a citable chunk, first in the grounded context
    first = result["retrieved"][0][0]
    assert first.meta.get("tool") == "budget_check"


def test_in_budget_names_cheapest_fit(pipeline):
    result = pipeline.answer("kurtis under 1000")
    assert "fit the ₹1,000 budget" in result["answer"]
    assert "nothing here fits" not in result["answer"]


def test_in_budget_boat_still_works(pipeline):
    result = pipeline.answer("boAt airdopes under 1500")
    assert "fit the ₹1,500 budget" in result["answer"]


def test_no_budget_query_has_no_budget_check(pipeline):
    result = pipeline.answer("maroon kurti")
    assert "Budget check" not in result["answer"]


def test_broad_budget_query_without_anchor(pipeline):
    # no product named: budget doc needs anchors, so nothing is invented
    result = pipeline.answer("best deals under 2000")
    assert all(d.meta.get("tool") != "budget_check" for d, _ in result["retrieved"])
