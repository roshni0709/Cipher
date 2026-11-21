"""Agentic planner tests: plan shape, budget parsing, tool execution and
pipeline integration, all offline and deterministic."""
import pytest

from pricematch.agent.planner import RulePlanner, _budget, _strip_budget, default_registry
from pricematch.agent.tools import Toolset
from pricematch.buy_or_wait import BuyOrWait
from pricematch.pipeline import seed
from pricematch.rag.pipeline import RagPipeline


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("planner") / "p.db")
    seed(path)
    return path


@pytest.fixture()
def planner(db_path):
    toolset = Toolset(db_path)
    yield RulePlanner(default_registry(toolset))
    toolset.close()


def test_budget_parsing():
    assert _budget("boAt airdopes under 1500") == 1500.0
    assert _budget("laptop below Rs 40,000") == 40000.0
    assert _budget("kettle") is None
    assert _strip_budget("boAt airdopes under 1500") == "boAt airdopes"


def test_plan_picks_tools_by_intent(planner):
    steps = [s.tool for s in planner.plan("is the boAt sale fake? should I wait?")]
    assert "price_check" in steps and "buy_or_wait" in steps
    steps2 = [s.tool for s in planner.plan("show me the kettle price history")]
    assert "price_history" in steps2


def test_run_produces_tool_docs_and_trace(planner):
    out = planner.run("boAt airdopes under 1500")
    assert out.trace and "budget" in out.trace[0]
    assert out.tool_docs, "expected grounded tool documents"
    doc = next(d for d in out.tool_docs if d["meta"]["tool"] == "buy_or_wait")
    assert "WAIT" in doc["text"] or "BUY" in doc["text"]
    assert "₹" in doc["text"]
    budgeted = [d for d in out.tool_docs if "budget" in d["text"]]
    assert budgeted, "budget comparison must appear in tool docs"


def test_run_unknown_product_is_graceful(planner):
    out = planner.run("washing machine under 20000")
    assert out.tool_docs == []


def test_pipeline_includes_plan_and_tool_chunks(db_path):
    pipe = RagPipeline(db_path, retriever_backend="tfidf",
                       generator_backend="template", bow=BuyOrWait())
    result = pipe.answer("boAt airdopes under 1500")
    assert result["plan"], "plan trace should be populated"
    tool_chunks = [d for d, _ in result["retrieved"] if d.meta.get("type") == "tool"]
    assert tool_chunks, "tool outputs should be first-class grounded context"
    assert "[1]" in result["answer"]
