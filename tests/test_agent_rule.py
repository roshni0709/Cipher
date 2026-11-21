from pricematch.agent.core import _rule_agent
from pricematch.agent.tools import Toolset
from pricematch.pipeline import seed


def test_rule_agent_end_to_end(tmp_path):
    db = str(tmp_path / "a.db")
    seed(db)
    ts = Toolset(db)
    out = _rule_agent("boAt airdopes under 1500", ts)
    assert "Best buy" in out
    assert "landed" in out
    assert "Verdict" in out
    ts.close()


def test_rule_agent_no_match(tmp_path):
    db = str(tmp_path / "b.db")
    seed(db)
    ts = Toolset(db)
    out = _rule_agent("ziggurat quantum flux capacitor", ts)
    assert "found nothing" in out
    ts.close()
