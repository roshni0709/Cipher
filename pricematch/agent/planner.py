"""Agentic tool-calling: the agent decides WHICH tools to call for a
question, runs them, and turns their outputs into grounded context.

The RulePlanner is deliberately transparent (regex intents, no LLM
needed, fully offline): it parses a budget, searches the catalog, then
fires only the tools the question implies — landed-cost comparison for
any shopping question, price history for trend questions, fake-discount
checks for "is this sale real", buy-or-wait for timing, alerts and
review summaries when those words appear. Every decision is recorded in
a plan trace the UI can show, so the "agent" is explainable rather than
a black box. Tools register in a plain registry, so new capabilities
(alerts, reviews) join without touching the planner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PRICE_RE = re.compile(r"(?:under|below|less than|max)\s*(?:rs\.?|inr|₹)?\s*([\d,]+)", re.I)

INTENT_RULES = [
    (re.compile(r"history|trend|chart|graph|over time", re.I), "price_history"),
    (re.compile(r"fake|discount|genuine|scam|inflat", re.I), "price_check"),
    (re.compile(r"buy|wait|when|timing|drop soon|good time", re.I), "buy_or_wait"),
    (re.compile(r"alert|notify|price.?drop", re.I), "price_alerts"),
    (re.compile(r"review|pros|cons|opinion|feedback", re.I), "review_summary"),
]


@dataclass
class Step:
    tool: str
    args: dict
    rationale: str


@dataclass
class PlanResult:
    query: str
    steps: list[Step] = field(default_factory=list)
    tool_docs: list[dict] = field(default_factory=list)  # rag.corpus.Document-shaped dicts
    trace: list[str] = field(default_factory=list)


class ToolRegistry:
    """name -> (description, callable). Shared by the planner and the
    LLM agent so both speak about the same tool set."""

    def __init__(self):
        self._tools: dict[str, tuple[str, object]] = {}

    def register(self, name: str, description: str, fn) -> None:
        self._tools[name] = (description, fn)

    def describe(self) -> list[dict]:
        return [{"name": n, "description": d} for n, (d, _f) in sorted(self._tools.items())]

    def call(self, name: str, **args):
        return self._tools[name][1](**args)


class RulePlanner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def plan(self, query: str) -> list[Step]:
        steps = [Step("search_products", {"query": _strip_budget(query)},
                      "find the products the question is about")]
        intents = [tool for rx, tool in INTENT_RULES if rx.search(query)]
        # a plain shopping question still needs comparison + verdict
        if not intents or "buy_or_wait" not in intents:
            intents.append("buy_or_wait")
        if not any(t in intents for t in ("price_check", "price_history", "price_alerts", "review_summary")):
            intents.insert(0, "price_check")
        seen = set()
        for tool in intents:
            if tool in seen:
                continue
            seen.add(tool)
            steps.append(Step(tool, {}, _RATIONALE[tool]))
        return steps

    def run(self, query: str, k_docs: int = 3,
            cluster_ids: set[str] | None = None) -> PlanResult:
        result = PlanResult(query=query)
        budget = _budget(query)
        if budget is not None:
            result.trace.append(f"parsed budget: ₹{budget:,.0f}")
        if cluster_ids:
            # The pipeline already resolved the named product(s). Do not let
            # broad SQL keyword search replace them with brand siblings.
            hits = [offer for cid in sorted(cluster_ids)
                    for offer in self.registry.call("cluster_offers", cluster_id=cid)]
        else:
            hits = self.registry.call("search_products", query=_strip_budget(query))
        result.trace.append(f"search_products → {len(hits)} offer(s)")
        if not hits:
            return result
        clusters = {}
        for hit in hits:
            clusters.setdefault(hit["cluster_id"], []).append(hit)
        for step in self.plan(query)[1:]:
            for cluster_id in list(clusters)[:k_docs]:
                doc = self._run_step(step, cluster_id, budget)
                if doc:
                    result.tool_docs.append(doc)
            result.steps.append(step)
            result.trace.append(f"{step.tool} → {step.rationale}")
        return result

    def _run_step(self, step: Step, cluster_id: str, budget: float | None) -> dict | None:
        ranked = self.registry.call("landed_costs", cluster_id=cluster_id)
        if not ranked:
            return None
        best = ranked[0]
        offers = self.registry.call("cluster_offers", cluster_id=cluster_id)
        best_offer = next((o for o in offers if o["site"] == best["site"]), offers[0])
        offer_id = best_offer["offer_id"]
        if step.tool == "price_check":
            check = self.registry.call("price_check", offer_id=offer_id)
            text = (f"[tool: price_check] {cluster_id} at {best['site']}: "
                    f"flags {', '.join(check['flags'])} — {check['detail']}")
        elif step.tool == "buy_or_wait":
            bow = self.registry.call("buy_or_wait", offer_id=offer_id)
            text = (f"[tool: buy_or_wait] {cluster_id} at {best['site']} "
                    f"(best landed ₹{best['landed_cost']:,.0f}): {bow['recommendation']} — "
                    f"{bow['reason']} (drop probability {bow['drop_probability']:.0%})")
        elif step.tool == "price_history":
            hist = self.registry.call("price_history", offer_id=offer_id)
            if hist["min"] is None:
                return None
            text = (f"[tool: price_history] {cluster_id} at {best['site']}: "
                    f"{len(hist['points'])} daily prices, min ₹{hist['min']:,.0f}, "
                    f"max ₹{hist['max']:,.0f}, latest ₹{hist['points'][-1]['price']:,.0f}")
        else:  # tools added later (alerts, reviews) return a ready dict with text
            out = self.registry.call(step.tool, cluster_id=cluster_id)
            if not out or not out.get("text"):
                return None
            text = out["text"]
        if budget is not None:
            within = "within" if best["landed_cost"] <= budget else "ABOVE"
            text += f" Best landed cost ₹{best['landed_cost']:,.0f} is {within} the ₹{budget:,.0f} budget."
        return {"doc_id": f"tool:{step.tool}:{cluster_id}", "text": text,
                "meta": {"type": "tool", "tool": step.tool, "cluster_id": cluster_id}}


_RATIONALE = {
    "landed_costs": "compare true landed costs",
    "price_check": "the question asks whether the discount is real",
    "buy_or_wait": "the question is about purchase timing (default for shopping questions)",
    "price_history": "the question asks about the price trend",
    "price_alerts": "the question mentions drops or alerts",
    "review_summary": "the question asks what reviewers think",
}


def _budget(query: str) -> float | None:
    m = PRICE_RE.search(query)
    return float(m.group(1).replace(",", "")) if m else None


def _strip_budget(query: str) -> str:
    return PRICE_RE.sub("", query).strip() or query


def _review_summary_tool(cluster_id: str) -> dict | None:
    """Planner-facing wrapper: pros/cons + trust signal as tool context."""
    from ..review_intel import summarize, trust  # local import: keeps sklearn out of cold paths
    s = summarize(cluster_id)
    if s["review_count"] == 0:
        return None
    t = trust(cluster_id)
    parts = [f"[tool: review_summary] {cluster_id}: {s['review_count']} reviews, "
             f"avg {s['avg_rating']} stars; review trust: {t['label']} ({t['score']}/100)."]
    if t["flags"]:
        parts.append("Suspicious review patterns (Manipulation flags): " + "; ".join(f["detail"] for f in t["flags"]) + ".")
    else:
        parts.append("No manipulation signals detected in this review set.")
    parts.append("Heuristic signals, not proof that reviews are fake or genuine. "
                 "Demo review data, not live store reviews.")
    if s["pros"]:
        parts.append("Pros: " + " | ".join(s["pros"]))
    if s["cons"]:
        parts.append("Cons: " + " | ".join(s["cons"]))
    return {"text": " ".join(parts)}


def _price_alerts_tool(toolset, cluster_id: str) -> dict | None:
    """Planner-facing wrapper: current drop-alert status for one cluster."""
    from ..alerts import scan, format_digest
    offers = toolset.cluster_offers(cluster_id)
    hists = {o["offer_id"]: toolset.price_history(o["offer_id"])["points"] for o in offers}
    titles = {o["offer_id"]: f"{cluster_id} at {o['site']}" for o in offers}
    digest = format_digest(scan(hists), titles)
    return {"text": f"[tool: price_alerts] {digest['subject']}. {digest['body']}"}


def default_registry(toolset) -> ToolRegistry:
    """The tool set the planner can choose from, wired to one Toolset."""
    reg = ToolRegistry()
    reg.register("search_products", "Search tracked offers by keywords", toolset.search_products)
    reg.register("landed_costs", "Rank a cluster's offers by true landed cost", toolset.landed_costs)
    reg.register("cluster_offers", "List all offers in one cluster", toolset.cluster_offers)
    reg.register("price_check", "Fake-discount analysis for one offer", toolset.price_check)
    reg.register("buy_or_wait", "Buy-or-wait prediction for one offer", toolset.buy_or_wait)
    reg.register("price_history", "Daily price history for one offer", toolset.price_history)
    reg.register("review_summary", "Extractive pros/cons + review-trust signal for one product",
                 _review_summary_tool)
    reg.register("price_alerts", "Whether any offer of one product has a live price-drop alert",
                 lambda cluster_id: _price_alerts_tool(toolset, cluster_id))
    return reg
