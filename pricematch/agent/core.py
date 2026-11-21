"""The LLM agent layer.

With OPENAI_API_KEY set (and `openai` installed): a real tool-calling
agent that plans, calls the project tools, and explains its pick.
Without them: a deterministic rule agent that parses queries like
"boAt airdopes under 1500" and walks the same tools, so the whole thing
runs offline and in CI.
"""
from __future__ import annotations

import os
import re

from .tools import Toolset

PRICE_RE = re.compile(r"(?:under|below|less than|max)\s*(?:rs\.?|inr|\u20b9)?\s*([\d,]+)", re.I)


def run_agent(query: str, toolset: Toolset) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _llm_agent(query, toolset)
        except ImportError:
            pass  # fall through to the rule agent
    return _rule_agent(query, toolset)


def _rule_agent(query: str, toolset: Toolset) -> str:
    budget = None
    m = PRICE_RE.search(query)
    if m:
        budget = float(m.group(1).replace(",", ""))
    clean = PRICE_RE.sub("", query)
    hits = toolset.search_products(clean)
    if not hits:
        return (f"I searched the tracked catalog for '{clean.strip()}' and found nothing. "
                "Try a model name or brand that has been collected.")
    clusters: dict[str, list[dict]] = {}
    for hit in hits:
        clusters.setdefault(hit["cluster_id"], []).append(hit)

    lines = [f"Query: {query}", f"Found {len(clusters)} matching product(s).", ""]
    for cluster_id, _ in list(clusters.items())[:3]:
        ranked = toolset.landed_costs(cluster_id)
        if budget is not None:
            ranked = [r for r in ranked if r["landed_cost"] <= budget] or ranked
        best = ranked[0]
        lines.append(f"* {cluster_id}")
        lines.append(f"  Best buy: {best['site']} at Rs {best['landed_cost']} landed "
                     f"(listed Rs {best['price']}, shipping Rs {best['shipping']}, "
                     f"coupon saves Rs {best['coupon_value']}). {best['risk']}.")
        offers = toolset.cluster_offers(cluster_id)
        best_offer = next((o for o in offers if o["site"] == best["site"]), offers[0])
        check = toolset.price_check(best_offer["offer_id"])
        lines.append(f"  Price check: {', '.join(check['flags'])}. {check['detail']}")
        bow = toolset.buy_or_wait(best_offer["offer_id"])
        lines.append(f"  Verdict: {bow['recommendation']} - {bow['reason']}")
        lines.append("")
    lines.append("Recommendation: take the best landed cost above; it already includes "
                 "shipping, coupons and return risk.")
    return "\n".join(lines)


def _llm_agent(query: str, toolset: Toolset) -> str:
    from openai import OpenAI

    client = OpenAI()
    messages = [
        {"role": "system", "content":
         "You are a shopping analyst. Use the tools to find the user's product, "
         "compare TRUE landed costs (price + shipping - coupon, noting return risk), "
         "check for fake discounts, and advise buy-or-wait. Explain the pick in "
         "5 short lines max, with rupee amounts."},
        {"role": "user", "content": query},
    ]
    for _ in range(6):
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=Toolset.SCHEMAS,
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content or ""
        for call in msg.tool_calls:
            import json

            result = toolset.call(call.function.name, json.loads(call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)})
    return "I could not finish within the tool-call budget."
