"""Grounded generation: turn retrieved chunks into an answer.

Two interchangeable backends:

- ``flan-t5``: google/flan-t5-small (~310 MB download on first use, then
  cached) writes fluent text conditioned on the numbered chunks and is
  asked to cite them as [n]. Needs requirements-ml.txt.
- ``template``: deterministic, extractive-grounded composition. Every
  sentence is built from the structured fields of one retrieved chunk,
  so it cannot say anything the data didn't. This is what the tests
  exercise, and the automatic fallback whenever the LLM is unavailable
  or errors.

Both return {"answer": str, "generator": str}; the caller displays the
retrieved chunks next to the answer, so citations are always verifiable.
"""
from __future__ import annotations

from .corpus import Document


def _money(v: float) -> str:
    return f"₹{v:,.0f}"


def _no_match_answer(query: str) -> dict:
    return {
        "answer": (f"I retrieved nothing in the tracked catalog for “{query}”. "
                   "Try a brand or model that has been collected "
                   "(the demo tracks boAt, Redmi, Noise, HP, Pigeon and one kurti), "
                   "or run the collector to add it."),
        "generator": "template",
    }


def _template(query: str, retrieved: list[tuple[Document, float]]) -> dict:
    if not retrieved:
        return _no_match_answer(query)
    sentences = []
    for i, (doc, _score) in enumerate(retrieved, start=1):
        m = doc.meta
        if m.get("type") == "intel":
            flags = ", ".join(m.get("flags", []))
            sentences.append(
                f"**{m['cluster_id']}** — best true landed cost is at **{m['best_site']}** "
                f"[{i}]. Price check: {flags} [{i}]. "
                f"Buy-or-wait: **{m['recommendation']}** — {m['reason']} [{i}]."
            )
        else:
            sentences.append(f"{doc.text} [{i}]")
    intro = f"Grounded in {len(retrieved)} retrieved chunk(s) from the tracked catalog:\n"
    outro = ("\nEvery sentence above cites the chunk it came from (listed under "
             "*Retrieved context* below), so nothing here is invented.")
    return {"answer": intro + "\n".join(f"* {s}" for s in sentences) + "\n" + outro,
            "generator": "template"}


def _flan(query: str, retrieved: list[tuple[Document, float]], llm) -> dict:
    # No [n] markers inside the context: small FLAN models degenerate and
    # echo them. Citations are attached after generation instead.
    context = "\n".join(doc.text for doc, _s in retrieved)
    prompt = (
        "You are a careful shopping assistant. Answer the question in two or "
        "three sentences using only the facts in the context. Mention exact "
        "prices in rupees and the buy-or-wait verdict when present.\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    out = llm(prompt).strip()
    if not _passes_quality_gate(out, retrieved):
        raise RuntimeError("llm output failed the grounding quality gate")
    cited = ", ".join(f"[{i}] {doc.meta.get('cluster_id', doc.doc_id)}"
                      for i, (doc, _s) in enumerate(retrieved, start=1))
    return {"answer": f"{out}\n\n*Sources: {cited} — see Retrieved context below.*",
            "generator": "flan-t5-small"}


def _passes_quality_gate(out: str, retrieved: list[tuple[Document, float]]) -> bool:
    """Reject degenerate or ungrounded small-LLM output so the caller can
    fall back to the extractive writer: answers must be sentence-length,
    mention at least one number (prices are the point), not be a bare
    citation token, and not just copy one retrieved chunk verbatim."""
    words = out.split()
    if len(words) < 12:
        return False
    if not any(ch.isdigit() for ch in out):
        return False
    if out.strip("[]() .") == "":
        return False
    for doc, _s in retrieved:
        if out.strip() and out.strip() in doc.text:
            return False
    return True


def generate(query: str, retrieved: list[tuple[Document, float]],
             backend: str = "auto", llm=None) -> dict:
    if not retrieved:
        return _no_match_answer(query)
    if backend in ("auto", "flan-t5") and llm is not None:
        try:
            return _flan(query, retrieved, llm)
        except Exception:
            pass  # any LLM failure degrades to the grounded template
    return _template(query, retrieved)
