"""The RAG pipeline: corpus -> retrieve -> grounded generate.

Models are loaded lazily and held for the process lifetime (the demo
caches one RagPipeline via st.cache_resource). The corpus is rebuilt per
question — it is tiny (one doc per offer + one per cluster) — so answers
always reflect the current database, including freshly collected prices.
"""
from __future__ import annotations

import re

from ..db.store import Store
from .anchor import anchor_clusters, catalog_tokens
from .corpus import Document, build_corpus
from .generator import generate
from .retriever import Retriever


class RagPipeline:
    def __init__(self, db_path: str, retriever_backend: str = "auto",
                 generator_backend: str = "auto", bow=None,
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "google/flan-t5-small"):
        self.db_path = db_path
        self.retriever_backend = retriever_backend
        self.generator_backend = generator_backend
        self.bow = bow
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self._embed_model = None
        self._embed_tried = False
        self._llm = None
        self._llm_tried = False

    def _get_embed_model(self):
        """Load MiniLM once; None (and remembered failure) means TF-IDF mode."""
        if self.retriever_backend == "tfidf":
            return None
        if not self._embed_tried:
            self._embed_tried = True
            try:
                from sentence_transformers import SentenceTransformer

                self._embed_model = SentenceTransformer(self.embed_model_name)
            except Exception:
                self._embed_model = None
        return self._embed_model

    def _get_llm(self):
        """Load FLAN-T5 once; None (and remembered failure) means template mode.
        Uses AutoModelForSeq2SeqLM directly: the old text2text-generation
        pipeline was removed in transformers v5."""
        if self.generator_backend == "template":
            return None
        if not self._llm_tried:
            self._llm_tried = True
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(self.llm_model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)

                def llm(prompt: str, max_new_tokens: int = 160) -> str:
                    inputs = tok(prompt, return_tensors="pt", truncation=True,
                                 max_length=1024)
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False)
                    return tok.decode(out[0], skip_special_tokens=True)

                self._llm = llm
            except Exception:
                self._llm = None
        return self._llm

    def _get_bow(self):
        """Shared buy-or-wait model; created on first use when not injected."""
        if self.bow is None:
            from ..buy_or_wait import BuyOrWait

            self.bow = BuyOrWait()
        return self.bow

    def answer(self, query: str, k: int = 4, use_planner: bool = True) -> dict:
        store = Store(self.db_path)
        try:
            docs = build_corpus(store, bow=self._get_bow())
            catalog = catalog_tokens(store.offers())
        finally:
            store.close()
        embed_model = self._get_embed_model()
        # when the query clearly names products or a category, retrieve ONLY
        # from those clusters' chunks; broad queries keep the whole corpus.
        # The score floor drops away inside an anchored pool: the anchor
        # itself is the relevance signal, and a plural variant ("kurtis" vs
        # a "Kurti" title) must not leave the answer with zero context.
        anchors = anchor_clusters(query, catalog)
        pool = docs
        if anchors:
            anchored = [d for d in docs if d.meta.get("cluster_id") in anchors]
            if anchored:
                pool = anchored
        retriever = Retriever(pool,
                              backend=("minilm" if embed_model is not None else "tfidf")
                              if self.retriever_backend == "auto" else self.retriever_backend,
                              embed_model=embed_model)
        retrieved = retriever.retrieve(query, k=k, embed_model=embed_model,
                                       min_score=-1.0 if pool is not docs else 0.05)
        plan_trace: list[str] = []
        if use_planner:
            tool_docs, plan_trace = self._run_planner(query, store_path=self.db_path, anchors=anchors)
            if anchors:
                # planner search has the same off-category failure mode;
                # an empty tool list is fine, the plan trace still explains
                tool_docs = [d for d in tool_docs
                             if d["meta"].get("cluster_id") in anchors]
            # agent tool outputs are first-class grounded context, ahead of
            # the retrieved chunks
            retrieved = ([(Document(d["doc_id"], d["text"], d["meta"]), 1.0)
                          for d in tool_docs] + retrieved)[: k + len(tool_docs)]
        budget_doc = self._budget_doc(query, anchors, pool)
        if budget_doc is not None:
            retrieved = [(budget_doc, 1.0)] + retrieved
        # Review questions need the full deterministic trust result and flags,
        # not a small language model's optional paraphrase of price evidence.
        reviews = bool(re.search(r"review|feedback|opinion", query, re.I))
        result = generate(query, retrieved,
                          backend="template" if reviews else self.generator_backend,
                          llm=None if reviews else self._get_llm())
        result["retriever"] = retriever.backend
        result["retrieved"] = retrieved
        result["query"] = query
        result["plan"] = plan_trace
        return result

    @staticmethod
    def _budget_doc(query: str, anchors: set[str], pool: list[Document]) -> Document | None:
        """Explicit budget verdict as a citable chunk.

        Offer docs carry true landed costs, so a budget question can be
        answered exactly: either name the cheapest in-budget offer, or say
        plainly that nothing fits and name the cheapest available. Without
        this chunk the generator would happily cite over-budget prices.
        """
        from ..agent.planner import _budget

        budget = _budget(query)
        if budget is None or not anchors:
            return None
        offers = [d for d in pool
                  if d.meta.get("type") == "offer" and "landed_cost" in d.meta]
        if not offers:
            return None
        within = [d for d in offers if d.meta["landed_cost"] <= budget]
        if within:
            best = min(within, key=lambda d: d.meta["landed_cost"])
            text = (f"Budget check: {len(within)} offer(s) fit the "
                    f"₹{budget:,.0f} budget; cheapest is {best.meta['cluster_id']} "
                    f"at {best.meta['site']} (true landed cost "
                    f"₹{best.meta['landed_cost']:,.0f}).")
            named = best
        else:
            named = min(offers, key=lambda d: d.meta["landed_cost"])
            over = named.meta["landed_cost"] - budget
            text = (f"Budget check: nothing here fits the ₹{budget:,.0f} budget. "
                    f"Cheapest available: {named.meta['cluster_id']} at "
                    f"{named.meta['site']}, true landed cost "
                    f"₹{named.meta['landed_cost']:,.0f} "
                    f"(₹{over:,.0f} over budget).")
        return Document(doc_id=f"tool:budget_check:{named.meta['cluster_id']}",
                        text=text,
                        meta={"type": "tool", "tool": "budget_check",
                              "cluster_id": named.meta["cluster_id"],
                              "budget": budget})

    def _run_planner(self, query: str, store_path: str,
                     anchors: set[str] | None = None) -> tuple[list[dict], list[str]]:
        """Agentic stage: the rule planner decides which tools to call and
        returns their outputs as grounded documents plus a human-readable
        plan trace. Any planner failure simply skips the stage."""
        try:
            from ..agent.planner import RulePlanner, default_registry
            from ..agent.tools import Toolset

            toolset = Toolset(store_path)
            try:
                planner = RulePlanner(default_registry(toolset))
                outcome = planner.run(query, cluster_ids=anchors)
                return outcome.tool_docs, outcome.trace
            finally:
                toolset.close()
        except Exception:
            return [], []
