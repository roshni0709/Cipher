"""Query anchoring for retrieval: when a question clearly names a product
or a category, keep the grounded context on those product clusters.

Pure vector/TF-IDF similarity happily surfaces a mixer grinder for
"kurtis under 1000" because the price tokens match. That reads as sloppy
in a demo and is wrong in principle: a query that names a product or
category is a FILTER, not just a ranking signal.

Design, precision-first:
- Catalog vocabulary comes from the database itself: cluster-id tokens
  (a controlled vocabulary: "maroon-kurti" -> {maroon, kurti}) plus the
  offer titles of each cluster. No hand-maintained product table.
- A query token anchors every cluster whose vocabulary contains it
  (plurals folded: kurtis -> kurti).
- A few GENERIC category words expand to their family (earbuds also
  matches "buds"/"tws"; appliances matches kettle, mixer, ...), because
  "earbuds under 2000" should compare all earbuds. Specific product-type
  words (kurti, jeans, kettle) do NOT expand: "kurtis" must not return
  jeans.
- Model-ish tokens containing a digit (t110, 13c, vs104) are
  product-specific: when they match, only those clusters anchor.
- No anchor hits at all (a genuinely broad query like "best deals under
  2000") means no filtering: retrieval behaves exactly as before.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "any", "are", "at", "be", "best", "below", "buy",
    "cheap", "cheapest", "deal", "deals", "discount", "do", "does", "fake",
    "for", "from", "genuine", "get", "give", "good", "i", "in", "is", "it",
    "less", "max", "me", "my", "new", "now", "of", "offer", "offers", "on",
    "or", "price", "prices", "review", "reviews", "rs", "say", "should",
    "show", "tell", "than", "the", "to", "today", "top", "under", "vs",
    "what", "which", "with", "worth",
}

# Generic category word -> extra tokens that should also match. Only words
# shoppers use as umbrella terms get an entry; specific product types stay
# precise. Keys and values are already plural-folded (kurti, not kurtis).
_EXPANSIONS = {
    "earbud": {"bud", "tws", "earphone"},
    "earphone": {"earbud", "bud", "tws"},
    "audio": {"earbud", "bud", "tws", "headphone", "earphone", "speaker"},
    "phone": {"mobile", "smartphone"},
    "laptop": {"notebook"},
    "apparel": {"kurti", "jean", "polo", "saree", "shoe", "sneaker", "tee",
                "clothing"},
    "clothing": {"kurti", "jean", "polo", "saree", "shoe", "sneaker", "tee",
                 "apparel"},
    "shoe": {"sneaker"},
    "appliance": {"kettle", "airfryer", "fryer", "mixer", "grinder",
                  "induction", "cooktop", "stove", "flask", "thermo"},
}


def _norm(tok: str) -> str:
    """Fold simple plurals so both query and catalog sides agree."""
    tok = tok.lower()
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def catalog_tokens(offers_df) -> dict[str, set[str]]:
    """cluster_id -> vocabulary: its id tokens plus every offer-title token."""
    catalog: dict[str, set[str]] = {}
    for row in offers_df.itertuples():
        cid = row.cluster_id
        toks = catalog.setdefault(
            cid, {_norm(t) for t in _TOKEN.findall(cid.lower())})
        toks.update(_norm(t) for t in _TOKEN.findall(str(row.title).lower()))
    return catalog


def anchor_clusters(query: str, catalog: dict[str, set[str]]) -> set[str]:
    """The clusters this query is about. Empty means 'no opinion': the
    caller must leave retrieval untouched."""
    # Full catalog IDs take precedence over loose brand/model words.
    # Split letter/digit boundaries: pro5 and Pro 5 are the same model.
    def phrase(text):
        text = re.sub(r"([a-z])([0-9])|([0-9])([a-z])",
                      lambda m: " ".join(x for x in m.groups() if x),
                      text.lower().replace("&", ""))
        return " ".join(_TOKEN.findall(text))
    normalized = " " + phrase(query) + " "
    named = {cid: phrase(cid) for cid in catalog
             if " " + phrase(cid) + " " in normalized}
    if named:
        return {cid for cid, name in named.items() if not any(
            name != other and " " + name + " " in " " + other + " "
            for other in named.values())}
    q = {_norm(t) for t in _TOKEN.findall(query.lower()) if not t.isdigit()}
    q -= STOPWORDS
    if not q:
        return set()
    expanded = set(q)
    for tok in q:
        expanded |= _EXPANSIONS.get(tok, set())
    anchors = {cid for cid, toks in catalog.items() if toks & expanded}
    model_toks = {t for t in q if any(c.isdigit() for c in t)}
    if model_toks:
        specific = {cid for cid in anchors if catalog[cid] & model_toks}
        if specific:
            return specific
    return anchors
