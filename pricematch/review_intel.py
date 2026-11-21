"""Review intelligence: extractive summarization and manipulation detection.

summarize() picks the most representative sentences from positive and
negative reviews using a TF-IDF centroid (deterministic, no model weights).
trust() scores a product's review set for manipulation patterns: bursts of
5-star activity, copy-paste duplicate text, and one-sided ratings.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data.reviews import reviews_for

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if len(s.strip()) >= 20]


def _norm_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _centroid_rank(sentences: list[str], top: int) -> list[str]:
    """Rank sentences by cosine similarity to the TF-IDF centroid of the set,
    skipping near-duplicates of already-picked sentences."""
    if len(sentences) <= top:
        return sentences
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(sentences)
    centroid = np.asarray(mat.mean(axis=0))
    sims = np.asarray(mat @ centroid.T).ravel()
    order = np.argsort(-sims, kind="stable")  # stable: deterministic ties
    picked: list[str] = []
    picked_toks: list[set] = []
    for i in order:
        toks = set(_norm_tokens(sentences[i]))
        if toks and all(len(toks & p) / len(toks | p) < 0.8 for p in picked_toks):
            picked.append(sentences[i])
            picked_toks.append(toks)
        if len(picked) == top:
            break
    return picked


def summarize(cluster_id: str, top: int = 3) -> dict:
    """Extractive pros/cons summary. Deterministic for a given review set."""
    reviews = reviews_for(cluster_id)
    pos = [s for r in reviews if r["rating"] >= 4 for s in _sentences(r["text"])]
    neg = [s for r in reviews if r["rating"] <= 3 for s in _sentences(r["text"])]
    return {
        "cluster_id": cluster_id,
        "review_count": len(reviews),
        "avg_rating": round(sum(r["rating"] for r in reviews) / len(reviews), 2) if reviews else None,
        "pros": _centroid_rank(pos, top),
        "cons": _centroid_rank(neg, top),
    }


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def trust(cluster_id: str) -> dict:
    """Review-trust analysis. Returns a 0-100 score plus human-readable flags."""
    reviews = reviews_for(cluster_id)
    if not reviews:
        return {"cluster_id": cluster_id, "score": None, "label": "no reviews",
                "flags": [], "review_count": 0}

    flags: list[dict] = []
    n = len(reviews)
    ratings = [r["rating"] for r in reviews]

    # 1) burst detection: most reviews inside any 7-day window, all glowing
    days = sorted(r["days_ago"] for r in reviews)
    best = 0
    for i, d in enumerate(days):
        j = i
        while j < n and days[j] - d <= 7:
            j += 1
        best = max(best, j - i)
    if n >= 6 and best / n >= 0.5:
        burst_reviews = [r for r in reviews if r["rating"] >= 4]
        if len(burst_reviews) / n >= 0.5 and sum(ratings) / n >= 4.5:
            flags.append({"flag": "review_burst",
                          "detail": f"{best} of {n} reviews posted within a 7-day window, nearly all 5-star"})

    # 2) copy-paste detection: near-duplicate review text
    toks = [set(_norm_tokens(r["text"])) for r in reviews]
    dup_pairs = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if _jaccard(toks[i], toks[j]) >= 0.8:
                dup_pairs += 1
    if pairs and dup_pairs / pairs >= 0.15:
        flags.append({"flag": "copy_paste_text",
                      "detail": f"{dup_pairs} pairs of reviews share near-identical text across different authors"})

    # 3) one-sided ratings
    five = sum(1 for r in ratings if r == 5)
    if n >= 6 and five / n >= 0.85:
        flags.append({"flag": "one_sided_ratings",
                      "detail": f"{five} of {n} reviews are 5-star with no critical balance"})

    score = max(5, 100 - 35 * len(flags))
    label = "high" if score >= 80 else ("medium" if score >= 50 else "low")
    return {"cluster_id": cluster_id, "score": score, "label": label,
            "flags": flags, "review_count": n,
            "avg_rating": round(sum(ratings) / n, 2)}
