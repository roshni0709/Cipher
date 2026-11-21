"""CLI: evaluate matchers on the sample catalog or WDC benchmark.

Examples:
  python -m pricematch.eval.run --dataset sample --compare
  python -m pricematch.eval.run --dataset wdc --category computers --size small --compare
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import metrics


def entity_split(df: pd.DataFrame, test_fraction: float = 0.3, seed: int = 13):
    """Split so no product cluster appears in both train and test.

    A random pair split leaks: the same product's listings land on both
    sides and the model memorizes products instead of learning matching.
    Positive pairs always stay together (both sides share a cluster);
    negative pairs whose two clusters fall in different splits are
    dropped and counted.
    """
    rng = np.random.RandomState(seed)
    clusters = np.unique(df[["left_cluster", "right_cluster"]].to_numpy().ravel())
    rng.shuffle(clusters)
    n_test = max(1, int(round(len(clusters) * test_fraction)))
    test_clusters = set(clusters[:n_test])
    in_test_l = df["left_cluster"].isin(test_clusters)
    in_test_r = df["right_cluster"].isin(test_clusters)
    test = df[in_test_l & in_test_r]
    train = df[~in_test_l & ~in_test_r]
    dropped = int((in_test_l != in_test_r).sum())
    return train, test, dropped


def load_dataset(args) -> pd.DataFrame:
    if args.dataset == "sample":
        from ..data import sample

        return sample.pairs()
    if args.dataset == "image-eval":
        from ..data import image_eval

        return image_eval.load_pairs()
    from ..data import wdc

    dest = wdc.download(args.category, "train")
    path = wdc.find_pair_file(dest, args.category, "train", args.size)
    return wdc.load_pairs(path, max_rows=args.max_rows)


def build_matcher(name: str):
    if name == "tfidf":
        from ..matching.tfidf import TfidfMatcher

        return TfidfMatcher()
    if name == "st":
        from ..matching.embeddings import TextEmbeddingMatcher

        return TextEmbeddingMatcher()
    if name == "multimodal":
        from ..matching.multimodal import MultimodalMatcher

        return MultimodalMatcher()
    raise ValueError(name)


def evaluate_one(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    matcher = build_matcher(name)
    if name == "multimodal":
        cov = metrics.image_coverage(test)
        if cov == 0.0:
            return {"matcher": name,
                    "error": "test split has no image-covered pairs; a multimodal "
                             "number without images is a text-only result. Use "
                             "--dataset image-eval or a pairs file with image paths."}
    if hasattr(matcher, "fit"):
        try:
            matcher.fit(train)
        except TypeError:
            matcher.fit(train, train["label"].to_numpy())
    train_scores = matcher.score_pairs(train)
    threshold, _ = metrics.best_f1(train["label"].to_numpy(), train_scores)
    test_scores = matcher.score_pairs(test)
    report = metrics.evaluate(test, test["label"].to_numpy(), test_scores, threshold)
    report["matcher"] = name
    report["image_coverage"] = round(metrics.image_coverage(test), 4)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sample", "wdc", "image-eval"], default="sample")
    parser.add_argument("--category", default="computers")
    parser.add_argument("--size", default="small")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--matcher", choices=["tfidf", "st", "multimodal"], default="tfidf")
    parser.add_argument("--compare", action="store_true",
                        help="Run the TF-IDF baseline AND the embedding matcher, side by side")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    df = load_dataset(args)
    train, test, dropped = entity_split(df)
    if int(test["label"].sum()) == 0 or len(test) < 4:
        # Tiny sets (sample, image-eval) leave too little for an entity-disjoint
        # split. Stratified pair split instead, LOUDLY labeled: same products can
        # appear in both sides, so scores are smoke-test indicators, not publishable.
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(df, test_size=0.4, random_state=13,
                                       stratify=df["label"])
        print(f"WARNING: set too small for an entity-disjoint split; using a "
              f"stratified pair split ({len(train)} train / {len(test)} test). "
              f"The same product can appear on both sides, so these scores are "
              f"smoke-test indicators only, not publishable numbers.")
    else:
        print(f"Split: entity-disjoint (no product cluster on both sides). "
              f"{len(train)} train / {len(test)} test pairs, "
              f"{dropped} cross-split negatives dropped. "
              f"Test positives: {int(test['label'].sum())}.")
    names = ["tfidf", "st"] if args.compare else [args.matcher]
    reports = []
    for name in names:
        try:
            reports.append(evaluate_one(name, train, test))
        except ImportError as exc:
            reports.append({"matcher": name, "error": str(exc)})
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            print(_pretty(r))
        if args.compare and all("f1" in r for r in reports):
            gain = reports[1]["f1"] - reports[0]["f1"]
            print(f"\nEmbedding matcher minus TF-IDF baseline: {gain:+.4f} F1 on this split. "
                  "Report both numbers with the split and dataset if you use them anywhere.")


def _pretty(r: dict) -> str:
    if "error" in r:
        return f"[{r['matcher']}] skipped: {r['error']}"
    cov = f"  image-coverage={r['image_coverage']}" if r.get("image_coverage") is not None else ""
    return (f"[{r['matcher']}] threshold={r['threshold']}  P={r['precision']}  "
            f"R={r['recall']}  F1={r['f1']}  top-1={r['top1_accuracy']}{cov}  "
            f"({r['n_pairs']} pairs, {r['n_positive']} positive)")


if __name__ == "__main__":
    main()
