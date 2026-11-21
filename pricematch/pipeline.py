"""One entry point for the whole project.

  python -m pricematch.pipeline seed      build the sample DB (offers + 60d history)
  python -m pricematch.pipeline eval      run the eval harness (see eval/run.py)
  python -m pricematch.pipeline agent     chat with the agent layer
  python -m pricematch.pipeline snapshot  collect prices from a URL list
  python -m pricematch.pipeline demo      launch the Streamlit demo
"""
from __future__ import annotations

import argparse
import sys


def ensure_db(db_path: str) -> bool:
    """Seed the sample database when none exists (fresh checkout or a new
    Streamlit Cloud deploy). Also reseeds when an existing DB belongs to an
    older version of the catalog (stale file kept as *.stale.db), so a demo
    never silently runs on outdated data. Returns True when it seeded."""
    from pathlib import Path

    path = Path(db_path)
    if path.exists():
        from .data import sample
        from .db.store import Store

        try:
            store = Store(db_path)
            n_clusters = int(store.offers()["cluster_id"].nunique())
            store.close()
        except Exception:
            n_clusters = -1  # unreadable/corrupt: reseed below
        if n_clusters == len(sample.CLUSTERS):
            return False
        # archive the old DB; timestamped name so a second stale event can
        # never crash on Windows (os.rename refuses to overwrite there)
        import time as _time
        stale = path.with_name(
            f"{path.stem}.stale-{_time.strftime('%Y%m%d-%H%M%S')}.db")
        path.rename(stale)
    seed(db_path)
    return True


def seed(db_path: str) -> None:
    from .data import sample
    from .db.store import Store

    store = Store(db_path)
    for offer in sample.offers():
        store.upsert_offer(offer)
        history = sample.make_history(offer)
        for row in history.itertuples():
            store.add_snapshot(offer["offer_id"], row.ts.date().isoformat(), row.price,
                               float(offer.get("shipping", 0)))
    store.close()
    n_clusters = len(sample.CLUSTERS)
    n_offers = len(sample.offers())
    print(f"Seeded {db_path}: {n_clusters} product clusters, {n_offers} offers, 60 days of prices each")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seed", "eval", "agent", "snapshot", "demo"])
    parser.add_argument("--db", default="pricematch.db")
    args, rest = parser.parse_known_args(argv)

    if args.command == "seed":
        seed(args.db)
    elif args.command == "eval":
        from .eval.run import main as eval_main

        eval_main(rest)
    elif args.command == "agent":
        from .agent.core import run_agent
        from .agent.tools import Toolset

        toolset = Toolset(args.db)
        query = " ".join(rest) or input("What are you looking for? ")
        print(run_agent(query, toolset))
        toolset.close()
    elif args.command == "snapshot":
        from .collector.snapshot import main as snap_main

        snap_main(rest + ["--db", args.db])
    elif args.command == "demo":
        import subprocess
        from pathlib import Path

        app = Path(__file__).parent / "demo" / "app.py"
        sys.exit(subprocess.call(["streamlit", "run", str(app), "--", "--db", args.db]))


if __name__ == "__main__":
    main()
