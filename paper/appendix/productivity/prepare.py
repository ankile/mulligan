"""Pinned collection tables; the throughput panels reuse the appendix's frozen protocol."""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from paper.appendix.artifacts import REAL_DIRS, cache_dir, load_inputs, source_paths
from paper.appendix.real_results import prepare as real
from paper.plotting import collection_success_mixed as collection

HERE = Path(__file__).resolve().parent
CACHE = cache_dir("productivity")
NAMES = (
    "real_world_dagger_collection_success",
    "real_world_dagger_collection_success_square",
    "real_world_success_throughput_headline",
)
SHARED = (
    "paper/plotting/collection_success_mixed.py",
    "paper/plotting/throughput_mixed.py",
    "paper/real_headline.py",
    "mulligan/real/lifecycle/stats.py",
)


def prepare_collection():
    raw = load_inputs(HERE)
    panels, records = [], []
    for panel in collection.REAL_PANELS:
        task = panel.task_key
        pinned = replace(
            panel,
            collection_csv=str(raw[f"real/collection/{REAL_DIRS[task]}/collection.csv"]),
            eval_csv=str(raw[f"real/collection/{REAL_DIRS[task]}/collector_eval.csv"]),
        )
        team = collection._ours_team(pd.read_csv(pinned.collection_csv), rel=pinned.collection_csv)
        lagged = collection._lagged_eval(pinned, tuple(team["round"].astype(str)))
        for label, frame in (("team", team), ("collector_held_out", lagged)):
            assert not frame["round"].duplicated().any()
            assert (
                (frame.n > 0).all()
                and (frame.successes >= 0).all()
                and (frame.successes <= frame.n).all()
            )
            for row in frame.to_dict("records"):
                records.append(dict(task=task, series=label, **row))
        panels.append(pinned)
    (CACHE / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(CACHE / "data/collection_points.csv", index=False)
    return tuple(panels)


def sources():
    return tuple(sorted(set(source_paths(HERE) + source_paths(real.HERE) + SHARED)))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare_collection()
    print(CACHE / "data/collection_points.csv")


if __name__ == "__main__":
    main()
