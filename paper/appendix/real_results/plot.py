"""Paper appendix builders; shared renderers receive only pinned snapshot inputs."""

import argparse
import pandas as pd
from paper.appendix.artifacts import source_paths
from mulligan.plotting import paper
from paper import real_headline as lh
from .prepare import HERE, CACHE, prepare

NAMES = (
    "real_world_headline_mulligan_vs_baseline_detailed",
    "real_world_marker_d2_substage",
    "real_world_square_d2_substage",
    "real_world_cf_ablation",
    "real_world_dagger_collection_success_detailed",
    "real_world_burden_progression",
    "real_world_success_speed",
    "real_world_success_throughput",
)
SHARED = (
    "paper/real_headline.py",
    "mulligan/real/lifecycle/stats.py",
    "paper/plotting/cf_ablation.py",
    "paper/plotting/dagger_collection_success.py",
    "paper/plotting/burden_progression.py",
)


def build_records(*, name):
    assert name in NAMES, name
    suite, stage = prepare()
    sources = source_paths(HERE) + SHARED
    if name == NAMES[0]:
        return lh.plot_suite_paper_headline_detailed(
            suite,
            sources=sources,
            xlabel="Round / checkpoint index",
            takeover_start=lh.PAPER_TAKEOVER_START,
        )
    if name in NAMES[1:3]:
        task = "marker_d2" if name == NAMES[1] else "square_d2"
        return lh.plot_paper_substage(stage[task], sources=sources)
    if name in NAMES[-2:]:
        metric = "success_duration" if name == NAMES[-2] else "success_throughput"
        return lh._plot_suite_efficiency_paper(
            suite, metric, sources=sources, xlabel="Round / checkpoint index"
        )
    with paper.paper_rc():
        if name == "real_world_cf_ablation":
            from paper.plotting import cf_ablation as mod

            fig = mod.build_figure(
                paths={
                    task: lh._data_path(stage[task], "headline_sr_all_arms") for task in mod.TASKS
                }
            )
        elif name == "real_world_dagger_collection_success_detailed":
            from paper.plotting import dagger_collection_success as mod

            tables = [
                (task, pd.read_csv(CACHE / "data" / task / "collection.csv"))
                for task in ("marker_d2", "square_d2", "routing_d2")
            ]
            fig = mod.build_figure(tables=tables)
        else:
            from paper.plotting import burden_progression as mod

            # Paper task order (Marker, Nut, Cable); mod.TASKS leads with the
            # main-text compact figure's Nut.
            order = ("marker_d2", "square_d2", "routing_d2")
            fig = mod.build_figure(
                tuple(sorted(mod.TASKS, key=lambda task: order.index(task[0]))),
                height_in=6.0,
                row_labels=True,
                data_dir=CACHE / "raw/real/collection",
            )
        return paper.save_paper_figure(fig, name, width_frac=1.0, sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=NAMES, default=NAMES)
    args = parser.parse_args()
    for name in args.only:
        record = build_records(name=name)
        print(record.proof_png)


if __name__ == "__main__":
    main()
