"""All-real main-text collection-success and throughput figures."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from paper.appendix.real_results.prepare import prepare as prepare_real
from mulligan.plotting import paper
from paper import real_headline as lh
from paper.plotting import collection_success_mixed as collection
from paper.plotting import throughput_mixed as throughput
from .prepare import CACHE, NAMES, prepare_collection, sources


# Shared with the side-by-side collection-burden figure (same float, same height).
SQUARE_HEIGHT_IN = 2.1
SQUARE_LEGEND_IN = 0.36


def collection_square(panels):
    panel = next(panel for panel in panels if panel.task_key == "square_d2")
    fig, ax = plt.subplots(figsize=paper.fig_size(0.30, height_in=SQUARE_HEIGHT_IN))
    collection._draw_real_panel(ax, panel, show_ylabel=True)
    ax.set_title(paper.TASK_TITLES["square_d2"])
    ax.set_xlabel(paper.ROUND_XLABEL)
    handles = [
        Line2D([0], [0], color=paper.OURS, marker="D", linestyle="-", label="Human-robot team"),
        Line2D(
            [0],
            [0],
            color=paper.OURS,
            marker="D",
            linestyle="--",
            markerfacecolor="white",
            label="Policy alone",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=1,
        frameon=False,
        fontsize=6.5,
        handlelength=1.7,
        borderaxespad=0.2,
    )
    fig.tight_layout(rect=(0, SQUARE_LEGEND_IN / SQUARE_HEIGHT_IN, 1, 1), pad=0.4)
    return fig


def build_records(*, name):
    assert name in NAMES
    width = 0.30 if name == "real_world_dagger_collection_success_square" else 1.0
    with paper.paper_rc():
        if name == "real_world_dagger_collection_success_square":
            fig = collection_square(prepare_collection())
        elif name == collection.NAME:
            fig = collection.build_figure(panels=prepare_collection())
        else:
            suite, _ = prepare_real()
            assert tuple(line.cfg.task_key for line in suite) == (
                "marker_d2",
                "square_d2",
                "routing_d2",
            )
            assert all(line.cfg.headline_metric == "full_success" for line in suite)
            records = []
            for idx, line in enumerate(suite):
                series = lh._suite_efficiency_lineage_series(
                    line,
                    "success_throughput",
                    idx,
                    takeover_start=lh.PAPER_TAKEOVER_START.get(line.cfg.task_key),
                )
                for arm, values in series.items():
                    for point in zip(
                        values["ticks"],
                        values["ys"],
                        values["err_lo"],
                        values["err_hi"],
                        values["ks"],
                        values["source_arms"],
                        strict=True,
                    ):
                        records.append(
                            dict(
                                task=line.cfg.task_key,
                                series=arm,
                                round=point[0],
                                throughput_per_min=point[1],
                                error_low=point[2],
                                error_high=point[3],
                                completed_tasks=point[4],
                                source_arm=point[5],
                            )
                        )
            (CACHE / "data").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(CACHE / "data/throughput_points.csv", index=False)
            fig = throughput.build_figure(suite)
        return paper.save_paper_figure(fig, name, width_frac=width, sources=sources())
