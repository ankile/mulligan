"""Per-round training-data composition of the real-world collection campaigns.

One 100%-stacked bar per round and campaign, split by frame source; the number above
each bar is that round's human-controlled frames. Reads the same pinned cells as
``tab:appendix-data-composition``.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paper.appendix.artifacts import load_inputs, source_paths
from mulligan.plotting import paper
from .prepare import ARMS, HERE, ROUNDS, TASKS, composition

NAMES = ("real_world_data_composition",)
# Human sources are one green family (initial demos darkest), autonomous frames the
# neutral fill: arm colors (gray/teal/purple) stay reserved for arm identity.
SEGMENTS = (
    ("initial", "Initial demonstrations (R0)", "#196b1f"),
    ("demonstration", "Counterfactual demonstrations", "#2ca02c"),
    ("correction", "Corrections", "#98df8a"),
    ("autonomous", "Autonomous", "#d9d9d9"),
)


def kilo(frames: int) -> str:
    # Whole thousands from 10k up keep the six Route Cable labels apart.
    return f"{frames / 1000:.0f}k" if frames >= 10_000 else f"{frames / 1000:.1f}k"


def build_figure(cells: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(
        len(ARMS),
        len(TASKS),
        figsize=paper.fig_size(1.0, height_in=3.1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    labels = ("R0",) + ROUNDS
    for row, arm in enumerate(ARMS):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            for x, r in enumerate(labels):
                (c,) = [
                    c for c in cells if c["task"] == task and c["arm"] == arm and c["round"] == r
                ]
                parts = dict(c, initial=c["demonstration"] if r == "R0" else 0)
                if r == "R0":
                    parts["demonstration"] = 0
                total = sum(parts[key] for key, *_ in SEGMENTS)
                bottom = 0.0
                for key, _, color in SEGMENTS:
                    share = 100 * parts[key] / total
                    ax.bar(x, share, bottom=bottom, width=0.72, color=color, linewidth=0)
                    bottom += share
                human = total - parts["autonomous"]
                ax.text(
                    x, 102, kilo(human), ha="center", va="bottom", fontsize=5.5, color=paper.INK
                )
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 112)
            ax.set_yticks((0, 25, 50, 75, 100))
            ax.spines["left"].set_bounds(0, 100)
            paper.style_axes(ax)
            if row == 0:
                ax.set_title(paper.TASK_TITLES[task])
            if row == len(ARMS) - 1:
                ax.set_xlabel(paper.ROUND_XLABEL)
            if col == 0:
                ax.set_ylabel(f"{paper.COLLECTION_ARM_LABELS[arm]}\nframes (%)")
    fig.legend(
        handles=[Patch(color=color, label=label) for _, label, color in SEGMENTS],
        loc="lower center",
        ncol=len(SEGMENTS),
        frameon=False,
        fontsize=6.5,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8)
    return fig


def build_records(*, name):
    assert name in NAMES, name
    cells = composition(load_inputs(HERE))
    with paper.paper_rc():
        fig = build_figure(cells)
        return paper.save_paper_figure(fig, name, width_frac=1.0, sources=source_paths(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(build_records(name=NAMES[0]).proof_png)


if __name__ == "__main__":
    main()
