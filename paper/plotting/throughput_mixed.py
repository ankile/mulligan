"""Main-text throughput for all three real tasks, from pinned appendix evidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from mulligan.plotting import paper
from paper import real_headline as real
from paper.real_headline import PAPER_TAKEOVER_START

NAME = "real_world_success_throughput_headline"

# Print geometry in inches: the one-row legend strip keeps its absolute size,
# so the panels absorb any height change. No round xlabel: the R0, R1, ...
# ticks name the rounds on their own.
FIG_HEIGHT_IN = 1.5
BOTTOM_STRIP_IN = 0.2


def _draw_real_panel(
    ax: plt.Axes, line: real.SuiteLineHeadlineSpec, panel_idx: int
) -> list[tuple[plt.Axes, float]]:
    tick_labels = tuple(label for label, _ in line.panel_groups())
    x_by_round = real._round_x(tick_labels)
    series = real._suite_efficiency_lineage_series(
        line,
        "success_throughput",
        panel_idx,
        takeover_start=PAPER_TAKEOVER_START.get(line.cfg.task_key),
    )
    panel_upper = real._draw_efficiency_lineage(ax, x_by_round, series)
    ax.set_title(f"{line.panel_title} (real)")
    ax.set_xticks(list(x_by_round.values()))
    ax.set_xticklabels(list(tick_labels))
    xlim = (-0.35, len(tick_labels) - 1 + 0.35)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, max(0.1, panel_upper * 1.18))
    paper.style_axes(ax)
    return [(ax, frac) for frac in real._round_axis_break_fracs(tick_labels, x_by_round, xlim)]


def build_figure(specs) -> plt.Figure:
    with paper.paper_rc():
        fig, axes = plt.subplots(1, 3, figsize=paper.fig_size(1.0, height_in=FIG_HEIGHT_IN))
        axis_breaks: list[tuple[plt.Axes, float]] = []
        for panel_idx, (axis, line) in enumerate(zip(axes, specs, strict=True)):
            axis_breaks.extend(_draw_real_panel(axis, line, panel_idx))
        axes[0].set_ylabel("Task units per minute")

        handles_by_label: dict[str, object] = {}
        for axis in axes:
            for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
                handles_by_label.setdefault(label, handle)
        baseline_label = paper.POLICY_ARM_LABELS["baseline"]
        dp_label = paper.POLICY_ARM_LABELS["mulligan_with_cf"]
        rerank_label = paper.POLICY_ARM_LABELS["final_iql"]
        fig.legend(
            [
                handles_by_label[baseline_label],
                (handles_by_label[dp_label], handles_by_label[rerank_label]),
            ],
            [baseline_label, rerank_label],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            columnspacing=1.6,
            fontsize=6.5,
            borderaxespad=0.2,
            frameon=False,
        )
        fig.tight_layout(rect=(0, BOTTOM_STRIP_IN / FIG_HEIGHT_IN, 1, 1), pad=0.4, w_pad=0.35)
        for axis, frac in axis_breaks:
            paper.x_axis_break(axis, x_frac=frac)
        return fig
