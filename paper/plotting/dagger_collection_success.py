"""Appendix three-panel team-only collection success: all three real tasks.

Marker (left), Nut (middle), and Cable (right) collection success per round on
fresh (non-CF) episodes, one line per collection arm (Baseline vs Ours), with
Wilson z=1 intervals. The per-task tables are rebuilt from the pinned collection
ledgers by :func:`paper.real_headline.build_dagger_collection_success_table`
(:mod:`paper.appendix.real_results.prepare`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from mulligan.plotting import paper
from mulligan.real.lifecycle.stats import wilson_1se
from paper.real_headline import _round_x

# Paper styling for the two collection arms (the full-size surface keeps its own
# COLLECTION_ARM_STYLES): Ours takes the paper-wide teal, not the lineage blue.
PAPER_COLLECTION_STYLES = (
    ("baseline_uniform", paper.COLLECTION_ARM_LABELS["baseline"], paper.BASELINE, "o", -0.06),
    ("mulligan_sobol", paper.COLLECTION_ARM_LABELS["mulligan"], paper.OURS, "D", 0.06),
)


def _draw_panel(ax, task_key, df, *, show_ylabel):
    round_labels = tuple(dict.fromkeys(df["round"].astype(str)))
    x_by_round = _round_x(round_labels)
    panel_lo = []
    for arm_key, label, color, marker, dx in PAPER_COLLECTION_STYLES:
        sub = df[df["arm_key"] == arm_key]
        if len(sub) != len(round_labels):
            raise RuntimeError(f"{task_key}: arm {arm_key} has {len(sub)} rounds")
        x = [x_by_round[r] + dx for r in sub["round"]]
        y = [100.0 * float(v) for v in sub["success_rate"]]
        bounds = [wilson_1se(int(k), int(n)) for k, n in zip(sub["successes"], sub["n"])]
        lo = [100.0 * bound[0] for bound in bounds]
        hi = [100.0 * bound[1] for bound in bounds]
        panel_lo.extend(lo)
        yerr = [
            [max(0.0, yi - lo_i) for yi, lo_i in zip(y, lo, strict=True)],
            [max(0.0, hi_i - yi) for yi, hi_i in zip(y, hi, strict=True)],
        ]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            elinewidth=1.0,
            zorder=3,
            label=label,
        )
    ax.set_title(paper.TASK_TITLES[task_key])
    if show_ylabel:
        ax.set_ylabel("Success rate (%)")
    ax.set_xticks(list(x_by_round.values()))
    ax.set_xticklabels(list(round_labels))
    ax.set_xlim(-0.4, len(round_labels) - 1 + 0.4)
    paper.style_axes(ax)
    return min(panel_lo)


# Fixed zoom: team success sits near ceiling, so the panels start at 50% (the
# axis break marks the truncated origin). Whiskers below the floor fail loud.
Y_LO = 50.0


def build_figure(*, tables: list[tuple[str, pd.DataFrame]]) -> plt.Figure:
    """Three-panel collection-success figure at print size (build inside
    ``paper_rc``); ``tables`` is ``[(task_key, collection table), ...]``."""
    fig, axes = plt.subplots(
        1, len(tables), figsize=paper.fig_size(1.0, height_in=2.3), sharey=True
    )
    lowers = [
        _draw_panel(ax, task_key, df, show_ylabel=(i == 0))
        for i, (ax, (task_key, df)) in enumerate(zip(axes, tables, strict=True))
    ]
    if min(lowers) < Y_LO:
        raise RuntimeError(
            f"collection-success whisker lower bound {min(lowers):.1f} sits below the "
            f"fixed {Y_LO:.0f}% zoom — widen Y_LO"
        )
    for ax in axes:
        ax.set_ylim(Y_LO, 101.5)
    # Headline layout grammar: horizontal figure-level legend strip under the
    # row, tightened inter-panel gap (sharey), broken-axis marker on the
    # truncated y origin.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels),
        columnspacing=1.6,
        fontsize=6.5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.21, 1, 1))
    fig.subplots_adjust(wspace=0.10)
    paper.round_xlabel(fig, y=0.105)
    for ax in axes:
        paper.y_axis_break(ax)
    return fig
