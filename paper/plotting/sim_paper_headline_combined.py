"""Combined real+sim headline (``headline_real_sim.pdf``).

Row 1: the three real-world task panels of the suite paper headline (marker
insertion, nut threading, cable routing) — the combined HG-DAgger-vs-Ours
two-series grammar, Wilson z=1 success intervals, per-lineage R0 rules — drawn by
:func:`paper.real_headline.draw_suite_paper_headline_panels`. Cable routing
uses mean clip score / 2 with one standard error over episodes.

Row 2: the simulation panels (:mod:`paper.plotting.sim_paper_headline`):
fixed-grid success rate for Square-Narrow and Square-Broad, plus the
Square-Narrow failure-rate (100 − SR) panel on a log axis.

Each row keeps its own legend strip (the real row shows the two deployed
lineages; the sim row shows all seven arms), but the colors mean the same arm
in both rows: gray HG-DAgger actor-only, dark-gray HG-DAgger+BoN (sim only),
teal Ours DP, purple Ours DP+BoN, warm muted hues for the autonomous arms.
Both legends explicitly key the lineage-colored static-R0 reference rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from mulligan.plotting import paper
from paper.appendix.artifacts import source_paths
from paper.appendix.real_results import prepare as real_results
from paper.appendix.real_results.prepare import paper_suite
from paper.plotting import sim_paper_headline as _SIM
from paper.real_headline import (
    PAPER_TAKEOVER_START,
    _display_path,
    draw_suite_paper_headline_panels,
    suite_paper_headline_legend,
)

SIM_PANELS = (
    ("success", "square_narrow"),
    ("success", "square_broad"),
    ("failure", "square_narrow"),
)
SIM_TITLES = (
    f"{paper.SIM_TASK_TITLES['square_narrow']} (sim)",
    f"{paper.SIM_TASK_TITLES['square_broad']} (sim)",
    f"{paper.SIM_TASK_TITLES['square_narrow']} failures",
)


def _paper_suite():
    specs = tuple(
        replace(line, task_progress=True) if line.cfg.task_key == "routing_d2" else line
        for line in paper_suite()
    )
    return specs, PAPER_TAKEOVER_START


# Print geometry in inches: the sim legend strip (three rows of arms) and
# the shared round xlabel above it keep their absolute size at any figure
# height, so the six panels absorb any height change.
FIG_HEIGHT_IN = 4.0
SIM_STRIP_IN = 0.60
XLABEL_Y_IN = 0.40


def _pack_row_horizontally(axes: Sequence[plt.Axes], *, gap: float) -> None:
    """Repack one row independently inside its existing left/right bounds.

    ``tight_layout`` treats the 2x3 array as one rigid grid: the separate
    ylabel on the bottom-right log panel then reserves the same wide column
    gutter in the real row, where it is not needed. Preserve the vertically
    solved positions, but let each row spend only the horizontal gap its own
    decorations require.
    """
    left = float(axes[0].get_position().x0)
    right = float(axes[-1].get_position().x1)
    width = (right - left - gap * (len(axes) - 1)) / len(axes)
    for index, axis in enumerate(axes):
        pos = axis.get_position()
        axis.set_position([left + index * (width + gap), pos.y0, width, pos.height])


def _distinguish_log_panel(fig: plt.Figure, axis: plt.Axes, previous: plt.Axes) -> None:
    """Mark the sole non-SR/log panel without consuming layout width."""
    # A quiet field change groups the panel as a different metric while
    # preserving the paper palette and all available data area.
    axis.set_facecolor("#F4F4F4")
    axis.text(
        0.97,
        0.96,
        "ERROR RATE · LOG SCALE",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        fontweight="bold",
        color=paper.INK,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": paper.INK,
            "linewidth": 0.6,
            "alpha": 0.92,
        },
        zorder=8,
    )
    # Put a separator in the already-budgeted inter-panel gap. It makes the
    # metric switch visible before the reader reaches the title or ticks.
    left = float(axis.get_position().x0)
    right = float(previous.get_position().x1)
    pos = axis.get_position()
    fig.add_artist(
        Line2D(
            [0.5 * (left + right), 0.5 * (left + right)],
            [pos.y0, pos.y1],
            transform=fig.transFigure,
            color=paper.INK,
            alpha=0.28,
            linewidth=0.8,
            zorder=2,
        )
    )


def build_figure() -> plt.Figure:
    """2x3 print-size render (build inside ``paper_rc``): real-task row over
    sim row, one legend strip per row."""
    specs, takeover = _paper_suite()
    if len(specs) != len(SIM_PANELS):
        raise RuntimeError(f"expected {len(SIM_PANELS)} real suite lines, got {len(specs)}")
    sim_rows = _SIM._read_rows()
    fig, axes = plt.subplots(2, 3, figsize=paper.fig_size(1.0, height_in=FIG_HEIGHT_IN))
    # Real row: one shared 0-100 axis (post-hoc sharey so the sim row keeps
    # its own scales); hide the inner tick labels like sharey=True would.
    for axis in axes[0, 1:]:
        axis.sharey(axes[0, 0])
        axis.tick_params(labelleft=False)
    real_breaks = draw_suite_paper_headline_panels(axes[0], specs, takeover_start=takeover)
    for axis, line in zip(axes[0], specs, strict=True):
        axis.set_title(f"{line.panel_title} (real)")
        if line.task_progress:
            axis.text(
                0.97,
                0.96,
                "CLIP SUCCESS RATE (%)",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.0,
                fontweight="bold",
                color=paper.INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            )
    # Sim row: the two zoomed SR panels share an axis; log-failure is free.
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 1].tick_params(labelleft=False)
    for axis, (metric, task_key), title in zip(axes[1], SIM_PANELS, SIM_TITLES, strict=True):
        _SIM.draw_headline_panel(axis, sim_rows, metric=metric, task_key=task_key)
        axis.set_title(title)
    axes[1, 0].set_ylabel("Success rate (%)")
    # Percent-formatted ticks plus the explicit title carry both the metric and
    # scale. Omitting a second vertical ylabel frees the third panel to float
    # toward its row instead of imposing its margin on the entire 2x3 grid.
    axes[1, 2].set_ylabel("")

    real_handles, real_labels = suite_paper_headline_legend(axes[0])
    sim_handles, sim_labels = _SIM.legend_entries({row["series_key"] for row in sim_rows})
    # Bottom strip: the sim legend, with one shared round xlabel above it (both
    # rows are round axes, so one figure-level label serves the whole grid).
    fig.legend(
        sim_handles,
        sim_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=6.0,
        borderaxespad=0.2,
        frameon=False,
    )
    # Keep the two legend strips, but spend less of the fixed print-size page
    # on padding around them so the six data panels get the recovered area.
    fig.tight_layout(rect=(0, SIM_STRIP_IN / FIG_HEIGHT_IN, 1, 1), pad=0.45, w_pad=0.55, h_pad=0.35)
    fig.subplots_adjust(hspace=0.58)
    # Independent row packing: the shared-y real row needs only title/tick
    # clearance; the sim row keeps a slightly wider gap for the log ticks.
    _pack_row_horizontally(axes[0], gap=0.045)
    _pack_row_horizontally(axes[1], gap=0.060)
    _distinguish_log_panel(fig, axes[1, 2], axes[1, 1])
    paper.round_xlabel(fig, y=XLABEL_Y_IN / FIG_HEIGHT_IN)
    # Real legend centered in the inter-row gap, between the real row's tick
    # labels and the sim row's titles (y from the final axes geometry).
    gap_mid = 0.5 * (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1)
    fig.legend(
        real_handles,
        real_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        loc="center",
        bbox_to_anchor=(0.5, gap_mid - 0.006),
        ncol=len(real_labels),
        columnspacing=1.6,
        frameon=False,
    )
    for axis, frac in real_breaks:
        paper.x_axis_break(axis, x_frac=frac)
    return fig


def build_records() -> paper.FigureRecord:
    with paper.paper_rc():
        return paper.save_paper_figure(
            build_figure(),
            "headline_real_sim",
            width_frac=1.0,
            sources=source_paths(real_results.HERE)
            + (
                "paper/plotting/sim_paper_headline.py",
                "paper/plotting/sim_paper_headline_combined.py",
                "paper/real_headline.py",
                _display_path(_SIM.CSV_PATH),
            ),
        )
