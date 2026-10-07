#!/usr/bin/env python3
"""Real-world DAgger collection-burden progression (Nut, Marker, Cable).

The two-message story for the three real-world tasks (one row each; the
main-text figure keeps the Nut row beside the simulation panel):

  * LEFT column (raw): fresh human-control share (%) per round. It FALLS over
    rounds for both arms as the collection policy improves and needs less
    operator help.
  * RIGHT column (collector-SR adjusted): the same share divided by the
    collection policy's failure rate ``(1 - collector_success_rate)``. Once we
    control for the fact that the Ours policy is *stronger* (fails less often),
    the Ours arm carries the RELATIVELY HIGHER burden per unit of policy
    weakness — it concentrates human time on the harder states it deliberately
    targets.

Metric:
  raw      = human_share_burden[view]["fresh_human_frame_share_pct"]
             (share of frames under human control over the arm's FRESH
             policy-first episodes only; same-start CF replays excluded from
             both numerator and denominator)
  adjusted = human_share_burden[view]["sr_adjusted_fresh_human_frame_share"]
           = raw / (1 - collector_success_rate)
             (collector_success_rate = held-out SR of that arm's collection
             policy from the neutral paired eval block recorded in the ingest)

The ingest retains only these aggregate shares (no per-episode share
distribution), so the series draw WITHOUT whiskers; the caption carries the
note. If the ingest ever persists per-episode shares, add episode-bootstrap
95% CIs here.

Data source: the pinned per-round collection summaries
``<data_dir>/{marker,square,routing}/rN/summary.json``, where ``data_dir`` is the
``real/collection`` tree of the paper evidence.
Arms: baseline = baseline_no_cf (uniform DAgger), Ours = mulligan_with_cf.
For Cable the collector-SR adjustment is degenerate (== raw) at early rounds
where full-success SR sits at floor, and becomes a real adjustment from R4 on.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from mulligan.plotting import paper

# (task_key, collection directory under data_dir, rounds)
TASKS = (
    ("square_d2", "square", (1, 2, 3, 4, 5)),
    ("marker_d2", "marker", (1, 2, 3, 4, 5)),
    ("routing_d2", "routing", (1, 2, 3, 4, 5)),
)
VIEWS = (
    ("baseline_no_cf", paper.COLLECTION_ARM_LABELS["baseline"], paper.BASELINE, "o", -0.04),
    ("mulligan_with_cf", paper.COLLECTION_ARM_LABELS["mulligan"], paper.OURS, "D", +0.04),
)


def load_task(
    task_key: str, directory: str, rounds: tuple[int, ...], *, data_dir: Path
) -> dict[str, dict[str, list[float]]]:
    """Return {view: {"raw": [...], "adj": [...], "sr": [...]}} per plotted round."""
    out: dict[str, dict[str, list[float]]] = {
        v: {"raw": [], "adj": [], "sr": []} for v, *_ in VIEWS
    }
    for r in rounds:
        summary = json.loads((data_dir / directory / f"r{r}" / "summary.json").read_text())[
            "human_share_burden"
        ]
        for view, *_ in VIEWS:
            arm = summary[view]  # KeyError = wrong pin, fail loudly
            out[view]["raw"].append(float(arm["fresh_human_frame_share_pct"]))
            out[view]["adj"].append(float(arm["sr_adjusted_fresh_human_frame_share"]))
            out[view]["sr"].append(float(arm["collector_success_rate"]))
    return out


# Second lines kept short: the right panel's title would otherwise clip at the
# figure edge at print size. The adjustment formula lives in the caption.
COL_TITLES = (
    "Raw human-collection burden\n(falls as the policy improves)",
    "Collector-SR-adjusted burden\n(divided by collector failure rate)",
)
COL_KEYS = ("raw", "adj")
COL_YLABEL = (
    "Human-control share (%)",
    "Adjusted share (%)",
)
# Shared y per column across the task rows for cross-task comparability
# (raw widened to fit Route Cable's higher fresh-human share); the compact
# single-task figure keeps the same limits so main text and appendix read on
# one scale.
COL_YLIM = ((0, 52), (0, 68))


def build_figure(
    tasks: tuple[tuple[str, str, tuple[int, ...]], ...],
    *,
    height_in: float,
    row_labels: bool,
    data_dir: Path,
) -> plt.Figure:
    """One burden figure: a (raw, SR-adjusted) column pair per task row.

    ``row_labels`` draws the rotated task name on each row's left edge — wanted
    on the multi-task figure, redundant on the single-task compact one (its
    caption names the task). Build inside :func:`mulligan.plotting.paper.paper_rc`.
    """
    # sharex=False: each row carries its own round ticks.
    fig, axes = plt.subplots(
        len(tasks),
        2,
        figsize=paper.fig_size(1.0, height_in=height_in),
        sharex=False,
        squeeze=False,
    )

    for row, (task_key, directory, rounds) in enumerate(tasks):
        data = load_task(task_key, directory, rounds, data_dir=data_dir)
        xs = list(range(len(rounds)))
        for col, ckey in enumerate(COL_KEYS):
            ax = axes[row, col]
            for view, legend, color, marker, dx in VIEWS:
                ax.plot(
                    [x + dx for x in xs],
                    data[view][ckey],
                    color=color,
                    marker=marker,
                    label=legend if (row == 0 and col == 0) else None,
                    zorder=3,
                )
            ax.set_ylim(*COL_YLIM[col])
            ax.set_xticks(xs)
            ax.set_xticklabels([f"R{r}" for r in rounds])
            paper.style_axes(ax)
            if row == 0:
                ax.set_title(COL_TITLES[col], fontsize=8.0, pad=6)
            if row == len(tasks) - 1:
                ax.set_xlabel(paper.ROUND_XLABEL)
            ax.set_ylabel(COL_YLABEL[col])
            if col == 0 and row_labels:
                # Row / task label on the far left, in the shared verb form and
                # the paper's small caps (the sink pass restyles titles and tick
                # labels only, so free-standing labels name themselves).
                paper.small_caps_text(
                    ax,
                    (-0.34, 0.5),
                    paper.TASK_TITLES[task_key],
                    size=9.0,
                    weight="bold",
                    rotation=90,
                )

    # Burden falls over rounds, so the raw panel's upper-right corner clears.
    paper.panel_legend(axes[0, 0], loc="upper right", fontsize=6.5)
    fig.tight_layout(rect=(0.03 if row_labels else 0.0, 0.0, 1, 1))
    return fig


def build_combined_figure(
    *, data_dir, sim_stats, sim_sr, sim_arms=None, width_frac=1.0, compact=False
) -> plt.Figure:
    """Main-text row of two: square_d2 collection burden over rounds (raw
    solid, failure-adjusted dashed, per arm) next to the Square-Narrow sim
    per-sampler intervention burden — the same burden story in the real
    protocol and in simulation. Build inside ``paper_rc``.
    """
    from paper.plotting.square_narrow_r1_intervention import (
        draw_panel as draw_sim_panel,
    )
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    task_key, directory, rounds = TASKS[0]
    data = load_task(task_key, directory, rounds, data_dir=data_dir)
    # Compact: the main-text float beside the 0.30-width collection-success
    # panel, which shares this height; the legend strip is sized in inches.
    height_in = 2.1 if compact else 2.6
    legend_strip_in = 0.36 if compact else 0.26
    fig, (ax_real, ax_sim) = plt.subplots(
        1,
        2,
        figsize=paper.fig_size(width_frac, height_in=height_in),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    xs = list(range(len(rounds)))
    # Metric via linestyle/fill (solid+filled = raw, dashed+open = adjusted),
    # arm via color — the line-panel analogue of the sim panel's plain-vs-hatch
    # bar encoding.
    for view, _legend, color, marker, dx in VIEWS:
        for ckey, linestyle, open_marker in (("raw", "-", False), ("adj", (0, (4, 3)), True)):
            ax_real.plot(
                [x + dx for x in xs],
                data[view][ckey],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markerfacecolor="white" if open_marker else color,
                markeredgecolor=color,
                zorder=3,
            )
    upper = max(v for view, *_ in VIEWS for k in COL_KEYS for v in data[view][k])
    ax_real.set_ylim(0, upper * 1.18)
    ax_real.set_xticks(xs)
    ax_real.set_xticklabels([f"R{r}" for r in rounds])
    ax_real.set_xlabel(paper.ROUND_XLABEL)
    ax_real.set_ylabel("Human-control share (%)")
    ax_real.set_title(f"{paper.TASK_TITLES[task_key]} (real)")
    paper.style_axes(ax_real)

    from paper.plotting.square_narrow_r1_intervention import ARM_ORDER

    draw_sim_panel(
        ax_sim, stats=sim_stats, sr=sim_sr, arms=ARM_ORDER if sim_arms is None else sim_arms
    )
    ax_sim.set_title(f"{paper.SIM_TASK_TITLES['square_narrow']} (sim)")

    if compact:
        ax_real.set_title(paper.TASK_TITLES[task_key])
        ax_sim.set_title(paper.SIM_TASK_TITLES["square_narrow"])
        ax_real.set_ylabel("Human-control share (%)")
        ax_sim.set_ylabel("Relative burden")
        ax_sim.set_xticklabels(["Uniform", "Guided\n(no CF)"])
        # At this print width, bar-value labels collide across adjacent pairs.
        # The axis, reference line, and caption already specify the ratio.
        for annotation in list(ax_sim.texts):
            annotation.remove()
        ax_sim.set_xlim(-0.6, 1.6)

    # Shared horizontal legend strip under the row (headline grammar): the two
    # collection arms, then the two metrics — each metric entry pairs its line
    # look (real panel) with its bar fill (sim panel).
    neutral = paper.INK
    arm_handles = [
        Line2D([0], [0], color=color, marker=marker, label=legend)
        for _view, legend, color, marker, _dx in VIEWS
    ]
    metric_handles = [
        (
            Line2D([0], [0], color=neutral, linestyle="-"),
            Patch(facecolor=paper.NEUTRAL_FILL, edgecolor="black"),
        ),
        (
            Line2D([0], [0], color=neutral, linestyle=(0, (4, 3))),
            Patch(facecolor=paper.NEUTRAL_FILL, edgecolor="black", hatch="///"),
        ),
    ]
    fig.legend(
        arm_handles + metric_handles,
        (
            [h.get_label() for h in arm_handles] + ["Raw", "Failure-adjusted"]
            if compact
            else [h.get_label() for h in arm_handles]
            + ["Raw", "Failure-adjusted ($\\div\\,(1-\\mathrm{SR})$)"]
        ),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2 if compact else 4,
        columnspacing=1.2,
        fontsize=6.5,
        borderaxespad=0.2 if compact else 0.5,
        frameon=False,
    )
    # The shared legend already owns the bottom strip; compact the remaining
    # outer/inter-panel padding so both plotting regions can grow.
    fig.tight_layout(rect=(0, legend_strip_in / height_in, 1, 1), pad=0.4, w_pad=0.65)
    return fig
