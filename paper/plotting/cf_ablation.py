"""Focused counterfactual-demonstrations ablation panels.

A deliberately spare panel grammar showing held-out full-success rate over
DAgger rounds for ONLY the two Ours arms that isolate the
counterfactual-demonstration treatment:

  * Ours (no CF demos)   -- DAgger targets hard states but collects NO
                            counterfactual (recovery) human demonstrations.
  * Ours (with CF demos) -- same start design, plus counterfactual demos at the
                            targeted hard states (the headline treatment).

:func:`draw_cf_panel` fills one caller-owned axis per task: the main-text
``ablations_sim_real`` carries compact Nut and Marker panels, and
:func:`build_figure` renders both at full size as the self-contained appendix
figure (``real_world_cf_ablation``).

The two arms share the R0 pre-DAgger Sobol source actor, so the no-CF/with-CF
split only begins at R1 (mirrors the *_headline_sr_all_arms footnote). The
no-CF arm was retired from the robot after the round where it stops (marker R2,
square R4; later rounds moved it off-robot), so each panel is capped at the
last round the no-CF arm was evaluated on-robot -- the paired region where the
head-to-head is real.

Data source: the pinned per-line headline all-arms tables. The renderer
recomputes Wilson ``z=1`` bounds from each pooled row's counts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mulligan.plotting import paper
from mulligan.real.lifecycle.stats import wilson_1se
from paper.real_headline import ARM_STYLES

ROUND_ORDER = ("R0", "R1", "R2", "R3", "R4", "R5")
ROUND_X = {label: idx for idx, label in enumerate(ROUND_ORDER)}

# The two ablation arms, drawn with-CF (treatment) first so it leads the legend.
ARMS = ("mulligan_with_cf", "mulligan_no_cf")
# Same small symmetric x-nudge as the paper headline so near-coincident points
# (e.g. marker R2: 48 vs 47) keep their Wilson whiskers legible.
ARM_DX = {"mulligan_with_cf": 0.04, "mulligan_no_cf": -0.04}


def _load_arm_table(path: Path) -> pd.DataFrame:
    """Pooled (round, arm) headline SR rows for the two ablation arms.

    ``is_pooled`` selects the one headline point per (round, arm); the per-block
    rows in the CSV are provenance-only and must not double-plot a pooled arm.
    """
    df = pd.read_csv(path)
    df = df[df["is_pooled"].astype(bool)]
    df = df[df["arm"].isin(ARMS)].copy()
    if df.empty:
        raise RuntimeError(f"{path}: no pooled rows for arms {ARMS}")
    # One pooled row per (round, arm) -- a second would silently double-plot.
    dupes = df.duplicated(subset=["round", "arm"], keep=False)
    if dupes.any():
        raise RuntimeError(f"{path}: duplicate pooled (round, arm) rows:\n{df[dupes]}")
    df["round_idx"] = df["round"].map(ROUND_X)
    if df["round_idx"].isna().any():
        bad = sorted(df.loc[df["round_idx"].isna(), "round"].unique())
        raise RuntimeError(f"{path}: round labels off the configured axis: {bad}")
    return df


def draw_cf_panel(ax: plt.Axes, task_key: str, *, path: Path) -> None:
    """Fill one CF-ablation panel (title, 0-100 y axis, round xticks); the
    caller owns ylabel, xlabel, and legend placement (arm labels are attached
    to the errorbar containers)."""
    df = _load_arm_table(path)

    # Cap at the last round the no-CF arm was actually evaluated on-robot: past
    # that there is nothing to compare against, so those with-CF points belong
    # to the headline figure, not this focused ablation.
    no_cf_rounds = df.loc[df["arm"] == "mulligan_no_cf", "round_idx"]
    if no_cf_rounds.empty:
        raise RuntimeError(f"{task_key}: no mulligan_no_cf rows to anchor the ablation cap")
    cap_idx = int(no_cf_rounds.max())
    df = df[df["round_idx"] <= cap_idx].copy()

    round_labels = tuple(ROUND_ORDER[i] for i in range(cap_idx + 1))

    for arm in ARMS:
        style = ARM_STYLES[arm]
        sub = df[df["arm"] == arm].sort_values("round_idx")
        if sub.empty:
            continue
        x = [ROUND_X[r] + ARM_DX[arm] for r in sub["round"]]
        y = [100.0 * v for v in sub["success_rate"]]
        bounds = [wilson_1se(int(k), int(n)) for k, n in zip(sub["successes"], sub["n"])]
        yerr = [
            [100.0 * (v - lo) for v, (lo, _) in zip(sub["success_rate"], bounds)],
            [100.0 * (hi - v) for v, (_, hi) in zip(sub["success_rate"], bounds)],
        ]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            elinewidth=1.0,
            label=paper.CF_ARM_LABELS[arm],
            zorder=3,
        )

    ax.set_title(paper.TASK_TITLES[task_key])
    ax.set_xticks([ROUND_X[r] for r in round_labels])
    ax.set_xticklabels(round_labels)
    ax.set_xlim(-0.4, cap_idx + 0.4)
    ax.set_ylim(0, 100)
    paper.style_axes(ax)


# Appendix panel order follows the paper's task order (Marker, Nut).
TASKS = ("marker_d2", "square_d2")


def build_figure(*, paths: dict[str, Path]) -> plt.Figure:
    """Two-panel appendix CF-ablation figure (Marker, Nut) at print size (build
    inside ``paper_rc``). Panel widths follow each task's round count so one
    round spans the same width in both panels."""
    caps = [
        int(_load_arm_table(paths[task]).query("arm == 'mulligan_no_cf'")["round_idx"].max())
        for task in TASKS
    ]
    fig, axes = plt.subplots(
        1,
        len(TASKS),
        figsize=paper.fig_size(1.0, height_in=2.2),
        sharey=True,
        gridspec_kw={"width_ratios": [cap + 0.8 for cap in caps]},
    )
    for ax, task in zip(axes, TASKS):
        draw_cf_panel(ax, task, path=paths[task])
        ax.set_xlabel(paper.ROUND_XLABEL)
    axes[0].set_ylabel("Success rate (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(
        handles, labels, loc="upper left", frameon=False, handlelength=1.6, fontsize=6.5
    )
    fig.tight_layout()
    return fig
