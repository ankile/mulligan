"""Collection success for Marker, Nut, and Cable (appendix and main-text Nut panel).

Human-robot completion of fresh episodes versus the actual collector policy's
previous-round held-out evaluation. :func:`paper.appendix.productivity.prepare.
prepare_collection` binds each panel to its pinned collection and eval tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from mulligan.plotting import paper
from mulligan.real.lifecycle.stats import wilson_1se
from paper.real_headline import _round_x

NAME = "real_world_dagger_collection_success"

TEAM_LABEL = "Human-robot team"
EVAL_LABEL = "Policy alone (held-out)"

TEAM_DX = -0.06
UNASSISTED_DX = 0.06


@dataclass(frozen=True)
class RealPanel:
    """One real-world panel: Ours team collection vs lagged collector eval."""

    task_key: str
    # Collection round -> headline arm of the frozen policy that collected it
    # (the previous round's eval). Round k collection uses the round k-1 policy.
    collectors: dict[str, str]
    # Pinned tables (``prepare_collection`` fills them): the collection-success
    # table and the collector eval table (pooled rows per round and arm).
    collection_csv: str = ""
    eval_csv: str = ""


# Collector arms are the policies that actually ran during Ours collection,
# not the later headline takeover (PAPER_TAKEOVER_START is an eval display
# rule). Marker R4/R5 collected with DP+IQL composites; earlier rounds
# collected with plain DP. Cable is on the paper's routing_d2 100-episode
# increments: R1-R4 each pool two 50-episode velocity-action sessions, R5 is the
# 100-episode session collected by the routing_d2 R4 checkpoint, and the
# collector series pools the source sessions' contemporaneous evaluations.
REAL_PANELS: tuple[RealPanel, ...] = (
    RealPanel(
        task_key="marker_d2",
        collectors={
            "R1": "mulligan_with_cf",  # R0 DP
            "R2": "mulligan_with_cf",  # R1 DP
            "R3": "mulligan_with_cf",  # R2 DP (pooled n=100)
            "R4": "final_marker_iql",  # R3 DP+IQL n=16
            "R5": "final_marker_iql",  # R4 DP+IQL composite
        },
    ),
    RealPanel(
        task_key="square_d2",
        collectors={
            "R1": "mulligan_with_cf",  # R0 DP
            "R2": "mulligan_with_cf",  # R1 DP
            "R3": "mulligan_with_cf",  # R2 DP
            "R4": "final_iql",  # R3 DP+IQL S1, n=16
            "R5": "final_iql",  # R4 DP+IQL S2, n=16
        },
    ),
    RealPanel(
        task_key="routing_d2",
        # Collector eval is episode full-success, matching collection's
        # episode-completion metric (the cable headline CSV is clip-progress).
        collectors={
            "R1": "mulligan_with_cf",  # source R0+R1 DP
            "R2": "mulligan_with_cf",  # source R2+R3 DP
            "R3": "mulligan_with_cf",  # source R4+R5 DP
            "R4": "mulligan_with_cf",  # source R6+R7 DP
            "R5": "mulligan_with_cf",  # routing_d2 R4 DP (UMI-relative)
        },
    ),
)


def _read_csv(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("panel table not bound; use prepare_collection()")
    return pd.read_csv(path)


def _ours_team(df: pd.DataFrame, *, rel: str) -> pd.DataFrame:
    sub = df[df["arm_key"] == "mulligan_sobol"].copy()
    if sub.empty:
        raise RuntimeError(f"{rel}: no mulligan_sobol collection rows")
    return sub.sort_values("round")


def _pooled(df: pd.DataFrame, *, rel: str) -> pd.DataFrame:
    if "is_pooled" not in df.columns:
        raise RuntimeError(f"{rel}: missing is_pooled")
    if df["is_pooled"].dtype != bool:
        raise RuntimeError(f"{rel}: is_pooled dtype {df['is_pooled'].dtype}, expected bool")
    pooled = df[df["is_pooled"]]
    if pooled.empty:
        raise RuntimeError(f"{rel}: no pooled eval rows")
    return pooled


def _lagged_eval(panel: RealPanel, team_rounds: tuple[str, ...]) -> pd.DataFrame:
    if team_rounds != tuple(panel.collectors):
        raise RuntimeError(
            f"{panel.task_key}: collection rounds {team_rounds} != "
            f"collector map {tuple(panel.collectors)}"
        )
    eval_df = _pooled(_read_csv(panel.eval_csv), rel=panel.eval_csv)
    rows: list[dict[str, object]] = []
    for coll_round, arm in panel.collectors.items():
        k = int(str(coll_round).removeprefix("R"))
        eval_round = f"R{k - 1}"
        hit = eval_df[(eval_df["round"].astype(str) == eval_round) & (eval_df["arm"] == arm)]
        if len(hit) != 1:
            raise RuntimeError(
                f"{panel.eval_csv}: expected one pooled {eval_round}/{arm} row, got {len(hit)}"
            )
        row = hit.iloc[0]
        rows.append(
            {
                "round": coll_round,
                "eval_round": eval_round,
                "arm": arm,
                "successes": int(row["successes"]),
                "n": int(row["n"]),
            }
        )
    return pd.DataFrame(rows)


def _errorbar(
    ax: plt.Axes,
    xs: list[float],
    successes: list[int],
    ns: list[int],
    *,
    color: str,
    marker: str,
    linestyle: str,
    open_marker: bool,
    markersize: float,
    zorder: float,
) -> None:
    y = [100.0 * k / n for k, n in zip(successes, ns, strict=True)]
    bounds = [wilson_1se(k, n) for k, n in zip(successes, ns, strict=True)]
    lo = [100.0 * bound[0] for bound in bounds]
    hi = [100.0 * bound[1] for bound in bounds]
    yerr = [
        [max(0.0, yi - lo_i) for yi, lo_i in zip(y, lo, strict=True)],
        [max(0.0, hi_i - yi) for yi, hi_i in zip(y, hi, strict=True)],
    ]
    ax.errorbar(
        xs,
        y,
        yerr=yerr,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markerfacecolor="white" if open_marker else color,
        markeredgecolor=color,
        markeredgewidth=1.0,
        elinewidth=1.0,
        zorder=zorder,
    )


def _draw_real_panel(ax: plt.Axes, panel: RealPanel, *, show_ylabel: bool) -> None:
    team = _ours_team(_read_csv(panel.collection_csv), rel=panel.collection_csv)
    team_rounds = tuple(team["round"].astype(str))
    lagged = _lagged_eval(panel, team_rounds)
    x_by_round = _round_x(team_rounds)
    team_successes = [int(v) for v in team["successes"]]
    team_ns = [int(v) for v in team["n"]]
    lagged_successes = [int(v) for v in lagged["successes"]]
    lagged_ns = [int(v) for v in lagged["n"]]
    for k, n, lk, ln, rnd in zip(
        team_successes, team_ns, lagged_successes, lagged_ns, team_rounds, strict=True
    ):
        if k / n <= lk / ln:
            raise RuntimeError(
                f"{panel.task_key} {rnd}: team {k}/{n} is not above lagged eval {lk}/{ln}"
            )
    _errorbar(
        ax,
        [x_by_round[r] + TEAM_DX for r in team_rounds],
        team_successes,
        team_ns,
        color=paper.OURS,
        marker="D",
        linestyle="-",
        open_marker=False,
        markersize=4.5,
        zorder=4,
    )
    _errorbar(
        ax,
        [x_by_round[r] + UNASSISTED_DX for r in team_rounds],
        lagged_successes,
        lagged_ns,
        color=paper.OURS,
        marker="D",
        linestyle="--",
        open_marker=True,
        markersize=4.5,
        zorder=3,
    )
    ax.set_title(f"{paper.TASK_TITLES[panel.task_key]} (real)")
    if show_ylabel:
        ax.set_ylabel("Success rate (%)")
    ax.set_xticks(list(x_by_round.values()))
    ax.set_xticklabels(list(team_rounds))
    ax.set_xlim(-0.4, len(team_rounds) - 1 + 0.4)
    ax.set_ylim(0, 103)
    ax.set_yticks(range(0, 101, 20))
    paper.style_axes(ax)


def build_figure(*, panels: tuple[RealPanel, ...]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=paper.fig_size(1.0, height_in=2.3), sharey=True)
    for i, panel in enumerate(panels):
        _draw_real_panel(axes[i], panel, show_ylabel=(i == 0))
    handles = [
        Line2D(
            [0],
            [0],
            color=paper.OURS,
            marker="D",
            markersize=4.5,
            linestyle="-",
            markerfacecolor=paper.OURS,
            markeredgecolor=paper.OURS,
        ),
        Line2D(
            [0],
            [0],
            color=paper.OURS,
            marker="D",
            markersize=4.5,
            linestyle="--",
            markerfacecolor="white",
            markeredgecolor=paper.OURS,
        ),
    ]
    fig.legend(
        handles,
        [TEAM_LABEL, EVAL_LABEL],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        columnspacing=1.4,
        fontsize=6.5,
        frameon=False,
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0, 0.21, 1, 1), pad=0.4, w_pad=0.35)
    paper.round_xlabel(fig, y=0.105)
    return fig
