"""Sim-headline panels of the paper's combined real+sim headline.

Reads the frozen per-seed row surface of the simulation baselines
(``paper/data/sim/partial_headline.csv``). Panel grammar: per task
(Square-Narrow / Square-Broad), success rate on a linear axis or failure rate
(100 − SR) on a log axis, mean line + seed-level Student-t 95% CIs + per-seed
dots. Color = arm identity under the paper-wide key (gray HG-DAgger, dark-gray
HG-DAgger+BoN, teal Ours DP, purple Ours DP+BoN, warm muted hues for the
autonomous arms); all arms are solid lines, dashed is reserved for the
static-BC R0 rules. Every plotted point must contain all five planned seeds;
partial rounds fail before rendering.

The panels are the bottom row of ``headline_real_sim``
(:mod:`paper.plotting.sim_paper_headline_combined`).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import t as t_dist

from mulligan.plotting import paper
from mulligan.plotting.colors import METHOD_COLORS

CSV_PATH = Path(__file__).resolve().parents[1] / "data/sim/partial_headline.csv"
PLANNED_SEEDS = 5

# Series keys of the frozen CSV; print-register labels and the
# paper-wide semantic colors. Color = arm identity, matching the real-world
# lineage grammar exactly: gray = HG-DAgger actor-only, dark gray = the
# HiL-IDQL arm, teal = HG-DAgger+Mulligan, purple = HiL-IDQL+Mulligan, the
# full recipe, and purple means nothing else paper-wide. All arms draw solid;
# dashed belongs exclusively to the static-BC R0 rules. The "P" plus marker
# is the paper-wide +BoN marker (as on the real rerank takeover points).
SERIES: dict[str, dict[str, object]] = {
    "human_baseline_n1": {
        "label": paper.POLICY_ARM_LABELS["baseline"],
        "color": paper.BASELINE,
        "marker": "o",
    },
    "human_baseline_n32": {
        "label": paper.POLICY_ARM_LABELS["baseline_bon"],
        "color": paper.BASELINE_RERANK,
        "marker": "P",
    },
    "mulligan_n1": {
        "label": paper.POLICY_ARM_LABELS["mulligan_with_cf"],
        "color": paper.OURS,
        "marker": "D",
    },
    "mulligan_n32": {
        "label": paper.POLICY_ARM_LABELS["final_iql"],
        "color": paper.OURS_RERANK,
        "marker": "P",
    },
    "auto_plain_il_n1": {
        "label": "Autonomous IL",
        "color": METHOD_COLORS["sim_auto_plain_il"],
        "marker": "D",
    },
    "auto_filtered_bc_n1": {
        "label": "Filtered BC",
        "color": METHOD_COLORS["sim_auto_filtered_bc"],
        "marker": "^",
    },
    "auto_iql_n32": {
        "label": "Batch-online IDQL",
        "color": METHOD_COLORS["sim_auto_iql"],
        "marker": "P",
    },
    "auto_iql_success_bc_n32": {
        "label": "Batch-online IDQL + success BC",
        "color": METHOD_COLORS["sim_auto_iql"],
        "marker": "h",
    },
}

TASK_TITLES = paper.SIM_TASK_TITLES


def _format_failure_percent(value: float, _position: int | None = None) -> str:
    """Render log-scale failure-rate ticks in the metric's native percent units."""

    return f"{value:g}%"


def _read_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH}: missing frozen sim headline CSV")
    with CSV_PATH.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{CSV_PATH}: no rows")
    unknown = sorted({row["series_key"] for row in rows} - set(SERIES))
    if unknown:
        raise RuntimeError(
            f"{CSV_PATH}: series {unknown} have no paper style; extend SERIES here "
            "(and the caption)"
        )
    _validate_complete_seed_groups(rows, source=CSV_PATH)
    return rows


def _validate_complete_seed_groups(rows: list[dict[str, str]], *, source: Path | str) -> None:
    """Fail loud unless every persisted plot point has seeds 1..5 exactly once."""
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["task_key"], row["series_key"], int(row["round"]))].append(row)
    expected_seeds = set(range(1, PLANNED_SEEDS + 1))
    for (task_key, series_key, round_number), group in sorted(groups.items()):
        seeds = [int(row["seed"]) for row in group]
        planned = {int(row["n_planned_seeds"]) for row in group}
        statuses = {row["point_status"] for row in group}
        if (
            len(group) != PLANNED_SEEDS
            or set(seeds) != expected_seeds
            or len(set(seeds)) != len(seeds)
            or planned != {PLANNED_SEEDS}
            or statuses != {"complete"}
        ):
            raise RuntimeError(
                f"{source}: INCOMPLETE SIM ROUND {task_key}/{series_key}/R{round_number}: "
                f"rows={len(group)}, seeds={sorted(seeds)}, planned={sorted(planned)}, "
                f"statuses={sorted(statuses)}; expected exactly seeds "
                f"{sorted(expected_seeds)} with status=complete"
            )


def _metric_values(success_values: list[float], metric: str) -> list[float]:
    if metric == "success":
        return success_values
    failures = [100.0 - value for value in success_values]
    if any(value <= 0 for value in failures):
        raise ValueError(
            "log-failure panel requires strictly positive per-seed failure rates; "
            f"found {failures} from success values {success_values}"
        )
    return failures


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    """Mean and two-sided Student-t 95% CI half-width over seed values."""
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()) if arr.size else float("nan"), float("nan")
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return float(arr.mean()), se * float(t_dist.ppf(0.975, arr.size - 1))


def draw_headline_panel(
    axis: plt.Axes, rows: list[dict[str, str]], *, metric: str, task_key: str
) -> None:
    """One sim-headline panel body onto ``axis``: every present series at the
    given metric ("success" linear, zoomed 40-101 / "failure" log), with mean
    line, seed-level Student-t 95% CIs, per-seed dots, and the N=32 lineages' R0 reference
    rules. The caller owns titles, axis labels, y-sharing, layout, and (for
    success panels) the post-layout y-axis-break mark."""
    task_rows = [row for row in rows if row["task_key"] == task_key]
    if not task_rows:
        raise RuntimeError(f"{CSV_PATH}: no rows for task {task_key}")
    axis_min_positive = math.inf
    r0_levels: dict[str, float] = {}
    for series_key, meta in SERIES.items():
        series_rows = [row for row in task_rows if row["series_key"] == series_key]
        if not series_rows:
            continue
        point_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in series_rows:
            point_rows[int(row["round"])].append(row)
        xs = np.asarray(sorted(point_rows), dtype=float)
        means: list[float] = []
        ci95s: list[float] = []
        for round_number in xs.astype(int):
            values = _metric_values(
                [float(row["overall_sr"]) for row in point_rows[round_number]], metric
            )
            mean, ci95 = _mean_ci95(values)
            # The CSV's own mean_sr column must agree with the mean recomputed here.
            csv_means = {float(row["mean_sr"]) for row in point_rows[round_number]}
            if len(csv_means) != 1:
                raise RuntimeError(
                    f"{task_key}/{series_key}/R{round_number}: inconsistent mean_sr "
                    f"values {sorted(csv_means)} in {CSV_PATH}"
                )
            expected = csv_means.pop() if metric == "success" else 100.0 - csv_means.pop()
            if abs(mean - expected) > 0.05:
                raise RuntimeError(
                    f"{task_key}/{series_key}/R{round_number}: recomputed mean "
                    f"{mean:.3f} != stored mean {expected:.3f}"
                )
            if metric == "failure":
                lower = mean if math.isnan(ci95) else mean - ci95
                if lower <= 0:
                    raise ValueError(
                        f"{task_key}/{series_key}/R{round_number} has non-positive "
                        f"failure-rate errorbar lower bound {lower:g}"
                    )
                axis_min_positive = min(axis_min_positive, lower, *values)
            means.append(mean)
            ci95s.append(0.0 if math.isnan(ci95) else ci95)
        if series_key in ("human_baseline_n32", "mulligan_n32") and int(xs[0]) == 0:
            r0_levels[series_key] = means[0]
        axis.plot(
            xs,
            means,
            color=meta["color"],
            linewidth=1.4,
            # Avoid an invalid all-zero dash pattern when a dashed series
            # currently contains only its R0 anchor.
            linestyle=("none" if len(xs) == 1 else str(meta.get("linestyle", "-"))),
            marker="none",
            zorder=3,
        )
        axis.errorbar(
            xs,
            means,
            yerr=ci95s,
            fmt="none",
            ecolor=meta["color"],
            elinewidth=1.2,
            zorder=3,
        )
        for x, mean in zip(xs, means, strict=True):
            axis.plot(
                x,
                mean,
                marker=str(meta["marker"]),
                markersize=4.5,
                markeredgecolor=meta["color"],
                markerfacecolor=meta["color"],
                markeredgewidth=1.0,
                linestyle="none",
                zorder=5,
            )
            seeds = _metric_values([float(row["overall_sr"]) for row in point_rows[int(x)]], metric)
            jitter = np.linspace(-0.05, 0.05, len(seeds)) if len(seeds) > 1 else np.zeros(1)
            axis.scatter(
                x + jitter, seeds, s=4.0, color=meta["color"], alpha=0.6, linewidths=0, zorder=4
            )
    if metric == "success":
        axis.set_ylim(0, 101)
        axis.set_yticks(range(0, 101, 20))
    else:
        if not math.isfinite(axis_min_positive):
            raise ValueError(f"{task_key} failure panel has no positive observations")
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(FuncFormatter(_format_failure_percent))
        axis.set_ylim(10 ** math.floor(math.log10(axis_min_positive)), 100)
        # Decade gridlines alone leave the log row nearly unruled; add
        # unlabeled 2x/5x sub-decade lines at half the major-grid weight.
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(axis="y", which="minor", color=paper.INK, alpha=0.10, linewidth=0.6)
    axis.set_xlim(-0.35, 3.35)
    axis.set_xticks(range(4), [f"R{round_number}" for round_number in range(4)])
    paper.style_axes(axis)
    # Each headline lineage's R0 (BC-only) level carried forward from R1
    # on, as in the real-world headline (the series itself passes through
    # its R0 marker); the N=32 arms are the headline lineages, and the
    # level is stored in the row's own metric space (SR or failure).
    for series_key, level in r0_levels.items():
        paper.r0_reference(axis, level, str(SERIES[series_key]["color"]), xmin=1.0 - 0.35)


def legend_entries(present: set[str]) -> tuple[list[Line2D], list[str]]:
    """Figure-legend handles/labels for every present series, in SERIES order
    (solid line + filled marker in the arm's color, matching the panel
    grammar)."""
    handles = [
        Line2D(
            [0],
            [0],
            color=meta["color"],
            marker=str(meta["marker"]),
            markerfacecolor=meta["color"],
            markeredgecolor=meta["color"],
            markeredgewidth=1.0,
            linestyle=str(meta.get("linestyle", "-")),
            linewidth=1.4,
            markersize=4.5,
            label=str(meta["label"]),
        )
        for series_key, meta in SERIES.items()
        if series_key in present
    ]
    handles.append(paper.static_bc_legend_handle())
    return handles, [str(handle.get_label()) for handle in handles]
