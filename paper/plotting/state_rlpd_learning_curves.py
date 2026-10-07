"""Intro figure: state-based RLPD vs HiL-IDQL+Mulligan vs HiL-SERL (``sim_state_rlpd_vs_mulligan_compact``).

Reads the frozen learning-curve CSVs in ``paper/data/online_rl``:

* ``state_rlpd_learning_curves{,_square_broad}.csv``: evaluation success (50 episodes
  per point) vs environment step of the RLPD runs, one row per run key
  (``<recipe id>/seed-<N>``) and step. Square-Narrow arms, 300k steps:
  ``square-narrow-rlpd`` (100 teleop demos) and ``square-narrow-rlpd-robomimic-ph``
  (the 200 robomimic PH demos). Square-Broad arms, 1M steps: ``square-broad-rlpd``
  (200 teleop demos) and ``square-broad-rlpd-mimicgen-core`` (the 1000 MimicGen core
  demos). Five learner seeds per arm, drawn as the mean with a ±1 SE band on the
  steps every seed has reached.
* ``state_rlpd_ours_reference{,_square_broad}.csv``: HiL-IDQL+Mulligan (DP+BoN, N=32)
  grid success per DAgger round at the cumulative number of collected frames.
* ``hilserl_sessions_eval.csv`` / ``hilserl_square_broad_eval.csv``: the one-seed
  HiL-SERL runs (operator-free to the fork checkpoint, then the forked session).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from mulligan.plotting import paper
from mulligan.plotting.colors import STATE_RLPD_CODEBASE_COLORS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "paper/data/online_rl"
# Axes read in thousands of environment steps.
STEP_FORMATTER_K = FuncFormatter(lambda v, _pos: f"{int(v / 1000)}k")
# Learner seeds per RLPD arm (run key ``<recipe id>/seed-<N>``).
SEEDS: tuple[int, ...] = tuple(range(1, 6))
TASKS: dict[str, dict[str, Path]] = {
    "square_narrow": {
        "curve_csv": DATA_DIR / "state_rlpd_learning_curves.csv",
        "ours_csv": DATA_DIR / "state_rlpd_ours_reference.csv",
    },
    "square_broad": {
        "curve_csv": DATA_DIR / "state_rlpd_learning_curves_square_broad.csv",
        "ours_csv": DATA_DIR / "state_rlpd_ours_reference_square_broad.csv",
    },
}


def aggregate_seeds(per_run: list[list[dict[str, object]]]):
    """Mean and ±1 SE across seeds of ``success_rate`` (in %) on the steps that
    EVERY seed has reached (a still-running seed truncates the arm rather than
    letting the mean silently drop to fewer seeds). Returns (steps, mean, se)."""
    common = set(int(r["step"]) for r in per_run[0])
    for rows in per_run[1:]:
        common &= set(int(r["step"]) for r in rows)
    if not common:
        raise RuntimeError("seeds share no evaluation steps")
    steps = sorted(common)
    values = np.asarray(
        [
            [100.0 * float(r["success_rate"]) for r in rows if int(r["step"]) in common]
            for rows in per_run
        ]
    )
    if values.shape[1] != len(steps):
        raise RuntimeError("duplicate evaluation steps within a seed")
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(per_run)) if len(per_run) > 1 else None
    return steps, mean, se


def run_keys(recipe: str) -> tuple[str, ...]:
    """The five seeds' run keys of an RLPD recipe."""
    return tuple(f"{recipe}/seed-{seed}" for seed in SEEDS)


# Per task: the RLPD recipes on this project's R0 (teleop) demos and on the
# released demos (five seeds each).
OVERVIEW_PANELS: tuple[dict[str, object], ...] = (
    {
        "task": "square_narrow",
        "ours_demos": run_keys("square-narrow-rlpd"),
        "released_demos": run_keys("square-narrow-rlpd-robomimic-ph"),
    },
    {
        "task": "square_broad",
        "ours_demos": run_keys("square-broad-rlpd"),
        "released_demos": run_keys("square-broad-rlpd-mimicgen-core"),
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# Both tasks on ONE axis at 0.36 textwidth (intro wrapfigure): Ours in purple,
# RLPD on our R0 demos in blue, RLPD on the released demos in neutral gray,
# HiL-SERL in its own hue; the task is carried by line style. One 500k-step axis: the Square-Narrow RL
# arms end at their 300k-step budget; the Square-Broad arms are the 1M-step
# seed groups drawn to 500k, far enough to show the released-demo plateau.
PAPER_COMPACT_NAME = "sim_state_rlpd_vs_mulligan_compact"
PAPER_COMPACT_WIDTH_FRAC = 0.36
PAPER_COMPACT_X_MAX = 500_000
# (overview panel index, line style, legend label, last step drawn)
PAPER_COMPACT_TASKS: tuple[tuple[int, object, str, int], ...] = (
    (0, "-", "Narrow", 300_000),
    (1, (0, (3.2, 1.6)), "Broad", PAPER_COMPACT_X_MAX),
)
PAPER_COMPACT_ARMS: tuple[tuple[str, str, str, float, int], ...] = (
    # (panel key of the run keys, color, legend label, line width, zorder).
    # The blue arm draws above the gray one: on Square-Broad it is flat at 0%
    # and would otherwise vanish under the gray band and the axis spine.
    ("ours_demos", STATE_RLPD_CODEBASE_COLORS["rlpd"], "RLPD, our R0 demos", 1.3, 4),
    ("released_demos", paper.INK, "RLPD, released demos", 1.1, 3),
)
PAPER_COMPACT_OURS_LABEL = "HiL-IDQL+Mulligan"
# Ours is context here (the RLPD gap is the takeaway): faded, no markers, so the
# solid/dashed task styles stay legible, and listed after the RLPD arms.
PAPER_COMPACT_OURS_ALPHA = 0.5
# HiL-SERL, one seed per task: the split system's zero-intervention run to the
# fork checkpoint, then the forked session (operator in the loop). The two share
# every transition up to the fork, so each curve is one run. No marker at the
# fork: the takeaway is the curve, and the protocol is in the paper appendix.
# Square-Narrow forks at 150k and ends at 300k; Square-Broad forks at 200k and
# ends at 500k, the axis end.
# Read from the frozen per-milestone HiL-SERL eval CSVs;
# (CSV, fork column, fork step, last step drawn, zorder), in PAPER_COMPACT_TASKS
# order. The Square-Broad curve draws UNDER the flat-at-0% own-demo RLPD arm
# (zorder 4) so the in-place "Broad: 0%" label keeps pointing at a visible blue
# line; its 2-8% points still show above it.
HILSERL_EVAL_INTERVAL = 10_000
HILSERL_CURVES: tuple[tuple[Path, str, int, int, float], ...] = (
    (DATA_DIR / "hilserl_sessions_eval.csv", "fork150k_sr", 150_000, 300_000, 4.5),
    (DATA_DIR / "hilserl_square_broad_eval.csv", "fork200k_sr", 200_000, PAPER_COMPACT_X_MAX, 3.5),
)
PAPER_COMPACT_HILSERL_LABEL = "HiL-SERL"


def hilserl_fork_curve(
    csv_path: Path, fork_col: str, fork_step: int, end_step: int
) -> tuple[list[int], list[float]]:
    """(steps, success %) of a HiL-SERL fork run, 0 to ``end_step`` at every 10k
    milestone: the no-human rows up to the fork, the fork's rows after it."""
    rows = {int(r["env_step"]): r for r in read_csv(csv_path)}
    steps = list(range(0, end_step + 1, HILSERL_EVAL_INTERVAL))
    missing = [
        s
        for s in steps
        if s not in rows or not rows[s]["split_nohuman_sr" if s <= fork_step else fork_col]
    ]
    if missing:
        raise RuntimeError(f"{csv_path}: HiL-SERL {fork_col} curve missing steps {missing}")
    fork_row = rows[fork_step]
    if fork_row["split_nohuman_sr"] != fork_row[fork_col]:
        raise RuntimeError(
            f"{csv_path}: {fork_col} at the fork ({fork_row[fork_col]}) != "
            f"no-human ({fork_row['split_nohuman_sr']})"
        )
    values = [
        100.0 * float(rows[s]["split_nohuman_sr" if s <= fork_step else fork_col]) for s in steps
    ]
    return steps, values


def build_paper_records() -> list[paper.FigureRecord]:
    """Paper figure ``sim_state_rlpd_vs_mulligan_compact.pdf``: state-based
    RLPD vs HiL-IDQL+Mulligan on Square-Narrow (solid) and
    Square-Broad (dashed), one axis, plus the one-seed HiL-SERL fork runs
    (fork150k on Square-Narrow, fork200k on Square-Broad), authored at 0.36
    textwidth for the introduction's wrapfigure."""
    from matplotlib.lines import Line2D

    sources: list[str] = [str(Path(__file__).resolve().relative_to(ROOT))]
    with paper.paper_rc():
        # Height budget: the intro wrapfigure (figure + caption) must end inside
        # the paragraph it starts in; a wrapfigure that crosses a paragraph
        # break leaves the following text narrow on newer TeX Live (Overleaf).
        # The legend + x-label block keeps its absolute height (0.85 in).
        height_in = 1.70
        fig, ax = plt.subplots(
            figsize=paper.fig_size(PAPER_COMPACT_WIDTH_FRAC, height_in=height_in)
        )
        fig.subplots_adjust(
            left=0.27, right=0.93, top=1 - 0.055 / height_in, bottom=0.85 / height_in
        )
        for panel_idx, linestyle, _task_label, x_end in PAPER_COMPACT_TASKS:
            panel = OVERVIEW_PANELS[panel_idx]
            cfg = TASKS[str(panel["task"])]
            sources += [
                str(Path(cfg["curve_csv"]).relative_to(ROOT)),
                str(Path(cfg["ours_csv"]).relative_to(ROOT)),
            ]
            curves = read_csv(Path(cfg["curve_csv"]))
            ours = read_csv(Path(cfg["ours_csv"]))
            for keys_field, color, _label, linewidth, zorder in PAPER_COMPACT_ARMS:
                per_run = []
                for key in panel[keys_field]:
                    rows = [r for r in curves if r["run_key"] == key and int(r["step"]) <= x_end]
                    if rows:
                        per_run.append(rows)
                if len(per_run) != len(SEEDS):
                    raise RuntimeError(
                        f"{PAPER_COMPACT_NAME} {panel['task']} {keys_field}: "
                        f"{len(per_run)} of {len(SEEDS)} seeds in {cfg['curve_csv']}"
                    )
                steps, mean, se = aggregate_seeds(per_run)
                if int(steps[-1]) != x_end:
                    raise RuntimeError(
                        f"{PAPER_COMPACT_NAME} {panel['task']} {keys_field}: curves stop at "
                        f"{steps[-1]} < {x_end}"
                    )
                ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.16, linewidth=0)
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    zorder=zorder,
                )
                if panel_idx == 1 and keys_field == "ours_demos":
                    # Flat-at-zero arm: name it in place so a reader sees it
                    # is a result, not the axis.
                    if float(mean.max()) > 2.0:
                        raise RuntimeError(
                            f"{PAPER_COMPACT_NAME}: Square-Broad own-demo RLPD peaks at "
                            f"{mean.max():.1f}%; the in-place 0% label is no longer true"
                        )
                    ax.annotate(
                        "our R0 demos,\nBroad: 0%",
                        xy=(0.98 * PAPER_COMPACT_X_MAX, float(mean[-1])),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        fontsize=7.0,
                        linespacing=1.1,
                        color=color,
                    )
            ax.errorbar(
                [int(r["cumulative_collected_frames"]) for r in ours],
                [100.0 * float(r["success_rate_mean"]) for r in ours],
                yerr=[100.0 * float(r["success_rate_se"]) for r in ours],
                color=paper.OURS_RERANK,
                alpha=PAPER_COMPACT_OURS_ALPHA,
                linestyle=linestyle,
                linewidth=1.6,
                capsize=1.5,
                zorder=5,
                clip_on=False,
            )
        hilserl_color = STATE_RLPD_CODEBASE_COLORS["hilserl"]
        for (csv_path, fork_col, fork_step, end_step, zorder), (_i, linestyle, _l, _x) in zip(
            HILSERL_CURVES, PAPER_COMPACT_TASKS, strict=True
        ):
            hilserl_steps, hilserl_values = hilserl_fork_curve(
                csv_path, fork_col, fork_step, end_step
            )
            sources.append(str(csv_path.relative_to(ROOT)))
            ax.plot(
                hilserl_steps,
                hilserl_values,
                color=hilserl_color,
                linestyle=linestyle,
                linewidth=1.3,
                zorder=zorder,
            )
        ax.set_xlim(0, PAPER_COMPACT_X_MAX)
        # Detached (R-style) spines below: a little headroom under zero keeps the
        # flat-at-0% arm's SE band inside the axes.
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(list(range(0, PAPER_COMPACT_X_MAX + 1, 250_000)))
        ax.set_xticks(list(range(0, PAPER_COMPACT_X_MAX + 1, 50_000)), minor=True)
        ax.xaxis.set_major_formatter(STEP_FORMATTER_K)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Success (%)")
        paper.style_axes(ax)
        paper.detach_spines(ax, x_bounds=(0, PAPER_COMPACT_X_MAX), y_bounds=(0, 100))
        handles = [
            *(
                Line2D([], [], color=color, linewidth=linewidth)
                for _f, color, _l, linewidth, _z in PAPER_COMPACT_ARMS
            ),
            Line2D([], [], color=paper.OURS_RERANK, alpha=PAPER_COMPACT_OURS_ALPHA, linewidth=1.6),
            # Second legend column (ncol=2 fills column-major): short labels only,
            # or the column runs past the figure edge.
            Line2D([], [], color=STATE_RLPD_CODEBASE_COLORS["hilserl"], linewidth=1.3),
            *(
                Line2D([], [], color=paper.INK, linestyle=linestyle, linewidth=1.2)
                for _i, linestyle, _l, _x in PAPER_COMPACT_TASKS
            ),
        ]
        labels = [
            *(label for _f, _c, label, _w, _z in PAPER_COMPACT_ARMS),
            PAPER_COMPACT_OURS_LABEL,
            PAPER_COMPACT_HILSERL_LABEL,
            *(label for _i, _ls, label, _x in PAPER_COMPACT_TASKS),
        ]
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            ncol=2,
            frameon=False,
            fontsize=6.5,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.0,
            labelspacing=0.28,
        )
        record = paper.save_paper_figure(
            fig,
            PAPER_COMPACT_NAME,
            width_frac=PAPER_COMPACT_WIDTH_FRAC,
            sources=tuple(sources),
        )
    return [record]
