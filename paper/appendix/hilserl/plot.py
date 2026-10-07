"""HiL-SERL sessions in simulation vs. RLPD: policy-only eval success per session.

Two panels on one success scale: Square-Narrow (0-300k; RLPD, the operator-free run,
the three operator-from-step-0 arms, the operator-buffer-then-none arm, and the fork
after takeoff) and Square-Broad (0-500k; RLPD, the operator-free run, and the fork at
200k). Reads the same pinned eval CSVs as ``tab:appendix-hilserl-sessions``.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper.appendix.artifacts import load_inputs, source_paths
from mulligan.plotting import paper
from mulligan.plotting.colors import METHOD_COLORS, STATE_RLPD_CODEBASE_COLORS
from .prepare import HERE, read_eval

NAMES = ("sim_hilserl_sessions",)
# RLPD and the operator forks keep their intro-figure hues (Fig. 2: blue / HiL-SERL
# green); the operator-free run is the neutral ink; the three operator-from-step-0 arms
# share one red (told apart by line style); the arm that kept the operator-filled buffer
# and dropped the operator is gold. Named palette entries only (mulligan.plotting.colors).
HILSERL_ARM_COLORS: dict[str, str] = {
    "rlpd": STATE_RLPD_CODEBASE_COLORS["rlpd"],
    "operator_free": METHOD_COLORS["gray_neutral"],
    "operator_fork": STATE_RLPD_CODEBASE_COLORS["hilserl"],
    "from_step0": METHOD_COLORS["hilserl_from_step0"],
    "buffer_then_none": METHOD_COLORS["hilserl_buffer_then_none"],
}
X_MAX = {"square_narrow": 300_000, "square_broad": 500_000}
ZORDER = {
    "operator_free": 3.5,
    "buffer_then_none": 3.6,
    "operator_fork": 4,
    "from_step0": 4.5,
}
# (task, eval-CSV column, color key, line style, legend label). The from-step-0 arms
# draw on top: they stay at 0-8% exactly where the other curves take off (100-130k).
CURVES: tuple[tuple[str, str, str, object, str], ...] = (
    ("square_narrow", "s1_sr", "from_step0", "-", "From step 0: dense"),
    ("square_narrow", "auto5k_sr", "from_step0", (0, (3.2, 1.6)), "From step 0: sparse bursts"),
    ("square_narrow", "curriculum_sr", "from_step0", (0, (1, 1.2)), "From step 0: long bouts"),
    ("square_narrow", "split_nohuman_sr", "operator_free", "-", "Operator-free"),
    ("square_broad", "split_nohuman_sr", "operator_free", "-", "Operator-free"),
    ("square_narrow", "auto5k_fork130k_sr", "buffer_then_none", "-", "Operator buffer, then none"),
    ("square_narrow", "fork150k_sr", "operator_fork", "-", "Operator fork"),
    ("square_broad", "fork200k_sr", "operator_fork", "-", "Operator fork"),
)


def series(rows: list[dict[str, str]], column: str, x_max: int) -> tuple[list[int], list[float]]:
    points = [
        (int(r["env_step"]), 100.0 * float(r[column]))
        for r in rows
        if r[column] and int(r["env_step"]) <= x_max
    ]
    assert points, column
    return [s for s, _ in points], [v for _, v in points]


def build_figure(evals: dict[str, list[dict[str, str]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=paper.fig_size(1.0, height_in=2.35), sharey=True)
    for ax, task in zip(axes, ("square_narrow", "square_broad"), strict=True):
        rows, x_max = evals[task], X_MAX[task]
        steps, mean = series(rows, "ref_mean_sr", x_max)
        _, se = series(rows, "ref_se", x_max)
        color = HILSERL_ARM_COLORS["rlpd"]
        ax.fill_between(
            steps,
            [m - s for m, s in zip(mean, se)],
            [m + s for m, s in zip(mean, se)],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(steps, mean, color=color, linewidth=1.3, zorder=3)
        for curve_task, column, color_key, style, _label in CURVES:
            if curve_task != task:
                continue
            x, y = series(rows, column, x_max)
            ax.plot(
                x,
                y,
                color=HILSERL_ARM_COLORS[color_key],
                linestyle=style,
                linewidth=1.3,
                zorder=ZORDER[color_key],
            )
        ax.set_xlim(0, x_max)
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(list(range(0, x_max + 1, 100_000)))
        ax.xaxis.set_major_formatter(lambda v, _pos: f"{v / 1000:.0f}k")
        ax.set_xlabel("Environment steps")
        ax.set_title(paper.SIM_TASK_TITLES[task])
        paper.style_axes(ax)
        paper.detach_spines(ax, x_bounds=(0, x_max), y_bounds=(0, 100))
    axes[0].set_ylabel("Eval success (%)")
    labels = ["RLPD (5 seeds)"]
    handles = [Line2D([], [], color=HILSERL_ARM_COLORS["rlpd"], linewidth=1.3)]
    for _task, _column, color_key, style, label in CURVES:
        if label in labels:
            continue
        labels.append(label)
        handles.append(
            Line2D([], [], color=HILSERL_ARM_COLORS[color_key], linestyle=style, linewidth=1.3)
        )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=6.5,
        handlelength=2.2,
        columnspacing=1.2,
        labelspacing=0.3,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.5)
    return fig


def build_records(*, name):
    assert name in NAMES, name
    inputs = load_inputs(HERE)
    evals = {
        task: read_eval(inputs[f"sim/results/hilserl/{task}/eval.csv"])
        for task in ("square_narrow", "square_broad")
    }
    with paper.paper_rc():
        fig = build_figure(evals)
        return paper.save_paper_figure(fig, name, width_frac=1.0, sources=source_paths(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(build_records(name=NAMES[0]).proof_png)


if __name__ == "__main__":
    main()
