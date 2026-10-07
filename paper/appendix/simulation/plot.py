"""Native print-size simulation appendix figures from the locked archive."""

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import qmc, t
from mulligan.plotting import paper
from mulligan.plotting.colors import METHOD_COLORS
from paper.appendix.artifacts import source_paths
from .prepare import CACHE, CONDITIONS, HERE, extract

# Same semantic identities as the simulation headline, using the shared palette.
SERIES = {
    "human_baseline_n1": (paper.POLICY_ARM_LABELS["baseline"], paper.BASELINE, "o"),
    "human_baseline_n32": (paper.POLICY_ARM_LABELS["baseline_bon"], paper.BASELINE_RERANK, "P"),
    "mulligan_n1": (paper.POLICY_ARM_LABELS["mulligan_with_cf"], paper.OURS, "D"),
    "mulligan_n32": (paper.POLICY_ARM_LABELS["final_iql"], paper.OURS_RERANK, "P"),
    "auto_plain_il_n1": ("Autonomous IL", METHOD_COLORS["sim_auto_plain_il"], "D"),
    "auto_filtered_bc_n1": ("Filtered BC", METHOD_COLORS["sim_auto_filtered_bc"], "^"),
    "auto_iql_n32": ("Batch-online IDQL", METHOD_COLORS["sim_auto_iql"], "P"),
}


def critic(data):
    fig, ax = plt.subplots(figsize=paper.fig_size(1.0, height_in=3.35))
    for i, (condition, label) in enumerate(CONDITIONS):
        values = data.loc[data.condition == condition, "success_rate"].to_numpy()
        assert len(values) >= 3
        mean = values.mean()
        se = values.std(ddof=1) / np.sqrt(len(values))
        color = paper.BASELINE if i == 0 else paper.OURS_RERANK
        ax.errorbar(mean, i, xerr=se, fmt="o", color=color, capsize=2, markersize=4)
        ax.scatter(
            values,
            i + np.linspace(-0.11, 0.11, len(values)),
            s=10,
            color=color,
            alpha=0.45,
            edgecolors="none",
        )
        ax.text(100.5, i, f"{mean:.2f} ± {se:.2f}  (n={len(values)})", va="center", fontsize=7)
    ax.set_yticks(range(len(CONDITIONS)), [label for _, label in CONDITIONS])
    ax.invert_yaxis()
    ax.set_xlim(80, 100)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_xlabel("Final-step success rate (%)")
    paper.style_axes(ax, ygrid=False, xgrid=True)
    fig.subplots_adjust(left=0.38, right=0.78, top=0.96, bottom=0.15)
    return fig


def bucket(data):
    fig, ax = plt.subplots(figsize=paper.fig_size(1.0, height_in=2.3))
    rates = 100 * data.successes / data.rollouts
    overall = 100 * data.successes.sum() / data.rollouts.sum()
    colors = np.where(data["rank"] <= 20, paper.SOBOL, paper.NEUTRAL_FILL)
    ax.bar(data["rank"], rates, width=0.84, color=colors, edgecolor="none")
    ax.axhline(overall, color=paper.INK, ls="--", lw=0.9)
    ax.text(79, overall - 3, f"Overall {overall:.2f}%", ha="right", va="top", fontsize=7)
    share = (data.rollouts - data.successes).iloc[:20].sum() / (
        data.rollouts - data.successes
    ).sum()
    ax.text(2, 108, f"Hardest 20/80 states: {share:.1%} of failures", va="top", fontsize=7.5)
    ax.set(
        xlim=(0.25, 80.75),
        ylim=(0, 112),
        xlabel="Initial-state rank (hardest → easiest)",
        ylabel="Success rate (%)",
    )
    ax.set_xticks([1, 20, 40, 60, 80])
    ax.set_yticks([0, 25, 50, 75, 100])
    paper.style_axes(ax)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.96, bottom=0.23)
    return fig


def coverage(data):
    fig = plt.figure(figsize=paper.fig_size(1.0, height_in=3.55))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.9, 1], hspace=0.92, wspace=0.85)
    metrics = []
    for i, (arm, label, color) in enumerate(
        [("uniform", "Uniform HG-DAgger", paper.BASELINE), ("sobol", "Sobol", paper.SOBOL)]
    ):
        d = data.loc[data.arm == arm]
        yaw = (d.nut_yaw.to_numpy() + np.pi) % (2 * np.pi)
        points = np.column_stack(((d.nut_y - 0.11) / 0.115, yaw / (2 * np.pi)))
        assert np.all((points >= 0) & (points <= 1))
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        marginal = np.mean(
            [np.histogram(points[:, j], bins=10, range=(0, 1))[0].std() for j in range(2)]
        )
        metrics.append([qmc.discrepancy(points, method="CD"), distance.min(), marginal])
        ax = fig.add_subplot(gs[0, i * 3 : (i + 1) * 3])
        ax.scatter(
            d.nut_y,
            np.rad2deg(yaw) - 180,
            s=8,
            color=color,
            marker="o" if i == 0 else "D",
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_title(f"{label} (n=100)")
        ax.set_xlabel("Nut y (m)")
        if i == 0:
            ax.set_ylabel("Nut yaw (deg)")
        ax.set(xlim=(0.105, 0.230), ylim=(-185, 185))
        ax.set_yticks([-180, -90, 0, 90, 180])
        paper.style_axes(ax, xgrid=True)
    for j, title in enumerate(
        [
            "Centered discrepancy\n(lower is better)",
            "Min. normalized distance\n(higher is better)",
            "10-bin count std.\n(lower is better)",
        ]
    ):
        ax = fig.add_subplot(gs[1, 2 * j : 2 * (j + 1)])
        values = np.array(metrics)[:, j]
        ax.bar([0, 1], values, color=[paper.BASELINE, paper.SOBOL], width=0.65)
        ax.set_xticks([0, 1], ["Uniform", "Sobol"])
        ax.set_title(title, fontsize=7.5)
        for i, v in enumerate(values):
            ax.text(
                i, v, f"{v:.4f}" if v < 0.1 else f"{v:.2f}", ha="center", va="bottom", fontsize=7
            )
        ax.set_ylim(0, values.max() * 1.35)
        paper.style_axes(ax)
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.10, top=0.91)
    (CACHE / "data/coverage_metrics.json").write_text(
        json.dumps(
            dict(
                zip(
                    ["uniform", "sobol"],
                    [
                        dict(
                            zip(
                                [
                                    "centered_discrepancy",
                                    "min_normalized_distance",
                                    "marginal_count_std",
                                ],
                                v,
                            )
                        )
                        for v in metrics
                    ],
                )
            ),
            indent=2,
        )
        + "\n"
    )
    return fig


def efficiency(data, metric):
    column = {
        "speed": "seed_mean_success_seconds",
        "throughput": "seed_success_throughput_per_min",
    }[metric]
    assert set(data.series_key) == set(SERIES)
    fig, axes = plt.subplots(1, 2, figsize=paper.fig_size(1.0, height_in=2.55))
    for ax, task in zip(axes, ("square_narrow", "square_broad"), strict=True):
        limits = []
        for key, (_, color, marker) in SERIES.items():
            sub = data.loc[(data.task_key == task) & (data.series_key == key)]
            xs = []
            means = []
            errors = []
            for round_, g in sub.groupby("round"):
                values = g.sort_values("seed")[column].to_numpy()
                assert len(values) == 5 and np.isfinite(values).all()
                mean = values.mean()
                err = t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5)
                xs.append(round_)
                means.append(mean)
                errors.append(err)
                limits.extend([mean - err, mean + err, *values])
                ax.scatter(
                    round_ + np.linspace(-0.05, 0.05, 5),
                    values,
                    s=4,
                    color=color,
                    alpha=0.6,
                    linewidth=0,
                    zorder=4,
                )
            assert xs, (task, key)
            ax.errorbar(
                xs,
                means,
                yerr=errors,
                color=color,
                marker=marker,
                markersize=4.5,
                lw=1.4,
                elinewidth=1.2,
            )
        if metric == "speed":
            pad = max(0.15, 0.10 * (max(limits) - min(limits)))
            ax.set_ylim(min(limits) - pad, max(limits) + pad)
        else:
            ax.set_ylim(0, max(limits) * 1.10)
        ax.set_title(paper.SIM_TASK_TITLES[task])
        ax.set_xlim(-0.35, 3.35)
        ax.set_xticks(range(4), [f"R{i}" for i in range(4)])
        paper.style_axes(ax)
    axes[0].set_ylabel("Time per task unit (s)" if metric == "speed" else "Task units per minute")
    handles = [
        Line2D([0], [0], label=label, color=color, marker=marker, markersize=4.5, lw=1.4)
        for label, color, marker in SERIES.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0),
        fontsize=6,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1), pad=0.45, w_pad=0.65)
    paper.round_xlabel(fig, y=0.125)
    if metric == "speed":
        for ax in axes:
            paper.y_axis_break(ax)
    return fig


def build_records(*, name: str):
    paths = extract()
    with paper.paper_rc():
        if name == "value_learning_bar_chart":
            fig = critic(pd.read_csv(paths["critic_objective_seeds.csv"]))
        elif name == "square_narrow_r2_bucket_success_sorted":
            fig = bucket(pd.read_csv(paths["bucket_counts.csv"]))
        elif name == "square_narrow_init_distribution_sobol_vs_uniform":
            fig = coverage(pd.read_csv(paths["initial_states.csv"]))
        elif name in ("sim_success_speed", "sim_success_throughput"):
            fig = efficiency(
                pd.read_csv(paths["efficiency_seeds.csv"]), name.removeprefix("sim_success_")
            )
        else:
            raise ValueError(name)
        return paper.save_paper_figure(fig, name, width_frac=1.0, sources=source_paths(HERE))
