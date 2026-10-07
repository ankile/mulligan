"""Paper appendix figures from the pinned value-learning extract."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from mulligan.plotting import paper
from mulligan.real.lifecycle.stats import wilson_ci
from paper.appendix.artifacts import fetch, sha256, source_paths, write_json

from .prepare import CACHE, HERE, POLICIES, STEPS

FAMILIES = (
    ("baseline", paper.POLICY_ARM_LABELS["baseline"], paper.BASELINE, "o"),
    ("dp", paper.POLICY_ARM_LABELS["mulligan_with_cf"], paper.OURS, "s"),
    ("iql", paper.POLICY_ARM_LABELS["final_iql"], paper.OURS_RERANK, "P"),
)


def load_tables() -> dict[str, pd.DataFrame]:
    dataset = json.loads((HERE / "dataset.json").read_text())
    assert dataset["sources_sha256"] == sha256(HERE / "sources.json")
    tables = {}
    for row in dataset["files"]:
        path = fetch(CACHE.name, row)
        frame = pd.read_csv(path)
        assert len(frame) == row["rows"]
        tables[Path(row["path"]).stem] = frame
    return tables


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    return float((rankdata(score)[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def analyze(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seed = tables["fqe_seeds"]
    assert set(seed.policy) == set(POLICIES) and len(seed) == 48
    assert not seed.duplicated(["policy", "seed"]).any()
    grouped = seed.groupby("policy").fqe_value.agg(["mean", "std", "count"])
    assert grouped["count"].eq(3).all()
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
    fqe = tables["fqe_policies"].merge(grouped, on="policy", validate="one_to_one")
    fqe["robot_sr"] = fqe.robot_successes / fqe.robot_n
    assert len(fqe) == 16 and fqe.robot_n.isin([50, 100]).all()
    fit_groups = [("all", fqe)] + [(family, fqe[fqe.family.eq(family)]) for family, *_ in FAMILIES]
    correlations = {}
    for name, frame in fit_groups:
        slope, intercept = np.polyfit(frame.robot_sr, frame["mean"], 1)
        correlations[name] = dict(
            n=len(frame),
            rho=float(spearmanr(frame.robot_sr, frame["mean"]).statistic),
            slope=float(slope),
            intercept=float(intercept),
        )
    nine = fqe[~fqe.family.eq("baseline") & fqe["round"].gt(0)]
    correlations["nine_non_baseline"] = dict(
        n=len(nine), rho=float(spearmanr(nine.robot_sr, nine["mean"]).statistic)
    )
    probes = tables["training_probes"]
    episode = tables["training_episodes"]
    training = []
    # Reuse the same stratified episode resamples at every checkpoint.
    rng = np.random.default_rng(20260910)
    pi = rng.integers(0, 23, (5000, 23))
    ni = rng.integers(0, 27, (5000, 27))
    auc_samples = {}
    for step in STEPS:
        p = probes[probes.step.eq(step)].sort_values("manifest_idx")
        e = episode[episode.step.eq(step)].sort_values("manifest_idx")
        assert len(p) == len(e) == 50 and p.success.sum() == e.success.sum() == 23
        pd.testing.assert_frame_equal(
            p[["manifest_idx", "success"]].reset_index(drop=True),
            e[["manifest_idx", "success"]].reset_index(drop=True),
        )
        y, score = p.success.to_numpy(), p.q_cand_max.to_numpy()
        auc = auroc(y, score)
        recorded = tables["training_checkpoints"].set_index("step").loc[step, "recorded_auroc"]
        np.testing.assert_allclose(auc, recorded, atol=1e-12)
        pos, neg = score[y == 1], score[y == 0]
        pair = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        draws = pair[pi[:, :, None], ni[:, None, :]].mean(axis=(1, 2))
        auc_samples[step] = draws
        lo, hi = np.quantile(draws, [0.025, 0.975])
        values = e.loc[e.success.eq(1), "v_s0"]
        training.append(
            dict(
                step=step,
                auroc=auc,
                lo=float(lo),
                hi=float(hi),
                v_mean=float(values.mean()),
                v_se=float(values.std(ddof=1) / np.sqrt(len(values))),
            )
        )
    delta = auc_samples[475000] - auc_samples[150000]
    analysis = dict(
        correlations=correlations,
        training=training,
        auroc_delta_475k_minus_150k_ci=np.quantile(delta, [0.025, 0.975]).tolist(),
        note="Descriptive fits; repeated policies are not independent robot experiments.",
    )
    write_json(CACHE / "analysis.json", analysis)
    return fqe, pd.DataFrame(training), analysis


def build_fqe(fqe: pd.DataFrame, seed: pd.DataFrame, analysis: dict) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=paper.fig_size(1.0, height_in=3.05), sharex=True, sharey=True
    )
    fig.subplots_adjust(left=0.115, right=0.96, bottom=0.29, top=0.88, wspace=0.13)
    for ax in axes:
        for family, label, color, marker in FAMILIES:
            rows = fqe[fqe.family.eq(family)]
            for row in rows.itertuples():
                lo, hi = wilson_ci(int(row.robot_successes), int(row.robot_n))
                ax.errorbar(
                    row.robot_sr * 100,
                    row.mean,
                    xerr=[[100 * (row.robot_sr - lo)], [100 * (hi - row.robot_sr)]],
                    yerr=row.se,
                    fmt=marker,
                    color=color,
                    ms=4,
                    capsize=1.5,
                    elinewidth=0.7,
                    alpha=0.85,
                    zorder=3,
                )
                values = seed.loc[seed.policy.eq(row.policy), "fqe_value"]
                ax.scatter(np.repeat(row.robot_sr * 100, 3), values, s=5, color=color, zorder=4)
        ax.set_xlim(0, 100)
        ax.set_ylim(0.14, 0.425)
        ax.set_xticks([0, 25, 50, 75, 100])
        paper.style_axes(ax)
        ax.set_xlabel("Robot success (%)")
        paper.y_axis_break(ax)
    axes[0].set_ylabel(r"FQE estimate $V(s_0)$")
    axes[0].set_title("(a) Pooled fit")
    axes[1].set_title("(b) Within-family fits")
    fit = analysis["correlations"]["all"]
    x = np.array([fqe.robot_sr.min(), fqe.robot_sr.max()])
    axes[0].plot(x * 100, fit["intercept"] + fit["slope"] * x, color=paper.INK, ls="--", lw=1.1)
    axes[0].text(
        0.04, 0.96, rf"$n=16$, $\rho={fit['rho']:.2f}$", transform=axes[0].transAxes, va="top"
    )
    for i, (family, label, color, marker) in enumerate(FAMILIES):
        rows = fqe[fqe.family.eq(family)]
        fit = analysis["correlations"][family]
        x = np.array([rows.robot_sr.min(), rows.robot_sr.max()])
        axes[1].plot(x * 100, fit["intercept"] + fit["slope"] * x, color=color, ls="--", lw=1.1)
        axes[1].text(
            0.03,
            0.96 - i * 0.11,
            rf"$n={len(rows)}$, $\rho={fit['rho']:+.2f}$",
            color=paper.darken(color),
            transform=axes[1].transAxes,
            va="top",
        )
    handles = [
        Line2D([], [], color=color, marker=marker, ls="none", label=label)
        for _, label, color, marker in FAMILIES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.08),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    paper.small_caps_text(
        axes[0],
        (0.52, 0.04),
        "Insert Marker · historical R0–R5 policies",
        ha="center",
        size=7.5,
        xycoords="figure fraction",
    )
    return fig


def build_training(training: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=paper.fig_size(1.0, height_in=2.55), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.25, top=0.85, wspace=0.47)
    x = training.step / 1000
    color = paper.OURS_RERANK
    axes[0].fill_between(x, training.lo, training.hi, color=color, alpha=0.15, linewidth=0)
    axes[0].plot(x, training.auroc, "o-", color=color, ms=3.5)
    paper.r0_reference(axes[0], 0.5, paper.INK)
    axes[0].set_ylim(0.25, 0.85)
    paper.y_axis_break(axes[0])
    axes[0].set_title("(a) Outcome discrimination")
    axes[0].set_ylabel("Candidate-score AUROC")
    axes[1].errorbar(
        x, training.v_mean, yerr=training.v_se, fmt="s--", color=color, ms=3.5, capsize=2
    )
    axes[1].set_ylim(-0.01, 0.115)
    axes[1].set_title("(b) Initial state value")
    axes[1].set_ylabel(r"Mean $V(s_0)$, successes")
    for ax in axes:
        ax.set_xlim(110, 490)
        ax.set_xticks([150, 250, 350, 450])
        ax.set_xlabel("Optimizer updates (thousands)")
        paper.style_axes(ax)
    paper.small_caps_text(
        axes[0],
        (0.52, 0.04),
        "Insert Marker · R5 critic recipe · same 50 held-out rollouts",
        ha="center",
        size=7.5,
        xycoords="figure fraction",
    )
    return fig


def build_records(*, name: str) -> paper.FigureRecord:
    tables = load_tables()
    fqe, training, analysis = analyze(tables)
    sources = source_paths(HERE)
    with paper.paper_rc():
        if name == "real_world_value_fqe":
            figure = build_fqe(fqe, tables["fqe_seeds"], analysis)
        else:
            assert name == "real_world_value_training"
            figure = build_training(training)
        return paper.save_paper_figure(figure, name, width_frac=1.0, sources=sources)
