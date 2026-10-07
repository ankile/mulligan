"""Paired appendix readouts computed from the same frozen per-state outcomes."""

import numpy as np
import pandas as pd
from paper import real_headline as lh
from mulligan.real.lifecycle.stats import (
    bootstrap_paired_delta,
    exact_mcnemar_pvalue,
    signflip_permutation_pvalue,
)
from paper.appendix.artifacts import write_table

from .prepare import CACHE, save


def _per_start(cfg, round_spec, arm):
    blocks = []
    for src in lh._arm_sr_sources(round_spec, round_spec.policy_arms[arm]):
        frame = pd.read_csv(src.eval_dir / "paired_round_outcomes.csv")
        assert not frame.manifest_idx.duplicated().any()
        blocks.append(frame.set_index("manifest_idx")[f"{src.paired_prefix}_success"].astype(float))
    joined = pd.concat(blocks, axis=1)
    assert not joined.isna().any().any(), "same-start pooling requires the same manifest IDs"
    return joined.mean(axis=1), int(joined.to_numpy().sum()), joined.size


def build_tables(suite):
    rows = []
    for line in suite:
        cfg = line.cfg
        final = cfg.rounds[-1]
        arms = {arm: _per_start(cfg, final, arm) for arm in cfg.headline_arms}
        baseline = "baseline"
        dp = "mulligan_with_cf"
        rerank = next(a for a in cfg.headline_arms if a not in (baseline, dp))
        for a, b in ((dp, baseline), (rerank, baseline), (rerank, dp)):
            av, ak, an = arms[a]
            bv, bk, bn = arms[b]
            np.testing.assert_array_equal(av.index, bv.index)
            seed = 2026090101 if cfg.task_key == "square_d2" else 20260909
            delta, lo, hi = bootstrap_paired_delta(av.to_numpy(), bv.to_numpy(), seed=seed)
            if not (set(av) <= {0.0, 1.0} and set(bv) <= {0.0, 1.0}):
                p = signflip_permutation_pvalue((av - bv).to_numpy(), seed=seed)
                method = "paired sign flip on per-start mean success"
            else:
                p = exact_mcnemar_pvalue(
                    int(((av == 1) & (bv == 0)).sum()), int(((av == 0) & (bv == 1)).sum())
                )
                method = "exact McNemar"
            rows.append(
                dict(
                    task=cfg.task_key,
                    round=final.round_label,
                    a=a,
                    b=b,
                    paired_starts=len(av),
                    a_successes=ak,
                    a_n=an,
                    b_successes=bk,
                    b_n=bn,
                    delta=delta,
                    ci_lo=lo,
                    ci_hi=hi,
                    p=p,
                    method=method,
                )
            )
    save(pd.DataFrame(rows), CACHE / "data/final_round_comparisons.csv")
    routing = suite[-1].cfg
    paired = pd.read_csv(routing.rounds[0].eval_dir / "paired_round_outcomes.csv")
    rows = []
    comparisons = []
    for r in routing.rounds:
        scores = {
            a: paired[f"{v.paired_prefix}_score"].to_numpy(int) for a, v in r.policy_arms.items()
        }
        for arm, score in scores.items():
            assert set(score) <= {0, 1, 2}
            rows.append(
                dict(
                    round=r.round_label,
                    arm=arm,
                    n=len(score),
                    full_successes=int((score == 2).sum()),
                    mean_clip_score=float(score.mean()),
                )
            )
        for a, b in [("mulligan_with_cf", "baseline")] + (
            [("final_iql", "mulligan_with_cf")] if "final_iql" in scores else []
        ):
            delta, lo, hi = bootstrap_paired_delta(scores[a], scores[b], seed=20260907)
            p = signflip_permutation_pvalue(scores[a] - scores[b], seed=20260907)
            comparisons.append(
                dict(round=r.round_label, a=a, b=b, delta=delta, ci_lo=lo, ci_hi=hi, p=p)
            )
    save(pd.DataFrame(rows), CACHE / "data/routing_progress.csv")
    save(pd.DataFrame(comparisons), CACHE / "data/routing_score_comparisons.csv")


def write_tables(*, check=False):
    """Regenerate the real-result tabular fragments; ``check`` requires the manuscript's bytes."""
    from .prepare import prepare

    prepare()
    labels = {"marker_d2": r"\marker", "square_d2": r"\nut", "routing_d2": r"\cable"}
    fragments = {}
    actor_label = r"\shortstack[l]{HG-DAgger+\\Mulligan}"
    rerank_label = r"\shortstack[l]{HiL-IDQL+\\Mulligan}"
    rows = []
    for task in ("marker_d2", "square_d2", "routing_d2"):
        r = pd.read_csv(CACHE / "data" / f"{task}_campaign.csv").query("is_pooled").iloc[0]
        rows.append(
            f"{labels[task]} & ${r.n}$ & ${r.baseline_successes}$ (${100 * r.baseline_successes / r.n:.1f}\\%$) & ${r.treatment_successes}$ (${100 * r.treatment_successes / r.n:.1f}\\%$) & ${100 * r.paired_delta:+.1f}$ pp $[{100 * r.paired_delta_bootstrap_ci95_lo:+.0f}, {100 * r.paired_delta_bootstrap_ci95_hi:+.0f}]$ & $\\mathbf{{{r.mcnemar_exact_pvalue:.4f}}}$ \\\\"
        )
    fragments["campaign"] = (
        r"\begin{tabular}{lccccc}"
        + "\n"
        + r"\hline"
        + "\n"
        + r"Task & Paired $n$ & HG-DAgger & "
        + actor_label
        + r" & Paired $\Delta$ [$95\%$ CI] & Exact McNemar $p$ \\"
        + "\n"
        + r"\hline"
        + "\n"
        + "\n".join(rows)
        + "\n"
        + r"\hline"
        + "\n"
        + r"\end{tabular}"
        + "\n"
    )
    df = pd.read_csv(CACHE / "data/final_round_comparisons.csv")

    def success(k, n):
        return f"${k}/{n}$ (${100 * k / n:.0f}\\%$)"

    def contrast(r):
        p = f"{r.p:.4f}" if r.p < 0.1 else f"{r.p:.3f}"
        ptext = f"\\mathbf{{p = {p}}}" if r.p < 0.05 else f"p = {p}"
        lo, hi = round(100 * r.ci_lo), round(100 * r.ci_hi)
        bounds = f"[{lo:+d}, {hi:+d}]".replace("+0", "0")
        return f"${100 * r.delta:+.0f}$ pp ${bounds}$, ${ptext}$"

    rows = []
    for task in labels:
        sub = df[df.task == task].reset_index(drop=True)
        dp, rb, rd = [r for _, r in sub.iterrows()]
        rows.append(
            f"{labels[task]} R5 & HG-DAgger & {success(dp.b_successes, dp.b_n)} & --- & --- \\\\"
        )
        rows.append(
            f" & {actor_label} & {success(dp.a_successes, dp.a_n)} & "
            + ("vs.\\ base & " + contrast(dp))
            + r" \\"
        )
        rows.append(
            f" & {rerank_label} & {success(rb.a_successes, rb.a_n)} & vs.\\ base & {contrast(rb)} \\\\"
        )
        rows.append(f" & & & vs.\\ actor & {contrast(rd)} \\\\")
        rows.append(r"\hline")
    fragments["final_round"] = (
        r"\begin{tabular}{llccc}"
        + "\n"
        + r"\hline"
        + "\n"
        + r"Task & Arm & Success & Contrast & $\Delta$ [CI], $p$ \\"
        + "\n"
        + r"\hline"
        + "\n"
        + "\n".join(rows)
        + "\n"
        + r"\end{tabular}"
        + "\n"
    )
    df = pd.read_csv(CACHE / "data/routing_progress.csv")
    rows = []
    for idx, (round_, sub) in enumerate(df.groupby("round", sort=False)):
        sub = sub.set_index("arm")
        counts = []
        scores = []
        for arm in ("baseline", "mulligan_with_cf", "final_iql"):
            counts.append(
                str(int(sub.loc[arm, "full_successes"])) if arm in sub.index else r"\text{---}"
            )
            scores.append(
                f"{sub.loc[arm, 'mean_clip_score']:.2f}" if arm in sub.index else r"\text{---}"
            )
        rows.append(
            f"{round_} & ${100 * (idx + 1)}$ & $"
            + "/".join(counts)
            + r"$ & $"
            + "/".join(scores)
            + r"$ \\"
        )
    fragments["routing_progress"] = (
        r"\begin{tabular}{lccc}"
        + "\n"
        + r"\hline"
        + "\n"
        + r"Round & Training episodes / arm & Full successes & Mean clip score \\"
        + "\n"
        + r"\hline"
        + "\n"
        + "\n".join(rows)
        + "\n"
        + r"\hline"
        + "\n"
        + r"\end{tabular}"
        + "\n"
    )
    paths = []
    for name, body in fragments.items():
        text = "% Generated by paper.appendix.real_results.statistics.\n" + body
        paths.append(write_table(f"real_results_{name}.tex", text, check=check))
    return paths
