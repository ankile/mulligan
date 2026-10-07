"""Verify the frozen Cable (routing_d2) evaluation tables and reproduce their endpoint comparisons.

Checks every file listed in ``paper/data/real/routing/provenance.json`` against its recorded
sha256, reconciles all 15 arms (six checkpoints; three arms from R3) of the full-success and
clip-score headline tables against the per-start paired outcomes and the policy summary,
and recomputes the final-round (R5) paired contrasts: bootstrap 95% interval (20,000
resamples, seed 20260909) and exact McNemar p-value.

    python -m paper.stats.routing_d2_verify
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mulligan.plotting.paper import POLICY_ARM_LABELS
from mulligan.real.lifecycle.stats import bootstrap_paired_delta, exact_mcnemar_pvalue

DATA = Path(__file__).resolve().parents[1] / "data/real/routing"
SEED = 20260909
N_STARTS, N_ROLLOUTS = 50, 750
LINEAGE = {f"R{i}": f"r{r}" for i, r in enumerate((0, 2, 4, 6, 8, 9))}
SUFFIX = {"baseline": "b", "mulligan_with_cf": "o", "final_iql": "i"}
# Final-round contrasts (lineage r9): HiL-IDQL+Mulligan, HG-DAgger+Mulligan, HG-DAgger.
FINAL_PAIRS = (("r9i", "r9b"), ("r9o", "r9b"), ("r9i", "r9o"))


def arm_label(prefix: str) -> str:
    """Paper round and arm name of a paired-outcome prefix, e.g. ``r9i`` -> R5 HiL-IDQL+Mulligan."""
    lineage, suffix = prefix[:-1], prefix[-1]
    (round_label,) = [r for r, lin in LINEAGE.items() if lin == lineage]
    (arm,) = [a for a, s in SUFFIX.items() if s == suffix]
    return f"{round_label} {POLICY_ARM_LABELS[arm]}"


def check_hashes() -> int:
    manifest = json.loads((DATA / "provenance.json").read_text())
    for filename, entry in manifest["files"].items():
        actual = hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise AssertionError(f"{DATA / filename}: sha256 {actual} != provenance.json")
    return len(manifest["files"])


def check_headlines(paired: pd.DataFrame) -> None:
    summary = pd.read_csv(DATA / "policy_summary.csv").set_index("policy_name")
    assert len(paired) == N_STARTS and paired.manifest_idx.nunique() == N_STARTS
    assert len(summary) == 15 and summary.episodes.sum() == N_ROLLOUTS
    full = pd.read_csv(DATA / "routing_d2_headline_sr.csv")
    progress = pd.read_csv(DATA / "routing_d2_headline_task_progress.csv")
    assert len(full) == len(progress) == 15
    for row in full.itertuples():
        prefix = LINEAGE[row.round] + SUFFIX[row.arm]
        success = paired[prefix + "_success"].to_numpy(bool)
        scores = paired[prefix + "_score"].to_numpy(int)
        assert set(scores) <= {0, 1, 2}
        np.testing.assert_array_equal(success, scores == 2)
        assert row.n == len(success) == summary.loc[row.source_policy_name, "episodes"]
        assert row.successes == success.sum() == summary.loc[row.source_policy_name, "successes"]
        np.testing.assert_allclose(row.success_rate, success.mean())
        graded = progress[(progress["round"] == row.round) & (progress.arm == row.arm)].iloc[0]
        assert graded.n == 2 * len(scores) and graded.successes == scores.sum()
        np.testing.assert_allclose(graded.success_rate, scores.mean() / 2)


def final_round_contrasts(paired: pd.DataFrame) -> list[dict]:
    rows = []
    for name_a, name_b in FINAL_PAIRS:
        a = paired[name_a + "_success"].to_numpy(bool)
        b = paired[name_b + "_success"].to_numpy(bool)
        delta, lo, hi = bootstrap_paired_delta(a, b, seed=SEED)
        wins, losses = int((a & ~b).sum()), int((~a & b).sum())
        rows.append(
            dict(
                a=name_a,
                b=name_b,
                a_successes=int(a.sum()),
                b_successes=int(b.sum()),
                n=len(a),
                delta=delta,
                ci_lo=lo,
                ci_hi=hi,
                wins=wins,
                losses=losses,
                p=exact_mcnemar_pvalue(wins, losses),
            )
        )
    return rows


def verify() -> list[dict]:
    """Run every check; return the final-round contrasts."""
    check_hashes()
    prov = json.loads((DATA / "provenance.json").read_text())
    assert prov["reviewed_episode_count"] == N_ROLLOUTS
    paired = pd.read_csv(DATA / "paired_round_outcomes.csv")
    check_headlines(paired)
    return final_round_contrasts(paired)


def main() -> None:
    for r in verify():
        print(
            f"{r['a']} ({arm_label(r['a'])}) vs {r['b']} ({arm_label(r['b'])}): "
            f"{r['a_successes']}/{r['n']} vs {r['b_successes']}/{r['n']}, "
            f"{r['delta'] * 100:+.1f} pp, 95% CI [{r['ci_lo'] * 100:+.1f}, {r['ci_hi'] * 100:+.1f}], "
            f"p={r['p']:.6f}"
        )
    print(
        f"Verified source hashes, all 15 arms / {N_ROLLOUTS} outcomes, clip scores, "
        "and endpoint tests."
    )


if __name__ == "__main__":
    main()
