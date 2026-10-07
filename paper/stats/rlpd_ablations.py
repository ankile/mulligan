"""RLPD recipe ablations on Square-Narrow (A18, baselines appendix, "RLPD ablations" bullet).

Two single-knob changes to the 5-seed RLPD reference (R0, 100 demos, UTD 20): early
termination of doomed training episodes, and UTD 40 (critic updates only). Each arm has
five seeds with eval success every 10k environment steps up to 300k. The paper quotes the
seed counts, the ~80k-step zero-success phase (median seed's first nonzero eval) and that
neither knob changed learning speed: the 95% Welch interval of each arm's whole-curve mean
eval minus the reference's lies within +/-0.12.

Recomputes every metric of ``rlpd_square_narrow_ablations_summary.json`` from the per-seed
curves in ``rlpd_square_narrow_ablations_eval.csv`` (both frozen from the original ingest),
checks them against the frozen summary, and prints the quoted numbers.

    python -m paper.stats.rlpd_ablations
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper.stats.sim_contrasts import welch

DATA = Path(__file__).resolve().parents[1] / "data/online_rl"
EVAL_CSV = DATA / "rlpd_square_narrow_ablations_eval.csv"
SUMMARY = DATA / "rlpd_square_narrow_ablations_summary.json"
REFERENCE = "reference_utd20"
ARMS = (REFERENCE, "early_kill", "utd40")
EVAL_STEPS = np.arange(0, 300_001, 10_000)


def first_crossing_k(curve: np.ndarray, threshold: float) -> float:
    hits = np.flatnonzero(curve >= threshold)
    if not len(hits):
        raise ValueError(f"curve never reaches {threshold}")
    return float(EVAL_STEPS[hits[0]] / 1000)


METRICS = {
    "auc_0_300k": lambda c: float(c.mean()),
    "sr_300k": lambda c: float(c[-1]),
    "sr_mean_250_300k": lambda c: float(c[EVAL_STEPS >= 250_000].mean()),
    "first_step_k_sr_ge_0.2": lambda c: first_crossing_k(c, 0.2),
    "first_step_k_sr_ge_0.5": lambda c: first_crossing_k(c, 0.5),
}


def curves(path: Path = EVAL_CSV) -> dict[str, np.ndarray]:
    """{arm: (seeds, eval steps) success rates}, seeds in ascending order."""
    rows: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows[row["arm"]][int(row["seed"])][int(row["env_step"])] = float(
                row["eval_success_rate"]
            )
    if tuple(sorted(rows)) != tuple(sorted(ARMS)):
        raise ValueError(f"{path}: arms {sorted(rows)}, expected {sorted(ARMS)}")
    out = {}
    for arm in ARMS:
        by_seed = rows[arm]
        for seed, by_step in by_seed.items():
            if sorted(by_step) != EVAL_STEPS.tolist():
                raise ValueError(
                    f"{path}: {arm} seed {seed} does not cover {len(EVAL_STEPS)} evals"
                )
        out[arm] = np.array([[by_seed[s][int(k)] for k in EVAL_STEPS] for s in sorted(by_seed)])
    return out


def summarize(per_arm_curves: dict[str, np.ndarray]) -> dict:
    """The frozen summary's layout: per-arm means and each arm minus the reference (Welch)."""
    out: dict = {"n_seeds": {k: len(v) for k, v in per_arm_curves.items()}, "metrics": {}}
    for name, fn in METRICS.items():
        per_arm = {k: np.array([fn(c) for c in v]) for k, v in per_arm_curves.items()}
        entry: dict = {"mean": {k: round(float(v.mean()), 4) for k, v in per_arm.items()}}
        for arm, values in per_arm.items():
            if arm == REFERENCE:
                continue
            diff, lo, hi, p = welch(values, per_arm[REFERENCE])
            entry[f"{arm}_minus_reference"] = {
                "diff": round(diff, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "welch_p": round(p, 3),
            }
        out["metrics"][name] = entry
    return out


def check_summary(summary: dict, path: Path = SUMMARY) -> None:
    frozen = json.loads(path.read_text())
    if summary != frozen:
        raise AssertionError(f"recomputed summary differs from {path.name}")


def median_first_nonzero_k(arm_curves: np.ndarray) -> float:
    """Median over seeds of the first eval step (k) with nonzero success."""
    firsts = [EVAL_STEPS[np.flatnonzero(c > 0)[0]] / 1000 for c in arm_curves]
    return float(np.median(firsts))


def quoted() -> dict:
    """The numbers the appendix quotes, after checking the summary against the curves."""
    per_arm = curves()
    summary = summarize(per_arm)
    check_summary(summary)
    auc = summary["metrics"]["auc_0_300k"]
    bound = max(abs(x) for arm in ARMS[1:] for x in auc[f"{arm}_minus_reference"]["ci95"])
    return {
        "n_seeds": summary["n_seeds"],
        "zero_phase_k": {arm: median_first_nonzero_k(c) for arm, c in per_arm.items()},
        "auc_mean": auc["mean"],
        "auc_diff": {arm: auc[f"{arm}_minus_reference"]["diff"] for arm in ARMS[1:]},
        "auc_ci95": {arm: auc[f"{arm}_minus_reference"]["ci95"] for arm in ARMS[1:]},
        # "within +/-0.12": the widest interval bound, rounded up to two decimals.
        "auc_ci_bound": math.ceil(bound * 100) / 100,
    }


def main() -> None:
    q = quoted()
    print(f"Summary {SUMMARY.name} recomputed from {EVAL_CSV.name}: equal.")
    print("Seeds per arm: " + ", ".join(f"{k} {v}" for k, v in q["n_seeds"].items()))
    print(
        "Median seed's first nonzero eval (k): "
        + ", ".join(f"{k} {v:g}" for k, v in q["zero_phase_k"].items())
    )
    for arm in ARMS[1:]:
        lo, hi = q["auc_ci95"][arm]
        print(
            f"Whole-curve mean eval, {arm} - reference: "
            f"{q['auc_diff'][arm]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        )
    print(f"All intervals within +/-{q['auc_ci_bound']:.2f} (paper: +/-0.12).")


if __name__ == "__main__":
    main()
