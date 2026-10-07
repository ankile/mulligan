"""Derive the main-text ablation and burden panel data from hashed raw inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.appendix.artifacts import REAL_DIRS, cache_dir, load_inputs
from paper.plotting.composition_analysis import (
    PredecessorPolicy,
    compute_predecessor_grid_sr,
)

HERE = Path(__file__).resolve().parent
CACHE = cache_dir("component_panels")
NAMES = ("ablations_sim_real", "collection_burden_real_sim")
CELLS = {
    (
        "no_cf",
        "human_only",
        "baseline_uniform",
    ): "square_narrow_r1_baseline_uniform_no_cf_human_only",
    (
        "no_cf",
        "straddled_auto_success",
        "baseline_uniform",
    ): "square_narrow_r1_baseline_uniform_no_cf_straddled_auto_success",
    (
        "no_cf",
        "human_only",
        "mulligan",
    ): "square_narrow_r1_mulligan_no_cf_human_only",
    (
        "with_cf",
        "human_only",
        "mulligan",
    ): "square_narrow_r1_mulligan_with_cf_human_only",
}
SHARED = (
    "paper/plotting/square_narrow_r1_sampler.py",
    "paper/plotting/square_narrow_r1_intervention.py",
    "paper/plotting/cf_ablation.py",
    "paper/plotting/burden_progression.py",
    "paper/plotting/composition_analysis.py",
    "paper/real_headline.py",
    "mulligan/real/lifecycle/stats.py",
)


def prepare():
    raw = load_inputs(HERE)
    out = CACHE / "data"
    out.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(raw["sim/critic/frozen_actor_results.csv"])
    groups = {}
    selected = []
    for key, cell in CELLS.items():
        group = frozen.loc[(frozen.family == "hil") & (frozen.cell_key == cell)].sort_values("seed")
        assert group.seed.tolist() == [1, 2, 3, 4, 5], cell
        assert set(group.status) == {"grid_complete"}
        assert np.isfinite(group.grid_n32_pct).all() and group.grid_n32_pct.between(0, 100).all()
        groups[key] = group.grid_n32_pct.tolist()
        selected.append(group)
    pd.concat(selected).to_csv(out / "sampler_seeds.csv", index=False)
    cf_paths = {}
    for task, cap in (("square_d2", 4), ("marker_d2", 2)):
        path = raw[f"real/results/{REAL_DIRS[task]}/headline/{task}_headline_sr_all_arms.csv"]
        frame = pd.read_csv(path)
        frame = frame.loc[
            frame.is_pooled
            & frame.arm.isin(["mulligan_with_cf", "mulligan_no_cf"])
            & frame["round"].isin([f"R{r}" for r in range(cap + 1)])
        ].copy()
        assert not frame.duplicated(["round", "arm"]).any()
        for arm, rounds in (
            ("mulligan_with_cf", range(cap + 1)),
            ("mulligan_no_cf", range(1, cap + 1)),
        ):
            assert set(frame.loc[frame.arm == arm, "round"]) == {f"R{r}" for r in rounds}
        assert (
            (frame.n > 0).all()
            and (frame.successes >= 0).all()
            and (frame.successes <= frame.n).all()
        )
        assert np.allclose(frame.success_rate, frame.successes / frame.n)
        cf_paths[task] = out / f"{task}_cf.csv"
        frame.to_csv(cf_paths[task], index=False)
    stats = {
        r["arm"]: r
        for r in json.loads(raw["sim/collection/square_narrow/r1_dataset_stats.json"].read_text())
    }
    # These are the policies that collected the R1 datasets, not the frozen-actor
    # DIVL heads evaluated in the sampler ablation.
    policies = {
        "baseline-uniform": PredecessorPolicy(
            task_label="Square-Narrow", stage="r0", arm="baseline_uniform"
        ),
        "mulligan": PredecessorPolicy(task_label="Square-Narrow", stage="r0", arm="sobol"),
    }
    sr = compute_predecessor_grid_sr(
        policies, raw["sim/collection/square_narrow/predecessor_grid.csv"]
    )
    records = []
    for arm in policies:
        k, n = stats[arm]["n_eps_with_intv"], stats[arm]["n_episodes"]
        mean, failure, seeds = sr[arm]
        assert 0 <= k <= n and n > 0 and 0 < failure < 1 and seeds == 5
        records.append(
            dict(
                arm=arm,
                intervention_episodes=k,
                episodes=n,
                predecessor_sr=mean,
                failure_divisor=failure,
                seeds=seeds,
            )
        )
    pd.DataFrame(records).to_csv(out / "sim_collection.csv", index=False)
    return groups, cf_paths, stats, sr


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
    print(CACHE / "data")


if __name__ == "__main__":
    main()
