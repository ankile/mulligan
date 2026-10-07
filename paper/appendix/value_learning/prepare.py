"""Extract the locked value-learning studies from their pinned producer outputs.

Run with python -m paper.appendix.value_learning.prepare. ``sources.json`` pins every
producer file by SHA-256 and size; ``dataset.json`` pins the five derived tables, and the
extract must reproduce them byte for byte. No training, inference, or live lookup happens here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.appendix.artifacts import cache_dir, fetch, sha256

HERE = Path(__file__).resolve().parent
CACHE = cache_dir("value_learning")
RAW = CACHE / "raw/real/value_learning"
TABLES = "real/value_learning/tables/"
BLOCK = "r4_baseline_no_cf_heldout_s2026070604"
STEPS = (125000, 150000, 200000, 225000, 250000, 300000, 400000, 450000, 475000)
POLICIES = tuple(
    f"{arm}_r{r}"
    for arm, rounds in (("baseline", range(6)), ("dp", range(6)), ("iql", range(2, 6)))
    for r in rounds
)


def extract() -> dict:
    lock = json.loads((HERE / "sources.json").read_text())
    for row in lock["sources"]:
        fetch(CACHE.name, row)
    policies, seeds, checkpoints, probes, episodes = [], [], [], [], []
    for key in POLICIES:
        d = json.loads((RAW / "eval" / key / "summary.json").read_text())
        assert d["gamma"] == 0.995 and d["bootstrap_timeouts"]
        assert d["checkpoint_selection"] == "final" and d["fqe_steps"] == 40000
        assert d["initial_states"] == 150 and d["initial_unique_manifests"] == 50
        assert d["transition_protocol"] == "primary_max_common_pure_mulligan_pre_r5_eval"
        t = d["targets"][key]
        assert t["robot_sr"] == t["robot_successes"] / t["robot_n"]
        assert sorted(r["seed"] for r in t["runs"]) == [1, 2, 3]
        np.testing.assert_allclose(
            np.mean([r["mean"] for r in t["runs"]]), t["fqe_seed_mean"], atol=1e-7
        )
        policies.append(
            dict(
                policy=key,
                family=key.split("_r")[0],
                round=int(key.split("_r")[1]),
                robot_successes=t["robot_successes"],
                robot_n=t["robot_n"],
                policy_bank_sha256=d["policy_bank_sha256"],
                initial_values_sha256=d["initial_values_sha256"],
            )
        )
        for run in t["runs"]:
            assert run["selected_step"] == 40000
            seeds.append(dict(policy=key, seed=run["seed"], fqe_value=run["mean"]))
    reference_probe = None
    reference_episodes = None
    for step in STEPS:
        folder = RAW / "training" / str(step)
        summary = json.loads((folder / "summary.json").read_text())
        probe = pd.read_csv(folder / "candidate_probe.csv")
        episode = pd.read_csv(folder / "per_episode_features.csv")
        assert len(probe) == 50 and probe["block"].eq(BLOCK).all()
        assert probe["arm"].eq("baseline_no_cf").all()
        assert probe["num_candidates"].eq(32).all()
        assert probe["manifest_idx"].nunique() == 50 and probe["success"].sum() == 23
        episode = episode[episode["block"].eq(BLOCK)].sort_values("manifest_idx")
        probe = probe.sort_values("manifest_idx")
        assert len(episode) == 50 and not episode["in_training"].any()
        identity = ["manifest_idx", "episode_index", "success"]
        pd.testing.assert_frame_equal(
            probe[identity].reset_index(drop=True), episode[identity].reset_index(drop=True)
        )
        if reference_probe is None:
            reference_probe = probe[identity].reset_index(drop=True)
            reference_episodes = episode[identity + ["repo_id"]].reset_index(drop=True)
        else:
            pd.testing.assert_frame_equal(reference_probe, probe[identity].reset_index(drop=True))
            pd.testing.assert_frame_equal(
                reference_episodes, episode[identity + ["repo_id"]].reset_index(drop=True)
            )
        assert np.isfinite(probe["q_cand_max"]).all() and np.isfinite(episode["v_s0"]).all()
        p = probe[identity + ["q_cand_max", "num_candidates"]].copy()
        e = episode[identity + ["repo_id", "v_s0"]].copy()
        p.insert(0, "step", step)
        e.insert(0, "step", step)
        probes.append(p)
        episodes.append(e)
        checkpoints.append(
            dict(
                step=step,
                critic_artifact=summary["iql_artifact"],
                recorded_auroc=summary["candidate_probe"]["auroc_q_cand_max"],
            )
        )
    tables = {
        "fqe_policies.csv": pd.DataFrame(policies),
        "fqe_seeds.csv": pd.DataFrame(seeds),
        "training_probes.csv": pd.concat(probes, ignore_index=True),
        "training_episodes.csv": pd.concat(episodes, ignore_index=True),
        "training_checkpoints.csv": pd.DataFrame(checkpoints),
    }
    (CACHE / "data").mkdir(parents=True, exist_ok=True)
    dataset = json.loads((HERE / "dataset.json").read_text())
    if dataset["sources_sha256"] != sha256(HERE / "sources.json"):
        raise RuntimeError("dataset.json was not derived from the current sources.json")
    expected = {row["path"]: row for row in dataset["files"]}
    for name, frame in tables.items():
        path = CACHE / "data" / name
        frame.to_csv(path, index=False, float_format="%.17g")
        row = expected[TABLES + name]
        if len(frame) != row["rows"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"{path}: extract differs from the dataset.json lock")
    print("Extracted", {name: len(frame) for name, frame in tables.items()})
    return dataset


def main() -> None:
    extract()


if __name__ == "__main__":
    main()
