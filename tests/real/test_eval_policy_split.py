"""Policy-view selection and lineage, using a real local LeRobot dataset."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from mulligan.real.eval import split_policies as split
from tests.unit.sim_dataset import build_sim_dataset


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    build_sim_dataset(
        root,
        episode_lengths=(2, 3, 2),
        extra_features={"policy_id": {"dtype": "int64", "shape": (1,), "names": None}},
        overrides=lambda ep, t, n: {
            "policy_id": np.array([ep % 2], np.int64),
            "task": "marker_d2",
        },
    )
    results = {
        "summary": [
            {"policy_id": pid, "model_id": "hf://org/shared", "name": f"arm{pid}"}
            for pid in range(2)
        ],
        "rollouts": [
            {
                "episode_index": ep,
                "policy_id": ep % 2,
                "model_id": "hf://org/shared",
                "round": ep,
                "outcome": "failure" if ep == 0 else "success",
                "num_steps": n,
                "manifest_idx": 10 + ep,
            }
            for ep, n in enumerate((2, 3, 2))
        ],
    }
    (root / "results.json").write_text(json.dumps(results))
    return root, results


def run_split(monkeypatch, root, out):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split",
            "--source-repo",
            "local/source",
            "--source-root",
            str(root),
            "--output-root",
            str(out),
            "--target",
            "arm0=local/zero",
            "--target",
            "1=local/one",
        ],
    )
    return split.main()


def test_shared_checkpoint_arms_split_by_name_and_id(source, tmp_path, monkeypatch):
    root, _ = source
    out = tmp_path / "out"
    assert run_split(monkeypatch, root, out) == 0
    for name, indices in (("zero", [0, 2]), ("one", [1])):
        child = out / name
        rows = [
            json.loads(line)
            for line in (child / "meta/real_eval_policy_split_manifest.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["source_episode_index"] for r in rows] == indices
        assert [r["episode_index"] for r in rows] == list(range(len(indices)))
        assert [r["manifest_idx"] for r in rows] == [10 + i for i in indices]
        assert all(r["source_current_success"] for r in rows)
        frames = pd.concat(pd.read_parquet(p) for p in (child / "data").glob("*/*.parquet"))
        assert sorted(frames.episode_index.unique()) == list(range(len(indices)))
        assert {int(np.asarray(x).item()) for x in frames.policy_id} == {indices[0] % 2}
        lineage = json.loads((child / "meta/dataset_lineage.json").read_text())
        assert lineage["parent_repo_id"] == "local/source"
        assert lineage["producer_model_ids"] == ["hf://org/shared"]
    first = json.loads(
        (out / "zero/meta/real_eval_policy_split_manifest.jsonl").read_text().splitlines()[0]
    )
    assert first["source_result_outcome"] == "failure" and first["source_current_success"]


@pytest.mark.parametrize("problem", ["duplicate", "missing", "wrong_policy"])
def test_inconsistent_results_fail_before_writing(source, tmp_path, monkeypatch, problem):
    root, results = source
    if problem == "duplicate":
        results["rollouts"].append(results["rollouts"][0])
    elif problem == "missing":
        results["rollouts"].pop(0)
    else:
        results["rollouts"][0]["policy_id"] = 1
    (root / "results.json").write_text(json.dumps(results))
    out = tmp_path / "out"
    with pytest.raises(
        SystemExit, match="duplicate episode|missing saved episode|dataset policy_id"
    ):
        run_split(monkeypatch, root, out)
    assert not out.exists()


def test_shared_model_selector_is_unavailable_but_unique_selectors_survive():
    entries = split._policy_entries(
        {
            "summary": [
                {"policy_id": 0, "model_id": "shared", "name": "first"},
                {"policy_id": 1, "model_id": "shared", "name": "second"},
            ]
        }
    )
    assert "shared" not in entries
    assert entries["first"]["policy_id"] == 0
    assert entries["1"]["name"] == "second"


def test_duplicate_policy_ids_are_rejected():
    with pytest.raises(SystemExit, match="Duplicate policy_id=0"):
        split._policy_entries(
            {
                "summary": [
                    {"policy_id": 0, "model_id": "first"},
                    {"policy_id": 0, "model_id": "second"},
                ]
            }
        )
