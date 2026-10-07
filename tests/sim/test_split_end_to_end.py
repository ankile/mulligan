"""Episode -> split assignment end to end on a tiny local collection.

The real collector (headless, scripted operator, protocol quota) writes a LeRobot
dataset and its ledger; ``split_protocol_quota`` and ``split_blind`` then split that
dataset into per-arm datasets, and each output episode is traced back to its
manifest row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mulligan.data import split_blind, split_protocol_quota
from mulligan.sim.collect import dagger
from tests.sim.test_dagger_operator import (
    STATES,
    FakeActor,
    ScriptedOperator,
    _args,
    _read_frames,
    _write_states,
)

KEYS = ["nut_x", "nut_y", "nut_yaw"]
# Manifest rows: (arm, policy label).
ROWS = [("baseline_uniform", "baseline"), ("mulligan", "ours"), ("mulligan", "ours")]


@pytest.fixture(scope="module")
def collection(tmp_path_factory) -> dict:
    """A blinded DAgger collection with the protocol quota; one failure is also saved."""
    tmp = tmp_path_factory.mktemp("collection")
    manifest = tmp / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "square_narrow",
                "keys": KEYS,
                "match_tolerance": 1e-3,
                "states": [
                    {**state, "sources": [arm], "source": arm, "policy_source": label}
                    for state, (arm, label) in zip(STATES, ROWS)
                ],
            }
        )
    )
    states_file = tmp / "states.json"
    _write_states(states_file, STATES)
    ledger = tmp / "ledger.jsonl"
    args = _args(
        tmp,
        "--sampler=list",
        f"--initial-states-file={states_file}",
        "--no-sampler-shuffle",
        f"--adaptive-protocol-quota-manifest={manifest}",
        "--adaptive-protocol-quota-targets=no_cf=1,with_cf=2",
        "--adaptive-protocol-quota-arms=no_cf=baseline_uniform,mulligan;with_cf=mulligan",
        f"--adaptive-protocol-quota-ledger={ledger}",
    )
    outcomes = iter(["0"] + ["1"] * 20)  # the first episode fails and is redrawn

    def script(start):
        key = next(outcomes)
        if start.human_first:
            return [("human", 2, "h"), ("policy", 1, "h"), ("human", 1, key)]
        return [("policy", 2, "h"), ("human", 2, key)]

    log: list[str] = []
    actors = {label: FakeActor(label, log) for label in ("ours", "baseline")}
    operator = ScriptedOperator(
        scripts=[script] * 20, choices=[lambda options: "c" if "c" in options else "n"] * 20
    )
    collector = dagger.DaggerCollector(
        args,
        operator=operator,
        actor=actors["ours"],
        env_name="NutAssemblySquare",
        robot_name="Panda",
        router=dagger.PolicyRouter(manifest, actors),
    )
    collector.setup()
    collector.run()
    assert collector.protocol_quota.is_complete()
    return {
        "manifest": manifest,
        "ledger": ledger,
        "source_root": tmp / "data" / "dagger-test",
        "rows": [json.loads(line) for line in ledger.read_text().splitlines()],
        "out": tmp / "splits",
    }


def _manifest_idx(env_state) -> int:
    start = np.asarray(env_state, dtype=float)
    nut = (start[7], start[8])
    for idx, state in enumerate(STATES):
        if abs(state["nut_x"] - nut[0]) < 1e-3 and abs(state["nut_y"] - nut[1]) < 1e-3:
            return idx
    raise AssertionError(f"episode start {nut} is not a manifest row")


def _episodes(dataset_dir: Path) -> list[dict]:
    """(first-frame manifest row, frame count, success) per episode of an output dataset."""
    frames = _read_frames(dataset_dir)
    out = []
    for _, rows in frames.groupby("episode_index", sort=True):
        out.append(
            {
                "manifest_idx": _manifest_idx(rows["observation.environment_state"].iloc[0]),
                "frames": len(rows),
                "success": int(np.asarray(rows["success"].iloc[-1]).reshape(-1)[0]),
            }
        )
    return out


def test_protocol_quota_split_assigns_episodes_by_the_ledger(collection):
    rows, out = collection["rows"], collection["out"] / "protocol"
    targets = {
        "no_cf.baseline_uniform": "local/no-cf-baseline",
        "no_cf.mulligan": "local/no-cf-mulligan",
        "with_cf.mulligan": "local/with-cf-mulligan",
    }
    assert (
        split_protocol_quota.main(
            [
                "--source-repo=local/dagger-test",
                f"--source-root={collection['source_root']}",
                f"--manifest={collection['manifest']}",
                f"--ledger={collection['ledger']}",
                "--expected-per-protocol=no_cf=1,with_cf=2",
                "--expected-protocol-arms=no_cf=baseline_uniform,mulligan;with_cf=mulligan",
                *[f"--target={key}={repo}" for key, repo in targets.items()],
                f"--output-root={out}",
                "--drop-visual-features",
            ]
        )
        == 0
    )

    source = _episodes(collection["source_root"])
    for key, repo in targets.items():
        protocol, arm = key.split(".")
        credited = [
            row for row in rows if arm in row.get("credited_protocol_arms", {}).get(protocol, [])
        ]
        assert credited, key
        expected = [source[row["episode_index"]] for row in credited]
        got = _episodes(out / repo.split("/")[1])
        # Same episodes (start row, length, outcome), in ledger order.
        assert got == expected, key
        for row in credited:
            assert row["success"]
            assert ROWS[row["manifest_idx"]][0] == arm
            if protocol == "no_cf":
                assert not row["is_counterfactual"]
    # The failed first episode is in the source but in no split.
    assert not rows[0]["success"] and not rows[0].get("credited_protocol_arms")


def test_blind_split_assigns_successful_unique_starts_by_manifest_source(collection, capsys):
    out = collection["out"] / "blind"
    source = _episodes(collection["source_root"])
    # split_blind keeps the first successful episode of each manifest row.
    kept: dict[int, dict] = {}
    for episode in source:
        if episode["success"] and episode["manifest_idx"] not in kept:
            kept[episode["manifest_idx"]] = episode
    # In source order (kept is filled in source order).
    expected = {
        arm: [ep for idx, ep in kept.items() if ROWS[idx][0] == arm]
        for arm in ("baseline_uniform", "mulligan")
    }
    assert all(expected.values())
    duplicates = sum(ep["success"] for ep in source) - len(kept)
    assert duplicates > 0  # counterfactual replays share a start

    assert (
        split_blind.main(
            [
                "--source-repo=local/dagger-test",
                f"--source-root={collection['source_root']}",
                f"--manifest={collection['manifest']}",
                "--target=baseline_uniform=local/blind-baseline",
                "--target=mulligan=local/blind-mulligan",
                "--expected-per-source="
                + ",".join(f"{arm}={len(eps)}" for arm, eps in expected.items()),
                f"--output-root={out}",
                "--drop-visual-features",
            ]
        )
        == 0
    )
    assert f"dropping {duplicates} duplicate manifest matches" in capsys.readouterr().out
    assert _episodes(out / "blind-baseline") == expected["baseline_uniform"]
    assert _episodes(out / "blind-mulligan") == expected["mulligan"]


def test_blind_split_refuses_a_start_outside_the_manifest_tolerance(collection, tmp_path):
    manifest = json.loads(collection["manifest"].read_text())
    for state in manifest["states"]:
        state["nut_x"] += 0.1
    shifted = tmp_path / "manifest.json"
    shifted.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="did not match quota manifest within tolerance"):
        split_blind.main(
            [
                "--source-repo=local/dagger-test",
                f"--source-root={collection['source_root']}",
                f"--manifest={shifted}",
                "--target=baseline_uniform=local/blind-baseline",
                "--target=mulligan=local/blind-mulligan",
                f"--output-root={tmp_path / 'out'}",
                "--drop-visual-features",
            ]
        )
