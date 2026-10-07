"""Tests for mixed-arm-cardinality heldout ingest in
``mulligan.real.lifecycle.heldout_eval``.

An eval can retire an arm mid-session (manifest_eval
``--drop-fixed-policy``): early rounds carry all arms, later rounds carry only
the survivors. The ingest must (a) analyze the surviving arms across ALL rounds
even though early rounds carry an extra retired arm, (b) stay unchanged for
the uniform all-arms-present case, and (c) fail loudly with guidance if a
configured arm has only partial coverage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from mulligan.real.lifecycle import heldout_eval
from mulligan.real.lifecycle.heldout_eval import (
    HeldoutEvalConfig,
    build_paired_rounds,
    build_policy_summary,
    build_pairwise_summary,
    load_manifest,
    load_results,
)
from mulligan.real.eval.outcome_results import FrameOutcome
from mulligan.real.lifecycle.tasks import get_task_spec

# control=0, a7=1, clean=2, h33=3 (retired), i4=4
ARM_IDS = {"control": 0, "a7": 1, "clean": 2, "h33": 3, "i4": 4}
SURVIVORS = ("control", "a7", "clean", "i4")
# Per-round, per-arm success pattern (deterministic). 3 rounds.
SUCCESS = {
    "control": [True, False, True],
    "a7": [True, True, False],
    "clean": [False, True, True],
    "i4": [True, True, True],
    "h33": [False, False, False],  # retired arm: only ran rounds 1-2
}
MANIFEST = {
    idx: {
        "pen_x": 0.01 * idx,
        "pen_y": -0.02 * idx,
        "pen_yaw": 0.05 * idx,
        "holder_x": 0.1524,  # 6 in, on the marker_d2 holder grid
        "holder_y": 0.0,
        "sobol_stream_index": idx,
    }
    for idx in range(3)
}


def _rollout(arm: str, round_id: int) -> dict:
    idx = round_id - 1
    m = MANIFEST[idx]
    return {
        "policy_id": ARM_IDS[arm],
        "round": round_id,
        "manifest_idx": idx,
        "outcome": "success" if SUCCESS[arm][idx] else "failure",
        "num_steps": 100 + idx,
        "episode_index": round_id * 10 + ARM_IDS[arm],
        "pen_x": m["pen_x"],
        "pen_y": m["pen_y"],
        "pen_yaw": m["pen_yaw"],
    }


def _payload(arms_by_round: dict[int, tuple[str, ...]]) -> dict:
    rollouts = []
    arm_rounds: dict[str, list[int]] = {}
    for round_id, arms in sorted(arms_by_round.items()):
        for arm in arms:
            rollouts.append(_rollout(arm, round_id))
            arm_rounds.setdefault(arm, []).append(round_id)
    summary = []
    for arm, rounds in arm_rounds.items():
        successes = sum(SUCCESS[arm][r - 1] for r in rounds)
        summary.append(
            {
                "name": arm,
                "policy_id": ARM_IDS[arm],
                "num_rounds": len(rounds),
                "successes": successes,
                "failures": len(rounds) - successes,
                "model_id": f"model://{arm}",
            }
        )
    return {"rollouts": rollouts, "summary": summary}


def _cfg(tmp_path: Path, policy_names: tuple[str, ...]) -> HeldoutEvalConfig:
    return HeldoutEvalConfig(
        task="marker_d2",
        eval_repo="org/dummy",
        manifest_path=tmp_path / "manifest.json",
        expected_total_rounds=3,
        policy_names=policy_names,
        policy_labels={n: n for n in policy_names},
        policy_colors={n: "#000000" for n in policy_names},
        policy_prefixes={n: n for n in policy_names},
        data_dir=tmp_path,
        plot_path=tmp_path / "plot.svg",
        plot_title="t",
        plot_caption_prefix="c",
        bootstrap_seed=1,
        n_boot=200,
    )


def _write_manifest(path: Path) -> str:
    payload = {
        "task": get_task_spec("marker_d2").task_name,
        "states": [
            {
                "manifest_idx": idx,
                "pen_x": row["pen_x"],
                "pen_y": row["pen_y"],
                "pen_yaw": row["pen_yaw"],
                "holder_x": row["holder_x"],
                "holder_y": row["holder_y"],
                "sobol_stream_index": row["sobol_stream_index"],
            }
            for idx, row in MANIFEST.items()
        ],
    }
    text = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_load_manifest_checks_sha_pin(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_sha = _write_manifest(manifest_path)
    cfg = dataclasses.replace(
        _cfg(tmp_path, ("control", "a7")),
        expected_manifest_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="manifest sha256"):
        load_manifest(cfg, get_task_spec("marker_d2"))
    cfg = dataclasses.replace(cfg, expected_manifest_sha256=manifest_sha)
    assert set(load_manifest(cfg, get_task_spec("marker_d2"))) == set(MANIFEST)


def test_uniform_all_arms_present(tmp_path):
    # Backward-compat: 4 arms in all 3 rounds → 3 paired rounds, n=3 each.
    spec = get_task_spec("marker_d2")
    payload = _payload({1: SURVIVORS, 2: SURVIVORS, 3: SURVIVORS})
    cfg = _cfg(tmp_path, SURVIVORS)
    paired = build_paired_rounds(cfg, spec, payload, MANIFEST)
    assert len(paired) == 3
    policy = build_policy_summary(cfg, payload, paired)
    assert set(policy["episodes"]) == {3}
    assert dict(zip(policy["policy_name"], policy["successes"])) == {
        "control": 2,
        "a7": 2,
        "clean": 2,
        "i4": 3,
    }
    pairwise = build_pairwise_summary(cfg, paired)
    assert len(pairwise) == len(SURVIVORS) - 1  # each vs baseline


def test_survivors_analyzed_across_all_rounds_with_extra_retired_arm(tmp_path):
    # Intent (a): h33 retired after round 2. Config lists the 4 survivors; rounds
    # 1-2 carry an EXTRA h33 rollout that must be ignored, and ALL 3 rounds count.
    spec = get_task_spec("marker_d2")
    payload = _payload({1: (*SURVIVORS, "h33"), 2: (*SURVIVORS, "h33"), 3: SURVIVORS})
    cfg = _cfg(tmp_path, SURVIVORS)
    paired = build_paired_rounds(cfg, spec, payload, MANIFEST)
    assert len(paired) == 3  # all rounds kept despite extra h33 in 1-2
    assert "h33_success" not in paired.columns  # retired arm ignored
    policy = build_policy_summary(cfg, payload, paired)
    assert set(policy["episodes"]) == {3}
    assert dict(zip(policy["policy_name"], policy["successes"])) == {
        "control": 2,
        "a7": 2,
        "clean": 2,
        "i4": 3,
    }


def test_partial_coverage_configured_arm_fails_loudly(tmp_path):
    # Intent (b) guard: configuring the retired arm (h33, only rounds 1-2) must
    # raise with actionable guidance, not silently mis-count.
    spec = get_task_spec("marker_d2")
    payload = _payload({1: (*SURVIVORS, "h33"), 2: (*SURVIVORS, "h33"), 3: SURVIVORS})
    cfg = _cfg(tmp_path, (*SURVIVORS, "h33"))
    # issubset keeps only rounds 1-2 (round 3 lacks h33) → 2 paired rounds.
    paired = build_paired_rounds(cfg, spec, payload, MANIFEST)
    assert len(paired) == 2
    with pytest.raises(RuntimeError, match="partial coverage|share full coverage"):
        build_policy_summary(cfg, payload, paired)


def test_missing_configured_arm_in_a_round_is_skipped(tmp_path):
    # A round missing a configured survivor is incomplete and dropped.
    spec = get_task_spec("marker_d2")
    payload = _payload({1: SURVIVORS, 2: ("control", "a7", "clean"), 3: SURVIVORS})
    cfg = _cfg(tmp_path, SURVIVORS)
    paired = build_paired_rounds(cfg, spec, payload, MANIFEST)
    assert set(paired["round"]) == {1, 3}  # round 2 (no i4) dropped


def test_max_rounds_analyzes_the_round_prefix(tmp_path):
    # max_rounds=2 on a 3-round eval: the summary counts all 3 rounds, the analysis
    # uses rounds 1-2 only.
    spec = get_task_spec("marker_d2")
    payload = _payload({1: SURVIVORS, 2: SURVIVORS, 3: SURVIVORS})
    cfg = dataclasses.replace(_cfg(tmp_path, SURVIVORS), max_rounds=2)
    paired = build_paired_rounds(cfg, spec, payload, MANIFEST)
    assert set(paired["round"]) == {1, 2}
    policy = build_policy_summary(cfg, payload, paired)
    assert set(policy["episodes"]) == {2}
    assert dict(zip(policy["policy_name"], policy["successes"])) == {
        name: sum(SUCCESS[name][:2]) for name in SURVIVORS
    }
    build_pairwise_summary(cfg, paired)

    payload["summary"][0]["successes"] += 1
    with pytest.raises(RuntimeError, match="disagree"):
        build_policy_summary(cfg, payload, paired)


def test_lying_summary_num_rounds_fails_loudly(tmp_path):
    # A drifting/lying summary that PASSES the coverage guard (num_rounds == paired
    # n) but disagrees with the raw rollouts must be caught, not silently mixed.
    # Round 3 has no i4 → paired = {1, 2}. control ran all 3 rounds, but its summary
    # lies with num_rounds=2 (== paired n). The raw-vs-summary guard must fire.
    spec = get_task_spec("marker_d2")
    payload = _payload({1: SURVIVORS, 2: SURVIVORS, 3: ("control", "a7", "clean")})
    cfg = _cfg(tmp_path, SURVIVORS)
    paired = build_paired_rounds(cfg, spec, payload, MANIFEST)
    assert set(paired["round"]) == {1, 2}
    for row in payload["summary"]:
        if row["name"] in ("control", "a7", "clean"):
            row["num_rounds"] = 2  # lie: they actually ran 3 rounds
            row["successes"] = sum(SUCCESS[row["name"]][:2])
    with pytest.raises(RuntimeError, match="raw rollout|summary and rollouts disagree"):
        build_policy_summary(cfg, payload, paired)


def test_load_results_preserves_eval_time_backup_on_canonical_rerun(tmp_path, monkeypatch):
    cfg = dataclasses.replace(
        _cfg(tmp_path, ("control", "a7")),
        outcome_overrides_filename=".outcome_edit_progress.json",
    )
    spec = get_task_spec("marker_d2")
    raw_payload = {
        "args": {
            "environment": spec.task_name,
            "hf_repo_id": cfg.eval_repo,
            "random_arena_slots": 0,
            "num_action_samples": None,
        },
        "round_plans": [{}, {}, {}],
        "summary": [
            {
                "name": "control",
                "policy_id": 0,
                "num_rounds": 1,
                "successes": 0,
                "failures": 1,
                "success_rate": 0.0,
            },
            {
                "name": "a7",
                "policy_id": 1,
                "num_rounds": 1,
                "successes": 0,
                "failures": 1,
                "success_rate": 0.0,
            },
        ],
        "rollouts": [
            {
                "round": 1,
                "policy_id": 0,
                "episode_index": 0,
                "outcome": "failure",
                "num_steps": 2,
            },
            {
                "round": 1,
                "policy_id": 1,
                "episode_index": 1,
                "outcome": "timeout",
                "num_steps": 4,
            },
        ],
    }
    edit_record = {
        "changed_episodes": {
            "1": {
                "new_outcome": "success",
                "outcome_frame": 2,
                "soft_truncate": False,
            }
        }
    }
    frame_outcomes = {
        0: FrameOutcome("failure", 2),
        1: FrameOutcome("success", 3),
    }
    fetched_payloads = [raw_payload]

    def fetch_payload(_repo):
        return json.loads(json.dumps(fetched_payloads[-1]))

    monkeypatch.setattr(heldout_eval, "fetch_results_payload", fetch_payload)
    monkeypatch.setattr(
        heldout_eval, "load_outcome_edit_record", lambda *args, **kwargs: edit_record
    )
    monkeypatch.setattr(
        heldout_eval, "load_frame_outcomes_from_hf", lambda *args, **kwargs: frame_outcomes
    )

    first = load_results(cfg, spec)
    backup_path = cfg.data_dir / "results_eval_time.json"
    results_path = cfg.data_dir / "results.json"
    assert json.loads(backup_path.read_text())["rollouts"][1]["outcome"] == "timeout"
    assert json.loads(results_path.read_text())["rollouts"][1]["outcome"] == "success"
    assert first["summary"][1]["successes"] == 1

    fetched_payloads.append(json.loads(results_path.read_text()))
    load_results(cfg, spec)
    assert json.loads(backup_path.read_text())["rollouts"][1]["outcome"] == "timeout"


def _live_mark_payload(spec) -> dict:
    return {
        "args": {
            "environment": spec.task_name,
            "hf_repo_id": "org/dummy",
            "random_arena_slots": 0,
            "num_action_samples": None,
        },
        "round_plans": [{}, {}, {}],
        "summary": [
            {"name": "control", "policy_id": 0, "num_rounds": 1, "successes": 1, "failures": 0},
            {"name": "a7", "policy_id": 1, "num_rounds": 1, "successes": 1, "failures": 0},
        ],
        "rollouts": [
            # success with the live first-clip mark: the normal case
            {
                "round": 1,
                "policy_id": 0,
                "episode_index": 0,
                "outcome": "success",
                "num_steps": 9,
                "subtask_frames": [4],
            },
            # success where the operator missed the mark key
            {
                "round": 1,
                "policy_id": 1,
                "episode_index": 1,
                "outcome": "success",
                "num_steps": 9,
                "subtask_frames": [],
            },
        ],
    }


def test_live_marks_success_without_mark_needs_explicit_acceptance(tmp_path, monkeypatch):
    spec = get_task_spec("routing_d2")
    assert spec.num_subtask_marks == 1
    base = dataclasses.replace(
        _cfg(tmp_path, ("control", "a7")),
        task="routing_d2",
        subtask_scoring=True,
        subtask_marks_from_live_records=True,
    )
    payload = _live_mark_payload(spec)
    monkeypatch.setattr(
        heldout_eval, "fetch_results_payload", lambda _repo: json.loads(json.dumps(payload))
    )
    frame_outcomes = {0: FrameOutcome("success", 9), 1: FrameOutcome("success", 9)}
    monkeypatch.setattr(
        heldout_eval, "load_frame_outcomes_from_hf", lambda *args, **kwargs: frame_outcomes
    )
    with pytest.raises(RuntimeError, match=r"SUCCESS episodes with 0 mid-episode marks .*\[1\]"):
        load_results(base, spec)
    # Listing the wrong episode fails just as loudly.
    with pytest.raises(RuntimeError, match="live_success_zero_mark_episodes"):
        load_results(dataclasses.replace(base, live_success_zero_mark_episodes=(0,)), spec)
    accepted = load_results(dataclasses.replace(base, live_success_zero_mark_episodes=(1,)), spec)
    assert accepted["_subtask_mark_counts"] == {0: 1, 1: 1}
    assert accepted["_live_success_zero_mark_episodes"] == [1]


def test_live_zero_mark_acceptance_requires_live_marks(tmp_path):
    with pytest.raises(ValueError, match="requires subtask_marks_from_live_records"):
        dataclasses.replace(
            _cfg(tmp_path, ("control", "a7")),
            subtask_scoring=True,
            outcome_overrides_filename=".outcome_edit_progress.json",
            live_success_zero_mark_episodes=(1,),
        )
