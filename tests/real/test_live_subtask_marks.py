"""Live operator sub-goal marks ('g' / numpad '3') during real evals.

Covers the whole minimal path: key handling -> reward=1.0 spike written by
``finalize_episode_data`` -> results.json provenance -> graded results table ->
ingest validator accepting the spike once a review record lists it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulligan.real.collect.dataset_features import (
    ACTION_INFO_KEYS,
    finalize_episode_data,
    init_supplementary_lists,
)
from mulligan.real.eval.common import (
    PolicyEntry,
    RolloutRecord,
    episode_score,
    load_previous_results,
    num_subtask_marks_for_task,
    print_results,
    rollout_outcome_line,
    save_results_file,
)
from mulligan.real.eval.outcome_results import _detect_frame_outcome
from mulligan.real.collect.rollout import record_subtask_mark
from mulligan.real.operator_ui.keys import (
    OPERATOR_KEY_ALIASES,
    apply_operator_key_alias,
    key_label,
)


# --------------------------------------------------------------------------- #
# Key handling
# --------------------------------------------------------------------------- #


def test_numpad_3_aliases_to_subgoal_key() -> None:
    assert OPERATOR_KEY_ALIASES["3"] == "g"
    assert apply_operator_key_alias("3") == "g"
    assert key_label("g") == "'g'/numpad'3'"
    # Outcome digits stay native (no alias) so the numpad bottom row is 1/3/9 + 0.
    assert apply_operator_key_alias("1") == "1"
    assert apply_operator_key_alias("9") == "9"


def test_record_subtask_mark_lands_on_most_recent_frame() -> None:
    frames: list[int] = []
    assert record_subtask_mark(frames, step=42, subtask_marks=1) is True
    assert frames == [41]


def test_record_subtask_mark_ignores_presses_beyond_cap() -> None:
    frames = [10]
    assert record_subtask_mark(frames, step=50, subtask_marks=1) is False
    assert frames == [10]


def test_record_subtask_mark_ignores_before_first_frame_and_for_markless_tasks() -> None:
    frames: list[int] = []
    assert record_subtask_mark(frames, step=0, subtask_marks=1) is False
    assert record_subtask_mark(frames, step=7, subtask_marks=0) is False
    assert frames == []


# --------------------------------------------------------------------------- #
# Spike written at save time
# --------------------------------------------------------------------------- #


def _episode_data(n_frames: int) -> dict:
    data = {
        "observations": [np.zeros(7, dtype=np.float32) for _ in range(n_frames + 1)],
        "joint_positions": [np.zeros(7, dtype=np.float32) for _ in range(n_frames + 1)],
        "actions": [np.zeros(7, dtype=np.float32) for _ in range(n_frames)],
        "rewards": [0.0] * n_frames,
        "dones": [0] * n_frames,
    }
    init_supplementary_lists(data)
    for key in ACTION_INFO_KEYS:
        data[key] = [np.zeros(1, dtype=np.float32) for _ in range(n_frames)]
    data["joint_velocities"] = [np.zeros(7, dtype=np.float32) for _ in range(n_frames)]
    data["cartesian_velocities"] = [np.zeros(6, dtype=np.float32) for _ in range(n_frames)]
    return data


_OBS = {"robot_state": {"joint_velocities": [0.0] * 7}}


def test_finalize_writes_spike_only_at_mark_and_keeps_terminal_labels() -> None:
    data = _episode_data(6)
    finalize_episode_data(data, _OBS, is_success=True, is_terminal=True, subtask_frames=[2])
    # 6 valid frames + 1 padded terminal row
    assert data["rewards"] == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert data["dones"] == [0, 0, 0, 0, 0, 1, 1]
    assert data["steps_to_go"] == [6, 5, 4, 3, 2, 1, 0]


def test_finalize_failure_with_mark_scores_one() -> None:
    data = _episode_data(4)
    finalize_episode_data(data, _OBS, is_success=False, is_terminal=True, subtask_frames=[1])
    assert data["rewards"] == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert data["dones"] == [0, 0, 0, 1, 1]


def test_finalize_timeout_mark_on_last_valid_frame_does_not_leak_into_padding() -> None:
    # Mirrors the outcome editor: a timeout may carry its mark on the final valid
    # frame (no terminal transition to conflict with); the padded row stays 0.
    data = _episode_data(4)
    finalize_episode_data(data, _OBS, is_success=False, is_terminal=False, subtask_frames=[3])
    assert data["rewards"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert data["dones"] == [0, 0, 0, 0, 0]


def test_finalize_without_marks_is_unchanged() -> None:
    data = _episode_data(3)
    finalize_episode_data(data, _OBS, is_success=True, is_terminal=True)
    assert data["rewards"] == [0.0, 0.0, 1.0, 1.0]


def test_finalize_rejects_mark_on_terminal_outcome_frame() -> None:
    data = _episode_data(4)
    with pytest.raises(ValueError, match="coincides with the outcome frame"):
        finalize_episode_data(data, _OBS, is_success=True, is_terminal=True, subtask_frames=[3])


def test_finalize_rejects_mark_outside_episode() -> None:
    data = _episode_data(4)
    with pytest.raises(ValueError, match="outside the episode"):
        finalize_episode_data(data, _OBS, is_success=True, is_terminal=True, subtask_frames=[4])


def test_live_spike_is_the_shape_the_ingest_validator_expects() -> None:
    # Exactly what the outcome-review apply writes: reward=1.0, done=0 at the mark; the
    # validator accepts it once a review record lists the frame, and still fails
    # loud on an unreviewed live spike (skip is not allowed for live-marked episodes).
    data = _episode_data(6)
    finalize_episode_data(data, _OBS, is_success=False, is_terminal=True, subtask_frames=[2])
    n = len(data["rewards"])
    ep = pd.DataFrame(
        {
            "episode_index": [0] * n,
            "frame_index": list(range(n)),
            "reward": data["rewards"],
            "done": data["dones"],
            "is_valid": [1] * (n - 1) + [0],
        }
    )
    outcome = _detect_frame_outcome(ep, subtask_frames=(2,))
    assert outcome.outcome == "failure"
    assert outcome.expected_num_steps == 6
    with pytest.raises(RuntimeError, match="nonzero reward before outcome frame"):
        _detect_frame_outcome(ep)


# --------------------------------------------------------------------------- #
# Graded score + results.json provenance
# --------------------------------------------------------------------------- #


def test_task_spec_lookup() -> None:
    assert num_subtask_marks_for_task("routing_d2") == 1
    assert num_subtask_marks_for_task("not-a-registered-task") == 0


def test_episode_score_is_marks_plus_success() -> None:
    assert episode_score("success", (120,)) == 2
    assert episode_score("failure", (120,)) == 1
    assert episode_score("timeout", ()) == 0
    assert episode_score("success", ()) == 1


def test_rollout_outcome_line_appends_score_only_for_graded_tasks() -> None:
    assert rollout_outcome_line("A", "failure", 300, [], 0) == "Policy A: FAILURE (300 steps)"
    assert (
        rollout_outcome_line("A", "failure", 300, [120], 1)
        == "Policy A: FAILURE (300 steps) | score 1/2 (sub-goal marks at frames [120])"
    )


def _record(policy_id: int, outcome: str, frames: tuple[int, ...], episode_index: int):
    return RolloutRecord(
        round_num=episode_index // 2,
        policy_id=policy_id,
        model_id=f"hf://org/model-{policy_id}",
        anonymous_label="AB"[policy_id],
        outcome=outcome,
        num_steps=100,
        episode_index=episode_index,
        subtask_frames=frames,
    )


def test_print_results_adds_graded_score_column(capsys) -> None:
    records = [
        _record(0, "success", (40,), 0),
        _record(1, "failure", (55,), 1),
        _record(0, "timeout", (), 2),
        _record(1, "success", (30,), 3),
    ]
    policies = [
        PolicyEntry(model_id="hf://org/model-0", policy_id=0, policy=None, results=[True, False]),
        PolicyEntry(model_id="hf://org/model-1", policy_id=1, policy=None, results=[False, True]),
    ]
    print_results(policies, ["mulligan", "baseline"], rollout_records=records, num_subtask_marks=1)
    out = capsys.readouterr().out
    assert "Score (max 2/ep)" in out
    assert "2/4 (1.00/2 per ep)" in out  # ours: 2 + 0
    assert "3/4 (1.50/2 per ep)" in out  # baseline: 1 + 2
    assert "1/2 (50.0%)" in out


def test_print_results_binary_only_without_marks(capsys) -> None:
    policies = [PolicyEntry(model_id="hf://org/m", policy_id=0, policy=None, results=[True])]
    print_results(policies)
    out = capsys.readouterr().out
    assert "Score" not in out
    assert "1/1 (100.0%)" in out


def test_print_results_graded_requires_matching_record_count() -> None:
    policies = [PolicyEntry(model_id="hf://org/m", policy_id=0, policy=None, results=[True, True])]
    with pytest.raises(RuntimeError, match="rollout records"):
        print_results(
            policies,
            rollout_records=[_record(0, "success", (1,), 0)],
            num_subtask_marks=1,
        )


def test_results_file_round_trips_subtask_frames(tmp_path: Path) -> None:
    policies = [PolicyEntry(model_id="hf://org/m", policy_id=0, policy=None, results=[False])]
    records = [_record(0, "failure", (77,), 0)]
    out = tmp_path / "results.json"
    save_results_file(out, policies, records, argparse.Namespace(), "ds")
    payload = json.loads(out.read_text())
    assert payload["rollouts"][0]["subtask_frames"] == [77]
    loaded, _policy_map = load_previous_results(out)
    assert loaded[0].subtask_frames == (77,)


def test_results_file_without_subtask_key_loads_as_no_marks(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    out.write_text(
        json.dumps(
            {
                "summary": [],
                "rollouts": [
                    {
                        "round": 0,
                        "policy_id": 0,
                        "model_id": "hf://org/m",
                        "outcome": "success",
                        "num_steps": 10,
                        "episode_index": 0,
                    }
                ],
            }
        )
    )
    loaded, _policy_map = load_previous_results(out)
    assert loaded[0].subtask_frames == ()
