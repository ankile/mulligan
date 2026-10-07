from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from mulligan.real.eval.outcome_results import (
    _detect_frame_outcome,
    apply_outcome_edit_record,
    canonicalize_results_payload,
    canonicalize_results_file,
    live_subtask_frames_from_results,
    load_frame_outcomes_from_root,
    subtask_frames_for_validation,
    subtask_frames_from_record,
    validate_results_against_frame_outcomes,
)


def _write_tiny_eval(root: Path) -> None:
    root.mkdir()
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame(
        [
            {"episode_index": 0, "data/chunk_index": 0, "data/file_index": 0},
            {"episode_index": 1, "data/chunk_index": 0, "data/file_index": 0},
        ]
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(
        [
            {
                "episode_index": 0,
                "frame_index": 0,
                "reward": 0.0,
                "done": 0,
                "success": 0,
                "is_valid": 1,
            },
            {
                "episode_index": 0,
                "frame_index": 1,
                "reward": 0.0,
                "done": 1,
                "success": 0,
                "is_valid": 1,
            },
            {
                "episode_index": 0,
                "frame_index": 2,
                "reward": 0.0,
                "done": 1,
                "success": 0,
                "is_valid": 1,
            },
            {
                "episode_index": 1,
                "frame_index": 0,
                "reward": 0.0,
                "done": 0,
                "success": 1,
                "is_valid": 1,
            },
            {
                "episode_index": 1,
                "frame_index": 1,
                "reward": 0.0,
                "done": 0,
                "success": 1,
                "is_valid": 1,
            },
            {
                "episode_index": 1,
                "frame_index": 2,
                "reward": 1.0,
                "done": 1,
                "success": 1,
                "is_valid": 1,
            },
            {
                "episode_index": 1,
                "frame_index": 3,
                "reward": 1.0,
                "done": 1,
                "success": 1,
                "is_valid": 1,
            },
        ]
    ).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    (root / "results.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "name": "baseline",
                        "policy_id": 0,
                        "num_rounds": 2,
                        "successes": 0,
                        "failures": 2,
                        "success_rate": 0.0,
                    }
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
                        "round": 2,
                        "policy_id": 0,
                        "episode_index": 1,
                        "outcome": "timeout",
                        "num_steps": 4,
                    },
                ],
            },
            indent=2,
        )
        + "\n"
    )
    (root / ".outcome_edit_progress.json").write_text(
        json.dumps(
            {
                "changed_episodes": {
                    "1": {
                        "new_outcome": "success",
                        "outcome_frame": 2,
                        "soft_truncate": False,
                    }
                },
                "skipped_episodes": [],
            },
            indent=2,
        )
        + "\n"
    )


def test_stale_results_fail_against_frame_outcomes(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    payload = json.loads((root / "results.json").read_text())

    with pytest.raises(RuntimeError, match="episode 1.*timeout.*success"):
        validate_results_against_frame_outcomes(payload, load_frame_outcomes_from_root(root))


def test_canonicalize_results_file_applies_edits_and_preserves_backup(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)

    results_path, backup_path, reconciliation = canonicalize_results_file(root)

    assert results_path == root / "results.json"
    assert backup_path == root / "results_eval_time.json"
    assert reconciliation is not None
    assert reconciliation["success_flips"] == [
        {"episode_index": 1, "old": "timeout", "new": "success"}
    ]
    raw = json.loads(backup_path.read_text())
    assert raw["rollouts"][1]["outcome"] == "timeout"
    canonical = json.loads(results_path.read_text())
    assert canonical["rollouts"][1]["outcome"] == "success"
    assert canonical["rollouts"][1]["num_steps"] == 3
    assert canonical["summary"][0]["successes"] == 1
    assert canonical["_outcome_edit_reconciliation"]["episodes_reviewed"] == 1

    _, second_backup, _ = canonicalize_results_file(root)
    assert second_backup is None


def test_canonicalize_results_file_refuses_mismatched_existing_backup(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    (root / "results_eval_time.json").write_text(json.dumps({"stale": True}) + "\n")

    with pytest.raises(RuntimeError, match="stale provenance"):
        canonicalize_results_file(root)

    still_raw = json.loads((root / "results.json").read_text())
    assert still_raw["rollouts"][1]["outcome"] == "timeout"


def test_canonicalize_results_file_refreshes_resumed_eval_prefix_backup(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    results_path = root / "results.json"
    current = json.loads(results_path.read_text())
    current.update(
        {
            "timestamp": "finished",
            "dataset_name": "same-eval",
            "arena_session_id": "session-1",
            "arena_submitted_round_indices": [0, 1],
            "arena_submitted_rollouts": [[0, 0], [1, 0]],
            "phase_stops": [{"visit_id": "first"}, {"visit_id": "second"}],
        }
    )
    results_path.write_text(json.dumps(current, indent=2) + "\n")
    previous = copy.deepcopy(current)
    previous["timestamp"] = "checkpoint"
    previous["rollouts"] = previous["rollouts"][:1]
    previous["summary"][0].update(
        {"num_rounds": 1, "successes": 0, "failures": 1, "success_rate": 0.0}
    )
    previous["arena_submitted_round_indices"] = [0]
    previous["arena_submitted_rollouts"] = [[0, 0]]
    previous["phase_stops"] = [{"visit_id": "first"}]
    (root / "results_eval_time.json").write_text(json.dumps(previous, indent=2) + "\n")

    _, backup_path, _ = canonicalize_results_file(root)

    assert backup_path == root / "results_eval_time.json"
    assert json.loads(backup_path.read_text()) == current

    for key in ("arena_submitted_rollouts", "phase_stops"):
        divergent = copy.deepcopy(previous)
        divergent[key][0] = [999, 0] if key == "arena_submitted_rollouts" else {"visit_id": "other"}
        results_path.write_text(json.dumps(current) + "\n")
        backup_path.write_text(json.dumps(divergent) + "\n")
        with pytest.raises(RuntimeError, match="stale provenance"):
            canonicalize_results_file(root)


def test_canonicalize_results_file_resumed_prefix_tolerates_new_cli_arg(tmp_path: Path) -> None:
    # A resume under a newer eval build adds a CLI key to args (e.g.
    # arena_session_status). Shared keys must still match.
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    results_path = root / "results.json"
    current = json.loads(results_path.read_text())
    current.update(
        {
            "timestamp": "finished",
            "dataset_name": "same-eval",
            "arena_session_id": "session-1",
            "arena_submitted_round_indices": [0, 1],
            "args": {"environment": "routing_d2", "arena_session_status": "testing"},
        }
    )
    results_path.write_text(json.dumps(current, indent=2) + "\n")
    previous = copy.deepcopy(current)
    previous["timestamp"] = "checkpoint"
    previous["rollouts"] = previous["rollouts"][:1]
    previous["summary"][0].update(
        {"num_rounds": 1, "successes": 0, "failures": 1, "success_rate": 0.0}
    )
    previous["arena_submitted_round_indices"] = [0]
    previous["args"] = {"environment": "routing_d2"}
    (root / "results_eval_time.json").write_text(json.dumps(previous, indent=2) + "\n")

    _, backup_path, _ = canonicalize_results_file(root)
    assert backup_path == root / "results_eval_time.json"
    assert json.loads(backup_path.read_text()) == current

    # A shared key with a DIFFERENT value is still a different run (restore the
    # raw payload first: an already-reconciled results.json skips the guard).
    results_path.write_text(json.dumps(current, indent=2) + "\n")
    previous["args"] = {"environment": "marker_d2"}
    (root / "results_eval_time.json").write_text(json.dumps(previous, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="stale provenance"):
        canonicalize_results_file(root)


def test_canonicalize_results_file_refuses_divergent_same_eval_prefix(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    results_path = root / "results.json"
    current = json.loads(results_path.read_text())
    current.update({"dataset_name": "same-eval", "arena_session_id": "session-1"})
    results_path.write_text(json.dumps(current, indent=2) + "\n")
    previous = copy.deepcopy(current)
    previous["rollouts"] = previous["rollouts"][:1]
    previous["rollouts"][0]["outcome"] = "success"
    (root / "results_eval_time.json").write_text(json.dumps(previous, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="stale provenance"):
        canonicalize_results_file(root)


def test_canonicalize_results_file_preserves_backup_for_second_edit(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    canonicalize_results_file(root)
    backup_before = json.loads((root / "results_eval_time.json").read_text())
    assert backup_before["rollouts"][1]["outcome"] == "timeout"

    data_path = root / "data/chunk-000/file-000.parquet"
    frames = pd.read_parquet(data_path)
    frames.loc[
        (frames["episode_index"] == 1) & (frames["frame_index"] == 2),
        ["reward", "done"],
    ] = [0.0, 0]
    frames.to_parquet(data_path, index=False)
    (root / ".outcome_edit_progress.json").write_text(
        json.dumps(
            {
                "changed_episodes": {
                    "1": {
                        "new_outcome": "success",
                        "outcome_frame": 3,
                        "soft_truncate": False,
                    }
                },
                "skipped_episodes": [],
            },
            indent=2,
        )
        + "\n"
    )

    _, backup_path, reconciliation = canonicalize_results_file(root)

    assert backup_path == root / "results_eval_time.json"
    assert reconciliation is not None
    assert reconciliation["num_steps_patches"] == 1
    assert json.loads(backup_path.read_text()) == backup_before
    canonical = json.loads((root / "results.json").read_text())
    assert canonical["rollouts"][1]["num_steps"] == 4


def test_canonicalize_results_file_reconciled_without_backup_fails_when_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    canonicalize_results_file(root)
    (root / "results_eval_time.json").unlink()

    with pytest.raises(RuntimeError, match="missing.*eval-time provenance"):
        canonicalize_results_file(root)


def test_canonicalize_results_file_reconciled_without_backup_fails_on_second_edit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    canonicalize_results_file(root)
    (root / "results_eval_time.json").unlink()
    data_path = root / "data/chunk-000/file-000.parquet"
    frames = pd.read_parquet(data_path)
    frames.loc[
        (frames["episode_index"] == 1) & (frames["frame_index"] == 2),
        ["reward", "done"],
    ] = [0.0, 0]
    frames.to_parquet(data_path, index=False)
    (root / ".outcome_edit_progress.json").write_text(
        json.dumps(
            {
                "changed_episodes": {
                    "1": {
                        "new_outcome": "success",
                        "outcome_frame": 3,
                        "soft_truncate": False,
                    }
                }
            }
        )
    )

    with pytest.raises(RuntimeError, match="missing.*eval-time provenance"):
        canonicalize_results_file(root)


def test_validation_fails_for_missing_result_episode(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    payload = json.loads((root / "results.json").read_text())
    payload["rollouts"] = payload["rollouts"][:1]

    with pytest.raises(RuntimeError, match="missing from results.json"):
        validate_results_against_frame_outcomes(payload, load_frame_outcomes_from_root(root))


def test_validation_fails_for_duplicate_result_episode(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    payload = json.loads((root / "results.json").read_text())
    payload["rollouts"].append(dict(payload["rollouts"][0]))

    with pytest.raises(RuntimeError, match="duplicate episode_index"):
        validate_results_against_frame_outcomes(payload, load_frame_outcomes_from_root(root))


def test_frame_outcome_validation_rejects_mixed_terminal_trace(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    data_path = root / "data/chunk-000/file-000.parquet"
    frames = pd.read_parquet(data_path)
    frames.loc[
        (frames["episode_index"] == 1) & (frames["frame_index"] == 3),
        "reward",
    ] = 0.0
    frames.to_parquet(data_path, index=False)

    with pytest.raises(RuntimeError, match="changes reward after the outcome frame"):
        load_frame_outcomes_from_root(root)


def _spike_episode() -> pd.DataFrame:
    # Success terminal at frame 4 with a recorded mid-episode reward spike at
    # frame 2 (done=0, is_valid=1). Frame 5 is the invalid terminal pad.
    return pd.DataFrame(
        [
            {"episode_index": 3, "frame_index": 0, "reward": 0.0, "done": 0, "is_valid": 1},
            {"episode_index": 3, "frame_index": 1, "reward": 0.0, "done": 0, "is_valid": 1},
            {"episode_index": 3, "frame_index": 2, "reward": 1.0, "done": 0, "is_valid": 1},
            {"episode_index": 3, "frame_index": 3, "reward": 0.0, "done": 0, "is_valid": 1},
            {"episode_index": 3, "frame_index": 4, "reward": 1.0, "done": 1, "is_valid": 1},
            {"episode_index": 3, "frame_index": 5, "reward": 1.0, "done": 1, "is_valid": 0},
        ]
    )


def test_detect_frame_outcome_tolerates_recorded_subtask_spike() -> None:
    ep = _spike_episode()
    outcome = _detect_frame_outcome(ep, subtask_frames=(2,))
    assert outcome.outcome == "success"
    # expected_num_steps is the terminal frame + 1, unaffected by the spike.
    assert outcome.expected_num_steps == 5


def test_detect_frame_outcome_raises_on_unlabeled_mid_episode_reward() -> None:
    ep = _spike_episode()
    with pytest.raises(RuntimeError, match="nonzero reward before outcome frame"):
        _detect_frame_outcome(ep)  # no subtask_frames -> spike is an anomaly


def test_detect_frame_outcome_raises_when_other_frame_unlabeled() -> None:
    ep = _spike_episode()
    # Add a second, unrecorded spike at frame 1; only frame 2 is labeled.
    ep.loc[ep["frame_index"] == 1, "reward"] = 1.0
    with pytest.raises(RuntimeError, match="nonzero reward before outcome frame"):
        _detect_frame_outcome(ep, subtask_frames=(2,))


def test_detect_frame_outcome_rejects_corrupt_spike_value() -> None:
    # A labeled subtask frame must carry exactly reward=1.0 (the only value the
    # editor writes); a corrupt 0.5 there must fail loud, not pass as tolerated.
    ep = _spike_episode()
    ep.loc[ep["frame_index"] == 2, "reward"] = 0.5
    with pytest.raises(RuntimeError, match="reward != 1.0"):
        _detect_frame_outcome(ep, subtask_frames=(2,))


def test_subtask_frames_from_record_extracts_and_ignores_absent_key() -> None:
    record = {
        "changed_episodes": {
            "3": {"new_outcome": "success", "outcome_frame": 4, "subtask_frames": [2, 2]},
            "4": {"new_outcome": "failure", "outcome_frame": 5},
        }
    }
    # Deduplicated: hand-edited duplicates collapse to one recorded frame.
    assert subtask_frames_from_record(record) == {3: [2]}
    assert subtask_frames_from_record(None) == {}


def _write_spike_eval(root: Path, *, include_record: bool) -> None:
    """Local fixture: one success episode with a mid-episode subtask reward spike."""
    root.mkdir()
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame([{"episode_index": 0, "data/chunk_index": 0, "data/file_index": 0}]).to_parquet(
        root / "meta/episodes/chunk-000/file-000.parquet", index=False
    )
    ep = _spike_episode()
    ep["episode_index"] = 0
    ep["success"] = 1
    ep.to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    (root / "results.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "name": "teleop",
                        "policy_id": 0,
                        "num_rounds": 1,
                        "successes": 1,
                        "failures": 0,
                        "success_rate": 1.0,
                    }
                ],
                "rollouts": [
                    {
                        "round": 1,
                        "policy_id": 0,
                        "episode_index": 0,
                        "outcome": "success",
                        "num_steps": 5,
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    if include_record:
        (root / ".outcome_edit_progress.json").write_text(
            json.dumps(
                {
                    "changed_episodes": {
                        "0": {
                            "new_outcome": "success",
                            "outcome_frame": 4,
                            "soft_truncate": False,
                            "subtask_frames": [2],
                        }
                    },
                    "skipped_episodes": [],
                },
                indent=2,
            )
            + "\n"
        )


def test_canonicalize_results_file_tolerates_recorded_spike(tmp_path: Path) -> None:
    """End-to-end record→detector wiring: the dotfile's subtask_frames must reach
    _detect_frame_outcome through canonicalize_results_file."""
    root = tmp_path / "eval"
    _write_spike_eval(root, include_record=True)

    results_path, _, reconciliation = canonicalize_results_file(root)

    assert results_path == root / "results.json"
    assert reconciliation is not None
    canonical = json.loads(results_path.read_text())
    assert canonical["rollouts"][0]["outcome"] == "success"
    assert canonical["rollouts"][0]["num_steps"] == 5


def test_canonicalize_results_file_raises_on_spike_without_record(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_spike_eval(root, include_record=False)

    with pytest.raises(RuntimeError, match="nonzero reward before outcome frame"):
        canonicalize_results_file(root)


def test_summary_policy_ids_must_match_rollout_policy_ids(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _write_tiny_eval(root)
    payload = json.loads((root / "results.json").read_text())
    payload["rollouts"][1]["policy_id"] = 1

    with pytest.raises(RuntimeError, match="missing summary rows"):
        canonicalize_results_payload(payload)

    payload = json.loads((root / "results.json").read_text())
    payload["summary"].append(dict(payload["summary"][0]))
    with pytest.raises(RuntimeError, match="duplicate policy_id"):
        canonicalize_results_payload(payload)


def test_soft_truncated_timeout_patches_num_steps() -> None:
    payload = {
        "summary": [
            {
                "name": "policy",
                "policy_id": 0,
                "num_rounds": 1,
                "successes": 0,
                "failures": 1,
                "success_rate": 0.0,
            }
        ],
        "rollouts": [
            {
                "round": 1,
                "policy_id": 0,
                "episode_index": 7,
                "outcome": "failure",
                "num_steps": 5,
            }
        ],
    }
    reconciliation = apply_outcome_edit_record(
        payload,
        {
            "changed_episodes": {
                "7": {
                    "new_outcome": "timeout",
                    "outcome_frame": 2,
                    "soft_truncate": True,
                }
            }
        },
    )

    assert payload["rollouts"][0]["outcome"] == "timeout"
    assert payload["rollouts"][0]["num_steps"] == 3
    assert reconciliation["num_steps_patches"] == 1


def test_reedit_can_expand_previously_shortened_num_steps() -> None:
    payload = {
        "_outcome_edit_reconciliation": {
            "overrides_file": ".outcome_edit_progress.json",
            "episodes_reviewed": 1,
            "outcome_class_changes": 1,
            "num_steps_patches": 1,
            "success_flips": [],
        },
        "summary": [
            {
                "name": "policy",
                "policy_id": 0,
                "num_rounds": 1,
                "successes": 0,
                "failures": 1,
                "success_rate": 0.0,
            }
        ],
        "rollouts": [
            {
                "round": 1,
                "policy_id": 0,
                "episode_index": 7,
                "outcome": "failure",
                "num_steps": 2,
            }
        ],
    }
    reconciliation = apply_outcome_edit_record(
        payload,
        {
            "changed_episodes": {
                "7": {
                    "new_outcome": "failure",
                    "outcome_frame": 3,
                    "soft_truncate": False,
                }
            }
        },
    )

    assert payload["rollouts"][0]["num_steps"] == 4
    assert reconciliation["num_steps_patches"] == 1


def test_outcome_edit_record_requires_changed_episodes() -> None:
    payload = {
        "summary": [{"name": "policy", "policy_id": 0}],
        "rollouts": [{"policy_id": 0, "episode_index": 0, "outcome": "failure", "num_steps": 1}],
    }

    with pytest.raises(KeyError, match="changed_episodes"):
        apply_outcome_edit_record(payload, {})


def _live_results_payload() -> dict:
    # Rollout rows as save_results_file writes them: live 'g' marks land in
    # subtask_frames; an older row has no key at all.
    return {
        "rollouts": [
            {"episode_index": 0, "outcome": "failure", "num_steps": 10, "subtask_frames": [4]},
            {"episode_index": 1, "outcome": "success", "num_steps": 12, "subtask_frames": [7, 7]},
            {"episode_index": 2, "outcome": "timeout", "num_steps": 9, "subtask_frames": []},
            {"episode_index": 3, "outcome": "failure", "num_steps": 8},
        ]
    }


def test_live_subtask_frames_from_results_reads_rollout_marks() -> None:
    assert live_subtask_frames_from_results(_live_results_payload()) == {0: [4], 1: [7]}
    assert live_subtask_frames_from_results(None) == {}


def test_subtask_frames_for_validation_keeps_live_marks_on_unreviewed_episodes() -> None:
    # No review record at all: every live mark survives (partial review of an
    # in-progress eval must still validate the untouched episodes).
    assert subtask_frames_for_validation(None, _live_results_payload()) == {0: [4], 1: [7]}


def test_subtask_frames_for_validation_record_overrides_live() -> None:
    record = {
        "changed_episodes": {
            # Reviewed with a MOVED mark: the record wins over the live frame.
            "0": {"new_outcome": "failure", "outcome_frame": 9, "subtask_frames": [5]},
            # Reviewed, marks cleared: apply zeroed the live spike, so none tolerated.
            "1": {"new_outcome": "failure", "outcome_frame": 11, "subtask_frames": []},
            # Pre-subtask record (no key): same as cleared.
            "3": {"new_outcome": "failure", "outcome_frame": 7},
        }
    }
    assert subtask_frames_for_validation(record, _live_results_payload()) == {0: [5]}
    # Record-only (collection datasets have no results.json).
    assert subtask_frames_for_validation(record, None) == {0: [5]}


def test_unreviewed_live_spike_validates_with_results_provenance() -> None:
    # An unreviewed episode carrying a live spike must
    # classify once results.json documents the mark, and still fail without it.
    ep = _spike_episode()
    payload = {
        "rollouts": [
            {"episode_index": 3, "outcome": "success", "num_steps": 5, "subtask_frames": [2]}
        ]
    }
    frames = subtask_frames_for_validation(None, payload)
    assert _detect_frame_outcome(ep, subtask_frames=tuple(frames[3])).outcome == "success"
    with pytest.raises(RuntimeError, match="nonzero reward before outcome frame"):
        _detect_frame_outcome(ep, subtask_frames=tuple(subtask_frames_from_record(None).get(3, ())))
