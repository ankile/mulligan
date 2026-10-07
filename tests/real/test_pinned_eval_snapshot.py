from __future__ import annotations

import pytest

from mulligan.real.lifecycle.pinned_eval_snapshot import (
    check_label_history,
    hybrid_subtask_mark_counts,
    stale_subtask_marks,
)


def test_hybrid_marks_prefer_review_and_retain_unreviewed_live_marks() -> None:
    payload = {
        "rollouts": [
            {"episode_index": 0, "outcome": "failure", "subtask_frames": []},
            {"episode_index": 1, "outcome": "failure", "subtask_frames": [7]},
            {"episode_index": 2, "outcome": "success", "subtask_frames": [5]},
        ]
    }
    record = {
        "changed_episodes": {
            "0": {"new_outcome": "failure", "subtask_frames": [11]},
        },
        "skipped_episodes": [],
    }

    marks, reviewed, unreviewed = hybrid_subtask_mark_counts(payload, record, num_subtask_marks=1)

    assert marks == {0: 1, 1: 1, 2: 1}
    assert reviewed == (0,)
    assert unreviewed == (1, 2)


def test_hybrid_marks_reject_success_without_mark() -> None:
    payload = {
        "rollouts": [
            {"episode_index": 7, "outcome": "success", "subtask_frames": []},
        ]
    }
    with pytest.raises(RuntimeError, match=r"terminal-success episodes .*\[7\]"):
        hybrid_subtask_mark_counts(
            payload,
            {"changed_episodes": {}, "skipped_episodes": []},
            num_subtask_marks=1,
        )


def _event(episode: int, frames: list[int]) -> dict:
    return {
        "episode_index": episode,
        "payload": {"new_outcome": "failure", "subtask_frames": frames},
    }


def _record(**episodes: list[int]) -> dict:
    return {
        "changed_episodes": {
            key: {"new_outcome": "failure", "subtask_frames": frames}
            for key, frames in episodes.items()
        }
    }


def test_label_history_accepts_append_only_re_reviews() -> None:
    events = [_event(0, [3]), _event(1, []), _event(0, [])]
    check_label_history(events, _record(**{"0": [], "1": []}), reviewed_episodes=(0, 1))


def test_label_history_rejects_stale_latest_event() -> None:
    events = [_event(0, []), _event(0, [3])]
    with pytest.raises(RuntimeError, match=r"differs from the outcome record for episodes \[0\]"):
        check_label_history(events, _record(**{"0": []}), reviewed_episodes=(0,))


def test_label_history_rejects_coverage_gap() -> None:
    with pytest.raises(RuntimeError, match=r"reviewed without history=\[1\]"):
        check_label_history(
            [_event(0, [])], _record(**{"0": [], "1": []}), reviewed_episodes=(0, 1)
        )


def test_stale_results_marks_are_reported() -> None:
    payload = {
        "rollouts": [
            {"episode_index": 0, "subtask_frames": [4]},
            {"episode_index": 1, "subtask_frames": [9]},
            {"episode_index": 2, "subtask_frames": [6]},
        ]
    }
    record = {
        "changed_episodes": {
            "0": {"new_outcome": "failure", "subtask_frames": []},
            "1": {"new_outcome": "failure", "subtask_frames": [9, 9]},
        }
    }
    assert stale_subtask_marks(payload, record) == (0,)


def test_markless_task_reviewed_set_is_the_outcome_record() -> None:
    payload = {"rollouts": [{"episode_index": i, "outcome": "success"} for i in range(3)]}
    record = {
        "changed_episodes": {"0": {"new_outcome": "success"}, "2": {"new_outcome": "success"}}
    }

    marks, reviewed, unreviewed = hybrid_subtask_mark_counts(payload, record, num_subtask_marks=0)

    assert marks == {0: 0, 1: 0, 2: 0}
    assert reviewed == (0, 2)
    assert unreviewed == (1,)


def test_markless_task_rejects_live_marks() -> None:
    payload = {"rollouts": [{"episode_index": 0, "outcome": "failure", "subtask_frames": [4]}]}
    with pytest.raises(RuntimeError, match=r"episodes \[0\] carry some"):
        hybrid_subtask_mark_counts(payload, {"changed_episodes": {}}, num_subtask_marks=0)


def test_label_history_accepts_unchanged_pre_ledger_reviews() -> None:
    record = _record(**{"0": [], "1": []})
    check_label_history(
        [_event(1, [])],
        record,
        reviewed_episodes=(0, 1),
        pre_ledger_entries={"0": record["changed_episodes"]["0"]},
    )


def test_label_history_rejects_pre_ledger_review_changed_since() -> None:
    with pytest.raises(RuntimeError, match=r"reviewed without history=\[0\]"):
        check_label_history(
            [_event(1, [])],
            _record(**{"0": [], "1": []}),
            reviewed_episodes=(0, 1),
            pre_ledger_entries={"0": {"new_outcome": "success", "subtask_frames": []}},
        )
