"""Accounting over outcome-edited collected data must cut off at the valid prefix.

The outcome editor (``mulligan.tools.outcome_review.apply_outcome_edits``) marks the
task-completion / done frame and, under soft truncation, sets ``is_valid=0`` on
every trailing post-outcome frame (retract / reset / operator handling). All
frame-count accounting must exclude that invalid suffix; only the physical
stored span (``saved_frames``, used by video renderers) keeps it. These tests
pin the shared ``valid_prefix_length`` helper and the split-script
``_summarize_saved_trajectory`` accounting.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mulligan.real.eval.outcome_results import (
    _detect_frame_outcome,
    first_done_inclusive_length,
    task_effective_prefix_length,
    valid_prefix_length,
)
from mulligan.data.split_protocol_quota import _summarize_saved_trajectory


def test_valid_prefix_length_variants() -> None:
    assert valid_prefix_length([1, 1, 1]) == 3  # fully valid
    assert valid_prefix_length([1, 1, 1, 0]) == 3  # terminal padding only
    assert valid_prefix_length([1, 1, 0, 0, 0]) == 2  # soft-truncated junk suffix


def test_valid_prefix_length_rejects_corruption() -> None:
    with pytest.raises(RuntimeError, match="not a valid prefix"):
        valid_prefix_length([1, 0, 1])  # valid frame after invalid one
    with pytest.raises(RuntimeError, match="no valid frames"):
        valid_prefix_length([0, 0])
    with pytest.raises(RuntimeError, match="empty is_valid"):
        valid_prefix_length([])
    with pytest.raises(RuntimeError, match="must be 0/1"):
        valid_prefix_length([1, 2])


def test_task_effective_prefix_stops_at_first_terminal_or_invalid() -> None:
    assert task_effective_prefix_length([0, 0, 1, 1, 1], [1, 1, 1, 1, 0]) == 3
    assert task_effective_prefix_length([0, 0, 0, 0], [1, 1, 0, 0]) == 2
    assert first_done_inclusive_length([0, 0, 0]) == 3


def test_task_effective_prefix_rejects_corrupt_terminal_tail() -> None:
    with pytest.raises(RuntimeError, match="done returns to zero"):
        task_effective_prefix_length([0, 1, 0], [1, 1, 1])
    with pytest.raises(RuntimeError, match="length mismatch"):
        task_effective_prefix_length([0, 1], [1])


def test_saved_trajectory_counts_only_valid_prefix() -> None:
    # 6 stored frames: valid prefix of 4 (policy,policy,human,human) then 2
    # soft-truncated junk frames that are human + intervention. Accounting must
    # exclude the junk; saved_frames keeps the full stored span for video.
    stats = _summarize_saved_trajectory(
        7,
        source_values=[0, 0, 1, 1, 1, 1],
        intervention_values=[0, 0, 1, 0, 1, 1],
        valid_values=[1, 1, 1, 1, 0, 0],
        done_values=[0, 0, 0, 1, 1, 1],
        success_values={1},
    )
    assert stats["saved_frames"] == 6  # full retained span (video path)
    assert stats["saved_valid_frames"] == 4  # cut off at is_valid prefix
    assert stats["saved_policy_frames"] == 2
    assert stats["saved_human_frames"] == 2  # junk human frames excluded
    assert stats["saved_intervention_count"] == 1  # junk interventions excluded
    assert stats["saved_success"] is True
    assert stats["saved_final_done"] is True  # done at last valid frame
    assert stats["saved_trajectory_kind"] == "mixed_human_success"


def test_saved_trajectory_terminal_padding_only() -> None:
    stats = _summarize_saved_trajectory(
        0,
        source_values=[0, 0, 0, 0],
        intervention_values=[0, 0, 0, 0],
        valid_values=[1, 1, 1, 0],  # only terminal padding invalid
        done_values=[0, 0, 0, 0],
        success_values={0},
    )
    assert stats["saved_frames"] == 4
    assert stats["saved_valid_frames"] == 3
    assert stats["saved_policy_frames"] == 3
    assert stats["saved_human_frames"] == 0
    assert stats["saved_trajectory_kind"] == "policy_only_failure"


def test_saved_trajectory_rejects_bad_patterns() -> None:
    with pytest.raises(RuntimeError, match="not a valid prefix"):
        _summarize_saved_trajectory(
            1,
            source_values=[0, 1, 0],
            intervention_values=[0, 0, 0],
            valid_values=[1, 0, 1],
            done_values=[0, 0, 1],
            success_values={1},
        )
    with pytest.raises(ValueError, match="mixed success"):
        _summarize_saved_trajectory(
            2,
            source_values=[0, 0],
            intervention_values=[0, 0],
            valid_values=[1, 0],
            done_values=[0, 1],
            success_values={0, 1},
        )


def test_detect_frame_outcome_truncates_num_steps_at_done() -> None:
    # Success at frame 2 (done=1), then two soft-truncated junk frames.
    ep = pd.DataFrame(
        [
            {"episode_index": 5, "frame_index": 0, "reward": 0.0, "done": 0, "is_valid": 1},
            {"episode_index": 5, "frame_index": 1, "reward": 0.0, "done": 0, "is_valid": 1},
            {"episode_index": 5, "frame_index": 2, "reward": 1.0, "done": 1, "is_valid": 1},
            {"episode_index": 5, "frame_index": 3, "reward": 1.0, "done": 1, "is_valid": 0},
            {"episode_index": 5, "frame_index": 4, "reward": 1.0, "done": 1, "is_valid": 0},
        ]
    )
    outcome = _detect_frame_outcome(ep)
    assert outcome.outcome == "success"
    assert outcome.expected_num_steps == 3  # frames 0..2, junk excluded
