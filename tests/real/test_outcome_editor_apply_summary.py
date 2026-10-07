import pandas as pd
import pytest

from mulligan.tools.outcome_review import _episode_success_and_frames


@pytest.mark.parametrize(
    "marks, expected_counts", [(None, (0, 0)), ({0: [0, 1], 1: [0]}, (2, 1)), ({0: [1]}, (1, 0))]
)
def test_episode_success_and_frames_matches_applied_rows(marks, expected_counts):
    frames = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "success": [1, 1, 1, 0, 0],
            "is_valid": [1, 1, 0, 1, 0],
        }
    )

    assert _episode_success_and_frames(frames, subtask_frames_by_episode=marks) == [
        {
            "episode_index": 0,
            "success": True,
            "num_frames": 2,
            "num_subtask_marks": expected_counts[0],
        },
        {
            "episode_index": 1,
            "success": False,
            "num_frames": 1,
            "num_subtask_marks": expected_counts[1],
        },
    ]


def test_episode_success_and_frames_rejects_nonconstant_success():
    frames = pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "success": [0, 1],
            "is_valid": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="success must be episode-constant"):
        _episode_success_and_frames(frames)


def test_episode_success_and_frames_rejects_nonbinary_success():
    frames = pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "success": [2, 2],
            "is_valid": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="success must be binary 0/1"):
        _episode_success_and_frames(frames)
