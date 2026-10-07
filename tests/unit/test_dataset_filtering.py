#!/usr/bin/env python3
"""filter_episodes: the include_failures / include_policy_data matrix on a local dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mulligan.data.constants import DataSource, EpisodeOutcome
from mulligan.data.transforms import filter_episodes

HUMAN, AUTO = int(DataSource.HUMAN), int(DataSource.AUTONOMOUS)
SUCC, FAIL = int(EpisodeOutcome.SUCCESS), int(EpisodeOutcome.FAILURE)
REPO_ID = "local/filtering-matrix"


def _build_dagger_lerobot_dataset(root: Path, episodes: list[dict]):
    """Create a tiny video-free DAgger-style LeRobotDataset with per-frame
    ``source`` / ``success`` / ``is_valid`` columns.

    Each ``episodes`` entry is ``{"len": int, "source": int, "success": int}``;
    all frames of an episode share the given source/success and is_valid=1.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["da", "db"]},
        "source": {"dtype": "int64", "shape": (1,), "names": None},
        "success": {"dtype": "int64", "shape": (1,), "names": None},
        "is_valid": {"dtype": "int64", "shape": (1,), "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="local/dagger-episodes",
        fps=10,
        features=features,
        root=root,
        use_videos=False,
    )
    for ep in episodes:
        for t in range(ep["len"]):
            v = float(t)
            ds.add_frame(
                {
                    "observation.state": np.array([v, v + 0.5], np.float32),
                    "action": np.array([v, -v], np.float32),
                    "source": np.array([ep["source"]], np.int64),
                    "success": np.array([ep["success"]], np.int64),
                    "is_valid": np.array([1], np.int64),
                    "task": "dagger",
                }
            )
        ds.save_episode()
    ds.finalize()


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("filtering") / "ds"
    _build_dagger_lerobot_dataset(
        root,
        [
            {"len": 5, "source": HUMAN, "success": SUCC},  # ep0
            {"len": 4, "source": HUMAN, "success": FAIL},  # ep1
            {"len": 3, "source": AUTO, "success": SUCC},  # ep2
            {"len": 6, "source": AUTO, "success": FAIL},  # ep3
        ],
    )
    return root


def test_default_keeps_human_successes_only(dataset_root):
    kept, frames = filter_episodes(REPO_ID, root=dataset_root)
    assert set(kept) == {0}
    assert frames == 5


def test_include_failures_keeps_all_human_episodes(dataset_root):
    kept, frames = filter_episodes(
        REPO_ID, root=dataset_root, include_failures=True, include_policy_data=False
    )
    assert set(kept) == {0, 1}
    assert frames == 9


def test_include_policy_data_keeps_all_successes(dataset_root):
    kept, frames = filter_episodes(
        REPO_ID, root=dataset_root, include_failures=False, include_policy_data=True
    )
    assert set(kept) == {0, 2}
    assert frames == 8


def test_no_filtering_returns_none_and_counts_every_frame(dataset_root):
    kept, frames = filter_episodes(
        REPO_ID, root=dataset_root, include_failures=True, include_policy_data=True
    )
    assert kept is None
    assert frames == 18
