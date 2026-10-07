"""A tiny local LeRobot dataset in the released sim schema (Square state observations).

Each episode has ``T`` valid frames followed by one padded frame (``is_valid=0``) that
carries the terminal observation, as in the released ``mulligan/sim-*`` datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

STATE_DIM = 9
ENV_STATE_DIM = 14
ACTION_DIM = 7
FPS = 20

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (ENV_STATE_DIM,),
        "names": None,
    },
    "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    "steps_to_go": {"dtype": "int64", "shape": (1,), "names": None},
    "source": {"dtype": "int64", "shape": (1,), "names": None},
    "success": {"dtype": "int64", "shape": (1,), "names": None},
    "is_valid": {"dtype": "int64", "shape": (1,), "names": None},
    "reward": {"dtype": "float32", "shape": (1,), "names": None},
    "done": {"dtype": "int64", "shape": (1,), "names": None},
}


def build_sim_dataset(
    root: Path,
    *,
    episode_lengths=(12, 10, 14, 11),
    source: int = 1,
    seed: int = 0,
    extra_features: dict | None = None,
    overrides: Callable[[int, int, int], dict] | None = None,
) -> Path:
    """Write the dataset to ``root`` (a new directory) and return it. Every episode is a
    success: reward 1 and done 1 on its last valid frame. ``overrides(episode, t, length)``
    returns per-frame values that replace the defaults (and fill ``extra_features``)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(seed)
    ds = LeRobotDataset.create(
        f"local/{Path(root).name}",
        fps=FPS,
        features={**FEATURES, **(extra_features or {})},
        root=root,
        use_videos=False,
    )
    for ep, length in enumerate(episode_lengths):
        for t in range(length + 1):
            valid = t < length
            last = t == length - 1
            frame = {
                "observation.state": rng.standard_normal(STATE_DIM).astype(np.float32),
                "observation.environment_state": rng.standard_normal(ENV_STATE_DIM).astype(
                    np.float32
                ),
                "action": rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
                "steps_to_go": np.array([max(length - 1 - t, 0)], np.int64),
                "source": np.array([source], np.int64),
                "success": np.array([1], np.int64),
                "is_valid": np.array([int(valid)], np.int64),
                "reward": np.array([1.0 if last else 0.0], np.float32),
                "done": np.array([int(last)], np.int64),
                "task": "square",
            }
            if overrides is not None:
                frame.update(overrides(ep, t, length))
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    return Path(root)
