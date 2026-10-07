"""A task's R0 demos as RLPD transitions, read from the LeRobot dataset at the
task's pinned revision (``tasks.py``: 100 Square-Narrow demos for ``square_narrow``,
200 Square-Broad demos for ``square_broad``). RLPD (``--offline_data=teleop``) and HiL-SERL
both train on these arrays:

- frame ``t`` of a ``T``-frame episode holds ``obs[t]``, ``action[t]`` and the
  result of stepping it (``reward[t]``, ``done[t]``); the final frame is a
  padding row (``is_valid == 0``) whose observation is the genuine terminal
  observation, so an episode yields ``T-1`` transitions with
  ``next_obs[t] = obs[t+1]``;
- ``done`` (== ``reward``) is 1 exactly at the success transition, so
  ``masks = 1 - done`` is 0 there and 1 elsewhere: the critic never bootstraps
  through the sparse terminal (get this wrong and Q diverges);
- actions are clipped to ``[-(1-1e-5), 1-1e-5]`` (``RoboD4RLDataset(clip_to_eps=True)``);
- ``dones`` (episode-boundary flags) are 1 at the last transition of each demo;
- all arrays float32, observations in the robomimic order (``obs.py``).

``TaskSpec.demos_sha256`` pins the content so a wrong cache revision fails loudly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from mulligan.baselines.hilserl.obs import ACTION_DIM, PROPRIO_DIM, assemble_obs
from mulligan.baselines.hilserl.replay import TRANSITION_KEYS, Batch
from mulligan.baselines.hilserl.tasks import TaskSpec

ACTION_CLIP_EPS = 1e-5


def content_sha256(transitions: Batch) -> str:
    h = hashlib.sha256()
    for k in TRANSITION_KEYS:
        arr = np.ascontiguousarray(transitions[k])
        h.update(k.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def dataset_root(task: TaskSpec, root: Optional[Path] = None) -> Path:
    """Local snapshot of the task's dataset at the pinned revision (downloads once)."""
    if root is not None:
        return Path(root)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            task.demo_repo_id,
            repo_type="dataset",
            revision=task.demo_revision,
            allow_patterns=["meta/*", "meta/**", "data/*", "data/**"],
        )
    )


def load_frames(task: TaskSpec, root: Optional[Path] = None):
    """All frames of the task's dataset as a DataFrame sorted by global ``index``."""
    import pandas as pd

    root = dataset_root(task, root)
    info = json.loads((root / "meta" / "info.json").read_text())
    if info["total_episodes"] != task.num_episodes or info["total_frames"] != task.num_frames:
        raise ValueError(
            f"{root}: expected {task.num_episodes} episodes / {task.num_frames} frames, got {info['total_episodes']} / {info['total_frames']}"
        )
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}/data")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    if len(df) != task.num_frames:
        raise ValueError(f"{len(df)} frames read, expected {task.num_frames}")
    return df


def load_demo_transitions(
    task: TaskSpec, root: Optional[Path] = None, check_sha: bool = True
) -> Batch:
    """The task's RLPD transitions (float32) in demo order (16,233 for square_narrow,
    35,486 for square_broad)."""
    df = load_frames(task, root)
    obs_l, next_l, act_l, rew_l, mask_l, done_l = [], [], [], [], [], []
    for ep_idx in sorted(df["episode_index"].unique()):
        g = df[df["episode_index"] == ep_idx].sort_values("index")
        state = np.stack(g["observation.state"].values).astype(np.float64)
        env_state = np.stack(g["observation.environment_state"].values).astype(np.float64)
        action = np.stack(g["action"].values).astype(np.float64)
        reward = np.asarray(g["reward"].values, dtype=np.float64).ravel()
        done = np.asarray(g["done"].values, dtype=np.int64).ravel()
        is_valid = np.asarray(g["is_valid"].values, dtype=np.int64).ravel()
        if (
            state.shape[1] != PROPRIO_DIM
            or env_state.shape[1] != task.object_dim
            or action.shape[1] != ACTION_DIM
        ):
            raise ValueError(
                f"episode {ep_idx}: shapes {state.shape} {env_state.shape} {action.shape}"
            )
        # Padding-row contract (same asserts as the hdf5 builder).
        if not (is_valid[:-1].all() and is_valid[-1] == 0):
            raise ValueError(f"episode {ep_idx}: is_valid must be 1..1,0")
        if not (np.array_equal(action[-1], action[-2]) and reward[-1] == reward[-2]):
            raise ValueError(f"episode {ep_idx}: padding row must copy the previous action/reward")
        n = len(g) - 1
        obs = assemble_obs(state[:, 0:3], state[:, 3:7], state[:, 7:9], env_state)
        obs_l.append(obs[:n])
        next_l.append(obs[1:])
        # RoboD4RLDataset(clip_to_eps=True): clip in float64 (hdf5 dtype) then cast.
        lim = 1 - ACTION_CLIP_EPS
        act_l.append(np.clip(action[:n], -lim, lim).astype(np.float32))
        rew_l.append(reward[:n].astype(np.float32))
        mask_l.append((1.0 - done[:n].astype(np.float32)).astype(np.float32))
        ep_done = np.zeros(n, dtype=np.float32)
        ep_done[-1] = 1.0  # RoboD4RLDataset dones_float: episode boundary
        done_l.append(ep_done)
    transitions = dict(
        observations=np.concatenate(obs_l).astype(np.float32),
        actions=np.concatenate(act_l).astype(np.float32),
        rewards=np.concatenate(rew_l).astype(np.float32),
        masks=np.concatenate(mask_l).astype(np.float32),
        dones=np.concatenate(done_l).astype(bool),
        next_observations=np.concatenate(next_l).astype(np.float32),
    )
    n = len(transitions["observations"])
    if n != task.num_transitions:
        raise ValueError(f"{n} transitions built, expected {task.num_transitions}")
    if transitions["observations"].shape[1] != task.obs_dim:
        raise ValueError(
            f"observations are {transitions['observations'].shape[1]}-D, expected {task.obs_dim}"
        )
    if (
        int(transitions["rewards"].sum()) != task.num_episodes
        or int((transitions["masks"] == 0).sum()) != task.num_episodes
    ):
        raise ValueError("every demo must end in exactly one success transition with mask 0")
    if check_sha:
        sha = content_sha256(transitions)
        if sha != task.demos_sha256:
            raise ValueError(
                f"{task.key} demo content sha256 {sha} != pinned {task.demos_sha256}; wrong dataset revision or a changed builder"
            )
    return transitions
