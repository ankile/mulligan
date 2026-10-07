#!/usr/bin/env python3
"""
Collect autonomous policy rollouts and save them to a LeRobotDataset.

Loads an IDQL checkpoint (``policy_type: idql`` in ``metadata.json``; this covers
the DIVL agents), rolls it out in vectorized Square environments and saves every
trajectory with success/reward/done annotations and the initial simulator state.
This is the autonomous-rollout step of the paper's rounds (the ``*-policy-rollouts``
datasets): one episode per start of a locked start manifest, with ``--audit-output``
and dataset lineage.

Usage:
    python -m mulligan.sim.collect.rollouts \
        --checkpoint hf://mulligan/sim-square-narrow-r01-mulligan-divl@<revision>/seed-1 \
        --initial-states-json-file data/sim/start_manifests/<starts>.json \
        --audit-output rollouts/audit.json \
        --num-episodes 100 \
        --max-steps 400 \
        --num-envs 10 \
        --num-action-samples 32 \
        --seed 2026080311 \
        --dataset-name square-narrow-policy-rollouts

``--checkpoint`` is a local checkpoint directory or
``hf://<org>/<repo>@<revision>/<subdir>`` (see :func:`mulligan.release.hub.resolve_checkpoint`).
The environment comes from the checkpoint's ``metadata.json`` unless ``--env`` is
given. Without ``--initial-states-json-file`` the environment draws its own starts.
Environment resets are unseeded unless ``--env-seed`` is given; ``--seed`` seeds only
the policy's sampling.
"""

import argparse
import hashlib
import json
import os
import platform
import random
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import HfApi

# MuJoCo needs the CGL backend on macOS.
if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"

import robosuite.macros as macros
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.env_state_names import get_environment_state_names
from mulligan.real.collect.hf_utils import add_license_arg
from mulligan.release.hub import HF_SCHEME, resolve_checkpoint
from mulligan.sim.envs import create_robosuite_env, resolve_env_name
from mulligan.sim.placement import nut_qpos
from mulligan.sim.vec_env import AsyncVectorEnv
from mulligan.utils.progress import emit_completion

macros.IMAGE_CONVENTION = "opencv"


STATE_KEYS_NARROW = ("nut_x", "nut_y", "nut_yaw")
STATE_KEYS_BROAD = ("nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y")
# Panda robot state: eef_pos (3) + eef_quat (4) + gripper_qpos (2).
ROBOT_STATE_DIM = 9
# Index where the nut joint starts in the MuJoCo qpos (robosuite 1.5 + Panda).
OBJECT_QPOS_START_IDX = 9


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_collection_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(env_name, robot_name, camera_names, camera_height, camera_width):
    """Module-level environment factory (picklable for AsyncVectorEnv workers)."""
    return create_robosuite_env(
        env_name=env_name,
        robot_name=robot_name,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
    )


def _load_initial_states(path: Path, state_keys: tuple[str, ...]) -> list[dict]:
    payload = json.loads(path.read_text())
    states = payload.get("states") or payload.get("points")
    if states is None:
        raise ValueError(f"{path}: expected top-level states or points")
    for idx, state in enumerate(states):
        missing = [key for key in state_keys if key not in state]
        if missing:
            raise ValueError(f"{path}: state {idx} missing key(s): {missing}")
    return states


def _state_to_nut_qpos(state: dict) -> np.ndarray:
    return nut_qpos(float(state["nut_x"]), float(state["nut_y"]), float(state["nut_yaw"]))


def _broad_state_to_placements(state: dict) -> list:
    """Square-Broad placements for one start: peg body position, then nut qpos."""
    return [
        ("body_pos", "peg1", [float(state["peg_x"]), float(state["peg_y"])]),
        ("qpos", OBJECT_QPOS_START_IDX, _state_to_nut_qpos(state)),
    ]


def _checkpoint_provenance(checkpoint: str) -> tuple[str | None, str | None]:
    """(remote reference, local path) for the audit and lineage records."""
    if checkpoint.startswith(HF_SCHEME):
        return checkpoint, None
    return None, checkpoint


def _write_rollout_audit(
    *,
    path: Path,
    episodes: list[dict],
    states: list[dict] | None,
    args,
    env_name: str,
    robot_name: str,
    policy_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for fallback_idx, episode in enumerate(episodes):
        idx = int(episode.get("requested_episode_index", fallback_idx))
        state = states[idx] if states is not None else {}
        state_fields = {}
        for key, value in state.items():
            if key in STATE_KEYS_BROAD:
                state_fields[key] = float(value)
            elif isinstance(value, bool):
                state_fields[key] = bool(value)
            elif isinstance(value, int):
                state_fields[key] = int(value)
            elif isinstance(value, float):
                state_fields[key] = float(value)
        records.append(
            {
                **state_fields,
                "success": int(bool(episode["success"])),
                "length": int(episode.get("length", max(0, len(episode["actions"]) - 1))),
                "reward": float(max(episode["rewards"]) if episode["rewards"] else 0.0),
                "_diagnostic_index": idx,
            }
        )
    remote_ref, local_path = _checkpoint_provenance(args.checkpoint)
    meta = {
        "dataset_name": args.dataset_name,
        "checkpoint_artifact": remote_ref,
        "checkpoint_path": local_path,
        "env": env_name,
        "robot": robot_name,
        "num_episodes": len(episodes),
        "max_steps": args.max_steps,
        "num_envs": args.num_envs,
        "policy_type": policy_type,
        "num_action_samples": args.num_action_samples,
        "collection_seed": args.seed,
        "env_seed": args.env_seed,
        "initial_states_json_file": args.initial_states_json_file,
        "initial_states_sha256": (
            _sha256_file(Path(args.initial_states_json_file))
            if args.initial_states_json_file
            else None
        ),
    }
    path.write_text(json.dumps({"records": records, "_meta": meta}, indent=2) + "\n")
    print(f"✓ Rollout audit written to {path}")


def _write_rollout_lineage(*, dataset_dir: Path, audit_path: Path, args, policy_type: str) -> Path:
    """Persist the provenance of the rollout inside the dataset."""
    if not args.initial_states_json_file:
        raise ValueError("rollout lineage requires --initial-states-json-file")
    remote_ref, local_path = _checkpoint_provenance(args.checkpoint)
    lineage = {
        # Schema name of the released rollout datasets; kept as is.
        "schema": "mulligan.sim.policy_rollout_lineage.v1",
        "dataset_name": args.dataset_name,
        "predecessor_artifact": remote_ref,
        "predecessor_checkpoint": local_path,
        "policy_type": policy_type,
        "num_action_samples": args.num_action_samples,
        "collection_seed": args.seed,
        "num_episodes": args.num_episodes,
        "initial_states_path": args.initial_states_json_file,
        "initial_states_sha256": _sha256_file(Path(args.initial_states_json_file)),
        "source": "autonomous_policy",
    }
    if args.env_seed is not None:
        lineage["env_seed"] = args.env_seed
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = meta_dir / "lineage.json"
    lineage_path.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n")
    (meta_dir / "rollout_audit.json").write_text(audit_path.read_text())
    return lineage_path


def _apply_idql_collection_overrides(
    policy, *, policy_type: str, num_action_samples: int | None
) -> None:
    """Apply collection-only IDQL overrides after loading a checkpoint.

    The override is deliberately strict: silently ignoring an N=1/N=32 request
    would change the deployed policy and invalidate a baseline lineage.
    """
    if num_action_samples is None:
        return
    if policy_type != "idql":
        raise ValueError(
            "--num-action-samples is only valid for IDQL checkpoints; "
            f"loaded policy type is {policy_type!r}"
        )
    if num_action_samples not in {1, 32}:
        raise ValueError(f"--num-action-samples must be 1 or 32, got {num_action_samples}")
    policy.config.num_action_samples = num_action_samples


def load_idql_policy(
    checkpoint_path: Path,
    device: str = "cpu",
):
    """Load an IDQL policy from a local checkpoint directory.

    Returns:
        (policy, state_mean, state_std, action_min, action_max)
    """
    from mulligan.agents.idql import IDQLPolicy
    from mulligan.training.normalization import Normalizer

    print(f"Loading IDQL policy from {checkpoint_path}")

    stats_path = checkpoint_path / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found at {stats_path}. "
            "Make sure the IDQL checkpoint was saved with normalization stats."
        )
    stats = json.loads(stats_path.read_text())

    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    action_min = torch.tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.tensor(stats["action_max"], dtype=torch.float32, device=device)
    state_min = torch.tensor(stats["state_min"], dtype=torch.float32, device=device)
    state_max = torch.tensor(stats["state_max"], dtype=torch.float32, device=device)

    policy = IDQLPolicy.load(checkpoint_path, device=device)
    policy.set_normalizer(
        Normalizer(
            state_mean=state_mean,
            state_std=state_std,
            action_min=action_min,
            action_max=action_max,
            device=device,
            state_min=state_min,
            state_max=state_max,
        )
    )
    policy.eval()

    print(f"✓ IDQL policy loaded from {checkpoint_path}")
    return policy, state_mean, state_std, action_min, action_max


def _robot_state(obs: dict, robot_state_keys: list[str]) -> np.ndarray:
    return np.concatenate([obs[k].flatten() for k in robot_state_keys if k in obs])


def collect_trajectories(
    policy,
    vec_env,
    num_episodes: int,
    max_steps: int,
    device: str,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    camera_names: Optional[list] = None,
    robot_state_dim: int = ROBOT_STATE_DIM,
    initial_object_qpos: Optional[np.ndarray] = None,
    initial_placements: Optional[list] = None,
) -> list[dict]:
    """
    Collect IDQL policy rollouts using vectorized environments.

    Args:
        policy: IDQL policy
        vec_env: Vectorized robosuite environment
        num_episodes: Total number of episodes to collect
        max_steps: Maximum steps per episode
        device: Device to run policy on
        state_mean, state_std: z-score statistics of the concatenated robot + object state
        action_min, action_max: bounds for mapping policy actions from [-1, 1]
        camera_names: Cameras whose images are recorded (default: none)
        robot_state_dim: Leading state dimensions that form ``observation.state``
        initial_object_qpos: Nut qpos per episode, shape (N >= num_episodes, 7)
        initial_placements: Placement lists per episode (Square-Broad nut + peg)

    Returns:
        List of episode dicts with observations (T+1), environment_state (T+1),
        actions/rewards/dones (T real + 1 padded), success, length, the initial
        simulator qpos/qvel and camera images.
    """
    policy.eval()

    num_envs = len(vec_env)
    completed_episodes = []
    camera_names = camera_names or []
    robot_state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

    if initial_object_qpos is not None:
        if initial_object_qpos.shape[0] < num_episodes:
            raise ValueError(
                f"initial_object_qpos has {initial_object_qpos.shape[0]} states "
                f"but {num_episodes} episodes need {num_episodes}"
            )
        if initial_object_qpos.shape[1] != 7:
            raise ValueError(
                f"initial_object_qpos must have shape (N, 7), got {initial_object_qpos.shape}"
            )
        print(f"Using {num_episodes} pre-specified initial object states")
    if initial_placements is not None and len(initial_placements) < num_episodes:
        raise ValueError(
            f"initial_placements has {len(initial_placements)} starts but {num_episodes} "
            f"episodes need {num_episodes}"
        )

    # Each round runs num_envs episodes in parallel.
    num_rounds = (num_episodes + num_envs - 1) // num_envs

    print(f"Collecting {num_episodes} episodes using {num_envs} parallel environments")
    print(f"Running {num_rounds} rounds...")
    print()

    episodes_collected = 0
    qpos_offset = 0  # next pre-specified start

    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")
        active_envs = min(num_envs, num_episodes - round_idx * num_envs)
        batch_episode_start_idx = round_idx * num_envs

        # Idle workers in a short final round replay the first start of the round.
        if initial_placements is not None:
            batch_placements = [
                initial_placements[qpos_offset + i]
                for i in range(active_envs)
                if qpos_offset + i < len(initial_placements)
            ]
            while len(batch_placements) < num_envs:
                batch_placements.append(initial_placements[qpos_offset])
            obs_list, initial_qpos_list, initial_qvel_list = vec_env.reset_with_placements(
                batch_placements
            )
            qpos_offset += active_envs
        elif initial_object_qpos is not None:
            batch_qpos = [
                initial_object_qpos[qpos_offset + i]
                for i in range(active_envs)
                if qpos_offset + i < len(initial_object_qpos)
            ]
            while len(batch_qpos) < num_envs:
                batch_qpos.append(initial_object_qpos[qpos_offset])
            obs_list, initial_qpos_list, initial_qvel_list = (
                vec_env.reset_to_object_states_with_sim_state(batch_qpos, OBJECT_QPOS_START_IDX)
            )
            qpos_offset += active_envs
        else:
            obs_list, initial_qpos_list, initial_qvel_list = vec_env.reset_with_sim_state()

        # Reset policy internal state (action queues).
        policy.reset()

        states = [
            np.concatenate([_robot_state(obs, robot_state_keys), obs["object-state"].flatten()])
            if "object-state" in obs
            else _robot_state(obs, robot_state_keys)
            for obs in obs_list
        ]

        env_episodes = []
        for env_idx in range(num_envs):
            episode_data = {
                "observations": [],
                "environment_state": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "success": False,
                # Initial MuJoCo state for exact environment reset.
                "initial_sim_qpos": initial_qpos_list[env_idx].copy(),
                "initial_sim_qvel": initial_qvel_list[env_idx].copy(),
                "requested_episode_index": batch_episode_start_idx + env_idx,
            }
            for cam_name in camera_names:
                episode_data[f"{cam_name}_image"] = []
            env_episodes.append(episode_data)

        # Observations are stored before each step, so an episode of T actions
        # ends with T+1 observations (the last one for TD bootstrapping).
        env_done = [env_idx >= active_envs for env_idx in range(num_envs)]
        env_episode_lengths = [0] * num_envs

        while not all(env_done):
            # z-score the state, split robot / object state, map actions from [-1, 1].
            state_batch = torch.from_numpy(np.stack(states)).float().to(device)
            state_normalized = (state_batch - state_mean) / state_std
            obs_dict = {
                "observation.state": state_normalized[:, :robot_state_dim],
                "observation.environment_state": state_normalized[:, robot_state_dim:],
            }
            with torch.no_grad():
                action_normalized = policy.select_action(obs_dict)
            action_scale = (action_max - action_min) / 2.0
            action_bias = (action_max + action_min) / 2.0
            actions = (action_normalized * action_scale + action_bias).cpu().numpy()

            # Store obs[t] with the action generated from it.
            for env_idx in range(num_envs):
                if env_done[env_idx]:
                    continue
                episode_data = env_episodes[env_idx]
                obs = obs_list[env_idx]
                episode_data["observations"].append(_robot_state(obs, robot_state_keys).copy())
                if "object-state" not in obs:
                    raise KeyError(
                        f"'object-state' not found in observations. "
                        f"Available keys: {list(obs.keys())}."
                    )
                episode_data["environment_state"].append(obs["object-state"].copy())
                episode_data["actions"].append(actions[env_idx].copy())
                for cam_name in camera_names:
                    img_key = f"{cam_name}_image"
                    if img_key in obs:
                        episode_data[img_key].append(obs[img_key].copy())

            obs_list, rewards, dones, infos = vec_env.step(list(actions))

            for env_idx in range(num_envs):
                if env_done[env_idx]:
                    continue
                episode_data = env_episodes[env_idx]
                episode_data["rewards"].append(rewards[env_idx])

                # Mark success (reward 1) terminal: the envs ignore done.
                done = dones[env_idx]
                if rewards[env_idx] == 1.0:
                    done = True
                episode_data["dones"].append(done)

                if infos[env_idx].get("success", False):
                    episode_data["success"] = True

                env_episode_lengths[env_idx] += 1

                obs = obs_list[env_idx]
                robot_state = _robot_state(obs, robot_state_keys)
                states[env_idx] = (
                    np.concatenate([robot_state, obs["object-state"].flatten()])
                    if "object-state" in obs
                    else robot_state
                )

                episode_done = (
                    dones[env_idx]
                    or infos[env_idx].get("success", False)
                    or env_episode_lengths[env_idx] >= max_steps
                )
                if not episode_done:
                    continue

                env_done[env_idx] = True
                episode_data["observations"].append(robot_state.copy())
                episode_data["environment_state"].append(obs["object-state"].copy())

                episode_data["length"] = int(env_episode_lengths[env_idx])

                # Pad action, reward and done of the final frame with the last real values.
                episode_data["actions"].append(episode_data["actions"][-1].copy())
                episode_data["rewards"].append(episode_data["rewards"][-1])
                episode_data["dones"].append(episode_data["dones"][-1])
                for cam_name in camera_names:
                    img_key = f"{cam_name}_image"
                    if img_key in obs:
                        episode_data[img_key].append(obs[img_key].copy())

                if episodes_collected < num_episodes:
                    completed_episodes.append(episode_data)
                    episodes_collected += 1
                    status = "✓ SUCCESS" if episode_data["success"] else "✗ FAILURE"
                    print(
                        f"  Episode {episodes_collected}/{num_episodes} (env {env_idx}) - {status} "
                        f"(length: {env_episode_lengths[env_idx]})"
                    )

        print()

    policy.train()
    return completed_episodes


def save_episodes_to_dataset(
    episodes: list[dict],
    dataset_path: Path,
    dataset_name: str,
    task_name: str,
    camera_names: list,
    robot_type: str,
):
    """
    Save collected episodes to a new LeRobotDataset at ``dataset_path / dataset_name``.

    Args:
        episodes: Episode dicts from :func:`collect_trajectories`
        dataset_path: Parent directory of the dataset
        dataset_name: Name of the dataset
        task_name: Task identifier (``<env>_<robot>``)
        camera_names: Cameras whose images were recorded
        robot_type: Robot type string
    """
    full_dataset_path = dataset_path / dataset_name
    if full_dataset_path.exists():
        raise FileExistsError(
            f"Dataset already exists at {full_dataset_path}. "
            "Pick a new --dataset-name or remove the existing directory."
        )
    print(f"Creating new dataset at {full_dataset_path}")

    first_ep = episodes[0]
    obs_shape = first_ep["observations"][0].shape[0]
    action_shape = first_ep["actions"][0].shape[0]
    env_state_shape = first_ep["environment_state"][0].shape[0]
    sim_qpos_shape = first_ep["initial_sim_qpos"].shape[0]
    sim_qvel_shape = first_ep["initial_sim_qvel"].shape[0]

    env_state_names = get_environment_state_names(task_name.split("_")[0], env_state_shape)

    # Not shared with the teleop/DAgger schemas: the three collectors write different columns.
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs_shape,),
            "names": [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "eef_quat_x",
                "eef_quat_y",
                "eef_quat_z",
                "eef_quat_w",
                "gripper_qpos_left",
                "gripper_qpos_right",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_shape,),
            "names": [
                "delta_eef_pos_x",
                "delta_eef_pos_y",
                "delta_eef_pos_z",
                "delta_eef_rot_x",
                "delta_eef_rot_y",
                "delta_eef_rot_z",
                "gripper_action",
            ],
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (env_state_shape,),
            "names": env_state_names,
        },
        "steps_to_go": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["steps_to_go"],
        },
        # 0 = policy
        "source": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["source_id"],
        },
        "success": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["success_flag"],
        },
        # 0 = padded final frame, 1 = valid state-action pair
        "is_valid": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["is_valid_flag"],
        },
        # Reward received after taking the action
        "reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
        "done": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["done_flag"],
        },
        # Initial MuJoCo state (for exact environment reset)
        "initial_sim_qpos": {
            "dtype": "float32",
            "shape": (sim_qpos_shape,),
            "names": [f"qpos_{i}" for i in range(sim_qpos_shape)],
        },
        "initial_sim_qvel": {
            "dtype": "float32",
            "shape": (sim_qvel_shape,),
            "names": [f"qvel_{i}" for i in range(sim_qvel_shape)],
        },
    }

    for cam_name in camera_names:
        cam_key = f"{cam_name}_image"
        if cam_key in first_ep:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video",
                "shape": first_ep[cam_key][0].shape,
                "names": ["height", "width", "channels"],
            }

    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=20,
        root=str(full_dataset_path),
        robot_type=robot_type.lower(),
        features=features,
    )
    print(f"✓ Dataset created at {full_dataset_path}")

    print(f"Saving {len(episodes)} episodes to dataset...")
    for ep_idx, episode in enumerate(episodes):
        # T+1 frames: T action steps plus the padded final frame.
        num_frames = len(episode["actions"])
        success_flag = 1 if episode["success"] else 0

        for frame_idx in range(num_frames):
            is_last_frame = frame_idx == num_frames - 1
            frame = {
                "task": task_name,
                "observation.state": episode["observations"][frame_idx].astype(np.float32),
                "observation.environment_state": episode["environment_state"][frame_idx].astype(
                    np.float32
                ),
                "action": episode["actions"][frame_idx].astype(np.float32),
                "steps_to_go": np.array([num_frames - 1 - frame_idx], dtype=np.int64),
                "source": np.array([0], dtype=np.int64),
                "success": np.array([success_flag], dtype=np.int64),
                "is_valid": np.array([0 if is_last_frame else 1], dtype=np.int64),
                "reward": np.array([episode["rewards"][frame_idx]], dtype=np.float32),
                "done": np.array([episode["dones"][frame_idx]], dtype=np.int64),
                "initial_sim_qpos": episode["initial_sim_qpos"].astype(np.float32),
                "initial_sim_qvel": episode["initial_sim_qvel"].astype(np.float32),
            }
            for cam_name in camera_names:
                cam_key = f"{cam_name}_image"
                if cam_key in episode:
                    frame[f"observation.images.{cam_name}"] = episode[cam_key][frame_idx]
            dataset.add_frame(frame)

        dataset.save_episode()
        if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == len(episodes):
            print(f"  Saved {ep_idx + 1}/{len(episodes)} episodes")

    dataset.finalize()
    print(f"✓ Dataset finalized with {dataset.num_episodes} total episodes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Local checkpoint directory or hf://<org>/<repo>@<revision>/<subdir>",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="square_narrow / square_broad or a robosuite env ID (default: from metadata.json)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot type (default: from metadata.json)",
    )

    # Collection args
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument(
        "--max-steps", type=int, default=400, help="Episode step budget (the paper: 400)"
    )
    parser.add_argument("--num-envs", type=int, default=4)

    # Dataset args
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to the Hugging Face Hub (your namespace unless the name has one)",
    )
    add_license_arg(parser)

    # System args
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
        help="Device for policy inference (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--num-action-samples",
        type=int,
        choices=[1, 32],
        default=None,
        help=(
            "Override the checkpoint's number of candidate action chunks for this "
            "collection run (1 for BC, 32 for IQL best-of-N)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python/NumPy/torch for reproducible policy sampling.",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="Seed the environment resets (worker i gets env_seed + i). Default: unseeded.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="Cameras to record (e.g. agentview robot0_eye_in_hand). Default: from metadata.json.",
    )
    parser.add_argument(
        "--initial-states-json-file",
        type=str,
        default=None,
        help=(
            "Start manifest with top-level states/points (nut_x, nut_y, nut_yaw, plus "
            "peg_x, peg_y on Square-Broad). One episode per start, in order."
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=str,
        default=None,
        help="JSON audit sidecar with per-episode start, success, length and reward.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.audit_output and not args.initial_states_json_file:
        parser.error("--audit-output requires --initial-states-json-file")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.env_seed is not None and args.env_seed < 0:
        parser.error("--env-seed must be >= 0")
    # Before any policy loading or rollouts: the dataset is written only at the end.
    if (Path(args.dataset_path) / args.dataset_name).exists():
        parser.error(
            f"dataset {Path(args.dataset_path) / args.dataset_name} already exists; "
            "pick a new --dataset-name or remove the directory"
        )
    # One episode per start, in order.
    starts_needed = args.num_episodes

    if args.seed is not None:
        _seed_collection_rngs(args.seed)
        print(f"Collection RNG seed: {args.seed}")

    print("=" * 70)
    print("Policy Rollout Trajectory Collection")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Dataset: {args.dataset_name}")
    print("=" * 70)
    print()

    checkpoint_dir = resolve_checkpoint(args.checkpoint)
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found; expected a sim IDQL checkpoint")
    metadata = json.loads(metadata_path.read_text())
    policy_type = metadata.get("policy_type")
    if policy_type != "idql":
        raise ValueError(
            f"{checkpoint_dir}: rollout collection supports IDQL checkpoints "
            f"(policy_type 'idql'), got {policy_type!r}"
        )

    policy, state_mean, state_std, action_min, action_max = load_idql_policy(
        checkpoint_path=checkpoint_dir,
        device=args.device,
    )
    _apply_idql_collection_overrides(
        policy,
        policy_type=policy_type,
        num_action_samples=args.num_action_samples,
    )
    # Audits and lineage record the resolved setting even when the CLI used the default.
    args.num_action_samples = int(policy.config.num_action_samples)
    print("IDQL action selection settings:")
    print(f"  num_action_samples: {policy.config.num_action_samples}")
    print()

    camera_names = args.cameras if args.cameras is not None else metadata.get("cameras", [])
    camera_height = metadata.get("camera_height")
    camera_width = metadata.get("camera_width")
    if camera_names and (camera_height is None or camera_width is None):
        camera_height = 256
        camera_width = 256
        print("Camera resolution not in metadata.json; using 256x256")

    env_ref = args.env or metadata.get("env_name")
    if env_ref is None:
        raise ValueError(f"{metadata_path} has no env_name; pass --env")
    env_name = resolve_env_name(env_ref)
    robot_type = args.robot or metadata.get("robot_name")
    if robot_type is None:
        raise ValueError(f"{metadata_path} has no robot_name; pass --robot")

    print("Environment configuration:")
    print(f"  Environment: {env_name}")
    print(f"  Robot: {robot_type}")
    print(f"  Cameras: {camera_names if camera_names else 'None (state-only)'}")
    if camera_names:
        print(f"  Camera resolution: {camera_height}x{camera_width}")
    print(f"  Env seed: {args.env_seed if args.env_seed is not None else 'unseeded'}")
    print()

    audit_initial_states = None
    initial_object_qpos = None
    initial_placements = None
    if args.initial_states_json_file:
        if env_name == "NutAssemblySquare":
            audit_initial_states = _load_initial_states(
                Path(args.initial_states_json_file), STATE_KEYS_NARROW
            )
        elif env_name == "Square_D1":
            audit_initial_states = _load_initial_states(
                Path(args.initial_states_json_file), STATE_KEYS_BROAD
            )
        else:
            raise ValueError(
                "--initial-states-json-file is only wired for NutAssemblySquare and Square_D1"
            )
        if len(audit_initial_states) < starts_needed:
            raise ValueError(
                f"{args.initial_states_json_file} has {len(audit_initial_states)} states, "
                f"but --num-episodes={starts_needed} are needed"
            )
        if env_name == "NutAssemblySquare":
            initial_object_qpos = np.stack(
                [_state_to_nut_qpos(state) for state in audit_initial_states]
            ).astype(np.float32)
        else:
            initial_placements = [
                _broad_state_to_placements(state) for state in audit_initial_states
            ]
        print(
            f"Loaded {len(audit_initial_states)} initial states from {args.initial_states_json_file}"
        )
    print()

    print(f"Creating {args.num_envs} async environment workers...")
    make_env = partial(
        _make_env,
        env_name=env_name,
        robot_name=robot_type,
        camera_names=camera_names if camera_names else None,
        camera_height=camera_height,
        camera_width=camera_width,
    )
    env_fns = [make_env for _ in range(args.num_envs)]
    vec_env = AsyncVectorEnv(env_fns, seed=args.env_seed)
    print()

    collect = partial(
        collect_trajectories,
        policy=policy,
        vec_env=vec_env,
        max_steps=args.max_steps,
        device=args.device,
        state_mean=state_mean,
        state_std=state_std,
        action_min=action_min,
        action_max=action_max,
        camera_names=camera_names,
        initial_object_qpos=initial_object_qpos,
        initial_placements=initial_placements,
    )

    print("Collecting trajectories...")
    print()
    episodes = collect(num_episodes=args.num_episodes)

    episodes = sorted(episodes, key=lambda ep: int(ep["requested_episode_index"]))

    num_successes = sum(1 for ep in episodes if ep["success"])
    success_rate = (num_successes / len(episodes)) * 100 if episodes else 0
    avg_length = np.mean([len(ep["actions"]) for ep in episodes]) if episodes else 0
    print("Collection complete!")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Successes: {num_successes}/{len(episodes)} ({success_rate:.1f}%)")
    print(f"  Average length: {avg_length:.1f} steps")
    print()

    if args.audit_output:
        _write_rollout_audit(
            path=Path(args.audit_output),
            episodes=episodes,
            states=audit_initial_states,
            args=args,
            env_name=env_name,
            robot_name=robot_type,
            policy_type=policy_type,
        )

    task_name = f"{env_name}_{robot_type}"
    dataset_path = Path(args.dataset_path)
    save_episodes_to_dataset(
        episodes=episodes,
        dataset_path=dataset_path,
        dataset_name=args.dataset_name,
        task_name=task_name,
        camera_names=camera_names,
        robot_type=robot_type,
    )

    lineage_path = None
    if args.audit_output:
        lineage_path = _write_rollout_lineage(
            dataset_dir=dataset_path / args.dataset_name,
            audit_path=Path(args.audit_output),
            args=args,
            policy_type=policy_type,
        )

    vec_env.close()

    if args.push_to_hub:
        print()
        print("=" * 70)
        print("Pushing dataset to the Hugging Face Hub")
        print("=" * 70)
        hub_repo_id = args.dataset_name
        if "/" not in hub_repo_id:
            hub_repo_id = f"{HfApi().whoami()['name']}/{hub_repo_id}"
        print(f"Pushing to: {hub_repo_id}")
        print(f"  License: {args.license}")
        dataset = LeRobotDataset(
            repo_id=hub_repo_id,
            root=str(dataset_path / args.dataset_name),
        )
        dataset.push_to_hub(private=False, license=args.license)
        print(f"✓ Dataset pushed to https://huggingface.co/datasets/{hub_repo_id}")

    print()
    print("Done!")
    emit_completion(
        job_type="policy-rollout",
        success=True,
        current=len(episodes),
        total=args.num_episodes,
        exit_code=0,
        message=(
            f"policy rollout collection complete: {args.dataset_name} "
            f"({num_successes}/{len(episodes)} successes)"
        ),
        metrics={
            "num_episodes": len(episodes),
            "num_successes": num_successes,
            "success_rate": success_rate / 100.0,
            "num_action_samples": args.num_action_samples,
            "collection_seed": args.seed,
            "env_seed": args.env_seed,
            "initial_states_sha256": (
                _sha256_file(Path(args.initial_states_json_file))
                if args.initial_states_json_file
                else None
            ),
        },
        outputs=[
            str(dataset_path / args.dataset_name) + "/",
            *([str(Path(args.audit_output))] if args.audit_output else []),
            *([str(lineage_path)] if lineage_path is not None else []),
        ],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit_completion(
            job_type="policy-rollout",
            success=False,
            exit_code=1,
            message=f"policy rollout collection failed: {type(exc).__name__}: {exc}",
        )
        raise
