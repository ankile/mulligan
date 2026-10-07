#!/usr/bin/env python3
# ruff: noqa: E402

"""
Sim training entry point: IDQL and DIVL on state observations.

- IDQL (``--policy.type idql``): IQL critics (V by expectile regression, chunked Q by
  TD) and a diffusion-policy actor trained with BC; best-of-N reranking at inference.
- DIVL (``--policy.type idql_divl``): IDQL with a distributional value network; the
  paper trains it on top of a frozen IDQL actor (``--training.update_components
  critic_value_only --pretrained_artifact <IDQL checkpoint>``).

Each run writes ``<system.checkpoint_dir>/<policy>_<env>_<timestamp>/`` with
``config.json``, ``checkpoints/`` (best/periodic/final) and ``videos/``.

Example (IDQL on the round-0 teleop data of Square-Narrow):
    python -m mulligan.training.train \\
        --dataset.repo_ids mulligan/sim-square-narrow-c00-teleop-sobol \\
        --env.name NutAssemblySquare --env.robot Panda \\
        --policy.type idql --policy.num_action_samples 32 \\
        --training.training_steps 150000 --eval.freq 150000

The training command of each released sim checkpoint comes from ``configs/sim/recipes.json``
(``python -m mulligan.sim.recipes train-argv <recipe> --stage <stage> --seed <seed>``,
``docs/sim.md``).
Re-running the same command with the same ``--wandb.run_name`` resumes from the last
local resume checkpoint.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial

# Suppress warnings before any other imports
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress common noisy loggers
import logging
import json

logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Now safe to import everything else
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path

from mulligan import apply_runtime_patches

apply_runtime_patches()

import draccus
import numpy as np
import torch
import wandb
from torch.profiler import record_function

from mulligan.data.constants import DataSource, EpisodeOutcome

# Set image convention and MuJoCo backend before other imports
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from mulligan.configs import TrainConfig
from mulligan.agents.factory import (
    create_policy,
    extract_normalization_stats,
    print_policy_info,
)
from mulligan.sim.vec_env import AsyncVectorEnv, SyncVectorEnv
from mulligan.utils.progress import ProgressEtaEstimator, emit_completion, emit_progress
from mulligan.training.batch_processor import create_batch_processor
from mulligan.training.checkpoint_manager import create_checkpoint_manager
from mulligan.training.evaluation import (
    evaluate_iql_policy_with_qv_values,
    make_fixed_eval_initial_states,
)
from mulligan.training.fast_dataset_loader import load_multiple_datasets_fast
from mulligan.training.precision import configure_torch_precision, require_compiled_bf16_support
from mulligan.training.resume import AutoResumeManager, should_save_resume_checkpoint
from mulligan.training.straddled_sampler import GPUBatchSampler
from mulligan.training.timer import TrainingTimer
from mulligan.release.hub import create_artifact_metadata, upload_checkpoint_to_wandb


from mulligan.training.fast_dataset_loader import (
    prepare_chunked_data,
    refresh_random_padding_for_batch,
)


def _resolve_dataset_path(repo_id: str, default_root: str | None) -> tuple[str, str | None]:
    """
    Resolve the correct repo_id and root for a dataset.

    LeRobot's root parameter is the FULL path to the dataset, not a parent directory.
    This function handles:
    - Hub datasets (like "mulligan/sim-square-narrow-c00-teleop-sobol"): repo_id as-is, root=None
    - Local full paths (like "data/my-dataset"): Extract name, use path as root

    Args:
        repo_id: Either a HF Hub ID ("user/dataset") or a local path ("data/dataset-name")
        default_root: Default root directory (used for Hub datasets if specified)

    Returns:
        Tuple of (resolved_repo_id, resolved_root)
    """
    path = Path(repo_id)

    # Check if this is a local path that exists
    if path.exists() and path.is_dir():
        # Local dataset - use basename as repo_id, full path as root
        return path.name, str(path)

    # Check if it looks like a Hub ID (contains exactly one "/" with user/repo format)
    if "/" in repo_id and repo_id.count("/") == 1:
        parts = repo_id.split("/")
        # If the first part doesn't exist as a local directory, treat as Hub ID
        # IMPORTANT: Hub datasets must use root=None so LeRobotDatasetMetadata
        # resolves to HF_LEROBOT_HOME/repo_id with per-dataset stats.
        # Using default_root would make all Hub datasets share a single
        # stats.json at default_root/meta/stats.json, causing wrong normalization.
        if not Path(parts[0]).exists():
            return repo_id, None

    # Fallback: check if combining with default_root creates a valid path
    if default_root:
        combined_path = Path(default_root) / repo_id
        if combined_path.exists() and combined_path.is_dir():
            return repo_id, str(combined_path)

    # Default: treat as Hub ID (root=None so LeRobot resolves to HF_LEROBOT_HOME)
    return repo_id, None


def _get_training_horizons(policy_type: str, cfg_policy) -> tuple[int, int]:
    """Return `(prediction_horizon, execution_horizon)` for data preparation."""
    if policy_type not in ("idql", "idql_divl"):
        raise ValueError(f"unsupported policy_type {policy_type!r}")
    prediction_horizon = cfg_policy.chunk_size
    if prediction_horizon is None or prediction_horizon < 1:
        raise ValueError(
            f"chunk_size must be >= 1 for policy_type={policy_type!r}, got {prediction_horizon}"
        )
    execution_horizon = cfg_policy.n_action_steps or prediction_horizon
    return prediction_horizon, execution_horizon


@dataclass
class _Episodes:
    """Per-episode facts of the flat frame tensors, one entry per (dataset, episode) pair."""

    frame_episode: torch.Tensor  # (N,) dense episode id of every frame
    success: torch.Tensor  # outcome of the episode's first frame
    has_human: torch.Tensor  # any frame with source == HUMAN

    @property
    def pure_autonomous_success(self) -> torch.Tensor:
        """Successful episodes without a single human frame."""
        return (self.success == EpisodeOutcome.SUCCESS) & ~self.has_human


def _episodes(
    dataset_indices: torch.Tensor,
    episode_indices: torch.Tensor,
    success: torch.Tensor,
    source: torch.Tensor,
) -> _Episodes:
    """Group frames by (dataset, episode) in one pass instead of a mask per episode."""
    _, frame_episode = torch.unique(
        torch.stack([dataset_indices, episode_indices], dim=1), dim=0, return_inverse=True
    )
    n = frame_episode.numel()
    n_episodes = int(frame_episode.max()) + 1 if n else 0
    frames = torch.arange(n, device=frame_episode.device)
    first = torch.full((n_episodes,), n, dtype=torch.long, device=frames.device)
    first = first.scatter_reduce(0, frame_episode, frames, reduce="amin")
    human_frames = torch.zeros(n_episodes, dtype=torch.long, device=frames.device)
    human_frames.index_add_(0, frame_episode, (source == DataSource.HUMAN).long())
    return _Episodes(
        frame_episode=frame_episode,
        success=success[first],
        has_human=human_frames > 0,
    )


def _require_policy_stratum(
    *,
    mode: str,
    available_frames: int,
    required_frames: int,
    strict: bool,
) -> None:
    """Fail when a locked actor-data mode cannot form its requested stratum."""
    if strict and available_frames < required_frames:
        raise ValueError(
            f"strict policy_data_mode={mode} requires at least {required_frames} "
            f"eligible frames, found {available_frames}"
        )


def _seed_training_rngs(seed: int) -> None:
    """Seed every RNG consumed by model initialization and batch sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("resume state holds CUDA RNG state but CUDA is not available")
        torch.cuda.set_rng_state_all(state["cuda"])


# Config keys a resumed run may change: extending a run to more steps.
_RESUME_MUTABLE_KEYS = frozenset({("training", "training_steps")})


def _check_resume_config(saved: dict, current: dict) -> None:
    """Fail when a resumed run's config differs from the saved one (except training_steps)."""

    def flatten(d, prefix=()):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flatten(v, prefix + (k,))
            else:
                yield prefix + (k,), v

    a, b = dict(flatten(saved)), dict(flatten(current))
    diff = {
        ".".join(k): (a.get(k), b.get(k))
        for k in sorted(a.keys() | b.keys())
        if k not in _RESUME_MUTABLE_KEYS and a.get(k) != b.get(k)
    }
    if diff:
        raise RuntimeError(
            "the resume state was saved with a different config (saved, current): "
            f"{diff}. Only training.training_steps may change on resume; use a new "
            "wandb.run_name for a different run."
        )


def _make_robosuite_eval_env(
    env_name: str,
    robot_name: str,
    save_video: bool,
):
    """
    Module-level factory function for creating evaluation environments.

    This MUST be at module level (not a nested function) to be picklable,
    which is required for multiprocessing with 'spawn' start method on macOS.
    Linux uses 'fork' which doesn't require pickling, but macOS M chips require 'spawn'.
    """
    from mulligan.sim.envs import create_robosuite_env

    return create_robosuite_env(
        env_name=env_name,
        robot_name=robot_name,
        camera_names=None,
        camera_height=None,
        camera_width=None,
        controller=None,
        render_camera="agentview",
        has_renderer=False,
        visual_aids=False,
        render_size=(240, 320),
        use_render_wrapper=save_video,
    )


@draccus.wrap()
def train(cfg: TrainConfig):
    """Main training loop."""
    # The config as launched; train() fills derived fields (dims, dataset_info) later.
    launch_config = cfg.to_dict()
    if cfg.training.seed is not None:
        _seed_training_rngs(cfg.training.seed)
        print(f"Training RNG seed: {cfg.training.seed}")
    if (
        cfg.training.amp_dtype == "bfloat16"
        and cfg.training.compile_actor
        and cfg.training.update_components != "critic_value_only"
    ):
        require_compiled_bf16_support(cfg.system.device)
    tf32_enabled = configure_torch_precision(cfg.training.enable_tf32)
    if cfg.training.enable_tf32 and not tf32_enabled:
        print("TF32 requested but CUDA is not available; ignoring training.enable_tf32.")

    # Parse comma-separated repo IDs
    repo_ids = cfg.dataset.get_repo_id_list()

    # Get policy type string from draccus ChoiceRegistry
    policy_type = cfg.policy.get_choice_name(type(cfg.policy))

    env_name = cfg.env.name.lower()

    # Collect dataset metadata
    print("Collecting dataset metadata...")
    dataset_info = {}
    fps = None

    # Resolve dataset paths - each dataset may have different root
    # Local datasets use their full path as root, Hub datasets use default root
    resolved_datasets = []
    for repo_id in repo_ids:
        resolved_repo_id, resolved_root = _resolve_dataset_path(repo_id, cfg.dataset.root)
        resolved_datasets.append((repo_id, resolved_repo_id, resolved_root))

    print("\n=== Datasets ===")
    for original_id, resolved_repo_id, resolved_root in resolved_datasets:
        print(f"Processing dataset: {original_id}")
        requested_revision = (
            cfg.dataset.revisions[original_id] if cfg.dataset.revisions is not None else None
        )
        if resolved_root and resolved_root != cfg.dataset.root:
            print(f"  (local path, root={resolved_root})")
        temp_meta = LeRobotDatasetMetadata(
            resolved_repo_id,
            root=resolved_root,
            revision=requested_revision,
        )

        if fps is None:
            fps = temp_meta.fps
        else:
            if fps != temp_meta.fps:
                raise ValueError(f"FPS mismatch between datasets: {fps} != {temp_meta.fps}")

        # Store with original ID for consistency in logs
        dataset_info[original_id] = {
            "total_episodes": temp_meta.total_episodes,
            "total_frames": temp_meta.total_frames,
            "revision": temp_meta.revision,
            "fps": temp_meta.fps,
            "resolved_repo_id": resolved_repo_id,
            "resolved_root": resolved_root,
        }
        print(f"  {temp_meta.total_episodes} episodes ({temp_meta.total_frames} timesteps)")
        print()

    assert fps is not None, "FPS must be set"

    # Resume is local: a named run (wandb.run_name) re-uses its rolling resume checkpoint.
    run_name_wb = cfg.wandb.run_name
    auto_resume_enabled = run_name_wb is not None and cfg.training.resume_checkpoint_freq > 0
    resume_manager = AutoResumeManager(
        Path(cfg.system.checkpoint_dir),
        run_name=run_name_wb or "",
        enabled=auto_resume_enabled,
    )
    resume_manager.acquire_lock()
    resume_meta = resume_manager.load_metadata()
    if resume_meta is not None and resume_meta.completed:
        raise RuntimeError(
            f"Run '{run_name_wb}' is already marked completed locally. "
            "Use a new wandb.run_name for a fresh training run."
        )

    resume_state = resume_manager.load_training_state() if resume_meta is not None else None
    if resume_state is not None:
        if "config" not in resume_state:
            raise RuntimeError(
                f"{resume_manager.state_path} has no saved config; it cannot be checked "
                "against this run. Use a new wandb.run_name."
            )
        _check_resume_config(resume_state["config"], launch_config)
    if resume_meta is not None:
        run_dir = Path(resume_meta.run_dir)
        run_name = run_dir.name
        print(f"✓ Resuming existing local run directory: {run_dir}")
    else:
        # Microseconds keep simultaneous launches apart.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = f"{policy_type}_{env_name}_{timestamp}"
        run_dir = Path(cfg.system.checkpoint_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(run_dir / "config.json")
        print(f"✓ Config saved to {run_dir / 'config.json'}")
        if auto_resume_enabled:
            resume_manager.initialize_fresh_run(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"{policy_type} training")
    print("=" * 60)
    print(f"Datasets: {', '.join(repo_ids)}")
    print(f"Root: {cfg.dataset.root if cfg.dataset.root else 'HuggingFace Hub (default cache)'}")
    print(f"Device: {cfg.system.device}")
    print(
        f"Precision: amp_dtype={cfg.training.amp_dtype}, "
        f"enable_tf32={cfg.training.enable_tf32}, "
        f"tf32_active={tf32_enabled}, "
        f"compile_actor={cfg.training.compile_actor}, "
        f"compile_mode={cfg.training.compile_mode}"
    )
    print(f"Batch size: {cfg.training.batch_size}")
    print(
        f"Actor LR: {cfg.training.lr_actor}, Critic LR: {cfg.training.lr_critic}, Value LR: {cfg.training.lr_value}"
    )
    print(
        f"Gradient clipping: Actor {cfg.training.max_grad_norm_actor}, Critic {cfg.training.max_grad_norm_critic}, Value {cfg.training.max_grad_norm_value}"
    )

    # Print policy-specific configuration
    cfg.policy.print_config_info()

    print(f"Run directory: {run_dir}")
    print("=" * 60)
    print()

    # Load datasets
    print("Loading datasets...")
    lerobot_datasets = []
    for original_id in repo_ids:
        info = dataset_info[original_id]
        dataset = LeRobotDataset(
            info["resolved_repo_id"],
            root=info["resolved_root"],
            revision=info["revision"],
        )
        lerobot_datasets.append(dataset)
        print(f"  {original_id}: {info['total_episodes']} episodes, {len(dataset)} frames")

    total_frames = sum(len(d) for d in lerobot_datasets)
    print(f"\nTotal: {total_frames} frames")

    # Remove camera keys from all datasets
    print("\nFiltering out camera keys to skip video decoding...")
    for dataset in lerobot_datasets:
        camera_keys = dataset.meta.camera_keys
        if camera_keys:
            # Image camera features are hf_dataset columns; video features are separate files
            # (not columns). Capture the image ones from meta (no data load) so we only force
            # the auto-loading `hf_dataset` property when there's a real column to drop —
            # that property never returns None, so an `is not None` guard would trigger an
            # eager full-dataset materialization on every (often video-only) datamix.
            features = dataset.meta.info.features
            image_camera_keys = [
                k for k in camera_keys if features.get(k, {}).get("dtype") == "image"
            ]
            for key in camera_keys:
                features.pop(key, None)
            for key in image_camera_keys:
                dataset.reader.hf_dataset = dataset.hf_dataset.remove_columns([key])
    print("✓ Camera keys removed from all datasets")

    # Fast preprocessing using PyArrow vectorized operations
    print("\nPre-processing offline datasets into tensor format (fast PyArrow mode)...")

    start_time = time.time()

    # Use fast loader that directly accesses Arrow tables
    data = load_multiple_datasets_fast(
        lerobot_datasets,
        intervention_negative_reward=cfg.dataset.intervention_negative_reward,
        reward_shift=cfg.dataset.reward_shift,
    )

    # Extract tensors from result dict
    offline_states = data["states"]
    offline_actions = data["actions"]
    offline_rewards = data["rewards"]
    offline_next_states = data["next_states"]
    offline_source = data["source"]
    offline_success = data["success"]
    offline_intervention = data["intervention"]
    offline_dataset_indices = data["dataset_indices"]

    # Save base (per-step) rewards BEFORE chunking for computing episode bounds
    # After chunking, data["rewards"] becomes cumulative discounted rewards,
    # which would give incorrect bounds if summed (overlapping chunks cause overcounting)
    base_rewards = data["rewards"].clone()

    total_transitions = len(offline_states)
    load_time = time.time() - start_time

    print(f"✓ Pre-processed {total_transitions:,} valid transitions in {load_time:.2f}s")
    print(f"  States: {offline_states.shape}, Actions: {offline_actions.shape}")

    episode_indices = data["episode_indices"]

    prediction_horizon, execution_horizon = _get_training_horizons(policy_type, cfg.policy)
    gamma = cfg.policy.gamma

    print(
        "\nPreparing data with "
        f"prediction_horizon={prediction_horizon}, execution_horizon={execution_horizon}, "
        f"gamma={gamma}..."
    )
    critic_data = prepare_chunked_data(data, chunk_size=execution_horizon, gamma=gamma)
    if prediction_horizon == execution_horizon:
        # Equal horizons can share the prepared tensors; only decoupled IDQL
        # runs need a second action tensor for actor BC.
        policy_data = critic_data
    else:
        # This intentionally stores a second action tensor only when the actor
        # prediction horizon differs from the critic/Q execution horizon.
        policy_data = prepare_chunked_data(data, chunk_size=prediction_horizon, gamma=gamma)

    # Extract prepared data
    offline_actions = critic_data["actions"]  # (N, execution_horizon, action_dim)
    offline_action_is_pad = critic_data["action_is_pad"]  # (N, execution_horizon)
    offline_policy_actions = policy_data["actions"]  # (N, prediction_horizon, action_dim)
    offline_policy_action_is_pad = policy_data["action_is_pad"]  # (N, prediction_horizon)
    offline_policy_chunk_valid = policy_data["chunk_valid"]  # 0 if actor chunk crosses boundary
    offline_next_states = critic_data["next_states"]  # Chunk-aligned next states
    offline_rewards = critic_data["rewards"]  # Cumulative discounted over chunk
    offline_masks = critic_data["masks"]  # Bootstrap decision (1=bootstrap, 0=don't)
    offline_chunk_valid = critic_data["chunk_valid"]  # 0 if crosses episode boundary

    if execution_horizon > 1 or prediction_horizon > 1:
        padding_ratio = offline_action_is_pad.float().mean().item()
        policy_padding_ratio = offline_policy_action_is_pad.float().mean().item()
        valid_ratio = offline_chunk_valid.mean().item()
        print(f"  Critic actions shape: {offline_actions.shape}")
        print(f"  Actor actions shape: {offline_policy_actions.shape}")
        print(f"  Critic action padding ratio: {padding_ratio:.2%}")
        print(f"  Actor action padding ratio: {policy_padding_ratio:.2%}")
        print(
            f"  Critic valid chunks ratio: {valid_ratio:.2%} "
            f"({int(offline_chunk_valid.sum().item()):,} valid)"
        )
    else:
        print("  Single-step mode (chunk_size=1)")

    # Print intervention statistics if intervention negative reward is enabled
    if cfg.dataset.intervention_negative_reward not in (None, 0.0):
        intervention_frames = (offline_intervention == 1).sum().item()
        print(
            f"\n✓ Intervention-based negative rewards enabled (penalty: {cfg.dataset.intervention_negative_reward})"
        )
        print(
            f"  Intervention frames: {intervention_frames:,} ({100 * intervention_frames / total_transitions:.1f}%)"
        )
        if intervention_frames == 0:
            print("  WARNING: No intervention frames found in dataset. Flag will have no effect.")

    # Print reward shift info
    if cfg.dataset.reward_shift != 0.0:
        print(
            f"\n✓ Reward shift applied: {cfg.dataset.reward_shift:+.1f} "
            "(all rewards shifted by this constant)"
        )

    # Compute empirical min/max undiscounted sums from episode rewards
    # IMPORTANT: Use base_rewards (per-step rewards before chunking), NOT offline_rewards
    # (which are cumulative discounted after chunking). Using chunked rewards would
    # overcount due to overlapping chunk windows.
    # Group rewards by (dataset_index, episode_index) to compute per-episode statistics
    episode_rewards = {}
    for i in range(len(base_rewards)):
        key = (offline_dataset_indices[i].item(), episode_indices[i].item())
        if key not in episode_rewards:
            episode_rewards[key] = []
        episode_rewards[key].append(base_rewards[i].item())

    episode_returns = [sum(rews) for rews in episode_rewards.values()]
    episode_negative_sums = [sum(r for r in rews if r < 0) for rews in episode_rewards.values()]
    episode_positive_sums = [sum(r for r in rews if r > 0) for rews in episode_rewards.values()]

    empirical_negative_min = min(episode_negative_sums)  # Most negative sum
    empirical_positive_max = max(episode_positive_sums)  # Most positive sum

    print(f"\n✓ Empirical reward bounds from {len(episode_returns)} episodes:")
    print(f"  Undiscounted returns: [{min(episode_returns):.3f}, {max(episode_returns):.3f}]")
    print(f"  Most negative sum (all negatives): {empirical_negative_min:.3f}")
    print(f"  Most positive sum (all positives): {empirical_positive_max:.3f}")

    target_value_clip_range = None
    if cfg.policy.clip_targets_to_range:
        target_value_clip_range = (empirical_negative_min, empirical_positive_max)
        print(
            f"  Critic target clip range: [{target_value_clip_range[0]:.3f}, {target_value_clip_range[1]:.3f}]"
        )
    else:
        print("  Critic target clipping: DISABLED")

    # Load and aggregate normalization statistics across ALL datasets
    from lerobot.datasets.compute_stats import aggregate_stats

    print("\nLoading normalization statistics from all dataset metadata...")
    all_dataset_stats = []
    primary_features = None
    for original_id in repo_ids:
        info = dataset_info[original_id]
        meta = LeRobotDatasetMetadata(
            info["resolved_repo_id"],
            root=info["resolved_root"],
            revision=info["revision"],
        )
        all_dataset_stats.append(meta.stats)
        if primary_features is None:
            primary_features = meta.features
    stats = aggregate_stats(all_dataset_stats)
    print(f"  Aggregated stats from {len(all_dataset_stats)} datasets")

    assert stats is not None, (
        "Dataset does not have pre-computed statistics. "
        "Please ensure the dataset was created with LeRobot and has meta/stats.json"
    )

    # Extract normalization statistics using factory function
    norm_stats = extract_normalization_stats(stats)

    # Recompute state normalization stats from the actual loaded data.
    # This ensures stats match the exact data used for training, accounting
    # for any filtering, episode limits, or data mode transformations.
    # Action stats are unchanged.
    _rd = norm_stats.state_dim  # robot state dimension
    norm_stats.state_mean = offline_states.mean(dim=0)
    norm_stats.state_std = offline_states.std(dim=0).clamp(min=1e-6)
    norm_stats.state_min = offline_states.min(dim=0).values
    norm_stats.state_max = offline_states.max(dim=0).values
    norm_stats.robot_state_mean = offline_states[:, :_rd].mean(dim=0)
    norm_stats.robot_state_std = offline_states[:, :_rd].std(dim=0).clamp(min=1e-6)
    norm_stats.env_state_mean = offline_states[:, _rd:].mean(dim=0)
    norm_stats.env_state_std = offline_states[:, _rd:].std(dim=0).clamp(min=1e-6)

    # Dimensions used below (state tensor splitting, checkpoint metadata)
    input_dim = norm_stats.input_dim
    action_dim = norm_stats.action_dim
    state_dim = norm_stats.state_dim  # Robot state dimension (for state tensor splitting)

    print("✓ Normalization statistics loaded (recomputed from loaded data)")
    print(
        f"  State dim: {input_dim} (robot: {norm_stats.state_dim}, env: {norm_stats.env_state_dim})"
    )
    print(f"  Action dim: {action_dim}")
    print("  Action bounds (normalized): [-1.0, 1.0] (tanh-bounded actor)")
    print()

    # Populate runtime values in config
    cfg.state_dim = input_dim
    cfg.action_dim = action_dim
    cfg.num_frames = total_transitions
    cfg.dataset_info = dataset_info

    # Create W&B config from draccus config
    config = asdict(cfg)

    wandb_init_kwargs = {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "name": run_name_wb,
        "group": cfg.wandb.group,
        "notes": cfg.wandb.notes,
        "config": config,
        "mode": "online" if cfg.wandb.enabled else "disabled",
    }
    if os.environ.get("WANDB_INIT_TIMEOUT"):
        wandb_init_kwargs["settings"] = wandb.Settings(
            init_timeout=float(os.environ["WANDB_INIT_TIMEOUT"])
        )

    wandb.init(**wandb_init_kwargs)
    if cfg.wandb.enabled:
        print(f"✓ Initialized Weights & Biases (project: {cfg.wandb.project})")

        if target_value_clip_range is not None:
            wandb.run.summary["target_value_clip_range"] = target_value_clip_range

        print()

    # Create policy using factory (handles all policy types uniformly)
    policy_components = create_policy(
        policy_type=policy_type,
        norm_stats=norm_stats,
        dataset_features=primary_features,
        device=cfg.system.device,
        cfg_policy=cfg.policy,
        target_value_clip_range=target_value_clip_range,
        value_support_range=(empirical_negative_min, empirical_positive_max),
        pretrained_artifact=cfg.pretrained_artifact,
        pretrained_load_components=(
            "actor" if cfg.training.update_components == "critic_value_only" else "all"
        ),
    )

    # Extract components from factory result
    policy = policy_components.policy
    preprocessor = policy_components.preprocessor
    normalizer = policy_components.normalizer
    hidden_dims = policy_components.hidden_dims

    if cfg.training.compile_actor and cfg.training.update_components != "critic_value_only":
        if not str(cfg.system.device).startswith("cuda"):
            print("Actor compile requested but device is not CUDA; skipping torch.compile.")
        else:
            policy.configure_actor_compile(enabled=True, mode=cfg.training.compile_mode)
            print(
                "✓ Compiled diffusion actor loss "
                f"(mode={cfg.training.compile_mode}, checkpoint-safe callable path)"
            )
    elif cfg.training.compile_actor:
        print("Actor compile requested but actor is frozen; skipping torch.compile.")

    print_policy_info(policy_components)

    # Create TensorDataset from pre-processed data
    from torch.utils.data import TensorDataset

    # Single dataset with all data + source/success flags for BC masking
    # masks = bootstrap decision (1=bootstrap, 0=don't), chunk_valid = 0 if crosses episode boundary
    dataset_tensor_keys = [
        "states",
        "actions",
        "rewards",
        "next_states",
        "masks",
        "source",
        "success",
        "action_is_pad",
        "chunk_valid",
        "policy_actions",
        "policy_action_is_pad",
        "policy_chunk_valid",
        "dataset_indices",
        "episode_indices",
    ]
    dataset_tensors = [
        offline_states,
        offline_actions,
        offline_rewards,
        offline_next_states,
        offline_masks,
        offline_source,
        offline_success,
        offline_action_is_pad,
        offline_chunk_valid,
        offline_policy_actions,
        offline_policy_action_is_pad,
        offline_policy_chunk_valid,
        offline_dataset_indices,
        episode_indices,
    ]
    offline_dataset = TensorDataset(*dataset_tensors)

    # =========================================================================
    # STRADDLED SAMPLING: Compute index sets for critic vs policy data
    # =========================================================================
    # This sampling strategy follows the Human-in-the-Loop SERL (HiL-SERL) paper:
    # "HiL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL"
    # https://hil-serl.github.io/
    #
    # Critic training: 50% human-success + 50% DAgger data (dataset_idx > 0)
    # Policy training: human-only (default) or straddled with pure autonomous successes
    #
    # Index overlap is intentional:
    # The human_success_indices and dagger_indices sets may overlap. This occurs when
    # human demonstrations exist in DAgger datasets (e.g., human corrections during
    # autonomous rollouts). The overlap follows HiL-SERL practice -
    # high-quality human demonstrations are emphasized in training,
    # and appearing in both index sets achieves this naturally. The same transition
    # may be sampled twice in a batch (once from each stratum), which provides
    # implicit upweighting of valuable human data.

    print("\n=== Straddled Sampling Setup ===")

    # Human-success indices: source=HUMAN AND success=SUCCESS (from any dataset)
    human_success_mask = (offline_source == DataSource.HUMAN) & (
        offline_success == EpisodeOutcome.SUCCESS
    )
    human_success_indices = human_success_mask.nonzero(as_tuple=True)[0]
    print(f"Human-success data: {len(human_success_indices):,} frames")

    # DAgger indices: all data from datasets with index > 0
    dagger_mask = offline_dataset_indices > 0
    dagger_indices = dagger_mask.nonzero(as_tuple=True)[0]
    print(f"DAgger data (dataset_idx > 0): {len(dagger_indices):,} frames")

    # Outcome-balanced strata for critic_sampling_mode="balanced_outcome":
    #   A = success-episode frames (the high-return mode), and
    #   B = failure/correction signal = failed-episode frames OR DAgger human-
    #       intervention frames (which carry the intervention penalty and mark
    #       near-failure states; dataset_idx > 0 excludes R0 teleop, which is all
    #       HUMAN-but-success).
    # Balancing A/B 50/50 keeps the failure tail of the return distribution from
    # being washed out at high SR — the regime where the distributional (DIVL)
    # value collapses onto the success mode under the default success-upweighted
    # straddle. Strata may overlap (a recovered intervention in a success episode
    # is in both) — same intentional-overlap semantics as the straddle above.
    success_outcome_indices = (offline_success == EpisodeOutcome.SUCCESS).nonzero(as_tuple=True)[0]
    fail_intv_mask = (offline_success != EpisodeOutcome.SUCCESS) | (
        (offline_source == DataSource.HUMAN) & (offline_dataset_indices > 0)
    )
    fail_intv_indices = fail_intv_mask.nonzero(as_tuple=True)[0]
    print(
        f"Outcome strata: success={len(success_outcome_indices):,} frames, "
        f"fail/intervened={len(fail_intv_indices):,} frames"
    )

    # Compute pure autonomous success indices for optional policy straddled mode
    # Episodes where ALL frames have source=AUTONOMOUS AND success=SUCCESS
    pure_auto_success_indices = None
    if cfg.dataset.policy_data_mode == "straddled_auto_success":
        episodes = _episodes(
            offline_dataset_indices, episode_indices, offline_success, offline_source
        )
        pure = episodes.pure_autonomous_success
        pure_auto_mask = pure[episodes.frame_episode]
        pure_episode_count = int(pure.sum())

        pure_auto_success_indices = pure_auto_mask.nonzero(as_tuple=True)[0]
        print(
            f"Pure autonomous successes: {len(pure_auto_success_indices):,} frames "
            f"({pure_episode_count} episodes)"
        )

    # =========================================================================
    # MOVE DATASET TO GPU: Keep entire dataset in GPU memory for fast access
    # =========================================================================
    # For low-dimensional observations, the dataset easily fits in GPU memory.
    # This eliminates CPU→GPU transfers during training (about 60% of training time otherwise).
    # NOTE: Must happen before sampler creation so indices can be moved to GPU.
    if cfg.system.device != "cpu":
        print(f"\n=== Moving Dataset to {cfg.system.device.upper()} ===")
        # Move all tensors in TensorDataset to GPU
        offline_dataset.tensors = tuple(t.to(cfg.system.device) for t in offline_dataset.tensors)

        # Calculate and print GPU memory usage
        total_bytes = sum(t.element_size() * t.numel() for t in offline_dataset.tensors)
        total_mb = total_bytes / (1024 * 1024)
        print(f"✓ Moved {len(offline_dataset.tensors)} tensors to GPU ({total_mb:.1f} MB)")

        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"  GPU memory: {allocated_mb:.1f} MB allocated, {reserved_mb:.1f} MB reserved")

    # =========================================================================
    # Create GPU batch samplers for fast direct indexing
    # =========================================================================
    # Instead of DataLoader (which has Python overhead), we use direct GPU tensor
    # indexing with GPUBatchSampler. This avoids per-sample __getitem__ calls
    # and collation overhead.
    batch_size = cfg.training.batch_size

    # Store tensors as dict for direct indexing (already on GPU from above).
    gpu_data = dict(zip(dataset_tensor_keys, offline_dataset.tensors, strict=True))

    if cfg.dataset.critic_sampling_mode == "human_only":
        from mulligan.data.critic_sampling import human_chunk_indices

        critic_human_indices = human_chunk_indices(
            (gpu_data["source"] == DataSource.HUMAN)
            & (gpu_data["success"] == EpisodeOutcome.SUCCESS),
            gpu_data["action_is_pad"],
        )
        if len(critic_human_indices) < batch_size:
            raise ValueError(
                "Human-only critic requires one full batch of eligible chunks: "
                f"{len(critic_human_indices)} < {batch_size}"
            )
        critic_sampler = GPUBatchSampler(
            indices_a=critic_human_indices,
            indices_b=None,
            batch_size=batch_size,
            device=cfg.system.device,
        )
        print(f"\nCritic sampler: human-only ({len(critic_human_indices):,} chunks)")
        if cfg.wandb.enabled:
            wandb.summary["critic_data/eligible_human_chunks"] = len(critic_human_indices)
            wandb.summary["critic_data/total_chunks"] = len(gpu_data["source"])
    elif cfg.dataset.critic_sampling_mode == "flat":
        # Flat: uniform sampling across ALL data (no straddling)
        all_indices = torch.arange(len(offline_dataset.tensors[0]), device=cfg.system.device)
        critic_sampler = GPUBatchSampler(
            indices_a=all_indices,
            indices_b=None,
            batch_size=batch_size,
            device=cfg.system.device,
        )
        print(f"\nCritic sampler: flat (uniform across all {len(all_indices):,} frames)")
    elif cfg.dataset.critic_sampling_mode == "balanced_outcome":
        # 50% success-episode / 50% fail-or-intervened transitions.
        if (
            len(success_outcome_indices) >= batch_size // 2
            and len(fail_intv_indices) >= batch_size // 2
        ):
            critic_sampler = GPUBatchSampler(
                indices_a=success_outcome_indices,
                indices_b=fail_intv_indices,
                batch_size=batch_size,
                device=cfg.system.device,
            )
            print("\nCritic sampler: balanced_outcome (50% success, 50% fail/intervened)")
        else:
            print(
                f"\nWARNING: Not enough data for balanced_outcome critic sampling "
                f"(success={len(success_outcome_indices)}, fail/intv={len(fail_intv_indices)}, "
                f"need >= {batch_size // 2} each). Falling back to flat."
            )
            all_indices = torch.arange(len(offline_dataset.tensors[0]), device=cfg.system.device)
            critic_sampler = GPUBatchSampler(
                indices_a=all_indices,
                indices_b=None,
                batch_size=batch_size,
                device=cfg.system.device,
            )
            print("Critic sampler: all data (shuffled)")
    elif len(dagger_indices) >= batch_size // 2:
        # Default straddled: 50/50 human-success / DAgger
        critic_sampler = GPUBatchSampler(
            indices_a=human_success_indices,
            indices_b=dagger_indices,
            batch_size=batch_size,
            device=cfg.system.device,
        )
        print("\nCritic sampler: straddled (50% human-success, 50% DAgger)")
    else:
        # Fallback: not enough DAgger data, use all data
        _require_policy_stratum(
            mode="critic_straddled",
            available_frames=len(dagger_indices),
            required_frames=batch_size // 2,
            strict=cfg.dataset.strict_policy_data_mode,
        )
        print(
            f"\nWARNING: Not enough DAgger data ({len(dagger_indices)}) for straddled sampling. "
            f"Falling back to all data for critic."
        )
        all_indices = torch.arange(len(offline_dataset.tensors[0]), device=cfg.system.device)
        critic_sampler = GPUBatchSampler(
            indices_a=all_indices,
            indices_b=None,
            batch_size=batch_size,
            device=cfg.system.device,
        )
        print("Critic sampler: all data (shuffled)")

    if cfg.dataset.policy_data_mode == "straddled_auto_success":
        if (
            pure_auto_success_indices is not None
            and len(pure_auto_success_indices) >= batch_size // 2
        ):
            policy_sampler = GPUBatchSampler(
                indices_a=human_success_indices,
                indices_b=pure_auto_success_indices,
                batch_size=batch_size,
                device=cfg.system.device,
            )
            print("Policy sampler: straddled (50% human-success, 50% pure autonomous success)")
        else:
            # Fallback: not enough pure autonomous data
            n_pure = len(pure_auto_success_indices) if pure_auto_success_indices is not None else 0
            _require_policy_stratum(
                mode="straddled_auto_success",
                available_frames=n_pure,
                required_frames=batch_size // 2,
                strict=cfg.dataset.strict_policy_data_mode,
            )
            print(
                f"\nWARNING: Not enough pure autonomous success data ({n_pure}) for straddled sampling. "
                f"Falling back to human-only for policy."
            )
            policy_sampler = GPUBatchSampler(
                indices_a=human_success_indices,
                indices_b=None,
                batch_size=batch_size,
                device=cfg.system.device,
            )
            print("Policy sampler: human-only")
    elif cfg.dataset.policy_data_mode == "straddled_all":
        # Mirror critic sampling: 50% human-success + 50% all DAgger data
        if len(dagger_indices) >= batch_size // 2:
            policy_sampler = GPUBatchSampler(
                indices_a=human_success_indices,
                indices_b=dagger_indices,
                batch_size=batch_size,
                device=cfg.system.device,
            )
            print("Policy sampler: straddled_all (50% human-success, 50% all DAgger)")
        else:
            _require_policy_stratum(
                mode="straddled_all",
                available_frames=len(dagger_indices),
                required_frames=batch_size // 2,
                strict=cfg.dataset.strict_policy_data_mode,
            )
            print(
                f"\nWARNING: Not enough DAgger data ({len(dagger_indices)}) for straddled sampling. "
                f"Falling back to human-only for policy."
            )
            policy_sampler = GPUBatchSampler(
                indices_a=human_success_indices,
                indices_b=None,
                batch_size=batch_size,
                device=cfg.system.device,
            )
            print("Policy sampler: human-only")
    else:
        # Default: human_only mode
        policy_sampler = GPUBatchSampler(
            indices_a=human_success_indices,
            indices_b=None,
            batch_size=batch_size,
            device=cfg.system.device,
        )
        print("Policy sampler: human-only")

    print(f"\nPolicy data mode: {cfg.dataset.policy_data_mode}")

    # Compute frame counts for critic and policy training pools
    # Critic pool: unique frames the critic/value network trains on
    critic_num_frames = len(critic_sampler.indices_a)
    if critic_sampler.indices_b is not None:
        # Union of both index sets (they may overlap)
        critic_num_frames = len(
            torch.unique(torch.cat([critic_sampler.indices_a, critic_sampler.indices_b]))
        )

    # Policy pool: unique frames the policy/actor trains on
    policy_num_frames = len(policy_sampler.indices_a)
    if policy_sampler.indices_b is not None:
        policy_num_frames = len(
            torch.unique(torch.cat([policy_sampler.indices_a, policy_sampler.indices_b]))
        )

    print("\nTraining pool sizes:")
    print(f"  Critic/value: {critic_num_frames:,} unique frames")
    print(f"  Policy/actor: {policy_num_frames:,} unique frames")
    print("=" * 40)

    # Log frame counts to wandb
    if cfg.wandb.enabled:
        wandb.config.update(
            {"critic_num_frames": critic_num_frames, "policy_num_frames": policy_num_frames},
            allow_val_change=True,
        )
        wandb.run.summary["critic_num_frames"] = critic_num_frames
        wandb.run.summary["policy_num_frames"] = policy_num_frames

    # Create optimizers based on policy type
    if cfg.training.update_components == "critic_value_only":
        policy.set_critic_value_only_training()
        optim_params = policy.get_optim_params()
        optimizers = {
            "critic": torch.optim.AdamW(
                optim_params["critic"],
                lr=cfg.training.lr_critic,
                weight_decay=cfg.training.weight_decay,
            ),
            "value": torch.optim.AdamW(
                optim_params["value"],
                lr=cfg.training.lr_value,
                weight_decay=cfg.training.weight_decay,
            ),
        }
        max_grad_norms = {
            "critic": cfg.training.max_grad_norm_critic,
            "value": cfg.training.max_grad_norm_value,
        }
        print("IDQL update components: critic_value_only (pretrained actor frozen)")
    else:
        optim_params = policy.get_optim_params()
        optimizers = {
            name: torch.optim.AdamW(
                optim_params[name], lr=lr, weight_decay=cfg.training.weight_decay
            )
            for name, lr in (
                ("actor", cfg.training.lr_actor),
                ("critic", cfg.training.lr_critic),
                ("value", cfg.training.lr_value),
            )
        }
        max_grad_norms = {
            "actor": cfg.training.max_grad_norm_actor,
            "critic": cfg.training.max_grad_norm_critic,
            "value": cfg.training.max_grad_norm_value,
        }

    # critic_value_only: the loaded actor and its normalizer must not change.
    freeze_actor = cfg.training.update_components == "critic_value_only"
    critic_value_actor_hash_parent = policy.actor_hash() if freeze_actor else None
    critic_value_normalizer_hash_parent = policy.normalizer_hash() if freeze_actor else None

    def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    # Create checkpoint manager (the saved stats are the policy's normalizer)
    normalizer_stats = {
        "state_mean": normalizer.state_mean,
        "state_std": normalizer.state_std,
        "action_min": normalizer.action_min,
        "action_max": normalizer.action_max,
        "state_min": normalizer.state_min,
        "state_max": normalizer.state_max,
    }
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        policy_type=policy_type,
        normalizer_stats=normalizer_stats,
        dataset_size=total_frames,
        hidden_dims=hidden_dims,
    )

    # Create batch processor
    batch_processor = create_batch_processor(
        preprocessor=preprocessor,
        state_dim=state_dim,
        policy_type=policy_type,
        device=cfg.system.device,
    )

    # Create vectorized evaluation environments
    eval_vec_env = None
    eval_initial_states = None
    if cfg.eval.freq is not None:
        env_mode = "sync" if cfg.eval.sync_envs else "async"
        print(f"Creating evaluation environment(s): {cfg.env.name} ({cfg.env.robot})")
        print(f"  Using {cfg.eval.num_envs} {env_mode} parallel workers")
        if cfg.eval.fixed_initial_states:
            eval_initial_states = make_fixed_eval_initial_states(
                env_name=cfg.env.name,
                num_episodes=cfg.eval.episodes,
                seed=cfg.eval.fixed_initial_state_seed,
            )
            print(
                "  Fixed Sobol eval states enabled: "
                f"{len(eval_initial_states)} placements, seed={cfg.eval.fixed_initial_state_seed}"
            )

        if cfg.eval.save_video:
            print(
                "  Video saving enabled - using offscreen rendering with 'agentview' camera at 320x240"
            )

        # Create environment factory using partial with module-level function
        # (module-level functions are picklable, required for 'spawn' on macOS)
        make_eval_env = partial(
            _make_robosuite_eval_env,
            env_name=cfg.env.name,
            robot_name=cfg.env.robot,
            save_video=cfg.eval.save_video,
        )

        # Create list of environment factory functions
        env_fns = [make_eval_env for _ in range(cfg.eval.num_envs)]

        # Create vectorized environment
        if cfg.eval.sync_envs:
            eval_vec_env = SyncVectorEnv(env_fns)
            print(
                f"✓ Sync vectorized evaluation environment created with {cfg.eval.num_envs} workers"
            )
        else:
            eval_vec_env = AsyncVectorEnv(env_fns)
            print(
                f"✓ Async vectorized evaluation environment created with {cfg.eval.num_envs} parallel workers"
            )
        print()

    # Training loop
    print("Starting training...")
    print("=" * 60)

    best_success_rate = 0.0
    step = 1
    last_resume_checkpoint_step = 0

    if resume_state is not None:
        policy.load_state_dict(resume_state["policy_state_dict"])
        for name, optimizer_state in resume_state["optimizer_state_dicts"].items():
            if name in optimizers:
                optimizers[name].load_state_dict(optimizer_state)
                _move_optimizer_to_device(optimizers[name], cfg.system.device)
        best_success_rate = float(resume_state["best_success_rate"])
        last_resume_checkpoint_step = int(resume_state["last_checkpoint_step"])
        step = last_resume_checkpoint_step + 1
        print(
            f"✓ Restored local training state from step {last_resume_checkpoint_step} "
            f"(best SR: {best_success_rate:.1f}%)"
        )

    critic_value_actor_hash_after_resume = policy.actor_hash() if freeze_actor else None
    if critic_value_actor_hash_parent != critic_value_actor_hash_after_resume:
        raise RuntimeError(
            "critic/value-only resume state changed the frozen actor relative to parent: "
            f"parent={critic_value_actor_hash_parent}, "
            f"after_resume={critic_value_actor_hash_after_resume}"
        )
    if resume_state is not None:
        # Restored last so the continuation draws the same batches and noise as an
        # uninterrupted run.
        critic_sampler.load_state_dict(resume_state["sampler_states"]["critic"])
        policy_sampler.load_state_dict(resume_state["sampler_states"]["policy"])
        _restore_rng_state(resume_state["rng_state"])
        del resume_state

    def save_resume_checkpoint(current_step: int, reason: str) -> None:
        nonlocal last_resume_checkpoint_step
        if not auto_resume_enabled:
            return
        if current_step <= last_resume_checkpoint_step:
            return
        resume_manager.save_training_state(
            policy_state_dict=policy.state_dict(),
            optimizer_state_dicts={name: opt.state_dict() for name, opt in optimizers.items()},
            step=current_step,
            best_success_rate=best_success_rate,
            extra_state={
                "config": launch_config,
                "rng_state": _capture_rng_state(),
                "sampler_states": {
                    "critic": critic_sampler.state_dict(),
                    "policy": policy_sampler.state_dict(),
                },
            },
        )
        last_resume_checkpoint_step = current_step
        print(f"✓ Saved local resume checkpoint at step {current_step} ({reason})")

    timer = TrainingTimer()
    cumulative_timings = defaultdict(float)
    eta = ProgressEtaEstimator(start_current=step - 1)
    progress_run_id = run_name_wb or run_name

    def _progress(phase: str, current: int, message: str, **kwargs) -> None:
        emit_progress(
            job_type="training",
            run_id=progress_run_id,
            phase=phase,
            progress=min(1.0, max(0.0, current / max(1, cfg.training.training_steps))),
            current=current,
            total=cfg.training.training_steps,
            eta_s=eta.eta_s(current, cfg.training.training_steps),
            message=message,
            **kwargs,
        )

    def _scalar_metric(value):
        if isinstance(value, torch.Tensor):
            return value.item()
        if isinstance(value, (int, float)):
            return float(value)
        return None

    _progress(
        "start",
        step - 1,
        f"training started: {run_name}",
        metrics={"policy_type": policy_type, "env": cfg.env.name},
        # Trailing "/" = directory (the run owns everything under it)
        outputs=[f"{run_dir}/"],
    )

    while step <= cfg.training.training_steps:
        # Sample batches using direct GPU tensor indexing (no DataLoader overhead)
        with record_function("train/data_loading"), timer("data_loading"):
            # Sample critic batch indices (on GPU)
            critic_idx = critic_sampler.sample()

            # Direct GPU tensor indexing - single operation per tensor
            state = gpu_data["states"][critic_idx]
            action = gpu_data["actions"][critic_idx]
            reward = gpu_data["rewards"][critic_idx]
            next_state = gpu_data["next_states"][critic_idx]
            masks = gpu_data["masks"][critic_idx]
            source = gpu_data["source"][critic_idx]
            success = gpu_data["success"][critic_idx]
            action_is_pad = gpu_data["action_is_pad"][critic_idx]
            chunk_valid = gpu_data["chunk_valid"][critic_idx]

            if cfg.training.update_components != "critic_value_only":
                # Sample policy batch indices (on GPU) only when the actor is trainable.
                policy_idx = policy_sampler.sample()
                policy_state = gpu_data["states"][policy_idx]
                policy_action = gpu_data["policy_actions"][policy_idx]
                policy_reward = gpu_data["rewards"][policy_idx]
                policy_next_state = gpu_data["next_states"][policy_idx]
                policy_masks = gpu_data["masks"][policy_idx]
                policy_source = gpu_data["source"][policy_idx]
                policy_success = gpu_data["success"][policy_idx]
                policy_action_is_pad = gpu_data["policy_action_is_pad"][policy_idx]
                policy_chunk_valid = gpu_data["policy_chunk_valid"][policy_idx]

        policy.train()

        # =====================================================================
        # BATCH CREATION: Use batch processor for unified handling
        # =====================================================================
        with record_function("train/preprocessing"), timer("preprocessing"):
            # Critic batch (for V/Q updates)
            critic_batch = batch_processor.prepare_critic_batch(
                state=state,
                next_state=next_state,
                action=action,
                action_is_pad=action_is_pad,
                reward=reward,
                masks=masks,
                source=source,
                success=success,
                chunk_valid=chunk_valid,
            )

            policy_batch = None
            if cfg.training.update_components != "critic_value_only":
                policy_batch = batch_processor.prepare_policy_batch(
                    state=policy_state,
                    next_state=policy_next_state,
                    action=policy_action,
                    action_is_pad=policy_action_is_pad,
                    reward=policy_reward,
                    masks=policy_masks,
                    source=policy_source,
                    success=policy_success,
                    chunk_valid=policy_chunk_valid,
                )

        # =====================================================================
        # REFRESH RANDOM PADDING for terminal chunks
        # =====================================================================
        # This ensures the Q-function sees different random actions for padded
        # positions at each step, helping it learn invariance to post-terminal
        # actions rather than overfitting to specific random values.
        with record_function("train/refresh_random_padding"):
            critic_batch = refresh_random_padding_for_batch(critic_batch)

        # =====================================================================
        # MODEL UPDATE: critic_batch for V/Q, policy_batch for actor
        # =====================================================================
        with record_function("train/model_update"), timer("model_update"):
            # critic_batch drives the V/Q updates, policy_batch the actor update
            with record_function("train/policy_update"):
                update_kwargs = {
                    "batch": critic_batch,
                    "policy_batch": policy_batch,
                    "optimizers": optimizers,
                    "step": step,
                    "max_grad_norms": max_grad_norms,
                    "amp_dtype": cfg.training.amp_dtype,
                    "update_components": cfg.training.update_components,
                }
                update_info = policy.update(**update_kwargs)

        # =====================================================================
        # UNIFIED LOGGING (policy-agnostic)
        # =====================================================================
        if step % 1000 == 0:
            # Build console log string from available metrics
            loss_keys = ("losses/value", "losses/critic", "losses/actor")
            log_parts = [f"step: {step}/{cfg.training.training_steps}"]
            for key in loss_keys:
                if key in update_info:
                    val = update_info[key]
                    val = val.item() if isinstance(val, torch.Tensor) else val
                    log_parts.append(f"{key.split('/')[-1]}: {val:.6f}")
            print(" ".join(log_parts))
            timer.print_stats(prefix="  ")
            print()

            metric_payload = {}
            for key in loss_keys:
                if key in update_info:
                    scalar = _scalar_metric(update_info[key])
                    if scalar is not None:
                        metric_payload[key] = scalar
            _progress(
                "train",
                step,
                f"step {step}/{cfg.training.training_steps}",
                metrics=metric_payload,
            )

            # Capture timing stats before reset
            timing_totals = timer.timings.copy()

            # Update cumulative timings
            for k, v in timing_totals.items():
                cumulative_timings[k] += v

            timer.reset()

        # Log to wandb (independent of the console cadence above)
        if cfg.wandb.enabled and step % cfg.wandb.log_freq == 0:
            total_cumulative_time = sum(cumulative_timings.values())
            # Start with all metrics from update_info (convert tensors to floats)
            wandb_log = {}
            for key, val in update_info.items():
                if isinstance(val, torch.Tensor):
                    wandb_log[key] = val.item()
                else:
                    wandb_log[key] = val

            # Add dataset statistics
            wandb_log["dataset/reward_mean"] = reward.mean().item()
            wandb_log["dataset/reward_std"] = reward.std().item()
            # masks = bootstrap decision (1 = bootstrap, 0 = don't)
            wandb_log["dataset/bootstrap_fraction"] = masks.mean().item()
            wandb_log["dataset/chunk_valid_fraction"] = chunk_valid.mean().item()

            # Keep logged critic actions distinct from actor BC targets.
            wandb_log["actions/logged_critic_mean"] = critic_batch["action"].mean().item()
            wandb_log["actions/logged_critic_std"] = critic_batch["action"].std().item()

            wandb_log["histograms/reward"] = wandb.Histogram(reward.detach().cpu().numpy())

            # Add cumulative timing metrics
            wandb_log["timing_cumulative/total_hours"] = total_cumulative_time / 3600
            for key, total_time in cumulative_timings.items():
                wandb_log[f"timing_cumulative/hours_{key}"] = total_time / 3600
                if total_cumulative_time > 0:
                    wandb_log[f"timing_cumulative/share_{key}"] = total_time / total_cumulative_time

            wandb_log["step"] = step
            wandb.log(wandb_log, step=step)

        # Evaluation
        if cfg.eval.freq is not None and step % cfg.eval.freq == 0:
            # IDQL/DIVL with eval.eval_num_action_samples: one evaluation per listed N;
            # otherwise one evaluation at the configured N.
            eval_n_values = None
            if cfg.eval.eval_num_action_samples:
                eval_n_values = cfg.eval.eval_num_action_samples
            _progress(
                "eval_start",
                step,
                f"evaluation started at step {step}",
                metrics={
                    "eval_episodes": cfg.eval.episodes,
                    "eval_num_action_samples": eval_n_values or [cfg.policy.num_action_samples],
                },
            )

            if eval_n_values:
                # Per-N evaluation (also for a single N): run eval for each N, log with /nX suffix
                # Best model is selected from the LAST N value (typically largest)
                original_n = cfg.policy.num_action_samples
                all_n_metrics = {}
                wandb_log = {}

                for n_idx, n_val in enumerate(eval_n_values):
                    is_last_n = n_idx == len(eval_n_values) - 1
                    # An eval exception ends train() before any checkpoint is written, so a
                    # skipped restore below never reaches a saved config.
                    policy.config.num_action_samples = n_val

                    with (
                        record_function(f"train/evaluation_n{n_val}"),
                        timer(f"evaluation_n{n_val}"),
                    ):
                        print(f"\nEvaluating policy (n={n_val})...")
                        n_metrics, video_path, plot_path = evaluate_iql_policy_with_qv_values(
                            policy,
                            eval_vec_env,
                            normalizer=normalizer,
                            num_episodes=cfg.eval.episodes,
                            max_steps=cfg.eval.max_steps,
                            device=cfg.system.device,
                            gamma=cfg.policy.gamma,
                            save_video=cfg.eval.save_video and is_last_n,
                            save_qv_plots=is_last_n,
                            output_dir=str(video_dir),
                            step=step,
                            max_video_episodes=cfg.eval.max_video_episodes,
                            fixed_initial_states=eval_initial_states,
                        )

                    all_n_metrics[n_val] = n_metrics

                    print(f"\nEvaluation Results (step {step}, n={n_val}):")
                    print(f"  Success Rate: {n_metrics['success_rate']:.1f}%")
                    print(f"  Avg Reward: {n_metrics['avg_reward']:.3f}")
                    print(f"  Avg Length: {n_metrics['avg_length']:.1f}")
                    success_length_str = (
                        f"{n_metrics['avg_success_length']:.1f}"
                        if not np.isnan(n_metrics["avg_success_length"])
                        else "N/A"
                    )
                    print(f"  Avg Success Length: {success_length_str}")

                    # Log per-N metrics to wandb
                    if cfg.wandb.enabled:
                        suffix = f"/n{n_val}"
                        wandb_log[f"eval/success_rate{suffix}"] = n_metrics["success_rate"]
                        wandb_log[f"eval/episode_length_mean{suffix}"] = n_metrics["avg_length"]
                        wandb_log[f"eval/success_length_mean{suffix}"] = n_metrics[
                            "avg_success_length"
                        ]

                    if is_last_n:
                        if video_path is not None:
                            print(f"  Video saved to: {video_path}")
                        if plot_path is not None:
                            print(f"  Q/V-trajectory plots saved to: {plot_path}")
                    print()

                # Restore original N
                policy.config.num_action_samples = original_n

                # Primary metrics come from the last (largest) N value
                primary_n = eval_n_values[-1]
                metrics = all_n_metrics[primary_n]

                # Update best success rate based on primary N
                is_best = False
                if metrics["success_rate"] > best_success_rate:
                    best_success_rate = metrics["success_rate"]
                    is_best = True

                _progress(
                    "eval",
                    step,
                    f"eval success_rate={metrics['success_rate']:.1f}%",
                    metrics={
                        "success_rate": float(metrics["success_rate"]),
                        "best_success_rate": float(best_success_rate),
                    },
                )

                # Log primary metrics (unqualified) + best tracking
                if cfg.wandb.enabled:
                    wandb_log["eval/success_rate"] = metrics["success_rate"]
                    wandb_log["eval/success_rate_best"] = best_success_rate
                    wandb_log["eval/episode_length_mean"] = metrics["avg_length"]
                    wandb_log["eval/success_length_mean"] = metrics["avg_success_length"]
                    wandb_log["eval/episode_length_histogram"] = wandb.Histogram(
                        metrics["episode_lengths"]
                    )

                    # Add value function diagnostics from primary N
                    diagnostic_categories = [
                        "mc/",
                        "td/",
                        "calibration/",
                        "start_state/",
                        "stability/",
                        "ranking/",
                        "histograms/",
                    ]
                    for key, value in metrics.items():
                        if any(key.startswith(cat) for cat in diagnostic_categories):
                            if key.startswith("histograms/"):
                                wandb_log[key] = wandb.Histogram(value)
                            else:
                                wandb_log[key] = value

                    if video_path is not None:
                        wandb_log["eval/video"] = wandb.Video(video_path, format="mp4")
                    if plot_path is not None:
                        plot_file = Path(plot_path)
                        if plot_file.exists() and plot_file.stat().st_size > 0:
                            try:
                                wandb_log["plots/qv_trajectories"] = wandb.Image(plot_path)
                            except Exception as e:
                                print(f"  Warning: Failed to load Q/V plot for wandb: {e}")
                        else:
                            print(f"  Warning: Q/V plot file missing or empty: {plot_path}")
                    wandb.log(wandb_log, step=step)

            else:
                # Evaluation at the policy's configured N
                with record_function("train/evaluation"), timer("evaluation"):
                    print("\nEvaluating policy...")
                    metrics, video_path, plot_path = evaluate_iql_policy_with_qv_values(
                        policy,
                        eval_vec_env,
                        normalizer=normalizer,
                        num_episodes=cfg.eval.episodes,
                        max_steps=cfg.eval.max_steps,
                        device=cfg.system.device,
                        gamma=cfg.policy.gamma,
                        save_video=cfg.eval.save_video,
                        save_qv_plots=True,
                        output_dir=str(video_dir),
                        step=step,
                        max_video_episodes=cfg.eval.max_video_episodes,
                        fixed_initial_states=eval_initial_states,
                    )

                print(f"\nEvaluation Results (step {step}):")
                print(f"  Success Rate: {metrics['success_rate']:.1f}%")
                print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
                print(f"  Avg Length: {metrics['avg_length']:.1f}")
                success_length_str = (
                    f"{metrics['avg_success_length']:.1f}"
                    if not np.isnan(metrics["avg_success_length"])
                    else "N/A"
                )
                print(f"  Avg Success Length: {success_length_str}")
                if video_path is not None:
                    print(f"  Video saved to: {video_path}")
                if plot_path is not None:
                    print(f"  Q/V-trajectory plots saved to: {plot_path}")
                print()

                # Update best success rate
                is_best = False
                if metrics["success_rate"] > best_success_rate:
                    best_success_rate = metrics["success_rate"]
                    is_best = True

                _progress(
                    "eval",
                    step,
                    f"eval success_rate={metrics['success_rate']:.1f}%",
                    metrics={
                        "success_rate": float(metrics["success_rate"]),
                        "best_success_rate": float(best_success_rate),
                    },
                )

                # Log to wandb
                if cfg.wandb.enabled:
                    wandb_log = {
                        "eval/success_rate": metrics["success_rate"],
                        "eval/success_rate_best": best_success_rate,
                        "eval/episode_length_mean": metrics["avg_length"],
                        "eval/success_length_mean": metrics["avg_success_length"],
                        "eval/episode_length_histogram": wandb.Histogram(
                            metrics["episode_lengths"]
                        ),
                    }

                    diagnostic_categories = [
                        "mc/",
                        "td/",
                        "calibration/",
                        "start_state/",
                        "stability/",
                        "ranking/",
                        "histograms/",
                    ]
                    for key, value in metrics.items():
                        if any(key.startswith(cat) for cat in diagnostic_categories):
                            if key.startswith("histograms/"):
                                wandb_log[key] = wandb.Histogram(value)
                            else:
                                wandb_log[key] = value

                    if video_path is not None:
                        wandb_log["eval/video"] = wandb.Video(video_path, format="mp4")
                    if plot_path is not None:
                        plot_file = Path(plot_path)
                        if plot_file.exists() and plot_file.stat().st_size > 0:
                            try:
                                wandb_log["plots/qv_trajectories"] = wandb.Image(plot_path)
                            except Exception as e:
                                print(f"  Warning: Failed to load Q/V plot for wandb: {e}")
                        else:
                            print(f"  Warning: Q/V plot file missing or empty: {plot_path}")
                    wandb.log(wandb_log, step=step)

            # Save best model
            if is_best and cfg.training.save_best_model:
                best_checkpoint_path = checkpoint_manager.save_checkpoint(
                    policy=policy,
                    name="best_model",
                    step=step,
                    success_rate=best_success_rate,
                )
                print(f"✓ Saved best model (success rate: {best_success_rate:.1f}%)")

                # Upload to W&B
                if cfg.wandb.enabled:
                    metadata = create_artifact_metadata(
                        repo_id=",".join(repo_ids),
                        success_rate=best_success_rate / 100.0,
                        avg_reward=metrics["avg_reward"],
                        step=step,
                        is_best=True,
                        dataset_size=total_frames,
                        env=cfg.env.name,
                        robot=cfg.env.robot,
                        policy_type=policy_type,
                        state_dim=input_dim,
                        action_dim=action_dim,
                    )
                    artifact_name = f"{run_name}-best-step-{step}"
                    upload_checkpoint_to_wandb(
                        checkpoint_path=best_checkpoint_path,
                        artifact_name=artifact_name,
                        description=f"Best {policy_type} policy (success: {best_success_rate:.1f}%, step: {step})",
                        metadata=metadata,
                    )
                print()

        # Save periodic checkpoint
        if (
            cfg.training.periodic_checkpoint_freq > 0
            and step > 0
            and step % cfg.training.periodic_checkpoint_freq == 0
        ):
            checkpoint_manager.save_checkpoint(
                policy=policy,
                name=f"checkpoint_{step}",
                step=step,
            )
            print(f"✓ Saved checkpoint at step {step}")
            print()

        if auto_resume_enabled and should_save_resume_checkpoint(
            step=step, resume_checkpoint_freq=cfg.training.resume_checkpoint_freq
        ):
            # Commit resume progress only after all side effects for this step
            # have completed. In particular, eval boundaries must not advance
            # resume state before the eval metrics are logged, otherwise a
            # preemption during eval resumes at step+1 and leaves a permanent
            # hole in the learning curve.
            with record_function("train/save_resume_checkpoint"):
                save_resume_checkpoint(step, "step_complete")

        step += 1
    # The loop exits one past the last completed update.
    final_step = step - 1

    _progress(
        "finalize",
        cfg.training.training_steps,
        f"saving final model: {run_name}",
        status="running",
        metrics={"best_success_rate": float(best_success_rate)},
        outputs=[f"{run_dir}/"],
    )

    # Save final model and upload to W&B
    if cfg.training.save_final_model:
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            policy=policy,
            name="final_model",
            step=final_step,
        )
        print(f"✓ Saved final model to {final_checkpoint_path}")

        if freeze_actor:
            critic_value_actor_hash_after = policy.actor_hash()
            if critic_value_actor_hash_parent != critic_value_actor_hash_after:
                raise RuntimeError(
                    "critic/value-only training changed the frozen actor: "
                    f"before={critic_value_actor_hash_parent}, "
                    f"after={critic_value_actor_hash_after}"
                )
            critic_value_normalizer_hash_after = policy.normalizer_hash()
            if critic_value_normalizer_hash_parent != critic_value_normalizer_hash_after:
                raise RuntimeError(
                    "critic/value-only training changed the inherited parent normalizer: "
                    f"before={critic_value_normalizer_hash_parent}, "
                    f"after={critic_value_normalizer_hash_after}"
                )
            critic_value_summary = {
                "schema_version": 1,
                "parent_artifact": cfg.pretrained_artifact,
                "dataset_repo_ids": repo_ids,
                "dataset_revisions": {
                    repo_id: dataset_info[repo_id]["revision"] for repo_id in repo_ids
                },
                "training_seed": cfg.training.seed,
                "training_steps": cfg.training.training_steps,
                "update_components": cfg.training.update_components,
                "policy_type": policy_type,
                "prediction_horizon": prediction_horizon,
                "execution_horizon": execution_horizon,
                "actor_sha256_before": critic_value_actor_hash_parent,
                "actor_sha256_after_resume": critic_value_actor_hash_after_resume,
                "actor_sha256_after": critic_value_actor_hash_after,
                "normalizer_sha256_before": critic_value_normalizer_hash_parent,
                "normalizer_sha256_after": critic_value_normalizer_hash_after,
            }
            summary_path = final_checkpoint_path / "critic_value_summary.json"
            summary_path.write_text(
                json.dumps(critic_value_summary, indent=2, sort_keys=True) + "\n"
            )
            print(f"✓ Wrote frozen-actor critic/value summary to {summary_path}")

        if cfg.wandb.enabled:
            metadata = create_artifact_metadata(
                repo_id=",".join(repo_ids),
                success_rate=best_success_rate / 100.0 if best_success_rate > 0 else 0.0,
                avg_reward=0.0,
                step=final_step,
                is_best=False,
                dataset_size=total_frames,
                env=cfg.env.name,
                robot=cfg.env.robot,
                policy_type=policy_type,
                state_dim=input_dim,
                action_dim=action_dim,
            )
            artifact_name = f"{run_name}-final-step-{final_step}"
            upload_checkpoint_to_wandb(
                checkpoint_path=final_checkpoint_path,
                artifact_name=artifact_name,
                description=f"Final {policy_type} policy (step: {final_step})",
                metadata=metadata,
            )
            print("✓ Uploaded final model artifact to W&B")

    # Cleanup
    if eval_vec_env is not None:
        eval_vec_env.close()

    # Finish wandb
    if cfg.wandb.enabled:
        wandb.finish()
        print("✓ Finished Weights & Biases logging")

    # Mark completion only after the final artifact upload and W&B finalization,
    # so a crash in between leaves the run resumable.
    resume_manager.mark_completed()

    print("\nDone!")
    emit_completion(
        job_type="training",
        success=True,
        run_id=progress_run_id,
        current=cfg.training.training_steps,
        total=cfg.training.training_steps,
        message=f"training complete: {run_name}",
        metrics={"best_success_rate": float(best_success_rate)},
        outputs=[f"{run_dir}/"],
    )


def main():
    train()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit_completion(
            job_type="training",
            success=False,
            exit_code=1,
            message=f"training failed: {type(exc).__name__}: {exc}",
        )
        raise
