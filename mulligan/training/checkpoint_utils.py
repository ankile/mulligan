"""
Utilities for saving and loading agent checkpoints with normalization statistics.

This module provides functions to save/load complete BC checkpoints including:
- Policy weights (policy.pt)
- Normalization statistics (stats.json)
- Metadata for environment configuration (metadata.json)
"""

import json
from pathlib import Path
from typing import Optional

import torch


def save_normalization_stats(
    checkpoint_path: Path,
    normalizer=None,
    state_mean=None,
    state_std=None,
    action_min=None,
    action_max=None,
    state_min=None,
    state_max=None,
):
    """
    Save normalization statistics to checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory
        normalizer: Normalizer instance with state/action statistics (optional if individual tensors provided)
        state_mean: State mean tensor (optional if normalizer provided)
        state_std: State std tensor (optional if normalizer provided)
        action_min: Action min tensor (optional if normalizer provided)
        action_max: Action max tensor (optional if normalizer provided)
        state_min: State min tensor for MIN_MAX normalization (optional, required for IDQL)
        state_max: State max tensor for MIN_MAX normalization (optional, required for IDQL)

    Either normalizer OR (state_mean, state_std, action_min, action_max) must be provided.
    For IDQL policies, state_min and state_max should also be provided.
    """
    if normalizer is not None:
        # Use normalizer object
        state_mean = normalizer.state_mean
        state_std = normalizer.state_std
        action_min = normalizer.action_min
        action_max = normalizer.action_max
        # Also extract state_min/state_max if available
        if (
            state_min is None
            and hasattr(normalizer, "state_min")
            and normalizer.state_min is not None
        ):
            state_min = normalizer.state_min
        if (
            state_max is None
            and hasattr(normalizer, "state_max")
            and normalizer.state_max is not None
        ):
            state_max = normalizer.state_max
    elif any(x is None for x in [state_mean, state_std, action_min, action_max]):
        raise ValueError(
            "Either normalizer OR all of (state_mean, state_std, action_min, action_max) must be provided"
        )

    stats = {
        "state_mean": state_mean.cpu().numpy().tolist(),
        "state_std": state_std.cpu().numpy().tolist(),
        "action_min": action_min.cpu().numpy().tolist(),
        "action_max": action_max.cpu().numpy().tolist(),
    }

    # Add state_min/state_max if provided (required for IDQL MIN_MAX conversion)
    if state_min is not None:
        stats["state_min"] = state_min.cpu().numpy().tolist()
    if state_max is not None:
        stats["state_max"] = state_max.cpu().numpy().tolist()

    stats_path = checkpoint_path / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(
    checkpoint_path: Path,
    device: str = "cpu",
    *,
    require_state_minmax: bool = False,
):
    """
    Load normalization statistics from checkpoint directory and create Normalizer.

    Every released checkpoint (IDQL, DIVL) stores action_min/max in stats.json.

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load tensors to
        require_state_minmax: Fail unless state_min/state_max are present (required by IDQL).

    Returns:
        Normalizer instance with loaded statistics
    """
    from mulligan.training.normalization import Normalizer

    stats_path = checkpoint_path / "stats.json"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found at {stats_path}. "
            "Make sure the checkpoint was saved with save_normalization_stats()."
        )

    with open(stats_path, "r") as f:
        stats = json.load(f)

    required_fields = ["state_mean", "state_std", "action_min", "action_max"]
    if require_state_minmax:
        required_fields.extend(["state_min", "state_max"])
    for field in required_fields:
        if field not in stats:
            raise ValueError(
                f"Checkpoint stats.json is missing required field '{field}'. "
                "Checkpoints saved by mulligan.training.train include it."
            )

    tensors = {
        field: torch.tensor(stats[field], dtype=torch.float32, device=device)
        for field in required_fields
    }
    if (tensors["state_std"] <= 0).any():
        raise ValueError("Checkpoint state_std must be strictly positive")
    state_min = (
        torch.tensor(stats["state_min"], dtype=torch.float32, device=device)
        if "state_min" in stats
        else None
    )
    state_max = (
        torch.tensor(stats["state_max"], dtype=torch.float32, device=device)
        if "state_max" in stats
        else None
    )

    return Normalizer(
        state_mean=tensors["state_mean"],
        state_std=tensors["state_std"],
        state_min=state_min,
        state_max=state_max,
        action_min=tensors["action_min"],
        action_max=tensors["action_max"],
        device=device,
    )


def save_checkpoint_metadata(
    checkpoint_path: Path,
    env_name: str,
    robot_name: str,
    state_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    use_layer_norm: bool,
    dataset_repo_id: str,
    policy_type: str,
    success_rate: Optional[float] = None,
    step: Optional[int] = None,
):
    """
    Save checkpoint metadata for automatic environment configuration.

    Args:
        checkpoint_path: Path to checkpoint directory
        env_name: Environment name (e.g., "Lift")
        robot_name: Robot name (e.g., "Panda")
        state_dim: State observation dimension
        action_dim: Action dimension
        hidden_dims: MLP hidden layer dimensions
        use_layer_norm: Whether layer normalization was used
        dataset_repo_id: Dataset repository ID
        policy_type: Policy family written to metadata.json ("idql")
        success_rate: Evaluation success rate (optional)
        step: Training step (optional)
    """
    metadata = {
        "policy_type": policy_type,
        "env_name": env_name,
        "robot_name": robot_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "use_layer_norm": use_layer_norm,
        "dataset_repo_id": dataset_repo_id,
    }

    if success_rate is not None:
        metadata["success_rate"] = success_rate
    if step is not None:
        metadata["step"] = step

    metadata_path = checkpoint_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """
    Load checkpoint metadata.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dictionary with checkpoint metadata
    """
    metadata_path = checkpoint_path / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Checkpoint metadata not found at {metadata_path}. "
            "Make sure the checkpoint was saved with save_checkpoint_metadata()."
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata
