#!/usr/bin/env python3
"""
Checkpoint saving for the training loop.

Writes ``policy.pt`` (via ``policy.save``), ``stats.json`` and ``metadata.json``: the
format of the released sim checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mulligan.training.checkpoint_utils import (
    save_checkpoint_metadata,
    save_normalization_stats,
)

if TYPE_CHECKING:
    from mulligan.configs.train import TrainConfig


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""

    checkpoint_dir: Path
    env_name: str
    robot_name: str
    dataset_repo_ids: list[str]
    state_dim: int
    action_dim: int
    hidden_dims: list[int]
    use_layer_norm: bool
    dataset_size: int


class CheckpointManager:
    """Saves IDQL/DIVL checkpoints with their normalizer and metadata."""

    def __init__(
        self,
        config: CheckpointConfig,
        policy_type: str,
        normalizer_stats: dict[str, torch.Tensor],
    ):
        """
        Args:
            config: Checkpoint configuration
            policy_type: "idql" or "idql_divl"
            normalizer_stats: Dict with keys: state_mean, state_std, action_min, action_max,
                             state_min, state_max
        """
        if policy_type not in ("idql", "idql_divl"):
            raise ValueError(f"unsupported policy_type {policy_type!r}")
        self.config = config
        self.policy_type = policy_type
        self.normalizer_stats = normalizer_stats

    def save_checkpoint(
        self,
        policy: torch.nn.Module,
        name: str,
        step: int,
        success_rate: float | None = None,
    ) -> Path:
        """
        Save a checkpoint with all necessary metadata.

        Args:
            policy: Policy to save
            name: Checkpoint name (e.g., "best_model", "checkpoint_1000", "final_model")
            step: Current training step
            success_rate: Optional success rate for metadata

        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_path = self.config.checkpoint_dir / name
        policy.save(checkpoint_path)
        save_normalization_stats(
            checkpoint_path,
            state_mean=self.normalizer_stats["state_mean"],
            state_std=self.normalizer_stats["state_std"],
            action_min=self.normalizer_stats["action_min"],
            action_max=self.normalizer_stats["action_max"],
            state_min=self.normalizer_stats["state_min"],
            state_max=self.normalizer_stats["state_max"],
        )
        metadata_kwargs: dict[str, Any] = {
            "env_name": self.config.env_name,
            "robot_name": self.config.robot_name,
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "hidden_dims": self.config.hidden_dims,
            "use_layer_norm": self.config.use_layer_norm,
            "dataset_repo_id": ",".join(self.config.dataset_repo_ids),
            # Both IDQL variants are "idql"; the pickled config class
            # (IDQLPolicyConfig / IDQLDIVLConfig) tells them apart.
            "policy_type": "idql",
            "step": step,
        }
        if success_rate is not None:
            metadata_kwargs["success_rate"] = success_rate
        save_checkpoint_metadata(checkpoint_path, **metadata_kwargs)
        return checkpoint_path


def create_checkpoint_manager(
    checkpoint_dir: Path,
    cfg: TrainConfig,
    policy_type: str,
    normalizer_stats: dict[str, torch.Tensor],
    dataset_size: int,
    hidden_dims: list[int],
) -> CheckpointManager:
    """Create a CheckpointManager from the training config (runtime dims must be set)."""
    if cfg.state_dim is None or cfg.action_dim is None:
        raise ValueError("cfg.state_dim and cfg.action_dim must be populated before checkpointing")
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        env_name=cfg.env.name,
        robot_name=cfg.env.robot,
        dataset_repo_ids=cfg.dataset.get_repo_id_list(),
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_dims=hidden_dims,
        use_layer_norm=cfg.policy.use_layer_norm,
        dataset_size=dataset_size,
    )
    return CheckpointManager(
        config=config, policy_type=policy_type, normalizer_stats=normalizer_stats
    )
