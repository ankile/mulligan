#!/usr/bin/env python3
"""
Batch preparation for the IDQL/DIVL training loop.

Splits the concatenated state into robot and environment state, applies the
preprocessor, and adds the observation-step dimension the diffusion actor expects.
"""

from __future__ import annotations

from typing import Any, Callable, Final

import torch

# Keys that get a temporal dimension (B, D) -> (B, 1, D)
TEMPORAL_KEYS: Final[frozenset[str]] = frozenset(
    [
        "observation.state",
        "observation.environment_state",
        "action",
        "next.observation.state",
        "next.observation.environment_state",
    ]
)

# Type alias for batch dictionaries (values can be tensors, bools, or other types)
BatchDict = dict[str, Any]
PreprocessorFn = Callable[[dict[str, Any]], dict[str, Any]]


class BatchProcessor:
    """Builds critic and actor batches for IDQL/DIVL."""

    def __init__(
        self,
        preprocessor: PreprocessorFn,
        state_dim: int,
        device: str | torch.device,
    ):
        """
        Args:
            preprocessor: Preprocessor function for normalization
            state_dim: Dimension of robot state (for splitting concatenated state)
            device: Device for tensors
        """
        self.preprocessor = preprocessor
        self.state_dim = state_dim
        self.device = device

    def _base_batch(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: torch.Tensor,
        reward: torch.Tensor | None,
        masks: torch.Tensor | None,
        source: torch.Tensor | None,
        success: torch.Tensor | None,
        chunk_valid: torch.Tensor | None,
    ) -> BatchDict:
        batch = {
            "observation.state": state[:, : self.state_dim],
            "observation.environment_state": state[:, self.state_dim :],
            "action": action,
            "action_is_pad": action_is_pad,
            "next.observation.state": next_state[:, : self.state_dim],
            "next.observation.environment_state": next_state[:, self.state_dim :],
        }
        for key, value in (
            ("reward", reward),
            ("masks", masks),
            ("source", source),
            ("success", success),
            ("chunk_valid", chunk_valid),
        ):
            if value is not None:
                batch[key] = value
        return batch

    def _finish(self, batch: BatchDict) -> BatchDict:
        batch = self.preprocessor(batch)
        for key in batch:
            if (
                key in TEMPORAL_KEYS
                and isinstance(batch[key], torch.Tensor)
                and batch[key].ndim == 2
            ):
                batch[key] = batch[key].unsqueeze(1)
        return batch

    def prepare_critic_batch(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: torch.Tensor,
        reward: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        success: torch.Tensor | None = None,
        chunk_valid: torch.Tensor | None = None,
    ) -> BatchDict:
        """
        Prepare batch for critic/value network updates.

        Args:
            state: Concatenated state tensor (batch, state_dim + env_state_dim)
            next_state: Concatenated next state tensor
            action: Action tensor (may have temporal dim from chunking)
            action_is_pad: Action padding mask
            reward, masks, source, success, chunk_valid: TD-learning keys

        Returns:
            Preprocessed batch dict ready for policy.update()
        """
        batch = self._base_batch(
            state, next_state, action, action_is_pad, reward, masks, source, success, chunk_valid
        )
        return self._finish(batch)

    def prepare_policy_batch(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: torch.Tensor,
        reward: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        success: torch.Tensor | None = None,
        chunk_valid: torch.Tensor | None = None,
    ) -> BatchDict:
        """Prepare batch for actor/BC updates (may come from different data than the critic batch)."""
        batch = self._base_batch(
            state, next_state, action, action_is_pad, reward, masks, source, success, chunk_valid
        )
        return self._finish(batch)


def create_batch_processor(
    preprocessor: PreprocessorFn,
    state_dim: int,
    policy_type: str,
    device: str | torch.device,
) -> BatchProcessor:
    """Create the BatchProcessor for ``policy_type`` ("idql" or "idql_divl")."""
    if policy_type not in ("idql", "idql_divl"):
        raise ValueError(f"unsupported policy_type {policy_type!r}")
    return BatchProcessor(
        preprocessor=preprocessor,
        state_dim=state_dim,
        device=device,
    )
