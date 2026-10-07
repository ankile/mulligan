#!/usr/bin/env python3
"""
IDQL (Implicit Diffusion Q-Learning) policy implementation.

This module implements a policy that learns values using IQL (Implicit Q-Learning)
and uses a Diffusion Policy as the actor. At inference time, it samples N actions
from the diffusion model and selects the one with the highest Q-value.

Key components:
- V-network: State value function learned via expectile regression
- Q-network: Action-value function learned via Q-learning with V as target
- Actor: Diffusion Policy trained with standard BC (diffusion denoising loss)
- Policy extraction: Sample N actions, select highest Q-value

Normalization strategy:
- During training: MIN_MAX normalization for diffusion actor (matches LeRobot's default)
- During inference: Converts from z-score (Q/V network format) to MIN_MAX (diffusion format)
- Q/V networks: Use z-score normalized states throughout

Reference:
- IQL: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022
- Diffusion Policy: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
"""

from __future__ import annotations

import copy
import hashlib
import itertools
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import record_function

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from mulligan.agents.iql_utils import (
    adaptive_tau,
    distributional_value_loss,
    expectile_loss,
    hl_gauss_target,
)
from mulligan.configs.policy import IDQLDIVLConfig, IDQLPolicyConfig
from mulligan.networks.distributional_v import DistributionalVNetwork
from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.training.normalization import Normalizer, zscore_to_actor_inputs
from mulligan.training.precision import autocast_context


class IDQLPolicy(nn.Module):
    """
    IDQL policy with diffusion actor, Q-network critics, and V-network.

    Combines:
    - Actor: DiffusionPolicy for action generation (trained with BC/diffusion loss)
    - Critics: Multiple Q-networks for action-value estimation (double Q-learning)
    - Value Network: V-network for state value estimation (IQL)
    - Target Networks: Delayed copies for stable learning

    Uses double-clipped Q-learning: takes the minimum over all Q-networks
    when computing Q-values to reduce overestimation bias.

    At inference, samples N actions from the diffusion model and selects
    the one with the highest Q-value (min over all critics).

    IMPORTANT - Normalizer Requirement:
        The normalizer is REQUIRED for inference (select_action)
        and training (update method). It must be provided either:
        1. At construction time via the `normalizer` argument, OR
        2. Before first use via `set_normalizer(normalizer)`

        The normalizer is used to convert between z-score normalization (used by Q/V networks)
        and MIN_MAX normalization (expected by the DiffusionPolicy actor).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor: DiffusionPolicy,
        config: IDQLPolicyConfig | None = None,
        clip_targets_to_range: tuple[float, float] | None = None,
        normalizer: Normalizer | None = None,
        robot_state_dim: int | None = None,
    ):
        """
        Initialize IDQL policy.

        Args:
            state_dim: Dimension of state observations
            action_dim: Dimension of action space
            actor: Pre-configured DiffusionPolicy from LeRobot
            config: Policy configuration (architecture + training hyperparameters).
                    Created with defaults if None.
            clip_targets_to_range: Optional (min, max) tuple for clipping Q-value targets
                                   to an empirical range. Computed from dataset statistics
                                   when config.clip_targets_to_range is True.
            normalizer: Normalizer with state statistics for z-score <-> MIN_MAX conversion.
                       REQUIRED for inference and training. If not provided here, must be set
                       via set_normalizer() before calling select_action() or update().
                       The normalizer must have state_min/state_max set for MIN_MAX conversion.
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or IDQLPolicyConfig()
        self.clip_targets_to_range = clip_targets_to_range
        self._normalizer = normalizer
        self.robot_state_dim = robot_state_dim
        self.env_state_dim = state_dim - robot_state_dim if robot_state_dim is not None else None
        # IDQLPolicy.chunk_size is the critic/execution horizon. The diffusion
        # actor may predict a longer horizon via the actor config's chunk_size.
        self.chunk_size = actor.config.n_action_steps

        if (
            self.robot_state_dim is not None
            and self.env_state_dim is not None
            and self.env_state_dim < 1
        ):
            raise ValueError("robot_state_dim must be smaller than state_dim")

        # Actor: DiffusionPolicy (passed in, already configured)
        self.actor: DiffusionPolicy = actor
        self._compiled_actor_forward = None

        # Extract architecture params from config
        hidden_dims = self.config.hidden_dims
        use_layer_norm = self.config.use_layer_norm

        # Critics: Multiple Q-networks for double Q-learning (take min over all).
        self.critics: nn.ModuleList = nn.ModuleList(
            [
                QNetwork(
                    state_dim=state_dim,
                    action_dim=action_dim * self.chunk_size,
                    hidden_dims=hidden_dims,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(self.config.num_q_networks)
            ]
        )

        # Value Network: V-network for state value estimation (IQL).
        # DIVL swaps the scalar V for a categorical distributional V over a
        # fixed action-value support; everything else (critic, actor, rerank)
        # is unchanged. The flag drives the two branched loss methods below.
        self._is_divl = isinstance(self.config, IDQLDIVLConfig)
        if self._is_divl:
            divl_config: IDQLDIVLConfig = self.config
            if divl_config.v_min is None or divl_config.v_max is None:
                raise ValueError(
                    "IDQLDIVLConfig.v_min/v_max must be set before constructing the "
                    "policy. The factory derives them from the empirical return range "
                    "(or a CLI override) and writes them back into the config."
                )
            self.value: VNetwork | DistributionalVNetwork = DistributionalVNetwork(
                state_dim=state_dim,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
                num_atoms=divl_config.num_atoms,
                v_min=divl_config.v_min,
                v_max=divl_config.v_max,
            )
        else:
            self.value = VNetwork(
                state_dim=state_dim,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
            )

        # Target Networks: Delayed copies for stable learning
        self.target_critics: nn.ModuleList[QNetwork] = nn.ModuleList(
            [copy.deepcopy(c) for c in self.critics]
        )
        self.target_value: VNetwork = copy.deepcopy(self.value)

        # Freeze target networks (no gradients)
        for target_critic in self.target_critics:
            for param in target_critic.parameters():
                param.requires_grad = False
        for param in self.target_value.parameters():
            param.requires_grad = False

        # Action queue for caching selected action chunks (like DiffusionPolicy)
        # Each element is one timestep: (batch_size, action_dim)
        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.chunk_size)

        # Cached Q/V values for the current action chunk (for visualization)
        # These are computed once per chunk and returned for each step within the chunk
        self._cached_q_value: torch.Tensor | None = None
        self._cached_v_value: torch.Tensor | None = None

        self._critic_value_only_training = False

    def configure_actor_compile(self, enabled: bool, mode: str = "default") -> None:
        """Compile the actor loss path without replacing the actor module.

        Compiling ``self.actor`` directly wraps it in an OptimizedModule and changes
        state-dict key names. Compiling the bound ``forward`` callable preserves the
        registered module and keeps checkpoints compatible with uncompiled runs.
        This intentionally bypasses module-level forward/pre-forward hooks on
        ``self.actor``; do not use this path if actor hooks become part of
        training semantics.
        """
        if not enabled:
            self._compiled_actor_forward = None
            return

        if self.actor._forward_hooks:
            raise RuntimeError(
                "Cannot compile actor.forward because actor forward hooks are registered. "
                "The checkpoint-safe compile path bypasses module-level hooks."
            )
        if self.actor._forward_pre_hooks:
            raise RuntimeError(
                "Cannot compile actor.forward because actor forward pre-hooks are registered. "
                "The checkpoint-safe compile path bypasses module-level hooks."
            )

        self._compiled_actor_forward = torch.compile(
            self.actor.forward,
            mode=mode,
            fullgraph=False,
        )

    def set_normalizer(self, normalizer: Normalizer) -> None:
        """
        Set the normalizer for z-score <-> MIN_MAX conversion.

        This method MUST be called before select_action() or update() if the
        normalizer was not provided in __init__.

        The normalizer is used to convert between:
        - Z-score normalization: used by Q/V networks (zero mean, unit variance)
        - MIN_MAX normalization: expected by DiffusionPolicy actor (values in [-1, 1])

        Args:
            normalizer: Normalizer with state statistics. Must have state_min and state_max
                       set (in addition to state_mean and state_std) for MIN_MAX conversion.

        Raises:
            ValueError: If normalizer.state_min or normalizer.state_max is None.

        Example:
            >>> policy = IDQLPolicy(state_dim=10, action_dim=7, actor=diffusion_policy)
            >>> normalizer = Normalizer(state_mean=..., state_std=..., state_min=..., state_max=...)
            >>> policy.set_normalizer(normalizer)  # Now policy is ready for inference
            >>> action = policy.select_action(obs)
        """
        if normalizer.state_min is None or normalizer.state_max is None:
            raise ValueError(
                "Normalizer must have state_min and state_max set for MIN_MAX conversion. "
                "Pass state_min and state_max when constructing the Normalizer."
            )
        self._normalizer = normalizer

    def get_optim_params(self) -> dict[str, Any]:
        """
        Return parameter groups for each optimizer.

        Returns:
            Dict mapping optimizer names to parameter iterators:
            - "actor": Actor (diffusion) network parameters
            - "critic": Combined parameters from all Q-networks
            - "value": Value network parameters
        """
        return {
            "actor": self.actor.parameters(),
            "critic": (p for p in self.critics.parameters() if p.requires_grad),
            "value": self.value.parameters(),
        }

    def set_critic_value_only_training(self) -> None:
        """Freeze the loaded actor and train only the freshly initialized Q/V modules."""
        self._critic_value_only_training = True
        self.actor.requires_grad_(False)
        self.actor.eval()
        self.critics.requires_grad_(True)
        self.value.requires_grad_(True)
        self.target_critics.requires_grad_(False)
        self.target_value.requires_grad_(False)
        self.target_critics.eval()
        self.target_value.eval()

    def train(self, mode: bool = True) -> IDQLPolicy:
        """Respect frozen-network contracts during partial-component training."""
        super().train(mode)
        if self._critic_value_only_training:
            self.actor.eval()
            self.target_critics.eval()
            self.target_value.eval()
        return self

    @staticmethod
    def _state_dict_sha256(state_dict: dict[str, Tensor]) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(state_dict.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def actor_hash(self) -> str:
        """Return a reproducible hash for the actor frozen in critic/value-only mode."""
        return self._state_dict_sha256(self.actor.state_dict())

    def normalizer_hash(self) -> str:
        """Hash every tensor that defines the actor/critic coordinate system."""
        if self._normalizer is None:
            raise ValueError("normalizer hash requires a configured normalizer")
        tensors = {
            "state_mean": self._normalizer.state_mean,
            "state_std": self._normalizer.state_std,
            "action_min": self._normalizer.action_min,
            "action_max": self._normalizer.action_max,
        }
        if self._normalizer.state_min is None or self._normalizer.state_max is None:
            raise ValueError("normalizer hash requires state_min and state_max")
        tensors["state_min"] = self._normalizer.state_min
        tensors["state_max"] = self._normalizer.state_max
        return self._state_dict_sha256(tensors)

    def _stack_critic_outputs(
        self, critics: nn.ModuleList, critic_state: Tensor, action: Tensor
    ) -> Tensor:
        """Run every critic and return Q-values shaped `(num_q_networks, B, 1)`."""
        return torch.stack([c(critic_state, action) for c in critics], dim=0)

    def _hl_gauss_sigma(self) -> float:
        """HL-Gauss smoothing sigma in value units (ratio * atom spacing).

        Only meaningful for DIVL (distributional V); derived from the config's
        ``hl_gauss_sigma_ratio`` and the distributional V-network's atom support.
        """
        atom_width = (self.value.v_max - self.value.v_min) / (self.value.num_atoms - 1)
        return self.config.hl_gauss_sigma_ratio * atom_width

    def compute_loss_value(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Compute IQL value network loss using expectile regression.

        The value network is trained to predict the expectile of the Q-function,
        which approximates the maximum Q-value without explicit maximization.

        Args:
            batch: Dict with "observation.state" and "action" tensors

        Returns:
            Dict with "loss_value" and auxiliary metrics
        """
        state = batch["observation.state"]
        action = batch["action"]
        critic_state = batch.get("critic_state", state)

        with torch.no_grad():
            # Stack Q-values from all critics and aggregate
            q_all = self._stack_critic_outputs(self.critics, critic_state, action)
            q_values = self._aggregate_q_values(q_all)  # (batch_size, 1)

        if self._is_divl:
            # DIVL: fit a categorical distribution over the value support toward
            # the HL-Gauss projection of the (no-grad) scalar Q target (min over critics).
            logits = self.value(state)  # (B, num_atoms)
            soft_labels = hl_gauss_target(q_values, self.value.atoms, self._hl_gauss_sigma())
            value_loss = distributional_value_loss(logits, soft_labels).mean()
            with torch.no_grad():
                v_expected = self.value.expected_value(state)  # (B, 1), for logging
            diff = q_values - v_expected
            return {
                "loss_value": value_loss,
                "v_mean": v_expected.mean().detach(),
                "v_std": v_expected.std().detach(),
                "q_data_mean": q_values.mean().detach(),
                "advantage_mean": diff.mean().detach(),
            }

        # Scalar IQL: L_V = expectile_loss(Q(s,a) - V(s))
        v_values = self.value(state)
        diff = q_values - v_values
        value_loss = expectile_loss(diff, expectile=self.config.expectile).mean()

        return {
            "loss_value": value_loss,
            "v_mean": v_values.mean().detach(),
            "v_std": v_values.std().detach(),
            "q_data_mean": q_values.mean().detach(),
            "advantage_mean": diff.mean().detach(),
        }

    def compute_loss_critic(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Compute IQL critic loss using Q-learning with target V.

        The critic is trained with standard TD learning, using the target
        value network to compute bootstrap targets.

        For chunked actions, uses the QC paper approach:
            Q(s, chunk) = cumulative_reward + gamma^k * masks * V(next_state)
        where k = chunk_size, and next_state is the state after executing all actions.

        Args:
            batch: Dict with:
                - observation.state: current state
                - action: action chunk (B, chunk_size, action_dim)
                - reward: cumulative discounted reward over chunk
                - next.observation.state: next state (chunk-aligned, i.e., s_{t+chunk_size})
                - masks: bootstrap decision (1 = bootstrap, 0 = don't)
                - chunk_valid: 0 if chunk crosses episode boundary, 1 otherwise

        Returns:
            Dict with "loss_critic" and auxiliary metrics
        """
        state = batch.get("critic_state", batch["observation.state"])
        action = batch["action"]  # Flattened: (B, chunk_size * action_dim)
        reward = batch["reward"]
        next_state = batch["next.observation.state"]
        masks = batch["masks"]  # Bootstrap decision (1 = bootstrap, 0 = don't)
        chunk_valid = batch["chunk_valid"]  # For weighting loss (always present)

        # Use stored chunk_size (action is flattened, so can't derive from shape)
        chunk_size = self.chunk_size

        # Compute effective discount factor
        effective_gamma = self.config.gamma**chunk_size

        # Compute target Q-values
        with torch.no_grad():
            if self._is_divl:
                # DIVL: bootstrap with a tau-quantile of the (distributional)
                # target value, with optional entropy-adaptive tau (lower /
                # less-optimistic tau on high-entropy, uncertain states).
                cfg: IDQLDIVLConfig = self.config
                norm_entropy = self.target_value.normalized_entropy(next_state)
                tau = adaptive_tau(
                    norm_entropy,
                    tau_base=cfg.tau_base,
                    tau_min=cfg.tau_min,
                    tau_max=cfg.tau_max,
                    alpha=cfg.tau_entropy_alpha,
                )
                target_v = self.target_value.quantile(next_state, tau)
            else:
                target_v = self.target_value(next_state)

            # Optionally clip targets to empirical range
            if self.clip_targets_to_range is not None:
                min_val, max_val = self.clip_targets_to_range
                target_v = target_v.clamp(min=min_val, max=max_val)

            # TD target: reward + gamma^k * masks * V(next_state)
            # masks = 1 means bootstrap, masks = 0 means don't bootstrap
            # reward, masks are already (B, 1) from prepare_chunked_data
            y = reward + effective_gamma * masks.float() * target_v

        # Compute current Q-values from all critics and train each with the same target.
        q_stack = self._stack_critic_outputs(self.critics, state, action)
        q_all = list(q_stack.unbind(dim=0))

        # Critic loss weight from chunk validity; chunk_valid is already (B, 1) from
        # prepare_chunked_data. Terminal chunks (valid=0, mask=0) are included since they
        # don't require bootstrapping: the Q-target is just cumulative reward, so padded
        # actions don't affect correctness. Weight = 1 if valid=1 OR mask=0 (terminal),
        # 0 only if valid=0 AND mask=1 (truncated): weight = valid + (1 - valid) * (1 - masks)
        critic_weight = chunk_valid + (1.0 - chunk_valid) * (1.0 - masks)

        # Per-critic MSE losses are summed over the ensemble.
        per_critic_losses = [
            (F.mse_loss(q, y, reduction="none") * critic_weight.float()).mean() for q in q_all
        ]
        critic_loss = sum(per_critic_losses)

        # Compute TD error for logging (uses configured aggregation)
        with torch.no_grad():
            q_agg = self._aggregate_q_values(torch.stack(q_all, dim=0))
            td_error = (q_agg - y).abs().mean()

        result = {
            "loss_critic": critic_loss,
            "q_mean": q_agg.mean().detach(),
            "q_std": q_agg.std().detach(),
            "target_mean": y.mean().detach(),
            "td_error": td_error,
            "effective_gamma": effective_gamma,
            "chunk_size": chunk_size,
            # Diagnostic metrics for debugging Q-value issues
            "reward_mean": reward.mean().detach(),
            "masks_mean": masks.mean().detach(),
            "target_v_mean": target_v.mean().detach(),
            "bootstrap_term_mean": (effective_gamma * masks.float() * target_v).mean().detach(),
            # Chunk validity metrics
            "chunk_valid_mean": chunk_valid.mean().detach(),
            "critic_weight_mean": critic_weight.mean().detach(),
        }
        return result

    def compute_loss_actor(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Compute diffusion actor loss (pure BC via diffusion denoising).

        The actor is trained with standard diffusion BC loss - no Q-gradient.
        Policy extraction happens at inference via N-sample Q-max selection.

        IMPORTANT: The batch must have MIN_MAX normalized states (matching LeRobot's
        DiffusionPolicy default). The update() method handles converting from z-score
        (as provided by IDQLPreprocessor) to MIN_MAX before calling this method.

        Args:
            batch: Dict with MIN_MAX normalized observations and actions:
                   - observation.state: MIN_MAX normalized robot state (B, T, D)
                   - observation.environment_state: Raw (unnormalized) env state (B, T, D)
                   - action: MIN_MAX normalized action (B, T, D)
                   Must have temporal dimension for diffusion policy.

        Returns:
            Dict with "loss_actor" and auxiliary metrics
        """
        # Call diffusion policy's forward which computes BC loss
        # The diffusion policy expects (B, T, D) format
        actor_forward = self._compiled_actor_forward or self.actor
        loss, _ = actor_forward(batch)

        return {
            "loss_actor": loss,
        }

    def _prepare_diffusion_actor_batch(
        self, actor_source_batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Convert IDQL z-score observations to the actor's normalization."""
        if self._normalizer is None:
            raise ValueError(
                "Normalizer not set. Required for z-score <-> MIN_MAX conversion during training."
            )

        robot_state_zscore = actor_source_batch["observation.state"]
        has_temporal = robot_state_zscore.dim() == 3
        if has_temporal:
            actor_batch_size, n_obs_steps = robot_state_zscore.shape[:2]
            robot_state_zscore = robot_state_zscore.reshape(-1, robot_state_zscore.shape[-1])

        env_state_zscore = actor_source_batch.get("observation.environment_state")
        if env_state_zscore is not None and has_temporal:
            env_state_zscore = env_state_zscore.reshape(-1, env_state_zscore.shape[-1])
        robot_state_minmax, env_state_raw = zscore_to_actor_inputs(
            self._normalizer, robot_state_zscore, env_state_zscore
        )
        if has_temporal:
            robot_state_minmax = robot_state_minmax.reshape(actor_batch_size, n_obs_steps, -1)

        actor_batch = {
            "observation.state": robot_state_minmax,
            "action": actor_source_batch["action"],
        }
        if env_state_raw is not None:
            if has_temporal:
                env_state_raw = env_state_raw.reshape(actor_batch_size, n_obs_steps, -1)
            actor_batch["observation.environment_state"] = env_state_raw
        if "action_is_pad" in actor_source_batch:
            actor_batch["action_is_pad"] = actor_source_batch["action_is_pad"]
        return actor_batch

    def _apply_actor_update(
        self,
        actor_source_batch: dict[str, Tensor],
        *,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
        amp_dtype: str,
    ) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        with record_function("idql/actor_batch_conversion"):
            actor_batch = self._prepare_diffusion_actor_batch(actor_source_batch)

        with record_function("idql/actor_loss_forward"), autocast_context(device, amp_dtype):
            actor_loss = self.compute_loss_actor(actor_batch)["loss_actor"]

        with record_function("idql/actor_backward_step"):
            optimizer.zero_grad()
            actor_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_grad_norm if max_grad_norm > 0 else float("inf"),
            )
            optimizer.step()

        return {
            "losses/actor": actor_loss.detach(),
            "gradients/actor_norm": grad_norm,
        }

    def reset(self) -> None:
        """
        Reset policy state (clears action queue, cached values, and resets diffusion policy).

        This method MUST be called on env.reset() to clear the cached action chunk.
        Similar to DiffusionPolicy.reset(), this ensures we don't use stale actions
        from a previous episode.
        """
        self.actor.reset()
        self._action_queue.clear()
        self._cached_q_value = None
        self._cached_v_value = None

    def _aggregate_q_values(self, q_stacked: torch.Tensor) -> torch.Tensor:
        """Minimum over the critic ensemble (dim 0): clipped double-Q."""
        return q_stacked.min(dim=0).values

    def _flatten_trailing_dims(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse `(B, T, D)` into `(B, T * D)`; pass through 2D inputs unchanged.

        Used for both state tensors and action chunks where the middle (time /
        chunk) dimension needs to be merged into the feature dimension.
        """
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        return x

    def _build_flat_state(
        self, robot_state: torch.Tensor, env_state: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Build the flat state representation used by the value network and the critics."""
        robot_state_flat = self._flatten_trailing_dims(robot_state)
        if env_state is None:
            return robot_state_flat
        env_state_flat = self._flatten_trailing_dims(env_state)
        return torch.cat([robot_state_flat, env_state_flat], dim=-1)

    def _expand_critic_state(self, state: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Repeat critic state inputs across sampled actions."""
        return state.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, state.shape[-1])

    def _sample_and_evaluate_actions(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.device]:
        """
        Shared logic for sampling N actions and computing Q-values.

        This helper method contains the common normalization, sampling, and Q-computation
        logic of select_action().

        Args:
            obs: Observation dict with keys:
                - "observation.state": z-score normalized robot state (batch_size, robot_dim)
                - "observation.environment_state": z-score normalized env state (batch_size, env_dim)
                  (REQUIRED - IDQL needs both robot and environment state)

        Returns:
            tuple of:
                - sampled_actions: Tensor of shape (N, batch_size, action_dim)
                - q_values: Tensor of shape (N, batch_size) - min Q over all critics
                - batch_size: int
                - device: torch.device
        """
        # Get robot state (required)
        robot_state_zscore = obs["observation.state"]  # (B, robot_dim)
        batch_size = robot_state_zscore.shape[0]
        device = robot_state_zscore.device
        num_samples = self.config.num_action_samples

        # Environment state is REQUIRED for IDQL policy: the critics consume the
        # concatenated [robot_state, env_state].
        if "observation.environment_state" not in obs:
            raise ValueError(
                "observation.environment_state is required for IDQL policy. "
                "The Q/V networks operate on concatenated [robot_state, env_state]. "
                "Ensure your preprocessor provides both observation keys."
            )
        env_state_zscore = obs["observation.environment_state"]  # (B, env_dim)

        state_for_q = self._build_flat_state(robot_state_zscore, env_state_zscore)

        # Convert from z-score to MIN_MAX normalization for diffusion actor
        if self._normalizer is None:
            raise ValueError(
                "Normalizer not set. Call set_normalizer() or pass normalizer in constructor."
            )

        # Robot state: z-score -> raw -> MIN_MAX; env state: z-score -> raw (LeRobot's
        # diffusion normalizer passes ENV features through unchanged).
        robot_state_minmax, env_state_raw = zscore_to_actor_inputs(
            self._normalizer, robot_state_zscore, env_state_zscore
        )

        # Diffusion model expects (B, n_obs_steps, dim) - add temporal dimension
        # and expand for N samples: (B, 1, D) -> (B*N, 1, D)
        diffusion_obs = {
            "observation.state": robot_state_minmax.unsqueeze(1).repeat_interleave(
                num_samples, dim=0
            ),
            "observation.environment_state": env_state_raw.unsqueeze(1).repeat_interleave(
                num_samples, dim=0
            ),
        }

        # Generate actions for all samples in one forward pass
        with torch.no_grad():
            # Shape: (B*N, n_action_steps, action_dim)
            all_actions = self.actor.diffusion.generate_actions(diffusion_obs)

        # Reshape to (N, B, action_dim) for Q-value comparison
        sampled_actions = all_actions.reshape(batch_size, num_samples, -1).permute(1, 0, 2)

        # Compute Q-values for all sampled actions
        # Expand state to match sampled actions
        state_expanded = self._expand_critic_state(state_for_q, num_samples)

        # Reshape for Q-network: (N * batch_size, state_dim) and (N * batch_size, action_dim)
        actions_flat = sampled_actions.reshape(
            num_samples * batch_size, self.chunk_size * self.action_dim
        )

        # Compute Q-values from all critics and aggregate
        with torch.no_grad():
            q_all = self._stack_critic_outputs(
                self.critics, state_expanded, actions_flat
            )  # (num_q_networks, N * batch_size, 1)
            q_values = self._aggregate_q_values(q_all)  # (N * batch_size, 1)

        # Reshape Q-values: (N, batch_size)
        q_values = q_values.reshape(num_samples, batch_size)

        return sampled_actions, q_values, batch_size, device

    def _select_action_indices(
        self, q_values: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Pick the highest-Q of the N sampled candidates per batch element (index into axis 0)."""
        return q_values.argmax(dim=0)  # (batch_size,)

    def select_action(
        self, obs: dict, return_values: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given observation dict using N-sample Q-max selection with caching.

        This method implements the same caching pattern as DiffusionPolicy.select_action:
        - When the action queue is empty: sample N action chunks, score with Q-values,
          select the chunk with highest Q, and cache it in the queue
        - Pop and return one action (single timestep) from the queue per call

        This allows IDQL to amortize the expensive diffusion sampling + Q-scoring
        over chunk_size timesteps, just like DiffusionPolicy amortizes generation.

        Normalization handling:
        - Input: z-score normalized states (separate keys from evaluation preprocessing)
        - For Q-network: Uses either flat or split robot/env critic inputs
        - For diffusion actor: Converts to MIN_MAX normalized (LeRobot's DiffusionPolicy default)

        Args:
            obs: Observation dict with keys:
                - "observation.state": z-score normalized robot state (batch_size, robot_dim)
                - "observation.environment_state": z-score normalized env state (batch_size, env_dim)
                  (REQUIRED - IDQL needs both robot and environment state for Q/V networks)
            return_values: If True, also return cached Q and V values for visualization.
                          The Q/V values are computed once per chunk and repeated for each step.

        Returns:
            If return_values=False:
                Action tensor of shape (batch_size, action_dim) - a single timestep
            If return_values=True:
                Tuple of (action, q_value, v_value) where:
                - action: (batch_size, action_dim)
                - q_value: (batch_size,) - Q-value of the selected action chunk
                - v_value: (batch_size,) - V-value at the state when chunk was sampled
        """
        # If action queue is empty, sample new actions and select best chunk
        if len(self._action_queue) == 0:
            sampled_actions, q_values, batch_size, device = self._sample_and_evaluate_actions(obs)
            # sampled_actions: (N, B, chunk_size * action_dim)
            # q_values: (N, B)

            # Select the action chunk (argmax, or top-k sampling)
            best_indices = self._select_action_indices(q_values, batch_size, device)

            # Cache the Q-value of the selected action for each batch element
            # Shape: (batch_size,)
            self._cached_q_value = q_values[best_indices, torch.arange(batch_size, device=device)]

            # Compute and cache V-value at the current state
            # We need the flat concatenated state for the V-network
            robot_state = obs["observation.state"]
            env_state = obs["observation.environment_state"]
            state_for_v = self._build_flat_state(robot_state, env_state)
            with torch.no_grad():
                # Route through compute_v_value, the single entry point for
                # V estimates (kept overridable for future variants).
                self._cached_v_value = self.compute_v_value(state_for_v).squeeze(
                    -1
                )  # (batch_size,)

            # Gather best actions: (B, chunk_size * action_dim)
            best_actions_flat = sampled_actions[
                best_indices, torch.arange(batch_size, device=device)
            ]

            # Reshape to (B, chunk_size, action_dim)
            best_actions = best_actions_flat.reshape(batch_size, self.chunk_size, self.action_dim)

            # Transpose to (chunk_size, B, action_dim) and extend queue
            # Each element in the queue is one timestep: (B, action_dim)
            self._action_queue.extend(best_actions.transpose(0, 1))

        # Pop and return one action
        action = self._action_queue.popleft()

        if return_values:
            return action, self._cached_q_value, self._cached_v_value

        return action

    def compute_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        env_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Q-value from critics (aggregated over all Q-networks).

        Args:
            state: Flat concatenated state, or the robot state when ``env_state`` is given.
            action: Action tensor of shape (batch_size, action_dim) or (batch_size, T, action_dim)
            env_state: Optional env-state tensor, concatenated after ``state``.

        Returns:
            Q-value tensor of shape (batch_size, 1) - aggregated over all critics
        """
        critic_state = self._build_flat_state(state, env_state)
        action = self._flatten_trailing_dims(action)
        q_all = self._stack_critic_outputs(self.critics, critic_state, action)
        return self._aggregate_q_values(q_all)

    def compute_v_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute V-value from value network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            V-value tensor of shape (batch_size, 1)
        """
        if self._is_divl:
            # DIVL's V is distributional; the scalar readout for viz/logging is
            # the expected value. (V is never used at inference for reranking —
            # select_action scores on scalar Q only.)
            return self.value.expected_value(state)
        return self.value(state)

    def update_targets(self):
        """
        Update target networks using Polyak averaging.

        target_params = tau * params + (1 - tau) * target_params
        """
        tau = self.config.tau

        with torch.no_grad():
            # Update all target critics
            for critic, target_critic in zip(self.critics, self.target_critics):
                for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                    target_param.lerp_(param, tau)

            # Update target value network
            for param, target_param in zip(self.value.parameters(), self.target_value.parameters()):
                target_param.lerp_(param, tau)

    def update(
        self,
        batch: dict[str, Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        step: int = 0,
        max_grad_norms: dict[str, float] | None = None,
        policy_batch: dict[str, Tensor] | None = None,
        amp_dtype: str = "none",
        update_components: str = "all",
    ) -> dict[str, Tensor | float]:
        """
        Perform a full training update on the batch.

        This method encapsulates the entire training step for IDQL:
        1. Value network update (IQL expectile regression) - uses `batch`
        2. Critic update (Q-learning with target V) - uses `batch`
        3. Actor update (Diffusion BC loss) - uses `policy_batch` if provided, else `batch`
        4. Target network update (Polyak averaging)

        Args:
            batch: Preprocessed critic batch dict (for V/Q updates) with:
                - observation.state: Normalized robot state tensor (B, T, D) or (B, D)
                - observation.environment_state: (optional) Normalized env state (B, T, D) or (B, D)
                - action: Normalized action tensor
                - reward, done: RL training data
                - next.observation.state: Normalized next robot state tensor
                - next.observation.environment_state: (optional) Normalized next env state
            optimizers: Dict of optimizers with keys "actor", "critic", "value"
            step: Current training step (unused, kept for interface compatibility)
            max_grad_norms: Optional dict of max gradient norms per component.
                           Use 0 or negative to disable clipping.
            policy_batch: Optional separate batch for actor/BC updates. If None, uses `batch`.
                         This enables straddled sampling where critic and actor train on different data.
            amp_dtype: CUDA autocast dtype for forward/loss computation.
            update_components: "all", or "critic_value_only" to keep the loaded actor
                frozen (call set_critic_value_only_training() first).

        Returns:
            Dict containing all metrics ready for logging (wandb-compatible keys)
        """
        if max_grad_norms is None:
            max_grad_norms = {"actor": 0, "critic": 0, "value": 0}

        if update_components == "critic_value_only":
            if not self._critic_value_only_training:
                raise RuntimeError(
                    "critic_value_only update requested before set_critic_value_only_training()"
                )
            if set(optimizers) != {"critic", "value"}:
                raise ValueError(
                    "critic_value_only update requires only critic/value optimizers, "
                    f"got {sorted(optimizers)}"
                )
        elif update_components != "all":
            raise ValueError(f"unknown update_components {update_components!r}")

        # =====================================================================
        # Prepare batches for IQL (need flat concatenated state) vs Diffusion
        # =====================================================================
        # For Q/V networks, we need flat concatenated state: (B, state_dim)
        # For diffusion, we pass the original batch (with temporal dimension)

        # Get state components
        robot_state = batch["observation.state"]
        action = batch["action"]
        env_state = batch.get("observation.environment_state")

        # Value network still consumes the flat concatenated state
        state_flat = self._build_flat_state(robot_state, env_state)

        critic_state = self._build_flat_state(robot_state, env_state)

        # Flatten action temporal dimension into the action dimension
        action_flat = self._flatten_trailing_dims(action)

        # Create batch for Q/V networks with flat concatenated tensors
        iql_batch = {
            "observation.state": state_flat,
            "action": action_flat,
            "critic_state": critic_state,
            "reward": batch["reward"],
            "masks": batch["masks"],  # Bootstrap decision (1 = bootstrap, 0 = don't)
        }
        if "chunk_valid" in batch:
            iql_batch["chunk_valid"] = batch["chunk_valid"]

        # Handle next state similarly - concatenate robot + env state
        if "next.observation.state" in batch:
            next_robot_state = batch["next.observation.state"]
            next_env_state = batch.get("next.observation.environment_state")
            iql_batch["next.observation.state"] = self._build_flat_state(
                next_robot_state, next_env_state
            )

        # =====================================================================
        # Value network update (IQL expectile regression)
        # =====================================================================
        device = next(self.parameters()).device

        with record_function("idql/value_loss_forward"), autocast_context(device, amp_dtype):
            value_loss_dict = self.compute_loss_value(iql_batch)
            value_loss = value_loss_dict["loss_value"]

        with record_function("idql/value_backward_step"):
            optimizers["value"].zero_grad()
            value_loss.backward()

            max_norm = max_grad_norms.get("value", 0)
            if max_norm > 0:
                grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm)
            else:
                grad_norm_value = torch.nn.utils.clip_grad_norm_(
                    self.value.parameters(), float("inf")
                )
            optimizers["value"].step()

        # =====================================================================
        # Critic update (Q-learning with target V)
        # =====================================================================
        with record_function("idql/critic_loss_forward"), autocast_context(device, amp_dtype):
            critic_loss_dict = self.compute_loss_critic(iql_batch)
            critic_loss = critic_loss_dict["loss_critic"]

        with record_function("idql/critic_backward_step"):
            optimizers["critic"].zero_grad()
            critic_loss.backward()

            # Chain parameters from all critics for gradient clipping
            all_critic_params = list(itertools.chain(*[c.parameters() for c in self.critics]))
            max_norm = max_grad_norms.get("critic", 0)
            if max_norm > 0:
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(all_critic_params, max_norm)
            else:
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(all_critic_params, float("inf"))
            optimizers["critic"].step()

        actor_update: dict[str, Tensor | float] = {}
        if update_components == "all":
            actor_source_batch = policy_batch if policy_batch is not None else batch
            actor_update = self._apply_actor_update(
                actor_source_batch,
                optimizer=optimizers["actor"],
                max_grad_norm=max_grad_norms.get("actor", 0),
                amp_dtype=amp_dtype,
            )

        # =====================================================================
        # Target network update
        # =====================================================================
        with record_function("idql/update_targets"):
            self.update_targets()

        # =====================================================================
        # Build result dict with wandb-ready keys
        # =====================================================================
        result = {
            # === LOSSES ===
            "losses/value": value_loss.detach(),
            "losses/critic": critic_loss.detach(),
            # === Q-VALUES ===
            "q_values/q_mean": critic_loss_dict["q_mean"],
            "q_values/q_std": critic_loss_dict["q_std"],
            "q_values/target_mean": critic_loss_dict["target_mean"],
            # === V-VALUES ===
            "v_values/v_mean": value_loss_dict["v_mean"],
            "v_values/v_std": value_loss_dict["v_std"],
            "v_values/target_v_mean": critic_loss_dict["target_v_mean"],
            # === TD ERRORS ===
            "td_errors/mean": critic_loss_dict["td_error"],
            # === GRADIENTS ===
            "gradients/critic_norm": grad_norm_critic,
            "gradients/value_norm": grad_norm_value,
            # === DIAGNOSTIC METRICS (for debugging Q-value issues) ===
            "debug/effective_gamma": critic_loss_dict["effective_gamma"],
            "debug/chunk_size": critic_loss_dict["chunk_size"],
            "debug/reward_mean": critic_loss_dict["reward_mean"],
            "debug/masks_mean": critic_loss_dict["masks_mean"],
            "debug/bootstrap_term_mean": critic_loss_dict["bootstrap_term_mean"],
            "critic_weight_mean": critic_loss_dict["critic_weight_mean"],
        }
        if update_components == "all":
            result["losses/actor"] = actor_update["losses/actor"]
            result["gradients/actor_norm"] = actor_update["gradients/actor_norm"]
        return result

    def load_actor_checkpoint(self, path: Path, *, strict_contract: bool = True) -> None:
        """Load only a pretrained IDQL actor, allowing a different child Q/V topology."""
        checkpoint_path = path / "policy.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"IDQL checkpoint is missing {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {"actor_state_dict", "actor_config", "state_dim", "action_dim", "config"}
        missing = required - checkpoint.keys()
        if missing:
            raise KeyError(f"IDQL checkpoint {checkpoint_path} is missing keys {sorted(missing)}")
        if int(checkpoint["state_dim"]) != self.state_dim:
            raise ValueError(
                f"parent state_dim={checkpoint['state_dim']} does not match current {self.state_dim}"
            )
        if int(checkpoint["action_dim"]) != self.action_dim:
            raise ValueError(
                f"parent action_dim={checkpoint['action_dim']} does not match current {self.action_dim}"
            )
        if strict_contract:
            parent_config = checkpoint["config"]
            actor_contract_fields = (
                "chunk_size",
                "n_action_steps",
                "down_dims",
                "num_train_timesteps",
                "num_inference_steps",
            )
            for field in actor_contract_fields:
                parent_value = getattr(parent_config, field)
                current_value = getattr(self.config, field)
                if parent_value != current_value:
                    raise ValueError(
                        f"parent actor config {field}={parent_value!r} does not match "
                        f"current {current_value!r}"
                    )
            parent_actor = checkpoint["actor_config"]
            if parent_actor.horizon != self.actor.config.horizon:
                raise ValueError("parent actor prediction horizon does not match current actor")
            if parent_actor.n_action_steps != self.actor.config.n_action_steps:
                raise ValueError("parent actor execution horizon does not match current actor")
        self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)

    def load_complete_checkpoint(self, path: Path, *, strict_contract: bool = True) -> None:
        """Load every learned component from an IDQL checkpoint into this policy."""
        checkpoint_path = path / "policy.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"IDQL checkpoint is missing {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "actor_state_dict",
            "actor_config",
            "critics_state_dicts",
            "value_state_dict",
            "target_critics_state_dicts",
            "target_value_state_dict",
            "state_dim",
            "action_dim",
            "config",
        }
        missing = required - checkpoint.keys()
        if missing:
            raise KeyError(f"IDQL checkpoint {checkpoint_path} is missing keys {sorted(missing)}")
        if int(checkpoint["state_dim"]) != self.state_dim:
            raise ValueError(
                f"parent state_dim={checkpoint['state_dim']} does not match current {self.state_dim}"
            )
        if int(checkpoint["action_dim"]) != self.action_dim:
            raise ValueError(
                f"parent action_dim={checkpoint['action_dim']} does not match current {self.action_dim}"
            )
        if len(checkpoint["critics_state_dicts"]) != len(self.critics):
            raise ValueError(
                f"parent has {len(checkpoint['critics_state_dicts'])} critic modules, "
                f"current policy has {len(self.critics)}"
            )
        if len(checkpoint["target_critics_state_dicts"]) != len(self.target_critics):
            raise ValueError("parent target-critic module count does not match current policy")

        if strict_contract:
            parent_config = checkpoint["config"]
            contract_fields = (
                "hidden_dims",
                "use_layer_norm",
                "num_q_networks",
                "chunk_size",
                "n_action_steps",
                "down_dims",
                "num_train_timesteps",
                "num_inference_steps",
            )
            for field in contract_fields:
                parent_value = getattr(parent_config, field)
                current_value = getattr(self.config, field)
                if parent_value != current_value:
                    raise ValueError(
                        f"parent config {field}={parent_value!r} does not match "
                        f"current {current_value!r}"
                    )
            parent_actor = checkpoint["actor_config"]
            if parent_actor.horizon != self.actor.config.horizon:
                raise ValueError("parent actor prediction horizon does not match current actor")
            if parent_actor.n_action_steps != self.actor.config.n_action_steps:
                raise ValueError("parent actor execution horizon does not match current actor")

        if self._is_divl:
            # load_state_dict would silently replace the child's atoms buffer, leaving the
            # config and the HL-Gauss sigma on a support the network does not use.
            parent_config = checkpoint["config"]
            parent_atoms = checkpoint["value_state_dict"].get("atoms")
            if parent_atoms is None or not isinstance(parent_config, IDQLDIVLConfig):
                raise ValueError("a DIVL policy can load only a DIVL parent in full")
            if not torch.equal(parent_atoms.cpu(), self.value.atoms.cpu()):
                raise ValueError(
                    "parent DIVL value support "
                    f"[{parent_config.v_min!r}, {parent_config.v_max!r}] x {parent_config.num_atoms} "
                    f"does not match current [{self.config.v_min!r}, {self.config.v_max!r}] x "
                    f"{self.config.num_atoms}; pass --policy.v_min={parent_config.v_min!r} "
                    f"--policy.v_max={parent_config.v_max!r} "
                    f"--policy.num_atoms={parent_config.num_atoms}"
                )

        self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        self.value.load_state_dict(checkpoint["value_state_dict"], strict=True)
        self.target_value.load_state_dict(checkpoint["target_value_state_dict"], strict=True)
        for module, state_dict in zip(self.critics, checkpoint["critics_state_dicts"], strict=True):
            module.load_state_dict(state_dict, strict=True)
        for module, state_dict in zip(
            self.target_critics, checkpoint["target_critics_state_dicts"], strict=True
        ):
            module.load_state_dict(state_dict, strict=True)

    def save(self, path: Path):
        """
        Save full IDQL checkpoint.

        Saves:
        - Actor (diffusion) state dict
        - Actor config (for reconstruction during load)
        - Critics state dicts (list for multiple Q-networks)
        - Value network state dict
        - Target networks state dicts
        - Model hyperparameters
        - Config
        """
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_config": self.actor.config,  # Save diffusion actor config for reconstruction
                "critics_state_dicts": [c.state_dict() for c in self.critics],
                "value_state_dict": self.value.state_dict(),
                "target_critics_state_dicts": [c.state_dict() for c in self.target_critics],
                "target_value_state_dict": self.target_value.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "robot_state_dim": self.robot_state_dim,
                "config": self.config,
            },
            path / "policy.pt",
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        device: str = "cpu",
        config: IDQLPolicyConfig | None = None,
        actor: nn.Module | None = None,
    ) -> IDQLPolicy:
        """Build a policy from a loaded ``policy.pt`` payload (see :meth:`save`).

        ``actor`` replaces the actor rebuilt from the pickled ``actor_config``; its
        weights are still loaded from the checkpoint.
        """
        from lerobot.policies.factory import get_policy_class

        required = {
            "actor_state_dict",
            "actor_config",
            "critics_state_dicts",
            "value_state_dict",
            "target_critics_state_dicts",
            "target_value_state_dict",
            "state_dim",
            "action_dim",
            "config",
        }
        missing = required - checkpoint.keys()
        if missing:
            raise KeyError(f"IDQL checkpoint is missing keys {sorted(missing)}")

        if actor is None:
            actor_config = checkpoint["actor_config"]
            actor = get_policy_class(actor_config.type)(actor_config)
        model = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            actor=actor,
            config=config if config is not None else checkpoint["config"],
            robot_state_dim=checkpoint.get("robot_state_dim"),
        )
        model.actor.load_state_dict(checkpoint["actor_state_dict"])
        model.value.load_state_dict(checkpoint["value_state_dict"])
        model.target_value.load_state_dict(checkpoint["target_value_state_dict"])
        for critic, state_dict in zip(
            model.critics, checkpoint["critics_state_dicts"], strict=True
        ):
            critic.load_state_dict(state_dict)
        for critic, state_dict in zip(
            model.target_critics, checkpoint["target_critics_state_dicts"], strict=True
        ):
            critic.load_state_dict(state_dict)
        return model.to(device)

    @classmethod
    def load(
        cls,
        path: Path,
        actor: nn.Module | None = None,
        device: str = "cpu",
        config: IDQLPolicyConfig | None = None,
    ) -> IDQLPolicy:
        """Load ``path / "policy.pt"``; ``actor`` / ``config`` override the saved ones.

        The argument order is the original ``IDQLPolicy.load(path, actor, device, config)``."""
        checkpoint = torch.load(Path(path) / "policy.pt", map_location="cpu", weights_only=False)
        return cls.from_checkpoint(checkpoint, device=device, config=config, actor=actor)
