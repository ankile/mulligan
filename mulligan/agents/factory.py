#!/usr/bin/env python3
"""
Policy factory for creating policies with unified interface.

This module centralizes all policy creation logic, eliminating scattered
conditionals in train.py and enabling clean policy-agnostic training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import torch

from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import get_policy_class, make_policy_config

from mulligan.agents.idql import IDQLPolicy
from mulligan.agents.processors import make_idql_pre_post_processors
from mulligan.release.hub import resolve_checkpoint
from mulligan.training.checkpoint_utils import load_normalization_stats
from mulligan.training.normalization import Normalizer

if TYPE_CHECKING:
    from mulligan.configs.policy import IDQLPolicyConfig, PolicyConfig


@dataclass
class NormalizationStats:
    """Container for normalization statistics."""

    # State statistics (z-score)
    state_mean: torch.Tensor
    state_std: torch.Tensor
    robot_state_mean: torch.Tensor
    robot_state_std: torch.Tensor
    env_state_mean: torch.Tensor
    env_state_std: torch.Tensor

    # State min/max (for MIN_MAX normalization)
    state_min: torch.Tensor
    state_max: torch.Tensor

    # Action statistics
    action_min: torch.Tensor
    action_max: torch.Tensor

    # Dimensions
    state_dim: int
    env_state_dim: int
    action_dim: int

    @property
    def input_dim(self) -> int:
        """Total input dimension (robot + environment state)."""
        return self.state_dim + self.env_state_dim


@dataclass
class PolicyComponents:
    """Container for all policy-related components."""

    policy: IDQLPolicy
    preprocessor: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    postprocessor: Callable[[torch.Tensor], torch.Tensor]
    normalizer: Normalizer
    policy_type: str
    hidden_dims: list[int]

    # Metadata for logging/checkpointing
    num_params: int
    num_actor_params: int
    num_critic_params: int | None
    num_value_params: int | None


def extract_normalization_stats(
    stats: dict[str, dict[str, Any]], device: str = "cpu"
) -> NormalizationStats:
    """
    Extract normalization statistics from dataset metadata.

    Args:
        stats: Dataset statistics dict from LeRobotDatasetMetadata.stats
        device: Device to place tensors on

    Returns:
        NormalizationStats container with all statistics
    """
    available_keys = list(stats.keys())

    def get_stats(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        if key not in stats:
            raise ValueError(
                f"Statistics for '{key}' not found in dataset metadata. "
                f"Available keys: {available_keys}"
            )
        return (
            torch.from_numpy(stats[key]["mean"]).float().to(device),
            torch.from_numpy(stats[key]["std"]).float().clamp(min=1e-6).to(device),
        )

    def get_stats_minmax(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        if key not in stats:
            raise ValueError(
                f"Statistics for '{key}' not found in dataset metadata. "
                f"Available keys: {available_keys}"
            )
        return (
            torch.from_numpy(stats[key]["min"]).float().to(device),
            torch.from_numpy(stats[key]["max"]).float().to(device),
        )

    # Extract statistics
    robot_state_mean, robot_state_std = get_stats("observation.state")
    env_state_mean, env_state_std = get_stats("observation.environment_state")
    state_mean = torch.cat([robot_state_mean, env_state_mean])
    state_std = torch.cat([robot_state_std, env_state_std])

    robot_state_min, robot_state_max = get_stats_minmax("observation.state")
    env_state_min, env_state_max = get_stats_minmax("observation.environment_state")
    state_min = torch.cat([robot_state_min, env_state_min])
    state_max = torch.cat([robot_state_max, env_state_max])

    if "action" not in stats:
        raise ValueError(
            f"Statistics for 'action' not found in dataset metadata. "
            f"Available keys: {available_keys}"
        )
    action_min = torch.from_numpy(stats["action"]["min"]).float().to(device)
    action_max = torch.from_numpy(stats["action"]["max"]).float().to(device)

    return NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        robot_state_mean=robot_state_mean,
        robot_state_std=robot_state_std,
        env_state_mean=env_state_mean,
        env_state_std=env_state_std,
        state_min=state_min,
        state_max=state_max,
        action_min=action_min,
        action_max=action_max,
        state_dim=robot_state_mean.shape[0],
        env_state_dim=env_state_mean.shape[0],
        action_dim=action_min.shape[0],
    )


def create_policy(
    policy_type: str,
    norm_stats: NormalizationStats,
    dataset_features: dict[str, Any],
    device: str,
    cfg_policy: PolicyConfig,
    target_value_clip_range: tuple[float, float] | None = None,
    value_support_range: tuple[float, float] | None = None,
    pretrained_artifact: str | None = None,
    pretrained_load_components: str = "all",
) -> PolicyComponents:
    """
    Create a policy with its normalizer and processors.

    Args:
        policy_type: "idql" or "idql_divl"
        norm_stats: Normalization statistics from dataset
        dataset_features: Dataset features dict from LeRobotDatasetMetadata.features
        device: Device to place policy on
        cfg_policy: Policy configuration from TrainConfig
        target_value_clip_range: Optional clip range for critic targets
        value_support_range: Optional (min, max) empirical return range used to
            derive the DIVL distributional value support when v_min/v_max are
            unset in the config. Ignored by non-DIVL policies.
        pretrained_artifact: Optional pretrained checkpoint (local directory, ``hf://`` URI
            or W&B artifact; see :func:`mulligan.release.hub.resolve_checkpoint`)
        pretrained_load_components: For IDQL parents, either ``"all"`` or ``"actor"``.
            Actor-only loading supports frozen-actor critic/value training when the child
            value-head topology differs from the parent.

    Returns:
        PolicyComponents containing policy and all related components
    """
    # Import here to avoid circular imports at module level
    from mulligan.configs.policy import IDQLPolicyConfig

    if policy_type in ("idql", "idql_divl"):
        # IDQLDIVLConfig inherits from IDQLPolicyConfig: only the value network differs.
        return _create_idql_policy(
            norm_stats=norm_stats,
            dataset_features=dataset_features,
            device=device,
            cfg_policy=cast(IDQLPolicyConfig, cfg_policy),
            target_value_clip_range=target_value_clip_range,
            value_support_range=value_support_range,
            policy_type=policy_type,
            pretrained_artifact=pretrained_artifact,
            pretrained_load_components=pretrained_load_components,
        )
    raise ValueError(f"Unknown policy_type {policy_type!r}; expected idql or idql_divl")


def _resolve_divl_value_support(
    cfg_policy: IDQLPolicyConfig,
    value_support_range: tuple[float, float] | None,
) -> None:
    """Resolve DIVL's distributional value support in-place on ``cfg_policy``.

    If ``v_min``/``v_max`` are already set (CLI override), they are kept. Otherwise
    they are derived from the empirical return range (``value_support_range``)
    with a small symmetric margin so the support brackets observed returns, then
    written back into the config so the checkpoint reconstructs the same support
    at eval time.
    """
    from mulligan.configs.policy import IDQLDIVLConfig

    cfg_policy = cast(IDQLDIVLConfig, cfg_policy)
    if cfg_policy.v_min is not None and cfg_policy.v_max is not None:
        print(
            f"  DIVL value support (config override): "
            f"[{cfg_policy.v_min:.3f}, {cfg_policy.v_max:.3f}], num_atoms={cfg_policy.num_atoms}"
        )
        return

    if value_support_range is None:
        raise ValueError(
            "DIVL requires a value support range: set --policy.v_min/--policy.v_max "
            "explicitly, or run via the training pipeline which derives it from the "
            "empirical return range."
        )

    lo, hi = float(value_support_range[0]), float(value_support_range[1])
    if not hi > lo:
        raise ValueError(f"value_support_range must have max > min, got ({lo}, {hi})")
    # Small symmetric margin (5% of the span) so the outermost atoms bracket the
    # observed returns rather than sitting exactly on them.
    margin = 0.05 * (hi - lo)
    cfg_policy.v_min = lo - margin
    cfg_policy.v_max = hi + margin
    print(
        f"  DIVL value support (derived from returns [{lo:.3f}, {hi:.3f}] + 5% margin): "
        f"[{cfg_policy.v_min:.3f}, {cfg_policy.v_max:.3f}], num_atoms={cfg_policy.num_atoms}"
    )


def _load_parent_normalizer(
    checkpoint_path: Path,
    current_stats: NormalizationStats,
    *,
    device: str,
) -> Normalizer:
    """Load and validate the coordinate system paired with a full parent checkpoint."""
    parent = load_normalization_stats(
        checkpoint_path,
        device=device,
        require_state_minmax=True,
    )
    expected_state_shape = current_stats.state_mean.shape
    expected_action_shape = current_stats.action_min.shape
    if parent.state_mean.shape != expected_state_shape:
        raise ValueError(
            "parent normalizer state shape "
            f"{tuple(parent.state_mean.shape)} does not match current "
            f"{tuple(expected_state_shape)}"
        )
    if parent.action_min.shape != expected_action_shape:
        raise ValueError(
            "parent normalizer action shape "
            f"{tuple(parent.action_min.shape)} does not match current "
            f"{tuple(expected_action_shape)}"
        )
    if not torch.equal(parent.action_min, current_stats.action_min.to(device)):
        raise ValueError("parent action_min does not match the current dataset contract")
    if not torch.equal(parent.action_max, current_stats.action_max.to(device)):
        raise ValueError("parent action_max does not match the current dataset contract")
    if parent.state_min is None or parent.state_max is None:
        raise ValueError("parent IDQL normalizer is missing state_min/state_max")

    current_state_tensors = (
        current_stats.state_mean.to(device),
        current_stats.state_std.to(device),
        current_stats.state_min.to(device),
        current_stats.state_max.to(device),
    )
    parent_state_tensors = (
        parent.state_mean,
        parent.state_std,
        parent.state_min,
        parent.state_max,
    )
    state_stats_match = all(
        torch.equal(parent_value, current_value)
        for parent_value, current_value in zip(
            parent_state_tensors, current_state_tensors, strict=True
        )
    )
    if state_stats_match:
        print("  Parent normalizer exactly matches current dataset statistics")
    else:
        print(
            "  WARNING: current dataset state statistics differ from the parent; "
            "inheriting parent stats so frozen actor/Q coordinates remain valid"
        )
    return parent


def _create_idql_policy(
    norm_stats: NormalizationStats,
    dataset_features: dict[str, Any],
    device: str,
    cfg_policy: IDQLPolicyConfig,
    target_value_clip_range: tuple[float, float] | None,
    value_support_range: tuple[float, float] | None = None,
    policy_type: str = "idql",
    pretrained_artifact: str | None = None,
    pretrained_load_components: str = "all",
) -> PolicyComponents:
    """Create an IDQL or DIVL policy (Diffusion actor + Q/V networks).

    ``policy_type='idql_divl'`` keeps IDQLPolicy but the IDQLDIVLConfig drives
    a distributional value network; we derive its value support from the
    empirical return range here when v_min/v_max are unset. See
    ``DistributionalVNetwork`` for the behavioral details.
    """
    if policy_type not in ("idql", "idql_divl"):
        raise ValueError(f"_create_idql_policy got unexpected policy_type={policy_type!r}")
    if pretrained_load_components not in ("all", "actor"):
        raise ValueError(
            "pretrained_load_components must be 'all' or 'actor', "
            f"got {pretrained_load_components!r}"
        )
    if pretrained_load_components == "actor" and pretrained_artifact is None:
        raise ValueError("actor-only pretrained loading requires pretrained_artifact")

    # DIVL: derive the distributional value support from the empirical return
    # range when not explicitly set, and write it back into the config so the
    # checkpoint reconstructs the same support at eval time.
    if policy_type == "idql_divl":
        _resolve_divl_value_support(cfg_policy, value_support_range)

    input_dim = norm_stats.input_dim
    action_dim = norm_stats.action_dim
    hidden_dims = cfg_policy.hidden_dims

    pretrained_path = None
    if pretrained_artifact is not None:
        pretrained_path = resolve_checkpoint(pretrained_artifact)

    # A full pretrained checkpoint is inseparable from its normalizer. Building
    # processors from newly aggregated dataset stats and then loading frozen
    # weights changes the function represented by both actor and critic.
    if pretrained_path is not None:
        normalizer = _load_parent_normalizer(
            pretrained_path,
            norm_stats,
            device=device,
        )
    else:
        normalizer = Normalizer(
            state_mean=norm_stats.state_mean,
            state_std=norm_stats.state_std,
            action_min=norm_stats.action_min,
            action_max=norm_stats.action_max,
            device=device,
            state_min=norm_stats.state_min,
            state_max=norm_stats.state_max,
        )

    # Create preprocessor/postprocessor
    preprocessor, postprocessor = make_idql_pre_post_processors(
        normalizer=normalizer,
        device=device,
    )

    # Convert dataset features to policy features
    features = dataset_to_policy_features(dataset_features)
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    input_features = {k: v for k, v in input_features.items() if v.type != FeatureType.VISUAL}

    # Build diffusion policy kwargs
    policy_kwargs: dict[str, Any] = {
        "input_features": input_features,
        "output_features": output_features,
        "device": device,
    }

    # Chunk size handling for diffusion.
    # `horizon` is what the actor predicts; `n_action_steps` is what we execute per query
    # (and what IDQL uses as its chunk_size for critic input shape). When n_action_steps
    # is unset, we default to chunk_size so the chunk and the executed steps coincide.
    if cfg_policy.chunk_size:
        n_action_steps = getattr(cfg_policy, "n_action_steps", None) or cfg_policy.chunk_size
        policy_kwargs["horizon"] = cfg_policy.chunk_size
        policy_kwargs["n_action_steps"] = n_action_steps
        policy_kwargs["down_dims"] = (512,)
        policy_kwargs["drop_n_last_frames"] = 0
        policy_kwargs["do_mask_loss_for_padding"] = True
        print(
            "  Diffusion actor horizons: "
            f"horizon={cfg_policy.chunk_size}, n_action_steps={n_action_steps}"
        )

    # Apply diffusion-specific overrides
    if hasattr(cfg_policy, "get_diffusion_overrides"):
        overrides = cfg_policy.get_diffusion_overrides()
        for k, v in overrides.items():
            policy_kwargs[k] = v

    # Create diffusion actor
    lerobot_config = make_policy_config("diffusion", **policy_kwargs)
    diffusion_policy = get_policy_class("diffusion")(lerobot_config)
    diffusion_policy.to(device)

    policy = IDQLPolicy(
        state_dim=input_dim,
        action_dim=action_dim,
        actor=diffusion_policy,
        config=cfg_policy,
        clip_targets_to_range=target_value_clip_range if cfg_policy.clip_targets_to_range else None,
        normalizer=normalizer,
    )
    policy.to(device)

    if pretrained_path is not None:
        if pretrained_load_components == "actor":
            policy.load_actor_checkpoint(pretrained_path, strict_contract=True)
            print(f"  Loaded frozen actor and parent normalizer from {pretrained_artifact}")
        else:
            policy.load_complete_checkpoint(pretrained_path, strict_contract=True)
            print(
                f"  Loaded complete IDQL checkpoint and parent normalizer "
                f"from {pretrained_artifact}"
            )

    # Compute parameter counts (num_value_params=None when there is no V-network).
    num_params = sum(p.numel() for p in policy.parameters())
    num_actor_params = sum(p.numel() for p in policy.actor.parameters())
    num_critic_params = sum(p.numel() for c in policy.critics for p in c.parameters())
    num_value_params = sum(p.numel() for p in policy.value.parameters())

    return PolicyComponents(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        normalizer=normalizer,
        policy_type=policy_type,
        hidden_dims=hidden_dims,
        num_params=num_params,
        num_actor_params=num_actor_params,
        num_critic_params=num_critic_params,
        num_value_params=num_value_params,
    )


def print_policy_info(components: PolicyComponents) -> None:
    """Print policy creation summary."""
    print(f"✓ {components.policy_type} policy created with {components.num_params:,} parameters")
    print(f"  Actor: {components.num_actor_params:,} parameters")
    if components.num_critic_params is not None:
        print(f"  Critic: {components.num_critic_params:,} parameters")
    if components.num_value_params is not None:
        print(f"  Value Network: {components.num_value_params:,} parameters")
    print()
