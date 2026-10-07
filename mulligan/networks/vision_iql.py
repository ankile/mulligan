#!/usr/bin/env python3
"""
Unified VisionIQL module for end-to-end IQL training with visual encoders.

Owns the encoder, Q1, Q2, V, and target encoder/Q critics. DDP wraps the whole module
so encoder gradients are properly averaged across ranks.
"""

import copy
import math
from itertools import chain

import torch
import torch.nn as nn

from mulligan.agents.iql_utils import (
    adaptive_tau,
    distributional_value_loss,
    expectile_loss,
    hl_gauss_target,
)
from mulligan.networks.distributional_v import DistributionalVNetwork
from mulligan.networks.action_film import apply_action_film


def truncated_td_lambda_weights(
    horizon_valid: torch.Tensor,
    td_lambda: float,
) -> torch.Tensor:
    """Return per-row geometric TD(lambda) weights with mass on the last valid horizon."""
    if horizon_valid.ndim != 2 or horizon_valid.shape[1] < 1:
        raise ValueError(
            "horizon_valid must have shape (batch, num_horizons) with num_horizons >= 1"
        )
    if horizon_valid.dtype != torch.bool:
        raise ValueError(f"horizon_valid must be bool, got {horizon_valid.dtype}")
    if not 0.0 <= td_lambda <= 1.0:
        raise ValueError(f"td_lambda must be in [0, 1], got {td_lambda}")
    if not bool(horizon_valid[:, 0].all()):
        raise ValueError("every TD(lambda) row must have its shortest horizon valid")
    # Valid horizons must form a prefix; otherwise the final-mass rule is ambiguous.
    if bool(((~horizon_valid[:, :-1]) & horizon_valid[:, 1:]).any()):
        raise ValueError("TD(lambda) valid horizons must form a contiguous prefix")

    n_horizons = horizon_valid.shape[1]
    exponent = torch.arange(n_horizons, device=horizon_valid.device)
    lam = horizon_valid.new_tensor(td_lambda, dtype=torch.float32)
    weights = (1.0 - lam) * lam.pow(exponent)[None, :]
    weights = weights.expand(horizon_valid.shape[0], -1).clone()
    valid_count = horizon_valid.sum(dim=1)
    last_index = valid_count - 1
    weights.scatter_(1, last_index[:, None], lam.pow(last_index.float())[:, None])
    return weights * horizon_valid.to(weights.dtype)


def build_truncated_td_lambda_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discount_powers: torch.Tensor,
    bootstrap_values: torch.Tensor,
    horizons: torch.Tensor,
    horizon_valid: torch.Tensor,
    *,
    gamma: float,
    td_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the exact deterministic mixture of all valid chunk-aligned TD targets."""
    if rewards.ndim != 2 or dones.shape != rewards.shape:
        raise ValueError(
            f"rewards/dones must have identical 2-D shapes, got {rewards.shape}/{dones.shape}"
        )
    if horizons.ndim != 1 or horizons.numel() != horizon_valid.shape[1]:
        raise ValueError("horizons must be 1-D and match horizon_valid's second dimension")
    if horizon_valid.shape[0] != rewards.shape[0]:
        raise ValueError("horizon_valid batch dimension must match rewards")
    if bootstrap_values.shape == (*horizon_valid.shape, 1):
        bootstrap_values = bootstrap_values.squeeze(-1)
    if bootstrap_values.shape != horizon_valid.shape:
        raise ValueError(
            f"bootstrap_values must have shape {tuple(horizon_valid.shape)} or that shape + (1,)"
        )
    if horizons.is_floating_point():
        raise ValueError(f"horizons must contain integer step counts, got {horizons.dtype}")
    horizons = horizons.to(device=rewards.device, dtype=torch.long)
    if bool((horizons < 1).any()) or bool((horizons[1:] <= horizons[:-1]).any()):
        raise ValueError("horizons must be positive and strictly increasing")
    if int(horizons[-1]) > rewards.shape[1]:
        raise ValueError(
            f"maximum horizon {int(horizons[-1])} exceeds reward width {rewards.shape[1]}"
        )
    if discount_powers.numel() != rewards.shape[1]:
        raise ValueError(
            f"discount_powers has {discount_powers.numel()} entries for reward width "
            f"{rewards.shape[1]}"
        )

    pre_done = dones.float().cumsum(dim=1).roll(1, 1)
    pre_done[:, 0] = 0
    reward_mask = (pre_done == 0).to(rewards.dtype)
    discounted_prefix = (rewards.float() * discount_powers.reshape(1, -1) * reward_mask).cumsum(
        dim=1
    )
    gather_index = (horizons - 1)[None, :].expand(rewards.shape[0], -1)
    horizon_rewards = discounted_prefix.gather(1, gather_index)
    horizon_done = dones.float().cumsum(dim=1).gather(1, gather_index).clamp(max=1.0)
    bootstrap_discount = gamma ** horizons.to(dtype=rewards.dtype)
    targets = horizon_rewards + (
        bootstrap_discount[None, :] * (1.0 - horizon_done) * bootstrap_values
    )
    weights = truncated_td_lambda_weights(horizon_valid.to(rewards.device), td_lambda).to(
        rewards.dtype
    )
    mixed_target = (targets * weights).sum(dim=1, keepdim=True)
    return mixed_target, targets, weights


class VisionIQL(nn.Module):
    """Unified IQL module: encoder + Q1/Q2 + V with target Q critics.

    forward() computes all losses internally and returns a dict with
    'total' (for .backward()) plus detached metrics for logging.

    When ``distributional`` is True (DIVL), the scalar V is replaced by a
    categorical ``DistributionalVNetwork`` (the caller is responsible for
    passing such a ``v_net``). The double-Q scalar critic is unchanged; only
    the V loss (HL-Gauss cross-entropy) and the Q TD bootstrap (a tau-quantile
    of the online distribution, with optional entropy-adaptive tau) differ.
    With distributional=False the same target-Q/online-V topology uses scalar
    expectile regression.
    """

    def __init__(
        self,
        encoder: nn.Module,
        q1: nn.Module,
        q2: nn.Module,
        v_net: nn.Module,
        camera_keys: list[str],
        separate_encoders: bool,
        expectile: float,
        gamma: float,
        tau: float,
        clip_targets_min: float | None = None,
        clip_targets_max: float | None = None,
        distributional: bool = False,
        hl_gauss_sigma_ratio: float = 0.75,
        tau_base: float = 0.7,
        tau_min: float = 0.5,
        tau_max: float = 0.95,
        tau_entropy_alpha: float = 0.0,
        image_norm_mean: torch.Tensor | None = None,
        image_norm_std: torch.Tensor | None = None,
        encoder_autocast_bf16: bool = False,
        channels_last: bool = False,
        action_film_head: bool = False,
        action_film_hidden: int = 256,
        action_film_arm_dims: int = 6,
        action_film_step_dim: int | None = None,
        action_film_state_dim: int | None = None,
    ):
        super().__init__()
        if distributional and not isinstance(v_net, DistributionalVNetwork):
            raise ValueError(
                "distributional=True requires v_net to be a DistributionalVNetwork, "
                f"got {type(v_net).__name__}"
            )
        for name, value in (
            ("clip_targets_min", clip_targets_min),
            ("clip_targets_max", clip_targets_max),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided, got {value}")
        if (
            clip_targets_min is not None
            and clip_targets_max is not None
            and clip_targets_min > clip_targets_max
        ):
            raise ValueError(
                f"clip_targets_min ({clip_targets_min}) exceeds "
                f"clip_targets_max ({clip_targets_max})"
            )
        # Online networks (participate in DDP gradient sync)
        self.encoder = encoder
        self.q1 = q1
        self.q2 = q2
        self.v_net = v_net

        # Canonical IQL target topology: target Q critics provide the fixed
        # expectile labels for V, while the ONLINE V provides the Q bootstrap.
        # The target encoder is the representation half of the target critics.
        self.encoder_target = copy.deepcopy(encoder)
        self.q1_target = copy.deepcopy(q1)
        self.q2_target = copy.deepcopy(q2)
        for p in chain(
            self.encoder_target.parameters(),
            self.q1_target.parameters(),
            self.q2_target.parameters(),
        ):
            p.requires_grad = False
        self.encoder_target.eval()
        self.q1_target.eval()
        self.q2_target.eval()

        # Config
        self.camera_keys = camera_keys
        if separate_encoders:
            if not isinstance(encoder, nn.ModuleDict):
                raise ValueError(
                    "separate_encoders=True expects encoder to be an nn.ModuleDict "
                    "keyed by camera name"
                )
            bad_keys = [k for k in camera_keys if not k.startswith("observation.images.")]
            if bad_keys:
                raise ValueError(
                    "separate_encoders=True expects camera keys to start with "
                    f"'observation.images.', got: {bad_keys}"
                )
        self._enc_keys = [k.removeprefix("observation.images.") for k in camera_keys]
        self.separate_encoders = separate_encoders

        # Image normalization mirrored from the DP the encoder came from. Applied
        # in encode() so both online AND target encoders see the input
        # distribution they were trained behind. None = identity (encoders trained
        # on raw [0, 1] inputs).
        if (image_norm_mean is None) != (image_norm_std is None):
            raise ValueError("image_norm_mean and image_norm_std must be provided together")
        self.image_normalization_enabled = image_norm_mean is not None
        if self.image_normalization_enabled:
            self.register_buffer("image_norm_mean", image_norm_mean.reshape(1, 3, 1, 1).float())
            self.register_buffer("image_norm_std", image_norm_std.reshape(1, 3, 1, 1).float())
            if (
                not torch.isfinite(self.image_norm_mean).all()
                or not (self.image_norm_std > 0).all()
            ):
                raise ValueError(
                    "image normalization stats must be finite with std > 0; got "
                    f"mean={image_norm_mean.flatten().tolist()}, "
                    f"std={image_norm_std.flatten().tolist()}"
                )
        self.expectile = expectile
        self.gamma = gamma
        self.tau = tau
        # Throughput knobs; both apply to online AND target encode paths so
        # Polyak-coupled encoders always see matched numerics/layout.
        self.encoder_autocast_bf16 = encoder_autocast_bf16
        self.channels_last = channels_last
        self.clip_targets_min = clip_targets_min
        self.clip_targets_max = clip_targets_max

        # Action FiLM head: observation-conditioned learned action scaling
        # on the critic action input. A small MLP on the fused critic state emits
        # per-arm-dim [mean, log_std]; the critic scores Q on
        # (a - mean)/exp(log_std) applied to the leading `arm_dims` dims of each
        # action step (gripper dim untouched). The FINAL layer is zero-initialized
        # so mean=0, log_std=0 => identity at init. When off, no film_head is
        # built and no op runs. The online path uses `film_head` (trained through
        # the Q loss); the no-grad target-Q V-label path uses the Polyak-updated
        # `film_head_target`. The deploy path (mulligan/real/policy/vision_idql.py)
        # applies the same head to the z-scored candidate actions before Q.
        self.action_film_head = action_film_head
        self.action_film_arm_dims = action_film_arm_dims
        self.action_film_step_dim = action_film_step_dim
        if action_film_head:
            if action_film_state_dim is None or action_film_step_dim is None:
                raise ValueError(
                    "action_film_head=True requires action_film_state_dim and "
                    "action_film_step_dim to be provided"
                )
            if not 0 < action_film_arm_dims <= action_film_step_dim:
                raise ValueError(
                    "action_film_arm_dims must be in (0, action_film_step_dim="
                    f"{action_film_step_dim}], got {action_film_arm_dims}"
                )
            self.film_head = nn.Sequential(
                nn.Linear(action_film_state_dim, action_film_hidden),
                nn.ReLU(),
                nn.Linear(action_film_hidden, 2 * action_film_arm_dims),
            )
            # Zero-init the FINAL layer (weights AND bias) => identity at init.
            nn.init.zeros_(self.film_head[-1].weight)
            nn.init.zeros_(self.film_head[-1].bias)
            self.film_head_target = copy.deepcopy(self.film_head)
            for p in self.film_head_target.parameters():
                p.requires_grad = False
            self.film_head_target.eval()

        # DIVL (distributional V) knobs. Inert when distributional=False.
        self.distributional = distributional
        self.hl_gauss_sigma_ratio = hl_gauss_sigma_ratio
        self.tau_base = tau_base
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_entropy_alpha = tau_entropy_alpha
        if distributional:
            # HL-Gauss smoothing sigma in value units (ratio * atom spacing).
            atom_width = (v_net.v_max - v_net.v_min) / (v_net.num_atoms - 1)
            self._hl_gauss_sigma = hl_gauss_sigma_ratio * atom_width

    def train(self, mode: bool = True):
        """Set online modules' mode while target critics remain deterministic."""
        super().train(mode)
        self.encoder_target.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        if self.action_film_head:
            self.film_head_target.eval()
        return self

    def _film_actions(
        self, actions: torch.Tensor, state: torch.Tensor, *, target: bool
    ) -> torch.Tensor:
        """FiLM-transform the critic's action input conditioned on ``state``.

        Returns ``actions`` unchanged when the head is OFF (no unconditional op,
        so flag-off leaves the actions untouched). ``target=True`` uses the no-grad
        ``film_head_target`` on the target state (the target-Q V-label path);
        ``target=False`` uses the online ``film_head`` on the online state.
        """
        if not self.action_film_head:
            return actions
        head = self.film_head_target if target else self.film_head
        film_params = head(state)
        return apply_action_film(
            actions,
            film_params,
            arm_dims=self.action_film_arm_dims,
            step_dim=self.action_film_step_dim,
        )

    def encode(
        self, images: dict[str, torch.Tensor], encoder: nn.Module | None = None
    ) -> torch.Tensor:
        """Encode per-camera images -> (B, n_cameras * feat_dim).

        Expects float [0,1] images (post-augmentation); applies the mirrored DP
        image normalization before the encoder when enabled.
        """
        enc = encoder if encoder is not None else self.encoder
        features = []
        for cam_key, enc_key in zip(self.camera_keys, self._enc_keys):
            img = images[cam_key]
            if self.image_normalization_enabled:
                img = (img - self.image_norm_mean) / self.image_norm_std
            if self.channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
            cam_enc = enc[enc_key] if self.separate_encoders else enc
            if self.encoder_autocast_bf16 and img.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feat = cam_enc(img)
                feat = feat.float()
            else:
                feat = cam_enc(img)
            features.append(feat)
        return torch.cat(features, dim=-1)

    def forward_encoded_states(
        self,
        curr_state: torch.Tensor,
        next_state: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        discount_powers: torch.Tensor,
        target_curr_state: torch.Tensor | None = None,
        bootstrap_next_state: torch.Tensor | None = None,
        valid_steps: torch.Tensor | None = None,
        td_lambda_bootstrap_states: torch.Tensor | None = None,
        td_lambda_horizons: torch.Tensor | None = None,
        td_lambda_horizon_valid: torch.Tensor | None = None,
        td_lambda: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Loss computation from already-encoded full IQL states.

        ``curr_state`` and ``next_state`` are the tensors normally produced by
        ``encode(images) + proprio``. This is used by the embedding-cache trainer
        path, while ``forward()`` below preserves the image-based API.

        ``valid_steps`` (multi-step TD only) is the per-sample count of real
        environment steps in the reward window. A timeout anchor truncated by
        its episode boundary has ``valid_steps < k``: its padded positions
        repeat the last valid row (so they must be masked out of the reward
        sum, or that reward is double-counted) and its bootstrap uses the
        truncated successor with a per-sample ``gamma**valid_steps`` discount.
        ``None`` preserves the fixed ``gamma**k`` path bit-for-bit.

        Label-producing target states may carry an additional cached-view axis:
        ``target_curr_state`` / ``bootstrap_next_state`` accept ``(B, K, D)``
        and ``td_lambda_bootstrap_states`` accepts ``(B, H, K, D)``. Q/V scalar
        outputs are computed per view and averaged across K; frozen features are
        never averaged before the nonlinear heads.
        """
        k = dones.shape[1]
        td_lambda_args = (
            td_lambda_bootstrap_states,
            td_lambda_horizons,
            td_lambda_horizon_valid,
            td_lambda,
        )
        td_lambda_enabled = any(value is not None for value in td_lambda_args)
        if td_lambda_enabled and any(value is None for value in td_lambda_args):
            raise ValueError(
                "TD(lambda) requires bootstrap states, horizons, horizon-valid mask, and lambda"
            )
        if td_lambda_enabled and valid_steps is not None:
            raise ValueError("TD(lambda) supplies per-horizon validity; valid_steps must be None")

        # Frozen-encoder caches have identical online and target features. Image
        # callers pass the separately encoded target state below.
        if target_curr_state is None:
            target_curr_state = curr_state
        # The no-grad online-V bootstrap reads `bootstrap_next_state` (independently
        # sampled cached views, when given) and defaults to the active `next_state`.
        if bootstrap_next_state is None:
            bootstrap_next_state = next_state

        # V loss (expectile regression against min of the target Qs).
        with torch.no_grad():
            # The target-Q V-label path FiLM-transforms the action input with the
            # Polyak target head on the target state (identity when the head is off).
            target_heads = [self.q1_target, self.q2_target]
            if target_curr_state.ndim == 2:
                target_actions = self._film_actions(actions, target_curr_state, target=True)
                q1_for_v = self.q1_target(target_curr_state, target_actions)
                q2_for_v = self.q2_target(target_curr_state, target_actions)
                q_min = torch.min(q1_for_v, q2_for_v)
            elif target_curr_state.ndim == 3:
                b_target, n_target_views, target_state_dim = target_curr_state.shape
                if b_target != curr_state.shape[0]:
                    raise ValueError("target-current state batch size mismatch")
                flat_target_curr = target_curr_state.reshape(
                    b_target * n_target_views, target_state_dim
                )
                flat_target_actions_raw = (
                    actions[:, None, :]
                    .expand(b_target, n_target_views, actions.shape[-1])
                    .reshape(b_target * n_target_views, actions.shape[-1])
                )
                flat_target_actions = self._film_actions(
                    flat_target_actions_raw, flat_target_curr, target=True
                )
                flat_q_min = torch.stack(
                    [head(flat_target_curr, flat_target_actions) for head in target_heads], dim=0
                ).amin(dim=0)
                q_min = flat_q_min.reshape(b_target, n_target_views, -1).mean(dim=1)
            else:
                raise ValueError(
                    "target_curr_state must have shape (B,D) or (B,K,D), got "
                    f"{tuple(target_curr_state.shape)}"
                )

        # Only the Q loss trains the encoder.
        v_state = curr_state.detach()
        if self.distributional:
            # DIVL: fit a categorical V toward the HL-Gauss projection of the
            # (no-grad) scalar Q target. v_val is the scalar readout (= mean of
            # the categorical distribution) for logging only.
            v_logits = self.v_net(v_state)  # (B, num_atoms)
            soft_labels = hl_gauss_target(q_min, self.v_net.atoms, self._hl_gauss_sigma)
            value_loss = distributional_value_loss(v_logits, soft_labels).mean()
            v_val = self.v_net.expected_value(v_state)  # (B, 1), for metrics
        else:
            v_val = self.v_net(v_state)
            value_loss = expectile_loss(q_min - v_val, self.expectile).mean()

        # Q loss (TD learning with online V, matching canonical IQL).
        with torch.no_grad():
            # Dropout in an opt-in online V head must not inject noise into TD
            # labels. Temporarily use eval mode, then restore the caller's mode
            # before returning to gradient-bearing training.
            v_was_training = self.v_net.training
            self.v_net.eval()
            try:

                def target_v_per_state(states: torch.Tensor) -> torch.Tensor:
                    if self.distributional:
                        # Each view gets its own entropy-adaptive quantile before
                        # scalar values are averaged over the nuisance samples.
                        norm_entropy = self.v_net.normalized_entropy(states)
                        quantile_tau = adaptive_tau(
                            norm_entropy,
                            tau_base=self.tau_base,
                            tau_min=self.tau_min,
                            tau_max=self.tau_max,
                            alpha=self.tau_entropy_alpha,
                        )
                        return self.v_net.quantile(states, quantile_tau)
                    return self.v_net(states)

                if bootstrap_next_state.ndim == 2:
                    v_target_next = target_v_per_state(bootstrap_next_state)
                elif bootstrap_next_state.ndim == 3:
                    b_next, n_next_views, next_state_dim = bootstrap_next_state.shape
                    if b_next != curr_state.shape[0]:
                        raise ValueError("bootstrap-next state batch size mismatch")
                    flat_next = bootstrap_next_state.reshape(b_next * n_next_views, next_state_dim)
                    v_target_next = (
                        target_v_per_state(flat_next).reshape(b_next, n_next_views, -1).mean(dim=1)
                    )
                else:
                    raise ValueError(
                        "bootstrap_next_state must have shape (B,D) or (B,K,D), got "
                        f"{tuple(bootstrap_next_state.shape)}"
                    )
                v_target_horizons = None
                if td_lambda_enabled:
                    if td_lambda_bootstrap_states.ndim not in (3, 4):
                        raise ValueError(
                            "td_lambda_bootstrap_states must have shape (B,H,D) or (B,H,K,D)"
                        )
                    b_td, n_td = td_lambda_bootstrap_states.shape[:2]
                    if b_td != curr_state.shape[0]:
                        raise ValueError("TD(lambda) bootstrap-state batch size mismatch")
                    if td_lambda_bootstrap_states.ndim == 3:
                        state_dim_td = td_lambda_bootstrap_states.shape[-1]
                        flat_bootstrap = td_lambda_bootstrap_states.reshape(
                            b_td * n_td, state_dim_td
                        )
                        v_target_horizons = target_v_per_state(flat_bootstrap).reshape(
                            b_td, n_td, 1
                        )
                    else:
                        n_td_views, state_dim_td = td_lambda_bootstrap_states.shape[-2:]
                        flat_bootstrap = td_lambda_bootstrap_states.reshape(
                            b_td * n_td * n_td_views, state_dim_td
                        )
                        v_target_horizons = (
                            target_v_per_state(flat_bootstrap)
                            .reshape(b_td, n_td, n_td_views, 1)
                            .mean(dim=2)
                        )
            finally:
                self.v_net.train(v_was_training)
            torch._assert_async(
                torch.isfinite(v_target_next).all(),
                "IQL online-V bootstrap is non-finite before clipping",
            )
            td_targets_by_horizon = None
            td_lambda_weights = None
            if td_lambda_enabled:
                td_target, td_targets_by_horizon, td_lambda_weights = (
                    build_truncated_td_lambda_target(
                        rewards,
                        dones,
                        discount_powers,
                        v_target_horizons,
                        td_lambda_horizons,
                        td_lambda_horizon_valid,
                        gamma=self.gamma,
                        td_lambda=td_lambda,
                    )
                )
            else:
                # Discounted reward with done masking.
                pre_done = dones.float().cumsum(dim=1).roll(1, 1)
                pre_done[:, 0] = 0
                reward_mask = (pre_done == 0).float()
            if not td_lambda_enabled and valid_steps is not None:
                if valid_steps.shape != (dones.shape[0],):
                    raise ValueError(
                        f"valid_steps must have shape ({dones.shape[0]},), got "
                        f"{tuple(valid_steps.shape)} — a smaller tensor would silently "
                        "broadcast one horizon across the batch"
                    )
                if valid_steps.is_floating_point():
                    raise ValueError(
                        f"valid_steps must be an integer step count, got {valid_steps.dtype}"
                    )
                steps = valid_steps.to(device=dones.device).reshape(-1, 1).float()
                if bool((steps < 1).any()) or bool((steps > k).any()):
                    raise ValueError(
                        f"valid_steps must lie in [1, {k}] for a {k}-step reward window"
                    )
                valid_mask = (
                    torch.arange(k, device=dones.device, dtype=torch.float32)[None, :] < steps
                ).float()
                reward_mask = reward_mask * valid_mask
                dones_windowed = dones.float() * valid_mask
                bootstrap_discount = self.gamma**steps
            elif not td_lambda_enabled:
                dones_windowed = dones.float()
                bootstrap_discount = self.gamma**k
            if not td_lambda_enabled:
                b_rewards = (rewards.float() * discount_powers * reward_mask).sum(
                    dim=1, keepdim=True
                )
                torch._assert_async(
                    torch.isfinite(b_rewards).all(),
                    "IQL discounted reward target is non-finite before clipping",
                )
                b_masks = 1.0 - dones_windowed.max(dim=1, keepdim=True).values
                td_target = b_rewards + bootstrap_discount * b_masks * v_target_next
            if self.clip_targets_min is not None or self.clip_targets_max is not None:
                # Bound the complete Bellman label, including terminal rewards;
                # clipping only V(s') does not bound r + gamma V(s').
                td_target = td_target.clamp(
                    min=self.clip_targets_min,
                    max=self.clip_targets_max,
                )
            torch._assert_async(torch.isfinite(td_target).all(), "IQL TD target is non-finite")

        # The online Q path FiLM-transforms the action input with the online head
        # on the online state (identity when the head is off). The head trains
        # through the Q loss.
        online_actions = self._film_actions(actions, curr_state, target=False)
        q1_pred = self.q1(curr_state, online_actions)
        q2_pred = self.q2(curr_state, online_actions)
        critic_loss = ((q1_pred - td_target) ** 2).mean() + ((q2_pred - td_target) ** 2).mean()

        total_loss = value_loss + critic_loss

        torch._assert_async(torch.isfinite(total_loss), "VisionIQL total loss is non-finite")
        metrics = {
            "total": total_loss,
            "value_loss": value_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "q1_mean": q1_pred.detach().mean(),
            "q2_mean": q2_pred.detach().mean(),
            "v_mean": v_val.detach().mean(),
            "advantage_mean": (q_min - v_val).detach().mean(),
            "td_target_mean": td_target.detach().mean(),
        }
        if td_lambda_enabled:
            valid_float = td_lambda_horizon_valid.to(td_targets_by_horizon.dtype)
            valid_count = valid_float.sum(dim=1)
            target_mean = (td_targets_by_horizon * valid_float).sum(dim=1) / valid_count
            target_variance = (
                (td_targets_by_horizon - target_mean[:, None]).square() * valid_float
            ).sum(dim=1) / valid_count
            metrics.update(
                {
                    "td_lambda": td_target.detach().new_tensor(td_lambda),
                    "td_effective_horizon_mean": (
                        td_lambda_weights * td_lambda_horizons.to(td_lambda_weights.dtype)[None, :]
                    )
                    .sum(dim=1)
                    .mean()
                    .detach(),
                    "td_valid_horizon_count_mean": valid_count.mean().detach(),
                    "td_target_horizon_std_mean": target_variance.sqrt().mean().detach(),
                }
            )
            horizon_valid_count = valid_float.sum(dim=0)
            horizon_target_mean = (td_targets_by_horizon * valid_float).sum(
                dim=0
            ) / horizon_valid_count.clamp(min=1.0)
            horizon_target_mean = torch.where(
                horizon_valid_count > 0,
                horizon_target_mean,
                horizon_target_mean.new_full(horizon_target_mean.shape, float("nan")),
            )
            for horizon_idx, horizon in enumerate(td_lambda_horizons.tolist()):
                metrics[f"td_h{horizon}_valid_frac"] = valid_float[:, horizon_idx].mean().detach()
                metrics[f"td_h{horizon}_target_mean"] = horizon_target_mean[horizon_idx].detach()
        return metrics

    def forward(
        self,
        curr_images: dict[str, torch.Tensor],
        next_images: dict[str, torch.Tensor],
        proprio_curr: torch.Tensor,
        proprio_next: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        discount_powers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full image-based forward + loss computation.

        Returns dict with 'total' (has grad) plus detached metrics.
        """

        # Encode current images (with grads for the Q path).
        curr_visual = self.encode(curr_images)
        curr_state = torch.cat([curr_visual, proprio_curr], dim=-1)

        # Target-Q labels use the target encoder; online-V TD bootstrap uses the
        # online encoder. Both are no-grad paths.
        with torch.no_grad():
            target_curr_visual = self.encode(curr_images, encoder=self.encoder_target)
            target_curr_state = torch.cat([target_curr_visual, proprio_curr], dim=-1)
            next_visual = self.encode(next_images)
            next_state = torch.cat([next_visual, proprio_next], dim=-1)

        return self.forward_encoded_states(
            curr_state,
            next_state,
            actions,
            rewards,
            dones,
            discount_powers,
            target_curr_state=target_curr_state,
        )

    @torch.no_grad()
    def update_targets(self, *, update_encoder: bool = True):
        """Polyak-average online Q critics and optionally the encoder."""
        for target, online in zip(self.q1_target.parameters(), self.q1.parameters(), strict=True):
            target.lerp_(online, self.tau)
        for target, online in zip(self.q2_target.parameters(), self.q2.parameters(), strict=True):
            target.lerp_(online, self.tau)
        # Polyak the FiLM target head exactly like the target critics.
        if self.action_film_head:
            for target, online in zip(
                self.film_head_target.parameters(), self.film_head.parameters(), strict=True
            ):
                target.lerp_(online, self.tau)
        if not update_encoder:
            return
        for target, online in zip(
            self.encoder_target.parameters(), self.encoder.parameters(), strict=True
        ):
            target.lerp_(online, self.tau)
        # BatchNorm running statistics and counters are buffers rather than
        # parameters. Keep the target representation complete when the encoder
        # is unfrozen; frozen encoders are unaffected.
        for target, online in zip(
            self.encoder_target.buffers(), self.encoder.buffers(), strict=True
        ):
            if target.is_floating_point():
                target.lerp_(online, self.tau)
            else:
                target.copy_(online)
