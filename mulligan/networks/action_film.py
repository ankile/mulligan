"""Observation-conditioned action scaling ("action FiLM") for the IQL critics.

Shared by the vision critic (``mulligan.networks.vision_iql``) and the real-robot scorer
(``mulligan.real.policy.vision_idql``), so both apply one definition of the transform.
"""

from __future__ import annotations

import torch


def apply_action_film(
    actions: torch.Tensor,
    film_params: torch.Tensor,
    *,
    arm_dims: int,
    step_dim: int,
    log_std_min: float = -2.0,
    log_std_max: float = 2.0,
) -> torch.Tensor:
    """Observation-conditioned learned action-scaling ("action FiLM").

    Applies ``(a - mean) / exp(log_std)`` per action step to the LEADING
    ``arm_dims`` dims of each step's action, leaving the trailing (gripper) dims
    untouched. ``film_params`` are the FiLM head's output ``[mean(arm_dims),
    log_std(arm_dims)]`` for the SAME state the critic scores; the same per-arm
    mean/log_std is broadcast across every action step. ``log_std`` is clamped
    to ``[log_std_min, log_std_max]`` for numerical stability. This is a pure,
    module-free helper so the training path (``VisionIQL``) and every offline
    scorer share ONE definition of the transform (parity-pinned in tests).

    With a zero-initialised final FiLM layer, ``film_params`` is all-zero, so
    ``mean=0`` and ``log_std=0`` => the transform is the identity at init.

    Args:
        actions: flat action tensor ``(B, n_steps * step_dim)`` — the RAW or
            z-scored flat action chunk fed to the Q network (the FiLM transform
            is applied at whichever stage the critic scores; training + the
            scorers both apply it to the z-scored flat action).
        film_params: ``(B, 2 * arm_dims)`` FiLM head output.
        arm_dims: number of leading per-step dims to scale (e.g. 6 arm dims).
        step_dim: full per-step action dim (e.g. 7 = 6 arm + 1 gripper).
        log_std_min / log_std_max: clamp bounds for ``log_std``.

    Returns:
        Transformed flat action tensor, same shape/dtype/device as ``actions``.
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2-D (B, flat), got shape {tuple(actions.shape)}")
    b, flat = actions.shape
    if step_dim <= 0 or flat % step_dim != 0:
        raise ValueError(
            f"flat action dim {flat} is not a positive multiple of step_dim {step_dim}"
        )
    if not 0 < arm_dims <= step_dim:
        raise ValueError(f"arm_dims must be in (0, step_dim={step_dim}], got {arm_dims}")
    if film_params.shape != (b, 2 * arm_dims):
        raise ValueError(
            f"film_params must have shape ({b}, {2 * arm_dims}), got {tuple(film_params.shape)}"
        )
    n_steps = flat // step_dim
    mean = film_params[:, :arm_dims]
    log_std = film_params[:, arm_dims:].clamp(log_std_min, log_std_max)
    inv_std = torch.exp(-log_std)
    steps = actions.reshape(b, n_steps, step_dim)
    arm = steps[:, :, :arm_dims]
    rest = steps[:, :, arm_dims:]
    scaled_arm = (arm - mean[:, None, :]) * inv_std[:, None, :]
    return torch.cat([scaled_arm, rest], dim=-1).reshape(b, flat)
