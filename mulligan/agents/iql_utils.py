#!/usr/bin/env python3
"""
Shared utilities for IQL-based policies.

This module contains common functions used by IQL, IDQL, and related policies.
"""

import torch


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """
    Compute expectile regression loss.

    This is the asymmetric squared loss used in IQL for V-network updates.
    When expectile > 0.5, it assigns more weight to positive errors (overestimation).

    Args:
        diff: Difference tensor (Q - V)
        expectile: Expectile parameter (default: 0.7)

    Returns:
        Expectile loss tensor (per-sample, not reduced)
    """
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff**2)


def hl_gauss_target(
    target_scalar: torch.Tensor,
    atoms: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Project a scalar target onto a categorical support via HL-Gauss smoothing.

    HL-Gauss (Histogram Loss with Gaussian smoothing) places a Gaussian centered
    at ``target_scalar`` over the fixed support ``atoms`` and integrates it onto
    the atom bins, yielding a soft-label distribution. This is the standard,
    robust way to build cross-entropy targets for categorical value learning and
    is what DIVL uses to turn the (no-grad) scalar Q-target into a distribution
    the categorical V-network regresses toward.

    Args:
        target_scalar: Scalar targets of shape (B, 1) or (B,).
        atoms: Fixed support of shape (num_atoms,), assumed evenly spaced.
        sigma: Standard deviation of the smoothing Gaussian (value units).

    Returns:
        Soft-label distribution of shape (B, num_atoms), rows summing to 1.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if atoms.ndim != 1 or atoms.numel() < 2:
        raise ValueError(f"atoms must be a 1-D tensor with at least 2 entries, got {atoms.shape}")
    if not atoms.is_floating_point():
        raise TypeError(f"atoms must have floating dtype, got {atoms.dtype}")

    target = target_scalar.reshape(-1, 1).to(device=atoms.device, dtype=atoms.dtype)
    torch._assert_async(torch.isfinite(target).all(), "HL-Gauss targets must be finite")
    # Categorical support is a bounded value representation. Saturating targets
    # at its endpoints guarantees a well-conditioned edge-bin projection even
    # for arbitrarily large Q estimates; without this, float32 erf can saturate
    # both CDF terms and silently produce an all-zero target row.
    target = target.clamp(min=atoms[0], max=atoms[-1])
    num_atoms = atoms.shape[0]
    atom_width = (atoms[-1] - atoms[0]) / (num_atoms - 1)
    atom_diffs = atoms[1:] - atoms[:-1]
    torch._assert_async(
        torch.isfinite(atoms).all()
        & (atom_diffs > 0).all()
        & torch.isclose(atom_diffs, atom_width, atol=1e-6, rtol=1e-5).all(),
        "HL-Gauss atoms must be finite, strictly increasing, and evenly spaced",
    )

    # Bin edges midway between consecutive atoms, extended by half a bin on each
    # end. Targets outside the support saturate to the nearest endpoint.
    centers = atoms.reshape(1, -1)  # (1, num_atoms)
    lower_edges = centers - atom_width / 2.0
    upper_edges = centers + atom_width / 2.0

    sqrt2_sigma = sigma * (2.0**0.5)
    cdf_upper = 0.5 * (1.0 + torch.erf((upper_edges - target) / sqrt2_sigma))
    cdf_lower = 0.5 * (1.0 + torch.erf((lower_edges - target) / sqrt2_sigma))
    probs = (cdf_upper - cdf_lower).clamp_min(0.0)  # (B, num_atoms)

    # Renormalize so rows sum to 1 (mass falling outside [first_edge, last_edge]
    # is redistributed proportionally; without this, targets near/beyond the
    # support edges would yield rows summing to < 1).
    row_mass = probs.sum(dim=-1, keepdim=True)
    torch._assert_async(
        torch.isfinite(row_mass).all() & (row_mass > 0).all(),
        "HL-Gauss projection produced non-finite or zero probability mass",
    )
    probs = probs / row_mass
    torch._assert_async(
        torch.isfinite(probs).all()
        & (probs >= 0).all()
        & torch.isclose(
            probs.sum(dim=-1),
            torch.ones(probs.shape[0], device=probs.device, dtype=probs.dtype),
            atol=1e-5,
            rtol=1e-5,
        ).all(),
        "HL-Gauss projection must produce finite normalized probability rows",
    )
    return probs


def distributional_value_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between predicted categorical V and the HL-Gauss soft labels.

    Per-sample and unreduced, matching ``expectile_loss``'s contract so the two
    value-loss paths are interchangeable in the IDQL loss method.

    Args:
        logits: Predicted logits of shape (B, num_atoms).
        soft_labels: Target distribution of shape (B, num_atoms).

    Returns:
        Per-sample cross-entropy of shape (B, 1).
    """
    if logits.ndim != 2 or soft_labels.shape != logits.shape:
        raise ValueError(
            "distributional logits/label shape mismatch: "
            f"{tuple(logits.shape)=}, {tuple(soft_labels.shape)=}"
        )
    valid_inputs = (
        torch.isfinite(logits).all()
        & torch.isfinite(soft_labels).all()
        & (soft_labels >= 0).all()
        & torch.isclose(
            soft_labels.sum(dim=-1),
            torch.ones(
                soft_labels.shape[0],
                device=soft_labels.device,
                dtype=soft_labels.dtype,
            ),
            atol=1e-5,
            rtol=1e-5,
        ).all()
    )
    torch._assert_async(
        valid_inputs,
        "distributional value loss requires finite logits and normalized nonnegative labels",
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(soft_labels * log_probs).sum(dim=-1, keepdim=True)
    torch._assert_async(torch.isfinite(loss).all(), "distributional value loss is non-finite")
    return loss


def adaptive_tau(
    norm_entropy: torch.Tensor,
    tau_base: float,
    tau_min: float,
    tau_max: float,
    alpha: float,
) -> torch.Tensor:
    """Entropy-adaptive quantile level for the DIVL TD target.

    Lowers tau (a less-optimistic statistic) on high-entropy / uncertain states:
    ``tau = clip(tau_base - alpha * norm_entropy, tau_min, tau_max)``. With
    ``alpha = 0`` this returns a constant ``tau_base`` (fixed-tau behavior).

    Args:
        norm_entropy: Normalized entropy in [0, 1], shape (B, 1) or (B,).
        tau_base: Base quantile level (e.g. the IQL expectile, 0.7).
        tau_min: Lower clip bound for tau.
        tau_max: Upper clip bound for tau.
        alpha: Entropy sensitivity (0 disables adaptation).

    Returns:
        Per-sample tau of shape (B, 1).
    """
    tau = tau_base - alpha * norm_entropy.reshape(-1, 1)
    return tau.clamp(min=tau_min, max=tau_max)
