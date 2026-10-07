#!/usr/bin/env python3
"""Distributional V-network for DIVL (Distributional Implicit Value Learning).

This module provides an MLP-based value network that, instead of regressing a
single scalar V(s), predicts a *categorical distribution* over a fixed support
of dataset action-value atoms. An asymmetric statistic (a tau-quantile) is then
extracted from that distribution as the TD bootstrap target, mirroring IQL's
expectile regression at the optimum (Finch DIVL, Proposition 1) while preserving
rare-but-reproducible high-return modes that a scalar critic averages away.

This is a sibling to ``VNetwork`` — the constructor signature is intentionally
identical apart from the categorical-support arguments (``num_atoms``,
``v_min``, ``v_max``), so the surrounding IDQL code can swap one for the other
behind a flag. ``VNetwork`` is left untouched.

Reference:
- AgiBot Finch, "Learning while Deploying: Fleet-Scale RL for Generalist Robot
  Policies", arXiv:2605.00416 (May 2026) — DIVL.
- Bellemare et al., "A Distributional Perspective on Reinforcement Learning",
  ICML 2017 — categorical value distributions on a fixed support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mulligan.networks.mlp import MLP


class DistributionalVNetwork(nn.Module):
    """V-network predicting a categorical distribution over a fixed value support.

    The network emits ``num_atoms`` logits per state; softmax over them gives a
    categorical distribution on the fixed support
    ``atoms = linspace(v_min, v_max, num_atoms)`` (registered as a buffer so it
    is serialized in the state dict and restored on load). Scalar readouts
    (expected value, tau-quantile) are derived from this distribution.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int] = [256, 256, 256],
        use_layer_norm: bool = False,
        num_atoms: int = 101,
        v_min: float = -1.0,
        v_max: float = 1.0,
    ):
        """Initialize the distributional V-network.

        Args:
            state_dim: Dimension of state observations.
            hidden_dims: List of hidden layer dimensions.
            use_layer_norm: Whether to use layer normalization.
            num_atoms: Number of categorical support atoms.
            v_min: Lower edge of the value support.
            v_max: Upper edge of the value support.
        """
        super().__init__()

        if num_atoms < 2:
            raise ValueError(f"num_atoms must be >= 2, got {num_atoms}")
        if not v_max > v_min:
            raise ValueError(f"v_max ({v_max}) must be greater than v_min ({v_min})")

        self.state_dim = state_dim
        self.num_atoms = num_atoms
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        # MLP shares VNetwork's architecture but emits `num_atoms` logits.
        dims = [state_dim] + hidden_dims + [num_atoms]
        self.mlp = MLP(
            hidden_dims=dims,
            activation=nn.ReLU,
            activate_final=False,
            use_layer_norm=use_layer_norm,
        )

        # Fixed support, registered as a buffer so it moves with .to(device) and
        # is saved/restored in the state dict (eval-time reconstruction).
        atoms = torch.linspace(self.v_min, self.v_max, num_atoms)
        self.register_buffer("atoms", atoms)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw logits over the value support.

        Args:
            state: State tensor of shape (batch_size, state_dim).

        Returns:
            Logits tensor of shape (batch_size, num_atoms).
        """
        return self.mlp(state)

    def probs(self, state: torch.Tensor) -> torch.Tensor:
        """Return the categorical distribution (softmax over logits)."""
        return F.softmax(self.forward(state), dim=-1)

    def expected_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return the expected value E[V] = sum(p_i * atom_i).

        Scalar readout for logging / metrics so dashboards stay comparable to
        the scalar VNetwork's v_mean.

        Returns:
            Tensor of shape (batch_size, 1).
        """
        probs = self.probs(state)
        return (probs * self.atoms).sum(dim=-1, keepdim=True)

    def quantile(self, state: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
        """Return the tau-quantile of the categorical distribution via CDF inversion.

        The tau-quantile is the smallest atom whose cumulative probability is
        >= tau. This is the asymmetric statistic used as the TD bootstrap target
        in DIVL (analogous to IQL's expectile).

        Args:
            state: State tensor of shape (batch_size, state_dim).
            tau: Quantile level, scalar in [0, 1] or per-sample tensor of shape
                 (batch_size, 1) / (batch_size,).

        Returns:
            Tensor of shape (batch_size, 1) within [v_min, v_max].
        """
        probs = self.probs(state)  # (B, num_atoms)
        cdf = probs.cumsum(dim=-1)  # (B, num_atoms)

        if not torch.is_tensor(tau):
            tau = torch.as_tensor(tau, dtype=probs.dtype, device=probs.device)
        tau = tau.reshape(-1, 1) if tau.ndim > 0 else tau.reshape(1, 1)
        tau = tau.to(dtype=probs.dtype, device=probs.device)

        # First atom index whose CDF reaches tau (argmax returns the first
        # True). Float32 rounding can leave the final CDF entry just under 1.0,
        # so for tau very close to 1 no atom satisfies cdf >= tau and the row is
        # all-False; argmax would then return index 0 (=> v_min, the *opposite*
        # extreme). Guard that by mapping all-False rows to the top atom (v_max),
        # which is the correct high-quantile limit.
        reached = cdf >= tau  # (B, num_atoms)
        idx = reached.float().argmax(dim=-1)  # (B,)
        idx = torch.where(reached.any(dim=-1), idx, torch.full_like(idx, self.num_atoms - 1))
        return self.atoms[idx].unsqueeze(-1)  # (B, 1)

    def normalized_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Return categorical entropy normalized to [0, 1] by log(num_atoms).

        Used to drive the entropy-adaptive tau (lower / less-optimistic tau on
        high-entropy, uncertain states). Uniform logits -> ~1.0, peaked -> ~0.0.

        Returns:
            Tensor of shape (batch_size, 1).
        """
        probs = self.probs(state)
        # -sum p log p, guarding log(0) with clamp.
        entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1, keepdim=True)
        log_n = torch.log(torch.tensor(float(self.num_atoms), device=probs.device))
        return entropy / log_n
