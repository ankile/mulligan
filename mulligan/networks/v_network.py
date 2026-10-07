#!/usr/bin/env python3
"""
V-network implementation for IQL value function.

This module provides an MLP-based V-network that takes states as input
and outputs state values for Implicit Q-Learning (IQL).
"""

import torch
import torch.nn as nn

from mulligan.networks.mlp import MLP


class VNetwork(nn.Module):
    """
    V-network for estimating state values V(s).

    Takes state observations as input and outputs scalar V-values.
    Used in IQL for learning the value function via expectile regression.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int] = [256, 256, 256],
        use_layer_norm: bool = False,
    ):
        """
        Initialize V-network.

        Args:
            state_dim: Dimension of state observations
            hidden_dims: List of hidden layer dimensions
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.state_dim = state_dim

        # Build MLP with state as input and scalar V-value as output
        dims = [state_dim] + hidden_dims + [1]

        self.mlp = MLP(
            hidden_dims=dims,
            activation=nn.ReLU,
            activate_final=False,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute V(s).

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            V-value tensor of shape (batch_size, 1)
        """
        v_value = self.mlp(state)
        return v_value
