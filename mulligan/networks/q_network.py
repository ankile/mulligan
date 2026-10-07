#!/usr/bin/env python3
"""
Q-network implementation for TD3 critic.

This module provides an MLP-based Q-network that takes state-action pairs
as input and outputs Q-values for value-based RL algorithms like TD3.
"""

import torch
import torch.nn as nn

from mulligan.networks.mlp import MLP


class QNetwork(nn.Module):
    """
    Q-network for estimating action-values Q(s, a).

    Takes concatenated state-action pairs as input and outputs scalar Q-values.
    Used as the critic in actor-critic algorithms like TD3.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256, 256],
        use_layer_norm: bool = False,
    ):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state observations
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP with [state, action] as input and scalar Q-value as output
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [1]

        self.mlp = MLP(
            hidden_dims=dims,
            activation=nn.ReLU,
            activate_final=False,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q(s, a).

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)

        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        # Concatenate state and action
        sa = torch.cat([state, action], dim=-1)
        q_value = self.mlp(sa)
        return q_value
