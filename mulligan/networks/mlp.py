"""
Simple MLP implementation with configurable layers, activations, and layer normalization.

Modeled on RLPD's network architecture; written independently in PyTorch.
"""

from typing import Callable, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.

    Args:
        hidden_dims: Sequence of hidden layer dimensions (including output dimension).
        activation: Activation function to use between layers. Default: nn.ReLU.
        activate_final: Whether to apply activation after the final layer. Default: False.
        use_layer_norm: Whether to apply layer normalization after each layer. Default: False.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activation: Callable[[], nn.Module] = nn.ReLU,
        activate_final: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.activate_final = activate_final

        if len(hidden_dims) < 2:
            raise ValueError("MLP requires at least input dimension and one hidden dimension")

        layers = []
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]

            # Create linear layer
            linear = nn.Linear(in_dim, out_dim)

            nn.init.xavier_uniform_(linear.weight)

            layers.append(linear)

            # Add normalization and activation for all layers except the last
            # (unless activate_final is True)
            is_last_layer = i + 2 == len(hidden_dims)
            if not is_last_layer or activate_final:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, ..., input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, ..., output_dim)
        """
        out = self.network(x)

        return out
