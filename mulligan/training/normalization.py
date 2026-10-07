"""Normalization utilities for states and actions."""

import torch


class Normalizer:
    """
    Handles normalization and denormalization of states and actions.

    State normalization: Supports both z-score (zero mean, unit variance) and MIN_MAX (to [-1, 1])
    Action handling: Assumes actor outputs in [-1, 1] via tanh, scales to actual action bounds
    """

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        device: str = "cpu",
        state_min: torch.Tensor | None = None,
        state_max: torch.Tensor | None = None,
    ):
        """
        Initialize normalizer with statistics.

        Args:
            state_mean: Mean of state observations (for z-score normalization)
            state_std: Standard deviation of state observations (for z-score normalization)
            action_min: Minimum action values (actual environment bounds)
            action_max: Maximum action values (actual environment bounds)
            device: Device to store tensors on
            state_min: Minimum state values (for MIN_MAX normalization, optional)
            state_max: Maximum state values (for MIN_MAX normalization, optional)
        """
        self.device = device

        # State normalization (zero mean, unit variance)
        self.state_mean = state_mean.to(device)
        self.state_std = state_std.to(device)

        # State MIN_MAX normalization (optional, for matching LeRobot's DiffusionPolicy)
        if state_min is not None and state_max is not None:
            self.state_min = state_min.to(device)
            self.state_max = state_max.to(device)
            # Precompute scaling factors for MIN_MAX: [-1, 1] range
            # Formula: normalized = 2 * (value - min) / (max - min) - 1
            state_range = self.state_max - self.state_min
            # Avoid division by zero for constant features
            self.state_range = torch.where(
                state_range.abs() < 1e-8, torch.ones_like(state_range), state_range
            )
        else:
            self.state_min = None
            self.state_max = None
            self.state_range = None

        # Action bounds (actual environment limits)
        self.action_min = action_min.to(device)
        self.action_max = action_max.to(device)

        # Precompute action scaling factors for tanh output [-1, 1] -> [action_min, action_max]
        # Formula: action = (tanh_output + 1) / 2 * (max - min) + min
        #        = tanh_output * (max - min) / 2 + (max + min) / 2
        self.action_scale = (self.action_max - self.action_min) / 2.0  # Multiplier
        self.action_bias = (self.action_max + self.action_min) / 2.0  # Offset

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state to zero mean, unit variance (z-score normalization).

        Args:
            state: Unnormalized state tensor

        Returns:
            Normalized state tensor
        """
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, state_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state from zero mean, unit variance to original scale.

        Args:
            state_normalized: Normalized state tensor

        Returns:
            Unnormalized state tensor
        """
        return state_normalized * self.state_std + self.state_mean

    def normalize_state_minmax(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state to [-1, 1] using MIN_MAX normalization.

        This matches LeRobot's DiffusionPolicy default normalization for STATE features.
        Formula: normalized = 2 * (value - min) / (max - min) - 1

        Args:
            state: Unnormalized state tensor

        Returns:
            Normalized state tensor in [-1, 1]

        Raises:
            ValueError: If state_min/state_max were not provided during initialization
        """
        if self.state_min is None or self.state_max is None:
            raise ValueError(
                "MIN_MAX normalization requires state_min and state_max. "
                "Provide these during Normalizer initialization."
            )
        return 2.0 * (state - self.state_min) / self.state_range - 1.0

    def denormalize_state_minmax(self, state_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state from [-1, 1] MIN_MAX range to original scale.

        Args:
            state_normalized: Normalized state tensor in [-1, 1]

        Returns:
            Unnormalized state tensor

        Raises:
            ValueError: If state_min/state_max were not provided during initialization
        """
        if self.state_min is None or self.state_max is None:
            raise ValueError(
                "MIN_MAX denormalization requires state_min and state_max. "
                "Provide these during Normalizer initialization."
            )
        return (state_normalized + 1.0) / 2.0 * self.state_range + self.state_min

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Normalize action from [action_min, action_max] to [-1, 1].

        Used during training to convert demonstration actions to the range
        expected from the tanh-bounded actor output.

        Args:
            action: Unnormalized action tensor in [action_min, action_max]

        Returns:
            Normalized action tensor in [-1, 1]
        """
        # action_normalized = (action - bias) / scale
        return (action - self.action_bias) / self.action_scale

    def denormalize_action(self, action_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize action from [-1, 1] to [action_min, action_max].

        Used during inference to convert tanh-bounded actor output to
        actual environment action space.

        Args:
            action_normalized: Normalized action tensor in [-1, 1] (from tanh)

        Returns:
            Unnormalized action tensor in [action_min, action_max]
        """
        # action = action_normalized * scale + bias
        return action_normalized * self.action_scale + self.action_bias

    def to(self, device: str):
        """Move all tensors to the specified device."""
        self.device = device
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        if self.state_min is not None:
            self.state_min = self.state_min.to(device)
        if self.state_max is not None:
            self.state_max = self.state_max.to(device)
        if self.state_range is not None:
            self.state_range = self.state_range.to(device)
        self.action_min = self.action_min.to(device)
        self.action_max = self.action_max.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return self


def zscore_to_actor_inputs(
    normalizer, robot_zscore: torch.Tensor, env_zscore: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Critic-space (z-score) states -> diffusion-actor inputs.

    The robot state goes to MIN_MAX [-1, 1] (LeRobot's STATE normalization); the
    environment state goes back to raw values (the actor's normalizer passes ENV
    features through). ``normalizer`` needs ``state_mean``/``state_std``/
    ``state_min``/``state_range``; the robot state is the leading block.
    """
    d = robot_zscore.shape[-1]
    robot_raw = robot_zscore * normalizer.state_std[:d] + normalizer.state_mean[:d]
    robot_range = normalizer.state_range[:d].clamp(min=1e-8)
    robot_minmax = 2.0 * (robot_raw - normalizer.state_min[:d]) / robot_range - 1.0
    if env_zscore is None:
        return robot_minmax, None
    return robot_minmax, env_zscore * normalizer.state_std[d:] + normalizer.state_mean[d:]
