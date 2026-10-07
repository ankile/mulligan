"""Pre/post processors for the IDQL/DIVL agents.

The preprocessor z-score normalizes ``observation.state`` and
``observation.environment_state`` separately (the agents concatenate them for the Q/V
networks and convert to MIN_MAX for the diffusion actor) and maps actions to [-1, 1].
"""

from dataclasses import dataclass
from typing import Any

import torch

from mulligan.training.normalization import Normalizer

# Keys that pass through preprocessors unchanged (not normalized)
# These are RL-specific tensors needed for critic training
PASSTHROUGH_KEYS = [
    "reward",  # Cumulative discounted reward over chunk
    "masks",  # Bootstrap decision (1 = bootstrap, 0 = don't)
    "chunk_valid",  # For weighting critic loss (0 if crosses episode boundary)
    "action_is_pad",  # For BC loss masking
    "source",  # Data source indicator
    "success",  # Success flag
]


@dataclass
class IDQLPreprocessor:
    """
    Preprocessor for IDQL policy that normalizes but KEEPS keys separate.

    This keeps observation.state and observation.environment_state separate
    (rather than concatenating them) because IDQLPolicy.select_action()
    expects separate keys.

    The IDQLPolicy.update() method handles:
    - Concatenation for Q/V networks
    - Converting current states from z-score to MIN_MAX for diffusion actor

    Normalization strategy:
    - Current states: z-score normalization (IDQLPolicy.update() converts to
      MIN_MAX for diffusion actor, IDQLPolicy.select_action() also converts)
    - Next states (for Q/V networks): z-score normalization (consistent with IQL)
    - Actions: MIN_MAX normalization to [-1, 1]

    IMPORTANT: The diffusion actor expects MIN_MAX normalized states. The conversion
    from z-score to MIN_MAX is handled by IDQLPolicy methods, NOT by this preprocessor.
    This allows Q/V networks to use z-score normalization consistently.
    """

    normalizer: Normalizer
    device: str

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize batch data while keeping state keys separate for IDQL.

        Args:
            batch: Dict with LeRobot-style keys:
                - observation.state: Robot state
                - observation.environment_state: Environment state
                - action: Action tensor
                - next.observation.state: (optional) Next robot state
                - next.observation.environment_state: (optional) Next env state
                - reward, done, source, success: (optional) RL training data

        Returns:
            Normalized batch with SEPARATE state keys:
                - observation.state: Z-score normalized robot state (NOT concatenated)
                - observation.environment_state: Z-score normalized env state (separate)
                - next.observation.state: Z-score normalized (concatenated for Q/V)
                - action: MIN_MAX normalized action
                - Other keys passed through
        """
        result = {}

        # For current states: keep separate keys (IDQLPolicy.select_action expects this)
        # Use z-score normalization - select_action() handles conversion to MIN_MAX for diffusion
        if "observation.state" in batch:
            robot_state = batch["observation.state"]
            if not isinstance(robot_state, torch.Tensor):
                robot_state = torch.tensor(robot_state, dtype=torch.float32)
            robot_state = robot_state.to(self.device)
            # Z-score normalize using robot portion of normalizer stats
            robot_dim = robot_state.shape[-1]
            robot_mean = self.normalizer.state_mean[:robot_dim]
            robot_std = self.normalizer.state_std[:robot_dim]
            result["observation.state"] = (robot_state - robot_mean) / robot_std

        if "observation.environment_state" in batch:
            env_state = batch["observation.environment_state"]
            if not isinstance(env_state, torch.Tensor):
                env_state = torch.tensor(env_state, dtype=torch.float32)
            env_state = env_state.to(self.device)
            # Z-score normalize using env portion of normalizer stats
            robot_dim = (
                result["observation.state"].shape[-1] if "observation.state" in result else 0
            )
            env_mean = self.normalizer.state_mean[robot_dim:]
            env_std = self.normalizer.state_std[robot_dim:]
            result["observation.environment_state"] = (env_state - env_mean) / env_std

        # For next states: concatenate and z-score normalize (Q/V networks need concatenated)
        if "next.observation.state" in batch:
            next_robot_state = batch["next.observation.state"]
            if not isinstance(next_robot_state, torch.Tensor):
                next_robot_state = torch.tensor(next_robot_state, dtype=torch.float32)
            next_robot_state = next_robot_state.to(self.device)

            if "next.observation.environment_state" in batch:
                next_env_state = batch["next.observation.environment_state"]
                if not isinstance(next_env_state, torch.Tensor):
                    next_env_state = torch.tensor(next_env_state, dtype=torch.float32)
                next_env_state = next_env_state.to(self.device)
                next_state = torch.cat([next_robot_state, next_env_state], dim=-1)
            else:
                next_state = next_robot_state

            result["next.observation.state"] = self.normalizer.normalize_state(next_state)

        # Normalize action (MIN_MAX to [-1, 1])
        if "action" in batch:
            action = batch["action"]
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            action = action.to(self.device)
            result["action"] = self.normalizer.normalize_action(action)

        # Pass through other keys unchanged
        for key in PASSTHROUGH_KEYS:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    result[key] = val.to(self.device)
                else:
                    result[key] = val

        return result


@dataclass
class IQLPostprocessor:
    """
    Postprocessor for IQL-based policies that denormalizes actions.

    Used by both IQL-MLP and IDQL policies.
    """

    normalizer: Normalizer

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from policy output."""
        action_device = action.device
        action = action.to(self.normalizer.device)
        action_denorm = self.normalizer.denormalize_action(action)
        return action_denorm.to(action_device)


def make_idql_pre_post_processors(
    normalizer: Normalizer,
    device: str = "cpu",
) -> tuple[IDQLPreprocessor, IQLPostprocessor]:
    """
    Create pre/post processor pair for IDQL policy.

    IDQL needs a special preprocessor that:
    1. Uses MIN_MAX normalization for current states (matches LeRobot's DiffusionPolicy)
    2. Keeps observation.state and observation.environment_state SEPARATE
       (diffusion actor requires separate keys)
    3. Uses z-score normalization for next states (for Q/V networks)
    4. IDQLPolicy.update() handles concatenation for Q/V networks internally

    IMPORTANT: The normalizer must have state_min and state_max set for MIN_MAX normalization.

    Args:
        normalizer: Normalizer with state/action statistics (including state_min/state_max)
        device: Device to move tensors to during preprocessing

    Returns:
        Tuple of (preprocessor, postprocessor)
    """
    if normalizer.state_min is None or normalizer.state_max is None:
        raise ValueError(
            "IDQL preprocessor requires state_min and state_max for MIN_MAX normalization. "
            "Pass these when creating the Normalizer."
        )
    return (
        IDQLPreprocessor(normalizer=normalizer, device=device),
        IQLPostprocessor(normalizer=normalizer),
    )
