"""Environment state dimension names for the Square simulation tasks.

This module provides semantic names for the environment_state (object-state)
observation dimensions of the sim tasks (``NutAssemblySquare`` and MimicGen's
``Square_D0`` / ``Square_D1``). These names are written into the dataset metadata.

The environment_state corresponds to the "object-state" observation in Robosuite,
which concatenates all observations with the "object" modality.
"""


def _vec(prefix: str) -> list[str]:
    return [f"{prefix}_{axis}" for axis in "xyz"]


def _pos(obj: str) -> list[str]:
    return _vec(f"{obj}_pos")


def _quat(obj: str) -> list[str]:
    return [f"{obj}_quat_{axis}" for axis in "xyzw"]


def _pose(obj: str) -> list[str]:
    return _pos(obj) + _quat(obj)


def _nut() -> list[str]:
    """Nut pose relative to the end effector (7), then its absolute pose (7)."""
    return _pose("nut_to_eef") + _pose("nut")


def get_environment_state_names(env_name: str, env_state_shape: int) -> list[str]:
    """
    Get descriptive names for environment_state dimensions based on the environment.

    Args:
        env_name: Name of the sim environment (e.g., "NutAssemblySquare", "Square_D1_Panda")
        env_state_shape: Number of dimensions in environment_state (14: nut; 17: nut + peg position)

    Returns:
        List of descriptive names for each dimension
    """
    if "square" in env_name.lower():
        if env_state_shape == 17:
            return _nut() + _pos("peg")
        if env_state_shape == 14:
            return _nut()

    # No fallback - fail explicitly so we add proper names before collecting data
    raise ValueError(
        f"Environment state names not defined for environment '{env_name}' with shape {env_state_shape}. "
        f"Please add a case for this environment in mulligan/data/env_state_names.py to ensure "
        f"proper semantic naming of environment state dimensions."
    )
