"""Extract initial object poses from environment-state observations.

The Square (nut / peg) extractors the Sobol samplers, the collection scripts and the
dataset splitters use to read an episode's initial state out of
``observation.environment_state``.
"""

import numpy as np


def extract_nut_pose_from_env_state(
    env_state: np.ndarray, task: str = "NutAssemblySquare"
) -> tuple[float, float, float]:
    """
    Extract (x, y, yaw) from observation.environment_state.

    For NutAssemblySquare, the environment state (14D) is structured as:
    [nut_to_eef_pos(3), nut_to_eef_quat(4), nut_pos(3), nut_quat(4)]

    nut_pos is at indices [7:10]
    nut_quat is at indices [10:14] in (w, x, y, z) format

    Args:
        env_state: Environment state array (14D for NutAssemblySquare)
        task: Task name (currently only NutAssemblySquare supported)

    Returns:
        (x, y, yaw) tuple
    """
    if task not in ("NutAssemblySquare", "Square_D1"):
        raise ValueError(
            f"Task {task} not supported. Only NutAssemblySquare and Square_D1 implemented."
        )

    expected_dim = 14 if task == "NutAssemblySquare" else 17
    if len(env_state) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim}D environment state for {task}, got {len(env_state)}D"
        )

    # Extract nut position
    nut_pos = env_state[7:10]
    x, y = nut_pos[0], nut_pos[1]

    # Extract nut quaternion in xyzw format (robosuite converts to xyzw via T.convert_quat)
    nut_quat = env_state[10:14]

    # Convert quaternion to yaw
    # For rotation around Z only: yaw = 2 * atan2(qz, qw)
    # nut_quat is (x, y, z, w) = (qx, qy, qz, qw) in xyzw convention
    qx, qy, qz, qw = nut_quat
    yaw = 2 * np.arctan2(qz, qw)
    yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi]

    return float(x), float(y), float(yaw)


def extract_peg_pos_from_env_state(
    env_state: np.ndarray,
) -> tuple[float, float, float]:
    """Extract (x, y, z) peg position from 17D Square_D1 environment state.

    Square_D1 env state (17D):
    [nut_to_eef_pos(3), nut_to_eef_quat(4), nut_pos(3), nut_quat(4), peg_pos(3)]

    Peg position is at indices [14:17].
    """
    if len(env_state) != 17:
        raise ValueError(f"Expected 17D environment state for Square_D1, got {len(env_state)}D")
    return float(env_state[14]), float(env_state[15]), float(env_state[16])


def extract_sampler_state_from_env_state(
    env_state: np.ndarray,
    task: str = "NutAssemblySquare",
) -> tuple[float, ...]:
    """Extract the task-specific initial state used by continuous samplers.

    Returns:
        NutAssemblySquare: (nut_x, nut_y, nut_yaw)
        Square_D1: (nut_x, nut_y, nut_yaw, peg_x, peg_y)
    """
    nut_x, nut_y, nut_yaw = extract_nut_pose_from_env_state(env_state, task=task)
    if task == "Square_D1":
        peg_x, peg_y, _ = extract_peg_pos_from_env_state(env_state)
        return nut_x, nut_y, nut_yaw, peg_x, peg_y
    return nut_x, nut_y, nut_yaw
