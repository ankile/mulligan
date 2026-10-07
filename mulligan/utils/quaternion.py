"""Quaternion utilities (wxyz, scalar-first, as used by robosuite/MuJoCo).

z_rot_to_quat_wxyz() applies [cos(angle/2), 0, 0, sin(angle/2)] directly
WITHOUT mapping to [0, 2*pi). The Square tasks (NutAssemblySquare, Square_D1)
sample yaw from [0, 2*pi) natively, so callers map the angle first:
z_rot_to_quat_wxyz(yaw % (2*pi)).

Note: MuJoCo does not canonicalize quaternion sign in body_xquat.
  q and -q represent the same rotation but produce DIFFERENT observations.
  Using the wrong convention causes 5-10 stddev out-of-distribution
  observations that break the policy.
"""

import numpy as np


def z_rot_to_quat_wxyz(angle: float) -> np.ndarray:
    """Convert a z-rotation angle to a wxyz quaternion using the direct formula.

    Applies [cos(angle/2), 0, 0, sin(angle/2)] directly to the input angle
    WITHOUT mapping to [0, 2*pi). This matches robosuite's UniformRandomSampler
    which applies the formula to the raw sampled angle.

    The Square tasks' native sampler uses [0, 2*pi), so callers map the angle into
    that range first (each sampler's to_qpos() does).

    Args:
        angle: Rotation angle around z-axis in radians (any range).

    Returns:
        Quaternion in [w, x, y, z] format (MuJoCo/robosuite convention).
    """
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
