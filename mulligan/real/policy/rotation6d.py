"""Continuous 6D rotation representation (Zhou et al. 2019).

Zhou, Barnes, Lu, Yang, Li — "On the Continuity of Rotation Representations
in Neural Networks" (CVPR 2019). A 3x3 rotation matrix is mapped to a 6D
vector by dropping its third column (keeping the first two columns,
flattened to 6 numbers). The inverse recovers the full matrix by Gram-Schmidt
orthonormalisation of those two columns and a cross product for the third.
Unlike euler roll/pitch/yaw, this representation is CONTINUOUS over SO(3):
there is no ``+/-pi`` wrap-around seam, so a regression network never has to
predict a value that jumps discontinuously as the true rotation moves smoothly.

Why this exists: the euler position target
piles roll/yaw at both ``+pi`` and ``-pi`` (the same physical rotation lands at
opposite normalized extremes under MIN_MAX), which is a corrupt regression
target. The 6D form removes that seam.

Convention contract (single source of truth for the re-cut):
- Euler is interpreted as scipy extrinsic ``"xyz"`` = (roll about x, pitch about
  y, yaw about z) in radians. This is the DROID ``cartesian_position``
  orientation convention. Both the forward (euler -> r6) and inverse
  (r6 -> euler) use this SAME convention, so the dataset re-cut and the eventual
  eval-side decode are exact mutual inverses regardless of which physical
  convention the robot ultimately uses — the round-trip is self-consistent and
  pinned by unit tests.
- The 6D vector layout is ``[m[:,0] (3), m[:,1] (3)]`` = the first column
  followed by the second column of the rotation matrix.

These helpers are pure numpy/scipy and have NO silent fallbacks: malformed
shapes raise, and a degenerate (rank-deficient) 6D input raises rather than
returning a garbage rotation.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# scipy extrinsic xyz euler = roll(x), pitch(y), yaw(z). Lowercase = extrinsic.
EULER_CONVENTION = "xyz"

# 6D rotation has exactly 6 numbers (two columns of a 3x3 matrix).
R6_DIM = 6


def rotation_matrix_to_r6(matrix: np.ndarray) -> np.ndarray:
    """Map a rotation matrix (..., 3, 3) to its 6D representation (..., 6).

    Keeps the first two columns of the matrix (Zhou et al. 2019). Vectorised over
    any leading batch dimensions.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation_matrix_to_r6 expects (..., 3, 3); got {matrix.shape}")
    col0 = matrix[..., :, 0]
    col1 = matrix[..., :, 1]
    return np.concatenate([col0, col1], axis=-1).astype(np.float32)


def r6_to_rotation_matrix(r6: np.ndarray) -> np.ndarray:
    """Recover a rotation matrix (..., 3, 3) from a 6D vector (..., 6).

    Gram-Schmidt: b0 = normalize(a0); b1 = normalize(a1 - <b0,a1> b0);
    b2 = b0 x b1. Returns an orthonormal right-handed matrix with the recovered
    columns ``[b0, b1, b2]``.

    Raises on a degenerate input (zero-norm first column, or first two columns
    parallel so the orthogonalised second column has zero norm) — a silent
    normalize-by-epsilon would emit a non-rotation matrix and corrupt the decode.
    """
    r6 = np.asarray(r6, dtype=np.float64)
    if r6.shape[-1] != R6_DIM:
        raise ValueError(f"r6_to_rotation_matrix expects (..., 6); got {r6.shape}")

    a0 = r6[..., 0:3]
    a1 = r6[..., 3:6]

    n0 = np.linalg.norm(a0, axis=-1, keepdims=True)
    if np.any(n0 <= 1e-8):
        raise ValueError("Degenerate r6: first column has ~zero norm; cannot orthonormalize.")
    b0 = a0 / n0

    # Remove the b0 component from a1, then normalize.
    dot = np.sum(b0 * a1, axis=-1, keepdims=True)
    a1_orth = a1 - dot * b0
    n1 = np.linalg.norm(a1_orth, axis=-1, keepdims=True)
    if np.any(n1 <= 1e-8):
        raise ValueError("Degenerate r6: first two columns are parallel; cannot orthonormalize.")
    b1 = a1_orth / n1

    b2 = np.cross(b0, b1)
    # Stack as columns: matrix[..., :, i] = b_i.
    matrix = np.stack([b0, b1, b2], axis=-1)
    return matrix


def euler_to_r6(euler: np.ndarray) -> np.ndarray:
    """Map euler roll/pitch/yaw (..., 3) (radians) to 6D rotation (..., 6).

    Uses the module ``EULER_CONVENTION`` (extrinsic xyz).
    """
    euler = np.asarray(euler, dtype=np.float64)
    if euler.shape[-1] != 3:
        raise ValueError(f"euler_to_r6 expects (..., 3); got {euler.shape}")
    flat = euler.reshape(-1, 3)
    matrix = Rotation.from_euler(EULER_CONVENTION, flat).as_matrix()  # (N, 3, 3)
    r6 = rotation_matrix_to_r6(matrix)  # (N, 6)
    return r6.reshape(*euler.shape[:-1], R6_DIM)


def r6_to_euler(r6: np.ndarray) -> np.ndarray:
    """Map 6D rotation (..., 6) back to euler roll/pitch/yaw (..., 3) (radians).

    Inverse of :func:`euler_to_r6` up to the periodicity of euler angles (the
    recovered rotation is identical; the euler triple is the canonical one scipy
    returns for ``EULER_CONVENTION``).
    """
    r6 = np.asarray(r6, dtype=np.float64)
    if r6.shape[-1] != R6_DIM:
        raise ValueError(f"r6_to_euler expects (..., 6); got {r6.shape}")
    flat = r6.reshape(-1, R6_DIM)
    matrix = r6_to_rotation_matrix(flat)  # (N, 3, 3)
    euler = Rotation.from_matrix(matrix).as_euler(EULER_CONVENTION)  # (N, 3)
    return euler.reshape(*r6.shape[:-1], 3).astype(np.float32)
