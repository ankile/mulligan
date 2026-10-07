"""Unit tests for the continuous 6D rotation representation (Zhou et al. 2019).

Pins the round-trip correctness that the 6D-rotation position-action target
depends on: euler -> r6 -> rotation matrix / euler must reconstruct the
SAME rotation to < 1e-5, including rotations near the euler +/-pi seam, where a
MIN_MAX-normalized euler target is discontinuous.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mulligan.real.policy.rotation6d import (
    EULER_CONVENTION,
    R6_DIM,
    euler_to_r6,
    r6_to_euler,
    r6_to_rotation_matrix,
    rotation_matrix_to_r6,
)

TOL = 1e-5


def _rotations_equal(mat_a: np.ndarray, mat_b: np.ndarray, tol: float = TOL) -> bool:
    """True if two rotation matrices represent the same rotation (geodesic angle)."""
    # Relative rotation; its angle is 0 iff identical.
    rel = mat_a.swapaxes(-1, -2) @ mat_b
    angle = Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude()
    return bool(np.all(angle < tol))


def test_random_rotation_roundtrip_matrix() -> None:
    rots = Rotation.random(2000, random_state=42)
    matrices = rots.as_matrix()  # (N, 3, 3)

    r6 = rotation_matrix_to_r6(matrices)
    assert r6.shape == (2000, R6_DIM)

    recovered = r6_to_rotation_matrix(r6)
    assert recovered.shape == (2000, 3, 3)
    # Each recovered matrix is a proper rotation (det +1, orthonormal).
    dets = np.linalg.det(recovered)
    assert np.allclose(dets, 1.0, atol=TOL)
    assert _rotations_equal(matrices, recovered)


def test_random_euler_roundtrip() -> None:
    rng = np.random.default_rng(1)
    euler = rng.uniform(-np.pi, np.pi, size=(5000, 3))

    r6 = euler_to_r6(euler)
    assert r6.shape == (5000, R6_DIM)

    euler_back = r6_to_euler(r6)
    # Euler triples may differ by periodicity; compare the actual rotations.
    mat_in = Rotation.from_euler(EULER_CONVENTION, euler).as_matrix()
    mat_back = Rotation.from_euler(EULER_CONVENTION, euler_back).as_matrix()
    assert _rotations_equal(mat_in, mat_back)


def test_seam_rotations_roundtrip() -> None:
    """Rotations near the euler +/-pi seam — the whole reason for the re-cut."""
    seam = np.array(
        [
            [np.pi, 0.0, 0.0],
            [-np.pi, 0.0, 0.0],
            [np.pi - 1e-6, 0.0, 0.0],
            [-np.pi + 1e-6, 0.0, 0.0],
            [0.0, 0.0, np.pi],
            [0.0, 0.0, -np.pi],
            [np.pi, 0.0, np.pi],
            [-np.pi, 0.0, -np.pi],
            [3.14159, 0.0, -3.14159],
        ],
        dtype=np.float64,
    )
    r6 = euler_to_r6(seam)
    mat_in = Rotation.from_euler(EULER_CONVENTION, seam).as_matrix()
    mat_back = r6_to_rotation_matrix(r6)
    assert _rotations_equal(mat_in, mat_back)

    # The r6 encoding of +pi and -pi about the same axis is IDENTICAL (no seam):
    # the two rows are the same physical rotation, so their r6 must match.
    r6_plus = euler_to_r6(np.array([np.pi, 0.0, 0.0]))
    r6_minus = euler_to_r6(np.array([-np.pi, 0.0, 0.0]))
    assert np.allclose(r6_plus, r6_minus, atol=TOL)


def test_r6_dims_are_seam_free_bounded() -> None:
    """r6 components live in [-1, 1] and vary continuously (no bimodal +/-1 pileup)."""
    rng = np.random.default_rng(2)
    euler = rng.uniform(-np.pi, np.pi, size=(20000, 3))
    r6 = euler_to_r6(euler)
    assert r6.min() >= -1.0 - TOL
    assert r6.max() <= 1.0 + TOL
    # Sanity: a single yaw sweep through +/-pi produces a continuous r6 path, not
    # a jump. Adjacent yaw samples differ by a small step in r6.
    yaw = np.linspace(-np.pi, np.pi, 100000)
    euler_sweep = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], axis=1)
    r6_sweep = euler_to_r6(euler_sweep)
    step = np.linalg.norm(np.diff(r6_sweep, axis=0), axis=1)
    # Max per-sample step is tiny (continuous); the euler target would jump ~2pi.
    assert step.max() < 1e-3


def test_shapes_and_single_vector() -> None:
    euler = np.array([0.1, -0.2, 0.3])
    r6 = euler_to_r6(euler)
    assert r6.shape == (R6_DIM,)
    euler_back = r6_to_euler(r6)
    assert euler_back.shape == (3,)
    mat_in = Rotation.from_euler(EULER_CONVENTION, euler).as_matrix()
    mat_back = Rotation.from_euler(EULER_CONVENTION, euler_back).as_matrix()
    assert _rotations_equal(mat_in[None], mat_back[None])


def test_degenerate_r6_raises() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        r6_to_rotation_matrix(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="parallel"):
        # First two columns identical -> orthogonalised second column is zero.
        r6_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def test_bad_shapes_raise() -> None:
    with pytest.raises(ValueError):
        euler_to_r6(np.zeros(4))
    with pytest.raises(ValueError):
        r6_to_euler(np.zeros(5))
    with pytest.raises(ValueError):
        r6_to_rotation_matrix(np.zeros((2, 7)))
    with pytest.raises(ValueError):
        rotation_matrix_to_r6(np.zeros((3, 4)))
