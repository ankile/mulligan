"""Tests for the shared cyclic-aware manifest matcher.

These tests are the regression net for yaw wrapping: a manifest yaw outside
``[-pi, pi]`` must match a saved env_state yaw wrapped into that range. They
also cover the symmetric ``-pi``/``+pi`` boundary case that a matcher which
wraps values before building the KDTree gets wrong (it gives ``2*pi`` for
boundary-aligned states).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mulligan.utils.manifest_matching import (
    ANGULAR_KEY_SUFFIXES,
    ManifestMatcher,
    angular_idxs_for_keys,
    embed_for_cyclic_match,
    min_pairwise_cyclic_distance,
    wrap_to_pi,
)


def test_angular_key_suffix_default():
    assert "_yaw" in ANGULAR_KEY_SUFFIXES


def test_angular_idxs_for_square_broad_keys():
    keys = ["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"]
    assert angular_idxs_for_keys(keys) == [2]


def test_angular_idxs_for_square_narrow_keys():
    keys = ["nut_x", "nut_y", "nut_yaw"]
    assert angular_idxs_for_keys(keys) == [2]


def test_angular_idxs_for_no_angular_keys():
    assert angular_idxs_for_keys(["x", "y", "z"]) == []


def test_angular_idxs_for_multi_angular_keys():
    keys = ["nut_yaw", "x", "peg_yaw"]
    assert angular_idxs_for_keys(keys) == [0, 2]


def test_wrap_to_pi_scalars_and_arrays():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(math.pi) == pytest.approx(math.pi) or wrap_to_pi(math.pi) == pytest.approx(
        -math.pi
    )
    assert wrap_to_pi(2 * math.pi) == pytest.approx(0.0, abs=1e-12)
    assert wrap_to_pi(5.04) == pytest.approx(5.04 - 2 * math.pi, abs=1e-12)
    arr = np.array([0.0, math.pi / 2, 5.04, -5.04])
    expected = np.array([0.0, math.pi / 2, 5.04 - 2 * math.pi, -5.04 + 2 * math.pi])
    np.testing.assert_allclose(wrap_to_pi(arr), expected, atol=1e-12)


def test_embed_for_cyclic_match_no_angular():
    arr = np.array([[1.0, 2.0, 3.0]])
    out = embed_for_cyclic_match(arr, [])
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == np.float64


def test_embed_for_cyclic_match_one_angular():
    arr = np.array([[0.5, 0.0, 0.7]])
    out = embed_for_cyclic_match(arr, [1])
    expected = np.array([[0.5, math.cos(0.0), math.sin(0.0), 0.7]])
    np.testing.assert_allclose(out, expected)


def test_embed_for_cyclic_match_2pi_wrap_invariance():
    arr_a = np.array([[5.04]])
    arr_b = np.array([[5.04 - 2 * math.pi]])
    out_a = embed_for_cyclic_match(arr_a, [0])
    out_b = embed_for_cyclic_match(arr_b, [0])
    np.testing.assert_allclose(out_a, out_b, atol=1e-12)


def test_matcher_query_exact_square_narrow_match():
    manifest = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    matcher = ManifestMatcher.from_keys(manifest, ["nut_x", "nut_y", "nut_yaw"])
    idx, dist = matcher.query([1.0, 0.0, 0.0])
    assert idx == 1
    assert dist < 1e-9


def test_matcher_query_square_broad_with_2pi_yaw_wrap():
    """Regression: manifest yaw=5.04 matches saved -1.24."""
    keys = ["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"]
    manifest = np.array(
        [
            [0.0, 0.0, 5.04, 0.1, 0.1],
            [1.0, 0.0, 0.5, 0.1, 0.1],
        ]
    )
    matcher = ManifestMatcher.from_keys(manifest, keys)
    wrapped_yaw = 5.04 - 2.0 * math.pi  # ~ -1.2432
    idx, dist = matcher.query([0.0, 0.0, wrapped_yaw, 0.1, 0.1])
    assert idx == 0
    assert dist < 1e-3


def test_matcher_handles_pi_boundary_correctly():
    """``yaw = pi`` and ``yaw = -pi + eps`` are the same angle and must match.

    This is what the prior value-wrap implementation got wrong. The current
    chord-distance embedding handles it correctly.
    """
    eps = 1e-4
    manifest = np.array([[math.pi - eps]])
    matcher = ManifestMatcher(manifest, [0])
    idx, dist = matcher.query([-math.pi + eps])
    assert idx == 0
    # Two angles 2*eps apart in cyclic distance => chord distance approx 2*eps.
    assert dist == pytest.approx(2 * eps, rel=1e-3, abs=1e-6)


def test_matcher_query_within_tolerance_raises_on_miss():
    manifest = np.array([[0.0, 0.0, 0.0]])
    matcher = ManifestMatcher(manifest, [2])
    with pytest.raises(ValueError, match="did not match quota manifest"):
        matcher.query_within_tolerance([1.0, 1.0, 0.0], tolerance=0.5)


def test_matcher_query_within_tolerance_passes_on_hit():
    manifest = np.array([[0.0, 0.0, 5.04]])
    matcher = ManifestMatcher(manifest, [2])
    idx, dist = matcher.query_within_tolerance([0.0, 0.0, 5.04 - 2 * math.pi], tolerance=1e-3)
    assert idx == 0
    assert dist < 1e-3


def test_matcher_rejects_wrong_dim_query():
    manifest = np.array([[0.0, 0.0, 0.0]])
    matcher = ManifestMatcher(manifest, [2])
    with pytest.raises(ValueError, match="expected vector shape"):
        matcher.query([0.0, 0.0])


def test_matcher_rejects_empty_manifest():
    with pytest.raises(ValueError, match="at least one row"):
        ManifestMatcher(np.zeros((0, 3)), [2])


def test_matcher_rejects_bad_angular_idx():
    with pytest.raises(ValueError, match="out of range"):
        ManifestMatcher(np.zeros((1, 3)), [5])


def test_matcher_multi_angular_columns():
    """The cos/sin embedding scales to multiple angular columns."""
    keys = ["nut_yaw", "x", "peg_yaw"]
    manifest = np.array(
        [
            [0.0, 0.0, 0.0],
            [math.pi / 2, 0.0, math.pi / 2],
        ]
    )
    matcher = ManifestMatcher.from_keys(manifest, keys)
    # Query yaw0=5*pi/2 (== pi/2 cyclically) and yaw2=-3pi/2 (== pi/2 cyclically),
    # x=0 -> row 1
    idx, dist = matcher.query([5.0 * math.pi / 2, 0.0, -3.0 * math.pi / 2])
    assert idx == 1
    assert dist < 1e-3


def test_matcher_min_pairwise_distance_picks_close_pair():
    manifest = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [5.0, 0.0, 0.0]])
    matcher = ManifestMatcher(manifest, [2])
    d = matcher.min_pairwise_distance()
    assert d == pytest.approx(0.001, abs=1e-9)


def test_min_pairwise_cyclic_distance_treats_yaw_circularly():
    # yaw=0.0 vs yaw=2*pi-eps should be eps apart (delta-wrap), not 2*pi-eps.
    eps = 0.01
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2 * math.pi - eps]])
    d = min_pairwise_cyclic_distance(points, [2])
    assert d == pytest.approx(eps, abs=1e-9)


def test_min_pairwise_cyclic_distance_empty_and_single():
    assert math.isinf(min_pairwise_cyclic_distance(np.zeros((0, 3)), [2]))
    assert math.isinf(min_pairwise_cyclic_distance(np.zeros((1, 3)), [2]))
