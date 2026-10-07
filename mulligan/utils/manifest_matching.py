"""Cyclic-aware nearest-state matching for sampler manifests.

Sampler manifests describe initial states by (x, y, yaw, ...) keys. Angular
columns (anything ending in ``_yaw``) must be compared cyclically: a manifest
yaw of ``5.04 rad`` is the same angle as a saved env_state yaw of
``-1.24 rad`` (since ``5.04 - 2*pi = -1.24``), and the matcher must treat
them as identical.

This module is the *single* source of truth for that cyclic match, shared by
``mulligan/sim/collect/quota.py``, ``mulligan/data/split_blind.py`` and
``mulligan/sampling/sobol.py``. Separate copies of the wrap logic can drift apart,
for instance when a manifest holds yaws outside ``[-pi, pi]``.

If you need to compare a state vector to a manifest, build a
``ManifestMatcher`` here rather than hand-rolling another wrap helper.

Implementation note: ``ManifestMatcher`` projects each angular column to
``(cos(theta), sin(theta))`` before feeding the manifest to ``cKDTree``.
Euclidean distance in that embedding is the chord distance
``sqrt(2 - 2*cos(delta_theta))``, which is monotone in the absolute cyclic
angular difference and equals it (to first order) for small differences.
This is the mathematically correct cyclic metric and works seamlessly with
``cKDTree`` for any number of angular columns. Tolerance values such as
``1e-3`` are unchanged in meaning for the small-difference regime that
matters for tolerance checks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


ANGULAR_KEY_SUFFIXES: tuple[str, ...] = ("_yaw",)


def wrap_to_pi(values: np.ndarray | float) -> np.ndarray:
    """Wrap angular values into ``[-pi, pi]``.

    Works for scalars and arrays. Use this when you need a delta-wrap
    primitive (e.g. computing minimum pairwise angular distance in a FPS
    coverage calculation); for nearest-state matching, use ``ManifestMatcher``
    instead.
    """
    return np.mod(np.asarray(values, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi


def angular_idxs_for_keys(keys: list[str]) -> list[int]:
    """Return positions of angular columns based on manifest key suffixes.

    Any key ending in ``_yaw`` is considered angular. Extend
    ``ANGULAR_KEY_SUFFIXES`` if new task families introduce additional
    angular conventions.
    """
    return [i for i, k in enumerate(keys) if k.endswith(ANGULAR_KEY_SUFFIXES)]


def embed_for_cyclic_match(
    arr: np.ndarray,
    angular_idxs: list[int],
) -> np.ndarray:
    """Project ``arr`` into a cyclic-aware Euclidean embedding.

    Each angular column ``theta`` is replaced by two columns
    ``(cos(theta), sin(theta))``. Linear (non-angular) columns are kept
    unchanged. Euclidean distance in the result is the chord distance, which
    is cyclic-correct.

    Returns a contiguous float64 array. The output dimensionality is
    ``arr.shape[-1] + len(angular_idxs)``.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if not angular_idxs:
        return arr if arr.dtype is np.dtype(np.float64) else arr.astype(np.float64)
    angular_set = set(angular_idxs)
    columns: list[np.ndarray] = []
    for col_idx in range(arr.shape[-1]):
        col = arr[..., col_idx : col_idx + 1]
        if col_idx in angular_set:
            columns.append(np.cos(col))
            columns.append(np.sin(col))
        else:
            columns.append(col)
    return np.concatenate(columns, axis=-1)


class ManifestMatcher:
    """Nearest-manifest-entry lookup with cyclic-aware Euclidean distance.

    Build once from a manifest_arr (shape ``(N, D)``) plus the indices of
    angular columns. Each ``query(vec)`` returns ``(idx, dist)`` where
    ``dist`` is the chord distance in the cos/sin embedding — equal to the
    cyclic Euclidean distance for small angular differences. Use
    ``query_within_tolerance(vec, tol)`` when you want a loud failure on
    out-of-tolerance queries (i.e. defensive cross-checks).

    The matcher is the canonical home for "match a state vector to a
    manifest row". Direct ``cKDTree`` construction in calling code is the
    pattern that misses the yaw wrap; prefer this class.
    """

    def __init__(
        self,
        manifest_arr: np.ndarray,
        angular_idxs: list[int],
    ) -> None:
        manifest_arr = np.asarray(manifest_arr, dtype=np.float64)
        if manifest_arr.ndim != 2:
            raise ValueError(f"manifest_arr must be 2D (N, D), got shape {manifest_arr.shape}")
        if len(manifest_arr) == 0:
            raise ValueError("manifest_arr must contain at least one row")
        max_idx = manifest_arr.shape[1] - 1
        for ai in angular_idxs:
            if ai < 0 or ai > max_idx:
                raise ValueError(
                    f"angular index {ai} out of range for manifest with "
                    f"{manifest_arr.shape[1]} columns"
                )
        self.manifest_arr = manifest_arr
        self.angular_idxs = list(angular_idxs)
        self._embedded = embed_for_cyclic_match(manifest_arr, self.angular_idxs)
        self._tree = cKDTree(self._embedded)
        self._n_keys = manifest_arr.shape[1]

    @classmethod
    def from_keys(cls, manifest_arr: np.ndarray, keys: list[str]) -> "ManifestMatcher":
        """Convenience constructor that derives angular columns from key names."""
        return cls(manifest_arr, angular_idxs_for_keys(keys))

    def query(self, vec: np.ndarray | list[float] | tuple[float, ...]) -> tuple[int, float]:
        """Return ``(manifest_idx, chord_distance)`` for the nearest manifest row."""
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape != (self._n_keys,):
            raise ValueError(f"expected vector shape {(self._n_keys,)}, got {arr.shape}")
        embedded = embed_for_cyclic_match(arr[None, :], self.angular_idxs)[0]
        dist, idx = self._tree.query(embedded, k=1)
        return int(idx), float(dist)

    def query_within_tolerance(
        self,
        vec: np.ndarray | list[float] | tuple[float, ...],
        tolerance: float,
    ) -> tuple[int, float]:
        """Like :meth:`query`, but raise ``ValueError`` if ``dist > tolerance``."""
        idx, dist = self.query(vec)
        if dist > tolerance:
            raise ValueError(
                f"state did not match quota manifest within tolerance: "
                f"dist={dist:.4g} > {tolerance}"
            )
        return idx, dist

    def min_pairwise_distance(self) -> float:
        """Minimum pairwise chord distance between distinct manifest rows.

        Used as a self-consistency check: if two manifest rows are closer
        than ``2 * match_tolerance`` they are ambiguous and matches can flip
        between them under sub-tolerance noise.
        """
        if len(self._embedded) <= 1:
            return float("inf")
        dists, _ = self._tree.query(self._embedded, k=2)
        return float(dists[:, 1].min())


def min_pairwise_cyclic_distance(points: np.ndarray, angular_idxs: list[int]) -> float:
    """Minimum pairwise *delta-wrap* distance among ``points``.

    This is the delta-wrap counterpart of :meth:`ManifestMatcher.min_pairwise_distance`
    and is what the FPS coverage diagnostics report. Kept on the
    delta-wrap metric so logged ``coverage_p95`` / ``min_pairwise`` values
    stay numerically comparable across rounds.

    For new matching code, prefer ``ManifestMatcher.min_pairwise_distance``.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return float("inf")
    diffs = points[:, None, :] - points[None, :, :]
    for ai in angular_idxs:
        diffs[..., ai] = wrap_to_pi(diffs[..., ai])
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))
