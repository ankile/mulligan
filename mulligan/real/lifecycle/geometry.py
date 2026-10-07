"""Normalized periodic geometry, coverage metrics, and samplers.

The math here is the single source of truth shared by the start-design and
manifest tooling of both real task lines. Arrays are
``(n, K)`` state arrays over a task's JOINT sampling space; which columns are
periodic is given by a boolean ``periodic_mask`` (length ``K``). When
``periodic_mask`` is omitted the 3-DOF convention ``(x, y, periodic yaw)``
is assumed.

For a task with no sampled placements the joint space IS the 3-DOF pen, so all
of this reduces exactly to the original (x, y, yaw) behavior. For a task with a
grid-snapped placement (e.g. the marker_d2 holder) the placement's (x, y) are
just two more continuous columns in the same metric space — Sobol covers them
and FPS diversifies them with the same machinery — then snapped on output.

Conventions:
- ``normalise`` maps each dim to the unit cube; periodic dims wrap to [0, 1).
- distances are Euclidean in the unit cube with ``min(|d|, 1 - |d|)`` on
  periodic columns.
- coverage "fill" quantiles measure NN distance from a dense test cloud to a
  support set; cells/entropy discretize the unit cube on per-dim grid dims.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc

from mulligan.real.lifecycle.tasks import RealTaskSpec


def wrap_angle(values: np.ndarray | float) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return ((np.asarray(values, dtype=np.float64) + math.pi) % (2.0 * math.pi)) - math.pi


def _resolve_mask(periodic_mask: np.ndarray | None, k: int) -> np.ndarray:
    """Periodic-column mask of length ``k``. Defaults to the 3-DOF
    ``(x, y, periodic yaw)`` convention when ``None``; an explicit mask is required for any other
    dimensionality."""
    if periodic_mask is not None:
        mask = np.asarray(periodic_mask, dtype=bool)
        if mask.shape != (k,):
            raise ValueError(f"periodic_mask shape {mask.shape} != ({k},)")
        return mask
    if k == 3:
        return np.array([False, False, True])
    raise ValueError(f"periodic_mask is required for {k}-D states (no 3-DOF default)")


def normalise(
    arr: np.ndarray, bounds: np.ndarray, periodic_mask: np.ndarray | None = None
) -> np.ndarray:
    """Map (n, K) states into the unit cube; periodic columns wrapped then scaled."""
    arr = np.asarray(arr, dtype=np.float64)
    k = arr.shape[1] if arr.ndim == 2 else bounds.shape[0]
    mask = _resolve_mask(periodic_mask, k)
    out = np.empty_like(arr, dtype=np.float64)
    for d in range(k):
        if mask[d]:
            out[:, d] = (wrap_angle(arr[:, d]) + math.pi) / (2.0 * math.pi)
        else:
            out[:, d] = (arr[:, d] - bounds[d, 0]) / (bounds[d, 1] - bounds[d, 0])
    return out


def unit_dist_to_point(
    unit: np.ndarray,
    point: np.ndarray,
    periodic_mask: np.ndarray | None = None,
    axis_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Periodic L2 distance from each unit-cube row to one unit-cube point.

    ``axis_weights`` (length K) optionally scales each dimension's contribution
    BEFORE the L2 norm, so a weight < 1 makes that axis count for less when
    measuring "farness" (e.g. downweight grid-snapped placement dims so coverage
    spends its diversity budget on the pen pose rather than the holder cell).
    ``None`` ⇒ all-ones (the unweighted distance)."""
    mask = _resolve_mask(periodic_mask, unit.shape[1])
    delta = np.abs(unit - point[None, :])
    for d in np.nonzero(mask)[0]:
        delta[:, d] = np.minimum(delta[:, d], 1.0 - delta[:, d])
    if axis_weights is not None:
        delta = delta * np.asarray(axis_weights, dtype=np.float64)[None, :]
    return np.linalg.norm(delta, axis=1)


def pairwise_min_distance(
    candidates: np.ndarray,
    support: np.ndarray,
    bounds: np.ndarray,
    periodic_mask: np.ndarray | None = None,
    axis_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Min periodic unit-cube distance from each candidate to the support set.

    ``axis_weights`` is forwarded to :func:`unit_dist_to_point` (``None`` ⇒
    unweighted)."""
    if len(support) == 0:
        return np.full(len(candidates), np.inf)
    mask = _resolve_mask(periodic_mask, bounds.shape[0])
    cu = normalise(candidates, bounds, mask)
    su = normalise(support, bounds, mask)
    out = np.full(len(candidates), np.inf)
    for point in su:
        out = np.minimum(out, unit_dist_to_point(cu, point, mask, axis_weights))
    return out


def fill_quantile(
    test_cloud: np.ndarray,
    support: np.ndarray,
    bounds: np.ndarray,
    q: float = 0.95,
    periodic_mask: np.ndarray | None = None,
) -> float:
    """Quantile of NN distances from a dense test cloud to the support set.

    ``q`` is the 4th positional argument; ``periodic_mask`` follows it."""
    return float(np.quantile(pairwise_min_distance(test_cloud, support, bounds, periodic_mask), q))


def sobol_points(bounds: np.ndarray, seed: int, n: int, start: int = 0) -> np.ndarray:
    """Scrambled Sobol states in physical bounds (base-2 block, then sliced).

    Dimensionality is ``bounds.shape[0]`` so a joint pen+placement space is sampled
    in ONE Sobol stream (placement dims are snapped separately by the caller)."""
    d = bounds.shape[0]
    end = start + n
    m = math.ceil(math.log2(max(2, end)))
    raw = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m=m)[start:end]
    return bounds[:, 0][None, :] + raw * (bounds[:, 1] - bounds[:, 0])[None, :]


def uniform_points(bounds: np.ndarray, seed: int, n: int) -> np.ndarray:
    """Flat-uniform states in physical bounds (``np.random.default_rng``)."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n, bounds.shape[0]))
    return bounds[:, 0][None, :] + raw * (bounds[:, 1] - bounds[:, 0])[None, :]


def snap_to_grid(arr: np.ndarray, bounds: np.ndarray, snap: np.ndarray) -> np.ndarray:
    """Snap each column ``d`` with finite ``snap[d]`` to its grid (``lo + round(.)*snap``),
    clipped to ``[lo, hi]``; columns with ``nan`` snap pass through unchanged.

    Operator-convenience quantization applied to sampled-placement dims AFTER sampling;
    a no-op (all-nan snap) for tasks whose only sampled object is the continuous pen."""
    snap = np.asarray(snap, dtype=np.float64)
    out = np.array(arr, dtype=np.float64, copy=True)
    if out.size == 0:
        return out
    for d in range(out.shape[1]):
        if np.isfinite(snap[d]):
            lo, hi = bounds[d, 0], bounds[d, 1]
            out[:, d] = np.clip(lo + np.round((out[:, d] - lo) / snap[d]) * snap[d], lo, hi)
    return out


def cell_ids(
    arr: np.ndarray,
    bounds: np.ndarray,
    dims: tuple[int, ...],
    periodic_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grid-cell id per state over ``dims`` cells of the unit cube (one entry per dim)."""
    mask = _resolve_mask(periodic_mask, len(dims))
    unit = normalise(arr, bounds, mask)
    ids = np.zeros(len(arr), dtype=np.int64)
    mult = 1
    for d in range(len(dims)):
        c = np.clip((unit[:, d] * dims[d]).astype(int), 0, dims[d] - 1)
        ids += c * mult
        mult *= dims[d]
    return ids


def cell_entropy(
    arr: np.ndarray,
    bounds: np.ndarray,
    dims: tuple[int, ...],
    periodic_mask: np.ndarray | None = None,
) -> float:
    """Normalized Shannon entropy of grid-cell occupancy (1.0 = uniform)."""
    n_cells = int(np.prod(dims))
    if n_cells == 1:
        # A single cell carries no occupancy information; normalization would divide
        # by math.log(1) == 0 (-> nan). Entropy of a 1-cell grid is identically 0.
        return 0.0
    counts = np.bincount(cell_ids(arr, bounds, dims, periodic_mask), minlength=n_cells).astype(
        np.float64
    )
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / math.log(n_cells))


def edge_norm(arr: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Sim-analog edge measure on symmetric pen bounds: max(|x|/x_max, |y|/y_max)."""
    return np.maximum(np.abs(arr[:, 0]) / bounds[0, 1], np.abs(arr[:, 1]) / bounds[1, 1])


def dist_to_fixed(arr: np.ndarray, fixed_xy: tuple[float, float]) -> np.ndarray:
    """Planar distance from each state to a fixed scene object (meters)."""
    return np.hypot(arr[:, 0] - fixed_xy[0], arr[:, 1] - fixed_xy[1])


def weighted_fps(
    cand_arr: np.ndarray,
    cand_hard: np.ndarray,
    prior: np.ndarray,
    bounds: np.ndarray,
    beta: float,
    budget: int,
    periodic_mask: np.ndarray | None = None,
    axis_weights: np.ndarray | None = None,
) -> list[int]:
    """Greedy farthest-point selection of ``budget`` candidate indices.

    Objective per step: ``d_min * (1 + beta * hardness)`` where ``d_min`` is the
    current NN distance to prior + already-selected points. ``beta=0`` is pure
    coverage. Candidates with grid-snapped placement dims should be pre-snapped by
    the caller so FPS diversifies among valid placement points + the continuous pen.

    ``axis_weights`` (length K, ``None`` ⇒ all-ones) scales each
    dimension in the distance, so downweighting the placement (e.g. holder) dims
    makes FPS spread the pen pose instead of buying cheap "farness" by hopping
    holder cells while leaving the pen pose nearly fixed."""
    mask = _resolve_mask(periodic_mask, bounds.shape[0])
    cu = normalise(cand_arr, bounds, mask)
    selected: list[int] = []
    d_min = pairwise_min_distance(cand_arr, prior, bounds, mask, axis_weights)
    for _ in range(budget):
        obj = d_min * (1.0 + beta * cand_hard)
        obj[selected] = -np.inf
        pick = int(np.argmax(obj))
        selected.append(pick)
        d_min = np.minimum(d_min, unit_dist_to_point(cu, cu[pick], mask, axis_weights))
    return selected


def states_from_manifest_rows(
    rows: list[dict], keys: tuple[str, ...], periodic_mask: np.ndarray | None = None
) -> np.ndarray:
    """(n, K) state array from manifest/ledger dict rows over ``keys``; periodic
    columns wrapped. With a 3-tuple of pen keys (and no mask) the columns are
    (x, y, periodic yaw)."""
    k = len(keys)
    mask = _resolve_mask(periodic_mask, k)
    arr = np.asarray([[float(r[key]) for key in keys] for r in rows], dtype=np.float64)
    if arr.size:
        for d in np.nonzero(mask)[0]:
            arr[:, d] = wrap_angle(arr[:, d])
    return arr.reshape(-1, k)


class TaskGeometry:
    """Spec-bound convenience wrapper over the task's JOINT sampling space (pen +
    grid-snapped placements). For a task with no placements the joint space is the
    3-DOF pen, so every method reduces to the original behavior."""

    def __init__(self, spec: RealTaskSpec):
        self.spec = spec
        self.bounds = spec.sampling_bounds
        self.periodic_mask = spec.sampling_periodic_mask
        self.grid_dims = spec.sampling_grid_dims
        self.snap = spec.sampling_snap
        self.keys = spec.sampling_keys

    def normalise(self, arr: np.ndarray) -> np.ndarray:
        return normalise(arr, self.bounds, self.periodic_mask)

    def pairwise_min_distance(self, candidates: np.ndarray, support: np.ndarray) -> np.ndarray:
        return pairwise_min_distance(candidates, support, self.bounds, self.periodic_mask)

    def fill_quantile(self, test_cloud: np.ndarray, support: np.ndarray, q: float = 0.95) -> float:
        return fill_quantile(test_cloud, support, self.bounds, q, self.periodic_mask)

    def _draw_bounds(self) -> np.ndarray:
        """Sampling DRAW range. For grid-snapped placement dims, widen by half a grid cell
        (snap/2) on each side so a uniform draw + nearest-snap yields a UNIFORM distribution
        over the grid points: otherwise the two border points each capture only a half-cell
        and are sampled ~half as often as interior points. snap_to_grid still snaps + clips
        to the ORIGINAL grid bounds, so the snapped values stay on the same grid (the wider
        range just balances the per-point catchment widths). Pen dims (nan snap) are
        unchanged, so tasks with no placements draw exactly as before."""
        finite = np.isfinite(self.snap)
        if not finite.any():
            return self.bounds
        db = np.array(self.bounds, dtype=np.float64, copy=True)
        half = np.asarray(self.snap, dtype=np.float64)[finite] / 2.0
        db[finite, 0] -= half
        db[finite, 1] += half
        return db

    def sobol(self, seed: int, n: int, start: int = 0) -> np.ndarray:
        """Joint Sobol over pen + placements, with placement dims snapped to grid."""
        return snap_to_grid(
            sobol_points(self._draw_bounds(), seed, n, start), self.bounds, self.snap
        )

    def sobol_feasible(self, seed: int, n: int, start: int = 0) -> tuple[np.ndarray, list[int]]:
        """Joint Sobol draw of ``n`` snapped points whose sampled placements satisfy the task's
        ``placement_min_separation_m`` inter-placement constraint (e.g. routing's two clips must
        be >= 2 in apart, else their mounting taps fight for the same holes and the robot loader
        rejects the state). Returns ``(points (n, K), stream_indices)`` where ``stream_indices``
        are the ORIGINAL positions in the seed's Sobol stream (non-contiguous when rejection
        drops points), so the manifest keeps auditable Sobol provenance.

        When the task has no ``placement_min_separation_m`` constraint this is exactly
        :meth:`sobol` with contiguous indices ``start..start+n-1`` (the plain draw for pen /
        single-placement tasks). Prefix-stable: growing the draw only appends tail points, so the
        kept set is the deterministic first-``(start+n)``-feasible of
        the seed's stream, sliced to ``[start:start+n]``."""
        min_sep = self.spec.placement_min_separation_m
        if min_sep is None:
            return self.sobol(seed, n, start), list(range(start, start + n))
        need = start + n
        m = max(4 * need, 512)
        while True:
            pts = self.sobol(seed, m, 0)
            kept: list[np.ndarray] = []
            kept_idx: list[int] = []
            for i, point in enumerate(pts):
                row = {k: float(v) for k, v in zip(self.keys, point, strict=True)}
                if self.spec.min_placement_separation(row) >= min_sep - 1e-9:
                    kept.append(point)
                    kept_idx.append(i)
                    if len(kept) == need:
                        return np.asarray(kept[start:]), kept_idx[start:]
            if m > (1 << 20):
                raise RuntimeError(
                    f"rejection sampling could not find {need} feasible points for "
                    f"{self.spec.name!r} (min separation {min_sep} m) within {m} draws"
                )
            m *= 2

    def uniform(self, seed: int, n: int) -> np.ndarray:
        """Joint uniform over pen + placements, with placement dims snapped to grid."""
        return snap_to_grid(uniform_points(self._draw_bounds(), seed, n), self.bounds, self.snap)

    def cell_ids(self, arr: np.ndarray) -> np.ndarray:
        return cell_ids(arr, self.bounds, self.grid_dims, self.periodic_mask)

    def cell_entropy(self, arr: np.ndarray) -> float:
        return cell_entropy(arr, self.bounds, self.grid_dims, self.periodic_mask)

    def edge_norm(self, arr: np.ndarray) -> np.ndarray:
        return edge_norm(arr, self.bounds)

    def dist_to_fixed(self, arr: np.ndarray, obj: str) -> np.ndarray:
        return dist_to_fixed(arr, self.spec.fixed_objects[obj])

    def placement_axis_weights(self, weight: float) -> np.ndarray:
        """(K,) per-dim weight vector: 1.0 on the pen dims, ``weight`` on every
        grid-snapped placement dim (e.g. the holder). ``weight=1`` is the
        unweighted joint space; ``weight<1`` makes coverage spread the pen pose
        rather than buying farness by hopping placement cells. Tasks with no
        placement reduce to all-ones."""
        w = np.ones(len(self.keys), dtype=np.float64)
        w[len(self.spec.state_keys) :] = weight
        return w

    def weighted_fps(
        self,
        cand_arr: np.ndarray,
        cand_hard: np.ndarray,
        prior: np.ndarray,
        beta: float,
        budget: int,
        axis_weights: np.ndarray | None = None,
    ) -> list[int]:
        return weighted_fps(
            cand_arr,
            cand_hard,
            prior,
            self.bounds,
            beta,
            budget,
            self.periodic_mask,
            axis_weights,
        )

    def states_from_rows(self, rows: list[dict]) -> np.ndarray:
        return states_from_manifest_rows(rows, self.keys, self.periodic_mask)
