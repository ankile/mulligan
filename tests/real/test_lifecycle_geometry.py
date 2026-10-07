"""Unit tests for the N-D joint-sampling geometry in mulligan.real.lifecycle.geometry.

Pins (a) byte-identity of the 3-DOF path (default periodic_mask), (b) the
grid-snap quantization for sampled placements, and (c) marker_d2's joint pen+holder
Sobol/uniform sampling (holder lands on the 1-inch grid, pen stays continuous)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mulligan.real.lifecycle.geometry import (
    TaskGeometry,
    cell_entropy,
    normalise,
    snap_to_grid,
    sobol_points,
    uniform_points,
)
from mulligan.real.lifecycle.tasks import get_task_spec


def test_default_mask_matches_explicit_3dof():
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0], [-np.pi, np.pi]])
    arr = np.random.default_rng(0).uniform(-1.5, 1.5, (32, 3))
    assert np.array_equal(
        normalise(arr, bounds), normalise(arr, bounds, np.array([False, False, True]))
    )


def test_normalise_requires_mask_for_non_3dof():
    bounds = np.tile([0.0, 1.0], (5, 1))
    with pytest.raises(ValueError, match="periodic_mask is required"):
        normalise(np.zeros((2, 5)), bounds)


def test_snap_to_grid_snaps_finite_columns_passes_nan():
    bounds = np.array([[6.0, 8.0], [-4.0, 0.0]])
    snap = np.array([np.nan, 1.0])  # col 0 continuous, col 1 on a 1-unit grid
    out = snap_to_grid(np.array([[6.4, -3.6], [7.9, -0.2]]), bounds, snap)
    assert np.allclose(out[:, 0], [6.4, 7.9])  # nan-snap column unchanged
    assert np.allclose(out[:, 1], [-4.0, 0.0])  # snapped to nearest integer


def test_snap_to_grid_clips_to_bounds():
    bounds = np.array([[6.0, 8.0]])
    out = snap_to_grid(np.array([[9.0], [5.0]]), bounds, np.array([1.0]))
    assert np.allclose(out[:, 0], [8.0, 6.0])


def test_no_placement_taskgeometry_sobol_is_pen_passthrough():
    # A task without placements: TaskGeometry.sobol must equal the bare
    # 3-DOF sobol_points (all-nan snap -> pass-through).
    spec = dataclasses.replace(get_task_spec("marker_d2"), sampled_placements={})
    tg = TaskGeometry(spec)
    assert np.array_equal(tg.sobol(5, 16), sobol_points(spec.bounds_arr, 5, 16))
    assert np.array_equal(tg.uniform(9, 16), uniform_points(spec.bounds_arr, 9, 16))


def test_marker_d2_sobol_snaps_holder_keeps_pen_continuous():
    spec = get_task_spec("marker_d2")
    tg = TaskGeometry(spec)
    pts = tg.sobol(123, 64)
    assert pts.shape == (64, 5)
    holder = spec.sampled_placements["holder"]
    for x, y in pts[:, 3:5]:
        assert holder.is_on_grid(float(x), float(y))  # holder snapped to the grid
    pen_bounds = spec.bounds_arr
    assert (pts[:, 0] >= pen_bounds[0, 0] - 1e-9).all() and (
        pts[:, 0] <= pen_bounds[0, 1] + 1e-9
    ).all()
    # Pen x is continuous (not collapsed onto a coarse grid): many distinct values.
    assert len(np.unique(np.round(pts[:, 0], 6))) > 32


def test_marker_d2_sobol_covers_all_holder_grid_points():
    spec = get_task_spec("marker_d2")
    tg = TaskGeometry(spec)
    pts = tg.sobol(7, 200)
    hits = {(round(float(x), 6), round(float(y), 6)) for x, y in pts[:, 3:5]}
    assert len(hits) == 15  # joint Sobol over 200 draws covers the full 3x5 holder grid


def test_marker_d2_holder_snapping_is_uniform_across_grid():
    # Edge-effect fix: the sampling DRAW is widened by half a grid cell (snap/2) per side so
    # every holder grid point — including the border points — gets a full-cell snap
    # catchment. A large UNIFORM draw must therefore hit all 15 points with ~equal frequency.
    # WITHOUT the widening the 4 corner points catch only a quarter-cell (half in x AND y),
    # giving a ~4x max/min imbalance; the fix brings it near 1.0, so 1.35 cleanly pins it.
    from collections import Counter

    spec = get_task_spec("marker_d2")
    tg = TaskGeometry(spec)
    pts = tg.uniform(2024, 30000)
    counts = Counter((round(float(x), 6), round(float(y), 6)) for x, y in pts[:, 3:5])
    assert len(counts) == 15
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo < 1.35, (
        f"holder snapping not ~uniform: per-point counts {sorted(counts.values())}"
    )


def test_cell_entropy_single_cell_grid_is_zero_not_nan():
    # A 1-cell grid carries no occupancy information; normalization would divide by
    # math.log(1) == 0 (-> nan). cell_entropy must special-case it to a clean 0.0.
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0], [-np.pi, np.pi]])
    arr = np.random.default_rng(0).uniform(-1.0, 1.0, (16, 3))
    ent = cell_entropy(arr, bounds, (1, 1, 1))
    assert ent == 0.0
    assert not np.isnan(ent)
