"""Well-formedness invariants for the real-task registry (RealTaskSpec).

Adding a real task is "one RealTaskSpec entry" (mulligan/real/lifecycle/tasks.py), but
that entry is the single source of truth for analysis-side geometry + the joint
sampling space, so a malformed spec would corrupt every downstream computation.
These tests (a) pin the invariants every registered spec must hold, (b) prove the
``_register`` guard rejects the malformations new tasks are likely to hit, and
(c) pin marker_d2's grid-snapped holder placement + its joint sampling space.

The rejection probes rely on ``_register`` validating BEFORE inserting into the
global registry, so a rejected probe never pollutes ``_TASK_SPECS``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mulligan.real.lifecycle.tasks import (
    INCH_TO_M,
    DPTrainingRecipe,
    GridSampledPlacement,
    RealTaskSpec,
    _register,
    get_task_spec,
    registered_task_specs,
)


def test_exactly_the_paper_tasks_are_registered():
    names = [spec.name for spec in registered_task_specs()]
    assert names == ["marker_d2", "square_d2", "routing_d2"]


def test_retired_tasks_are_not_registered():
    for name in ("insert_marker_d1", "Square_D1"):
        with pytest.raises(KeyError, match="unknown real task"):
            get_task_spec(name)


@pytest.mark.parametrize("spec", registered_task_specs(), ids=lambda s: s.name)
def test_registered_spec_is_well_formed(spec):
    # Free dims are variable-length (>= 1). The 3-DOF pen (x, y, periodic yaw) is
    # assumed ONLY when no explicit state_periodic_mask is given; a non-pen task (routing)
    # supplies its own mask.
    n_free = len(spec.state_keys)
    assert n_free >= 1
    assert len(spec.bounds) == n_free
    mask = spec.free_periodic_mask
    assert mask.shape == (n_free,)
    if spec.state_periodic_mask is None:
        assert n_free == 3 and spec.state_keys[2].endswith("_yaw")
    # fixed-object position dict and manifest-key dict must describe the same objects.
    assert set(spec.fixed_objects) == set(spec.fixed_object_keys)
    # All persisted manifest keys (free + fixed + placement) are unique.
    keys = spec.manifest_keys
    assert len(keys) == len(set(keys))
    # Bounds are non-degenerate intervals (one per free dim).
    arr = spec.bounds_arr
    assert arr.shape == (n_free, 2)
    assert np.all(arr[:, 0] < arr[:, 1])
    assert int(np.prod(spec.grid_dims)) > 0
    assert len(spec.grid_dims) == n_free
    # Each sampled placement has two distinct (x, y) keys (in manifest_keys) + a non-empty
    # grid; an optional discrete orientation contributes index + angle manifest columns.
    for placement in spec.sampled_placements.values():
        assert len(placement.keys) == 2 and placement.keys[0] != placement.keys[1]
        assert all(k in keys for k in placement.keys)
        assert placement.grid_dims[0] >= 2 and placement.grid_dims[1] >= 2
        assert len(placement.grid_points()) == placement.grid_dims[0] * placement.grid_dims[1]
        if placement.has_orient:
            assert placement.orient_key in keys and placement.orient_angle_key in keys
            assert placement.n_orient >= 2
    # The joint sampling space is consistent: keys/bounds/mask/snap/grid all length K.
    k = len(spec.sampling_keys)
    assert spec.sampling_bounds.shape == (k, 2)
    assert spec.sampling_periodic_mask.shape == (k,)
    assert spec.sampling_snap.shape == (k,)
    assert len(spec.sampling_grid_dims) == k
    # Placement dims (x, y, orient index) are never periodic, so the joint space has exactly
    # as many periodic dims as the free space.
    assert int(spec.sampling_periodic_mask.sum()) == int(mask.sum())


def _pen_only_spec(**changes) -> RealTaskSpec:
    """An unregistered 3-DOF pen task with no placements and the default (4, 5, 4) grid."""
    kwargs = dict(
        name="registry_probe",
        task_name="registry_probe",
        state_keys=("pen_x", "pen_y", "pen_yaw"),
        bounds=get_task_spec("marker_d2").bounds,
    )
    kwargs.update(changes)
    return RealTaskSpec(**kwargs)


def test_no_placement_task_sampling_space_is_the_pen():
    # A task without sampled placements must have a sampling space identical to its
    # 3-DOF pen, so the shared geometry reduces to the pen geometry.
    spec = _pen_only_spec()
    assert spec.sampled_placements == {}
    assert spec.sampling_keys == spec.state_keys
    assert np.array_equal(spec.sampling_bounds, spec.bounds_arr)
    assert tuple(spec.sampling_grid_dims) == tuple(spec.grid_dims)
    assert list(spec.sampling_periodic_mask) == [False, False, True]
    assert np.all(np.isnan(spec.sampling_snap))


def _placement(
    keys: tuple[str, str] = ("holder_x", "holder_y"),
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (6.0 * INCH_TO_M, 8.0 * INCH_TO_M),
        (-4.0 * INCH_TO_M, 0.0),
    ),
    snap_m: float = INCH_TO_M,
    orient_key: str | None = None,
    orient_angle_key: str | None = None,
    orient_angles: tuple[float, ...] = (),
) -> GridSampledPlacement:
    return GridSampledPlacement(
        keys=keys,
        bounds=bounds,
        snap_m=snap_m,
        orient_key=orient_key,
        orient_angle_key=orient_angle_key,
        orient_angles=orient_angles,
    )


def _probe(**changes):
    """A placement-free pen spec with a fresh probe name and the given changes."""
    return _pen_only_spec(**changes)


def test_register_rejects_duplicate_name():
    dup = dataclasses.replace(get_task_spec("square_d2"))
    with pytest.raises(ValueError, match="duplicate RealTaskSpec name"):
        _register(dup)


def test_register_rejects_state_keys_bounds_length_mismatch():
    # state_keys and bounds must have matching length (here 4 keys vs the inherited 3 bounds).
    with pytest.raises(ValueError, match="matching length"):
        _register(_probe(state_keys=("pen_x", "pen_y", "pen_z", "pen_yaw")))


def test_register_rejects_non_pen_free_dims_without_mask():
    # Without an explicit state_periodic_mask the free dims must be the 3-DOF pen
    # (x, y, *_yaw); a 4-DOF free space (with matching bounds) and no mask is rejected.
    extra = (-0.1, 0.1)
    with pytest.raises(ValueError, match="must be the 3-DOF pen"):
        _register(
            _probe(
                state_keys=("pen_x", "pen_y", "pen_z", "pen_yaw"),
                bounds=(*get_task_spec("marker_d2").bounds, extra),
                grid_dims=(4, 5, 4, 2),
            )
        )


def test_register_rejects_non_yaw_third_key():
    with pytest.raises(ValueError, match="must be the 3-DOF pen"):
        _register(_probe(state_keys=("pen_x", "pen_y", "pen_theta")))


def test_register_rejects_state_periodic_mask_length_mismatch():
    # An explicit mask must match the number of free dims.
    with pytest.raises(ValueError, match="state_periodic_mask length"):
        _register(_probe(state_periodic_mask=(False, False)))


def test_register_rejects_mismatched_fixed_objects():
    with pytest.raises(ValueError, match="fixed_objects"):
        _register(_probe(fixed_objects={"peg": (0.1, 0.2)}))


def test_register_rejects_non_finite_fixed_object():
    with pytest.raises(ValueError, match="finite"):
        _register(
            _probe(
                fixed_objects={"peg": (float("nan"), 0.2)}, fixed_object_keys={"peg": ("px", "py")}
            )
        )


def test_register_rejects_placement_with_duplicate_keys():
    with pytest.raises(ValueError, match="two distinct manifest keys"):
        _register(_probe(sampled_placements={"holder": _placement(keys=("holder_x", "holder_x"))}))


def test_register_rejects_non_positive_snap():
    with pytest.raises(ValueError, match="snap_m must be"):
        _register(_probe(sampled_placements={"holder": _placement(snap_m=0.0)}))


def test_register_rejects_bounds_not_multiple_of_snap():
    # x range 0.5 in with a 1-in snap -> not an integer number of grid steps.
    with pytest.raises(ValueError, match="integer multiple"):
        _register(
            _probe(
                sampled_placements={
                    "holder": _placement(
                        bounds=((6.0 * INCH_TO_M, 6.5 * INCH_TO_M), (-4.0 * INCH_TO_M, 0.0))
                    )
                }
            )
        )


def test_register_rejects_placement_key_colliding_with_pen():
    with pytest.raises(ValueError, match="duplicate manifest keys"):
        _register(_probe(sampled_placements={"holder": _placement(keys=("pen_x", "holder_y"))}))


def test_register_rejects_grid_dims_length_mismatch():
    # grid_dims must have one cell count per free dim (here 2 cells for the 3-DOF pen).
    with pytest.raises(ValueError, match="one cell count per free dim"):
        _register(_probe(grid_dims=(4, 5)))


def test_register_rejects_zero_grid_dim():
    # A zero cell count would make np.prod(grid_dims) == 0 (no cells at all).
    with pytest.raises(ValueError, match="grid_dims cells must be >= 1"):
        _register(_probe(grid_dims=(0, 5, 4)))


def test_register_rejects_single_cell_grid_dims():
    # (1, 1, 1) -> n_cells == 1 -> cell_entropy would divide by math.log(1) == 0 (nan).
    with pytest.raises(ValueError, match="grid_dims must have >= 2 total cells"):
        _register(_probe(grid_dims=(1, 1, 1)))


def test_marker_d2_holder_is_a_grid_snapped_placement():
    spec = get_task_spec("marker_d2")
    assert spec.task_name == "marker_d2"
    assert spec.state_keys == ("pen_x", "pen_y", "pen_yaw")
    inch = INCH_TO_M
    arr = spec.bounds_arr
    assert np.allclose(arr[0], [-3 * inch, 3 * inch])
    assert np.allclose(arr[1], [-6 * inch, 6 * inch])

    holder = spec.sampled_placements["holder"]
    assert holder.keys == ("holder_x", "holder_y")
    assert np.allclose(holder.bounds, [[6 * inch, 8 * inch], [-4 * inch, 0.0]])
    assert np.isclose(holder.snap_m, inch)
    # x in {6,7,8} in, y in {-4,-3,-2,-1,0} in -> 3 x 5 = 15 grid points.
    assert holder.grid_dims == (3, 5)
    assert len(holder.grid_points()) == 15
    assert np.allclose(holder.axis_points(0), [6 * inch, 7 * inch, 8 * inch])

    # snap + on-grid behavior.
    assert np.allclose(holder.snap(6.4 * inch, -3.6 * inch), (6 * inch, -4 * inch))
    assert holder.is_on_grid(7 * inch, -2 * inch)
    assert not holder.is_on_grid(7.5 * inch, -2 * inch)  # off the 1-inch grid
    assert not holder.is_on_grid(9 * inch, -2 * inch)  # out of bounds

    # Manifest + joint sampling space carry the holder (x, y), NOT a categorical index.
    assert spec.manifest_keys == ("pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y")
    assert spec.sampling_keys == ("pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y")
    assert list(spec.sampling_periodic_mask) == [False, False, True, False, False]
    # Finer pen grid (6,12,8) JOINED with the holder's (3,5) grid -> joint coverage grid.
    assert spec.grid_dims == (6, 12, 8)
    assert spec.sampling_grid_dims == (6, 12, 8, 3, 5)
    assert np.isnan(spec.sampling_snap[:3]).all()
    assert np.allclose(spec.sampling_snap[3:], inch)


def test_square_d2_peg_is_a_grid_snapped_placement():
    # square_d2's peg is a placement sampled JOINTLY with the nut, snapped to a 1-inch grid.
    spec = get_task_spec("square_d2")
    assert spec.task_name == "square_d2"
    assert spec.state_keys == ("nut_x", "nut_y", "nut_yaw")
    inch = INCH_TO_M
    arr = spec.bounds_arr
    assert np.allclose(arr[0], [-5 * inch, 5 * inch])
    assert np.allclose(arr[1], [-9 * inch, 9 * inch])

    assert spec.fixed_objects == {} and spec.fixed_object_keys == {}
    peg = spec.sampled_placements["peg"]
    assert peg.keys == ("peg_x", "peg_y")
    assert np.allclose(peg.bounds, [[9.5 * inch, 11.5 * inch], [-4.5 * inch, -0.5 * inch]])
    assert np.isclose(peg.snap_m, inch)
    # 1-inch grid OFFSET BY 0.5 in so each peg center lands ON a physical marker dot (the
    # dots sit at half-integer inches): x in {9.5,10.5,11.5} in, y in {-4.5,-3.5,-2.5,-1.5,
    # -0.5} in -> 3 x 5 = 15 grid points. The forward values start at 9.5 in for physical
    # feasibility: a nut at nut_x_max=+5 in with its handle aimed forward reaches ~+8.2 in,
    # so a peg nearer than ~9 in could overlap the nut; peg_x >= 9.5 in keeps clearance > 0.
    assert peg.grid_dims == (3, 5)
    assert len(peg.grid_points()) == 15
    assert np.allclose(peg.axis_points(0), [9.5 * inch, 10.5 * inch, 11.5 * inch])
    assert np.allclose(
        peg.axis_points(1), [-4.5 * inch, -3.5 * inch, -2.5 * inch, -1.5 * inch, -0.5 * inch]
    )
    assert peg.is_on_grid(9.5 * inch, -1.5 * inch)  # a half-integer dot is on the grid
    assert not peg.is_on_grid(
        9 * inch, -2 * inch
    )  # integer inches are off the dot grid (offset 0.5)
    assert not peg.is_on_grid(8.5 * inch, -2.5 * inch)  # x < 9.5 in (below the feasibility floor)
    assert not peg.is_on_grid(12 * inch, -2.5 * inch)  # x > 11.5 in (out of bounds)

    # Manifest + joint sampling space carry the peg (x, y).
    assert spec.manifest_keys == ("nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y")
    assert spec.sampling_keys == ("nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y")
    assert list(spec.sampling_periodic_mask) == [False, False, True, False, False]
    # Finer nut grid (10,18,8) JOINED with the peg's (3,5) grid -> joint coverage grid.
    assert spec.grid_dims == (10, 18, 8)
    assert spec.sampling_grid_dims == (10, 18, 8, 3, 5)


def test_register_rejects_orient_key_without_angle_key():
    with pytest.raises(ValueError, match="both orient_key and orient_angle_key"):
        _register(
            _probe(
                sampled_placements={
                    "holder": _placement(orient_key="holder_oidx", orient_angles=(0.0, 1.0))
                }
            )
        )


def test_register_rejects_orient_with_single_angle():
    with pytest.raises(ValueError, match="needs >= 2 angles"):
        _register(
            _probe(
                sampled_placements={
                    "holder": _placement(
                        orient_key="holder_oidx",
                        orient_angle_key="holder_yaw",
                        orient_angles=(0.5,),
                    )
                }
            )
        )


def test_register_rejects_orient_with_duplicate_angles():
    with pytest.raises(ValueError, match="must be distinct"):
        _register(
            _probe(
                sampled_placements={
                    "holder": _placement(
                        orient_key="holder_oidx",
                        orient_angle_key="holder_yaw",
                        orient_angles=(0.5, 0.5, 1.0),
                    )
                }
            )
        )


# --- routing_d2: the first non-pen line (1-DOF free rope_x + 2 oriented clips) -------------

_ROUTING_ANGLES = (-np.pi / 4.0, -np.pi / 2.0, -3.0 * np.pi / 4.0)


def test_routing_d2_is_a_non_pen_task():
    spec = get_task_spec("routing_d2")
    inch = INCH_TO_M
    # The only free continuous DOF is the rope's x; it is non-periodic (explicit mask).
    assert spec.state_keys == ("rope_x",)
    assert spec.state_periodic_mask == (False,)
    assert list(spec.free_periodic_mask) == [False]
    assert spec.bounds_arr.shape == (1, 2)
    assert np.allclose(spec.bounds_arr[0], [-6.5 * inch, 9.5 * inch])
    assert spec.grid_dims == (16,)
    # 9 persisted manifest columns; 7 joint-sampling dims (orientation = index, not angle).
    assert spec.manifest_keys == (
        "rope_x",
        "clip_left_x",
        "clip_left_y",
        "clip_left_yaw",
        "clip_left_oidx",
        "clip_right_x",
        "clip_right_y",
        "clip_right_yaw",
        "clip_right_oidx",
    )
    assert spec.sampling_keys == (
        "rope_x",
        "clip_left_x",
        "clip_left_y",
        "clip_left_oidx",
        "clip_right_x",
        "clip_right_y",
        "clip_right_oidx",
    )
    # Nothing is periodic (no pen yaw); rope_x is the only continuous (nan-snap) dim.
    assert not spec.sampling_periodic_mask.any()
    assert np.isnan(spec.sampling_snap[0])
    assert np.allclose(spec.sampling_snap[1:], [inch, inch, 1.0, inch, inch, 1.0])
    assert spec.sampling_grid_dims == (16, 17, 10, 3, 17, 10, 3)


def test_routing_d2_clips_on_offset_grid_with_orientation():
    spec = get_task_spec("routing_d2")
    inch = INCH_TO_M
    left = spec.sampled_placements["clip_left"]
    right = spec.sampled_placements["clip_right"]
    # Both clips use the SAME 0.5-offset 1-inch dot grid as square_d2's peg, full x, split y.
    assert left.keys == ("clip_left_x", "clip_left_y")
    assert np.allclose(left.bounds, [[-6.5 * inch, 9.5 * inch], [0.5 * inch, 9.5 * inch]])
    assert np.allclose(right.bounds, [[-6.5 * inch, 9.5 * inch], [-9.5 * inch, -0.5 * inch]])
    assert left.grid_dims == (17, 10) and right.grid_dims == (17, 10)
    # half-integer dots are on-grid; integer inches are OFF (0.5 offset).
    assert left.is_on_grid(2.5 * inch, 3.5 * inch)
    assert not left.is_on_grid(2.0 * inch, 3.0 * inch)
    assert not left.is_on_grid(2.5 * inch, -1.5 * inch)  # left clip cannot be in the y<0 half
    # 3-way discrete orientation {-45, +45, +90} deg, shared by both clips.
    assert left.n_orient == 3 and right.n_orient == 3
    assert np.allclose(left.orient_angles, _ROUTING_ANGLES)
    assert np.allclose(right.orient_angles, _ROUTING_ANGLES)
    assert left.orient_key == "clip_left_oidx" and left.orient_angle_key == "clip_left_yaw"
    # index <-> angle round-trips; an off-set angle fails loud (no silent nearest-class snap).
    for i, a in enumerate(_ROUTING_ANGLES):
        assert left.angle_for_index(i) == pytest.approx(a)
        assert left.index_for_angle(a) == i
    with pytest.raises(ValueError, match="not within"):
        left.index_for_angle(0.0)  # 0 rad is not one of {-45, +45, +90} deg


def test_routing_d2_sampler_respects_grid_orientation_and_rope_bounds():
    from mulligan.real.lifecycle.geometry import TaskGeometry

    spec = get_task_spec("routing_d2")
    tg = TaskGeometry(spec)
    pts = tg.sobol(2026062901, 256)
    cols = {k: pts[:, i] for i, k in enumerate(spec.sampling_keys)}
    inch = INCH_TO_M
    # rope_x stays a continuous in-bounds value (not snapped to the dot grid).
    assert cols["rope_x"].min() >= -6.5 * inch - 1e-9
    assert cols["rope_x"].max() <= 9.5 * inch + 1e-9
    assert len({round(v, 6) for v in cols["rope_x"]}) > 50  # genuinely continuous
    for clip, ylo, yhi in (("clip_left", 0.5, 9.5), ("clip_right", -9.5, -0.5)):
        placement = spec.sampled_placements[clip]
        for x, y in zip(cols[f"{clip}_x"], cols[f"{clip}_y"]):
            assert placement.is_on_grid(x, y)
        oidx = cols[f"{clip}_oidx"].round().astype(int)
        assert set(oidx.tolist()) == {0, 1, 2}  # all 3 orientation classes covered
        assert np.allclose(cols[f"{clip}_oidx"], oidx)  # snapped to exact integers


def test_routing_d2_clip_min_separation_is_two_inches():
    spec = get_task_spec("routing_d2")
    assert spec.placement_min_separation_m == pytest.approx(2.0 * INCH_TO_M)
    # min_placement_separation reads each clip's (x, y) from a manifest-like row.
    close = {
        "clip_left_x": 0.0,
        "clip_left_y": 0.5 * INCH_TO_M,
        "clip_right_x": 0.0,
        "clip_right_y": -0.5 * INCH_TO_M,
    }
    far = {
        "clip_left_x": 0.0,
        "clip_left_y": 5.5 * INCH_TO_M,
        "clip_right_x": 0.0,
        "clip_right_y": -5.5 * INCH_TO_M,
    }
    assert spec.min_placement_separation(close) == pytest.approx(1.0 * INCH_TO_M)  # < 2 in
    assert spec.min_placement_separation(far) == pytest.approx(11.0 * INCH_TO_M)  # >= 2 in


def _two_placement_probe(**changes):
    placements = {
        "a": _placement(keys=("ax", "ay")),
        "b": _placement(keys=("bx", "by")),  # same (valid) default bounds, distinct keys
    }
    return _pen_only_spec(name="sep_probe", sampled_placements=placements, **changes)


def test_register_rejects_nonpositive_min_separation():
    with pytest.raises(ValueError, match="placement_min_separation_m must be"):
        _register(_two_placement_probe(placement_min_separation_m=0.0))


def test_register_rejects_min_separation_with_single_placement():
    # The probe has no placements; a separation constraint needs >= 2 to be meaningful.
    with pytest.raises(ValueError, match="needs >= 2 sampled placements"):
        _register(_probe(placement_min_separation_m=0.05))


def test_marker_d2_consumed_camera_roles_are_side1_wristleft():
    # marker_d2 stores 4 role-named cameras; the DP consumes side_1 + wrist_left, the train
    # default --camera-keys when none is passed explicitly.
    spec = get_task_spec("marker_d2")
    assert spec.consumed_camera_roles == ("side_1", "wrist_left")


def test_square_d2_consumed_camera_roles_are_side1_wristleft():
    # square_d2 collects in the same new room as marker_d2 (role-named cameras); the DP
    # consumes side_1 + wrist_left.
    spec = get_task_spec("square_d2")
    assert spec.consumed_camera_roles == ("side_1", "wrist_left")


def test_routing_d2_consumed_camera_roles_are_side1_side2_wristleft():
    # routing_d2 consumes BOTH side views PLUS the wrist: the rope spans the full y-axis, so the
    # opposing side cameras cover it
    # end-to-end, and the wrist view resolves the fine clip-seat manipulation.
    spec = get_task_spec("routing_d2")
    assert spec.consumed_camera_roles == ("side_1", "side_2", "wrist_left")


def test_consumed_camera_roles_default_is_empty():
    # The field is opt-in: every registered line sets it explicitly; the default is empty.
    assert _pen_only_spec().consumed_camera_roles == ()
    for spec in registered_task_specs():
        assert spec.consumed_camera_roles, spec.name


# --- DP training recipe (per-task defaults resolved by the DP trainer) -----------------


def test_dp_training_recipe_marker_d2_era_defaults():
    # The shared default recipe = the marker_d2-era real-DP recipe. Pin the load-bearing
    # numbers (a silent drift here would change every --task launch's hypers).
    r = DPTrainingRecipe()
    assert r.chunk_size == 12  # prediction horizon
    assert r.n_action_steps == 6  # execution horizon (predict 12 / execute 6)
    assert r.down_dims == (512, 1024)
    assert r.batch_size == 64
    assert r.training_steps == 50_000
    assert r.save_freq == 25_000
    assert r.eval_freq == 10_000
    assert r.drop_n_last_frames == 2  # fixed 2 (not the generic config default 7)


@pytest.mark.parametrize("name", ["marker_d2", "square_d2", "routing_d2"])
def test_every_spec_carries_expected_recipe(name):
    # The paper lines train longer and evaluate more densely than the shared default.
    expected = DPTrainingRecipe(training_steps=100_000, save_freq=25_000, eval_freq=5_000)
    assert get_task_spec(name).training == expected


def test_apply_real_training_recipe_resolution_precedence():
    # The DP trainer resolves: explicit CLI > --task recipe (+ station) > fallback.
    # Imported lazily so the rest of this (tasks-only) file stays light.
    import argparse

    from mulligan.real.train.policy import _apply_real_training_recipe

    recipe_args = dict(
        image_height=None,
        image_width=None,
        video_backend=None,
        chunk_size=None,
        n_action_steps=None,
        down_dims=None,
        batch_size=None,
        eval_batch_size=None,
        training_steps=None,
        save_freq=None,
        eval_freq=None,
        drop_n_last_frames=None,
    )

    def ns(**over):
        d = dict(task=None)
        d.update(recipe_args)
        d.update(over)
        return argparse.Namespace(**d)

    # --task marker_d2 fills the whole recipe + station capture defaults.
    a = ns(task="marker_d2")
    _apply_real_training_recipe(a)
    assert (a.chunk_size, a.n_action_steps, a.down_dims, a.batch_size) == (12, 6, "512,1024", 64)
    assert a.eval_batch_size == 64
    assert (a.training_steps, a.save_freq, a.eval_freq, a.drop_n_last_frames) == (
        100000,
        25000,
        5000,
        "2",
    )
    assert (a.image_height, a.image_width, a.video_backend) == (224, 224, "torchcodec")

    # No --task: generic fallbacks; the downstream-resolved args stay None.
    b = ns(task=None)
    _apply_real_training_recipe(b)
    assert (b.image_height, b.image_width, b.batch_size) == (240, 320, 8)
    assert b.eval_batch_size == 8
    assert (b.training_steps, b.save_freq, b.eval_freq) == (100000, 25000, 2500)
    assert (b.chunk_size, b.n_action_steps, b.down_dims, b.drop_n_last_frames, b.video_backend) == (
        None,
        None,
        None,
        None,
        None,
    )

    # Explicit CLI flags win over the recipe.
    c = ns(task="marker_d2", chunk_size=8, batch_size=128, eval_batch_size=256, image_height=256)
    _apply_real_training_recipe(c)
    assert (c.chunk_size, c.batch_size, c.eval_batch_size, c.image_height) == (8, 128, 256, 256)
    assert c.training_steps == 100000  # unspecified -> still from the recipe
    assert c.eval_freq == 5000


def test_lifecycle_package_reexports_registry_api():
    """The lifecycle package surface must mirror the public registry API in tasks.py.

    Parity with ``mulligan.real.stage_specs``, whose package ``__init__`` re-exports its
    full registry API. Each re-exported name must be the SAME object as the
    tasks-submodule attribute (a true re-export, not a shadowing redefinition).
    """
    import mulligan.real.lifecycle as lifecycle_pkg
    from mulligan.real.lifecycle import tasks

    for name in (
        "RealTaskSpec",
        "get_task_spec",
        "registered_task_specs",
        "find_task_spec_by_task_name",
    ):
        assert hasattr(lifecycle_pkg, name), f"lifecycle package does not re-export {name!r}"
        assert getattr(lifecycle_pkg, name) is getattr(tasks, name), (
            f"lifecycle.{name} is not the same object as tasks.{name}"
        )
        assert name in lifecycle_pkg.__all__, f"{name!r} missing from lifecycle.__all__"


def test_camera_crop_overrides_are_valid_station_role_boxes():
    """Every spec's camera_crop_overrides must be role-keyed, stored-frame (640x480)
    valid boxes strictly inside the frame — same contract as the station defaults."""
    from mulligan.real.robot.cameras import STATION_CAMERA_KEYS_BY_ROLE

    for spec in registered_task_specs():
        for role, box in spec.camera_crop_overrides.items():
            assert role in STATION_CAMERA_KEYS_BY_ROLE, (spec.name, role)
            x0, y0, x1, y1 = box
            assert 0 <= x0 < x1 <= 640, (spec.name, role, box)
            assert 0 <= y0 < y1 <= 480, (spec.name, role, box)


def test_routing_d2_side_crop_overrides_locked():
    """routing_d2's task-fit side ROIs (fit from the R0 reset-frame
    homography content envelopes). The station defaults are intentionally NOT these
    boxes; the overrides must ride on the spec so the other lines sharing side_1
    (marker_d2/square_d2) and any other side_2 consumer keep the station defaults."""
    from mulligan.real.robot.cameras import STATION_CAMERA_DEFAULT_CROPS

    spec = get_task_spec("routing_d2")
    assert spec.camera_crop_overrides == {
        "side_1": (140, 120, 560, 470),
        "side_2": (130, 120, 440, 445),
    }
    for role, box in spec.camera_crop_overrides.items():
        assert box != STATION_CAMERA_DEFAULT_CROPS[role], role
