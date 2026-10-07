"""Collection-path support for marker_d2 (the 4-key manifest + grid-snapped RED holder).

marker_d2 manifests carry per-row continuous (snapped) ``holder_x`` / ``holder_y`` in
the same operator frame as the pen. These tests pin that the collection-path loader/
renderer in the blind collector accepts the manifest, validates the holder lands on the
registry grid (fails loud otherwise), and that the operator card draws the chosen point
from the SAME source as the text readout (the target row) + the allowed grid from the
registry -- so they can never disagree. The 3-key marker / 5-key square paths are
untouched.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import math

from mulligan.real.collect.initial_states import (
    ArmSpec,
    MARKER_D2_INITIAL_STATE_KEYS,
    NUT_PEG_INITIAL_STATE_KEYS,
    _format_initial_state_target,
    _load_initial_state_manifest,
    _load_manifest_payload,
    _manifest_keys,
)
from mulligan.real.operator_ui.cards import (
    CardStyle,
    operator_card_pen_bounds_in,
    placement_card_points,
    write_initial_state_card,
)
from mulligan.real.lifecycle.tasks import INCH_TO_M, get_task_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT / "data/real/manifests/marker_d2/r00/"
    "marker_d2_r0_three_arm_uniform100_sobol100_evalsobol50_scrambled.json"
)
ARMS = [
    ArmSpec("baseline_uniform", "model-a"),
    ArmSpec("mulligan_sobol", "model-b"),
    ArmSpec("eval_heldout", "model-c"),
]


def test_manifest_keys_accepts_marker_d2():
    assert _manifest_keys({"keys": MARKER_D2_INITIAL_STATE_KEYS}) == MARKER_D2_INITIAL_STATE_KEYS
    with pytest.raises(ValueError, match="unsupported initial-state keys"):
        _manifest_keys({"keys": ["pen_x", "pen_y", "pen_yaw"]})
    assert _manifest_keys({"keys": NUT_PEG_INITIAL_STATE_KEYS}) == NUT_PEG_INITIAL_STATE_KEYS
    with pytest.raises(ValueError, match="unsupported initial-state keys"):
        _manifest_keys({"keys": ["pen_x", "pen_y"]})


def test_load_marker_d2_manifest_holder_on_grid():
    targets, meta = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="marker_d2")
    assert len(targets) == 250
    assert _manifest_keys(meta) == MARKER_D2_INITIAL_STATE_KEYS
    holder = get_task_spec("marker_d2").sampled_placements["holder"]
    for t in targets:
        assert t.holder_x is not None and t.holder_y is not None
        assert holder.is_on_grid(t.holder_x, t.holder_y)


def test_manifest_arms_match_sources_and_load_with_derived_arms():
    # The teleop collector DERIVES its arm allowlist from the manifest (a fixed allowlist
    # would crash the load once eval_heldout is present), so the manifest's declared `arms` MUST
    # equal its actual row sources,
    # and loading with arms derived from the manifest must accept all 250 rows (eval_heldout
    # must NOT be dropped).
    payload = _load_manifest_payload(MANIFEST)
    declared = list(payload["arms"])
    sources = sorted({row["source"] for row in payload["states"]})
    assert sorted(declared) == sources == ["baseline_uniform", "eval_heldout", "mulligan_sobol"]
    derived_arms = [ArmSpec(key, "teleop") for key in declared]
    targets, _ = _load_initial_state_manifest(MANIFEST, derived_arms, expected_task="marker_d2")
    assert len(targets) == 250


def test_format_includes_holder_coords():
    targets, _ = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="marker_d2")
    assert "HOLDER (" in _format_initial_state_target(targets[0])


def test_card_points_chosen_matches_row_and_is_a_grid_point():
    # The operator card's bold point must come from the SAME source as the text (the
    # row coords) and coincide with a registry grid point -> no text-vs-bold disagreement.
    targets, _ = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="marker_d2")
    target = targets[0]
    points = placement_card_points(target, "marker_d2")
    assert len(points) == 1
    chosen, grid = points[0]
    assert chosen is not None
    assert chosen == (target.holder_x, target.holder_y)
    assert len(grid) == 15
    assert min(abs(chosen[0] - gx) + abs(chosen[1] - gy) for gx, gy in grid) < 1e-9


def _bad_manifest(tmp_path: Path, mutate) -> Path:
    payload = json.loads(MANIFEST.read_text())
    row = copy.deepcopy(payload["states"][0])
    mutate(row)
    payload["states"] = [row]
    out = tmp_path / "bad_marker_d2_manifest.json"
    out.write_text(json.dumps(payload))
    return out


def test_off_grid_holder_fails_loud(tmp_path):
    def mutate(row):
        row["holder_x"] = float(row["holder_x"]) + 0.5 * 0.0254  # half-inch off the 1-inch grid

    with pytest.raises(ValueError, match="grid point"):
        _load_initial_state_manifest(
            _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="marker_d2"
        )


def test_missing_manifest_idx_fails_loud(tmp_path):
    # The required provenance key manifest_idx must NOT be fabricated from row order.
    def mutate(row):
        del row["manifest_idx"]

    with pytest.raises(ValueError, match="manifest_idx"):
        _load_initial_state_manifest(
            _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="marker_d2"
        )


def test_non_finite_holder_fails_loud(tmp_path):
    def mutate(row):
        row["holder_x"] = float("inf")

    with pytest.raises(ValueError, match="non-finite holder_x"):
        _load_initial_state_manifest(
            _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="marker_d2"
        )


def test_holder_keyed_manifest_unregistered_task_fails_loud(tmp_path):
    # A holder-keyed manifest whose task has NO registered spec (e.g. a future
    # marker_d2_r1) must NOT load the holder with zero on-grid validation. We point
    # expected_task at the unregistered name so the expected-task gate passes and the
    # spec-None path is what trips (the spec lookup is what has no grid to validate).
    payload = json.loads(MANIFEST.read_text())
    payload["task"] = "marker_d2_unregistered_xyz"
    payload["states"] = [copy.deepcopy(payload["states"][0])]
    out = tmp_path / "unregistered_marker_d2_manifest.json"
    out.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="holder keys"):
        _load_initial_state_manifest(out, list(ARMS), expected_task="marker_d2_unregistered_xyz")


def test_card_points_holder_target_unregistered_task_fails_loud():
    # Mirror of the parser fix on the operator-card path: a holder-carrying target whose
    # task has no registered spec must RAISE, not silently return [] (which would draw no
    # holder while the text readout still shows a HOLDER line).
    targets, _ = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="marker_d2")
    target = targets[0]
    assert target.holder_x is not None and target.holder_y is not None
    with pytest.raises(ValueError, match="holder coords"):
        placement_card_points(target, "marker_d2_unregistered_xyz")


def test_render_marker_d2_target_card(tmp_path):
    targets, meta = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="marker_d2")
    path = write_initial_state_card(
        targets[0],
        meta,
        output_dir=tmp_path,
        task_name="marker_d2",
        style=CardStyle(),
    )
    assert path.exists() and path.stat().st_size > 5_000


# --- the manifest loader validates the pen against the registry bounds ----------


def test_out_of_bounds_pen_x_fails_loud(tmp_path):
    # marker_d2 pen_x is bounded to +/-3 in (+/-0.0762 m). A stale/hand-edited manifest
    # with pen_x = 99 m must fail loud on load -- the holder grid check alone would NOT
    # catch an out-of-bounds pen (a 99 m pen_x must not be accepted).
    def mutate(row):
        row["pen_x"] = 99.0

    with pytest.raises(ValueError, match="pen_x.*out of bounds"):
        _load_initial_state_manifest(
            _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="marker_d2"
        )


def test_just_out_of_bounds_pen_y_fails_loud(tmp_path):
    # Just beyond +6 in (the +/-6 in pen_y bound) must also fail -- not only absurd values.
    def mutate(row):
        row["pen_y"] = 6.5 * INCH_TO_M

    with pytest.raises(ValueError, match="pen_y.*out of bounds"):
        _load_initial_state_manifest(
            _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="marker_d2"
        )


def test_in_bounds_pen_at_bound_loads(tmp_path):
    # Exactly on the +6 in pen_y bound (with a tiny tolerance) must load -- the check must
    # not reject the legitimate extremes the sampler can produce.
    def mutate(row):
        row["pen_y"] = 6.0 * INCH_TO_M

    targets, _ = _load_initial_state_manifest(
        _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="marker_d2"
    )
    assert len(targets) == 1
    assert abs(targets[0].pen_y - 6.0 * INCH_TO_M) < 1e-9


def test_out_of_range_pen_yaw_does_not_fail(tmp_path):
    # pen_yaw is periodic (rows skipped by the in-bounds check by design). A yaw outside
    # [-pi, pi] must NOT trip the loader -- only finiteness is required for yaw.
    def mutate(row):
        row["pen_yaw"] = 2.0 * math.pi + 0.3  # well outside [-pi, pi]

    targets, _ = _load_initial_state_manifest(
        _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="marker_d2"
    )
    assert len(targets) == 1
    assert abs(targets[0].pen_yaw - (2.0 * math.pi + 0.3)) < 1e-9


# --- the operator-card pen rectangle uses the marker_d2 +/-3/+/-6 in extent -----


def test_operator_card_pen_rect_uses_marker_d2_extent():
    # The drawn pen randomization rectangle is marker_d2's +/-3 in (x) / +/-6 in (y),
    # read from the registry.
    _, meta = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="marker_d2")
    x_min, x_max, y_min, y_max = operator_card_pen_bounds_in("marker_d2", meta)
    assert (x_min, x_max) == pytest.approx((-3.0, 3.0), abs=1e-6)
    assert (y_min, y_max) == pytest.approx((-6.0, 6.0), abs=1e-6)


def test_operator_card_pen_bounds_meta_registry_mismatch_fails_loud():
    # A manifest meta whose pen bounds disagree with the registry must fail loud rather
    # than silently drawing a rectangle that contradicts the validated manifest.
    _, meta = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="marker_d2")
    bad_meta = copy.deepcopy(meta)
    bad_meta["bounds"]["pen_x"] = [-0.05, 0.05]  # wrong vs registry +/-0.0762 m
    with pytest.raises(ValueError, match="disagree with the registry"):
        operator_card_pen_bounds_in("marker_d2", bad_meta)
