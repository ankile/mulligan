"""Collection-path support for square_d2 (the square 5-key manifest + grid-snapped peg).

square_d2 reuses the earlier fixed-peg Nut setup's exact 5 manifest keys
(``nut_x/nut_y/nut_yaw`` + ``peg_x/peg_y``), but the peg is no longer a FIXED object -- it is sampled JOINTLY with the nut
and snapped to a 1-inch registry grid. These tests pin that the shared collection-path
loader/validator/renderer in the blind collector:
  - validates the chosen peg lands on the registry grid (fails loud otherwise), via the
    same registry-driven helper marker_d2's holder uses;
  - leaves the fixed-peg path a no-op; and
  - draws the peg per-row plus the allowed-grid overlay on the operator card.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mulligan.real.collect.initial_states import (
    ArmSpec,
    NUT_PEG_INITIAL_STATE_KEYS,
    _format_initial_state_target,
    _initial_state_setup_subject,
    _load_initial_state_manifest,
    _manifest_keys,
    _validate_sampled_placements_against_registry,
)
from mulligan.real.operator_ui.cards import (
    CardStyle,
    placement_card_points,
    write_initial_state_card,
)
from mulligan.real.lifecycle.tasks import INCH_TO_M, get_task_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT / "data/real/manifests/square_d2/r00/"
    "square_d2_r0_three_arm_uniform100_sobol100_evalsobol50_scrambled.json"
)
ARMS = [
    ArmSpec("baseline_uniform", "model-a"),
    ArmSpec("mulligan_sobol", "model-b"),
    ArmSpec("eval_heldout", "model-c"),
]


# --- Registry-driven on-grid validation (no manifest needed) -------------------------------


def test_validate_placements_validates_square_d2_peg_on_grid():
    # square_d2's peg grid is offset by 0.5 in so the peg center lands on the physical
    # marker dots.
    row = {"peg_x": 9.5 * INCH_TO_M, "peg_y": -2.5 * INCH_TO_M}
    _validate_sampled_placements_against_registry(Path("mem"), row, 0, "square_d2")  # no raise


def test_validate_placements_rejects_off_grid_square_d2_peg():
    # An integer-inch peg position is off the physical-dot grid.
    row = {"peg_x": 9.0 * INCH_TO_M, "peg_y": -2.0 * INCH_TO_M}
    with pytest.raises(ValueError, match="grid point"):
        _validate_sampled_placements_against_registry(Path("mem"), row, 0, "square_d2")


def test_validate_placements_rejects_non_finite_square_d2_peg():
    row = {"peg_x": float("inf"), "peg_y": -2.5 * INCH_TO_M}
    with pytest.raises(ValueError, match="non-finite peg_x"):
        _validate_sampled_placements_against_registry(Path("mem"), row, 0, "square_d2")


def test_validate_placements_noop_for_fixed_peg_task():
    # A fixed-peg task has no sampled placements, so the helper must be a
    # no-op even for a peg value that is NOT on any grid -- the fixed-peg path is unchanged.
    row = {"peg_x": 9.5 * INCH_TO_M, "peg_y": -2.0 * INCH_TO_M}
    _validate_sampled_placements_against_registry(Path("mem"), row, 0, "fixed_peg_nut")  # no raise


def test_manifest_keys_accepts_square_d2_keys():
    # square_d2 reuses the fixed-peg setup's 5-key tuple, so the shared allowlist accepts it as-is.
    assert _manifest_keys({"keys": NUT_PEG_INITIAL_STATE_KEYS}) == NUT_PEG_INITIAL_STATE_KEYS


# --- Manifest-driven end-to-end (gated on the generated R0 manifest) -----------------------


def test_load_square_d2_manifest_peg_on_grid():
    targets, meta = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="square_d2")
    assert len(targets) == 250
    assert _manifest_keys(meta) == NUT_PEG_INITIAL_STATE_KEYS
    peg = get_task_spec("square_d2").sampled_placements["peg"]
    for t in targets:
        assert t.peg_x is not None and t.peg_y is not None
        assert peg.is_on_grid(t.peg_x, t.peg_y)


def test_manifest_arms_match_sources_and_load_with_derived_arms():
    payload = json.loads(MANIFEST.read_text())
    declared = list(payload["arms"])
    sources = sorted({row["source"] for row in payload["states"]})
    assert sorted(declared) == sources == ["baseline_uniform", "eval_heldout", "mulligan_sobol"]
    derived_arms = [ArmSpec(key, "teleop") for key in declared]
    targets, _ = _load_initial_state_manifest(MANIFEST, derived_arms, expected_task="square_d2")
    assert len(targets) == 250


def test_format_includes_peg_coords():
    targets, _ = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="square_d2")
    line = _format_initial_state_target(targets[0])
    assert "nut x=" in line and "peg x=" in line
    assert _initial_state_setup_subject(targets[0], "square_d2") == "nut and square peg"


def test_card_points_chosen_matches_row_and_is_a_grid_point():
    targets, _ = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="square_d2")
    target = targets[0]
    points = placement_card_points(target, "square_d2")
    assert len(points) == 1
    chosen, grid = points[0]
    assert chosen is not None
    assert chosen == (target.peg_x, target.peg_y)
    assert len(grid) == 15
    assert min(abs(chosen[0] - gx) + abs(chosen[1] - gy) for gx, gy in grid) < 1e-9
    grid_in = {(round(gx / INCH_TO_M, 1), round(gy / INCH_TO_M, 1)) for gx, gy in grid}
    assert grid_in == {(x, y) for x in (9.5, 10.5, 11.5) for y in (-4.5, -3.5, -2.5, -1.5, -0.5)}


def _bad_manifest(tmp_path: Path, mutate) -> Path:
    payload = json.loads(MANIFEST.read_text())
    row = copy.deepcopy(payload["states"][0])
    mutate(row)
    payload["states"] = [row]
    out = tmp_path / "bad_square_d2_manifest.json"
    out.write_text(json.dumps(payload))
    return out


def test_off_grid_peg_fails_loud(tmp_path):
    def mutate(row):
        row["peg_x"] = float(row["peg_x"]) + 0.5 * INCH_TO_M  # half-inch off the 1-inch grid

    with pytest.raises(ValueError, match="grid point"):
        _load_initial_state_manifest(
            _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="square_d2"
        )


@pytest.mark.parametrize(("key", "value"), [("nut_x", 99.0), ("nut_y", -0.5)])
def test_out_of_bounds_nut_fails_loud(tmp_path, key, value):
    # Same registry range check as the marker pen (nut_x +/-5 in, nut_y +/-9 in).
    def mutate(row):
        row[key] = value

    with pytest.raises(ValueError, match=f"{key}=.* out of bounds for square_d2"):
        _load_initial_state_manifest(
            _bad_manifest(tmp_path, mutate), list(ARMS), expected_task="square_d2"
        )


def test_released_square_d2_manifests_are_in_bounds():
    for path in sorted((REPO_ROOT / "data/real/manifests/square_d2").rglob("*.json")):
        payload = json.loads(path.read_text())
        arms = [ArmSpec(key, "m") for key in sorted({row["source"] for row in payload["states"]})]
        _load_initial_state_manifest(path, arms, expected_task="square_d2")


def test_render_square_d2_target_card(tmp_path):
    targets, meta = _load_initial_state_manifest(MANIFEST, list(ARMS), expected_task="square_d2")
    path = write_initial_state_card(
        targets[0],
        meta,
        output_dir=tmp_path,
        task_name="square_d2",
        style=CardStyle(),
    )
    assert path.exists() and path.stat().st_size > 5_000
