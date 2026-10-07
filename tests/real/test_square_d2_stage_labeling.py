"""Stage-labeling support for square_d2."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import mulligan.real.stage_specs as sl
from mulligan.real.stage_specs.schema import build_schema_fields

REPO = Path(__file__).resolve().parents[2]


def test_square_d2_stage_spec_registered_and_role_keyed() -> None:
    names = {spec.name for spec in sl.registered_label_specs()}
    assert "square_d2" in names

    spec = sl.get_label_task_spec("square_d2")
    assert spec.lifecycle_task == "square_d2"
    assert spec.dataset_repo_id == "mulligan/real-square-d2-c00-teleop-mixed"
    assert spec.side_camera_key == "observation.images.side_1"
    assert spec.wrist_camera_key == "observation.images.wrist_left"
    assert spec.events_csv is None  # per-run input, passed explicitly


def test_square_d2_keeps_the_inherited_square_contract() -> None:
    # The earlier fixed-peg spec is inlined into square_d2 (not registered).
    with pytest.raises(KeyError):
        sl.get_label_task_spec("Square_D1")
    square_d2 = sl.get_label_task_spec("square_d2")

    assert square_d2.ladder.success_level == 7
    assert square_d2.success_final_state == "nut_fully_seated_released"
    assert square_d2.released_field == "nut_released"
    assert square_d2.stage_field == "max_stage"
    assert "nut_hole_missed_peg" in square_d2.failure_modes
    assert [f.name for f in build_schema_fields(square_d2)][:3] == [
        "episode_index",
        "max_stage",
        "approach_reached",
    ]
    assert tuple(rule.name for rule in square_d2.sensor_rules)[:3] == (
        "jaws_never_closed_caps_s1",
        "jaws_never_closed_invalidates_final_state",
        "no_reopen_removes_release",
    )
    assert (square_d2.release_final_abs_max, square_d2.release_plateau_margin) == (0.6, 0.05)


def test_square_d2_prompt_is_square_v0p5_with_d2_context() -> None:
    from mulligan.real.stage_specs.square_d2 import _FIXED_PEG_PROMPTS

    square_d2 = sl.get_label_task_spec("square_d2")

    assert square_d2.default_prompt_variant == "v0p5_d2p3"
    base_prompt = square_d2.system_prompt("v0p5_d2p0")
    assert base_prompt.startswith("You are labeling square_d2")
    assert "peg is no longer fixed" in base_prompt
    assert "x in {9.5, 10.5, 11.5}" in base_prompt
    assert "side_1 (SIDE) and wrist_left (WRIST)" in base_prompt
    assert "Inherited calibrated square prompt follows" in base_prompt
    assert _FIXED_PEG_PROMPTS.assemble("v0p5") in base_prompt
    assert set(square_d2.variants) == {"v0p5_d2p0", "v0p5_d2p3"}
    prompt = square_d2.system_prompt("v0p5_d2p3")
    assert "HELD-TIMEOUT S2/S3 PEG-AREA GATE" in prompt
    assert "side of the peg" in prompt
    assert "S2 / nut_in_gripper_away_from_peg" in prompt


def test_register_rejects_unknown_lifecycle_task() -> None:
    from mulligan.real.stage_specs.tasks import register_label_task_spec

    spec = sl.get_label_task_spec("square_d2")
    bad = dataclasses.replace(
        spec,
        name="bad_lifecycle_probe",
        lifecycle_task="not_a_real_lifecycle_task",
    )
    with pytest.raises(ValueError, match="lifecycle_task"):
        register_label_task_spec(bad)
