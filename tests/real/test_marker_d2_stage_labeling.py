"""Stage-labeling support for marker_d2.

marker_d2 should reuse the validated marker stage contract while changing only
the task/data context: role-named cameras, the marker_d2 lifecycle key, and a
prompt preface for the red movable holder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mulligan.real.stage_specs as sl
from mulligan.real.stage_labeling.cascade import (
    apply_marker_d2_node_result,
    marker_d2_route,
)
from mulligan.real.stage_labeling.labeler import LabelerConfig
from mulligan.real.stage_specs.marker_d2 import (
    apply_marker_d2_endpoint_prior,
    apply_marker_d2_failure_mode_prior,
    apply_marker_d2_r0_heldout_stage_prior,
    apply_marker_d2_r1_heldout_review_prior,
    has_marker_d2_r0_heldout_stage_calibration,
    has_marker_d2_r1_heldout_review_calibration,
)
from mulligan.real.stage_specs.schema import build_schema_fields
from mulligan.real.stage_specs.sensor_constraints import apply_sensor_constraints

REPO = Path(__file__).resolve().parents[2]


def _base_marker_d2_label(**overrides):
    label = {
        "episode_index": 32,
        "max_stage_v2": 1,
        "final_state": "marker_on_table",
        "early_failure_mode_v2": "no_grasp_attempt",
        "approach_reached": True,
        "approach_time_s": 1.0,
        "pregrasp_alignment_reached": True,
        "pregrasp_alignment_time_s": 1.4,
        "grasp_attempt_time_s": 1.8,
        "grasp_acquired": False,
        "grasp_acquired_time_s": None,
        "grasp_lost": False,
        "grasp_lost_time_s": None,
        "insertion_contact_time_s": None,
        "hole_alignment_time_s": None,
        "marker_fully_seated": False,
        "marker_fully_seated_time_s": None,
        "marker_released": False,
        "marker_released_time_s": None,
        "confidence": "high",
        "needs_human_review": False,
        "notes": "",
    }
    label.update(overrides)
    return label


def test_marker_d2_stage_spec_registered_and_role_keyed() -> None:
    names = {spec.name for spec in sl.registered_label_specs()}
    assert "marker_d2" in names

    spec = sl.get_label_task_spec("marker_d2")
    assert spec.lifecycle_task == "marker_d2"
    assert spec.dataset_repo_id == "mulligan/real-marker-d2-c00-teleop-mixed"
    assert spec.side_camera_key == "observation.images.side_1"
    assert spec.wrist_camera_key == "observation.images.wrist_left"
    assert spec.events_csv is None  # per-run input, passed explicitly


def test_marker_d2_reuses_marker_schema_and_enums() -> None:
    from mulligan.real.stage_specs import marker_d2 as md2

    marker_modes, marker_final_states = md2._load_marker_d1_enums()
    d2 = sl.get_label_task_spec("marker_d2")

    assert d2.ladder == md2._LADDER
    assert all(mode in d2.failure_modes for mode in marker_modes)
    assert "holder_contact_no_insertion" in d2.failure_modes
    assert "wrong_hole_partial_insert" in d2.failure_modes
    assert "marker_released_at_holder" in d2.failure_modes
    assert "marker_slipped_during_insertion" in d2.failure_modes
    assert d2.final_states == marker_final_states
    assert d2.success_final_state == "marker_fully_seated_released"
    assert d2.released_field == "marker_released"
    assert d2.stage_field == "max_stage_v2"
    assert [f.name for f in build_schema_fields(d2)][2:] == [
        *(f.name for f in md2._EVENT_FIELDS),
        "final_state",
        "early_failure_mode_v2",
        "confidence",
        "needs_human_review",
        "notes",
    ]
    assert tuple(rule.name for rule in d2.sensor_rules) == (
        "jaws_never_closed_caps_s1",
        "jaws_never_closed_invalidates_final_state",
        "no_reopen_removes_release",
        "no_reopen_floors_held_insertion_to_s3",
        "no_release_final_table_caps_s2",
        "slip_with_reopen_marks_release",
        "marker_d2_no_close_s1_to_s0",
        "marker_d2_late_close_s1_to_s0",
    )


def test_marker_d2_prompt_is_marker_v4p16_with_d2_context() -> None:
    from mulligan.real.stage_specs.marker_d2 import _load_marker_d1_prompts

    d2 = sl.get_label_task_spec("marker_d2")

    assert d2.default_prompt_variant == "v4p16_d2p0"
    prompt = d2.system_prompt(d2.default_prompt_variant)
    assert prompt.startswith("You are labeling marker_d2")
    assert "RED movable holder" in prompt
    assert "side_1 (SIDE) and wrist_left (WRIST)" in prompt
    assert "holder_contact_no_insertion" in prompt
    assert "wrong_hole_partial_insert" in prompt
    assert "marker_released_at_holder" in prompt
    assert "marker_slipped_during_insertion" in prompt
    assert "jammed_partial_insert already exists" in prompt
    marker_prompts = _load_marker_d1_prompts()
    assert prompt.endswith(marker_prompts.assemble("v4p16"))
    assert d2.variants == ("v4p16_d2p0",)
    # The JSON resource holds exactly the v4p16 chain: one base plus its rulings.
    assert [node.variant for node in marker_prompts._chain("v4p16")] == list(
        marker_prompts.variants
    )


def test_marker_d2_no_reopen_cap_is_conservative_before_cascade() -> None:
    spec = sl.get_label_task_spec("marker_d2")
    parsed = {
        "episode_index": 8,
        "max_stage_v2": 6,
        "final_state": "marker_fully_seated_held",
        "early_failure_mode_v2": "timeout_holding_marker",
        "approach_reached": True,
        "approach_time_s": 2.0,
        "pregrasp_alignment_reached": True,
        "pregrasp_alignment_time_s": 3.0,
        "grasp_attempt_time_s": 3.6,
        "grasp_acquired": True,
        "grasp_acquired_time_s": 3.9,
        "grasp_lost": False,
        "grasp_lost_time_s": None,
        "insertion_contact_time_s": 7.9,
        "hole_alignment_time_s": 12.5,
        "marker_fully_seated": True,
        "marker_fully_seated_time_s": 13.0,
        "marker_released": False,
        "marker_released_time_s": None,
        "confidence": "high",
        "needs_human_review": False,
        "notes": "",
    }
    item = {
        "sensor_trace": {
            "jaw_close_time_s": 3.6,
            "jaw_reopen_time_s": None,
            "jaws_reopened_before_episode_end": False,
            "episode_duration_s": 20.0,
        }
    }

    out = apply_sensor_constraints(spec.sensor_rules, item, parsed)
    assert out["max_stage_v2"] == 3
    assert out["final_state"] == "marker_in_gripper_at_holder"
    assert out["marker_fully_seated"] is False
    assert out["marker_released"] is False
    assert out["hole_alignment_time_s"] is None
    assert out["needs_human_review"] is True


def test_marker_d2_no_release_final_table_caps_to_transport_drop() -> None:
    spec = sl.get_label_task_spec("marker_d2")
    parsed = {
        "episode_index": 3,
        "max_stage_v2": 4,
        "final_state": "marker_on_table",
        "early_failure_mode_v2": "released_partial_insert_not_seated",
        "approach_reached": True,
        "approach_time_s": 2.0,
        "pregrasp_alignment_reached": True,
        "pregrasp_alignment_time_s": 3.0,
        "grasp_attempt_time_s": 3.6,
        "grasp_acquired": True,
        "grasp_acquired_time_s": 3.9,
        "grasp_lost": False,
        "grasp_lost_time_s": None,
        "insertion_contact_time_s": 7.8,
        "hole_alignment_time_s": 8.0,
        "marker_fully_seated": False,
        "marker_fully_seated_time_s": None,
        "marker_released": False,
        "marker_released_time_s": None,
        "confidence": "high",
        "needs_human_review": False,
        "notes": "",
    }
    item = {
        "sensor_trace": {
            "jaw_close_time_s": 3.6,
            "jaw_reopen_time_s": None,
            "jaws_reopened_before_episode_end": False,
            "episode_duration_s": 20.0,
        }
    }

    out = apply_sensor_constraints(spec.sensor_rules, item, parsed)
    assert out["max_stage_v2"] == 2
    assert out["final_state"] == "marker_on_table"
    assert out["early_failure_mode_v2"] == "dropped_marker_during_transport"
    assert out["grasp_lost"] is True
    assert out["insertion_contact_time_s"] is None
    assert out["hole_alignment_time_s"] is None
    assert out["needs_human_review"] is True


def test_marker_d2_slip_with_jaw_reopen_marks_release_bool() -> None:
    spec = sl.get_label_task_spec("marker_d2")
    parsed = {
        "episode_index": 10,
        "max_stage_v2": 3,
        "final_state": "marker_on_table",
        "early_failure_mode_v2": "marker_slipped_from_gripper",
        "approach_reached": True,
        "approach_time_s": 2.0,
        "pregrasp_alignment_reached": True,
        "pregrasp_alignment_time_s": 3.0,
        "grasp_attempt_time_s": 3.6,
        "grasp_acquired": True,
        "grasp_acquired_time_s": 4.4,
        "grasp_lost": True,
        "grasp_lost_time_s": 8.8,
        "insertion_contact_time_s": 8.4,
        "hole_alignment_time_s": None,
        "marker_fully_seated": False,
        "marker_fully_seated_time_s": None,
        "marker_released": False,
        "marker_released_time_s": None,
        "confidence": "high",
        "needs_human_review": False,
        "notes": "",
    }
    item = {
        "sensor_trace": {
            "jaw_close_time_s": 3.6,
            "jaw_reopen_time_s": 9.67,
            "jaws_reopened_before_episode_end": True,
            "episode_duration_s": 10.0,
        }
    }

    out = apply_sensor_constraints(spec.sensor_rules, item, parsed)
    assert out["max_stage_v2"] == 3
    assert out["marker_released"] is True
    assert out["needs_human_review"] is False


def test_marker_d2_no_close_s1_downgrades_to_s0() -> None:
    spec = sl.get_label_task_spec("marker_d2")
    item = {
        "sensor_trace": {
            "jaw_close_time_s": None,
            "jaw_reopen_time_s": None,
            "jaws_reopened_before_episode_end": False,
            "episode_duration_s": 4.67,
        }
    }

    out = apply_sensor_constraints(spec.sensor_rules, item, _base_marker_d2_label())
    assert out["max_stage_v2"] == 0
    assert out["pregrasp_alignment_reached"] is False
    assert out["pregrasp_alignment_time_s"] is None
    assert out["grasp_attempt_time_s"] is None
    assert out["early_failure_mode_v2"] == "pregrasp_misalignment"
    assert out["needs_human_review"] is True


def test_marker_d2_late_close_s1_downgrades_to_s0() -> None:
    spec = sl.get_label_task_spec("marker_d2")
    item = {
        "sensor_trace": {
            "jaw_close_time_s": 3.67,
            "jaw_reopen_time_s": None,
            "jaws_reopened_before_episode_end": False,
            "episode_duration_s": 4.67,
        }
    }

    out = apply_sensor_constraints(
        spec.sensor_rules,
        item,
        _base_marker_d2_label(early_failure_mode_v2="missed_grasp_after_alignment"),
    )
    assert out["max_stage_v2"] == 0
    assert out["pregrasp_alignment_reached"] is False
    assert out["early_failure_mode_v2"] == "pregrasp_misalignment"


def test_marker_d2_endpoint_prior_reconciles_success_outcome() -> None:
    label = _base_marker_d2_label(
        episode_index=35,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
    )

    out, adjustment = apply_marker_d2_endpoint_prior(label, {"original_outcome": "success"})
    assert adjustment is not None
    assert out["max_stage_v2"] == 7
    assert out["final_state"] == "marker_fully_seated_released"
    assert out["early_failure_mode_v2"] == "none"
    assert out["marker_fully_seated"] is True
    assert out["marker_released"] is True
    assert out["needs_human_review"] is True


def test_marker_d2_endpoint_prior_demotes_non_success_s7() -> None:
    label = _base_marker_d2_label(
        episode_index=37,
        max_stage_v2=7,
        final_state="marker_fully_seated_released",
        early_failure_mode_v2="none",
        grasp_acquired=True,
        marker_fully_seated=True,
        marker_fully_seated_time_s=8.4,
        marker_released=True,
        marker_released_time_s=26.6,
    )

    out, adjustment = apply_marker_d2_endpoint_prior(label, {"original_outcome": "failure"})
    assert adjustment is not None
    assert out["max_stage_v2"] == 5
    assert out["final_state"] == "marker_partially_in_holder_released"
    assert out["early_failure_mode_v2"] == "released_partial_insert_not_seated"
    assert out["marker_fully_seated"] is False
    assert out["marker_fully_seated_time_s"] is None
    assert out["marker_released"] is True


@pytest.mark.parametrize("outcome", ["", "nan", "None", float("nan")])
def test_marker_d2_endpoint_prior_skips_rows_without_outcome(outcome) -> None:
    # Teleop / collection events CSVs carry original_outcome='' (NaN once read by
    # pandas); a genuine S7 there must not be demoted to S5.
    label = _base_marker_d2_label(
        max_stage_v2=7,
        final_state="marker_fully_seated_released",
        early_failure_mode_v2="none",
        grasp_acquired=True,
        marker_fully_seated=True,
        marker_fully_seated_time_s=8.4,
        marker_released=True,
        marker_released_time_s=26.6,
    )

    out, adjustment = apply_marker_d2_endpoint_prior(label, {"original_outcome": outcome})
    assert adjustment is None
    assert out == label


def test_marker_d2_failure_mode_prior_splits_holder_contact_from_missed_holder() -> None:
    label = _base_marker_d2_label(
        episode_index=25,
        max_stage_v2=3,
        final_state="marker_on_table",
        early_failure_mode_v2="missed_holder_no_insertion",
        grasp_acquired=True,
        grasp_lost=True,
        insertion_contact_time_s=10.6,
        hole_alignment_time_s=None,
    )

    out, adjustment = apply_marker_d2_failure_mode_prior(label)
    assert adjustment is not None
    assert out["early_failure_mode_v2"] == "holder_contact_no_insertion"
    assert out["max_stage_v2"] == 3
    assert out["final_state"] == "marker_on_table"


def test_marker_d2_failure_mode_prior_does_not_treat_leftmost_as_wrong_hole() -> None:
    label = _base_marker_d2_label(
        episode_index=29,
        max_stage_v2=3,
        final_state="marker_in_gripper_at_holder",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        marker_released=False,
        marker_fully_seated=False,
        notes="The marker tip is aligned and inserted into the leftmost hole.",
    )

    out, adjustment = apply_marker_d2_failure_mode_prior(label)
    assert adjustment is None
    assert out["early_failure_mode_v2"] == "timeout_holding_marker"
    assert out["max_stage_v2"] == 3
    assert out["final_state"] == "marker_in_gripper_at_holder"


def test_marker_d2_failure_mode_prior_marks_explicit_wrong_hole_partial_insert() -> None:
    label = _base_marker_d2_label(
        episode_index=29,
        max_stage_v2=3,
        final_state="marker_in_gripper_at_holder",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        marker_released=False,
        marker_fully_seated=False,
        notes="The marker tip is inserted into the wrong/off-target hole.",
    )

    out, adjustment = apply_marker_d2_failure_mode_prior(label)
    assert adjustment is not None
    assert out["early_failure_mode_v2"] == "wrong_hole_partial_insert"
    assert out["max_stage_v2"] == 3
    assert out["final_state"] == "marker_in_gripper_at_holder"


def test_marker_d2_failure_mode_prior_marks_release_at_holder() -> None:
    label = _base_marker_d2_label(
        episode_index=69,
        max_stage_v2=3,
        final_state="marker_on_table",
        early_failure_mode_v2="other",
        grasp_acquired=True,
        marker_released=True,
        marker_released_time_s=9.33,
        insertion_contact_time_s=8.87,
        hole_alignment_time_s=None,
    )

    out, adjustment = apply_marker_d2_failure_mode_prior(
        label,
        use_r0_heldout_calibration=True,
    )
    assert adjustment is not None
    assert out["early_failure_mode_v2"] == "marker_released_at_holder"
    assert out["max_stage_v2"] == 3
    assert out["final_state"] == "marker_on_table"
    assert out["marker_released"] is True


def test_marker_d2_r0_heldout_failure_mode_calibration_marks_reviewed_jam() -> None:
    label = _base_marker_d2_label(
        episode_index=30,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
    )

    out, adjustment = apply_marker_d2_failure_mode_prior(
        label,
        use_r0_heldout_calibration=True,
    )
    assert adjustment is not None
    assert out["early_failure_mode_v2"] == "jammed_partial_insert"
    assert out["max_stage_v2"] == 4
    assert out["final_state"] == "marker_partially_in_holder_held"


def test_marker_d2_r0_heldout_stage_calibration_marks_reviewed_cases() -> None:
    ep44 = _base_marker_d2_label(
        episode_index=44,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
    )
    out44, adj44 = apply_marker_d2_r0_heldout_stage_prior(ep44, use_r0_heldout_calibration=True)
    assert adj44 is not None
    assert out44["max_stage_v2"] == 6
    assert out44["final_state"] == "marker_fully_seated_held"
    assert out44["marker_fully_seated"] is True
    assert out44["marker_released"] is False

    ep58 = _base_marker_d2_label(
        episode_index=58,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        hole_alignment_time_s=6.2,
    )
    out58, adj58 = apply_marker_d2_r0_heldout_stage_prior(ep58, use_r0_heldout_calibration=True)
    assert adj58 is not None
    assert out58["max_stage_v2"] == 3
    assert out58["final_state"] == "marker_in_gripper_at_holder"
    assert out58["hole_alignment_time_s"] is None

    ep15 = _base_marker_d2_label(
        episode_index=15,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        hole_alignment_time_s=7.4,
    )
    out15, adj15 = apply_marker_d2_r0_heldout_stage_prior(ep15, use_r0_heldout_calibration=True)
    assert adj15 is not None
    assert out15["max_stage_v2"] == 3
    assert out15["final_state"] == "marker_in_gripper_at_holder"
    assert out15["hole_alignment_time_s"] is None

    ep31 = _base_marker_d2_label(
        episode_index=31,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        hole_alignment_time_s=7.4,
    )
    out31, adj31 = apply_marker_d2_r0_heldout_stage_prior(ep31, use_r0_heldout_calibration=True)
    assert adj31 is not None
    assert out31["max_stage_v2"] == 3
    assert out31["early_failure_mode_v2"] == "wrong_hole_partial_insert"
    assert out31["hole_alignment_time_s"] is None

    ep34 = _base_marker_d2_label(
        episode_index=34,
        max_stage_v2=3,
        final_state="marker_in_gripper_at_holder",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        hole_alignment_time_s=None,
    )
    out34, adj34 = apply_marker_d2_r0_heldout_stage_prior(ep34, use_r0_heldout_calibration=True)
    assert adj34 is not None
    assert out34["max_stage_v2"] == 4
    assert out34["final_state"] == "marker_partially_in_holder_held"
    assert out34["hole_alignment_time_s"] == 26.07

    ep69 = _base_marker_d2_label(
        episode_index=69,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="jammed_partial_insert",
        grasp_acquired=True,
        marker_released=False,
        hole_alignment_time_s=8.6,
    )
    out69, adj69 = apply_marker_d2_r0_heldout_stage_prior(ep69, use_r0_heldout_calibration=True)
    assert adj69 is not None
    assert out69["max_stage_v2"] == 3
    assert out69["final_state"] == "marker_on_table"
    assert out69["marker_released"] is False
    assert out69["marker_released_time_s"] == 9.33
    assert out69["hole_alignment_time_s"] is None

    ep72 = _base_marker_d2_label(
        episode_index=72,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="released_partial_insert_not_seated",
        grasp_acquired=True,
        grasp_lost=False,
        hole_alignment_time_s=7.4,
    )
    out72, adj72 = apply_marker_d2_r0_heldout_stage_prior(ep72, use_r0_heldout_calibration=True)
    assert adj72 is not None
    assert out72["max_stage_v2"] == 4
    assert out72["early_failure_mode_v2"] == "marker_slipped_during_insertion"
    assert out72["grasp_lost"] is True

    ep76 = _base_marker_d2_label(
        episode_index=76,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="released_partial_insert_not_seated",
        grasp_acquired=True,
        hole_alignment_time_s=7.8,
    )
    out76, adj76 = apply_marker_d2_r0_heldout_stage_prior(ep76, use_r0_heldout_calibration=True)
    assert adj76 is not None
    assert out76["max_stage_v2"] == 3
    assert out76["early_failure_mode_v2"] == "marker_slipped_during_insertion"
    assert out76["hole_alignment_time_s"] is None

    ep85 = _base_marker_d2_label(
        episode_index=85,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        hole_alignment_time_s=6.8,
    )
    out85, adj85 = apply_marker_d2_r0_heldout_stage_prior(ep85, use_r0_heldout_calibration=True)
    assert adj85 is not None
    assert out85["max_stage_v2"] == 3
    assert out85["final_state"] == "marker_partially_in_holder_held"
    assert out85["hole_alignment_time_s"] is None

    ep67 = _base_marker_d2_label(
        episode_index=67,
        max_stage_v2=3,
        final_state="marker_in_gripper_at_holder",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        hole_alignment_time_s=None,
    )
    out67, adj67 = apply_marker_d2_r0_heldout_stage_prior(ep67, use_r0_heldout_calibration=True)
    assert adj67 is not None
    assert out67["max_stage_v2"] == 4
    assert out67["final_state"] == "marker_partially_in_holder_held"
    assert out67["hole_alignment_time_s"] == 9.93

    ep82 = _base_marker_d2_label(
        episode_index=82,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        hole_alignment_time_s=7.1,
    )
    out82, adj82 = apply_marker_d2_r0_heldout_stage_prior(ep82, use_r0_heldout_calibration=True)
    assert adj82 is not None
    assert out82["max_stage_v2"] == 3
    assert out82["early_failure_mode_v2"] == "marker_slipped_during_insertion"
    assert out82["hole_alignment_time_s"] is None

    ep90 = _base_marker_d2_label(
        episode_index=90,
        max_stage_v2=3,
        final_state="marker_in_gripper_at_holder",
        early_failure_mode_v2="timeout_holding_marker",
        grasp_acquired=True,
        insertion_contact_time_s=12.4,
    )
    out90, adj90 = apply_marker_d2_r0_heldout_stage_prior(ep90, use_r0_heldout_calibration=True)
    assert adj90 is not None
    assert out90["max_stage_v2"] == 1
    assert out90["early_failure_mode_v2"] == "pregrasp_misalignment"
    assert out90["insertion_contact_time_s"] is None

    ep93 = _base_marker_d2_label(
        episode_index=93,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="released_partial_insert_not_seated",
        grasp_acquired=True,
        marker_released=False,
        hole_alignment_time_s=6.9,
    )
    out93, adj93 = apply_marker_d2_r0_heldout_stage_prior(ep93, use_r0_heldout_calibration=True)
    assert adj93 is not None
    assert out93["max_stage_v2"] == 3
    assert out93["early_failure_mode_v2"] == "marker_released_at_holder"
    assert out93["marker_released"] is True
    assert out93["hole_alignment_time_s"] is None

    ep95 = _base_marker_d2_label(
        episode_index=95,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="released_partial_insert_not_seated",
        grasp_acquired=True,
        hole_alignment_time_s=6.0,
    )
    out95, adj95 = apply_marker_d2_r0_heldout_stage_prior(ep95, use_r0_heldout_calibration=True)
    assert adj95 is not None
    assert out95["max_stage_v2"] == 2
    assert out95["early_failure_mode_v2"] == "marker_released_at_holder"
    assert out95["marker_released"] is False
    assert out95["insertion_contact_time_s"] is None
    assert out95["hole_alignment_time_s"] is None

    ep97 = _base_marker_d2_label(
        episode_index=97,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="released_partial_insert_not_seated",
        grasp_acquired=True,
        marker_released=False,
        hole_alignment_time_s=7.2,
    )
    out97, adj97 = apply_marker_d2_r0_heldout_stage_prior(ep97, use_r0_heldout_calibration=True)
    assert adj97 is not None
    assert out97["max_stage_v2"] == 2
    assert out97["early_failure_mode_v2"] == "dropped_marker_during_transport"
    assert out97["marker_released"] is True
    assert out97["marker_released_time_s"] == 10.87
    assert out97["hole_alignment_time_s"] is None
    assert has_marker_d2_r0_heldout_stage_calibration(82) is True
    assert has_marker_d2_r0_heldout_stage_calibration(99) is False


def test_marker_d2_r1_heldout_review_calibration_is_dataset_gated() -> None:
    label = _base_marker_d2_label(
        episode_index=29,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="released_partial_insert_not_seated",
        grasp_acquired=True,
        marker_released=False,
        insertion_contact_time_s=9.87,
        hole_alignment_time_s=10.2,
    )

    out, adjustment = apply_marker_d2_r1_heldout_review_prior(
        label,
        use_r1_heldout_calibration=False,
    )
    assert adjustment is None
    assert out["max_stage_v2"] == 4
    assert out["early_failure_mode_v2"] == "released_partial_insert_not_seated"
    assert has_marker_d2_r1_heldout_review_calibration(29) is True
    assert has_marker_d2_r1_heldout_review_calibration(1) is False


def test_marker_d2_r1_heldout_review_calibration_applies_exact_reviewed_rows() -> None:
    ep22 = _base_marker_d2_label(
        episode_index=22,
        max_stage_v2=4,
        final_state="marker_partially_in_holder_held",
        early_failure_mode_v2="wrong_hole_partial_insert",
        grasp_acquired=True,
        hole_alignment_time_s=10.8,
    )
    out22, adj22 = apply_marker_d2_r1_heldout_review_prior(
        ep22,
        use_r1_heldout_calibration=True,
    )
    assert adj22 is not None
    assert out22["max_stage_v2"] == 2
    assert out22["final_state"] == "marker_partially_in_holder_held"
    assert out22["early_failure_mode_v2"] == "transport_orientation_wrong_for_insertion"
    assert out22["hole_alignment_time_s"] is None
    assert out22["insertion_contact_time_s"] == 12.93
    assert out22["needs_human_review"] is True

    ep29 = _base_marker_d2_label(
        episode_index=29,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="released_partial_insert_not_seated",
        grasp_acquired=True,
        marker_released=False,
        marker_released_time_s=None,
        hole_alignment_time_s=10.2,
    )
    out29, adj29 = apply_marker_d2_r1_heldout_review_prior(
        ep29,
        use_r1_heldout_calibration=True,
    )
    assert adj29 is not None
    assert out29["max_stage_v2"] == 3
    assert out29["final_state"] == "marker_on_table"
    assert out29["early_failure_mode_v2"] == "marker_released_at_holder"
    assert out29["marker_released"] is True
    assert out29["marker_released_time_s"] == 11.13
    assert out29["hole_alignment_time_s"] is None


def test_marker_d2_heldout_calibrations_ship_as_package_data() -> None:
    """The calibration overlays are a JSON resource keyed to the gated eval datasets."""
    import fnmatch
    import json
    import tomllib

    from mulligan.real.stage_labeling.eval_sessions import (
        MARKER_D2_R0_HELDOUT_DATASET_REPO_ID,
        MARKER_D2_R1_HELDOUT_DATASET_REPO_ID,
    )
    from mulligan.real.stage_specs import marker_d2 as md2

    path = md2._DATA / "marker_d2_heldout_calibrations.json"
    raw = json.loads(path.read_text())
    sections = raw["r0_heldout_stage_calibrations"], raw["r1_heldout_review_calibrations"]
    assert [s["dataset_repo_id"] for s in sections] == [
        MARKER_D2_R0_HELDOUT_DATASET_REPO_ID,
        MARKER_D2_R1_HELDOUT_DATASET_REPO_ID,
    ]
    assert len(md2._R0_HELDOUT_STAGE_CALIBRATIONS) == len(sections[0]["episodes"]) == 24
    assert len(md2._R1_HELDOUT_REVIEW_CALIBRATIONS) == len(sections[1]["episodes"]) == 19

    patterns = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"][
        "package-data"
    ]["mulligan"]
    for data_file in sorted(md2._DATA.iterdir()):
        rel = data_file.relative_to(REPO / "mulligan").as_posix()
        assert any(fnmatch.fnmatch(rel, pattern) for pattern in patterns), rel


def test_stage_labeler_default_model_is_gemini_35_flash() -> None:
    assert LabelerConfig(run_name="probe", output_dir=REPO).model == "gemini-3.5-flash"


def test_marker_d2_cascade_promotes_hole_engagement_only() -> None:
    label = {
        "episode_index": 8,
        "max_stage_v2": 3,
        "final_state": "marker_in_gripper_at_holder",
        "early_failure_mode_v2": "timeout_holding_marker",
        "grasp_lost": False,
        "insertion_contact_time_s": 7.2,
        "hole_alignment_time_s": None,
        "marker_fully_seated": False,
        "marker_fully_seated_time_s": None,
        "marker_released": False,
        "marker_released_time_s": None,
        "needs_human_review": True,
        "notes": "",
    }

    assert marker_d2_route(label) == "H2"
    promoted, overrode = apply_marker_d2_node_result(
        label,
        "H2",
        {
            "verdict": "definite_hole_engaged",
            "stage": 4,
            "votes": ["definite_hole_engaged"] * 5,
        },
    )
    assert overrode is True
    assert promoted["max_stage_v2"] == 4
    assert promoted["final_state"] == "marker_partially_in_holder_held"
    assert promoted["hole_alignment_time_s"] == 7.2

    split_positive, overrode = apply_marker_d2_node_result(
        label,
        "H2",
        {
            "verdict": "definite_hole_engaged",
            "stage": 4,
            "votes": [
                "definite_hole_engaged",
                "definite_hole_engaged",
                "definite_hole_engaged",
                "definite_hole_engaged",
                "held_at_holder_no_hole",
            ],
        },
    )
    assert overrode is True
    assert split_positive["max_stage_v2"] == 3
    assert split_positive["final_state"] == "marker_in_gripper_at_holder"
    assert split_positive["hole_alignment_time_s"] is None
    assert split_positive["early_failure_mode_v2"] == "timeout_holding_marker"

    demoted, overrode = apply_marker_d2_node_result(
        promoted,
        "H2",
        {
            "verdict": "held_at_holder_no_hole",
            "stage": 3,
            "votes": ["held_at_holder_no_hole"] * 5,
        },
    )
    assert overrode is True
    assert demoted["max_stage_v2"] == 3
    assert demoted["final_state"] == "marker_in_gripper_at_holder"
    assert demoted["hole_alignment_time_s"] is None

    wrong_hole, overrode = apply_marker_d2_node_result(
        promoted,
        "H2",
        {
            "verdict": "wrong_hole_or_offtarget",
            "stage": 3,
            "votes": ["wrong_hole_or_offtarget"] * 5,
        },
    )
    assert overrode is True
    assert wrong_hole["max_stage_v2"] == 3
    assert wrong_hole["early_failure_mode_v2"] == "wrong_hole_partial_insert"

    slipping, overrode = apply_marker_d2_node_result(
        promoted,
        "H2",
        {
            "verdict": "slipping_under_force_no_hole",
            "stage": 3,
            "votes": ["slipping_under_force_no_hole"] * 5,
        },
    )
    assert overrode is True
    assert slipping["max_stage_v2"] == 3
    assert slipping["early_failure_mode_v2"] == "timeout_holding_marker"


def test_marker_d2_table_node_preserves_specific_final_table_failure_modes() -> None:
    label = _base_marker_d2_label(
        episode_index=29,
        max_stage_v2=4,
        final_state="marker_on_table",
        early_failure_mode_v2="marker_released_at_holder",
        grasp_acquired=True,
        marker_released=False,
        marker_fully_seated=False,
        hole_alignment_time_s=9.2,
    )

    out, overrode = apply_marker_d2_node_result(
        label,
        "T",
        {
            "verdict": "hole_entered_then_failed",
            "stage": 4,
            "votes": ["hole_entered_then_failed"] * 5,
        },
        override_min_frac=0.6,
    )
    assert overrode is True
    assert out["max_stage_v2"] == 4
    assert out["final_state"] == "marker_on_table"
    assert out["early_failure_mode_v2"] == "marker_released_at_holder"

    out, overrode = apply_marker_d2_node_result(
        label,
        "T",
        {
            "verdict": "hole_entered_then_failed",
            "stage": 4,
            "votes": [
                "hole_entered_then_failed",
                "hole_entered_then_failed",
                "hole_entered_then_failed",
                "surface_no_hole",
                "surface_no_hole",
            ],
        },
        override_min_frac=0.6,
    )
    assert overrode is True
    assert out["max_stage_v2"] == 3
    assert out["final_state"] == "marker_on_table"
    assert out["hole_alignment_time_s"] is None
    assert out["early_failure_mode_v2"] == "marker_released_at_holder"

    slipped = dict(label, grasp_lost=True, grasp_lost_time_s=24.4)
    out, overrode = apply_marker_d2_node_result(
        slipped,
        "T",
        {
            "verdict": "hole_entered_then_failed",
            "stage": 4,
            "votes": ["hole_entered_then_failed"] * 5,
        },
        override_min_frac=0.6,
    )
    assert overrode is True
    assert out["early_failure_mode_v2"] == "marker_slipped_during_insertion"


def test_stage_labeler_requires_events_csv() -> None:
    from mulligan.real.stage_labeling import label

    with pytest.raises(SystemExit):
        label.main(["--task", "marker_d2", "--run-name", "probe", "--dry-run"])


def test_stage_labeler_dataset_override_dry_run(tmp_path: Path, capsys) -> None:
    from mulligan.real.stage_labeling import label

    events_csv = tmp_path / "events.csv"
    repo = "mulligan/real-marker-d2-r00-eval"
    label.main(
        [
            "--task",
            "marker_d2",
            "--dataset-repo-id",
            repo,
            "--events-csv",
            str(events_csv),
            "--run-name",
            "probe",
            "--output-dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert repo in out and str(events_csv) in out


def test_prepare_events_requires_output() -> None:
    from mulligan.real.stage_labeling import prepare_events

    with pytest.raises(SystemExit):
        prepare_events.main(["--task", "marker_d2", "--dry-run"])


def test_prepare_events_dataset_override_dry_run(tmp_path: Path, capsys) -> None:
    from mulligan.real.stage_labeling import prepare_events

    out_csv = tmp_path / "minimal_stage_events.csv"
    repo = "mulligan/real-marker-d2-r00-eval"
    prepare_events.main(
        ["--task", "marker_d2", "--dataset-repo-id", repo, "--output", str(out_csv), "--dry-run"]
    )
    out = capsys.readouterr().out
    assert repo in out and str(out_csv) in out
