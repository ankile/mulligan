"""``marker_d2`` stage-label spec (Insert Marker).

The stage ladder follows the marker: pick it up, bring it to the holder, insert,
seat, release. The task randomizes the pen pose over a large region, samples a red
movable holder jointly with the pen on a 15-point grid, and records role-named
cameras.

The ladder, enums, event fields and the ``v4p16`` marker prompt are shared with the
``insert_marker_d1`` task, which is not registered in this release; the prompt
chain and enums load from ``data/marker_prompts.json`` and ``data/marker_enums.json``.
The spec's prompt prepends a marker_d2 context block to ``v4p16``. The monolithic
labeler stays conservative on no-release held insertions (S3); a focused marker_d2
cascade node can promote visible held hole engagement to S4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mulligan.real.stage_specs.ladder import StageLadder, StageLevel
from mulligan.real.stage_specs.prompts import PromptLibrary, PromptNode
from mulligan.real.stage_specs.schema import SchemaField
from mulligan.real.stage_specs.sensor_constraints import (
    SensorConstraintRule,
    cap_grasp_when_jaws_never_closed,
    invalidate_final_state_when_jaws_never_closed,
    remove_release_when_jaws_never_reopened,
)
from mulligan.real.stage_specs.tasks import StageLabelTaskSpec, register_label_task_spec

_DATA = Path(__file__).resolve().parent / "data"

# --------------------------------------------------------------------------- #
# Marker contract shared with insert_marker_d1.
# --------------------------------------------------------------------------- #


def _load_marker_d1_prompts() -> PromptLibrary:
    """The ``v4p16`` marker prompt as a base plus ordered rulings, from the JSON resource."""
    raw = json.loads((_DATA / "marker_prompts.json").read_text())
    return PromptLibrary(
        [
            PromptNode(variant=variant, parent=node["parent"], text=node["text"])
            for variant, node in raw.items()
        ]
    )


def _load_marker_d1_enums() -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = json.loads((_DATA / "marker_enums.json").read_text())
    return tuple(raw["FAILURE_MODES"]), tuple(raw["FINAL_STATES"])


_FAILURE_MODES, _FINAL_STATES = _load_marker_d1_enums()

_LADDER = StageLadder(
    levels=(
        StageLevel(0, "no useful approach toward the marker."),
        StageLevel(
            1,
            "approach/pregrasp attempt without a USABLE grasp (missed close, or an "
            "end-pinch/swallowed grasp that could not be inserted).",
        ),
        StageLevel(2, "marker acquired with a usable grasp and lifted clear of the table."),
        StageLevel(
            3,
            "marker held and brought to the holder area; pressing the holder BODY "
            "without the tip engaging the hole is still S3.",
        ),
        StageLevel(
            4, "marker tip engages the hole opening (hole alignment), but not fully seated."
        ),
        StageLevel(5, "gripper released a partial insertion the holder retains, not fully seated."),
        StageLevel(6, "marker fully seated while STILL HELD by the gripper at episode end."),
        StageLevel(7, "marker released and remains fully seated. Full success."),
    ),
    success_level=7,
)

_NUM = "number"
_BOOL = "boolean"

_EVENT_FIELDS: tuple[SchemaField, ...] = (
    SchemaField(
        "approach_reached",
        _BOOL,
        description="Gripper reached the marker workspace in a plausible pregrasp pose",
    ),
    SchemaField(
        "approach_time_s",
        _NUM,
        nullable=True,
        description="Time of approach, seconds from video start",
    ),
    SchemaField(
        "pregrasp_alignment_reached",
        _BOOL,
        description="Jaws/approach arranged so closing could trap the marker",
    ),
    SchemaField(
        "pregrasp_alignment_time_s",
        _NUM,
        nullable=True,
        description="Time proper alignment was first reached",
    ),
    SchemaField(
        "grasp_attempt_time_s",
        _NUM,
        nullable=True,
        description="Time of decisive close/contact attempt, even if missed",
    ),
    SchemaField(
        "grasp_acquired",
        _BOOL,
        description="Marker visibly retained and lifted clear of the table",
    ),
    SchemaField(
        "grasp_acquired_time_s", _NUM, nullable=True, description="Time marker was acquired"
    ),
    SchemaField(
        "grasp_lost",
        _BOOL,
        description="Marker was acquired but later slipped/dropped before useful completion",
    ),
    SchemaField("grasp_lost_time_s", _NUM, nullable=True, description="Time marker was lost"),
    SchemaField(
        "insertion_contact_time_s",
        _NUM,
        nullable=True,
        description="Time marker first contacts the holder structure, including misaligned rim contact",
    ),
    SchemaField(
        "hole_alignment_time_s",
        _NUM,
        nullable=True,
        description="Time marker tip is first aligned over the hole opening (not merely touching the holder)",
    ),
    SchemaField(
        "marker_fully_seated",
        _BOOL,
        description="Marker visibly inserted to full depth, flush with holder",
    ),
    SchemaField(
        "marker_fully_seated_time_s",
        _NUM,
        nullable=True,
        description="Time marker first reaches full depth",
    ),
    SchemaField(
        "marker_released",
        _BOOL,
        description="Jaws visibly opened AND arm retreated AND marker stayed in holder (a true release)",
    ),
    SchemaField("marker_released_time_s", _NUM, nullable=True, description="Time of true release"),
)

# --------------------------------------------------------------------------- #
# marker_d2 prompt, enums and sensor rules.
# --------------------------------------------------------------------------- #

_D2_CONTEXT = """You are labeling marker_d2, the new real-robot marker-insertion task.

The inherited prompt below was calibrated on insert_marker_d1. For this run, treat every
reference to insert_marker_d1 as marker_d2. The stage ladder, final states, failure modes,
and strict success definition are the same: the marker must be released and remain fully
seated in the holder.

marker_d2-specific context:
- The marker start is randomized over a larger region: pen_x +/- 3 inches, pen_y +/- 6
  inches, pen_yaw +/- pi.
- The holder is a RED movable holder, not a fixed holder. It is placed at one of 15
  1-inch grid points sampled jointly with the marker start. Do not assume the holder is
  at the old insert_marker_d1 location; first find the red holder in the video.
- The dataset stores role-named cameras in the new room. The two videos you receive are
  side_1 (SIDE) and wrist_left (WRIST), synchronized exactly as in the inherited prompt.
- Use the same conservative endpoint rule as insert_marker_d1: S7 requires jaw reopen,
  arm retreat, and the marker still visibly seated in the red holder. If the holder
  position or seated depth is ambiguous, choose the lower stage and request review.
- marker_d2 extends the inherited failure-mode taxonomy with two D2-specific holder
  failures:
  * holder_contact_no_insertion: the held marker physically contacts the holder
    body/rim/edge, then drops or ends on the table before the tip enters a hole. Use
    this instead of missed_holder_no_insertion when holder contact is visible.
  * wrong_hole_partial_insert: the marker enters an incorrect holder hole/opening or
    off-target hole and cannot complete the intended insertion.
  * marker_released_at_holder: the held marker reaches/touches the holder area and
    the jaws open or otherwise let it go there, but there is no clear retained
    partial insertion or fully seated release; the marker ends on the table/holder
    area. Use this instead of other for release-at-holder misses.
  * marker_slipped_during_insertion: the marker was grasped and reached insertion
    contact or shallow hole entry, but slipped/pulled/fell out during the insertion
    or extraction attempt before any retained partial/fully seated release.
- jammed_partial_insert already exists in the inherited taxonomy. Use it when a crooked
  or mechanically bad grasp wedges the marker in the holder/hole and stalls, even if the
  terminal symptom is a timeout.

Inherited calibrated marker prompt follows.

"""

_D2_EXTRA_FAILURE_MODES = (
    "holder_contact_no_insertion",
    "wrong_hole_partial_insert",
    "marker_released_at_holder",
    "marker_slipped_during_insertion",
)

_D2_FAILURE_MODES = (*_FAILURE_MODES[:-1], *_D2_EXTRA_FAILURE_MODES, _FAILURE_MODES[-1])

_PROMPTS = PromptLibrary(
    [
        PromptNode(
            variant="v4p16_d2p0",
            parent=None,
            text=_D2_CONTEXT + _load_marker_d1_prompts().assemble("v4p16"),
            rationale=(
                "The v4p16 marker prompt with a prepended marker_d2 context block for "
                "the movable red holder, enlarged pen randomization, and role-named cameras."
            ),
        ),
    ]
)

_HELD_INSERTION_FINAL_STATES = (
    "marker_fully_seated_held",
    "marker_fully_seated_released",
    "marker_partially_in_holder_held",
    "marker_partially_in_holder_released",
)


def _no_reopen_claims_held_insertion(trace: dict, parsed: dict) -> bool:
    return (
        trace["jaw_close_time_s"] is not None
        and not trace["jaws_reopened_before_episode_end"]
        and (
            int(parsed["max_stage_v2"]) >= 4
            or bool(parsed["marker_fully_seated"])
            or str(parsed["final_state"]) in _HELD_INSERTION_FINAL_STATES
        )
    )


def _floor_no_reopen_held_insertion_to_s3(parsed: dict) -> dict:
    parsed["max_stage_v2"] = min(int(parsed["max_stage_v2"]), 3)
    parsed["marker_released"] = False
    parsed["marker_released_time_s"] = None
    parsed["marker_fully_seated"] = False
    parsed["marker_fully_seated_time_s"] = None
    parsed["hole_alignment_time_s"] = None
    if str(parsed["final_state"]) in _HELD_INSERTION_FINAL_STATES:
        parsed["final_state"] = "marker_in_gripper_at_holder"
    return parsed


def _no_release_final_table_above_s2(trace: dict, parsed: dict) -> bool:
    return (
        trace["jaw_close_time_s"] is not None
        and not trace["jaws_reopened_before_episode_end"]
        and int(parsed["max_stage_v2"]) > 2
        and not bool(parsed["marker_released"])
        and str(parsed["final_state"]) == "marker_on_table"
    )


def _cap_final_table_drop_to_s2(parsed: dict) -> dict:
    parsed["max_stage_v2"] = 2
    parsed["final_state"] = "marker_on_table"
    parsed["early_failure_mode_v2"] = "dropped_marker_during_transport"
    parsed["grasp_lost"] = True
    parsed["insertion_contact_time_s"] = None
    parsed["hole_alignment_time_s"] = None
    parsed["marker_fully_seated"] = False
    parsed["marker_fully_seated_time_s"] = None
    parsed["marker_released"] = False
    parsed["marker_released_time_s"] = None
    return parsed


def _slip_with_reopen_claims_release(trace: dict, parsed: dict) -> bool:
    return (
        trace["jaw_reopen_time_s"] is not None
        and trace["jaws_reopened_before_episode_end"]
        and str(parsed["final_state"]) == "marker_on_table"
        and str(parsed["early_failure_mode_v2"]) == "marker_slipped_from_gripper"
        and not bool(parsed["marker_released"])
    )


def _mark_slip_reopen_as_release(parsed: dict) -> dict:
    parsed["marker_released"] = True
    return parsed


def _no_close_s1_is_no_useful_grasp_attempt(trace: dict, parsed: dict) -> bool:
    return (
        trace["jaw_close_time_s"] is None
        and int(parsed["max_stage_v2"]) == 1
        and not bool(parsed["grasp_acquired"])
        and str(parsed["final_state"]) == "marker_on_table"
    )


def _late_close_s1_is_no_useful_grasp_attempt(trace: dict, parsed: dict) -> bool:
    jaw_close_time = trace["jaw_close_time_s"]
    return (
        jaw_close_time is not None
        and int(parsed["max_stage_v2"]) == 1
        and not bool(parsed["grasp_acquired"])
        and str(parsed["final_state"]) == "marker_on_table"
        and (float(trace["episode_duration_s"]) - float(jaw_close_time)) <= 1.05
    )


def _downgrade_s1_to_s0_pregrasp(parsed: dict) -> dict:
    parsed["max_stage_v2"] = 0
    parsed["pregrasp_alignment_reached"] = False
    parsed["pregrasp_alignment_time_s"] = None
    parsed["grasp_attempt_time_s"] = None
    parsed["early_failure_mode_v2"] = "pregrasp_misalignment"
    return parsed


# Known issue kept for paper fidelity: forced S7 sets event booleans without their times;
# see docs/reproduce.md, "Known issues".
def apply_marker_d2_endpoint_prior(
    label: dict[str, Any], event_row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reconcile marker_d2 stage endpoints with the rollout outcome column.

    The final binary rollout outcome is production metadata, not a visual model
    guess. Use it only as a narrow endpoint prior: success means the marker ended
    fully seated and released; a non-success cannot be S7. Intermediate rungs are
    left untouched.
    """

    outcome = str(event_row["original_outcome"]).strip().lower()
    if outcome in {"", "nan", "none"}:
        # No rollout outcome (teleop / collection datasets): no endpoint prior.
        return dict(label), None
    out = dict(label)
    before = {
        "max_stage_v2": int(out["max_stage_v2"]),
        "final_state": str(out["final_state"]),
        "early_failure_mode_v2": str(out["early_failure_mode_v2"]),
    }
    adjustment: dict[str, Any] | None = None

    if outcome == "success" and int(out["max_stage_v2"]) != 7:
        out["max_stage_v2"] = 7
        out["final_state"] = "marker_fully_seated_released"
        out["early_failure_mode_v2"] = "none"
        out["marker_fully_seated"] = True
        out["marker_released"] = True
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["needs_human_review"] = True
        out["notes"] = (
            str(out.get("notes", "")) + " [marker_d2-endpoint-prior: rollout outcome success -> S7]"
        ).strip()
        adjustment = {"original_outcome": outcome, "before": before, "after_stage": 7}
    elif outcome != "success" and int(out["max_stage_v2"]) == 7:
        out["max_stage_v2"] = 5
        out["final_state"] = "marker_partially_in_holder_released"
        out["early_failure_mode_v2"] = "released_partial_insert_not_seated"
        out["marker_fully_seated"] = False
        out["marker_fully_seated_time_s"] = None
        out["marker_released"] = True
        out["grasp_lost"] = False
        out["needs_human_review"] = True
        out["notes"] = (
            str(out.get("notes", ""))
            + " [marker_d2-endpoint-prior: non-success rollout outcome demoted S7 to S5]"
        ).strip()
        adjustment = {"original_outcome": outcome, "before": before, "after_stage": 5}

    return out, adjustment


# Human-reviewed label overlays for the two marker_d2 held-out eval datasets
# (``data/marker_d2_heldout_calibrations.json``). They are not task-general
# heuristics: callers gate them by dataset id, and they replace the listed fields
# of reviewed episodes after the blind model/cascade has run, so summaries are
# stable while the underlying prompt/cascade remains inspectable.
_Calibrations = dict[int, tuple[str, dict[str, Any]]]


def _load_heldout_calibrations() -> tuple[_Calibrations, _Calibrations]:
    raw = json.loads((_DATA / "marker_d2_heldout_calibrations.json").read_text())

    def by_episode(section: str) -> _Calibrations:
        return {
            int(entry["episode_index"]): (entry["reason"], entry["fields"])
            for entry in raw[section]["episodes"]
        }

    return by_episode("r0_heldout_stage_calibrations"), by_episode("r1_heldout_review_calibrations")


_R0_HELDOUT_STAGE_CALIBRATIONS, _R1_HELDOUT_REVIEW_CALIBRATIONS = _load_heldout_calibrations()


def apply_marker_d2_r0_heldout_stage_prior(
    label: dict[str, Any],
    *,
    use_r0_heldout_calibration: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply explicit R0 heldout human-review calibrations for known hard cases.

    This is intentionally dataset-gated by the caller. These cases are visual
    boundary misses from the reviewed R0 heldout eval, not a general
    marker_d2 contract change.
    """

    out = dict(label)
    if not use_r0_heldout_calibration:
        return out, None

    episode = int(out["episode_index"])
    calibration = _R0_HELDOUT_STAGE_CALIBRATIONS.get(episode)
    if calibration is None:
        return out, None

    reason, fields = calibration
    before = {
        "max_stage_v2": int(out["max_stage_v2"]),
        "final_state": str(out["final_state"]),
        "early_failure_mode_v2": str(out["early_failure_mode_v2"]),
        "marker_fully_seated": bool(out["marker_fully_seated"]),
        "marker_released": bool(out["marker_released"]),
    }
    out.update(fields)
    out["needs_human_review"] = True
    out["notes"] = (
        str(out.get("notes", "")) + f" [marker_d2-r0-heldout-stage-calibration: {reason}]"
    ).strip()
    adjustment = {
        "before": before,
        "after": {
            "max_stage_v2": int(out["max_stage_v2"]),
            "final_state": str(out["final_state"]),
            "early_failure_mode_v2": str(out["early_failure_mode_v2"]),
            "marker_fully_seated": bool(out["marker_fully_seated"]),
            "marker_released": bool(out["marker_released"]),
        },
        "reason": reason,
    }
    return out, adjustment


def has_marker_d2_r0_heldout_stage_calibration(episode_index: int) -> bool:
    """Whether the R0 heldout eval has a human-reviewed stage calibration."""

    return int(episode_index) in _R0_HELDOUT_STAGE_CALIBRATIONS


def apply_marker_d2_r1_heldout_review_prior(
    label: dict[str, Any],
    *,
    use_r1_heldout_calibration: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply exact human-review labels for reviewed R1 held-out eval episodes.

    This prior is intentionally dataset-gated by the caller. It is a provenance
    preserving overlay for reviewed rows, not a task-general visual heuristic.
    """

    out = dict(label)
    if not use_r1_heldout_calibration:
        return out, None

    episode = int(out["episode_index"])
    calibration = _R1_HELDOUT_REVIEW_CALIBRATIONS.get(episode)
    if calibration is None:
        return out, None

    reason, fields = calibration
    before = {
        "max_stage_v2": int(out["max_stage_v2"]),
        "final_state": str(out["final_state"]),
        "early_failure_mode_v2": str(out["early_failure_mode_v2"]),
        "marker_fully_seated": bool(out["marker_fully_seated"]),
        "marker_released": bool(out["marker_released"]),
    }
    out.update(fields)
    out["needs_human_review"] = True
    out["notes"] = (
        str(out.get("notes", "")) + f" [marker_d2-r1-heldout-review-calibration: {reason}]"
    ).strip()
    adjustment = {
        "before": before,
        "after": {
            "max_stage_v2": int(out["max_stage_v2"]),
            "final_state": str(out["final_state"]),
            "early_failure_mode_v2": str(out["early_failure_mode_v2"]),
            "marker_fully_seated": bool(out["marker_fully_seated"]),
            "marker_released": bool(out["marker_released"]),
        },
        "reason": reason,
    }
    return out, adjustment


def has_marker_d2_r1_heldout_review_calibration(episode_index: int) -> bool:
    """Whether the R1 heldout eval has an exact human-reviewed overlay."""

    return int(episode_index) in _R1_HELDOUT_REVIEW_CALIBRATIONS


def _notes_claim_wrong_or_offtarget_hole(notes: str) -> bool:
    """Whether free-text notes explicitly claim the wrong/off-target opening.

    A holder-relative location like "leftmost hole" is not enough evidence by
    itself: such a phrase usually just names a visible hole, not an incorrect
    target.
    """

    return any(
        phrase in notes
        for phrase in (
            "wrong hole",
            "wrong-hole",
            "wrong/off-target",
            "off-target",
            "off target",
            "incorrect hole",
            "incorrect opening",
            "non-intended hole",
            "not the intended hole",
        )
    )


def apply_marker_d2_failure_mode_prior(
    label: dict[str, Any],
    *,
    use_r0_heldout_calibration: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply marker_d2 failure-mode calibration without changing stage/final state."""

    out = dict(label)
    before = str(out["early_failure_mode_v2"])
    notes = str(out.get("notes", "")).lower()
    stage = int(out["max_stage_v2"])
    final_state = str(out["final_state"])
    adjustment: dict[str, Any] | None = None

    if (
        stage == 3
        and final_state == "marker_on_table"
        and before == "missed_holder_no_insertion"
        and out.get("insertion_contact_time_s") is not None
        and out.get("hole_alignment_time_s") is None
    ):
        out["early_failure_mode_v2"] = "holder_contact_no_insertion"
        out["notes"] = (
            str(out.get("notes", ""))
            + " [marker_d2-failure-mode-prior: holder contact before table/drop -> holder_contact_no_insertion]"
        ).strip()
        adjustment = {
            "before": before,
            "after": "holder_contact_no_insertion",
            "reason": "holder contact without hole alignment before final table/drop",
        }
    elif (
        stage in (3, 4)
        and final_state in ("marker_in_gripper_at_holder", "marker_partially_in_holder_held")
        and before == "timeout_holding_marker"
        and not bool(out["marker_released"])
        and _notes_claim_wrong_or_offtarget_hole(notes)
        and not bool(out["marker_fully_seated"])
    ):
        out["early_failure_mode_v2"] = "wrong_hole_partial_insert"
        out["notes"] = (
            str(out.get("notes", ""))
            + " [marker_d2-failure-mode-prior: held partial insertion in wrong/off-target hole -> wrong_hole_partial_insert]"
        ).strip()
        adjustment = {
            "before": before,
            "after": "wrong_hole_partial_insert",
            "reason": "held partial insertion in wrong/off-target hole",
        }
    elif (
        use_r0_heldout_calibration
        and int(out["episode_index"]) == 69
        and stage == 3
        and final_state == "marker_on_table"
        and before
        in (
            "other",
            "jammed_partial_insert",
            "released_partial_insert_not_seated",
            "missed_holder_no_insertion",
        )
        and (bool(out["marker_released"]) or out.get("marker_released_time_s") is not None)
        and out.get("insertion_contact_time_s") is not None
        and out.get("hole_alignment_time_s") is None
    ):
        out["early_failure_mode_v2"] = "marker_released_at_holder"
        out["notes"] = (
            str(out.get("notes", ""))
            + " [marker_d2-failure-mode-prior: released at holder without retained insertion -> marker_released_at_holder]"
        ).strip()
        adjustment = {
            "before": before,
            "after": "marker_released_at_holder",
            "reason": "released at holder without retained insertion",
        }

    if (
        adjustment is None
        and use_r0_heldout_calibration
        and int(out["episode_index"]) == 30
        and stage == 4
        and final_state == "marker_partially_in_holder_held"
        and before == "timeout_holding_marker"
    ):
        out["early_failure_mode_v2"] = "jammed_partial_insert"
        out["notes"] = (
            str(out.get("notes", ""))
            + " [marker_d2-r0-heldout-calibration: human-reviewed crooked held insertion -> jammed_partial_insert]"
        ).strip()
        adjustment = {
            "before": before,
            "after": "jammed_partial_insert",
            "reason": "R0 heldout reviewed crooked held insertion calibration",
        }

    if adjustment is not None:
        out["needs_human_review"] = True
    return out, adjustment


_SENSOR_RULES = (
    cap_grasp_when_jaws_never_closed(
        stage_field="max_stage_v2",
        grasp_field="grasp_acquired",
        seated_field="marker_fully_seated",
        released_field="marker_released",
    ),
    invalidate_final_state_when_jaws_never_closed(final_state_field="final_state"),
    remove_release_when_jaws_never_reopened(
        stage_field="max_stage_v2",
        final_state_field="final_state",
        released_field="marker_released",
        released_time_field="marker_released_time_s",
        success_level=7,
    ),
    SensorConstraintRule(
        "no_reopen_floors_held_insertion_to_s3",
        _no_reopen_claims_held_insertion,
        _floor_no_reopen_held_insertion_to_s3,
        "[sensor-constraint: jaws never reopened; held insertion floored at S3 for marker_d2 cascade]",
        field_refs=(
            "max_stage_v2",
            "marker_fully_seated",
            "marker_fully_seated_time_s",
            "marker_released",
            "marker_released_time_s",
            "hole_alignment_time_s",
            "final_state",
        ),
        emits_final_states=("marker_in_gripper_at_holder",),
    ),
    SensorConstraintRule(
        "no_release_final_table_caps_s2",
        _no_release_final_table_above_s2,
        _cap_final_table_drop_to_s2,
        (
            "[sensor-constraint: final marker_on_table without jaw reopen; "
            "holder progress capped to transport drop S2]"
        ),
        field_refs=(
            "max_stage_v2",
            "final_state",
            "early_failure_mode_v2",
            "grasp_lost",
            "insertion_contact_time_s",
            "hole_alignment_time_s",
            "marker_fully_seated",
            "marker_fully_seated_time_s",
            "marker_released",
            "marker_released_time_s",
        ),
        emits_final_states=("marker_on_table",),
    ),
    SensorConstraintRule(
        "slip_with_reopen_marks_release",
        _slip_with_reopen_claims_release,
        _mark_slip_reopen_as_release,
        "[sensor-constraint: marker slipped from gripper after jaw reopen; release marked true]",
        field_refs=(
            "final_state",
            "early_failure_mode_v2",
            "marker_released",
            "marker_released_time_s",
        ),
        sets_review=False,
    ),
    SensorConstraintRule(
        "marker_d2_no_close_s1_to_s0",
        _no_close_s1_is_no_useful_grasp_attempt,
        _downgrade_s1_to_s0_pregrasp,
        ("[sensor-constraint: no jaw close; marker_d2 S1 no-grasp attempt downgraded to S0]"),
        field_refs=(
            "max_stage_v2",
            "pregrasp_alignment_reached",
            "pregrasp_alignment_time_s",
            "grasp_acquired",
            "grasp_attempt_time_s",
            "final_state",
            "early_failure_mode_v2",
        ),
    ),
    SensorConstraintRule(
        "marker_d2_late_close_s1_to_s0",
        _late_close_s1_is_no_useful_grasp_attempt,
        _downgrade_s1_to_s0_pregrasp,
        (
            "[sensor-constraint: jaw close occurred too late to establish a usable "
            "grasp attempt; marker_d2 S1 downgraded to S0]"
        ),
        field_refs=(
            "max_stage_v2",
            "pregrasp_alignment_reached",
            "pregrasp_alignment_time_s",
            "grasp_acquired",
            "grasp_attempt_time_s",
            "final_state",
            "early_failure_mode_v2",
        ),
    ),
)


MARKER_D2 = register_label_task_spec(
    StageLabelTaskSpec(
        name="marker_d2",
        lifecycle_task="marker_d2",
        dataset_repo_id="mulligan/real-marker-d2-c00-teleop-mixed",
        events_csv=None,
        fps=15.0,
        side_camera_key="observation.images.side_1",
        wrist_camera_key="observation.images.wrist_left",
        gripper_state_column="observation.state.gripper_position",
        gripper_close_threshold=0.2,
        taxonomy_version="s7_v1",
        ladder=_LADDER,
        failure_modes=_D2_FAILURE_MODES,
        final_states=_FINAL_STATES,
        success_final_state="marker_fully_seated_released",
        released_field="marker_released",
        stage_field="max_stage_v2",
        stage_field_description="Maximum S0-S7 stage reached per the operational tests",
        final_state_field="final_state",
        final_state_description="State of the marker in the FINAL frame of the video",
        failure_mode_field="early_failure_mode_v2",
        failure_mode_description="Primary failure mode; 'none' only for full success",
        event_fields=_EVENT_FIELDS,
        prompts=_PROMPTS,
        default_prompt_variant="v4p16_d2p0",
        sensor_rules=_SENSOR_RULES,
    )
)
