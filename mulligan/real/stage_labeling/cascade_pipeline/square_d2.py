"""Cascade runner for square_d2 stage-labeling runs."""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from mulligan.real.stage_specs import get_label_task_spec
from mulligan.real.stage_labeling.assets import (
    build_items,
    full_frame_window_montage,
    local_wrist_window_crop,
)
from mulligan.real.stage_labeling.cascade import RefinementNode, run_node
from mulligan.real.stage_labeling.cascade_pipeline.common import (
    CascadeRunOutputs,
    REPO_ROOT,
    SquareD2CascadeConfig,
    configure_gemini,
    consensus_label,
    load_events,
    load_raw,
    raw_samples_per_episode,
)
from mulligan.real.stage_labeling.consensus import build_consensus, summarize
from mulligan.real.stage_labeling.labeler import (
    LabelerConfig,
    client_route,
    make_client,
    write_outputs,
)
from mulligan.real.stage_specs.sensor_constraints import apply_sensor_constraints
from mulligan.real.stage_specs.square_d2 import (
    SQUARE_D2_R1_HELDOUT_DATASET_REPO_ID,
    apply_square_d2_endpoint_prior,
    apply_square_d2_r0_s1_failure_prior,
)

# Few-shot anchor media, rendered on first use from the reviewed episodes below.
_ANCHOR_ROOT = REPO_ROOT / "outputs/real/stage_labeling_anchors/square_d2"
_S1_SUBTYPE_ANCHOR_DIR = _ANCHOR_ROOT / "s1_subtype_anchor_crops"
_TRANSPORT_BOUNDARY_ANCHOR_DIR = _ANCHOR_ROOT / "transport_boundary_anchor_videos"
_CARRY_ENDPOINT_ANCHOR_DIR = _ANCHOR_ROOT / "carry_endpoint_anchor_videos"
_PEG_ARRIVAL_ANCHOR_DIR = _ANCHOR_ROOT / "peg_arrival_anchor_videos"
_TRANSPORT_BOUNDARY_ANCHOR_SOURCE_REPO = SQUARE_D2_R1_HELDOUT_DATASET_REPO_ID

_S1_SUBTYPE_ANCHORS: dict[int, tuple[str, float, str]] = {
    14: (
        "pregrasp_misalignment",
        5.60,
        "Negative anchor: the robot attempts the nut, but the jaws are not in a genuinely "
        "graspable alignment when contact/close happens.",
    ),
    15: (
        "pregrasp_misalignment",
        4.73,
        "Negative anchor: contact is a misaligned poke/bump, not a close from a pose that could "
        "cleanly trap the nut.",
    ),
    17: (
        "pregrasp_misalignment",
        3.80,
        "Negative borderline anchor adjudicated as pregrasp misalignment: do not promote a close "
        "to aligned miss unless the open jaws visibly bracket the nut in a graspable pose.",
    ),
    22: (
        "pregrasp_misalignment",
        3.60,
        "Negative anchor: the nut remains on the table after an off-center/misposed attempt.",
    ),
    9: (
        "pregrasp_misalignment",
        5.67,
        "Negative close-attempt anchor: even with a decisive close, the jaws are not cleanly "
        "bracketing the nut in the human-reviewed square_d2 sense.",
    ),
    23: (
        "missed_grasp_after_alignment",
        5.33,
        "Positive aligned-miss anchor: the jaws reach a plausible grasp pose and the miss is a "
        "small height/closure error, not a gross pregrasp offset.",
    ),
    27: (
        "missed_grasp_after_alignment",
        4.73,
        "Positive aligned-miss anchor: the jaws are barely aligned well enough, but close a bit "
        "too high for the grasp to stick.",
    ),
}

_S1_SUBTYPE_NODE = RefinementNode(
    name="square_d2_s1_failed_grasp_subtype",
    question=(
        "Focused square_d2 S1 failed-grasp subtype classifier. The target episode is already "
        "known to be S1 with final_state=nut_on_table and no usable grasp acquired. Decide ONLY "
        "which failure subtype applies.\n\n"
        "Choose 'missed_grasp_after_alignment' only when, at the decisive attempt, the open jaws "
        "visibly reached a plausible grasp pose around the nut body/handle and the failure was a "
        "small miss, height error, closure error, or slip during the close. This means "
        "pregrasp_alignment_reached=true.\n"
        "Choose 'pregrasp_misalignment' when the gripper went for the nut but never reached a "
        "genuinely graspable alignment: off-center, wrong angle, one jaw on top/beside the nut, "
        "poking/shoving/bumping the nut, or otherwise not bracketing the nut in a pose from which "
        "closing could plausibly capture it. Also choose 'pregrasp_misalignment' if there was no "
        "decisive jaw close/contact attempt. Borderline or ambiguous cases default to "
        "pregrasp_misalignment; do not promote merely because a close happened near the nut.\n\n"
        "Use the labeled reference strips as the calibration standard. For the target, use the "
        "full side/wrist videos for context and the local wrist strip for the decisive "
        "approach/close moment. In reasoning, first state whether the jaws bracket the nut in a "
        "graspable pose at the decisive moment, then give the verdict."
    ),
    evidence=("side_path", "wrist_path", "s1_subtype_crop", "grasp_window_crop"),
    verdict_to_stage={"pregrasp_misalignment": 1, "missed_grasp_after_alignment": 1},
    exemplar_key="s1_subtype_crop",
    samples=5,
)

_TRANSPORT_BOUNDARY_ANCHORS: dict[int, tuple[str, str]] = {
    62: (
        "lost_before_controlled_peg_arrival_off_table",
        "R1 reviewed S2 negative: after lift the nut is already hanging/falling before a "
        "controlled insertion-directed arrival at the peg. Later proximity/brushing does not "
        "earn S3.",
    ),
    94: (
        "lost_before_controlled_peg_arrival_on_table",
        "R1 reviewed S2 negative: the nut is lost during transport and ends on the table; do "
        "not regress this to an S1 grasp miss or promote it to S3 merely because the arm moves "
        "toward the peg region.",
    ),
    10: (
        "controlled_at_peg_no_hole_released",
        "R1 reviewed S3 negative for hole engagement: the nut reaches the peg area under "
        "control and is released near the peg, but the peg tip is not visibly inside the nut "
        "hole.",
    ),
    27: (
        "controlled_at_peg_no_hole_slipped",
        "R1 reviewed S3 negative for hole engagement: the nut is carried to the peg area, then "
        "slips/falls, but there is no positive peg-tip-in-hole moment and no fully seated nut.",
    ),
    37: (
        "definite_hole_engagement",
        "R1 reviewed success control: this shows what positive square-nut hole engagement looks "
        "like. Use it only as the visual standard for definite peg-in-hole evidence; a failed "
        "target still needs its own visible engagement moment.",
    ),
}

_TRANSPORT_BOUNDARY_NODE = RefinementNode(
    name="square_d2_transport_peg_engagement_boundary",
    question=(
        "Focused square_d2 transport / peg-engagement boundary classifier. The target already "
        "has a grasped nut and ends with the nut on the table after a failed transport / peg-area "
        "attempt. Decide ONLY the maximum progress boundary and the release/slip subtype.\n\n"
        "Definitions:\n"
        "- 'lost_before_controlled_peg_arrival_on_table': the nut was grasped/lifted, but it "
        "slipped into an uncontrolled dangling/falling trajectory before a controlled "
        "insertion-directed arrival at the peg; it ends on the table. This is S2.\n"
        "- 'lost_before_controlled_peg_arrival_off_table': same uncontrolled pre-arrival loss, "
        "but the nut drops off the table / out of the workspace. This is S2.\n"
        "- 'controlled_at_peg_no_hole_slipped': the nut is still under usable control when it "
        "reaches the peg area in an insertion-directed pose, but the peg tip never visibly enters "
        "the nut hole before the nut slips/falls. This is S3.\n"
        "- 'controlled_at_peg_no_hole_released': same controlled arrival with no visible "
        "peg-tip-in-hole, but the decisive failure is an intentional jaw opening/release near the "
        "peg. This is S3.\n"
        "- 'definite_hole_engagement': before the table-ending failure, the peg tip is visibly "
        "inside the nut's center hole, or the nut hole is visibly around the peg tip. This is S4 "
        "credit. Mere proximity, camera overlap, hovering beside/over the peg, contact with the "
        "peg exterior, or a release near the peg is NOT enough.\n\n"
        "Use the labeled reference videos as the calibration standard. For the target, inspect "
        "the full side/wrist videos for timing and depth, the release crop for release-vs-slip, "
        "and final crops for on-table versus off-table endpoint. In reasoning, first state "
        "whether the nut was still controlled at peg arrival, then whether the peg tip visibly "
        "entered the hole, then whether the failure was a slip/drop or a jaw release."
    ),
    evidence=(
        "side_path",
        "wrist_path",
        "release_moment_crop",
        "side_final_crop",
        "wrist_final_crop",
    ),
    verdict_to_stage={
        "lost_before_controlled_peg_arrival_on_table": 2,
        "lost_before_controlled_peg_arrival_off_table": 2,
        "controlled_at_peg_no_hole_slipped": 3,
        "controlled_at_peg_no_hole_released": 3,
        "definite_hole_engagement": 4,
    },
    exemplar_key="combo_path",
    samples=5,
)

_CARRY_ENDPOINT_ANCHORS: dict[int, tuple[str, str]] = {
    10: (
        "controlled_release_at_peg_not_seated_on_table",
        "R1 reviewed endpoint correction: the jaws open near the peg, but the nut is not seated "
        "and ends on the table; this is a release failure, not success.",
    ),
    18: (
        "passive_slip_after_controlled_peg_arrival_on_table",
        "R1 reviewed release-vs-slip correction: the gripper reopens late, but the nut has already "
        "slipped out after controlled peg-area arrival, so this is grasp loss rather than a "
        "controlled release.",
    ),
    94: (
        "passive_slip_before_controlled_peg_arrival_on_table",
        "R1 reviewed endpoint correction: the nut is acquired, then passively slips out before a "
        "controlled insertion-directed peg arrival and ends on the table.",
    ),
    143: (
        "dropped_during_transport_on_table",
        "R1 reviewed endpoint correction: the nut is dropped during transport before controlled "
        "peg-top/hole arrival and ends on the table, despite the initial model saying it was held.",
    ),
    89: (
        "still_held_away_from_peg",
        "R1 reviewed S2/S3 boundary: the nut is still held at the end, but only beside or "
        "away from the peg with no insertion-directed peg-top/hole control.",
    ),
    29: (
        "still_held_at_peg_not_seated",
        "R1 reviewed held-timeout control: the nut remains visibly held at the peg region, "
        "but is not fully seated and the jaws never reopen.",
    ),
    22: (
        "still_held_at_peg_fully_seated",
        "R1 reviewed held-timeout seated control: the nut reaches the base/fully seated pose "
        "while still held; no release happens, so this remains a timeout-held endpoint.",
    ),
    33: (
        "passive_slip_before_controlled_peg_arrival_on_table",
        "R1 reviewed endpoint correction: the nut was acquired, then lost before a controlled "
        "peg arrival and ends on the table, despite the initial model saying it was still held.",
    ),
    62: (
        "dropped_during_transport_off_table",
        "R1 reviewed endpoint correction: the nut was acquired, then dropped out of the "
        "workspace/off the table before controlled peg arrival.",
    ),
    14: (
        "passive_slip_after_controlled_peg_arrival_on_table",
        "R1 reviewed endpoint correction: the nut reaches the peg area under usable control, "
        "then slips/falls and ends on the table; do not leave it as still held.",
    ),
}

_CARRY_ENDPOINT_NODE = RefinementNode(
    name="square_d2_endpoint_action_verifier",
    question=(
        "Focused square_d2 endpoint-action verifier. The target already has a grasped nut and "
        "a failed or non-success endpoint. Decide ONLY what physically happened to the nut at "
        "the end: controlled release, passive slip/loss, transport drop, or still-held timeout. "
        "Do not re-score S0/S1 or full successes.\n\n"
        "Use the gripper sensor trace as proprioceptive evidence:\n"
        "- If jaws_reopened_before_episode_end is false, there was NO controlled release. The nut "
        "may still slip out of closed jaws, but never choose a release verdict without a reopen.\n"
        "- If jaws_reopened_before_episode_end is true, the key question is whether the nut was "
        "still visibly between the jaws immediately BEFORE the reopen. If yes, an opening of the "
        "jaws is a controlled release. If the nut had already left the jaws before the reopen, the "
        "later reopen is just an empty-gripper motion and the failure is slip/drop, not release.\n"
        "- The final crops decide where the nut ended; the side/wrist release-window strips, "
        "release crop, and full video decide whether the jaw opening released the nut or happened "
        "after the nut was already gone.\n"
        "- If the final frames are ambiguous and the jaws never reopen, prefer a still-held "
        "verdict. Choose a slip/drop verdict only with positive visual evidence that the nut left "
        "the jaws or ended on/off the table.\n\n"
        "Peg-area boundary for this verifier:\n"
        "- controlled peg arrival means the held nut/hole reaches the sampled peg-top or hole "
        "insertion region in a pose where continuing the same motion could insert the peg through "
        "the nut. Touching the side/exterior of the peg, hovering beside it, or passing near it is "
        "NOT controlled peg arrival.\n\n"
        "Verdicts:\n"
        "- 'still_held_away_from_peg': the nut is still in the gripper at the end, away from the "
        "peg or beside the peg with no insertion-directed peg-top/hole control. This is S2.\n"
        "- 'still_held_at_peg_not_seated': the nut is still in the gripper at the end and reaches "
        "the peg-top/hole region in an insertion-directed pose, but is not fully seated. This is S3.\n"
        "- 'still_held_at_peg_fully_seated': the nut is still in the gripper at the end and is "
        "visibly fully seated at the base, but no release happened. Keep the timeout-held endpoint.\n"
        "- 'passive_slip_before_controlled_peg_arrival_on_table': after acquisition, the nut "
        "passively slips out of the jaws before controlled peg-top/hole arrival and ends on the "
        "table. A later jaw reopen, if any, happens after the nut is already gone. This is S2.\n"
        "- 'passive_slip_after_controlled_peg_arrival_on_table': the nut reaches controlled "
        "peg-top/hole-region arrival, then passively slips/falls from the jaws and ends on the "
        "table. A later empty-gripper reopen is not a release. This is S3.\n"
        "- 'dropped_during_transport_on_table': a more forceful or collision-driven drop during "
        "transport before controlled peg-top/hole arrival; the nut ends on the table. This is S2.\n"
        "- 'dropped_during_transport_off_table': same transport drop, but the nut leaves the "
        "workspace/off the table. This is S2.\n"
        "- 'controlled_release_at_peg_not_seated_on_table': the nut is still visibly in/controlled "
        "by the jaws immediately before a jaw reopen near the peg, the jaws open, and the nut is "
        "not retained/seated, ending on the table. This is S3 and nut_released=true.\n\n"
        "In reasoning, follow this order: (1) was the nut still between the jaws immediately before "
        "any reopen, (2) if not, when did it leave the jaws, (3) did it reach controlled peg-top/"
        "hole-region arrival before leaving the jaws, and (4) where is the nut in the final crops."
    ),
    evidence=(
        "sensor_trace",
        "combo_path",
        "side_release_window_crop",
        "wrist_release_window_crop",
        "release_moment_crop",
        "side_final_crop",
        "wrist_final_crop",
    ),
    verdict_to_stage={
        "still_held_away_from_peg": 2,
        "still_held_at_peg_not_seated": 3,
        "still_held_at_peg_fully_seated": 3,
        "passive_slip_before_controlled_peg_arrival_on_table": 2,
        "passive_slip_after_controlled_peg_arrival_on_table": 3,
        "dropped_during_transport_on_table": 2,
        "dropped_during_transport_off_table": 2,
        "controlled_release_at_peg_not_seated_on_table": 3,
    },
    exemplar_key="combo_path",
    samples=5,
)

_CARRY_ENDPOINT_VERDICT_ALIASES = {
    "lost_before_controlled_peg_arrival_on_table": (
        "passive_slip_before_controlled_peg_arrival_on_table"
    ),
    "lost_before_controlled_peg_arrival_off_table": "dropped_during_transport_off_table",
    "dropped_before_controlled_peg_arrival_on_table": "dropped_during_transport_on_table",
    "lost_after_controlled_peg_arrival_on_table": (
        "passive_slip_after_controlled_peg_arrival_on_table"
    ),
    "released_at_peg_not_seated_on_table": "controlled_release_at_peg_not_seated_on_table",
}

_PEG_ARRIVAL_ANCHORS: dict[int, tuple[str, str]] = {
    89: (
        "no_controlled_peg_arrival",
        "R1 reviewed negative: the nut is still held near/beside the peg, but it only reaches "
        "the side/exterior region and never gets controlled peg-top/hole insertion-region control.",
    ),
    143: (
        "no_controlled_peg_arrival",
        "R1 reviewed negative: the nut is lost during transport before controlled peg-top/hole "
        "arrival, even though the arm later moves near the peg fixture.",
    ),
    10: (
        "controlled_peg_arrival",
        "R1 reviewed positive: the grasped nut reaches the peg-top/hole region under control "
        "before the failed release.",
    ),
    18: (
        "controlled_peg_arrival",
        "R1 reviewed positive: the grasped nut reaches controlled peg-area insertion posture "
        "before it slips/falls.",
    ),
    22: (
        "controlled_peg_arrival",
        "R1 reviewed positive: the nut reaches the peg/base region while still held, so S3 "
        "peg-arrival credit is warranted even though release never happens.",
    ),
    29: (
        "controlled_peg_arrival",
        "R1 reviewed positive held-timeout control: the nut remains held at the peg insertion "
        "region but is not released/seated.",
    ),
}

_PEG_ARRIVAL_NODE = RefinementNode(
    name="square_d2_strict_peg_arrival_boundary",
    question=(
        "Focused square_d2 S2/S3 peg-arrival verifier. The target already has a grasped nut. "
        "Decide ONLY whether the held nut ever reached controlled peg-top/hole insertion-region "
        "control. Do not decide release-vs-slip and do not decide full seating.\n\n"
        "Choose 'controlled_peg_arrival' only when the nut, while still controlled by the gripper, "
        "reaches the sampled peg top or the nut hole is brought into the peg insertion column so "
        "continuing the same motion could plausibly insert the peg through the hole. The nut may "
        "later fall or be released; this question is only about maximum progress before that.\n\n"
        "Choose 'no_controlled_peg_arrival' when the nut is merely near the peg, beside it, brushing "
        "the side/exterior, below/next to the peg, or when the arm moves near the fixture after the "
        "nut has already left the jaws. Side/exterior contact is S2, not S3. Ambiguous evidence "
        "defaults to no_controlled_peg_arrival.\n\n"
        "Use the labeled reference videos as the calibration standard. In reasoning, first say "
        "whether the nut is still controlled when it reaches the peg, then whether the nut/hole "
        "enters the peg-top insertion region rather than only the peg side/exterior."
    ),
    evidence=(
        "sensor_trace",
        "combo_path",
        "side_release_window_crop",
        "wrist_release_window_crop",
        "side_final_crop",
        "wrist_final_crop",
    ),
    verdict_to_stage={"no_controlled_peg_arrival": 2, "controlled_peg_arrival": 3},
    exemplar_key="combo_path",
    samples=5,
)


def _adjust_square_d2_raw_results(
    raw: list[dict[str, Any]],
    labels: dict[int, dict[str, Any]],
    refinements: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach square_d2 metadata while preserving untouched sample disagreement."""
    adjusted = []
    for record in raw:
        episode = int(record["episode_index"])
        out = dict(record)
        if bool(refinements[episode].get("overrode")) or bool(
            refinements[episode].get("label_prior_applied")
            or refinements[episode].get("endpoint_prior_applied")
        ):
            out["backbone_parsed"] = dict(record["parsed"])
            out["parsed"] = dict(labels[episode])
        else:
            out["parsed"] = dict(record["parsed"])
        out["cascade_refinement"] = refinements[episode]
        adjusted.append(out)
    return adjusted


def _s1_subtype_candidate(label: dict[str, Any]) -> bool:
    if int(label["max_stage"]) != 1:
        return False
    if str(label["final_state"]) != "nut_on_table":
        return False
    if bool(label.get("grasp_acquired")):
        return False
    if str(label.get("failure_mode")) not in {
        "aligned_grasp_miss",
        "missed_grasp_after_alignment",
    }:
        return False
    return _s1_subtype_timing_in_calibrated_band(label)


def _s1_subtype_timing_in_calibrated_band(label: dict[str, Any]) -> bool:
    """Guard the S1 subtype node to the reviewed aligned-miss timing boundary."""
    alignment_time = _optional_float(label.get("pregrasp_alignment_time_s"))
    attempt_time = _optional_float(label.get("grasp_attempt_time_s"))
    if alignment_time is None or attempt_time is None:
        return False
    delta = attempt_time - alignment_time
    return 0.7 <= delta <= 1.2


def _raw_s1_aligned_miss_timing_floor(
    raw_samples: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Detect strongly supported S1 aligned misses from raw backbone samples."""
    aligned_sample_count = 0
    in_band_samples: list[dict[str, Any]] = []
    for sample in raw_samples:
        parsed = sample["parsed"]
        if int(parsed["max_stage"]) != 1:
            continue
        if str(parsed["final_state"]) != "nut_on_table":
            continue
        if bool(parsed.get("grasp_acquired")):
            continue
        if str(parsed["failure_mode"]) not in {
            "aligned_grasp_miss",
            "missed_grasp_after_alignment",
        }:
            continue
        aligned_sample_count += 1
        alignment_time = _optional_float(parsed.get("pregrasp_alignment_time_s"))
        attempt_time = _optional_float(parsed.get("grasp_attempt_time_s"))
        if alignment_time is None or attempt_time is None:
            continue
        delta = attempt_time - alignment_time
        if 0.7 <= delta <= 1.2:
            in_band_samples.append(
                {
                    "sample_idx": int(sample["sample_idx"]),
                    "pregrasp_alignment_time_s": alignment_time,
                    "grasp_attempt_time_s": attempt_time,
                    "delta_s": delta,
                    "failure_mode": str(parsed["failure_mode"]),
                }
            )
    if aligned_sample_count < 2 or not in_band_samples:
        return None
    selected = min(in_band_samples, key=lambda row: abs(float(row["delta_s"]) - 0.95))
    return {
        "name": "square_d2_s1_aligned_miss_timing_floor",
        "aligned_sample_count": aligned_sample_count,
        "in_band_samples": in_band_samples,
        "selected_sample": selected,
    }


def _apply_s1_aligned_miss_timing_floor(
    label: dict[str, Any],
    floor: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if int(label["max_stage"]) != 1:
        raise ValueError(f"S1 timing floor received non-S1 label: {label}")
    if str(label["final_state"]) != "nut_on_table":
        raise ValueError(f"S1 timing floor received non-table final_state: {label}")
    if bool(label.get("grasp_acquired")):
        raise ValueError(f"S1 timing floor received acquired-grasp label: {label}")
    selected = floor["selected_sample"]
    before = {
        "failure_mode": label.get("failure_mode"),
        "pregrasp_alignment_reached": label.get("pregrasp_alignment_reached"),
        "pregrasp_alignment_time_s": label.get("pregrasp_alignment_time_s"),
    }
    out = dict(label)
    out["failure_mode"] = "missed_grasp_after_alignment"
    out["pregrasp_alignment_reached"] = True
    out["pregrasp_alignment_time_s"] = selected["pregrasp_alignment_time_s"]
    if out.get("grasp_attempt_time_s") is None:
        out["grasp_attempt_time_s"] = selected["grasp_attempt_time_s"]
    out["notes"] = (
        str(out.get("notes", ""))
        + " [square-d2-s1 timing floor: >=2 raw samples called aligned/missed grasp "
        f"and sample {selected['sample_idx']} had close-after-alignment delta "
        f"{selected['delta_s']:.2f}s]"
    ).strip()
    after = {
        "failure_mode": out.get("failure_mode"),
        "pregrasp_alignment_reached": out.get("pregrasp_alignment_reached"),
        "pregrasp_alignment_time_s": out.get("pregrasp_alignment_time_s"),
    }
    if before == after:
        return label, None
    return out, {"name": floor["name"], "before": before, "after": after, "evidence": floor}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)


def _s1_subtype_center_time(label: dict[str, Any], item: dict[str, Any]) -> float | None:
    for field in ("grasp_attempt_time_s", "pregrasp_alignment_time_s"):
        value = _optional_float(label.get(field))
        if value is not None:
            return value
    sensor_trace = item.get("sensor_trace") or {}
    return _optional_float(sensor_trace.get("jaw_close_time_s"))


def _s1_subtype_crop_path(item: dict[str, Any], center_time: float) -> Path:
    episode = int(item["episode_index"])
    return (
        Path(item["wrist_path"]).parent / f"episode_{episode:03d}_s1_subtype_t{center_time:.2f}.png"
    )


def _add_s1_subtype_crop(item: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    center_time = _s1_subtype_center_time(label, item)
    if center_time is None:
        return dict(item)
    out = _s1_subtype_crop_path(item, center_time)
    local_wrist_window_crop(
        Path(item["wrist_path"]),
        center_time,
        float(item["duration_s"]),
        out,
    )
    enriched = dict(item)
    enriched["s1_subtype_crop"] = str(out)
    enriched["s1_subtype_center_time_s"] = round(center_time, 2)
    return enriched


def _add_endpoint_action_release_windows(item: dict[str, Any]) -> dict[str, Any]:
    reopen_time = _optional_float((item.get("sensor_trace") or {}).get("jaw_reopen_time_s"))
    if reopen_time is None:
        return dict(item)
    episode = int(item["episode_index"])
    duration = float(item["duration_s"])
    asset_dir = Path(item["combo_path"]).parent
    side_out = asset_dir / f"episode_{episode:03d}_side_release_window_t{reopen_time:.2f}.png"
    wrist_out = asset_dir / f"episode_{episode:03d}_wrist_release_window_t{reopen_time:.2f}.png"
    full_frame_window_montage(Path(item["side_path"]), reopen_time, duration, side_out)
    full_frame_window_montage(Path(item["wrist_path"]), reopen_time, duration, wrist_out)
    enriched = dict(item)
    enriched["side_release_window_crop"] = str(side_out)
    enriched["wrist_release_window_crop"] = str(wrist_out)
    enriched["release_window_center_time_s"] = round(reopen_time, 2)
    return enriched


# Known issue kept for paper fidelity: the anchors are cut from the dataset being labeled, unlike
# the other anchor banks; see docs/reproduce.md, "Known issues".
def _ensure_s1_subtype_anchor_crops(
    item_by_episode: dict[int, dict[str, Any]],
) -> tuple[tuple[int, str, str, str], ...]:
    _S1_SUBTYPE_ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    few_shot: list[tuple[int, str, str, str]] = []
    for episode, (verdict, center_time, why) in _S1_SUBTYPE_ANCHORS.items():
        item = item_by_episode[episode]
        out = _S1_SUBTYPE_ANCHOR_DIR / f"episode_{episode:03d}_t{center_time:.2f}.png"
        local_wrist_window_crop(
            Path(item["wrist_path"]),
            center_time,
            float(item["duration_s"]),
            out,
        )
        few_shot.append((episode, verdict, why, str(out)))
    return tuple(few_shot)


def _s1_subtype_node_with_anchors(
    item_by_episode: dict[int, dict[str, Any]],
) -> RefinementNode:
    return dataclasses.replace(
        _S1_SUBTYPE_NODE,
        few_shot=_ensure_s1_subtype_anchor_crops(item_by_episode),
    )


def _ensure_transport_boundary_anchor_videos(
    spec: Any,
    item_by_episode: dict[int, dict[str, Any]],
) -> tuple[tuple[int, str, str, str], ...]:
    _TRANSPORT_BOUNDARY_ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    few_shot: list[tuple[int, str, str, str]] = []
    for episode, (verdict, why) in _TRANSPORT_BOUNDARY_ANCHORS.items():
        out = _TRANSPORT_BOUNDARY_ANCHOR_DIR / f"episode_{episode:03d}_{verdict}.mp4"
        if not out.exists():
            if spec.dataset_repo_id != _TRANSPORT_BOUNDARY_ANCHOR_SOURCE_REPO:
                raise FileNotFoundError(
                    f"{out} is missing. Regenerate square_d2 transport-boundary anchors from "
                    f"{_TRANSPORT_BOUNDARY_ANCHOR_SOURCE_REPO} before applying this node to "
                    f"{spec.dataset_repo_id}."
                )
            item = item_by_episode.get(episode)
            if item is None:
                raise FileNotFoundError(
                    f"missing built item for transport-boundary anchor episode {episode}; "
                    f"cannot create {out}"
                )
            shutil.copy2(item["combo_path"], out)
        few_shot.append((episode, verdict, why, str(out)))
    return tuple(few_shot)


def _transport_boundary_node_with_anchors(
    spec: Any,
    item_by_episode: dict[int, dict[str, Any]],
) -> RefinementNode:
    return dataclasses.replace(
        _TRANSPORT_BOUNDARY_NODE,
        few_shot=_ensure_transport_boundary_anchor_videos(spec, item_by_episode),
    )


def _ensure_carry_endpoint_anchor_videos(
    spec: Any,
    item_by_episode: dict[int, dict[str, Any]],
) -> tuple[tuple[int, str, str, str], ...]:
    _CARRY_ENDPOINT_ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    few_shot: list[tuple[int, str, str, str]] = []
    for episode, (verdict, why) in _CARRY_ENDPOINT_ANCHORS.items():
        out = _CARRY_ENDPOINT_ANCHOR_DIR / f"episode_{episode:03d}_{verdict}.mp4"
        if not out.exists():
            if spec.dataset_repo_id != _TRANSPORT_BOUNDARY_ANCHOR_SOURCE_REPO:
                raise FileNotFoundError(
                    f"{out} is missing. Regenerate square_d2 carry-endpoint anchors from "
                    f"{_TRANSPORT_BOUNDARY_ANCHOR_SOURCE_REPO} before applying this node to "
                    f"{spec.dataset_repo_id}."
                )
            item = item_by_episode.get(episode)
            if item is None:
                raise FileNotFoundError(
                    f"missing built item for carry-endpoint anchor episode {episode}; "
                    f"cannot create {out}"
                )
            shutil.copy2(item["combo_path"], out)
        few_shot.append((episode, verdict, why, str(out)))
    return tuple(few_shot)


def _carry_endpoint_node_with_anchors(
    spec: Any,
    item_by_episode: dict[int, dict[str, Any]],
) -> RefinementNode:
    return dataclasses.replace(
        _CARRY_ENDPOINT_NODE,
        few_shot=_ensure_carry_endpoint_anchor_videos(spec, item_by_episode),
    )


def _ensure_peg_arrival_anchor_videos(
    spec: Any,
    item_by_episode: dict[int, dict[str, Any]],
) -> tuple[tuple[int, str, str, str], ...]:
    _PEG_ARRIVAL_ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    few_shot: list[tuple[int, str, str, str]] = []
    for episode, (verdict, why) in _PEG_ARRIVAL_ANCHORS.items():
        out = _PEG_ARRIVAL_ANCHOR_DIR / f"episode_{episode:03d}_{verdict}.mp4"
        if not out.exists():
            if spec.dataset_repo_id != _TRANSPORT_BOUNDARY_ANCHOR_SOURCE_REPO:
                raise FileNotFoundError(
                    f"{out} is missing. Regenerate square_d2 peg-arrival anchors from "
                    f"{_TRANSPORT_BOUNDARY_ANCHOR_SOURCE_REPO} before applying this node to "
                    f"{spec.dataset_repo_id}."
                )
            item = item_by_episode.get(episode)
            if item is None:
                raise FileNotFoundError(
                    f"missing built item for peg-arrival anchor episode {episode}; "
                    f"cannot create {out}"
                )
            shutil.copy2(item["combo_path"], out)
        few_shot.append((episode, verdict, why, str(out)))
    return tuple(few_shot)


def _peg_arrival_node_with_anchors(
    spec: Any,
    item_by_episode: dict[int, dict[str, Any]],
) -> RefinementNode:
    return dataclasses.replace(
        _PEG_ARRIVAL_NODE,
        few_shot=_ensure_peg_arrival_anchor_videos(spec, item_by_episode),
    )


def _without_target_episode_few_shot(
    node: RefinementNode,
    episode: int,
    *,
    max_few_shot: int | None = None,
) -> RefinementNode:
    """Omit the target episode from its own few-shot bank.

    Reviewed episodes can be useful calibration anchors for future episodes, but
    showing an episode's own video as a labeled reference while scoring that same
    episode is not a meaningful no-overlay audit.
    """
    few_shot = tuple(entry for entry in node.few_shot if int(entry[0]) != episode)
    if max_few_shot is not None:
        if max_few_shot < 0:
            raise ValueError(f"max_few_shot must be nonnegative, got {max_few_shot}")
        few_shot = few_shot[:max_few_shot]
    if len(few_shot) == len(node.few_shot):
        return node
    return dataclasses.replace(node, few_shot=few_shot)


def _apply_s1_subtype_node_result(
    label: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], bool, float]:
    votes = list(result["votes"])
    if not votes:
        raise ValueError(f"S1 subtype node returned no votes: {result}")
    win_frac = max(Counter(votes).values()) / len(votes)
    verdict = str(result["verdict"])
    if verdict not in _S1_SUBTYPE_NODE.verdict_to_stage:
        raise ValueError(f"unknown square_d2 S1 subtype verdict {verdict!r}")

    out = dict(label)
    if verdict == "missed_grasp_after_alignment":
        out["failure_mode"] = "missed_grasp_after_alignment"
        out["pregrasp_alignment_reached"] = True
        if out.get("pregrasp_alignment_time_s") is None:
            out["pregrasp_alignment_time_s"] = out.get("grasp_attempt_time_s")
    else:
        out["failure_mode"] = "pregrasp_misalignment"
        out["pregrasp_alignment_reached"] = False
        out["pregrasp_alignment_time_s"] = None
    out["needs_human_review"] = bool(out.get("needs_human_review")) or win_frac < 0.8
    out["notes"] = (
        str(out.get("notes", "")) + f" [square-d2-s1-subtype verdict={verdict} votes={votes}]"
    ).strip()
    return out, out != label, win_frac


def _transport_boundary_candidate(label: dict[str, Any]) -> bool:
    stage = int(label["max_stage"])
    if stage not in {2, 3, 4}:
        return False
    if str(label["final_state"]) != "nut_on_table":
        return False
    if not bool(label.get("grasp_acquired")):
        return False
    return str(label.get("failure_mode")) in {
        "nut_slipped_from_gripper",
        "dropped_nut_during_transport",
        "retreat_from_peg_after_contact",
        "released_partial_not_seated",
    }


def _apply_transport_boundary_node_result(
    label: dict[str, Any],
    result: dict[str, Any],
    *,
    override_min_frac: float,
    s4_override_min_frac: float,
) -> tuple[dict[str, Any], bool, float]:
    votes = list(result["votes"])
    if not votes:
        raise ValueError(f"transport-boundary node returned no votes: {result}")
    win_frac = max(Counter(votes).values()) / len(votes)
    verdict = str(result["verdict"])
    if verdict not in _TRANSPORT_BOUNDARY_NODE.verdict_to_stage:
        raise ValueError(f"unknown square_d2 transport-boundary verdict {verdict!r}")

    stage = int(_TRANSPORT_BOUNDARY_NODE.verdict_to_stage[verdict])
    required_frac = s4_override_min_frac if stage >= 4 else override_min_frac
    if win_frac < required_frac:
        return dict(label), False, win_frac

    out = dict(label)
    out["max_stage"] = stage
    out["grasp_acquired"] = True
    out["nut_fully_seated"] = False
    out["nut_fully_seated_time_s"] = None
    out["needs_human_review"] = bool(out.get("needs_human_review")) or win_frac < 0.8

    if verdict == "lost_before_controlled_peg_arrival_on_table":
        out["final_state"] = "nut_on_table"
        out["failure_mode"] = "nut_slipped_from_gripper"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["peg_contact_time_s"] = None
        out["peg_alignment_time_s"] = None
        out["nut_released_time_s"] = None
    elif verdict == "lost_before_controlled_peg_arrival_off_table":
        out["final_state"] = "nut_dropped_off_table"
        out["failure_mode"] = "dropped_nut_during_transport"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["peg_contact_time_s"] = None
        out["peg_alignment_time_s"] = None
        out["nut_released_time_s"] = None
    elif verdict == "controlled_at_peg_no_hole_slipped":
        out["final_state"] = "nut_on_table"
        out["failure_mode"] = "nut_slipped_from_gripper"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["peg_alignment_time_s"] = None
        out["nut_released_time_s"] = None
    elif verdict == "controlled_at_peg_no_hole_released":
        out["final_state"] = "nut_on_table"
        out["failure_mode"] = "released_partial_not_seated"
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["nut_released"] = True
        out["peg_alignment_time_s"] = None
    elif verdict == "definite_hole_engagement":
        out["final_state"] = "nut_on_table"
        out["peg_alignment_time_s"] = out.get("peg_alignment_time_s") or out.get(
            "peg_contact_time_s"
        )
        if str(out.get("failure_mode")) not in {
            "nut_slipped_from_gripper",
            "retreat_from_peg_after_contact",
            "released_partial_not_seated",
        }:
            out["failure_mode"] = "retreat_from_peg_after_contact"
    else:
        raise ValueError(f"unknown square_d2 transport-boundary verdict {verdict!r}")

    out["notes"] = (
        str(out.get("notes", ""))
        + f" [square-d2-transport-boundary verdict={verdict} votes={votes}]"
    ).strip()
    return out, out != label, win_frac


def _carry_endpoint_candidate(label: dict[str, Any], event_row: dict[str, Any]) -> bool:
    if not bool(label.get("grasp_acquired")):
        return False
    stage = int(label["max_stage"])
    final_state = str(label["final_state"])
    failure_mode = str(label.get("failure_mode"))
    outcome = str(event_row["original_outcome"]).strip().lower()
    if stage in {2, 3} and final_state in {
        "nut_in_gripper_away_from_peg",
        "nut_in_gripper_at_peg",
    }:
        return failure_mode in {"timeout_holding_nut", "dropped_nut_during_transport"}
    # A blank outcome (teleop / collection datasets) carries no endpoint prior.
    if outcome not in {"", "nan", "none", "success"} and (
        final_state
        in {
            "nut_fully_seated_released",
            "nut_partially_on_peg_released",
            "nut_fully_seated_held",
            "nut_partially_on_peg_held",
        }
        or failure_mode == "none"
        or (stage >= 4 and bool(label.get("nut_released")))
    ):
        return True
    return False


def _clear_insertion_success_fields(out: dict[str, Any]) -> None:
    out["nut_fully_seated"] = False
    out["nut_fully_seated_time_s"] = None
    out["peg_alignment_time_s"] = None


def _carry_endpoint_verdict_family(verdict: str) -> str:
    verdict = _canonical_carry_endpoint_verdict(verdict)
    if verdict.startswith("still_held_"):
        return "held"
    if verdict == "controlled_release_at_peg_not_seated_on_table":
        return "released"
    if "slip_" in verdict or verdict.startswith("dropped_"):
        return "lost"
    raise ValueError(f"unknown square_d2 carry-endpoint verdict {verdict!r}")


def _canonical_carry_endpoint_verdict(verdict: str) -> str:
    return _CARRY_ENDPOINT_VERDICT_ALIASES.get(verdict, verdict)


def _carry_endpoint_label_family(label: dict[str, Any]) -> str:
    final_state = str(label["final_state"])
    if final_state in {"nut_in_gripper_away_from_peg", "nut_in_gripper_at_peg"}:
        return "held"
    if bool(label.get("nut_released")) or final_state.endswith("_released"):
        return "released"
    if bool(label.get("grasp_lost")) or final_state in {
        "nut_on_table",
        "nut_dropped_off_table",
    }:
        return "lost"
    return "other"


def _event_jaws_reopened(event_row: dict[str, Any]) -> bool:
    value = event_row["gripper_reopened_at_end"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"cannot parse gripper_reopened_at_end={value!r}")


def _relabel_release_verdict_from_held_label(label: dict[str, Any]) -> str:
    if int(label["max_stage"]) >= 3 or str(label["final_state"]) == "nut_in_gripper_at_peg":
        return "passive_slip_after_controlled_peg_arrival_on_table"
    return "passive_slip_before_controlled_peg_arrival_on_table"


def _apply_carry_endpoint_node_result(
    label: dict[str, Any],
    result: dict[str, Any],
    *,
    event_row: dict[str, Any],
    override_min_frac: float,
) -> tuple[dict[str, Any], bool, float]:
    votes = [_canonical_carry_endpoint_verdict(str(vote)) for vote in result["votes"]]
    if not votes:
        raise ValueError(f"carry-endpoint node returned no votes: {result}")
    win_frac = max(Counter(votes).values()) / len(votes)
    verdict = _canonical_carry_endpoint_verdict(str(result["verdict"]))
    if verdict not in _CARRY_ENDPOINT_NODE.verdict_to_stage:
        raise ValueError(f"unknown square_d2 carry-endpoint verdict {verdict!r}")
    family_votes = [_carry_endpoint_verdict_family(str(vote)) for vote in votes]
    family, family_count = Counter(family_votes).most_common(1)[0]
    family_win_frac = family_count / len(votes)
    current_family = _carry_endpoint_label_family(label)
    effective_win_frac = win_frac
    required_frac = override_min_frac
    jaws_reopened = _event_jaws_reopened(event_row)
    if not jaws_reopened and family == "released":
        return dict(label), False, win_frac
    if not jaws_reopened and current_family == "held" and family == "lost":
        if not verdict.startswith("passive_slip_"):
            return dict(label), False, win_frac
    if current_family == "released" and family == "lost":
        verdict = "controlled_release_at_peg_not_seated_on_table"
        votes = [
            "controlled_release_at_peg_not_seated_on_table"
            if _carry_endpoint_verdict_family(vote) == "lost"
            else vote
            for vote in votes
        ]
        win_frac = max(Counter(votes).values()) / len(votes)
        family = "released"
        family_votes = [_carry_endpoint_verdict_family(str(vote)) for vote in votes]
        family_count = Counter(family_votes)[family]
        family_win_frac = family_count / len(votes)
    if jaws_reopened and current_family == "held" and family == "released":
        verdict = _relabel_release_verdict_from_held_label(label)
        votes = [
            _relabel_release_verdict_from_held_label(label)
            if _carry_endpoint_verdict_family(vote) == "released"
            else vote
            for vote in votes
        ]
        win_frac = max(Counter(votes).values()) / len(votes)
        family = "lost"
        family_votes = [_carry_endpoint_verdict_family(str(vote)) for vote in votes]
        family_count = Counter(family_votes)[family]
        family_win_frac = family_count / len(votes)
    if jaws_reopened and current_family == "held" and family in {"lost", "released"}:
        # A reopen makes the current held endpoint suspicious unless the verifier
        # sees a regrasp/held endpoint strongly enough to win the family vote.
        required_frac = min(required_frac, 0.6)
        effective_win_frac = max(effective_win_frac, family_win_frac)
    if family != current_family and current_family not in family_votes:
        effective_win_frac = max(effective_win_frac, family_win_frac)
    if effective_win_frac < required_frac:
        return dict(label), False, win_frac

    out = dict(label)
    out["max_stage"] = int(_CARRY_ENDPOINT_NODE.verdict_to_stage[verdict])
    out["grasp_acquired"] = True
    out["needs_human_review"] = bool(out.get("needs_human_review")) or win_frac < 0.8

    if verdict == "still_held_away_from_peg":
        out["final_state"] = "nut_in_gripper_away_from_peg"
        out["failure_mode"] = "timeout_holding_nut"
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["nut_released"] = False
        out["nut_released_time_s"] = None
        out["peg_contact_time_s"] = None
        _clear_insertion_success_fields(out)
    elif verdict == "still_held_at_peg_not_seated":
        out["final_state"] = "nut_in_gripper_at_peg"
        out["failure_mode"] = "timeout_holding_nut"
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["nut_released"] = False
        out["nut_released_time_s"] = None
        _clear_insertion_success_fields(out)
    elif verdict == "still_held_at_peg_fully_seated":
        out["final_state"] = "nut_in_gripper_at_peg"
        out["failure_mode"] = "timeout_holding_nut"
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["nut_released"] = False
        out["nut_released_time_s"] = None
        # The square static-held convention floors held seated-looking episodes to S3,
        # and reviewed R0 examples show that a focused endpoint node over-calls this
        # boolean. Keep the endpoint family but do not assert full seating from this node.
        _clear_insertion_success_fields(out)
    elif verdict == "passive_slip_before_controlled_peg_arrival_on_table":
        out["final_state"] = "nut_on_table"
        out["failure_mode"] = "nut_slipped_from_gripper"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["nut_released_time_s"] = None
        out["peg_contact_time_s"] = None
        _clear_insertion_success_fields(out)
    elif verdict == "dropped_during_transport_off_table":
        out["final_state"] = "nut_dropped_off_table"
        out["failure_mode"] = "dropped_nut_during_transport"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["nut_released_time_s"] = None
        out["peg_contact_time_s"] = None
        _clear_insertion_success_fields(out)
    elif verdict == "dropped_during_transport_on_table":
        out["final_state"] = "nut_on_table"
        out["failure_mode"] = "dropped_nut_during_transport"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["nut_released_time_s"] = None
        out["peg_contact_time_s"] = None
        _clear_insertion_success_fields(out)
    elif verdict == "passive_slip_after_controlled_peg_arrival_on_table":
        out["final_state"] = "nut_on_table"
        out["failure_mode"] = "nut_slipped_from_gripper"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["nut_released_time_s"] = None
        _clear_insertion_success_fields(out)
    elif verdict == "controlled_release_at_peg_not_seated_on_table":
        out["final_state"] = "nut_on_table"
        out["failure_mode"] = "released_partial_not_seated"
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["nut_released"] = True
        _clear_insertion_success_fields(out)
    else:
        raise ValueError(f"unknown square_d2 carry-endpoint verdict {verdict!r}")

    out["notes"] = (
        str(out.get("notes", "")) + f" [square-d2-carry-endpoint verdict={verdict} votes={votes}]"
    ).strip()
    return out, out != label, effective_win_frac


# A blank original_outcome is safe here: only an explicit "timeout" makes a candidate.
def _peg_arrival_candidate(label: dict[str, Any], event_row: dict[str, Any]) -> bool:
    if not bool(label.get("grasp_acquired")):
        return False
    if int(label["max_stage"]) != 3:
        return False
    if bool(label.get("nut_fully_seated")) or bool(label.get("nut_released")):
        return False
    outcome = str(event_row["original_outcome"]).strip().lower()
    if outcome == "timeout" and str(label["final_state"]) == "nut_in_gripper_at_peg":
        return False
    return str(label["final_state"]) in {"nut_in_gripper_at_peg", "nut_on_table"}


def _apply_peg_arrival_node_result(
    label: dict[str, Any],
    result: dict[str, Any],
    *,
    override_min_frac: float,
) -> tuple[dict[str, Any], bool, float]:
    votes = list(result["votes"])
    if not votes:
        raise ValueError(f"peg-arrival node returned no votes: {result}")
    win_frac = max(Counter(votes).values()) / len(votes)
    verdict = str(result["verdict"])
    if verdict not in _PEG_ARRIVAL_NODE.verdict_to_stage:
        raise ValueError(f"unknown square_d2 peg-arrival verdict {verdict!r}")
    if verdict == "controlled_peg_arrival" or win_frac < override_min_frac:
        return dict(label), False, win_frac

    out = dict(label)
    out["max_stage"] = 2
    out["peg_contact_time_s"] = None
    out["peg_alignment_time_s"] = None
    out["nut_fully_seated"] = False
    out["nut_fully_seated_time_s"] = None
    if str(out["final_state"]) == "nut_in_gripper_at_peg":
        out["final_state"] = "nut_in_gripper_away_from_peg"
        out["failure_mode"] = "timeout_holding_nut"
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["nut_released"] = False
        out["nut_released_time_s"] = None
    elif str(out["final_state"]) == "nut_on_table":
        out["failure_mode"] = "dropped_nut_during_transport"
        out["grasp_lost"] = True
        out["nut_released"] = False
        out["nut_released_time_s"] = None
    else:
        raise ValueError(f"unexpected peg-arrival candidate final_state {out['final_state']!r}")
    out["needs_human_review"] = bool(out.get("needs_human_review")) or win_frac < 0.8
    out["notes"] = (
        str(out.get("notes", "")) + f" [square-d2-peg-arrival verdict={verdict} votes={votes}]"
    ).strip()
    return out, out != label, win_frac


def apply_square_d2_cascade(config: SquareD2CascadeConfig) -> CascadeRunOutputs:
    """Apply the square_d2-specific post-backbone pipeline."""
    spec = dataclasses.replace(
        get_label_task_spec("square_d2"),
        dataset_repo_id=config.dataset_repo_id,
        events_csv=config.events_csv,
    )
    input_run_dir = config.runs_dir / config.input_run_name
    raw = load_raw(input_run_dir)
    if config.included_episodes:
        requested = set(config.included_episodes)
        raw = [record for record in raw if int(record["episode_index"]) in requested]
        present = {int(record["episode_index"]) for record in raw}
        missing = sorted(requested - present)
        if missing:
            raise ValueError(
                f"requested square_d2 episodes are absent from {input_run_dir}: {missing}"
            )
        if not raw:
            raise ValueError(f"episode filter for {input_run_dir} produced no raw records")
    raw_by_episode: dict[int, list[dict[str, Any]]] = {}
    for record in raw:
        raw_by_episode.setdefault(int(record["episode_index"]), []).append(record)
    input_samples_per_episode = raw_samples_per_episode(raw)
    episodes = sorted({int(r["episode_index"]) for r in raw})
    event_rows = load_events(config.events_csv, set(episodes))
    anchor_episodes = (
        set(_S1_SUBTYPE_ANCHORS)
        | set(_TRANSPORT_BOUNDARY_ANCHORS)
        | set(_CARRY_ENDPOINT_ANCHORS)
        | set(_PEG_ARRIVAL_ANCHORS)
    )
    build_episodes = set(episodes) | anchor_episodes
    built_items = {
        int(item["episode_index"]): item
        for item in build_items(spec, config.build_dir, build_episodes)
    }
    items = {episode: built_items[episode] for episode in episodes}
    s1_subtype_node = _s1_subtype_node_with_anchors(built_items)
    transport_boundary_node = _transport_boundary_node_with_anchors(spec, built_items)
    carry_endpoint_node = _carry_endpoint_node_with_anchors(spec, built_items)
    peg_arrival_node = _peg_arrival_node_with_anchors(spec, built_items)

    consensus = build_consensus(raw, spec)
    labels: dict[int, dict[str, Any]] = {}
    refinements: dict[int, dict[str, Any]] = {}
    stage_rows: list[dict[str, Any]] = []
    labeler_config = LabelerConfig(
        run_name=config.output_run_name,
        output_dir=config.runs_dir,
        build_dir=config.build_dir,
        model=config.model,
        prompt_variant=spec.default_prompt_variant,
        samples=input_samples_per_episode,
        media_resolution=config.media_resolution,
        workers=config.workers,
    )
    client = None
    cascade_gemini_route: dict[str, Any] | None = None
    for row in consensus.to_dict(orient="records"):
        episode = int(row["episode_index"])
        label = apply_sensor_constraints(
            spec.sensor_rules, items[episode], consensus_label(row, spec)
        )
        backbone_stage = int(row[spec.stage_field])
        s1_node_result = None
        s1_node_win_frac = None
        s1_node_overrode = False
        node_item = None
        transport_node_result = None
        transport_node_win_frac = None
        transport_node_overrode = False
        carry_node_result = None
        carry_node_win_frac = None
        carry_node_overrode = False
        peg_arrival_node_result = None
        peg_arrival_node_win_frac = None
        peg_arrival_node_overrode = False
        endpoint_prior_adjustment = None
        s1_timing_floor_adjustment = None
        source = "backbone"
        if _s1_subtype_candidate(label):
            timing_floor = _raw_s1_aligned_miss_timing_floor(raw_by_episode[episode])
            if timing_floor is not None:
                label, s1_timing_floor_adjustment = _apply_s1_aligned_miss_timing_floor(
                    label, timing_floor
                )
                source = "s1_timing_floor"
                print(
                    f"episode {episode:03d}: square_d2 S1 timing floor {s1_timing_floor_adjustment}"
                )
            else:
                node_item = _add_s1_subtype_crop(items[episode], label)
                if client is None:
                    configure_gemini()
                    client = make_client()
                cascade_gemini_route = client_route(client)
                s1_node_result = run_node(
                    spec,
                    labeler_config,
                    client,
                    _without_target_episode_few_shot(s1_subtype_node, episode),
                    node_item,
                )
                label, s1_node_overrode, s1_node_win_frac = _apply_s1_subtype_node_result(
                    label, s1_node_result
                )
                source = "node_s1_subtype"
                print(
                    f"episode {episode:03d}: square_d2 S1 subtype {s1_node_result} "
                    f"overrode={s1_node_overrode}"
                )
        if _transport_boundary_candidate(label):
            if client is None:
                configure_gemini()
                client = make_client()
            cascade_gemini_route = client_route(client)
            transport_node_result = run_node(
                spec,
                labeler_config,
                client,
                _without_target_episode_few_shot(transport_boundary_node, episode),
                items[episode],
            )
            (
                label,
                transport_node_overrode,
                transport_node_win_frac,
            ) = _apply_transport_boundary_node_result(
                label,
                transport_node_result,
                override_min_frac=config.transport_override_min_frac,
                s4_override_min_frac=config.transport_s4_override_min_frac,
            )
            source = "node_transport_boundary"
            print(
                f"episode {episode:03d}: square_d2 transport boundary "
                f"{transport_node_result} overrode={transport_node_overrode}"
            )
        label, endpoint_prior_adjustment = apply_square_d2_endpoint_prior(
            label, event_rows[episode]
        )
        if endpoint_prior_adjustment is not None and source == "backbone":
            source = "endpoint_prior"
        if _carry_endpoint_candidate(label, event_rows[episode]):
            endpoint_item = _add_endpoint_action_release_windows(items[episode])
            if client is None:
                configure_gemini()
                client = make_client()
            cascade_gemini_route = client_route(client)
            carry_node_result = run_node(
                spec,
                labeler_config,
                client,
                _without_target_episode_few_shot(
                    carry_endpoint_node,
                    episode,
                    max_few_shot=9,
                ),
                endpoint_item,
            )
            label, carry_node_overrode, carry_node_win_frac = _apply_carry_endpoint_node_result(
                label,
                carry_node_result,
                event_row=event_rows[episode],
                override_min_frac=config.carry_endpoint_override_min_frac,
            )
            source = "node_carry_endpoint"
            print(
                f"episode {episode:03d}: square_d2 carry endpoint "
                f"{carry_node_result} overrode={carry_node_overrode}"
            )
        if _peg_arrival_candidate(label, event_rows[episode]):
            peg_item = _add_endpoint_action_release_windows(items[episode])
            if client is None:
                configure_gemini()
                client = make_client()
            cascade_gemini_route = client_route(client)
            peg_arrival_node_result = run_node(
                spec,
                labeler_config,
                client,
                _without_target_episode_few_shot(peg_arrival_node, episode),
                peg_item,
            )
            (
                label,
                peg_arrival_node_overrode,
                peg_arrival_node_win_frac,
            ) = _apply_peg_arrival_node_result(
                label,
                peg_arrival_node_result,
                override_min_frac=config.peg_arrival_override_min_frac,
            )
            source = "node_peg_arrival"
            print(
                f"episode {episode:03d}: square_d2 peg arrival "
                f"{peg_arrival_node_result} overrode={peg_arrival_node_overrode}"
            )
        label, label_prior_adjustment = apply_square_d2_r0_s1_failure_prior(label)
        if label_prior_adjustment is not None and source == "backbone":
            source = "label_prior"
        labels[episode] = label
        if label_prior_adjustment is not None:
            print(
                f"episode {episode:03d}: square_d2 prior "
                f"{label_prior_adjustment['before']} -> {label_prior_adjustment['after']}"
            )
        if endpoint_prior_adjustment is not None:
            print(
                f"episode {episode:03d}: square_d2 endpoint prior "
                f"outcome={endpoint_prior_adjustment['original_outcome']} "
                f"{endpoint_prior_adjustment['before']} -> "
                f"{endpoint_prior_adjustment['after']}"
            )
        refinements[episode] = {
            "refined_by": (
                "peg_arrival"
                if peg_arrival_node_result is not None
                else (
                    "carry_endpoint"
                    if carry_node_result is not None
                    else (
                        "transport_boundary"
                        if transport_node_result is not None
                        else (
                            "S1_timing_floor"
                            if s1_timing_floor_adjustment is not None
                            else ("S1_subtype" if s1_node_result is not None else None)
                        )
                    )
                )
            ),
            "overrode": (
                s1_node_overrode
                or transport_node_overrode
                or carry_node_overrode
                or peg_arrival_node_overrode
                or s1_timing_floor_adjustment is not None
            ),
            "s1_subtype_center_time_s": (
                None if node_item is None else node_item.get("s1_subtype_center_time_s")
            ),
            "s1_subtype_win_frac": s1_node_win_frac,
            "s1_subtype_node_result": s1_node_result,
            "s1_timing_floor_applied": s1_timing_floor_adjustment is not None,
            "s1_timing_floor_adjustment": s1_timing_floor_adjustment,
            "transport_boundary_win_frac": transport_node_win_frac,
            "transport_boundary_node_result": transport_node_result,
            "carry_endpoint_win_frac": carry_node_win_frac,
            "carry_endpoint_node_result": carry_node_result,
            "peg_arrival_win_frac": peg_arrival_node_win_frac,
            "peg_arrival_node_result": peg_arrival_node_result,
            "endpoint_prior_applied": endpoint_prior_adjustment is not None,
            "endpoint_prior_adjustment": endpoint_prior_adjustment,
            "label_prior_applied": label_prior_adjustment is not None,
            "label_prior_adjustment": label_prior_adjustment,
        }
        stage_rows.append(
            {
                "episode_index": episode,
                "policy_short": event_rows[episode].get("policy_short", ""),
                "backbone_stage": backbone_stage,
                "s1_subtype_verdict": (
                    None if s1_node_result is None else s1_node_result["verdict"]
                ),
                "s1_subtype_win_frac": (
                    None if s1_node_win_frac is None else round(s1_node_win_frac, 3)
                ),
                "s1_timing_floor_applied": s1_timing_floor_adjustment is not None,
                "transport_boundary_verdict": (
                    None if transport_node_result is None else transport_node_result["verdict"]
                ),
                "transport_boundary_win_frac": (
                    None if transport_node_win_frac is None else round(transport_node_win_frac, 3)
                ),
                "transport_boundary_overrode": transport_node_overrode,
                "carry_endpoint_verdict": (
                    None if carry_node_result is None else carry_node_result["verdict"]
                ),
                "carry_endpoint_win_frac": (
                    None if carry_node_win_frac is None else round(carry_node_win_frac, 3)
                ),
                "carry_endpoint_overrode": carry_node_overrode,
                "peg_arrival_verdict": (
                    None if peg_arrival_node_result is None else peg_arrival_node_result["verdict"]
                ),
                "peg_arrival_win_frac": (
                    None
                    if peg_arrival_node_win_frac is None
                    else round(peg_arrival_node_win_frac, 3)
                ),
                "peg_arrival_overrode": peg_arrival_node_overrode,
                "endpoint_prior_applied": endpoint_prior_adjustment is not None,
                "label_prior_applied": label_prior_adjustment is not None,
                "refined_stage": int(label[spec.stage_field]),
                "source": source,
                "final_stage": int(label[spec.stage_field]),
            }
        )

    adjusted = _adjust_square_d2_raw_results(raw, labels, refinements)

    sample_labels_csv = write_outputs(
        spec,
        labeler_config,
        adjusted,
        gemini_route=cascade_gemini_route,
    )
    run_dir = sample_labels_csv.parent
    summary = summarize(run_dir, spec, events_csv=config.events_csv)

    stage_labels_path = run_dir / "stage_labels.csv"
    with stage_labels_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_index",
                "policy_short",
                "backbone_stage",
                "s1_subtype_verdict",
                "s1_subtype_win_frac",
                "s1_timing_floor_applied",
                "transport_boundary_verdict",
                "transport_boundary_win_frac",
                "transport_boundary_overrode",
                "carry_endpoint_verdict",
                "carry_endpoint_win_frac",
                "carry_endpoint_overrode",
                "peg_arrival_verdict",
                "peg_arrival_win_frac",
                "peg_arrival_overrode",
                "endpoint_prior_applied",
                "label_prior_applied",
                "refined_stage",
                "source",
                "final_stage",
            ],
        )
        writer.writeheader()
        writer.writerows(stage_rows)

    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    input_provenance_path = input_run_dir / "provenance.json"
    input_provenance = (
        json.loads(input_provenance_path.read_text()) if input_provenance_path.exists() else {}
    )
    provenance.update(
        {
            "source_run_name": config.input_run_name,
            "source_model": input_provenance.get("model"),
            "source_prompt_variant": input_provenance.get("prompt_variant"),
            "cascade": {
                "nodes": [
                    "square_d2_s1_subtype",
                    "square_d2_transport_peg_engagement_boundary",
                    "square_d2_endpoint_action_verifier",
                    "square_d2_strict_peg_arrival_boundary",
                ],
                "model": config.model,
                "label_priors": [
                    name
                    for name, enabled in (
                        (
                            "square_d2_endpoint_prior",
                            any(r.get("endpoint_prior_applied") for r in refinements.values()),
                        ),
                        (
                            "square_d2_s1_aligned_miss_timing_floor",
                            any(r.get("s1_timing_floor_applied") for r in refinements.values()),
                        ),
                        (
                            "square_d2_r0_s1_failure_prior",
                            any(r.get("label_prior_applied") for r in refinements.values()),
                        ),
                    )
                    if enabled
                ],
                "included_episodes": sorted(config.included_episodes),
                "anchor_episodes": sorted(anchor_episodes),
                "anchor_episodes_by_node": {
                    "square_d2_s1_subtype": sorted(_S1_SUBTYPE_ANCHORS),
                    "square_d2_transport_peg_engagement_boundary": sorted(
                        _TRANSPORT_BOUNDARY_ANCHORS
                    ),
                    "square_d2_endpoint_action_verifier": sorted(_CARRY_ENDPOINT_ANCHORS),
                    "square_d2_strict_peg_arrival_boundary": sorted(_PEG_ARRIVAL_ANCHORS),
                },
                "target_episode_few_shot_leave_one_out": True,
                "transport_boundary_override_min_frac": config.transport_override_min_frac,
                "transport_boundary_s4_override_min_frac": config.transport_s4_override_min_frac,
                "carry_endpoint_override_min_frac": config.carry_endpoint_override_min_frac,
                "peg_arrival_override_min_frac": config.peg_arrival_override_min_frac,
                "refinements": refinements,
            },
            "consensus_summary": summary,
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    return CascadeRunOutputs(
        run_dir=run_dir,
        sample_labels_csv=sample_labels_csv,
        labels_joined_csv=run_dir / "labels_joined.csv",
        provenance_json=provenance_path,
        summary_json=run_dir / "gemini_label_summary.json",
    )
