"""``square_d2`` stage-label spec.

square_d2 is the Thread Nut task: grasp the square nut, bring it to the peg, engage the
hole, seat it at the peg base, and release. The peg is sampled jointly with the nut on a
grid-snapped 15-point layout, and the datasets store role-named cameras.

The ladder, enums, event fields, sensor caps and release thresholds below are shared with
a fixed-peg variant of the task; its prompt chain (``v0`` -> ``v0p5``) is the base the
square_d2 prompts extend.
"""

from __future__ import annotations

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

SQUARE_D2_R1_HELDOUT_DATASET_REPO_ID = "mulligan/real-square-d2-r01-eval"

# --------------------------------------------------------------------------- #
# Fixed-peg contract: ladder, enums, event fields, the v0 -> v0p5 prompt chain and the
# sensor caps.
# --------------------------------------------------------------------------- #

_LADDER = StageLadder(
    levels=(
        StageLevel(0, "no useful approach toward the nut."),
        StageLevel(
            1,
            "approach/pregrasp attempt without a USABLE grasp: a missed close, "
            "poking the nut, or a grasp that never lifts the nut clear of the table.",
        ),
        StageLevel(2, "nut acquired with a usable grasp and lifted clear of the table."),
        StageLevel(
            3,
            "nut held and brought to the peg area in an insertion-directed pose "
            "(held roughly level over the peg region).",
        ),
        StageLevel(
            4,
            "the nut's centre hole engages the peg top (peg tip inside the hole / "
            "hole aligned over the peg), but the nut is not seated at the base.",
        ),
        StageLevel(
            5,
            "gripper released a partial insertion the peg retains (nut hung on the "
            "peg, not dropped to the base).",
        ),
        StageLevel(
            6,
            "nut fully seated (slid to the peg base, sitting flat) while STILL HELD "
            "by the gripper at episode end.",
        ),
        StageLevel(7, "nut released and remains fully seated at the peg base. Full success."),
    ),
    success_level=7,
)


_FAILURE_MODES: tuple[str, ...] = (
    "none",
    "no_useful_approach",
    "pregrasp_misalignment",
    "wrong_approach_direction",
    "no_grasp_attempt",
    "missed_grasp_after_alignment",
    "aligned_grasp_miss",
    "nut_slipped_from_gripper",
    "dropped_nut_during_transport",
    "transport_orientation_wrong_for_insertion",
    "nut_hole_missed_peg",
    "nut_hung_on_peg_not_seated",
    "retreat_from_peg_after_contact",
    "timeout_holding_nut",
    "released_partial_not_seated",
    "other",
)

_FINAL_STATES: tuple[str, ...] = (
    "nut_on_table",
    "nut_in_gripper_away_from_peg",
    "nut_in_gripper_at_peg",
    "nut_partially_on_peg_held",
    "nut_partially_on_peg_released",
    "nut_fully_seated_held",
    "nut_fully_seated_released",
    "nut_dropped_off_table",
    "unclear",
)


_NUM = "number"
_BOOL = "boolean"

_EVENT_FIELDS: tuple[SchemaField, ...] = (
    SchemaField(
        "approach_reached",
        _BOOL,
        description="Gripper reached the nut workspace in a plausible pregrasp pose",
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
        description="Jaws/approach arranged so closing could trap the nut",
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
        description="Nut visibly retained and lifted clear of the table",
    ),
    SchemaField("grasp_acquired_time_s", _NUM, nullable=True, description="Time nut was acquired"),
    SchemaField(
        "grasp_lost",
        _BOOL,
        description="Nut was acquired but later slipped/dropped before useful completion",
    ),
    SchemaField("grasp_lost_time_s", _NUM, nullable=True, description="Time nut was lost"),
    SchemaField(
        "peg_contact_time_s",
        _NUM,
        nullable=True,
        description="Time the nut/gripper first contacts the peg structure",
    ),
    SchemaField(
        "peg_alignment_time_s",
        _NUM,
        nullable=True,
        description="Time the nut's centre hole is first aligned over the peg top",
    ),
    SchemaField(
        "nut_fully_seated",
        _BOOL,
        description="Nut visibly slid to the peg base, sitting flat/flush",
    ),
    SchemaField(
        "nut_fully_seated_time_s",
        _NUM,
        nullable=True,
        description="Time nut first reaches the peg base",
    ),
    SchemaField(
        "nut_released",
        _BOOL,
        description="Jaws opened AND arm retreated AND nut stayed on the peg (a true release)",
    ),
    SchemaField("nut_released_time_s", _NUM, nullable=True, description="Time of true release"),
)


_BASE_PROMPT = """You are labeling real-robot video rollouts of a square-nut-on-peg insertion task (square_d1).

You receive, in order: (1) the SIDE camera video — fixed view, best for the table, the nut, gripper approach, and arm motion; (2) the WRIST camera video — mounted on the arm, the authority on whether the nut is actually between the jaws and over the peg. The two videos are synchronized views of the same episode.

Task: the robot must pick up a square nut from the table, carry it to a fixed vertical peg, lower it so the peg enters the nut's centre hole, and drop the nut down to the peg base so it sits flat.

Stage ladder (label the MAXIMUM stage reached):
- S0: no useful approach toward the nut.
- S1: approach/pregrasp attempt without a USABLE grasp. This includes closes that miss the nut, poking at the nut, and grasps that never lift the nut clear of the table. A grasp counts only if the nut is held in a way that could plausibly be carried to and dropped onto the peg.
- S2: nut acquired with a usable grasp and lifted clear of the table, even if later dropped.
- S3: nut held and brought to the peg area in an insertion-directed pose (held roughly level over the peg region). Carrying the nut near the peg while tilted or far above is still S3.
- S4: the nut's centre hole engages the peg top — the peg tip is inside the hole / the hole is aligned directly over the peg — but the nut is not yet seated at the base.
- S5: the gripper released a partial insertion that the peg retains: the nut is hung on the peg above the base, not dropped flat.
- S6: nut fully seated (slid to the peg base, sitting flat) while STILL HELD by the gripper at episode end.
- S7: nut released and remains fully seated at the peg base. Full success.

THE MOST COMMON FAILURE: the robot grasps the nut, carries it toward the peg, then makes alignment attempts without ever getting the hole over the peg, until the episode times out. That is S3 (or S4 only if the hole verifiably engages the peg) with failure mode timeout_holding_nut. A long static hold near the peg is evidence of FAILURE, never of success. Do not infer "seated" from time spent at the peg.

Insertion-phase event chain — each event requires POSITIVE visual evidence:
1. peg_contact: the nut or gripper first touches the peg structure.
2. peg_alignment: the nut's centre hole is over/around the peg top. Without this, the episode cannot be S4 or higher and nut_fully_seated must be false.
3. fully_seated: the nut reaches the peg base, sitting flat. Requires peg_alignment first. If you cannot verify the nut reached the base, it is NOT seated — say so in notes.
4. release: the gripper sensor trace is ground truth for jaw motion. If the jaws never reopened, NO release happened — never output S7 or a *_released final_state. If the jaws reopened, decide visually whether the nut stayed on the peg (release) or fell/left the peg (drop).

Track the nut itself, continuously — do not narrate the arm's intent:
- After the jaws close, CONFIRM the nut actually leaves the table. If it is still on the table, the grasp missed (S1).
- final_state must describe where the nut physically IS in the last frame, not where the trajectory was heading.

The square nut has 4-fold symmetry, so in-plane yaw rarely blocks insertion; the dominant grasp failures are a wrong approach direction and an aligned-but-missed close.

Be conservative: ambiguous evidence means the LOWER stage, with confidence set accordingly. Use null for times you cannot determine. Report times in seconds from the start of the video.
"""


# v0p1: base-seating, held-at-end, lift and S0/S1 tests, with zoomed crops as the
# authority. The held-at-end S6->S3 floor is also enforced code-side below.
_V0P1_RULING = """
DECISIVE TESTS (v0p1, from the first 18 hand-labels) — these override your overall impression:
- BASE-SEATING (S6/S7 require the nut AT THE PEG BASE): seated means the nut slid all the way DOWN the peg and sits FLAT on the base plate. A nut resting on the peg top, hung partway down, or pressed against the peg while still well above the base is NOT seated — that is S3 (held at the peg) or S4 (hole verifiably over the peg tip). If you cannot confirm the nut reached the base, do NOT call S6/S7; use S3/S4 and set needs_human_review.
- HELD-AT-END (jaws never reopened): a long static hold at the peg with the jaws still closed is S3 (or S4 only if the hole is verifiably over the peg), failure mode timeout_holding_nut — never S6. Do not credit seating for a nut the gripper is still holding unless base contact is unambiguous.
- LIFT CHECK (S2+ requires a real lift): the nut must visibly leave the table held between the fingers. If the jaws close but the nut stays on the table — drags, pivots, or simply does not rise — the grasp missed: S1, aligned_grasp_miss, even if the arm then travels to the peg.
- S0 vs S1: S1 requires a genuine pregrasp alignment over the nut (a graspable, insertion-directed pose) — set pregrasp_alignment_reached true only then. An approach that never achieves a graspable pose (misaligned, off to the side, wrong approach direction) is S0 with pregrasp_misalignment (or no_useful_approach), NOT S1. Within S1: no_grasp_attempt only when the jaws never close despite alignment; missed_grasp_after_alignment / aligned_grasp_miss when a close executed and failed to secure the nut.
- ZOOMED CROPS ARE THE AUTHORITY: the videos and full-frame stills are low-resolution for the small nut/peg/base region. When zoomed final-frame, grasp-moment, and release-moment crops are provided, use THEM to decide base-seating, grasp validity (did the nut leave the table?), and whether the nut stayed on the peg after the jaws reopened.
"""

# v0p4: S0/S1 judged by whether the open jaws straddle the nut before any close.
_V0P4_RULING = """
S0/S1 SHARPENING (v0p4) — the entire remaining error is the S0<->S1 line; this OVERRIDES the v0p1 S0/S1 wording:
A close executing and missing is NOT proof of alignment. Judge pregrasp_alignment_reached ONLY on whether the OPEN jaws actually straddle the nut squarely — the two fingers on opposite sides of the nut body, the nut centred in the jaw gap — in the frames just before any close, and decide this INDEPENDENTLY of whether a close then happened.
- Squarely-straddling pose, jaws then close but fail to secure the nut => S1, aligned_grasp_miss (alignment WAS reached).
- Squarely-straddling pose, jaws never close => S1, no_grasp_attempt (alignment reached, no close — do NOT drop to S0).
- Gripper bumps / shoves / pokes / closes on the nut from an OFF-CENTRE pose (a finger landing on top of or beside the nut, the nut knocked askew or pushed away, the jaws not cleanly straddling it) => S0, pregrasp_misalignment, NOT aligned_grasp_miss — even though a close executed and the nut moved. A nut DISPLACED by a clumsy close is evidence AGAINST alignment, not evidence of a grasp.
- No meaningful reach toward the nut at all => S0, no_useful_approach.
Set pregrasp_alignment_reached to exactly match this decision: true for S1 (both aligned cases), false for S0.
"""

# v0p5: replaces the v0p4 split with an attempt-based one: S1 is any genuine attempt to
# grasp the nut, and the failure mode records why it failed. The labeler sends no
# grasp-window crop with this prompt.
_V0P5_RULING = """
S0/S1 RE-DEFINITION (v0p5) — this REPLACES the v0p4 S0/S1 sharpening above. The STAGE records how far the robot got; the FAILURE MODE (not the stage) records why a grasp failed. Alignment quality does NOT decide S0 vs S1.
- S1 = the robot made a genuine attempt to grasp the NUT and came away without a usable grasp. This INCLUDES misaligned attempts: if the gripper went for the nut — descended onto/around it, bumped or shoved it, closed near it, or hovered in a grasp pose over it — but did not lift it away held, that is S1, however clumsy or off-centre the attempt was. Record the reason in failure_mode: pregrasp_misalignment (went for the nut but the jaws were not aligned to grasp it), aligned_grasp_miss (jaws closed on it but it slipped free), no_grasp_attempt (reached a grasp pose over it but never closed). pregrasp_misalignment is an S1 reason, NOT an S0 reason.
- S0 = no genuine grasp attempt at the nut: the gripper never really went for the nut — it stayed away, moved/searched elsewhere, or only drifted near without engaging it. failure_mode no_useful_approach.
- The deciding question is "did the robot attempt to grasp THE NUT?", never "was the attempt well-aligned?". A clumsy, misaligned, nut-knocking attempt is still an attempt: S1, pregrasp_misalignment.
pregrasp_alignment_reached stays a description of alignment quality (false for a misaligned S1 attempt) and no longer gates the stage.
"""

_FIXED_PEG_PROMPTS = PromptLibrary(
    [
        PromptNode(variant="v0", parent=None, text=_BASE_PROMPT),
        PromptNode(variant="v0p1", parent="v0", text=_V0P1_RULING),
        PromptNode(variant="v0p4", parent="v0p1", text=_V0P4_RULING),
        PromptNode(variant="v0p5", parent="v0p4", text=_V0P5_RULING),
    ]
)


_SEATED_OR_PARTIAL_HELD = (
    "nut_fully_seated_held",
    "nut_fully_seated_released",
    "nut_partially_on_peg_held",
    "nut_partially_on_peg_released",
)


def _held_at_end_above_s3(trace: dict, parsed: dict) -> bool:
    return (
        trace["jaw_close_time_s"] is not None
        and not trace["jaws_reopened_before_episode_end"]
        and (int(parsed["max_stage"]) >= 4 or bool(parsed["nut_fully_seated"]))
    )


def _floor_s3(parsed: dict) -> dict:
    parsed["max_stage"] = min(int(parsed["max_stage"]), 3)
    parsed["nut_fully_seated"] = False
    parsed["nut_fully_seated_time_s"] = None
    parsed["peg_alignment_time_s"] = None
    if str(parsed["final_state"]) in _SEATED_OR_PARTIAL_HELD:
        parsed["final_state"] = "nut_in_gripper_at_peg"
    return parsed


_SENSOR_RULES = (
    cap_grasp_when_jaws_never_closed(
        stage_field="max_stage",
        grasp_field="grasp_acquired",
        seated_field="nut_fully_seated",
        released_field="nut_released",
    ),
    invalidate_final_state_when_jaws_never_closed(final_state_field="final_state"),
    remove_release_when_jaws_never_reopened(
        stage_field="max_stage",
        final_state_field="final_state",
        released_field="nut_released",
        released_time_field="nut_released_time_s",
        success_level=7,
    ),
    SensorConstraintRule(
        "held_at_end_floors_s3",
        _held_at_end_above_s3,
        _floor_s3,
        "[sensor-constraint: jaws never reopened; held-at-end stage floored at S3 per static-hold ruling]",
        field_refs=(
            "max_stage",
            "nut_fully_seated",
            "nut_fully_seated_time_s",
            "peg_alignment_time_s",
            "final_state",
        ),
        emits_final_states=("nut_in_gripper_at_peg",),
    ),
)


# --------------------------------------------------------------------------- #
# square_d2.
# --------------------------------------------------------------------------- #

_D2_CONTEXT = """You are labeling square_d2, the new real-robot square-nut-on-peg task.

The inherited prompt below was calibrated on square_d1. For this run, treat every
reference to square_d1 as square_d2. The stage ladder, final states, failure modes,
and strict success definition are the same: the square nut must be released and
remain fully seated at the peg base.

square_d2-specific context:
- The nut randomization is unchanged from square_d1: nut_x +/- 5 inches, nut_y +/- 9
  inches, nut_yaw +/- pi.
- The peg is no longer fixed. Each episode samples the peg jointly with the nut and
  snaps it to a 1-inch physical marker-dot grid with a half-inch offset.
- The possible peg centers are x in {9.5, 10.5, 11.5} inches forward and y in
  {-4.5, -3.5, -2.5, -1.5, -0.5} inches left. Do not assume the old square_d1 fixed
  peg at (9, -2) inches; first locate the peg for the current episode.
- The dataset stores role-named cameras in the new room. The two videos you receive
  are side_1 (SIDE) and wrist_left (WRIST), synchronized exactly as in the inherited
  prompt.
- The inherited prompt sometimes says "fixed vertical peg". For square_d2, read that
  as "the current episode's sampled vertical peg"; all stage semantics remain the same.

Inherited calibrated square prompt follows.

"""

_D2P3_RULING = """
SQUARE_D2 HELD-TIMEOUT S2/S3 PEG-AREA GATE (v0p5_d2p3):
For held timeouts, do not call S3 / nut_in_gripper_at_peg merely because the nut
touches the peg exterior or passes beside the peg. S3 requires insertion-relevant
peg-area control: the nut is held over the peg top or aligned with the hole such
that continuing the same motion could plausibly insert the peg through the hole.

If the nut only bumps or rests against the side of the peg, never reaches the
peg-top / hole region, or is held beside the peg with no visible insertion path,
call S2 / nut_in_gripper_away_from_peg with failure_mode timeout_holding_nut.
Use peg_contact_time_s only for insertion-relevant peg-top/hole contact; side-only
contact with the peg exterior is not peg_contact_time_s for S3 credit.
"""

_PROMPTS = PromptLibrary(
    [
        PromptNode(
            variant="v0p5_d2p0",
            parent=None,
            text=_D2_CONTEXT + _FIXED_PEG_PROMPTS.assemble("v0p5"),
            rationale=(
                "The fixed-peg v0p5 prompt with a prepended context block for the sampled "
                "grid-snapped peg and role-named cameras."
            ),
        ),
        PromptNode(
            variant="v0p5_d2p3",
            parent="v0p5_d2p0",
            text=_D2P3_RULING,
            rationale=(
                "Held-timeout S2/S3 gate: side contact with the peg exterior is not "
                "insertion progress; S3 needs peg-top/hole-region control."
            ),
        ),
    ]
)


_S1_CANONICAL_FIELDS = (
    "max_stage",
    "final_state",
    "failure_mode",
    "pregrasp_alignment_reached",
    "pregrasp_alignment_time_s",
)


def apply_square_d2_r0_s1_failure_prior(
    label: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Canonicalize the reviewed square_d2 R0 S1 nut-on-table convention.

    The first R0 eval review batch showed a taxonomy mismatch: the model
    correctly found S1/nut_on_table failures but over-split them into
    ``no_grasp_attempt`` / ``aligned_grasp_miss`` and set
    ``pregrasp_alignment_reached=true``. Batch 11-20 exposed one narrower
    ``missed_grasp_after_alignment`` residual, but a focused subtype node
    regressed other reviewed cases, so the selected pipeline keeps this coarse
    prior until the boundary is separable on more evidence.
    """
    if int(label["max_stage"]) != 1:
        return label, None
    if str(label["final_state"]) != "nut_on_table":
        return label, None
    if bool(label.get("grasp_acquired")):
        return label, None
    if str(label.get("failure_mode")) == "missed_grasp_after_alignment":
        return label, None

    before = {field: label.get(field) for field in _S1_CANONICAL_FIELDS}
    out = dict(label)
    out["failure_mode"] = "pregrasp_misalignment"
    out["pregrasp_alignment_reached"] = False
    out["pregrasp_alignment_time_s"] = None
    after = {field: out.get(field) for field in _S1_CANONICAL_FIELDS}
    if before == after:
        return label, None
    out["notes"] = (
        str(out.get("notes", ""))
        + " [square-d2-r0 prior: S1 nut-on-table failure canonicalized to "
        "pregrasp_misalignment with pregrasp_alignment_reached=false]"
    ).strip()
    return out, {"name": "square_d2_r0_s1_failure_prior", "before": before, "after": after}


# Known issue kept for paper fidelity: forced S7 sets event booleans without their times;
# see docs/reproduce.md, "Known issues".
def apply_square_d2_endpoint_prior(
    label: dict[str, Any], event_row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reconcile square_d2 stage endpoints with strict rollout outcomes.

    The outcome-edited rollout result is production metadata, while the VLM stage
    label is the substage explanation. Use the outcome only as a narrow endpoint
    prior: a strict success is S7, and a strict non-success cannot remain S7.
    Intermediate non-endpoint rungs are left untouched.
    """

    outcome = str(event_row["original_outcome"]).strip().lower()
    if outcome in {"", "nan", "none"}:
        return dict(label), None

    out = dict(label)
    before = {
        "max_stage": int(out["max_stage"]),
        "final_state": str(out["final_state"]),
        "failure_mode": str(out["failure_mode"]),
        "nut_fully_seated": bool(out["nut_fully_seated"]),
        "nut_released": bool(out["nut_released"]),
    }
    adjustment: dict[str, Any] | None = None

    if outcome == "success" and int(out["max_stage"]) != 7:
        out["max_stage"] = 7
        out["final_state"] = "nut_fully_seated_released"
        out["failure_mode"] = "none"
        out["nut_fully_seated"] = True
        out["nut_released"] = True
        out["grasp_lost"] = False
        out["grasp_lost_time_s"] = None
        out["needs_human_review"] = True
        out["notes"] = (
            str(out.get("notes", "")) + " [square_d2-endpoint-prior: rollout outcome success -> S7]"
        ).strip()
        adjustment = {
            "name": "square_d2_endpoint_prior",
            "original_outcome": outcome,
            "before": before,
            "after": {
                "max_stage": 7,
                "final_state": "nut_fully_seated_released",
                "failure_mode": "none",
                "nut_fully_seated": True,
                "nut_released": True,
            },
        }
    non_success_claims_success_endpoint = outcome != "success" and (
        int(out["max_stage"]) == 7
        or str(out["final_state"]) == "nut_fully_seated_released"
        or str(out["failure_mode"]) == "none"
    )
    if non_success_claims_success_endpoint:
        out["max_stage"] = 5
        out["final_state"] = "nut_partially_on_peg_released"
        out["failure_mode"] = "released_partial_not_seated"
        out["nut_fully_seated"] = False
        out["nut_fully_seated_time_s"] = None
        out["nut_released"] = True
        out["grasp_lost"] = False
        out["needs_human_review"] = True
        out["notes"] = (
            str(out.get("notes", ""))
            + " [square_d2-endpoint-prior: non-success rollout outcome demoted success endpoint to S5]"
        ).strip()
        adjustment = {
            "name": "square_d2_endpoint_prior",
            "original_outcome": outcome,
            "before": before,
            "after": {
                "max_stage": 5,
                "final_state": "nut_partially_on_peg_released",
                "failure_mode": "released_partial_not_seated",
                "nut_fully_seated": False,
                "nut_released": True,
            },
        }

    return out, adjustment


SQUARE_D2 = register_label_task_spec(
    StageLabelTaskSpec(
        name="square_d2",
        lifecycle_task="square_d2",
        dataset_repo_id="mulligan/real-square-d2-c00-teleop-mixed",
        events_csv=None,
        fps=15.0,
        side_camera_key="observation.images.side_1",
        wrist_camera_key="observation.images.wrist_left",
        gripper_state_column="observation.state.gripper_position",
        gripper_close_threshold=0.2,
        # Relaxed release detection (vs marker's 0.5/0.3): a fixed-peg-setup release is
        # often a shallow partial-open-then-reclose — the gripper opens just enough
        # to drop a seated nut, then closes on air — so a recovering end-of-trace dip
        # off the hold plateau must still count as a release. (0.6, 0.05) catches both
        # no-reopen successes (eps 29/36) with zero held-timeout false-positives.
        release_final_abs_max=0.6,
        release_plateau_margin=0.05,
        taxonomy_version="s7_v1",
        ladder=_LADDER,
        failure_modes=_FAILURE_MODES,
        final_states=_FINAL_STATES,
        success_final_state="nut_fully_seated_released",
        released_field="nut_released",
        stage_field="max_stage",
        stage_field_description="Maximum S0-S7 stage reached per the operational tests",
        final_state_field="final_state",
        final_state_description="State of the nut in the FINAL frame of the video",
        failure_mode_field="failure_mode",
        failure_mode_description="Primary failure mode; 'none' only for full success",
        event_fields=_EVENT_FIELDS,
        prompts=_PROMPTS,
        default_prompt_variant="v0p5_d2p3",
        sensor_rules=_SENSOR_RULES,
    )
)
