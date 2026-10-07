"""``routing_d2`` stage-label spec — the first deformable-manipuland real line.

routing_d2 is a rope-into-clips routing task. A single flexible rope spans the
workspace y-axis; the robot must seat it into two pegboard clips in order — the
FIRST clip, then the SECOND clip — threading left->right across the board. The
rope is deformable, and the robot MAY RE-GRASP the rope between the two clips
(let go of one segment and take hold of another to reach the second clip). The
two clips are spread far apart on the board, so a wrist view cannot see both;
the policy and this labeler consume the two OPPOSING side cameras (``side_1`` and
``side_2``), which together cover the rope end-to-end. There is no wrist camera
in the label inputs — ``wrist_camera_key`` points at ``side_2`` (the second side
view) and is labeled ``SIDE-2`` in the prompt via the engine's camera-role labels.

PROMPT CHAIN. v0 is the base prompt; v0p1 adds the owes-to-seat decision ladder and the
SEATED = captured test; v0p2 adds the WRIST-L third camera; v0p2p1-v0p2p3 add calibration
rulings from reviewed hand-labels. The whole line runs through the single monolithic
labeler; there is no coarse/fine seat cascade.

WHY THE 3RD CAMERA — the fine sub-seat rung (off_axis S3 = reached-but-rotated vs
beside_on_table S4 = aligned-not-over-mouth vs partial_insertion S5 = engaged-at-mouth)
is NOT resolvable from the two fixed OPPOSING side views even at media_resolution=high
(the clips are small and far apart, so a side-only labeler collapses these three to S5).
The gripper-mounted wrist_left view CAN resolve it, since it travels to the attempted
clip, and the human graders always had it. The labeler therefore gets all three views
(side_1 + side_2 + wrist_left, the policy's and the human strip's cameras): the wrist
for the fine seat call, the sides for the grasp/global signal the wrist alone loses.

Sub-goal gates: the two seat events (``first_clip_seated`` at S5,
``second_clip_seated`` at S8) are the load-bearing rungs. ``first_clip_seated_time_s``
is cross-checked downstream against a human-marked reward frame, so the prompt
defines the seat moment crisply: the frame the rope is secured UNDER/INTO the
clip mouth and STAYS there.

Sensor caps (deliberate, documented choices):
- The ONLY proprioceptive cap is the jaws-never-closed grasp cap: if the gripper
  jaws never closed anywhere in the episode, nothing could have been grasped or
  seated, so cap the stage at S1, clear the grasp/both-seat/release booleans, and
  reset any grasp-implying final_state to ``unclear``. Routing needs a routing-
  shaped rule here rather than the shared ``cap_grasp_when_jaws_never_closed``
  factory: that factory clears exactly ONE ``seated_field`` and never touches
  final_state, whereas routing has TWO seat booleans (first + second clip) that
  must BOTH clear, and a final-state vocabulary with no ``_held``/``_released``
  suffix (so the suffix-based ``invalidate_final_state_when_jaws_never_closed``
  factory would no-op). This single hand-authored rule is the routing analogue of
  the factory, with field_refs/emits wired to routing's field names.
- ``remove_release_when_jaws_never_reopened`` is DELIBERATELY OMITTED. Routing
  episodes are multi-grasp-cycle (the robot re-grasps the rope between clips), so
  "the jaws never reopened" is ambiguous evidence — a mid-episode reopen for a
  re-grasp is not a terminal release, and a successful seat does not require a
  final reopen at all (the rope stays clipped whether or not the jaws part).
  Keying a release cap off jaw-reopen physics would mis-fire on both patterns.
- ``invalidate_final_state_when_jaws_never_closed`` is OMITTED as NOT APPLICABLE:
  routing's final_states carry no ``_held``/``_released`` suffix, so the
  suffix-based factory is a no-op on this vocabulary. Its intent (reset an
  impossible final_state when jaws never closed) is folded into the single grasp
  cap above instead.
"""

from __future__ import annotations

from mulligan.real.stage_specs.ladder import StageLadder, StageLevel
from mulligan.real.stage_specs.prompts import PromptLibrary, PromptNode
from mulligan.real.stage_specs.schema import SchemaField
from mulligan.real.stage_specs.sensor_constraints import SensorConstraintRule
from mulligan.real.stage_specs.tasks import StageLabelTaskSpec, register_label_task_spec


# --------------------------------------------------------------------------- #
# Stage ladder (S0..S10). Two COUNT gates: S6 = one clip seated (either clip), S10 =
# both clips seated (task success). The two clips carry SYMMETRIC rungs, each with FOUR
# achievements: reach -> align (correct orientation) -> engage the mouth (contact the
# lip, seating in progress) -> fully seat. Canonically clip 1 then clip 2, so S3-S5 are
# the first-seat approach and S7-S9 the second; but the seat gates are order-AGNOSTIC
# counts (one / both), so a policy that seats clip 2 first still reaches S6 with one clip
# done. The per-clip toggles (first_clip_seated / second_clip_seated) stay honest facts,
# so an out-of-order seat is visible in the bools + the second_seated_first_skipped mode.
# --------------------------------------------------------------------------- #

_LADDER = StageLadder(
    levels=(
        StageLevel(0, "no useful approach toward the rope."),
        StageLevel(
            1,
            "approach/contact with the rope without a USABLE grasp: a missed close, "
            "brushing or nudging the rope, or a grasp that never controls a segment "
            "well enough to route it.",
        ),
        StageLevel(
            2,
            "rope grasped and brought under control — a rope segment is held and "
            "lifted/tensioned enough to route it toward a clip, but not yet brought to "
            "the clip.",
            gate_field="rope_grasped",
        ),
        StageLevel(
            3,
            "the rope has been brought TO the FIRST clip (reached its mouth region) but "
            "is NOT yet aligned with the insertion axis — rotated off the seating axis, "
            "over the clip or beside it. Owes the rotation.",
        ),
        StageLevel(
            4,
            "the rope is ALIGNED with the FIRST clip's insertion axis and beside or "
            "hovering over the mouth, but NOT touching/engaged with the clip lip.",
        ),
        StageLevel(
            5,
            "the rope is aligned AND directly above/at the FIRST clip mouth, engaged "
            "with the lip and being pressed in — the seating is IN PROGRESS (first "
            "contact through partial insertion) but not yet fully secured.",
        ),
        StageLevel(
            6,
            "ONE clip is FULLY SEATED — the rope is secured under/into a clip mouth and "
            "stays there (sub-goal 1). Canonically this is the FIRST clip; if the policy "
            "seated the SECOND clip while the first is still empty (out of order), that "
            "also counts as one clip seated and reaches this rung.",
            gate_any_of=("first_clip_seated", "second_clip_seated"),
        ),
        StageLevel(
            7,
            "with ONE clip seated, the rope has been brought TO the remaining (still-"
            "unseated) clip — canonically the SECOND — reaching its mouth region (may "
            "include a re-grasp) but NOT yet aligned with the insertion axis.",
        ),
        StageLevel(
            8,
            "the rope is ALIGNED with the remaining clip's insertion axis and beside or "
            "hovering over the mouth, but NOT touching/engaged with the clip lip.",
        ),
        StageLevel(
            9,
            "the rope is aligned AND directly above/at the remaining clip mouth, engaged "
            "with the lip and being pressed in — seating IN PROGRESS, not yet secured.",
        ),
        StageLevel(
            10,
            "BOTH clips are FULLY SEATED — the rope is secured in both clip mouths and "
            "stays there. Full task success.",
            gate_all_of=("first_clip_seated", "second_clip_seated"),
        ),
    ),
    success_level=10,
)


# --------------------------------------------------------------------------- #
# Enums. The failure modes are the seating-error shapes observed in routing hand-labels.
# The taxonomy is clip-AGNOSTIC — the stage already says
# which clip is being attempted (S3-S6 = first clip, S7-S10 = second) — so a single
# "off_axis"/"beside_on_table"/... describes the failing seat at whichever clip.
# --------------------------------------------------------------------------- #

_FAILURE_MODES: tuple[str, ...] = (
    "none",  # full success only
    "grasp_miss",  # never got a usable grasp on the rope
    "not_reached_clip",  # grasped but never reached the clip (owes reach) — S2/S6
    "off_axis",  # reached but not oriented to the insertion axis (owes rotation) — S3/S7
    "beside_on_table",  # oriented but not over the mouth (owes a small translate) — S4/S8
    "hovering_over_clip",  # aligned above the mouth but never engages the lip — S4/S8
    "partial_insertion",  # engaged at the mouth, not pushed fully in (owes a push) — S5/S9
    "gripper_blocks_seat",  # engaged but the gripper fingers occupy the mouth — S5/S9
    "regrasp_miss",  # the re-grasp for the second clip misses — closes without securing the rope
    "regrasp_failed",  # the mid-episode re-grasp toward the second clip failed
    "stopped_after_first",  # seated clip1 then terminated/retracted without a genuine clip2 attempt — S6
    "second_seated_first_skipped",  # seated the SECOND clip without ever seating the first (out of order)
    "first_clip_unseated_after_seating",  # clip1 seated, then pulled/fell out during clip2 work
    "timeout_holding",  # terminal timeout still holding the rope, no decisive seat attempt
    "other",  # escape hatch — should now be rare; a persistent "other" is a taxonomy gap
)

# Declarative failure_mode -> forbidden max_stage set (consistency invariant).
# Each clip carries FOUR achievements; a seat-failure mode maps to exactly the rung
# whose achievement is still owed (clip1 rung / clip2 rung):
#   not_reached_clip     -> owes reach            -> S2 / S6
#   off_axis             -> reached, owes rotation -> S3 / S7
#   beside_on_table / hovering_over_clip -> aligned but not engaged -> S4 / S8
#   partial_insertion/gripper_blocks_seat -> engaged at the mouth, seating -> S5 / S9
_NOT_REACHED = frozenset(
    {2, 6}
)  # not_reached_clip: grasped(clip1)/seated-clip1(clip2), clip not reached
_REACHED_MISALIGNED = frozenset({3, 7})  # off_axis lives exactly here
_ALIGNED_NOT_ABOVE = frozenset({4, 8})  # beside_on_table: oriented but not over the mouth
_ENGAGED_SEATING = frozenset(
    {5, 9}
)  # partial_insertion / gripper_blocks_seat: at the mouth, seating
_ALL = frozenset(range(0, 11))
_FAILURE_MODE_FORBIDDEN_STAGES: tuple[tuple[str, frozenset[int]], ...] = (
    ("grasp_miss", frozenset(range(2, 11))),  # no usable grasp => max_stage <= 1
    ("not_reached_clip", _ALL - _NOT_REACHED),
    ("off_axis", _ALL - _REACHED_MISALIGNED),
    ("beside_on_table", _ALL - _ALIGNED_NOT_ABOVE),
    ("hovering_over_clip", _ALL - _ALIGNED_NOT_ABOVE),
    ("partial_insertion", _ALL - _ENGAGED_SEATING),
    ("gripper_blocks_seat", _ALL - _ENGAGED_SEATING),
    # the 2nd-clip re-grasp misses -> stranded at S6 (clip1 seated, nothing in hand).
    ("regrasp_miss", _ALL - frozenset({6})),
    # re-grasp secured but transfer to clip2 failed: clip1 seated (>=6) through clip2
    # engaged (9), not success.
    ("regrasp_failed", _ALL - frozenset({6, 7, 8, 9})),
    # seated clip1 (S6) then terminated (retracted / idle) without a genuine reach toward
    # clip2 -> stranded exactly at S6, like a give-up. Distinct from not_reached_clip (which
    # DID move toward clip2 and fell short) and regrasp_miss (a missed re-grab).
    ("stopped_after_first", _ALL - frozenset({6})),
    # OUT-OF-ORDER: the SECOND clip was seated while the first is still empty. That one
    # seat reaches the count milestone S6 (one clip), and if the policy then works on the
    # remaining (first) clip it can climb S7-S9 — but never S10 (that needs BOTH, at which
    # point it is success, not this mode). So valid S6-S9. The wrong order is carried by the
    # honest per-clip toggles (first=False, second=True) alongside this mode.
    ("second_seated_first_skipped", _ALL - frozenset({6, 7, 8, 9})),
    # Clip1 was genuinely seated (its historical seat time is retained), then
    # came out while seating clip2. With clip2 still seated, the terminal count
    # is one clip -> S6; S7-S9 remain available if the policy then works back
    # toward the now-empty first clip.
    ("first_clip_unseated_after_seating", _ALL - frozenset({6, 7, 8, 9})),
    ("timeout_holding", frozenset({10})),  # terminal non-success
    # "none" is governed by the success<=>failure biconditional; "other" is unconstrained.
)

_FINAL_STATES: tuple[str, ...] = (
    "rope_free",
    "rope_in_gripper",
    "rope_at_first_clip_unseated",
    "rope_in_first_clip",
    "rope_in_first_clip_at_second_unseated",
    "rope_in_second_clip_only",  # OUT-OF-ORDER: rope physically in clip2, clip1 empty (not success)
    "rope_in_both_clips",
    "unclear",
)

# A final_state that shows the rope physically IN a clip at the last frame requires
# that clip's seat bool to be True (safe direction only — a seat that later falls out
# ends in gripper/free/at-clip-unseated and carries no requirement). The two seat bools
# are honest per-clip facts, so each "in clip X" state pins exactly its clip's bool:
# rope_in_second_clip_only (out of order) pins second_clip_seated; rope_in_both_clips
# (success) pins both. success<=>failure_mode coherence is checked separately.
_FINAL_STATE_REQUIRES_GATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rope_in_first_clip", ("first_clip_seated",)),
    ("rope_in_first_clip_at_second_unseated", ("first_clip_seated",)),
    ("rope_in_second_clip_only", ("second_clip_seated",)),
    ("rope_in_both_clips", ("first_clip_seated", "second_clip_seated")),
)

# A final_state that shows the rope AT/IN a clip means the ladder REACHED that clip, so
# max_stage cannot be below the reach rung: "at the first clip" (seated or not) => the
# rope got to clip1 => S3+; anything at the SECOND clip => reached clip2 => S6+. (The
# seat states additionally pin their gate bool via _FINAL_STATE_REQUIRES_GATES above.)
# Both one-clip-seated states (in_first_clip / in_second_clip_only) pin S6 — the count
# milestone is order-agnostic, so an out-of-order clip2 seat is one clip seated == S6+.
_FINAL_STATE_REQUIRES_MIN_STAGE: tuple[tuple[str, int], ...] = (
    ("rope_at_first_clip_unseated", 3),
    ("rope_in_first_clip", 6),
    ("rope_in_second_clip_only", 6),
    ("rope_in_first_clip_at_second_unseated", 7),
    ("rope_in_both_clips", 10),
)

# Final states that imply the rope was at some point grasped/seated, i.e. that a
# jaw close must have occurred. Used by the never-closed grasp cap to reset an
# impossible final_state to "unclear" (the routing-shaped substitute for the
# suffix-based invalidate factory, which no-ops on this vocabulary).
_GRASP_IMPLYING_FINAL_STATES = frozenset(_FINAL_STATES) - {"rope_free", "unclear"}


# --------------------------------------------------------------------------- #
# Response-schema event fields (between the stage integer and final_state).
# --------------------------------------------------------------------------- #

_NUM = "number"
_BOOL = "boolean"

_EVENT_FIELDS: tuple[SchemaField, ...] = (
    SchemaField(
        "rope_grasped",
        _BOOL,
        description="A rope segment was visibly grasped and controlled (held/tensioned)",
    ),
    SchemaField(
        "rope_grasped_time_s",
        _NUM,
        nullable=True,
        description="Time the rope was first grasped, seconds from video start",
    ),
    SchemaField(
        "first_clip_contact_time_s",
        _NUM,
        nullable=True,
        description="Time the rope first contacts the FIRST clip mouth region",
    ),
    SchemaField(
        "first_clip_seated",
        _BOOL,
        description=(
            "Rope secured under/into the FIRST clip and still seated at episode end (sub-goal 1)"
        ),
    ),
    SchemaField(
        "first_clip_seated_time_s",
        _NUM,
        nullable=True,
        description=(
            "Time the rope is first SECURED under/into the first clip mouth; retain this "
            "historical time if it later comes out under "
            "first_clip_unseated_after_seating; cross-checked against a human reward frame"
        ),
    ),
    SchemaField(
        "regrasped",
        _BOOL,
        description="Rope released and re-grasped mid-episode (e.g. between the two clips)",
    ),
    SchemaField(
        "regrasped_time_s",
        _NUM,
        nullable=True,
        description="Time of the first mid-episode re-grasp of the rope",
    ),
    SchemaField(
        "second_clip_contact_time_s",
        _NUM,
        nullable=True,
        description="Time the rope first contacts the SECOND clip mouth region",
    ),
    SchemaField(
        "second_clip_seated",
        _BOOL,
        description="Rope secured under/into the SECOND clip and staying there (task success seat)",
    ),
    SchemaField(
        "second_clip_seated_time_s",
        _NUM,
        nullable=True,
        description="Time the rope is first secured under/into the second clip mouth and stays",
    ),
    SchemaField(
        "rope_released",
        _BOOL,
        description=(
            "TERMINAL release/settle: at episode end the gripper has let go of the rope "
            "and the rope is left in its final routed state (not held)"
        ),
    ),
    SchemaField(
        "rope_released_time_s",
        _NUM,
        nullable=True,
        description="Time of the terminal release of the rope",
    ),
    SchemaField(
        "failure_mode_clear_time_s",
        _NUM,
        nullable=True,
        description=(
            "Time the chosen failure_mode is first UNAMBIGUOUS on video — the single "
            "frame that best shows the failure (used as the negative-anchor frame for "
            "the boosting loop). Null for full success (failure_mode='none')"
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Base prompt (v0); calibration rulings are appended as child variants below.
# The SIDE-1 / SIDE-2 role tokens in the video part labels come from the spec's
# side_camera_label / wrist_camera_label; keep the prompt text consistent.
# --------------------------------------------------------------------------- #

_BASE_PROMPT = f"""You are labeling real-robot video rollouts of a rope-routing task (routing_d2).

You receive TWO synchronized fixed side-camera videos of the same episode: SIDE-1 and SIDE-2. These are OPPOSING side views of the workspace — NEITHER is a wrist (gripper-mounted) camera. The rope spans the full width of the board, so the two side views are placed on opposite sides to cover the rope end to end; use both together to follow the whole rope, the two clips, and the gripper.

Task: a single flexible ROPE lies across the workspace. The robot must seat the rope into TWO pegboard clips, in order — the FIRST clip, then the SECOND clip — threading the rope left->right across the board. Because the rope is deformable and the clips are far apart, the robot may RE-GRASP the rope between the two clips (let go of one part and take hold of another). Seating a clip means the rope is pressed under/into the clip mouth so the clip holds it.

{_LADDER.render()}

Sub-goal gates are COUNTS of seated clips, not clip-specific: S6 is reached when ONE clip is seated (secured under the clip and staying) and S10 (full success) when BOTH clips are seated. Canonically clip 1 is seated first, so S3-S5 are the approach to the first seat (reached-not-oriented S3, oriented-not-over-mouth S4, engaged-at-mouth S5) and S7-S9 the approach to the remaining clip's seat (S7/S8/S9). The order tracks distance-to-seat: reach the clip → clear the rotation into the insertion axis → bring it over the mouth and start pressing in → fully seat. ALWAYS set the two per-clip toggles (first_clip_seated / second_clip_seated) to what actually happened, independently.

OUT-OF-ORDER case: if the robot seats the SECOND clip while the FIRST is still empty, that is still ONE clip seated → S6 (it is NOT zero-credit). Set second_clip_seated=true, first_clip_seated=false, final_state=rope_in_second_clip_only, failure_mode=second_seated_first_skipped. It is a FAILURE (not success — success needs both), but it ranks at the one-clip milestone, above any zero-clip stall. If the policy then goes back and works on the first clip it can climb S7-S9; only both-seated is S10.

Event chain — each event requires POSITIVE visual evidence in the side views:
1. rope_grasped: a rope segment is visibly held and controlled (lifted/tensioned) by the gripper, not merely brushed.
2. first_clip_contact: the rope first reaches the FIRST clip mouth region.
3. first_clip_seated: the DECISIVE moment the rope is secured UNDER/INTO the first clip mouth and STAYS there. Report first_clip_seated_time_s as that frame — the instant it becomes and remains clipped, not the moment the gripper merely arrives near the clip. If you cannot see the rope captured by the clip, it is NOT seated.
4. regrasped: the gripper releases the rope and takes a new hold mid-episode (common between the two clips). Record regrasped_time_s at the first such re-grasp.
5. second_clip_contact / second_clip_seated: the same tests for the SECOND clip. second_clip_seated (with the first clip still seated) is full success.
6. rope_released: a TERMINAL settle — at episode end the gripper has let go and the rope is left in its final routed state.

Failure mode — pick the SINGLE best; 'none' ONLY for full success (S10). The stage says which seat is being attempted (S3-S5 = the first seat, S7-S9 = the second seat), so these describe the failing seat at whichever clip:
- grasp_miss: never got a usable grasp on the rope.
- not_reached_clip: the rope never REACHED the clip being attempted — still grasped, not brought to the clip. First clip: max_stage 2. Second clip (clip1 seated): max_stage 6. If it DID reach the clip but was rotated off-axis use off_axis; if oriented but not seated use beside_on_table / partial_insertion.
- off_axis: the rope reached the clip but is NOT correctly ORIENTED for insertion (rotated off the seating axis, ~45-90 deg) — wherever it sits, over the mouth OR beside it. The expensive rotation is still owed. This is the "reached but not lined up" rung. max_stage 3 (clip1) / 7 (clip2).
- beside_on_table: the rope is correctly ORIENTED for insertion but remains BESIDE the mouth, not touching the clip lip (owes a small lateral translate). If it is beside AND rotated wrong, that is off_axis, not this. max_stage 4 (clip1) / 8 (clip2).
- hovering_over_clip: the rope is correctly oriented and visibly HOVERS DIRECTLY ABOVE the mouth, but repeatedly stops/retracts without touching or engaging the lip. This is distinct from beside_on_table (lateral translation still owed) and partial_insertion (visible lip engagement). max_stage 4 (clip1) / 8 (clip2).
- partial_insertion: the rope is over the mouth, engaged with the lip and being pressed in, but not pushed fully in (halfway / almost seated) — owes a push. max_stage 5 (clip1) / 9 (clip2).
- gripper_blocks_seat: the rope is over the mouth but the gripper fingers occupy the clip mouth, leaving no room to seat — owes clearing the gripper. max_stage 5 (clip1) / 9 (clip2).
- regrasp_miss: the re-grasp for the second clip MISSES — the gripper closes without securing the rope, so nothing is carried toward clip2 (stranded at S6, clip1 seated). This is grasp_miss's analog for the second grab.
- regrasp_failed: the re-grasp secured the rope but the transfer toward the second clip still failed (e.g. dropped mid-move or the wrong segment taken).
- stopped_after_first: the first clip is seated (S6) and the robot then TERMINATES — retracts / opens / idles to the end — WITHOUT a genuine attempt to bring the rope to the second clip (it behaves as if the task were already done). max_stage 6. Use this rather than not_reached_clip when there was no real reach toward clip2, and rather than regrasp_miss when there was no re-grasp attempt at all.
- second_seated_first_skipped: OUT OF ORDER — the robot seats the SECOND clip but the FIRST clip was NEVER seated (it moved on to clip2 with clip1 still empty). This is NOT success, but it IS one clip seated → max_stage 6 (or S7-S9 if it then went back and worked on the first clip). Set second_clip_seated=true (with its time) and first_clip_seated=false, final_state=rope_in_second_clip_only. The seated-clip milestone counts; the wrong order is what this failure_mode records.
- first_clip_unseated_after_seating: the FIRST clip was visibly and properly seated, but the rope later pulled or fell back out while working on the SECOND clip. Preserve first_clip_seated_time_s as the historical first-seat event, but set first_clip_seated=false because clip1 is no longer seated at the end. When clip2 remains seated, set second_clip_seated=true, final_state=rope_in_second_clip_only, and max_stage=6 (or S7-S9 only if the policy then works back toward the now-empty first clip). This is distinct from second_seated_first_skipped: that mode means clip1 was NEVER seated and therefore has no first-seat timestamp.
- timeout_holding: terminal timeout while merely HOLDING the rope with NO decisive seat attempt at the current clip. If the rope was engaged at a clip mouth (over/against the lip, even off-center) use the seating-error mode (partial_insertion / gripper_blocks_seat) instead — an engaged-but-timed-out attempt is a seat error, not idle holding.
- other: none of the above (should be rare — a persistent 'other' is a taxonomy gap to report).

Output discipline:
- Judge by what the rope IS doing, not by the arm's apparent intent. A long static hold of the rope near a clip is NOT evidence of a seat — seating requires positive visual evidence of the rope captured at the clip mouth. Do not infer "seated" from time spent near the clip.
- final_state must describe where the rope physically IS in the LAST frame (rope_free / rope_in_gripper / rope_at_first_clip_unseated / rope_in_first_clip / rope_in_first_clip_at_second_unseated / rope_in_second_clip_only / rope_in_both_clips / unclear), not where the trajectory was heading.
- Be conservative: ambiguous evidence means the LOWER stage, with confidence set accordingly. Use null for times you cannot determine. Report all times in seconds from the start of the video.
"""


# v0p1: the decision procedure (judge the rung by what is still OWED to seat:
# off_axis S3 owes rotation, beside_on_table S4 owes translation, partial_insertion S5 is
# engaged at the mouth) and the SEATED = CAPTURED test.
_V0P1_RULING = """
DECISIVE TESTS (v0p1, from the first 20 R3 hand-labels) — these OVERRIDE your overall impression. v0 read most failed seats as "partial_insertion" and sometimes over-called a first seat; the fix is the seat definition and the reached-vs-oriented-vs-engaged distinction.

SEATED means CAPTURED (the load-bearing test): a clip is seated ONLY when you positively see the rope pressed UNDER/INTO the clip mouth and STAYING there after the gripper moves away or opens. A rope merely brought to the clip, pressed against the lip, or held near/at the mouth until the episode ends is NOT seated. If you cannot see the rope captured under the lip, the clip is NOT seated: set the seat bool false, keep max_stage in the APPROACH band (S3-S5 for the first clip, S7-S9 for the second), and pick the seating-error failure_mode below. When unsure whether a seat held, it did NOT.

WHICH APPROACH RUNG — judge by WHAT IS STILL OWED to seat, not by how close the gripper looks. Once the rope is grasped and moving toward a clip, walk this ladder and STOP at the first unmet step (first clip = S3/S4/S5, second clip = S7/S8/S9):
- Has the rope even REACHED the clip mouth region? If it is still short of the clip (owes the reach) -> not_reached_clip (S2 first / S6 second).
- REACHED but ROTATED off the insertion axis — over or beside the clip but not lined up to go in, owes the rotation into the seating axis -> off_axis (S3 / S7). This is the DEFAULT for "arrived at the clip but not lined up," and it is COMMON — do not skip past it to partial_insertion.
- Correctly ORIENTED but still BESIDE the mouth (owes lateral translation) -> beside_on_table (S4 / S8).
- Correctly ORIENTED and HOVERING directly ABOVE the mouth but never touching/engaging the lip -> hovering_over_clip (S4 / S8).
- Only when the rope is VISIBLY engaged with the lip and being pressed INTO the mouth (you can see it entering, partway in) is it partial_insertion (S5 / S9) — owes the final push; gripper_blocks_seat (S5/S9) when the fingers occupy the mouth and leave no room. RESERVE partial_insertion for VISIBLE engagement at the mouth; it is NOT the catch-all for "near the clip." Most failures in this task end at off_axis or beside_on_table, NOT at partial_insertion — do not advance the rung on optimism.

STOPPED-AFTER-FIRST REQUIRES A REAL FIRST SEAT: use stopped_after_first (S6) ONLY when the first clip is genuinely seated (captured and staying, per the SEATED test) and the robot then quits without a real second-clip attempt. If the first clip was never captured, it is a first-clip seating-error mode (S3-S5), NOT stopped_after_first — that S6 falsely credits a seat that never happened.

GRASP MUST CONTROL THE ROPE: reaching a clip (S3+) requires a CONTROLLED rope segment held in the gripper. If the jaws closed but never established control — the rope slips free, is only brushed/nudged, or is never lifted/tensioned — that is grasp_miss (S1), even if the arm then travels toward the clip. A rope that is not actually in hand cannot be seated.
"""


# v0p2: the WRIST-L third camera. The two fixed opposing side views cannot resolve
# whether the rope is captured vs rotated-off-axis vs aligned-beside; v0p2 feeds all three
# views (side_1 + side_2 + wrist_left) and uses the wrist for the fine seat call and the
# sides for grasp/global.
_V0P2_RULING = """
THIRD CAMERA — WRIST-L IS THE SEAT/ORIENTATION AUTHORITY (v0p2): you now ALSO receive a WRIST-L video — a gripper-mounted camera looking down at whatever the gripper is working on. Because it travels with the arm to the clip being attempted, it is the DECISIVE view for the fine call the two fixed SIDE views cannot make: whether the rope is CAPTURED under the clip lip (seated) versus merely near it, and which approach rung is owed — off_axis (reached but rotated off the insertion axis), beside_on_table (aligned but not over the mouth), partial_insertion / gripper_blocks_seat (engaged at the mouth). Make the SEATED test and the approach-rung / failure_mode choice from WRIST-L.
Keep using SIDE-1 and SIDE-2 for what the wrist cannot hold in one view: which clip is being attempted, whether the rope is actually grasped and under control (a jaws-closed-but-no-control attempt is grasp_miss S1, judged from the sides — do NOT let a wrist close-up of the rope near a clip talk you out of a missed grasp), and the overall left->right routing. When WRIST-L shows the rope captured under the lip and staying, that is the seat; when it shows the rope beside/over the mouth but not under the lip, it is NOT seated (use the S3/S4/S5 approach mode).
"""


# v0p2p1: event history vs terminal state. A seat can happen and later be lost: the event
# timestamp is kept while the terminal seat bool is false. Also the literal owed-action
# rung test and timestamps at the first positive evidence.
_V0P2P1_RULING = """
TRAJECTORY-HISTORY AND TIMESTAMP CALIBRATION (v0p2p1, from the second frozen R7 review):

Treat EVENT HISTORY and TERMINAL STATE as separate questions. Scrub the whole synchronized trajectory once to mark events, then inspect the final frames of all three cameras to assign terminal seat booleans and final_state. A seat timestamp means the rope was genuinely captured at that moment; it does not by itself prove the rope remained seated until the end. If clip 1 was captured and later pulled or fell out while the robot worked on clip 2, retain first_clip_seated_time_s, set terminal first_clip_seated=false, and use first_clip_unseated_after_seating. If clip 2 remains captured, set second_clip_seated=true and final_state=rope_in_second_clip_only. Do not collapse this into second_seated_first_skipped: that mode says clip 1 was NEVER seated.

For the S3/S4/S5 and S7/S8/S9 decision, name the single strongest WRIST-L frame and apply the owed-action test literally:
- rotated relative to the insertion axis -> off_axis (S3/S7);
- aligned but laterally beside/short of the mouth -> beside_on_table (S4/S8);
- aligned at the mouth with visible lip engagement -> partial_insertion (S5/S9);
- use gripper_blocks_seat only when the fingers visibly occupy the mouth at that engaged frame.
Do not infer the rung from elapsed time, arm intent, or proximity alone.

Timestamp each field at its own first positive evidence:
- contact = first frame the rope reaches the clip mouth region, before any later alignment/press;
- seated = first frame it is captured and remains captured long enough to establish the seat (retain it as historical evidence if the seat is later lost);
- failure_mode_clear = first frame at which the CHOSEN mechanism is unambiguous, not episode end and not merely first contact;
- final_state = the physical rope location in the last valid frames, independent of the high-water max_stage.
Use the synchronized SIDE views to disambiguate which clip and whether a regrasp/release occurred; use WRIST-L for rope-vs-mouth geometry.
"""


# v0p2p2: grasp control. Coherent rope motion with the closed gripper is the positive
# evidence; jaw closure, proximity and arm travel are not.
_V0P2P2_RULING = """
GRASP CONTROL REQUIRES ROPE MOTION (v0p2p2, from the third frozen R7 review):

Judge rope_grasped from the SIDE videos as a temporal cause-and-effect test, not from jaw closure, proximity, or the robot's intended motion. Mark a successful grasp only after at least two successive frames show the rope segment moving coherently with the closed gripper — lifted, tensioned, or translated as the gripper moves. The jaws closing near the rope, the arm then travelling toward a clip, or the wrist camera losing sight of the rope are not positive grasp evidence by themselves.

Scrub across the closure and name what the rope itself does. If the gripper moves away while the rope stays in the same place on the table, the attempt is grasp_miss: rope_grasped=false, rope_grasped_time_s=null, and max_stage cannot exceed S1. If the views never show coherent rope motion, prefer grasp_miss and request human review rather than inferring control from robot motion. In notes, briefly state the positive motion evidence (for example "rope lifts with gripper") or the decisive negative evidence ("gripper departs; rope remains fixed").
"""


# v0p2p3: geometry, endpoint and event ownership as concrete visual tests. Reward and
# proprioception are assembled by the pipeline; the prompt still asks Gemini for the
# corresponding visual account for auditing.
_V0P2P3_RULING = """
GEOMETRY, ENDPOINT, AND EVENT OWNERSHIP (v0p2p3, from the fourth frozen R7 review):

For WRIST-L approach geometry, decide ORIENTATION before proximity. First compare the rope segment's local direction with the clip's insertion axis. If it remains visibly curved, diagonal, or rotated relative to that axis, label off_axis (S3/S7) even when it is close to or partly over the clip. Use beside_on_table (S4/S8) only after the rope is visibly parallel/aligned and the remaining error is a lateral translation. Use partial_insertion (S5/S9) only when the ROPE ITSELF visibly reaches and engages the mouth/lip; the gripper or arm arriving over the clip is not rope contact.

Assign final_state from the last valid synchronized frames, after the action is over. A rope segment still moving with and retained by closed jaws ends rope_in_gripper, even if it previously reached a clip. An earlier high-water approach location does not own the endpoint. Conversely, use rope_at_first_clip_unseated only when the rope is physically left at that clip after the gripper no longer holds it.

For regrasped, require a completed release followed by a new controlled hold: the jaws open enough to relinquish the old segment, then close on a different segment and that segment moves with the gripper. Continuous adjustment without release is not a regrasp. Treat the supplied gripper-position series as timing evidence, and use the videos to confirm that the new close controls rope rather than air.

Do not write routine explanatory notes. Leave notes empty unless the episode exposes an exceptional ambiguity or a failure not represented by the schema. Calibrate confidence to the actual boundary evidence: a close S3/S4 or S4/S5 call is not high-confidence merely because all samples repeat it.
"""


_PROMPTS = PromptLibrary(
    [
        PromptNode(
            variant="v0",
            parent=None,
            text=_BASE_PROMPT,
            rationale=(
                "Base routing_d2 ladder and rope/two-clip enums for the rope-into-clips task."
            ),
        ),
        PromptNode(
            variant="v0p1",
            parent="v0",
            text=_V0P1_RULING,
            rationale=(
                "SEATED = captured, the owes-to-seat decision ladder "
                "(off_axis/beside_on_table are the reached-but-unseated rungs, not "
                "partial_insertion) and the stopped_after_first / grasp-control guards."
            ),
        ),
        PromptNode(
            variant="v0p2",
            parent="v0p1",
            text=_V0P2_RULING,
            rationale=(
                "The WRIST-L 3rd camera (side_1+side_2+wrist_left, the policy cameras): the "
                "wrist resolves the sub-seat rung the side views cannot; the sides keep "
                "grasp/global."
            ),
        ),
        PromptNode(
            variant="v0p2p1",
            parent="v0p2",
            text=_V0P2P1_RULING,
            rationale=(
                "Separate event history from terminal clip state, make "
                "clip-1-unseated-after-seating explicit, apply the literal owed-action rung "
                "test, and place contact/failure timestamps at their first positive evidence."
            ),
        ),
        PromptNode(
            variant="v0p2p2",
            parent="v0p2p1",
            text=_V0P2P2_RULING,
            rationale=(
                "Coherent rope displacement across frames is the necessary positive evidence "
                "for grasp control; jaw closure, proximity and arm travel are insufficient."
            ),
        ),
        PromptNode(
            variant="v0p2p3",
            parent="v0p2p2",
            text=_V0P2P3_RULING,
            rationale=(
                "Judge wrist orientation before proximity, "
                "require rope rather than arm contact, separate endpoint hold from high-water "
                "approach, require a release/new-hold cycle for regrasp, and reserve notes for "
                "exceptional cases."
            ),
        ),
    ]
)


# --------------------------------------------------------------------------- #
# Sensor caps. Single rule: the jaws-never-closed grasp cap, routing-shaped
# (two seat booleans + non-suffixed final_state). See the module docstring for
# why the shared single-seat factory and the suffix-based invalidate/never-reopen
# factories are not used here.
# --------------------------------------------------------------------------- #


def _jaws_never_closed(trace: dict, parsed: dict) -> bool:
    return trace["jaw_close_time_s"] is None and int(parsed["max_stage"]) > 1


# Known issue kept for paper fidelity: clears the booleans but not their times; see
# docs/reproduce.md, "Known issues".
def _cap_never_closed(parsed: dict) -> dict:
    parsed["max_stage"] = 1
    parsed["rope_grasped"] = False
    parsed["first_clip_seated"] = False
    parsed["second_clip_seated"] = False
    parsed["rope_released"] = False
    if str(parsed["final_state"]) in _GRASP_IMPLYING_FINAL_STATES:
        parsed["final_state"] = "unclear"
    # Jaws never closed => the rope was never grasped => grasp_miss (S<=1). Any seat sub-rung
    # mode (off_axis/beside_on_table/... require a grasp at S3+) is physically impossible here;
    # leaving it would put a seat mode on the capped S1 row, and the anchor's mode->min-stage
    # rule would force the stage back to S3 and demand a grasp time that cannot exist.
    # Capping the mode too keeps the row anchorable.
    # The cap only fires when max_stage was >1, so every prior mode (including 'none', which is
    # S10-success-only) is now inconsistent at S1 -> grasp_miss is the sole valid mode here.
    parsed["failure_mode"] = "grasp_miss"
    return parsed


_SENSOR_RULES = (
    SensorConstraintRule(
        "jaws_never_closed_caps_s1",
        _jaws_never_closed,
        _cap_never_closed,
        "[sensor-constraint: jaws never closed; stage capped at S1]",
        field_refs=(
            "max_stage",
            "rope_grasped",
            "first_clip_seated",
            "second_clip_seated",
            "rope_released",
            "final_state",
            "failure_mode",
        ),
        emits_final_states=("unclear",),
    ),
)


ROUTING_D2 = register_label_task_spec(
    StageLabelTaskSpec(
        name="routing_d2",
        lifecycle_task="routing_d2",
        dataset_repo_id=("mulligan/real-routing-d2-c00-teleop-mixed"),
        events_csv=None,
        fps=15.0,
        side_camera_key="observation.images.side_1",
        wrist_camera_key="observation.images.side_2",
        side_camera_label="SIDE-1",
        wrist_camera_label="SIDE-2",
        # The gripper-mounted wrist view fed as a 3rd stream to the labeler — the close-up
        # seat/orientation authority the two fixed side views cannot resolve. The labeler
        # (labeler.py build_parts) consumes extra cameras after side+wrist. See _V0P2_RULING.
        extra_camera_keys=("observation.images.wrist_left",),
        extra_camera_labels=("WRIST-L",),
        gripper_state_column="observation.state.gripper_position",
        gripper_close_threshold=0.2,
        taxonomy_version="s10_v1",
        ladder=_LADDER,
        failure_modes=_FAILURE_MODES,
        failure_mode_forbidden_stages=_FAILURE_MODE_FORBIDDEN_STAGES,
        historical_event_failure_modes=(
            ("first_clip_unseated_after_seating", "first_clip_seated"),
        ),
        final_state_requires_gates=_FINAL_STATE_REQUIRES_GATES,
        final_state_requires_min_stage=_FINAL_STATE_REQUIRES_MIN_STAGE,
        final_states=_FINAL_STATES,
        success_final_state="rope_in_both_clips",
        released_field="rope_released",
        stage_field="max_stage",
        stage_field_description="Maximum S0-S10 stage reached per the operational tests",
        final_state_field="final_state",
        final_state_description="State of the rope in the FINAL frame of the video",
        failure_mode_field="failure_mode",
        failure_mode_description="Primary failure mode; 'none' only for full success",
        event_fields=_EVENT_FIELDS,
        prompts=_PROMPTS,
        default_prompt_variant="v0p2p3",
        sensor_rules=_SENSOR_RULES,
    )
)
