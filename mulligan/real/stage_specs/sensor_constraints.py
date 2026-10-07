"""Declarative physics caps applied to a parsed VLM label.

Proprioception (the gripper trace) is ground truth the model cannot override:
jaws that never closed cannot have grasped; jaws that never reopened cannot have
released. The original labeler enforced this in a ~110-line imperative cascade
(``_apply_sensor_constraints``). Here each block becomes a
:class:`SensorConstraintRule` — a predicate over ``(sensor_trace, parsed)`` plus
a field mutation, an audit ``note_tag``, and a review flag — and
:func:`apply_sensor_constraints` folds the task's rule list in order, threading
the mutated label through so later rules observe earlier caps (preserving the
original sequential semantics exactly).

The rules are *task* data (a marker task caps on marker-release physics; a
nut-on-peg task would cap on peg-seating physics), but the fold engine is
task-agnostic. Note tags are appended to ``notes`` with a single leading space
and ``.strip()``, matching the original byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# A sensor trace as produced by the labeler's ``_sensor_trace``: jaw close/reopen
# times (seconds or None), the reopened-before-end flag, and episode duration.
SensorTrace = dict[str, Any]
ParsedLabel = dict[str, Any]


@dataclass(frozen=True)
class SensorConstraintRule:
    """One physics cap.

    ``predicate(trace, parsed) -> bool`` decides whether the rule fires on the
    *current* (possibly already-capped) label. ``mutate(parsed) -> parsed``
    returns a new label with field caps applied (identity for review-only
    rules). On fire, the engine sets ``needs_human_review`` (when
    ``sets_review``) and appends ``note_tag`` to ``notes``.
    """

    name: str
    predicate: Callable[[SensorTrace, ParsedLabel], bool]
    mutate: Callable[[ParsedLabel], ParsedLabel]
    note_tag: str
    sets_review: bool = True
    field_refs: tuple[str, ...] = ()
    """Schema field names this rule reads/writes. Declared so the registry can
    fail loud on a typo'd field reference (a closure's field access is otherwise
    invisible). Empty = unvalidated (e.g. equivalence-pinned hand-written rules)."""
    emits_final_states: tuple[str, ...] = ()
    """final_state values this rule may write (e.g. ``"unclear"``); validated to
    be members of the task's final-state enum at registration."""


def apply_sensor_constraints(
    rules: tuple[SensorConstraintRule, ...],
    item: dict[str, Any],
    parsed: ParsedLabel,
) -> ParsedLabel:
    """Fold ``rules`` over ``parsed`` in order; returns a new label dict.

    ``item["sensor_trace"]`` supplies the proprioceptive ground truth. Rules are
    applied sequentially: each sees the mutations of those before it.

    Note vs the original ``_apply_sensor_constraints``: this reads ``notes`` via
    ``.get(..., "")``, so a label missing the ``notes`` key is hardened here
    where the original would ``KeyError``. That input is unreachable in
    production (``notes`` is a required schema key enforced by the parser), so
    the two are equivalent on every real label; the difference is intentional
    robustness, not a behavior change.
    """
    parsed = dict(parsed)
    trace: SensorTrace = item["sensor_trace"]
    for rule in rules:
        if rule.predicate(trace, parsed):
            parsed = rule.mutate(dict(parsed))
            if rule.sets_review:
                parsed["needs_human_review"] = True
            parsed["notes"] = (str(parsed.get("notes", "")) + " " + rule.note_tag).strip()
    return parsed


def identity(parsed: ParsedLabel) -> ParsedLabel:
    """Mutation for review-only rules (flag, no field change)."""
    return parsed


# --------------------------------------------------------------------------- #
# Physics-general rule factories. These caps follow from gripper proprioception
# alone (a jaw that never closed cannot grasp; one that never reopened cannot
# release) and apply to ANY gripper task, so each task builds them from these
# factories rather than re-deriving them. Task-specific *perceptual* caps (e.g.
# "a long static hold is S3, not S4, because the S3/S4 boundary is unreadable at
# this camera resolution") stay in the task module — they are earned through
# that task's own hand-label calibration, not assumed.
# --------------------------------------------------------------------------- #


# Known issue kept for paper fidelity: clears the booleans but not their times; see
# docs/reproduce.md, "Known issues".
def cap_grasp_when_jaws_never_closed(
    *,
    stage_field: str,
    grasp_field: str,
    seated_field: str,
    released_field: str,
    note_tag: str = "[sensor-constraint: jaws never closed; stage capped at S1]",
) -> SensorConstraintRule:
    """If the jaws never closed and the stage is above S1, cap at S1 (no usable
    grasp) and clear the grasp/seated/released booleans."""

    def predicate(trace: SensorTrace, parsed: ParsedLabel) -> bool:
        return trace["jaw_close_time_s"] is None and int(parsed[stage_field]) > 1

    def mutate(parsed: ParsedLabel) -> ParsedLabel:
        parsed[stage_field] = 1
        parsed[grasp_field] = False
        parsed[seated_field] = False
        parsed[released_field] = False
        return parsed

    return SensorConstraintRule(
        "jaws_never_closed_caps_s1",
        predicate,
        mutate,
        note_tag,
        field_refs=(stage_field, grasp_field, seated_field, released_field),
    )


def invalidate_final_state_when_jaws_never_closed(
    *,
    final_state_field: str,
    held_or_released_suffixes: tuple[str, ...] = ("_held", "_released"),
    unclear_value: str = "unclear",
    note_tag: str = "[sensor-constraint: jaws never closed; claimed final_state impossible]",
) -> SensorConstraintRule:
    """If the jaws never closed, a *_held / *_released final_state is impossible
    (nothing was ever grasped); reset it to ``unclear``."""

    def predicate(trace: SensorTrace, parsed: ParsedLabel) -> bool:
        return trace["jaw_close_time_s"] is None and str(parsed[final_state_field]).endswith(
            held_or_released_suffixes
        )

    def mutate(parsed: ParsedLabel) -> ParsedLabel:
        parsed[final_state_field] = unclear_value
        return parsed

    return SensorConstraintRule(
        "jaws_never_closed_invalidates_final_state",
        predicate,
        mutate,
        note_tag,
        field_refs=(final_state_field,),
        emits_final_states=(unclear_value,),
    )


def remove_release_when_jaws_never_reopened(
    *,
    stage_field: str,
    final_state_field: str,
    released_field: str,
    released_time_field: str,
    success_level: int,
    released_suffix: str = "_released",
    unclear_value: str = "unclear",
    note_tag: str = "[sensor-constraint: jaws never reopened; release claim removed]",
) -> SensorConstraintRule:
    """If the jaws closed but never reopened, no release happened: cap below the
    success rung, clear the release boolean/time, and reset a *_released
    final_state to ``unclear``."""

    def predicate(trace: SensorTrace, parsed: ParsedLabel) -> bool:
        return (
            trace["jaw_close_time_s"] is not None
            and not trace["jaws_reopened_before_episode_end"]
            and (int(parsed[stage_field]) == success_level or bool(parsed[released_field]))
        )

    def mutate(parsed: ParsedLabel) -> ParsedLabel:
        parsed[stage_field] = min(int(parsed[stage_field]), success_level - 1)
        parsed[released_field] = False
        parsed[released_time_field] = None
        if str(parsed[final_state_field]).endswith(released_suffix):
            parsed[final_state_field] = unclear_value
        return parsed

    return SensorConstraintRule(
        "no_reopen_removes_release",
        predicate,
        mutate,
        note_tag,
        field_refs=(stage_field, final_state_field, released_field, released_time_field),
        emits_final_states=(unclear_value,),
    )
