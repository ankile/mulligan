import type { StageLabelRow } from "../../convex/stageConsistency";
import type { TrajectoryReviewSpec } from "../../convex/trajectoryReview";
import { patchActionOccurrence } from "./trajectoryActionEdits";

const ROUTING_PIN = "4d471b328cf0f163b173074c8c5edc5cbf1679130f3d72d8e9d0c49b21847b3b";
const SQUARE_PIN = "78346164d307a9e168fe94fa82a085201083e68f13cbb386cb3980b0f5a3bfae";
const MARKER_PINS = new Set([
  "f1b43f091236fda2389fa57d8cea7b8e15276e7835b42dd9af99c2ce25358927",
  "b190c25bbe62d2d863cb3300364fe0a50ea0aebccfd158009c5b474b5453fc2f",
]);
const EPSILON = 1e-6;
export type UnifiedEventCandidate = {
  id: string; actionId: string; stageId: string; attemptIndex: number;
  actionIndex: number; occurrenceIndex: number; transitionIndex: number;
  actionTimeS: number | null; stageTimeS: number | null; sameTime: boolean;
  relation: "equivalent" | "conditional"; explanation: string;
};
type Item = StageLabelRow;
function record(value: unknown): value is Item {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function records(value: unknown, name: string): Item[] {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  for (let index = 0; index < value.length; index++) {
    if (!Object.prototype.hasOwnProperty.call(value, index) || !record(value[index])) throw new Error(`${name} contains an invalid event`);
  }
  return value;
}
function validTime(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value) && value >= 0; }
function editableTime(value: unknown): value is number | null { return value === null || (typeof value === "number" && Number.isFinite(value)); }
function validAttempt(value: unknown): value is number { return typeof value === "number" && Number.isSafeInteger(value) && value > 0; }
function sameTime(left: unknown, right: unknown): boolean { return validTime(left) && validTime(right) && Math.abs(left - right) <= EPSILON; }
function requireIndex(items: Item[], index: number) {
  if (!Number.isSafeInteger(index) || index < 0 || index >= items.length) throw new Error("Event index is out of bounds");
}

/** Review presentation candidates, not automatic claims of physical identity.
 * Generic stage prerequisites never imply equal timestamps. Conditional pairs
 * require an explicit human shared-event choice. The inspected task pins and
 * unique occurrence/transition restrictions prevent guessing a retry or clip.
 * This function never normalizes input. Invalid structure fails visibly.
 */
export function getUnifiedEventCandidates(spec: TrajectoryReviewSpec, row: StageLabelRow): UnifiedEventCandidate[] {
  const actions = records(row.key_action_observations, "Key actions");
  const transitions = records(row.stage_transitions, "Stage transitions");
  const failures = records(row.failure_events, "Failure events");
  actions.forEach((action) => records(action.occurrences, "Action occurrences"));
  const pairs: [string, string, UnifiedEventCandidate["relation"], string][] = [];
  const isRouting = spec.task_definition.taskId === "routing_d2" && spec.task_definition_sha256 === ROUTING_PIN;
  if (isRouting) {
    pairs.push(["rope_grasped", "controlled_rope_not_at_clip", "equivalent", "Rope control and entry into S2 describe the same event."]);
    pairs.push(["first_clip_contact", "first_clip_reached_off_axis", "conditional", "Share one time if the rope arrived at the first clip off-axis. Keep separate times if its state changed after arrival."]);
    const seats = actions.flatMap((action) => ["first_clip_seated", "second_clip_seated"].includes(String(action.action_id)) && action.occurred === true
      ? records(action.occurrences, "Action occurrences").map((event) => ({ actionId: String(action.action_id), event })) : []);
    const lostSeat = failures.some((event) => event.failure_mode_id === "first_clip_unseated_after_seating") || row.failure_mode === "first_clip_unseated_after_seating";
    // One seat in the entire history identifies the remaining clip without
    // assumptions about reseating, losses, or which of two seats happened first.
    if (seats.length === 1 && !lostSeat && editableTime(seats[0].event.time_s) && validAttempt(seats[0].event.attempt_index)) {
      pairs.push([seats[0].actionId, "one_clip_seated", "equivalent", "This is the only recorded clip seating, so it also establishes the one-clip-seated stage."]);
      const remainingContact = seats[0].actionId === "first_clip_seated" ? "second_clip_contact" : "first_clip_contact";
      const stage = transitions.filter((event) => event.to_stage_id === "remaining_clip_reached_off_axis");
      if (stage.length === 1 && stage[0].attempt_index === seats[0].event.attempt_index && editableTime(stage[0].time_s) && (seats[0].event.time_s === null || stage[0].time_s === null || seats[0].event.time_s <= stage[0].time_s + EPSILON)) {
        pairs.push([remainingContact, "remaining_clip_reached_off_axis", "conditional", "Share one time if the rope reached the remaining clip off-axis while the other clip stayed seated. A later change of state is a separate event."]);
      }
    }
  }
  if (spec.task_definition.taskId === "square_d2" && spec.task_definition_sha256 === SQUARE_PIN) {
    pairs.push(["secure_nut_lift", "secure_nut_lift", "equivalent", "The secure nut lift and entry into S2 describe the same event."]);
    pairs.push(["arrive_at_peg", "retained_arrival_at_peg", "conditional", "Share one time if this arrival established the retained, insertion-directed pose over the peg. A later pose change is a separate event."]);
    pairs.push(["engage_peg_top", "hole_engaged_held", "conditional", "Share one time if peg entry established partial engagement while the gripper still held the nut. Keep a separate stage time if those conditions were established later."]);
    pairs.push(["retain_partial_after_release", "partial_insertion_released", "conditional", "Share one time if this observation established that the released nut stayed captured above the base as the gripper cleared. Later confirmation can be a separate event."]);
    pairs.push(["reach_peg_base", "fully_seated_held", "conditional", "Share one time if visible full seating occurred while the gripper still held the nut. Seating after release does not establish this held stage."]);
  }
  if (spec.task_definition.taskId === "marker_d2" && MARKER_PINS.has(spec.task_definition_sha256)) {
    // S2 requires transport as well as lift; secure_marker_lift alone is not
    // interchangeable with this stage. S7 terminal success remains independent.
    pairs.push(["transport_marker", "secure_retained_lift", "conditional", "Share one time if transport began with the marker securely retained and lifted clear. A lift before transport is a separate action."]);
    pairs.push(["arrive_at_holder", "retained_arrival_at_holder", "conditional", "Share one time if this arrival also established the retained, insertion-directed pose at the holder. A later pose change is a separate event."]);
    pairs.push(["enter_hole", "partial_insertion_held", "conditional", "Share one time if the marker tip entered the holder hole while still held. Keep a separate stage time if the held-insertion conditions were established later."]);
    pairs.push(["reach_full_depth", "fully_seated_held", "conditional", "Share one time if full-depth geometry became visible while the gripper still held the marker. Seating after release does not establish this held stage."]);
  }
  const candidates: UnifiedEventCandidate[] = [];
  for (const [actionId, stageId, relation, explanation] of pairs) {
    const matchingActions = actions.flatMap((action, actionIndex) => action.action_id === actionId ? [{ action, actionIndex }] : []);
    const matchingTransitions = transitions.flatMap((event, transitionIndex) => event.to_stage_id === stageId ? [{ event, transitionIndex }] : []);
    if (matchingActions.length !== 1 || matchingTransitions.length !== 1) continue;
    const { action, actionIndex } = matchingActions[0];
    const occurrences = records(action.occurrences, "Action occurrences");
    const { event: stage, transitionIndex } = matchingTransitions[0];
    if (action.occurred !== true || occurrences.length !== 1) continue;
    const occurrence = occurrences[0];
    if (!validAttempt(occurrence.attempt_index) || occurrence.attempt_index !== stage.attempt_index || !editableTime(occurrence.time_s) || !editableTime(stage.time_s)) continue;
    candidates.push({ id: `${actionId}:${stageId}:${occurrence.attempt_index}`, actionId, stageId, attemptIndex: occurrence.attempt_index,
      actionIndex, occurrenceIndex: 0, transitionIndex, actionTimeS: occurrence.time_s, stageTimeS: stage.time_s,
      relation, sameTime: sameTime(occurrence.time_s, stage.time_s), explanation });
  }
  // A single action cannot own two stage entries. This occurs in noncanonical
  // clip order; preserve the independent editors rather than selecting one.
  return candidates.filter((candidate) => candidates.filter((other) => other.actionIndex === candidate.actionIndex && other.occurrenceIndex === candidate.occurrenceIndex).length === 1);
}

/** Explicit shared-event edit; null preserves an unfinished draft for validation.
 * Re-resolve the semantic identity at click time, never trust stale array indices.
 */
export function setUnifiedEventTime(spec: TrajectoryReviewSpec, row: StageLabelRow, groupId: string, timeS: number | null): StageLabelRow {
  if (!editableTime(timeS)) throw new Error("Event time must be a finite number or null");
  const group = getUnifiedEventCandidates(spec, row).find((candidate) => candidate.id === groupId);
  if (!group) throw new Error("This shared event is no longer unambiguous; review the current events");
  const actions = records(row.key_action_observations, "Key actions");
  const transitions = records(row.stage_transitions, "Stage transitions");
  return { ...row,
    key_action_observations: actions.map((action, index) => index === group.actionIndex ? patchActionOccurrence(action, group.occurrenceIndex, { time_s: timeS }) : action),
    stage_transitions: transitions.map((event, index) => index === group.transitionIndex ? { ...event, time_s: timeS } : event),
  };
}

/** A summary field identifies an event only when mode and time match uniquely.
 * Ambiguous or stale imported summaries remain visible until explicit selection.
 */
export function getPrimaryFailureIndex(row: StageLabelRow): number | null {
  const failures = records(row.failure_events, "Failure events");
  const matches = failures.flatMap((event, index) => event.failure_mode_id === row.failure_mode &&
    ((event.time_s === null && row.primary_failure_time_s === null) || sameTime(event.time_s, row.primary_failure_time_s)) ? [index] : []);
  return matches.length === 1 ? matches[0] : null;
}
export function selectPrimaryFailure(row: StageLabelRow, index: number): StageLabelRow {
  const failures = records(row.failure_events, "Failure events"); requireIndex(failures, index);
  return { ...row, failure_mode: failures[index].failure_mode_id, primary_failure_time_s: failures[index].time_s };
}
export function patchFailureEvent(row: StageLabelRow, index: number, patch: StageLabelRow): StageLabelRow {
  const failures = records(row.failure_events, "Failure events"); requireIndex(failures, index);
  if (!record(patch) || Object.keys(patch).some((key) => !["time_s", "attempt_index", "failure_mode_id", "confidence", "evidence"].includes(key))) throw new Error("Unknown failure event field");
  const primaryIndex = getPrimaryFailureIndex(row);
  const result = { ...row, failure_events: failures.map((event, eventIndex) => eventIndex === index ? { ...event, ...patch } : event) };
  return primaryIndex === index ? selectPrimaryFailure(result, index) : result;
}
/** Removing the selected event clears its summary and requires a new decision.
 * It must not silently select a different failure or claim task success.
 */
export function removeFailureEvent(row: StageLabelRow, index: number): StageLabelRow {
  const failures = records(row.failure_events, "Failure events"); requireIndex(failures, index);
  const selected = getPrimaryFailureIndex(row) === index;
  return { ...row, failure_events: failures.filter((_, eventIndex) => index !== eventIndex),
    ...(selected ? { failure_mode: "", primary_failure_time_s: null } : {}),
  };
}
