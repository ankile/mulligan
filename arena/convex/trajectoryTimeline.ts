import type { StageLabelRow } from "./stageConsistency";
import type { TrajectoryReviewSpec } from "./trajectoryReview";

type Item = Record<string, unknown>;
export type TrajectoryTimelineAction = {
  kind: "action"; actionIndex: number; occurrenceIndex: number; timeS: number;
  actionId: string; label: string; attemptIndex: number;
};
export type TrajectoryTimelineEvent = {
  kind: "transition" | "failure"; index: number; timeS: number; label: string; attemptIndex: number;
};
export type TrajectoryTimelineRepair = "use_action_time" | "use_event_time";
export type TrajectoryTimelineIssue = {
  id: string; message: string; relation: "not_before" | "same_event";
  prerequisite: TrajectoryTimelineAction;
  dependent: TrajectoryTimelineEvent;
  repairs: TrajectoryTimelineRepair[];
};
const EPSILON = 1e-6;
const ROUTING_PIN = "4d471b328cf0f163b173074c8c5edc5cbf1679130f3d72d8e9d0c49b21847b3b";
const SQUARE_PIN = "78346164d307a9e168fe94fa82a085201083e68f13cbb386cb3980b0f5a3bfae";
function object(value: unknown): Item | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Item : null;
}
function items(value: unknown): Item[] { return Array.isArray(value) ? value.map((item) => object(item) ?? {}) : []; }
function time(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value) && value >= 0; }
function attempt(value: unknown): value is number { return typeof value === "number" && Number.isInteger(value) && value > 0; }
function actionEvents(tag: TrajectoryReviewSpec, row: StageLabelRow, actionId: string): TrajectoryTimelineAction[] {
  const definitions = tag.task_definition.keyActions;
  return items(row.key_action_observations).flatMap((action, actionIndex) => {
    if (action.action_id !== actionId || action.occurred !== true) return [];
    return items(action.occurrences).flatMap((event, occurrenceIndex) => time(event.time_s) && attempt(event.attempt_index)
      ? [{ kind: "action" as const, actionIndex, occurrenceIndex, timeS: event.time_s,
        actionId, label: definitions.find((definition) => definition.id === actionId)?.name ?? actionId,
        attemptIndex: event.attempt_index }] : []);
  });
}
function eventRef(tag: TrajectoryReviewSpec, event: Item, index: number, kind: "transition" | "failure"): TrajectoryTimelineEvent | null {
  if (!time(event.time_s) || !attempt(event.attempt_index)) return null;
  const stage = tag.task_definition.stages.find((stage) => stage.id === event.to_stage_id);
  return { kind, index, timeS: event.time_s, attemptIndex: event.attempt_index,
    label: kind === "transition" ? `${stage?.name ?? "Stage"} entry` : `Failure ${index + 1}` };
}
function routing(tag: TrajectoryReviewSpec): boolean {
  // These OR/clip-identity relations are not expressed by generic stageLinks.
  // Bind them to the inspected immutable definition, not a future task with reused IDs.
  return tag.task_definition.taskId === "routing_d2" && tag.task_definition_sha256 === ROUTING_PIN;
}

/** Pure review gate, separate from the unchanged pipeline prediction contract.
 * Required actions are lower bounds, not promises of identical timestamps.
 * An earlier occurrence may establish a prerequisite across attempt boundaries;
 * we never move an event across attempts or guess which retry a human meant.
 * Malformed/missing fields remain the responsibility of the shape validator.
 */
export function analyzeTrajectoryTimeline(tag: TrajectoryReviewSpec, row: StageLabelRow): TrajectoryTimelineIssue[] {
  const issues: TrajectoryTimelineIssue[] = [];
  const transitions = items(row.stage_transitions);
  const check = (actionIds: string[], dependent: TrajectoryTimelineEvent) => {
    const candidates = actionIds.flatMap((id) => actionEvents(tag, row, id))
      .filter((candidate) => candidate.attemptIndex <= dependent.attemptIndex).sort((a, b) => a.timeS - b.timeS);
    if (!candidates.length || candidates[0].timeS <= dependent.timeS + EPSILON) return;
    const prerequisite = candidates[0];
    const id = `${dependent.kind}:${dependent.index}:${actionIds.join("|")}`;
    // Multiple occurrences/OR alternatives need a manual video decision. A
    // single same-attempt occurrence permits an explicit, reversible alignment.
    const repairable = candidates.length === 1 && prerequisite.attemptIndex === dependent.attemptIndex;
    issues.push({ id, relation: "not_before", prerequisite, dependent,
      repairs: repairable ? ["use_action_time", "use_event_time"] : [],
      message: `${dependent.label} at ${dependent.timeS.toFixed(2)} s precedes ${prerequisite.label} at ${prerequisite.timeS.toFixed(2)} s. Review these times together${repairable ? "." : "; multiple events or attempts require a manual timing correction."}` });
  };
  transitions.forEach((transition, index) => {
    const event = eventRef(tag, transition, index, "transition");
    if (!event) return;
    for (const action of tag.task_definition.keyActions) {
      if (action.stageLinks.some((link) => link.stageId === transition.to_stage_id && link.relation === "required_for_entry")) check([action.id], event);
    }
    if (routing(tag)) {
      if (transition.to_stage_id === "one_clip_seated") check(["first_clip_seated", "second_clip_seated"], event);
      if (transition.to_stage_id === "remaining_clip_reached_off_axis") {
        const contact = remainingClipContact(tag, row);
        if (contact) check([contact], event);
      }
    }
  });
  if (routing(tag)) items(row.failure_events).forEach((failure, index) => {
    if (failure.failure_mode_id !== "off_axis") return;
    const event = eventRef(tag, failure, index, "failure");
    if (!event) return;
    // Use the most recent stage in this attempt. Do not infer an active clip
    // from max_stage or from a future transition in a retry.
    const reached = transitions.filter((transition) => transition.attempt_index === event.attemptIndex &&
      time(transition.time_s) && transition.time_s <= event.timeS + EPSILON)
      .sort((a, b) => (b.time_s as number) - (a.time_s as number))[0];
    if (["controlled_rope_not_at_clip", "first_clip_reached_off_axis", "first_clip_oriented_not_engaged", "first_clip_engaged_unseated"].includes(String(reached?.to_stage_id))) check(["first_clip_contact"], event);
    if (["one_clip_seated", "remaining_clip_reached_off_axis", "remaining_clip_oriented_not_engaged", "remaining_clip_engaged_unseated"].includes(String(reached?.to_stage_id))) {
      const contact = remainingClipContact(tag, row);
      if (contact) check([contact], event);
    }
  });
  for (const [actionId, stageId] of equivalentPairs(tag)) {
    const actions = actionEvents(tag, row, actionId);
    const matching = transitions.flatMap((transition, index) => transition.to_stage_id === stageId ? [{ transition, index }] : []);
    if (actions.length !== 1 || matching.length !== 1) continue;
    const dependent = eventRef(tag, matching[0].transition, matching[0].index, "transition");
    if (!dependent || actions[0].attemptIndex !== dependent.attemptIndex || Math.abs(actions[0].timeS - dependent.timeS) <= EPSILON) continue;
    const id = `${dependent.kind}:${dependent.index}:${actionId}`;
    const existing = issues.findIndex((issue) => issue.id === id);
    const issue: TrajectoryTimelineIssue = { id, relation: "same_event", prerequisite: actions[0], dependent,
      repairs: ["use_action_time", "use_event_time"],
      message: `${dependent.label} at ${dependent.timeS.toFixed(2)} s and ${actions[0].label} at ${actions[0].timeS.toFixed(2)} s describe the same event. Choose its time after reviewing the video.` };
    if (existing >= 0) issues[existing] = issue; else issues.push(issue);
  }
  return issues;
}
function equivalentPairs(tag: TrajectoryReviewSpec): [string, string][] {
  return routing(tag) ? [["rope_grasped", "controlled_rope_not_at_clip"]]
    : tag.task_definition.taskId === "square_d2" && tag.task_definition_sha256 === SQUARE_PIN
      ? [["secure_nut_lift", "secure_nut_lift"]] : [];
}
function assertEditableObjects(row: StageLabelRow): void {
  for (const key of ["stage_transitions", "failure_events", "key_action_observations"]) {
    const value = row[key];
    if (!Array.isArray(value) || Array.from({ length: value.length }, (_, index) => index).some((index) => !Object.prototype.hasOwnProperty.call(value, index) || !object(value[index]))) throw new Error(`Cannot edit malformed ${key}.`);
  }
  for (const action of row.key_action_observations as Item[]) {
    if (!Array.isArray(action.occurrences) || Array.from({ length: action.occurrences.length }, (_, index) => index).some((index) => !Object.prototype.hasOwnProperty.call(action.occurrences, index) || !object((action.occurrences as unknown[])[index]))) throw new Error("Cannot edit malformed action occurrences.");
  }
}
function remainingClipContact(tag: TrajectoryReviewSpec, row: StageLabelRow): string | null {
  // Earlier one-clip progress does not identify the currently seated clip
  // after an explicit loss; remaining-clip inference must decline that trace.
  if (items(row.failure_events).some((event) => event.failure_mode_id === "first_clip_unseated_after_seating") ||
      row.failure_mode === "first_clip_unseated_after_seating") return null;
  const first = actionEvents(tag, row, "first_clip_seated");
  const second = actionEvents(tag, row, "second_clip_seated");
  // One seated clip across this trace unambiguously identifies the remaining
  // clip. Both-seat/reseat/lost-seat histories cannot be reconstructed here.
  if (first.length === 1 && second.length === 0) return "second_clip_contact";
  if (second.length === 1 && first.length === 0) return "first_clip_contact";
  return null;
}
function setActionTime(row: StageLabelRow, ref: TrajectoryTimelineAction, timeS: number): StageLabelRow {
  const actions = items(row.key_action_observations);
  const action = actions[ref.actionIndex];
  const occurrences = items(action.occurrences).map((event, index) => index === ref.occurrenceIndex ? { ...event, time_s: timeS } : event);
  const validTimes = occurrences.map((event) => event.time_s).filter(time);
  return { ...row, key_action_observations: actions.map((item, index) => index === ref.actionIndex
    ? { ...item, occurrences, first_time_s: validTimes.length === occurrences.length ? Math.min(...validTimes) : null } : item) };
}
function setEventTime(row: StageLabelRow, ref: TrajectoryTimelineEvent, timeS: number): StageLabelRow {
  const key = ref.kind === "transition" ? "stage_transitions" : "failure_events";
  const events = items(row[key]);
  const updated = { ...row, [key]: events.map((event, index) => index === ref.index ? { ...event, time_s: timeS } : event) };
  // The primary failure is a summary of this event only when the association
  // was exact and unique before the edit; preserve ambiguous summaries.
  if (ref.kind === "failure" && row.failure_mode === events[ref.index].failure_mode_id &&
      time(row.primary_failure_time_s) && Math.abs(row.primary_failure_time_s - ref.timeS) <= EPSILON &&
      events.filter((event) => event.failure_mode_id === row.failure_mode && time(event.time_s) &&
        Math.abs(event.time_s - ref.timeS) <= EPSILON).length === 1) updated.primary_failure_time_s = timeS;
  return updated;
}

/** Explicit user repair. The original row and all evidence remain untouched.
 * Re-analyze at click time: stale or ambiguous repair requests fail loudly.
 * Other ordering/semantic problems are intentionally left visible to validation.
 */
export function repairTrajectoryTimeline(
  tag: TrajectoryReviewSpec, row: StageLabelRow, issueId: string, repair: TrajectoryTimelineRepair,
): StageLabelRow {
  assertEditableObjects(row);
  const issue = analyzeTrajectoryTimeline(tag, row).find((issue) => issue.id === issueId);
  if (!issue || !issue.repairs.includes(repair)) throw new Error("This timeline repair is no longer available; review the current events.");
  return repair === "use_action_time" ? setEventTime(row, issue.dependent, issue.prerequisite.timeS)
    : setActionTime(row, issue.prerequisite, issue.dependent.timeS);
}

/** Keep narrowly equivalent events together after an explicit edit, only if
 * they already agreed and each had exactly one occurrence in the same attempt.
 * Never runs on load; never repairs pre-existing disagreement. A stage may
 * require several actions or a later confirmation, so required_for_entry alone
 * is insufficient to enable this behavior.
 */
export function synchronizeEquivalentTimelineEdit(tag: TrajectoryReviewSpec, before: StageLabelRow, after: StageLabelRow): StageLabelRow {
  assertEditableObjects(before);
  assertEditableObjects(after);
  const pairs = equivalentPairs(tag);
  let result = after;
  for (const [actionId, stageId] of pairs) {
    const oldActions = actionEvents(tag, before, actionId); const newActions = actionEvents(tag, after, actionId);
    const oldTransitions = items(before.stage_transitions); const newTransitions = items(after.stage_transitions);
    const indices = oldTransitions.flatMap((event, i) => event.to_stage_id === stageId ? [i] : []);
    if (oldActions.length !== 1 || newActions.length !== 1 || indices.length !== 1 || oldTransitions.length !== newTransitions.length) continue;
    const index = indices[0]; const oldEvent = eventRef(tag, oldTransitions[index], index, "transition");
    const newEvent = eventRef(tag, newTransitions[index], index, "transition");
    if (!oldEvent || !newEvent || newTransitions[index].to_stage_id !== stageId ||
        oldActions[0].attemptIndex !== oldEvent.attemptIndex || newActions[0].attemptIndex !== newEvent.attemptIndex ||
        oldEvent.attemptIndex !== newEvent.attemptIndex || Math.abs(oldActions[0].timeS - oldEvent.timeS) > EPSILON) continue;
    const actionChanged = Math.abs(oldActions[0].timeS - newActions[0].timeS) > EPSILON;
    const eventChanged = Math.abs(oldEvent.timeS - newEvent.timeS) > EPSILON;
    if (actionChanged && !eventChanged) result = setEventTime(result, newEvent, newActions[0].timeS);
    if (eventChanged && !actionChanged) result = setActionTime(result, newActions[0], newEvent.timeS);
  }
  return result;
}
