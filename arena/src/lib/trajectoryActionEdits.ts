import type { StageLabelRow } from "../../convex/stageConsistency";

const occurrenceFields = ["attempt_index", "time_s", "confidence", "evidence"] as const;

function requireRecord(value: unknown, name: string): asserts value is StageLabelRow {
  if (value === null || typeof value !== "object" || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) {
    throw new Error(`${name} must be a plain object`);
  }
}

function requireFields(value: StageLabelRow, fields: readonly string[], name: string) {
  if (fields.some((field) => !Object.hasOwn(value, field))) {
    throw new Error(`${name} is missing required fields`);
  }
}

function requireOccurrences(value: unknown): asserts value is StageLabelRow[] {
  if (!Array.isArray(value)) throw new Error("Action occurrences must be an array");
  // Index explicitly: Array.every skips holes and could erase a malformed draft.
  for (let index = 0; index < value.length; index++) {
    if (!Object.hasOwn(value, index)) throw new Error("Action occurrences must not contain holes");
    requireRecord(value[index], "Action occurrence");
    requireFields(value[index], occurrenceFields, "Action occurrence");
  }
}

function occurrencesOf(action: StageLabelRow): StageLabelRow[] {
  requireRecord(action, "Action");
  requireFields(action, ["action_id", "occurred", "first_time_s", "occurrences"], "Action");
  if (typeof action.action_id !== "string" || !action.action_id) {
    throw new Error("Action must have a nonempty action_id");
  }
  requireOccurrences(action.occurrences);
  return action.occurrences;
}

function validTime(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function requireIndex(index: number, occurrences: readonly StageLabelRow[]) {
  if (!Number.isSafeInteger(index) || index < 0 || index >= occurrences.length) {
    throw new Error("Action occurrence index is out of bounds");
  }
}

/** Apply an explicit occurrence edit, never a load-time normalization.
 * Null/invalid times remain in the draft for the real validator to reject.
 * Neither time precision nor list order nor event metadata is changed.
 * Like other immutable review edits, untouched nested values are shared.
 */
export function setActionOccurrences(action: StageLabelRow, occurrences: readonly StageLabelRow[]): StageLabelRow {
  occurrencesOf(action);
  requireOccurrences(occurrences);
  let firstTime: number | null = null;
  for (const event of occurrences) {
    if (validTime(event.time_s) && (firstTime === null || event.time_s < firstTime)) firstTime = event.time_s;
  }
  return { ...action, occurred: occurrences.length > 0, first_time_s: firstTime, occurrences: [...occurrences] };
}

export function patchActionOccurrence(action: StageLabelRow, index: number, patch: StageLabelRow): StageLabelRow {
  const occurrences = occurrencesOf(action);
  requireIndex(index, occurrences);
  requireRecord(patch, "Action occurrence patch");
  if (Object.keys(patch).some((key) => !occurrenceFields.includes(key as typeof occurrenceFields[number]))) {
    throw new Error("Action occurrence patch contains an unknown field");
  }
  return setActionOccurrences(action, occurrences.map((event, i) => i === index ? { ...event, ...patch } : event));
}

export function removeActionOccurrence(action: StageLabelRow, index: number): StageLabelRow {
  const occurrences = occurrencesOf(action);
  requireIndex(index, occurrences);
  return setActionOccurrences(action, occurrences.filter((_, i) => i !== index));
}

/** An explicit No decision removes all occurrences and their redundant summary. */
export function clearActionOccurrences(action: StageLabelRow): StageLabelRow {
  return setActionOccurrences(action, []);
}

/** Explicit chronology repair. The UI must block this while an input is pending
 * or focused, because its editors are indexed by position. Equal times retain
 * their order, with each occurrence's attempt, confidence, and evidence intact.
 */
export function sortActionOccurrencesByTime(action: StageLabelRow): StageLabelRow {
  const occurrences = occurrencesOf(action);
  if (occurrences.some((event) => !validTime(event.time_s))) {
    throw new Error("Finish every occurrence time before sorting");
  }
  const sorted = occurrences.map((event, index) => ({ event, index }))
    .sort((a, b) => (a.event.time_s as number) - (b.event.time_s as number) || a.index - b.index)
    .map(({ event }) => event);
  return setActionOccurrences(action, sorted);
}
