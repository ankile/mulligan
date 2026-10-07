import { describe, expect, test } from "bun:test";
import type { StageLabelRow } from "../../convex/stageConsistency";
import { validateStageLabel } from "../../convex/stageConsistency";
import fixtures from "../fixtures/trajectory-review-fixtures.json";
import { normalizeStageSpec } from "../../src/lib/stage-spec";
import {
  clearActionOccurrences, patchActionOccurrence, removeActionOccurrence,
  setActionOccurrences, sortActionOccurrencesByTime,
} from "../../src/lib/trajectoryActionEdits";

const square = fixtures.synthetic.tasks.find((task) => task.source_name === "square_d2_v3")!;
const spec = normalizeStageSpec(square.spec);
const fixture = square.cases.find((item) => item.name === "valid_failure")!;
const seed = () => structuredClone(fixture.review_label!) as StageLabelRow;
const actionsOf = (row: StageLabelRow) => row.key_action_observations as StageLabelRow[];
const eventsOf = (action: StageLabelRow) => action.occurrences as StageLabelRow[];
const event = (time_s: unknown, evidence = "Visible jaw closure") => ({ attempt_index: 1, time_s, confidence: "medium", evidence });
const messages = (row: StageLabelRow) => validateStageLabel(spec, row, fixture.duration_s).map((v) => v.message);

function withAction(row: StageLabelRow, action: StageLabelRow, index = 2): StageLabelRow {
  return { ...row, key_action_observations: actionsOf(row).map((item, i) => i === index ? action : item) };
}

function deepFreeze(value: unknown) {
  if (value !== null && typeof value === "object") {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
}

describe("explicit trajectory action edits", () => {
  test("Square Action 3 No plus 3.75s remains intact until a chosen repair", () => {
    const row = seed();
    const action = actionsOf(row)[2];
    expect(action.action_id).toBe("close_jaws_on_nut");
    action.occurrences = [event(3.75)];
    const original = structuredClone(row);
    deepFreeze(row);
    expect(messages(row)).toContain("key_action_observations[2] absent-action fields conflict");
    const keepEvents = setActionOccurrences(action, eventsOf(action));
    expect(keepEvents).toEqual({ ...action, occurred: true, first_time_s: 3.75 });
    expect(messages(withAction(row, keepEvents))).toEqual([]);
    const keepNo = clearActionOccurrences(action);
    expect(keepNo).toEqual({ ...action, occurred: false, first_time_s: null, occurrences: [] });
    expect(messages(withAction(row, keepNo))).toEqual([]);
    expect(row).toEqual(original);
  });

  test("Yes creates an unfinished event and cannot pass confirmation until its time is entered", () => {
    const row = seed();
    const action = setActionOccurrences(actionsOf(row)[2], [event(null, "")]);
    expect(action).toMatchObject({ occurred: true, first_time_s: null });
    expect(messages(withAction(row, action))).toContain("key_action_observations[2].occurrences[0].time_s must be a finite number greater than or equal to zero");
    const timed = patchActionOccurrence(action, 0, { time_s: 3.75 });
    expect(timed.first_time_s).toBe(3.75);
    expect(messages(withAction(row, timed))).toEqual([]);
    expect(eventsOf(action)[0].time_s).toBeNull();
  });

  test("editing or marking a time synchronizes both summaries without rounding or inference", () => {
    const row = seed();
    const original = actionsOf(row)[2];
    const entered = setActionOccurrences(original, [event(3.75)]);
    const marked = patchActionOccurrence(entered, 0, { time_s: 58 / spec.fps });
    expect(marked.first_time_s).toBe(58 / spec.fps);
    expect(eventsOf(marked)[0].time_s).toBe(58 / spec.fps);
    expect(eventsOf(marked)[0].evidence).toBe(eventsOf(entered)[0].evidence);
    expect(messages(withAction(row, marked))).toEqual([]);
    expect(original).toMatchObject({ occurred: false, first_time_s: null, occurrences: [] });
  });

  test("clearing a time leaves the occurrence unfinished; removing the last occurrence means No", () => {
    const row = seed();
    const action = setActionOccurrences(actionsOf(row)[2], [event(3.75)]);
    const cleared = patchActionOccurrence(action, 0, { time_s: null });
    expect(cleared).toMatchObject({ occurred: true, first_time_s: null });
    expect(eventsOf(cleared)).toHaveLength(1);
    expect(messages(withAction(row, cleared)).length).toBeGreaterThan(0);
    const removed = removeActionOccurrence(cleared, 0);
    expect(removed).toMatchObject({ occurred: false, first_time_s: null, occurrences: [] });
    expect(messages(withAction(row, removed))).toEqual([]);
  });

  test("repeated occurrences survive edits, and only explicit sort repairs chronology", () => {
    const row = seed();
    row.attempt_count = 2;
    const action = setActionOccurrences(actionsOf(row)[2], [event(3.75, "First attempt"), { ...event(6.5, "Second attempt"), attempt_index: 2 }]);
    const edited = patchActionOccurrence(action, 1, { time_s: 2.25, confidence: "high" });
    expect(edited.first_time_s).toBe(2.25);
    expect(eventsOf(edited).map((item) => item.time_s)).toEqual([3.75, 2.25]);
    expect(messages(withAction(row, edited))).toContain("key_action_observations[2].occurrences must be chronologically nondecreasing by time_s");
    deepFreeze(edited);
    const sorted = sortActionOccurrencesByTime(edited);
    expect(eventsOf(sorted)).toEqual([eventsOf(edited)[1], eventsOf(edited)[0]]);
    expect(eventsOf(sorted)[0]).toMatchObject({ attempt_index: 2, confidence: "high", evidence: "Second attempt" });
    expect(messages(withAction(row, sorted))).toEqual([]);
    expect(eventsOf(edited).map((item) => item.time_s)).toEqual([3.75, 2.25]);
  });

  test("sort is stable for equal times and does not renumber attempts", () => {
    const action = setActionOccurrences(actionsOf(seed())[2], [event(5, "A"), { ...event(1, "B"), attempt_index: 3 }, event(1, "C")]);
    const sorted = sortActionOccurrencesByTime(action);
    expect(eventsOf(sorted).map((item) => item.evidence)).toEqual(["B", "C", "A"]);
    expect(eventsOf(sorted).map((item) => item.attempt_index)).toEqual([3, 1, 1]);
  });

  test("removing the earliest event derives the next first time and retains all other facts", () => {
    const action = setActionOccurrences(actionsOf(seed())[2], [event(0), event(3.75), event(9.125)]);
    expect(action.first_time_s).toBe(0);
    const result = removeActionOccurrence(action, 0);
    expect(result.first_time_s).toBe(3.75);
    expect(eventsOf(result)).toEqual(eventsOf(action).slice(1));
  });

  test("using a saved summary time replaces only the chosen occurrence timestamp", () => {
    const row = seed();
    const action = { ...actionsOf(row)[2], occurred: true, first_time_s: 3.75, occurrences: [event(1.125, "Original evidence"), event(6)] };
    expect(messages(withAction(row, action))).toContain("key_action_observations[2] first_time_s is inconsistent");
    const repaired = patchActionOccurrence(action, 0, { time_s: action.first_time_s });
    expect(eventsOf(repaired)).toEqual([event(3.75, "Original evidence"), event(6)]);
    expect(messages(withAction(row, repaired))).toEqual([]);
    expect(eventsOf(action)[0].time_s).toBe(1.125);
  });

  test("unknown metadata and unrelated label decisions are preserved for validator review", () => {
    const row = seed();
    const action = { ...actionsOf(row)[2], extra_metadata: { original: "keep" }, occurrences: [{ ...event(3.75), extra_evidence: { source: "keep" } }] };
    const original = structuredClone(action);
    deepFreeze(action);
    const patched = patchActionOccurrence(action, 0, { confidence: "high" });
    expect(patched.extra_metadata).toEqual(action.extra_metadata);
    expect(eventsOf(patched)[0]).toEqual({ ...eventsOf(action)[0], confidence: "high" });
    expect(action).toEqual(original);
    const next = withAction(row, patched);
    expect({ ...next, key_action_observations: null }).toEqual({ ...row, key_action_observations: null });
    expect(validateStageLabel(spec, next, fixture.duration_s).some((v) => v.code === "trajectory_shape")).toBe(true);
  });

  test("No does not demote a stage or rewrite the outcome when a required action is removed", () => {
    const success = structuredClone(square.cases.find((item) => item.name === "valid_success")!.review_label!) as StageLabelRow;
    const requiredId = spec.trajectory!.task_definition.keyActions.find((action) => action.stageLinks.some((link) => link.relation === "required_for_entry"))!.id;
    const index = actionsOf(success).findIndex((action) => action.action_id === requiredId);
    const next = withAction(success, clearActionOccurrences(actionsOf(success)[index]), index);
    expect(next.max_stage).toBe(success.max_stage);
    expect(next.task_success).toBe(success.task_success);
    expect(next.final_state).toBe(success.final_state);
    expect(messages(next).some((message) => message.includes("is required for reached stage"))).toBe(true);
  });

  test.each([null, -1, Number.NaN, Number.POSITIVE_INFINITY, "3.75"])("invalid or unfinished time %p is retained and cannot be sorted or confirmed", (time) => {
    const row = seed();
    const action = setActionOccurrences(actionsOf(row)[2], [event(time), event(6)]);
    expect(action.first_time_s).toBe(6);
    expect(eventsOf(action)[0].time_s).toBe(time);
    expect(messages(withAction(row, action)).length).toBeGreaterThan(0);
    expect(() => sortActionOccurrencesByTime(action)).toThrow("Finish every occurrence time");
  });

  test("out-of-duration time remains unchanged and is rejected by the real review validator", () => {
    const row = seed();
    const action = setActionOccurrences(actionsOf(row)[2], [event(fixture.duration_s + 1)]);
    expect(action.first_time_s).toBe(fixture.duration_s + 1);
    expect(validateStageLabel(spec, withAction(row, action), fixture.duration_s).some((v) => v.code === "trajectory_time_bounds")).toBe(true);
  });

  test.each(fixtures.synthetic.tasks)("current $source_name valid labels remain valid under a metadata-only occurrence edit", (task) => {
    const taskSpec = normalizeStageSpec(task.spec);
    const example = task.cases.find((item) => item.name === "valid_success")!;
    const row = structuredClone(example.review_label!) as StageLabelRow;
    const original = structuredClone(row);
    deepFreeze(row);
    const index = actionsOf(row).findIndex((action) => eventsOf(action).length > 0);
    const next = withAction(row, patchActionOccurrence(actionsOf(row)[index], 0, { evidence: "Human observation retained verbatim." }), index);
    expect(validateStageLabel(taskSpec, next, example.duration_s)).toEqual([]);
    expect(row).toEqual(original);
    expect(actionsOf(next).map((action) => action.action_id)).toEqual(actionsOf(row).map((action) => action.action_id));
  });

  test.each([-1, 0.5, 1, Number.NaN, Number.POSITIVE_INFINITY])("invalid index %p fails without touching the original", (index) => {
    const action = setActionOccurrences(actionsOf(seed())[2], [event(3.75)]);
    const original = structuredClone(action);
    expect(() => patchActionOccurrence(action, index, { time_s: 5 })).toThrow("index is out of bounds");
    expect(() => removeActionOccurrence(action, index)).toThrow("index is out of bounds");
    expect(action).toEqual(original);
  });

  test("malformed lists, missing fields, sparse arrays and unknown patches fail loudly", () => {
    const action = actionsOf(seed())[2];
    const sparse: StageLabelRow[] = [];
    sparse.length = 1;
    for (const invalid of [null, {}, [null], [3.75], [{}], sparse]) {
      expect(() => setActionOccurrences(action, invalid as StageLabelRow[])).toThrow();
    }
    expect(() => clearActionOccurrences({ ...action, occurrences: null })).toThrow();
    expect(() => clearActionOccurrences({ occurrences: [] })).toThrow("missing required fields");
    expect(() => clearActionOccurrences({ ...action, action_id: "" })).toThrow("nonempty action_id");
    const populated = setActionOccurrences(action, [event(3.75)]);
    expect(() => patchActionOccurrence(populated, 0, { action_id: "different" })).toThrow("unknown field");
    expect(() => patchActionOccurrence(populated, 0, null as unknown as StageLabelRow)).toThrow("plain object");
  });
});
