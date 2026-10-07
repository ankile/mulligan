import { describe, expect, test } from "bun:test";
import fixtures from "../fixtures/trajectory-review-fixtures.json";
import { validateStageLabel, type StageLabelRow } from "../../convex/stageConsistency";
import { normalizeStageSpec } from "../../src/lib/stage-spec";
import {
  getUnifiedEventCandidates, setUnifiedEventTime, getPrimaryFailureIndex,
  selectPrimaryFailure, patchFailureEvent, removeFailureEvent,
} from "../../src/lib/trajectoryUnifiedEvents";

const routing = normalizeStageSpec(fixtures.synthetic.tasks.find((task) => task.source_name === "routing_d2_v1")!.spec).trajectory!;
const square = normalizeStageSpec(fixtures.synthetic.tasks.find((task) => task.source_name === "square_d2_v3")!.spec).trajectory!;
const event = (time_s: number | null, attempt_index = 1) => ({ time_s, attempt_index, confidence: "high", evidence: "Original source evidence" });
const action = (action_id: string, time_s: number) => ({ action_id, occurred: true, first_time_s: time_s, occurrences: [event(time_s)] });
const transition = (to_stage_id: string, time_s: number) => ({ ...event(time_s), to_stage_id });
const actions = (row: StageLabelRow) => row.key_action_observations as StageLabelRow[];
const transitions = (row: StageLabelRow) => row.stage_transitions as StageLabelRow[];
const failures = (row: StageLabelRow) => row.failure_events as StageLabelRow[];
function sample(): StageLabelRow {
  return {
    key_action_observations: [action("rope_grasped", 4), action("first_clip_contact", 6.53), action("first_clip_seated", 10.6), action("second_clip_contact", 21.47)],
    stage_transitions: [transition("controlled_rope_not_at_clip", 4), transition("first_clip_reached_off_axis", 5.75), transition("one_clip_seated", 9.5), transition("remaining_clip_reached_off_axis", 18.5)],
    failure_events: [{ ...event(22), failure_mode_id: "off_axis" }, { ...event(23), failure_mode_id: "stuck" }],
    failure_mode: "off_axis", primary_failure_time_s: 22, task_success: false, notes: "Original source note",
  };
}
function freeze(value: unknown) {
  if (value !== null && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); }
}

describe("unified trajectory events", () => {
  test("episode 62 conflicts are grouped without changing any observations", () => {
    const row = sample(); const before = structuredClone(row); freeze(row);
    const groups = getUnifiedEventCandidates(routing, row);
    expect(groups.map((group) => group.stageId)).toEqual(["controlled_rope_not_at_clip", "first_clip_reached_off_axis", "one_clip_seated", "remaining_clip_reached_off_axis"]);
    expect(groups.map((group) => group.relation)).toEqual(["equivalent", "conditional", "equivalent", "conditional"]);
    expect(groups.filter((group) => !group.sameTime)).toHaveLength(3);
    expect(row).toEqual(before);
  });
  test("an explicit shared edit updates one physical time and both wire representations", () => {
    const row = sample(); const before = structuredClone(row); freeze(row);
    const group = getUnifiedEventCandidates(routing, row)[1];
    const result = setUnifiedEventTime(routing, row, group.id, 100 / 15);
    expect(transitions(result)[1].time_s).toBe(100 / 15);
    expect(actions(result)[1].first_time_s).toBe(100 / 15);
    expect((actions(result)[1].occurrences as StageLabelRow[])[0]).toEqual({ ...event(100 / 15) });
    expect(result.notes).toBe(before.notes);
    expect(transitions(result)[0]).toBe(transitions(row)[0]);
    expect(row).toEqual(before);
  });
  test("clearing and reentering a unified time preserves the event and its source fields", () => {
    for (const group of getUnifiedEventCandidates(routing, sample())) {
      const cleared = setUnifiedEventTime(routing, sample(), group.id, null);
      expect(actions(cleared)[group.actionIndex].first_time_s).toBeNull();
      expect(transitions(cleared)[group.transitionIndex].time_s).toBeNull();
      const restored = setUnifiedEventTime(routing, cleared, group.id, 7);
      expect(transitions(restored)[group.transitionIndex].time_s).toBe(7);
    }
  });
  test("negative shared draft input stays editable while confirmation validation rejects it", () => {
    const task = fixtures.synthetic.tasks.find((task) => task.source_name === "routing_d2_v1")!;
    const fixture = fixtures.real_campaign_cases.find((item) => item.name === "real_routing_d2_valid")!;
    const row = structuredClone(fixture.review_label!) as StageLabelRow;
    const group = getUnifiedEventCandidates(routing, row).find((candidate) => candidate.actionId === "rope_grasped")!;
    const invalid = setUnifiedEventTime(routing, row, group.id, -1);
    expect(transitions(invalid)[group.transitionIndex].time_s).toBe(-1);
    expect((actions(invalid)[group.actionIndex].occurrences as StageLabelRow[])[0].time_s).toBe(-1);
    expect(validateStageLabel(normalizeStageSpec(task.spec), invalid, fixture.duration_s).length).toBeGreaterThan(0);
    const repaired = setUnifiedEventTime(routing, invalid, group.id, group.actionTimeS);
    expect(transitions(repaired)[group.transitionIndex].time_s).toBe(group.actionTimeS);
  });
  test("generic stage prerequisites and unknown pins never become shared events", () => {
    const row = sample(); transitions(row).push(transition("first_clip_oriented_not_engaged", 8));
    expect(getUnifiedEventCandidates(routing, row)).toHaveLength(4);
    expect(getUnifiedEventCandidates({ ...routing, task_definition_sha256: "0".repeat(64) }, row)).toEqual([]);
  });
  test("equal conditional times remain conditional, so equality is not consent", () => {
    const row = sample(); transitions(row)[1].time_s = 6.53;
    expect(getUnifiedEventCandidates(routing, row).find((group) => group.stageId === "first_clip_reached_off_axis")).toMatchObject({ sameTime: true, relation: "conditional" });
  });
  test("retries, duplicate transitions, and cross-attempt records decline guessed associations", () => {
    const row = sample(); const id = getUnifiedEventCandidates(routing, row)[1].id;
    (actions(row)[1].occurrences as StageLabelRow[]).push(event(7));
    expect(getUnifiedEventCandidates(routing, row).some((group) => group.actionId === "first_clip_contact")).toBe(false);
    expect(() => setUnifiedEventTime(routing, row, id, 8)).toThrow("no longer unambiguous");
    const duplicate = sample(); transitions(duplicate).push(transition("controlled_rope_not_at_clip", 5));
    expect(getUnifiedEventCandidates(routing, duplicate).some((group) => group.actionId === "rope_grasped")).toBe(false);
    const differentAttempt = sample(); transitions(differentAttempt)[1].attempt_index = 2;
    expect(getUnifiedEventCandidates(routing, differentAttempt).some((group) => group.actionId === "first_clip_contact")).toBe(false);
  });
  test("second clip seated first uses canonical clip identity", () => {
    const row = sample(); actions(row)[2].action_id = "second_clip_seated";
    // The first clip becomes remaining; no earlier first-clip stage observation.
    transitions(row).splice(1, 1);
    const groups = getUnifiedEventCandidates(routing, row);
    expect(groups.find((group) => group.stageId === "one_clip_seated")?.actionId).toBe("second_clip_seated");
    expect(groups.find((group) => group.stageId === "remaining_clip_reached_off_axis")?.actionId).toBe("first_clip_contact");
  });
  test("losses, both seats, and remaining-clip stage before seating suppress clip inference", () => {
    const loss = sample(); failures(loss).push({ ...event(12), failure_mode_id: "first_clip_unseated_after_seating" });
    expect(getUnifiedEventCandidates(routing, loss).some((group) => ["one_clip_seated", "remaining_clip_reached_off_axis"].includes(group.stageId))).toBe(false);
    const both = sample(); actions(both).push(action("second_clip_seated", 24));
    expect(getUnifiedEventCandidates(routing, both).some((group) => group.stageId === "remaining_clip_reached_off_axis")).toBe(false);
    const early = sample(); transitions(early)[3].time_s = 9;
    expect(getUnifiedEventCandidates(routing, early).some((group) => group.stageId === "remaining_clip_reached_off_axis")).toBe(false);
  });
  test("one action cannot own two distinct stage entries", () => {
    const row = sample(); actions(row)[2].action_id = "second_clip_seated";
    expect(getUnifiedEventCandidates(routing, row).filter((group) => group.actionId === "first_clip_contact")).toEqual([]);
  });
  test("Square secure lift uses inspected equivalent relation", () => {
    const row = sample(); row.key_action_observations = [action("secure_nut_lift", 4)]; row.stage_transitions = [transition("secure_nut_lift", 5)];
    expect(getUnifiedEventCandidates(square, row)[0]).toMatchObject({ relation: "equivalent", sameTime: false });
  });
  test("malformed and sparse collections fail before editing rather than silently dropping data", () => {
    for (const key of ["key_action_observations", "stage_transitions", "failure_events"]) {
      const row = sample(); row[key] = new Array(2);
      expect(() => getUnifiedEventCandidates(routing, row)).toThrow("invalid event");
    }
    expect(() => setUnifiedEventTime(routing, sample(), "missing", NaN)).toThrow("finite");
  });
});

describe("primary failure is an event reference", () => {
  test("editing the selected failure updates its summary atomically and immutably", () => {
    const row = sample(); const before = structuredClone(row); freeze(row);
    expect(getPrimaryFailureIndex(row)).toBe(0);
    const next = patchFailureEvent(row, 0, { time_s: 24, failure_mode_id: "other" });
    expect(next).toMatchObject({ primary_failure_time_s: 24, failure_mode: "other" });
    expect(failures(next)[0].evidence).toBe(failures(row)[0].evidence);
    expect(row).toEqual(before);
  });
  test("clearing a selected time keeps the selection through the unfinished edit", () => {
    const row = patchFailureEvent(sample(), 0, { time_s: null });
    expect(row.primary_failure_time_s).toBeNull();
    expect(getPrimaryFailureIndex(row)).toBe(0);
    expect(patchFailureEvent(row, 0, { time_s: 21 }).primary_failure_time_s).toBe(21);
  });
  test("explicit selection changes primary; editing an unselected event does not", () => {
    const selected = selectPrimaryFailure(sample(), 1);
    expect(selected).toMatchObject({ failure_mode: "stuck", primary_failure_time_s: 23 });
    expect(patchFailureEvent(selected, 0, { time_s: 25 }).primary_failure_time_s).toBe(23);
  });
  test("ambiguous imported summary is not silently assigned or changed", () => {
    const row = sample(); failures(row).push({ ...failures(row)[0] });
    expect(getPrimaryFailureIndex(row)).toBeNull();
    expect(patchFailureEvent(row, 0, { time_s: 24 }).primary_failure_time_s).toBe(22);
    const stale = sample(); stale.primary_failure_time_s = 1;
    expect(getPrimaryFailureIndex(stale)).toBeNull();
    expect(patchFailureEvent(stale, 0, { time_s: 24 }).primary_failure_time_s).toBe(1);
  });
  test("removing primary requires a new decision instead of silently selecting another failure", () => {
    const row = removeFailureEvent(sample(), 0);
    expect(row).toMatchObject({ failure_mode: "", primary_failure_time_s: null, task_success: false });
    expect(failures(row)).toHaveLength(1);
    expect(getPrimaryFailureIndex(row)).toBeNull();
    expect(removeFailureEvent(sample(), 1)).toMatchObject({ failure_mode: "off_axis", primary_failure_time_s: 22 });
  });
  test("invalid patch and indices fail loudly", () => {
    expect(() => patchFailureEvent(sample(), 0, { invented: true })).toThrow("Unknown");
    expect(() => selectPrimaryFailure(sample(), -1)).toThrow("bounds");
    expect(() => removeFailureEvent(sample(), 2)).toThrow("bounds");
  });
});

describe("Marker and Square conditional event projections", () => {
  const cases = [
    { source: "marker_d2_v3", pairs: [["transport_marker", "secure_retained_lift"], ["arrive_at_holder", "retained_arrival_at_holder"], ["enter_hole", "partial_insertion_held"], ["reach_full_depth", "fully_seated_held"]] },
    { source: "marker_d2_v4", pairs: [["transport_marker", "secure_retained_lift"], ["arrive_at_holder", "retained_arrival_at_holder"], ["enter_hole", "partial_insertion_held"], ["reach_full_depth", "fully_seated_held"]] },
    { source: "square_d2_v3", pairs: [["arrive_at_peg", "retained_arrival_at_peg"], ["engage_peg_top", "hole_engaged_held"], ["retain_partial_after_release", "partial_insertion_released"], ["reach_peg_base", "fully_seated_held"]] },
  ];
  for (const { source, pairs } of cases) {
    test(`${source}: imported overlapping observations remain conditional until an explicit shared edit`, () => {
      const spec = normalizeStageSpec(fixtures.synthetic.tasks.find((task) => task.source_name === source)!.spec).trajectory!;
      const row: StageLabelRow = { key_action_observations: pairs.map(([actionId], i) => action(actionId, i + 1)),
        stage_transitions: pairs.map(([, stageId], i) => transition(stageId, i + 1.25)), failure_events: [] };
      const original = structuredClone(row); freeze(row);
      const groups = getUnifiedEventCandidates(spec, row);
      expect(groups.map((group) => [group.actionId, group.stageId])).toEqual(pairs);
      expect(groups.every((group) => group.relation === "conditional" && !group.sameTime)).toBe(true);
      for (const group of groups) {
        const revised = setUnifiedEventTime(spec, row, group.id, 100 / 15);
        expect(transitions(revised)[group.transitionIndex].time_s).toBe(100 / 15);
        expect(actions(revised)[group.actionIndex].first_time_s).toBe(100 / 15);
        expect(transitions(revised)[group.transitionIndex].evidence).toBe(transitions(row)[group.transitionIndex].evidence);
      }
      expect(row).toEqual(original);
      expect(getUnifiedEventCandidates({ ...spec, task_definition_sha256: "0".repeat(64) }, row)).toEqual([]);
    });
    test(`${source}: repeated actions and later attempts never receive a guessed pairing`, () => {
      const spec = normalizeStageSpec(fixtures.synthetic.tasks.find((task) => task.source_name === source)!.spec).trajectory!;
      const row: StageLabelRow = { key_action_observations: pairs.map(([actionId], i) => action(actionId, i + 1)),
        stage_transitions: pairs.map(([, stageId], i) => transition(stageId, i + 1.25)), failure_events: [] };
      (actions(row)[0].occurrences as StageLabelRow[]).push(event(10));
      transitions(row)[1].attempt_index = 2;
      expect(getUnifiedEventCandidates(spec, row).map((group) => group.actionId)).toEqual(pairs.slice(2).map(([actionId]) => actionId));
    });
  }
  test("Marker lift alone does not establish transported S2, and terminal S7 remains independent", () => {
    const spec = normalizeStageSpec(fixtures.synthetic.tasks.find((task) => task.source_name === "marker_d2_v4")!.spec).trajectory!;
    const row: StageLabelRow = { key_action_observations: [action("secure_marker_lift", 1), action("reach_full_depth", 2)],
      stage_transitions: [transition("secure_retained_lift", 1), transition("fully_seated_released", 2)], failure_events: [] };
    expect(getUnifiedEventCandidates(spec, row)).toEqual([]);
  });
  test("Square peg engagement and base arrival do not collapse terminal success into an action", () => {
    const row: StageLabelRow = { key_action_observations: [action("engage_peg_top", 1), action("reach_peg_base", 2)],
      stage_transitions: [transition("fully_seated_released", 3)], failure_events: [] };
    expect(getUnifiedEventCandidates(square, row)).toEqual([]);
  });
});
