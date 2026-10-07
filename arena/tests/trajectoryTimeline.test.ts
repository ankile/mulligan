import { describe, expect, test } from "bun:test";
import fixtures from "./fixtures/trajectory-review-fixtures.json";
import type { StageLabelRow } from "../convex/stageConsistency";
import { normalizeStageSpec } from "../src/lib/stage-spec";
import { analyzeTrajectoryTimeline, repairTrajectoryTimeline, synchronizeEquivalentTimelineEdit } from "../convex/trajectoryTimeline";

const tag = normalizeStageSpec(fixtures.synthetic.tasks.find((task) => task.source_name === "routing_d2_v1")!.spec).trajectory!;
const event = (time_s: number, attempt_index = 1) => ({ time_s, attempt_index, evidence: "source evidence", confidence: "high" });
const action = (action_id: string, time_s: number) => ({ action_id, occurred: true, first_time_s: time_s, occurrences: [event(time_s)] });
const transition = (to_stage_id: string, time_s: number) => ({ ...event(time_s), to_stage_id });
function sample(): StageLabelRow {
  return {
    key_action_observations: [action("rope_grasped", 4.6), action("first_clip_contact", 7.26), action("first_clip_seated", 9.53), action("second_clip_contact", 15.46)],
    stage_transitions: [transition("controlled_rope_not_at_clip", 4.75), transition("first_clip_reached_off_axis", 6.75), transition("one_clip_seated", 8.25), transition("remaining_clip_reached_off_axis", 15.25)],
    failure_events: [{ ...event(15.25), failure_mode_id: "off_axis" }],
    primary_failure_time_s: 15.25, failure_mode: "off_axis", notes: "Original model note",
  };
}
type Item = Record<string, unknown>;
function actions(row: StageLabelRow) { return row.key_action_observations as Item[]; }
function transitions(row: StageLabelRow) { return row.stage_transitions as Item[]; }

describe("trajectory timeline review consistency", () => {
  test("reports both same-event disagreement and four prerequisite disagreements without mutation", () => {
    const row = sample(); const original = structuredClone(row);
    const issues = analyzeTrajectoryTimeline(tag, row);
    expect(issues).toHaveLength(5);
    expect(issues.filter((issue) => issue.relation === "same_event")).toHaveLength(1);
    expect(issues.find((issue) => issue.dependent.kind === "failure")?.prerequisite.actionId).toBe("second_clip_contact");
    expect(row).toEqual(original);
  });
  test("repairing stage times first cannot hide a stale failure timestamp", () => {
    let row = sample(); const original = structuredClone(row);
    for (const issue of analyzeTrajectoryTimeline(tag, row).filter((issue) => issue.dependent.kind === "transition")) row = repairTrajectoryTimeline(tag, row, issue.id, "use_action_time");
    const remaining = analyzeTrajectoryTimeline(tag, row);
    expect(remaining).toHaveLength(1); expect(remaining[0].dependent.kind).toBe("failure");
    row = repairTrajectoryTimeline(tag, row, remaining[0].id, "use_action_time");
    expect(analyzeTrajectoryTimeline(tag, row)).toEqual([]);
    expect(row.primary_failure_time_s).toBe(15.46);
    expect((row.failure_events as Item[])[0].evidence).toBe("source evidence");
    expect(sample()).toEqual(original);
  });
  test("repairing failure first also resolves cleanly", () => {
    let row = sample();
    for (const issue of analyzeTrajectoryTimeline(tag, row).sort((a, b) => Number(b.dependent.kind === "failure") - Number(a.dependent.kind === "failure"))) row = repairTrajectoryTimeline(tag, row, issue.id, "use_action_time");
    expect(analyzeTrajectoryTimeline(tag, row)).toEqual([]);
  });
  test("using event time updates occurrence and derived first time without changing evidence", () => {
    const row = sample(); const issue = analyzeTrajectoryTimeline(tag, row)[0];
    const changed = repairTrajectoryTimeline(tag, row, issue.id, "use_event_time");
    expect(actions(changed)[issue.prerequisite.actionIndex].first_time_s).toBe(issue.dependent.timeS);
    expect((actions(changed)[issue.prerequisite.actionIndex].occurrences as Item[])[0].evidence).toBe("source evidence");
    expect(actions(row)[issue.prerequisite.actionIndex].first_time_s).toBe(issue.prerequisite.timeS);
  });
  test("generic supports-entry does not require equal times", () => {
    const row = sample(); transitions(row)[1].time_s = 8;
    expect(analyzeTrajectoryTimeline(tag, row).some((issue) => issue.dependent.index === 1 && issue.dependent.kind === "transition")).toBe(false);
  });
  test("unknown definition pin disables task-specific assumptions", () => {
    const unknown = { ...tag, task_definition_sha256: "0".repeat(64) };
    expect(analyzeTrajectoryTimeline(unknown, sample()).map((issue) => issue.dependent.index)).toEqual([1]);
  });
  test("repeated events and cross-attempt events cannot receive guessed repairs", () => {
    const row = sample(); (actions(row)[1].occurrences as Item[]).push(event(8));
    expect(analyzeTrajectoryTimeline(tag, row).find((issue) => issue.prerequisite.actionId === "first_clip_contact")?.repairs).toEqual([]);
    transitions(row)[1].attempt_index = 2;
    expect(analyzeTrajectoryTimeline(tag, row).find((issue) => issue.prerequisite.actionId === "first_clip_contact")?.repairs).toEqual([]);
  });
  test("second seat first uses canonical clip identity, not temporal naming", () => {
    const row = sample(); actions(row)[2].action_id = "second_clip_seated";
    actions(row)[1].first_time_s = 17; (actions(row)[1].occurrences as Item[])[0].time_s = 17;
    const issue = analyzeTrajectoryTimeline(tag, row).find((issue) => issue.dependent.kind === "failure");
    expect(issue?.prerequisite.actionId).toBe("first_clip_contact");
  });
  test("lost seat or both-seat history prevents inference of remaining clip", () => {
    const row = sample(); (row.failure_events as Item[]).unshift({ ...event(12), failure_mode_id: "first_clip_unseated_after_seating" });
    expect(analyzeTrajectoryTimeline(tag, row).some((issue) => issue.dependent.kind === "failure")).toBe(false);
    expect(analyzeTrajectoryTimeline(tag, row).some((issue) => issue.dependent.kind === "transition" && issue.dependent.index === 3)).toBe(false);
    const both = sample(); actions(both).push(action("second_clip_seated", 10));
    expect(analyzeTrajectoryTimeline(tag, both).some((issue) => issue.dependent.kind === "failure")).toBe(false);
  });
  test("failure in another attempt does not inherit previous attempt stage", () => {
    const row = sample(); (row.failure_events as Item[])[0].attempt_index = 2;
    expect(analyzeTrajectoryTimeline(tag, row).some((issue) => issue.dependent.kind === "failure")).toBe(false);
  });
  test("exactly agreeing equivalent events follow a later explicit edit in either direction", () => {
    const before = sample(); transitions(before)[0].time_s = 4.6;
    const after = structuredClone(before); (actions(after)[0].occurrences as Item[])[0].time_s = 5; actions(after)[0].first_time_s = 5;
    expect(transitions(synchronizeEquivalentTimelineEdit(tag, before, after))[0].time_s).toBe(5);
    const stageEdit = structuredClone(before); transitions(stageEdit)[0].time_s = 5;
    expect(actions(synchronizeEquivalentTimelineEdit(tag, before, stageEdit))[0].first_time_s).toBe(5);
    expect(transitions(before)[0].time_s).toBe(4.6);
  });
  test("sync does not normalize existing disagreement or prerequisite-only relations", () => {
    const before = sample(); const after = structuredClone(before); (actions(after)[0].occurrences as Item[])[0].time_s = 5;
    expect(transitions(synchronizeEquivalentTimelineEdit(tag, before, after))[0].time_s).toBe(4.75);
    transitions(before)[1].time_s = 7.26; const contactEdit = structuredClone(before); (actions(contactEdit)[1].occurrences as Item[])[0].time_s = 8;
    expect(transitions(synchronizeEquivalentTimelineEdit(tag, before, contactEdit))[1].time_s).toBe(7.26);
  });
  test("repair/sync fail before replacing malformed or sparse objects", () => {
    const row = sample(); const id = analyzeTrajectoryTimeline(tag, row)[0].id;
    for (const malformed of [null, 5]) {
      const broken = structuredClone(row); (broken.failure_events as unknown[]).push(malformed);
      expect(() => repairTrajectoryTimeline(tag, broken, id, "use_action_time")).toThrow("malformed");
      expect(() => synchronizeEquivalentTimelineEdit(tag, row, broken)).toThrow("malformed");
    }
    const sparse = structuredClone(row); delete (sparse.failure_events as unknown[])[0];
    expect(() => repairTrajectoryTimeline(tag, sparse, id, "use_action_time")).toThrow("malformed");
  });
  test("stale repair request fails loudly", () => {
    const row = sample(); const issue = analyzeTrajectoryTimeline(tag, row)[0];
    const fixed = repairTrajectoryTimeline(tag, row, issue.id, "use_action_time");
    expect(() => repairTrajectoryTimeline(tag, fixed, issue.id, "use_action_time")).toThrow("no longer available");
  });
});
