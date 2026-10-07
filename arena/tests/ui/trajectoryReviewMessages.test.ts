import { describe, expect, test } from "bun:test";
import type { StageLabelRow, Violation } from "../../convex/stageConsistency";
import { validateStageLabel } from "../../convex/stageConsistency";
import fixtures from "../fixtures/trajectory-review-fixtures.json";
import { normalizeStageSpec } from "../../src/lib/stage-spec";
import { trajectoryReviewMessage } from "../../src/lib/trajectoryReviewMessages";

const fixture = fixtures.synthetic.tasks.find((task) => task.source_name === "marker_d2_v4")!;
const spec = normalizeStageSpec(fixture.spec);
const valid = fixture.cases.find((item) => item.name === "valid_success")!.review_label!;
const hidden = "hidden-policy-identity-iql-model-77";
const explain = (violation: Violation) => trajectoryReviewMessage(violation, spec, true);

describe("policy-blind trajectory validation explanations", () => {
  test("real validator errors explain an independent summary without changing the annotation", () => {
    const row = structuredClone(valid) as StageLabelRow;
    row.task_success = false;
    const before = structuredClone(row);
    const violations = validateStageLabel(spec, row, 30);
    expect(violations.length).toBeGreaterThan(0);
    expect(violations.map(explain).join(" ")).toContain("Task success must agree");
    expect(violations.map(explain).join(" ")).not.toContain("Unblind");
    expect(row).toEqual(before);
  });

  test("real action inconsistency and occurrence errors identify safe numbered fields", () => {
    const row = structuredClone(valid) as StageLabelRow;
    const actions = row.key_action_observations as StageLabelRow[];
    actions[0].occurred = false;
    actions[1].first_time_s = 0.111;
    (actions[2].occurrences as StageLabelRow[])[0].attempt_index = 0;
    const messages = validateStageLabel(spec, row, 30).map(explain);
    expect(messages.some((message) => message.includes("Action 1 is marked No"))).toBe(true);
    expect(messages.some((message) => message.includes("Action 2 first time must match"))).toBe(true);
    expect(messages.some((message) => message.includes("Action 3 occurrence 1 attempt must be between"))).toBe(true);
  });

  test("real out-of-bounds times can be explained without exposing provider values", () => {
    const row = structuredClone(valid) as StageLabelRow;
    (row.stage_transitions as StageLabelRow[])[0].time_s = 40;
    const violations = validateStageLabel(spec, row, 30).filter((v) => v.code === "trajectory_time_bounds");
    expect(violations.map(explain)).toEqual(["Transition 1 time must be within the episode duration (30.000000 seconds)."]);
  });

  test("real extra keys and invalid IDs never enter blind messages", () => {
    const row = structuredClone(valid) as StageLabelRow;
    const action = (row.key_action_observations as StageLabelRow[])[0];
    action[hidden] = hidden;
    action.action_id = hidden;
    row.final_state = hidden;
    row.failure_mode = hidden;
    const violations = validateStageLabel(spec, row, 30);
    const shape = violations.find((v) => v.code === "trajectory_shape")!;
    expect(shape.message).toContain(hidden);
    expect(explain(shape)).toBe("Action 1 has an invalid structure. Unblind to inspect raw details.");
    expect(violations.map(explain).join(" ")).not.toContain(hidden);
    expect(trajectoryReviewMessage(shape, spec, false)).toBe(shape.message);
  });

  test("all semantic contract fixtures have actionable safe explanations", () => {
    let checked = 0;
    for (const task of fixtures.synthetic.tasks) {
      const taskSpec = normalizeStageSpec(task.spec);
      for (const item of task.cases) {
        for (const message of item.errors) {
          const result = trajectoryReviewMessage({ code: "trajectory_contract", message, fields: ["trajectory"] }, taskSpec, true);
          expect(result).not.toContain("Unblind");
          checked++;
        }
      }
    }
    expect(checked).toBeGreaterThan(30);
  });

  test.each([
    { code: hidden, message: hidden, fields: [hidden] },
    { code: "trajectory_contract", message: hidden, fields: ["trajectory"] },
    { code: "trajectory_contract", message: `key action ${hidden} is required for reached stage ${hidden}`, fields: ["trajectory"] },
    { code: "trajectory_contract", message: `stage_transitions[${hidden}] is not forward`, fields: ["trajectory"] },
    { code: "trajectory_contract", message: `key_action_observations[0].${hidden} must be an integer`, fields: ["trajectory"] },
    { code: "trajectory_contract", message: `stage_transitions[0].attempt_index must be between 1 and attempt_count (${hidden})`, fields: ["trajectory"] },
    { code: "trajectory_contract", message: "__proto__", fields: ["trajectory"] },
    { code: "trajectory_shape", message: hidden, fields: [`label.${hidden}`] },
    { code: "trajectory_shape", message: hidden, fields: ["label.key_action_observations[0]", hidden] },
    { code: "trajectory_time_bounds", message: `${hidden} must be within [0, 30.000000] seconds (up to 0.010s positive endpoint representation rounding is allowed)`, fields: [hidden] },
    { code: "trajectory_time_bounds", message: `primary_failure_time_s must be within [0, ${hidden}] seconds`, fields: ["primary_failure_time_s"] },
    { code: "trajectory_duration", message: hidden, fields: ["trajectory"] },
  ])("unknown codes, paths, and templates stay redacted %#", (violation) => {
    const result = explain(violation);
    expect(result).not.toContain(hidden);
    expect(result).toBe("An annotation field needs review. Unblind to inspect raw details.");
  });

  test("known shape location does not expose arbitrary extra-key names", () => {
    const violation = { code: "trajectory_shape", message: `label: unmapped fields=${hidden}`, fields: ["label"] };
    expect(explain(violation)).toBe("Annotation has an invalid structure. Unblind to inspect raw details.");
  });

  test("legacy raw-value errors do not bypass trajectory redaction", () => {
    expect(explain({ code: "failure_mode_unknown", message: `failure_mode=${hidden}`, fields: ["failure_mode"] })).not.toContain(hidden);
  });
});
