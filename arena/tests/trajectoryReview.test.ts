import { describe, expect, test } from "bun:test";
import fixtures from "./fixtures/trajectory-review-fixtures.json";
import { validateTrajectoryLabel } from "../convex/trajectoryContract";
import { canonicalizeStageLabel, validateStageLabel } from "../convex/stageConsistency";
import { blankTrajectoryReview, trajectoryFromReview } from "../convex/trajectoryReview";
import { normalizeStageSpec } from "../src/lib/stage-spec";

describe("trajectory review contract fixtures", () => {
  for (const task of fixtures.synthetic.tasks) {
    const spec = normalizeStageSpec(task.spec);
    for (const fixture of task.cases) {
      test(`${task.source_name}: ${fixture.name}`, () => {
        expect(validateTrajectoryLabel(fixture.canonical, spec.trajectory!.task_definition)).toEqual(fixture.errors);
        if (fixture.review_label) {
          expect(trajectoryFromReview(fixture.review_label)).toEqual(fixture.canonical);
          expect(canonicalizeStageLabel(spec, fixture.review_label)).toEqual({ label: fixture.review_label, unknownKeys: [] });
          const violations = validateStageLabel(spec, fixture.review_label, fixture.duration_s);
          expect(violations.filter((item) => item.code === "trajectory_contract").map((item) => item.message)).toEqual(fixture.errors);
          expect(violations.filter((item) => item.code === "trajectory_time_bounds").map((item) => item.message)).toEqual(fixture.time_errors);
        }
      });
    }
    test(`${task.source_name}: manual blank has identity and no inherited success`, () => {
      const blank = blankTrajectoryReview(spec.trajectory!, "org/test", 123);
      expect(blank.trajectory_identity).toEqual({ schema_version: "trajectory-label/v1", task_id: spec.task,
        taxonomy_version: spec.trajectory!.task_definition.taxonomyVersion, sample_id: "org/test#episode=123" });
      expect(blank.task_success).toBeNull();
      expect(blank.max_stage).toBeNull();
      expect(blank.stage_transitions).toEqual([]);
      expect(validateStageLabel(spec, blank, 30).length).toBeGreaterThan(0);
      expect(canonicalizeStageLabel(spec, { ...blank, silent_field: 1 }).unknownKeys).toEqual(["silent_field"]);
    });
  }
  for (const fixture of fixtures.real_campaign_cases) {
    test(`real campaign ${fixture.name}`, () => {
      const task = fixtures.synthetic.tasks.find((task) => task.source_name === fixture.source_name)!;
      const spec = normalizeStageSpec(task.spec);
      expect(trajectoryFromReview(fixture.review_label)).toEqual(fixture.canonical);
      expect(validateTrajectoryLabel(fixture.canonical, spec.trajectory!.task_definition)).toEqual(fixture.errors);
    });
  }
});


test("generic no-failure IDs use the task definition instead of the legacy none literal", () => {
  const task = fixtures.synthetic.tasks[0];
  const spec = JSON.parse(JSON.stringify(task.spec).replaceAll('"none"', '"no_failure"'));
  expect(normalizeStageSpec(spec).trajectory!.task_definition.successDefinition.noFailureModeId).toBe("no_failure");
});

test("generic structural gate detects nested fields the review adapter cannot expose", () => {
  const task = fixtures.synthetic.tasks[0]; const spec = normalizeStageSpec(task.spec);
  const fixture = task.cases.find((item) => item.name === "valid_success")!;
  const row = structuredClone(fixture.review_label!) as Record<string, unknown>;
  const action = (row.key_action_observations as Array<Record<string, unknown>>)[0];
  action.extra_unreviewable_field = "future-provider-data";
  expect(validateStageLabel(spec, row, 30).some((item) => item.code === "trajectory_shape")).toBe(true);
  delete action.extra_unreviewable_field; delete action.occurrences;
  expect(validateStageLabel(spec, row, 30).some((item) => item.code === "trajectory_shape")).toBe(true);
});

test("source-free episode identity does not round int64 episode IDs", () => {
  const spec = normalizeStageSpec(fixtures.synthetic.tasks[0].spec);
  const row = blankTrajectoryReview(spec.trajectory!, "org/test", 9007199254740993n);
  expect((row.trajectory_identity as Record<string, unknown>).sample_id).toBe("org/test#episode=9007199254740993");
});
