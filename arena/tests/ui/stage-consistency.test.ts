import { describe, expect, test } from "bun:test";
import {
  asBool,
  timeValue,
  validateStageLabel,
  type ExportedStageSpec,
  type StageLabelRow,
} from "../../convex/stageConsistency";
import fixturesDoc from "../fixtures/stage-consistency-fixtures.json";

/**
 * The consistency interpreter is pinned by tests/fixtures/stage-consistency-fixtures.json:
 * per spec, label rows and the violation codes each must produce. The specs equal
 * what mulligan/tools/export_arena_task_specs.py exports for the registered tasks
 * (tests/real/test_export_arena_task_specs.py checks that). A divergence here is a
 * rule-interpretation bug in stageConsistency.ts.
 */

interface Fixture {
  name: string;
  row: StageLabelRow;
  episode_duration_s: number | null;
  expected_codes: string[];
}

interface TaskFixtures {
  spec: ExportedStageSpec;
  fixtures: Fixture[];
}

// Keys are `${task}@${taxonomy_version}` so a candidate taxonomy's vectors
// coexist with the live taxonomy's instead of clobbering them.
const tasks = Object.entries(fixturesDoc as unknown as Record<string, TaskFixtures>);

test("fixtures file covers the registered tasks", () => {
  const taskNames = tasks.map(([key]) => key.split("@")[0]);
  expect(taskNames.sort()).toEqual(["marker_d2", "routing_d2", "square_d2"]);
});

for (const [key, { spec, fixtures }] of tasks) {
  describe(`oracle fixtures: ${key}`, () => {
    test("spec shape is interpretable", () => {
      expect(spec.task).toBe(key.split("@")[0]);
      expect(spec.taxonomy_version).toBe(key.split("@")[1]);
      expect(spec.ladder.levels.length).toBe(spec.ladder.max_stage + 1);
      expect(spec.failure_modes).toContain("none");
      expect(spec.final_states).toContain(spec.success_final_state);
      const bools = spec.event_fields.filter((f) => f.kind === "boolean").map((f) => f.name);
      const times = spec.event_fields.filter((f) => f.kind === "number").map((f) => f.name);
      expect(bools).toEqual(spec.bool_fields);
      expect(times).toEqual(spec.time_fields);
    });

    for (const fx of fixtures) {
      test(fx.name, () => {
        const violations = validateStageLabel(spec, fx.row, fx.episode_duration_s);
        const codes = [...new Set(violations.map((v) => v.code))].sort();
        expect(codes).toEqual(fx.expected_codes);
      });
    }
  });
}

describe("coercion helpers", () => {
  test("asBool", () => {
    expect(asBool(true)).toBe(true);
    expect(asBool(0)).toBe(false);
    expect(asBool(1.0)).toBe(true);
    expect(asBool(NaN)).toBeNull();
    expect(asBool("Yes")).toBe(true);
    expect(asBool(" false ")).toBe(false);
    expect(asBool("")).toBeNull();
    expect(asBool("nan")).toBeNull();
    expect(asBool("none")).toBeNull();
    expect(asBool(null)).toBeNull();
    expect(asBool(undefined)).toBeNull();
    expect(asBool("garbage")).toBeNull();
  });

  test("timeValue", () => {
    expect(timeValue(1.25)).toBe(1.25);
    expect(timeValue(0)).toBe(0);
    expect(timeValue(true)).toBeNull();
    expect(timeValue(NaN)).toBeNull();
    expect(timeValue("2.5")).toBe(2.5);
    expect(timeValue("")).toBeNull();
    expect(timeValue("null")).toBeNull();
    expect(timeValue("NaN")).toBeNull();
    expect(timeValue(undefined)).toBeNull();
  });
});
