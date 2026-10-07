import { normalizeTrajectoryReviewSpec } from "../../convex/trajectoryReview";
import type { ExportedStageSpec } from "../../convex/stageConsistency";

/**
 * Loud shape validation for a fetched stageTaskSpecs row's `spec` payload.
 *
 * The payload is produced by mulligan/tools/export_arena_task_specs.py and
 * stored opaque in Convex; numbers arrive as plain
 * float64. This narrows `unknown` to ExportedStageSpec and throws on anything
 * malformed — a partial spec must never render a partial form (a missing
 * constraint map would silently disable its rules).
 */
export function normalizeStageSpec(raw: unknown): ExportedStageSpec {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error("stage spec payload is not an object");
  }
  const spec = raw as Record<string, unknown>;
  const fail = (detail: string): never => {
    throw new Error(`malformed stage spec (${String(spec.task)}): ${detail}`);
  };
  const str = (key: string): string => {
    if (typeof spec[key] !== "string" || !spec[key]) fail(`missing string ${key}`);
    return spec[key] as string;
  };
  const strArray = (value: unknown, what: string): string[] => {
    if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) {
      fail(`${what} is not a string array`);
    }
    return value as string[];
  };

  const ladder = spec.ladder as Record<string, unknown> | undefined;
  if (!ladder || !Array.isArray(ladder.levels) || ladder.levels.length === 0) {
    fail("ladder.levels missing/empty");
  }
  const levels = (ladder!.levels as Record<string, unknown>[]).map((lvl, i) => {
    if (typeof lvl.sid !== "number" || lvl.sid !== i) fail(`level ${i} sid mismatch`);
    if (typeof lvl.text !== "string") fail(`level ${i} text missing`);
    return {
      sid: lvl.sid as number,
      text: lvl.text as string,
      gate_field: (lvl.gate_field ?? null) as string | null,
      gate_time_field: (lvl.gate_time_field ?? null) as string | null,
      gate_any_of: strArray(lvl.gate_any_of ?? [], `level ${i} gate_any_of`),
      gate_all_of: strArray(lvl.gate_all_of ?? [], `level ${i} gate_all_of`),
    };
  });
  if (typeof ladder!.success_level !== "number") fail("ladder.success_level missing");
  if (typeof ladder!.max_stage !== "number") fail("ladder.max_stage missing");

  const constraints = spec.constraints as Record<string, unknown> | undefined;
  for (const key of [
    "failure_mode_forbidden_stages",
    "historical_event_failure_modes",
    "final_state_requires_gates",
    "final_state_requires_min_stage",
  ]) {
    if (!constraints || typeof constraints[key] !== "object" || constraints[key] === null) {
      fail(`constraints.${key} missing`);
    }
  }
  // Value types too — a mistyped constraint value would crash the validator at
  // review time (banned.includes on a number) or silently change semantics.
  const con = constraints as Record<string, Record<string, unknown>>;
  for (const [mode, stages] of Object.entries(con.failure_mode_forbidden_stages)) {
    if (!Array.isArray(stages) || stages.some((n) => typeof n !== "number")) {
      fail(`forbidden stages for ${mode} is not a number array`);
    }
  }
  for (const [mode, bf] of Object.entries(con.historical_event_failure_modes)) {
    if (typeof bf !== "string") fail(`historical bool for ${mode} is not a string`);
  }
  for (const [state, gates] of Object.entries(con.final_state_requires_gates)) {
    if (!Array.isArray(gates) || gates.some((g) => typeof g !== "string")) {
      fail(`gate requirements for ${state} is not a string array`);
    }
  }
  for (const [state, minStage] of Object.entries(con.final_state_requires_min_stage)) {
    if (typeof minStage !== "number") fail(`min stage for ${state} is not a number`);
  }

  const eventFields = spec.event_fields;
  if (
    !Array.isArray(eventFields) ||
    eventFields.some(
      (f) =>
        typeof f !== "object" ||
        f === null ||
        typeof (f as Record<string, unknown>).name !== "string" ||
        typeof (f as Record<string, unknown>).kind !== "string"
    )
  ) {
    fail("event_fields malformed");
  }
  if (typeof spec.fps !== "number" || !(spec.fps > 0)) fail("fps missing/invalid");

  const out: ExportedStageSpec = {
    task: str("task"),
    lifecycle_task: str("lifecycle_task"),
    taxonomy_version: str("taxonomy_version"),
    taxonomy_hash: str("taxonomy_hash"),
    ladder: {
      header: String(ladder!.header ?? ""),
      success_level: ladder!.success_level as number,
      max_stage: ladder!.max_stage as number,
      levels,
    },
    failure_modes: strArray(spec.failure_modes, "failure_modes"),
    final_states: strArray(spec.final_states, "final_states"),
    success_final_state: str("success_final_state"),
    released_field: str("released_field"),
    stage_field: str("stage_field"),
    final_state_field: str("final_state_field"),
    failure_mode_field: str("failure_mode_field"),
    event_fields: (eventFields as { name: string; kind: string; description?: string }[]).map(
      (f) => ({ name: f.name, kind: f.kind, description: f.description ?? "" })
    ),
    bool_fields: strArray(spec.bool_fields, "bool_fields"),
    time_fields: strArray(spec.time_fields, "time_fields"),
    editable_fields: strArray(spec.editable_fields, "editable_fields"),
    constraints: spec.constraints as ExportedStageSpec["constraints"],
    fps: spec.fps as number,
  };
  if (spec.trajectory !== undefined) {
    out.trajectory = normalizeTrajectoryReviewSpec(spec.trajectory, out.task, out.taxonomy_version);
  }
  if (!out.trajectory && !out.failure_modes.includes("none")) fail("failure_modes lacks 'none'");
  if (!out.final_states.includes(out.success_final_state)) {
    fail("success_final_state not in final_states");
  }
  return out;
}
