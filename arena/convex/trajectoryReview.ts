import {
  TRAJECTORY_LABEL_SCHEMA_VERSION,
  validateTrajectoryLabel,
  validateTrajectoryTaskDefinition,
  type TrajectoryTaskDefinition,
} from "./trajectoryContract";
import type { StageLabelRow, Violation } from "./stageConsistency";

export interface TrajectoryReviewSpec {
  adapter_version: "trajectory-review/v1";
  task_definition: TrajectoryTaskDefinition;
  task_definition_sha256: string;
  response_schema: Record<string, unknown>;
}

export const TRAJECTORY_REVIEW_FIELDS = [
  "trajectory_identity", "max_stage", "max_stage_id", "failure_mode", "primary_failure_time_s",
  "final_state", "attempt_count", "task_success", "stage_transitions", "key_action_observations",
  "failure_events", "confidence", "needs_human_review", "review_reasons", "notes",
] as const;

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

/** The lossless inverse of the Python trajectory-review/v1 adapter. */
export function trajectoryFromReview(row: StageLabelRow): Record<string, unknown> {
  const identity = record(row.trajectory_identity) ?? {};
  return {
    ...identity,
    max_stage: { stage_id: row.max_stage_id, stage_index: row.max_stage },
    primary_failure: { failure_mode_id: row.failure_mode, time_s: row.primary_failure_time_s },
    final_state_id: row.final_state,
    attempt_count: row.attempt_count,
    task_success: row.task_success,
    stage_transitions: row.stage_transitions,
    key_action_observations: row.key_action_observations,
    failure_events: row.failure_events,
    confidence: row.confidence,
    needs_human_review: row.needs_human_review,
    review_reasons: row.review_reasons,
    notes: row.notes,
  };
}

export function normalizeTrajectoryReviewSpec(value: unknown, task: string, taxonomyVersion: string): TrajectoryReviewSpec {
  const tag = record(value);
  if (!tag || tag.adapter_version !== "trajectory-review/v1") throw new Error("Unsupported trajectory review adapter");
  const errors = validateTrajectoryTaskDefinition(tag.task_definition);
  if (errors.length) throw new Error(`Malformed trajectory task definition: ${errors.join("; ")}`);
  const definition = tag.task_definition as TrajectoryTaskDefinition;
  if (definition.taskId !== task || `trajectory-review/v1/${definition.taxonomyVersion}` !== taxonomyVersion) {
    throw new Error("Trajectory task definition must match the registered task and taxonomy");
  }
  if (typeof tag.task_definition_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(tag.task_definition_sha256)) {
    throw new Error("Trajectory task definition needs a SHA-256 content pin");
  }
  if (!record(tag.response_schema)) throw new Error("Trajectory response_schema must be an object");
  // The portal validator checks IDs/relations. These fields are rendered by the
  // review form and must be correctly typed too, before a spec can be shown.
  for (const [name, items] of Object.entries({ stages: definition.stages, keyActions: definition.keyActions,
    failureModes: definition.failureModes, finalStates: definition.finalStates })) {
    for (const item of items) {
      if (typeof item.description !== "string" || !item.description.trim()) throw new Error(`${name} descriptions must be nonempty strings`);
    }
  }
  const stringArray = (value: unknown) => Array.isArray(value) && value.every((item) => typeof item === "string");
  for (const stage of definition.stages) {
    if (typeof stage.name !== "string" || !stringArray(stage.entryCriteria) ||
        (stage.exclusions !== undefined && !stringArray(stage.exclusions))) throw new Error("Trajectory stage instructions are malformed");
  }
  if (definition.keyActions.some((action) => typeof action.name !== "string") ||
      definition.decisionRules.some((rule) => typeof rule.rule !== "string")) throw new Error("Trajectory action or decision instructions are malformed");
  return tag as unknown as TrajectoryReviewSpec;
}

export function canonicalizeTrajectoryReview(row: StageLabelRow): { label: StageLabelRow; unknownKeys: string[] } {
  const allowed = new Set<string>(TRAJECTORY_REVIEW_FIELDS);
  return { label: { ...row }, unknownKeys: Object.keys(row).filter((key) => !allowed.has(key)).sort() };
}

export function validateTrajectoryReview(
  tag: TrajectoryReviewSpec, row: StageLabelRow, durationS?: number | null,
): Violation[] {
  const canonical = trajectoryFromReview(row);
  const errors = validateTrajectoryLabel(canonical, tag.task_definition).map((message) => ({
    code: "trajectory_contract", message, fields: ["trajectory"],
  }));
  errors.push(...trajectoryShapeViolations(tag, canonical));
  const identity = record(row.trajectory_identity);
  const expectedIdentityFields = ["schema_version", "task_id", "taxonomy_version", "sample_id"];
  if (!identity || Object.keys(identity).sort().join(",") !== expectedIdentityFields.sort().join(",")) {
    errors.push({ code: "trajectory_identity", message: "trajectory_identity must contain exactly schema_version, task_id, taxonomy_version, and sample_id", fields: ["trajectory_identity"] });
  }
  if (durationS == null || !Number.isFinite(durationS) || durationS <= 0) {
    errors.push({ code: "trajectory_duration", message: "A trajectory review requires a positive episode duration", fields: ["trajectory"] });
    return errors;
  }
  const bound = (time: unknown, path: string) => {
    if (typeof time === "number" && Number.isFinite(time) && (time < 0 || time > durationS + 0.01)) {
      errors.push({ code: "trajectory_time_bounds", message: `${path} must be within [0, ${durationS.toFixed(6)}] seconds (up to 0.010s positive endpoint representation rounding is allowed)`, fields: [path] });
    }
  };
  for (const name of ["stage_transitions", "failure_events"] as const) {
    if (Array.isArray(row[name])) row[name].forEach((event, i) => bound(record(event)?.time_s, `${name}[${i}].time_s`));
  }
  if (Array.isArray(row.key_action_observations)) {
    row.key_action_observations.forEach((action, i) => {
      const item = record(action);
      bound(item?.first_time_s, `key_action_observations[${i}].first_time_s`);
      if (Array.isArray(item?.occurrences)) item.occurrences.forEach((event, j) =>
        bound(record(event)?.time_s, `key_action_observations[${i}].occurrences[${j}].time_s`));
    });
  }
  bound(row.primary_failure_time_s, "primary_failure_time_s");
  return errors;
}

/** Source-free assisted annotation. Nothing is inherited from a human outcome. */
export function blankTrajectoryReview(tag: TrajectoryReviewSpec, repoId: string, episodeIndex: number | bigint): StageLabelRow {
  const task = tag.task_definition;
  return {
    trajectory_identity: { schema_version: TRAJECTORY_LABEL_SCHEMA_VERSION,
      task_id: task.taskId, taxonomy_version: task.taxonomyVersion, sample_id: `${repoId}#episode=${episodeIndex}` },
    max_stage: null, max_stage_id: "", failure_mode: "", primary_failure_time_s: null,
    final_state: "", attempt_count: 1, task_success: null,
    stage_transitions: [], failure_events: [],
    key_action_observations: task.keyActions.map((action) => ({
      action_id: action.id, occurred: false, first_time_s: null, occurrences: [],
    })),
    confidence: "medium", needs_human_review: false, review_reasons: [], notes: "",
  };
}

/** Source trajectory-task/v1 pins use sorted compact UTF-8 JSON, unlike the
 * Arena transport's typed digest. Task definitions contain only integer stage
 * indices; fractional numeric extensions need a versioned serialization rule. */
export async function trajectoryTaskDefinitionDigest(definition: TrajectoryTaskDefinition): Promise<string> {
  const encode = (value: unknown): string => {
    if (value === null || typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
    if (typeof value === "number") {
      if (!Number.isSafeInteger(value)) throw new Error("Trajectory task definition numeric fields must be safe integers");
      return String(value);
    }
    if (Array.isArray(value)) return `[${value.map(encode).join(",")}]`;
    const item = record(value);
    if (!item) throw new Error("Trajectory task definition must contain JSON values");
    const keys = Object.keys(item).sort((a, b) => {
      const x = Array.from(a, (char) => char.codePointAt(0)!); const y = Array.from(b, (char) => char.codePointAt(0)!);
      for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
      return x.length - y.length;
    });
    return `{${keys.map((key) => `${JSON.stringify(key)}:${encode(item[key])}`).join(",")}}`;
  };
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(encode(definition)));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Match the Python adapter's recursive exact-key gate. A future source field
 * cannot become confirmed gold while it is hidden from this review adapter. */
export function trajectoryShapeViolations(tag: TrajectoryReviewSpec, canonical: unknown): Violation[] {
  const violations: Violation[] = [];
  const walk = (value: unknown, schema: Record<string, unknown>, path: string) => {
    const fail = (message: string) => violations.push({ code: "trajectory_shape", message: `${path}: ${message}`, fields: [path] });
    if (schema.type === "OBJECT") {
      const item = record(value); const properties = record(schema.properties);
      if (!properties) throw new Error("Trajectory response schema object needs properties");
      if (!item) { fail("must be an object"); return; }
      const missing = Object.keys(properties).filter((key) => !(key in item));
      const extra = Object.keys(item).filter((key) => !(key in properties));
      if (missing.length || extra.length) fail(`missing fields=${missing.sort().join(", ")}; unmapped fields=${extra.sort().join(", ")}`);
      for (const [key, child] of Object.entries(properties)) {
        const childSchema = record(child); if (!childSchema) throw new Error("Malformed trajectory response schema property");
        if (key in item) walk(item[key], childSchema, `${path}.${key}`);
      }
    } else if (schema.type === "ARRAY") {
      if (!Array.isArray(value)) { fail("must be an array"); return; }
      const items = record(schema.items); if (!items) throw new Error("Trajectory response schema array needs items");
      value.forEach((item, i) => walk(item, items, `${path}[${i}]`));
    }
  };
  walk(canonical, tag.response_schema, "label");
  return violations;
}


export function validTrajectoryIdentity(tag: TrajectoryReviewSpec, value: unknown): boolean {
  const identity = record(value);
  return identity !== null && Object.keys(identity).sort().join(",") === "sample_id,schema_version,task_id,taxonomy_version" &&
    identity.schema_version === TRAJECTORY_LABEL_SCHEMA_VERSION && identity.task_id === tag.task_definition.taskId &&
    identity.taxonomy_version === tag.task_definition.taxonomyVersion && typeof identity.sample_id === "string" && identity.sample_id.trim().length > 0;
}
