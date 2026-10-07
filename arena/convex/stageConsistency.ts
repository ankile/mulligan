import { canonicalizeTrajectoryReview, validateTrajectoryReview, type TrajectoryReviewSpec } from "./trajectoryReview";

/**
 * Interpreter for the stage-label consistency rules, driven ENTIRELY by the
 * exported spec data (stageTaskSpecs.spec, written by
 * mulligan/tools/export_arena_task_specs.py) — no task names or field names
 * appear here. Shared by convex/stageReviews.ts (server gate on
 * confirmed/corrected saves) and the stage-review form (instant feedback), so
 * there is exactly one implementation. tests/fixtures/stage-consistency-fixtures.json
 * pins it: per spec, label rows and the violation codes each must produce.
 *
 * Violation CODES are the contract (the fixtures compare codes); messages are
 * human framing.
 */

export interface StageSpecLevel {
  sid: number;
  text: string;
  gate_field: string | null;
  gate_time_field: string | null;
  gate_any_of: string[];
  gate_all_of: string[];
}

export interface ExportedStageSpec {
  trajectory?: TrajectoryReviewSpec;
  task: string;
  lifecycle_task: string;
  taxonomy_version: string;
  taxonomy_hash: string;
  ladder: {
    header: string;
    success_level: number;
    max_stage: number;
    levels: StageSpecLevel[];
  };
  failure_modes: string[];
  final_states: string[];
  success_final_state: string;
  released_field: string;
  stage_field: string;
  final_state_field: string;
  failure_mode_field: string;
  event_fields: { name: string; kind: string; description: string }[];
  bool_fields: string[];
  time_fields: string[];
  editable_fields: string[];
  constraints: {
    failure_mode_forbidden_stages: Record<string, number[]>;
    historical_event_failure_modes: Record<string, string>;
    final_state_requires_gates: Record<string, string[]>;
    final_state_requires_min_stage: Record<string, number>;
  };
  fps: number;
}

export interface Violation {
  code: string;
  message: string;
  fields: string[];
}

export type StageLabelRow = Record<string, unknown>;

// Upper-bound tolerance for the time-in-clip check (consensus rounds event
// times to 2 decimals; a last-frame time can round up half a unit).
const TIME_ROUND_TOL_S = 5e-3;

const NO_FAILURE = "none";

/** Strict bool, else null when missing/unparseable. */
export function asBool(v: unknown): boolean | null {
  if (typeof v === "boolean") return v;
  if (v === null || v === undefined) return null;
  if (typeof v === "number") {
    if (Number.isNaN(v)) return null;
    return Boolean(v);
  }
  const s = String(v).trim().toLowerCase();
  if (s === "true" || s === "1" || s === "yes") return true;
  if (s === "false" || s === "0" || s === "no") return false;
  return null; // includes "", "nan", "none", and anything unrecognized
}

// Strict decimal syntax for STRING-typed timestamps/stages: plain ASCII
// decimals + exponent; no "inf", underscores, hex, or non-ASCII digits.
const DECIMAL_RE = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
const INT_RE = /^[+-]?\d+$/;

/** Prototype-safe record lookup: constraint maps are plain JSON objects, so a
 * key like "toString"/"constructor" must resolve to undefined, not to
 * Object.prototype members (which crash downstream .includes/iteration). */
function lookup<T>(map: Record<string, T>, key: string): T | undefined {
  return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined;
}

/** A present timestamp as number, else null. */
export function timeValue(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "boolean") return null;
  if (typeof v === "number") return Number.isNaN(v) ? null : v;
  const s = String(v).trim().toLowerCase();
  if (s === "" || s === "nan" || s === "none" || s === "null") return null;
  if (!DECIMAL_RE.test(s)) return null;
  return Number(s);
}

/** int(row[stage]) semantics: bools count, floats truncate (±inf/NaN
 * unparseable), strings parse the strict ASCII-integer grammar. */
function parseStage(v: unknown): number | null {
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "number") return Number.isFinite(v) ? Math.trunc(v) : null;
  if (typeof v === "string") {
    const s = v.trim();
    if (INT_RE.test(s)) return parseInt(s, 10);
    return null;
  }
  return null;
}

function gateBools(lvl: StageSpecLevel): string[] {
  if (lvl.gate_field != null) return [lvl.gate_field];
  return lvl.gate_any_of.length ? lvl.gate_any_of : lvl.gate_all_of;
}

function gateReduce(lvl: StageSpecLevel): "single" | "any" | "all" {
  if (lvl.gate_any_of.length) return "any";
  if (lvl.gate_all_of.length) return "all";
  return "single";
}

function gated(lvl: StageSpecLevel): boolean {
  return lvl.gate_field != null || lvl.gate_any_of.length > 0 || lvl.gate_all_of.length > 0;
}

function gateSatisfied(lvl: StageSpecLevel, vals: Record<string, boolean>): boolean {
  const bools = gateBools(lvl).map((f) => vals[f]);
  const reduce = gateReduce(lvl);
  if (reduce === "all") return bools.every(Boolean);
  if (reduce === "any") return bools.some(Boolean);
  return bools[0];
}

function gateTimeFields(lvl: StageSpecLevel): string[] {
  if (lvl.gate_field != null) {
    return lvl.gate_time_field != null ? [lvl.gate_time_field] : [];
  }
  return gateBools(lvl).map((f) => `${f}_time_s`);
}

/** The single timestamp representing when a rung's gate is satisfied: any ->
 * min of present constituent times, all -> max, single -> its own. */
function gateOrderTime(lvl: StageSpecLevel, times: Record<string, number | null>): number | null {
  const present = gateTimeFields(lvl)
    .map((f) => times[f])
    .filter((t): t is number => t !== null && t !== undefined);
  if (!present.length) return null;
  return gateReduce(lvl) === "all" ? Math.max(...present) : Math.min(...present);
}

/** Values whose absence is deliberate (missing cell), as opposed to junk. */
function isMissingish(v: unknown): boolean {
  if (v === null || v === undefined) return true;
  if (typeof v === "number") return Number.isNaN(v);
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    return s === "" || s === "nan" || s === "none" || s === "null";
  }
  return false;
}

/**
 * Canonicalize a label row against the spec's field kinds BEFORE storage:
 * stage -> number, bools -> boolean, times -> number, enums/notes -> trimmed
 * string; missing-ish values are OMITTED. Uncoercible junk is KEPT RAW so the
 * validator flags it instead of a silent drop. Keys outside `editable_fields`
 * are returned in `unknownKeys` — the caller must reject them (they would
 * otherwise ride into gold, and stringly-typed values would manufacture false
 * reviewer disagreements against web-typed rows).
 */
export function canonicalizeStageLabel(
  spec: ExportedStageSpec,
  label: StageLabelRow
): { label: StageLabelRow; unknownKeys: string[] } {
  if (spec.trajectory) return canonicalizeTrajectoryReview(label);
  const boolSet = new Set(spec.bool_fields);
  const timeSet = new Set(spec.time_fields);
  const allowed = new Set(spec.editable_fields);
  const out: StageLabelRow = {};
  const unknownKeys: string[] = [];
  for (const [key, raw] of Object.entries(label)) {
    if (!allowed.has(key)) {
      unknownKeys.push(key);
      continue;
    }
    if (key === "notes") {
      // Notes are free text: only true absence is missing — a reviewer's note
      // reading exactly "none"/"nan" must not be silently dropped.
      if (raw === null || raw === undefined || raw === "") continue;
      out[key] = String(raw);
      continue;
    }
    if (isMissingish(raw)) continue;
    if (key === spec.stage_field) {
      const stage = parseStage(raw);
      out[key] = stage !== null ? stage : raw;
    } else if (boolSet.has(key)) {
      const b = asBool(raw);
      out[key] = b !== null ? b : raw;
    } else if (timeSet.has(key)) {
      const t = timeValue(raw);
      out[key] = t !== null ? t : raw;
    } else {
      out[key] = String(raw).trim();
    }
  }
  return { label: out, unknownKeys: unknownKeys.sort() };
}

export function validateStageLabel(
  spec: ExportedStageSpec,
  row: StageLabelRow,
  episodeDurationS?: number | null
): Violation[] {
  if (spec.trajectory) return validateTrajectoryReview(spec.trajectory, row, episodeDurationS);
  const out: Violation[] = [];
  const ladder = spec.ladder;
  const timeFieldSet = new Set(spec.time_fields);

  if (!(spec.stage_field in row)) {
    return [
      {
        code: "missing_stage",
        message: `row has no ${spec.stage_field}`,
        fields: [spec.stage_field],
      },
    ];
  }
  const maxStage = parseStage(row[spec.stage_field]);
  if (maxStage === null) {
    return [
      {
        code: "unparseable_stage",
        message: `${spec.stage_field}=${JSON.stringify(row[spec.stage_field])} is not an integer`,
        fields: [spec.stage_field],
      },
    ];
  }
  if (!(0 <= maxStage && maxStage <= ladder.success_level)) {
    out.push({
      code: "stage_out_of_range",
      message: `${spec.stage_field}=${maxStage} outside ladder [0, ${ladder.success_level}]`,
      fields: [spec.stage_field],
    });
  }

  const gatedLevels = ladder.levels.filter(gated);

  // The achieved-then-lost exception feeds the GATE checks too: max_stage is a
  // HIGH-WATER mark, so a gate bool achieved and then lost (terminal false,
  // historical event time retained, matching failure mode active) still counts
  // as reached.
  const failureMode = String(row[spec.failure_mode_field] ?? "").trim();
  const historicalBool: string | undefined = lookup(
    spec.constraints.historical_event_failure_modes,
    failureMode
  );
  const achieved = (field: string, terminal: boolean): boolean =>
    terminal ||
    (field === historicalBool && timeValue(row[`${field}_time_s`]) !== null);
  // The high-water reading applies to SINGLE and ANY gates only: an ALL gate
  // asserts a JOINT state, which a lost constituent falsifies at the terminal
  // snapshot (reached-then-lost joint states are encoded via the mode's
  // forbidden-stage set instead).
  const gateTrueFor = (
    lvl: StageSpecLevel,
    vals: Record<string, boolean>
  ): boolean => {
    if (gateReduce(lvl) === "all") return gateSatisfied(lvl, vals);
    return gateSatisfied(
      lvl,
      Object.fromEntries(
        Object.entries(vals).map(([f, v]) => [f, achieved(f, v)])
      )
    );
  };

  // --- gate biconditionals: max_stage >= sid  <=>  gate predicate  <=>  time present ---
  for (const lvl of gatedLevels) {
    const fields = gateBools(lvl);
    const vals: Record<string, boolean | null> = {};
    for (const f of fields) vals[f] = asBool(row[f]);
    const missing = fields.filter((f) => vals[f] === null);
    if (missing.length) {
      for (const f of missing) {
        out.push({
          code: "gate_bool_missing",
          message: `gate bool ${f} (S${lvl.sid}) is missing/unparseable`,
          fields: [f],
        });
      }
      continue;
    }
    const gateTrue = gateTrueFor(lvl, vals as Record<string, boolean>);
    const reached = maxStage >= lvl.sid;
    if (gateTrue !== reached) {
      out.push({
        code: "gate_stage_mismatch",
        message:
          `gate S${lvl.sid} (${gateReduce(lvl)} of ${fields.join(", ")})=${gateTrue} but ` +
          `${spec.stage_field}=${maxStage} (${reached ? ">=" : "<"} S${lvl.sid}) — must agree`,
        fields: [...fields, spec.stage_field],
      });
    }
    const tfield = lvl.gate_field != null ? lvl.gate_time_field : null;
    if (tfield && timeFieldSet.has(tfield)) {
      const b = vals[fields[0]];
      const present = timeValue(row[tfield]) !== null;
      const retainedHistorical =
        fields[0] === historicalBool && b === false && present;
      if (b !== present && !retainedHistorical) {
        out.push({
          code: "gate_time_mismatch",
          message:
            `${fields[0]}=${b} but ${tfield} is ${present ? "present" : "absent"} — ` +
            "a seated/grasped gate needs its timestamp (and vice versa)",
          fields: [fields[0], tfield],
        });
      }
    }
  }

  // --- every boolean event paired with a *_time_s: bool <=> time present ---
  // (failureMode / historicalBool computed above the gate loop, which shares
  // the achieved-then-lost exception.)
  const singleGateFields = new Set(
    gatedLevels.filter((l) => l.gate_field != null).map((l) => l.gate_field as string)
  );
  for (const bf of spec.bool_fields) {
    if (singleGateFields.has(bf)) continue; // covered by the single-gate check above
    const tf = `${bf}_time_s`;
    if (!timeFieldSet.has(tf)) continue;
    const b = asBool(row[bf]);
    if (b === null) continue; // a non-gate optional bool may legitimately be absent
    const present = timeValue(row[tf]) !== null;
    const retainedHistoricalTime = bf === historicalBool && b === false && present;
    if (b !== present && !retainedHistoricalTime) {
      out.push({
        code: "bool_time_mismatch",
        message: `${bf}=${b} but ${tf} is ${present ? "present" : "absent"} — must agree`,
        fields: [bf, tf],
      });
    }
  }
  if (historicalBool !== undefined) {
    const historicalTime = timeFieldSet.has(`${historicalBool}_time_s`)
      ? `${historicalBool}_time_s`
      : null;
    const historicalValue = asBool(row[historicalBool]);
    if (historicalValue !== false) {
      out.push({
        code: "historical_event_terminal_bool",
        message:
          `${failureMode} requires terminal ${historicalBool}=false, got ` +
          JSON.stringify(row[historicalBool]),
        fields: [spec.failure_mode_field, historicalBool],
      });
    }
    if (historicalTime === null || timeValue(row[historicalTime]) === null) {
      out.push({
        code: "historical_event_time_missing",
        message:
          `${failureMode} means ${historicalBool} occurred earlier, so ` +
          `${historicalTime ?? historicalBool + "_time_s"} must be present`,
        fields: [spec.failure_mode_field, historicalTime ?? historicalBool],
      });
    }
  }

  // --- gate timestamps must run in ladder order ---
  const allTimes: Record<string, number | null> = {};
  for (const tf of spec.time_fields) allTimes[tf] = timeValue(row[tf]);
  let prevT: number | null = null;
  let prevField: string | null = null;
  for (const lvl of gatedLevels) {
    const t = gateOrderTime(lvl, allTimes);
    if (t === null) continue;
    const fieldLabel = gateTimeFields(lvl).join(", ");
    if (prevT !== null && t < prevT) {
      out.push({
        code: "gate_time_order",
        message:
          `S${lvl.sid} gate time ${t} precedes ${prevField}=${prevT} — ` +
          "later gates cannot occur earlier",
        fields: [prevField as string, fieldLabel],
      });
    }
    prevT = t;
    prevField = fieldLabel;
  }

  // --- success coherence + calibrated failure mode ---
  const fm = row[spec.failure_mode_field];
  if (fm !== null && fm !== undefined) {
    const isSuccess = maxStage === ladder.success_level;
    const noFailure = String(fm).trim().toLowerCase() === NO_FAILURE;
    if (isSuccess !== noFailure) {
      out.push({
        code: "success_failure_mismatch",
        message:
          `${spec.stage_field}=${maxStage} (success=${isSuccess}) but ` +
          `${spec.failure_mode_field}=${JSON.stringify(fm)} (none=${noFailure}) — must agree`,
        fields: [spec.stage_field, spec.failure_mode_field],
      });
    }
    if (isSuccess) {
      const fs = row[spec.final_state_field];
      if (fs !== null && fs !== undefined && String(fs) !== spec.success_final_state) {
        out.push({
          code: "success_final_state",
          message:
            `success (${spec.stage_field}=${maxStage}) requires ` +
            `${spec.final_state_field}=${spec.success_final_state}, got ${JSON.stringify(fs)}`,
          fields: [spec.final_state_field],
        });
      }
    }
    const fmS = String(fm).trim();
    if (fmS && !spec.failure_modes.includes(fmS)) {
      out.push({
        code: "failure_mode_unknown",
        message: `${spec.failure_mode_field}=${JSON.stringify(fmS)} is not in the task taxonomy`,
        fields: [spec.failure_mode_field],
      });
    }
    const banned = lookup(spec.constraints.failure_mode_forbidden_stages, fmS);
    if (banned && banned.includes(maxStage)) {
      out.push({
        code: "failure_mode_stage_conflict",
        message:
          `${spec.failure_mode_field}=${fmS} is incompatible with ` +
          `${spec.stage_field}=${maxStage}`,
        fields: [spec.failure_mode_field, spec.stage_field],
      });
    }
  }

  // --- final_state coherence ---
  const fs = row[spec.final_state_field];
  if (fs !== null && fs !== undefined) {
    const fsS = String(fs).trim();
    if (fsS && !spec.final_states.includes(fsS)) {
      // Mirror of failure_mode_unknown: a typo'd final_state must not pass
      // both validators into gold.
      out.push({
        code: "final_state_unknown",
        message: `${spec.final_state_field}=${JSON.stringify(fsS)} is not in the task taxonomy`,
        fields: [spec.final_state_field],
      });
    }
    for (const gate of lookup(spec.constraints.final_state_requires_gates, fsS) ?? []) {
      if (asBool(row[gate]) !== true) {
        out.push({
          code: "final_state_seat_conflict",
          message:
            `${spec.final_state_field}=${fsS} places the object in the seat, so ` +
            `${gate} must be true (got ${JSON.stringify(row[gate])})`,
          fields: [spec.final_state_field, gate],
        });
      }
    }
    const minStage = lookup(spec.constraints.final_state_requires_min_stage, fsS);
    if (minStage !== undefined && maxStage < minStage) {
      out.push({
        code: "final_state_stage_conflict",
        message:
          `${spec.final_state_field}=${fsS} means the target was reached, so ` +
          `${spec.stage_field} must be >= ${minStage} (got ${maxStage})`,
        fields: [spec.final_state_field, spec.stage_field],
      });
    }
  }

  // --- time bounds ---
  if (episodeDurationS !== null && episodeDurationS !== undefined) {
    for (const tf of spec.time_fields) {
      const t = timeValue(row[tf]);
      if (t !== null && !(0.0 <= t && t <= episodeDurationS + TIME_ROUND_TOL_S)) {
        out.push({
          code: "time_out_of_range",
          message: `${tf}=${t} outside episode [0, ${episodeDurationS.toFixed(2)}]s`,
          fields: [tf],
        });
      }
    }
  }

  return out;
}
