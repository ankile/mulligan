/**
 * Canonical outcome reconciliation for real eval results.json payloads.
 * Port of mulligan/real/eval/outcome_results.py (payload/file layers; the
 * frame classifier lives in frames.ts).
 */

import { dumpsIndent2Sorted, dumpsSorted } from "./pyjson";
import type { Json } from "./pyjson";
import { OUTCOME_NAMES } from "./progress";
import type { ProgressRecord } from "./progress";
import type { FrameOutcome } from "./frames";

type Payload = Record<string, unknown>;
type Rollout = Record<string, unknown>;

export interface Reconciliation {
  overrides_file: string;
  episodes_reviewed: number;
  outcome_class_changes: number;
  num_steps_patches: number;
  success_flips: Array<{ episode_index: number; old: string; new: string }>;
}

const RESUMABLE_DYNAMIC_KEYS = new Set([
  "timestamp",
  "summary",
  "rollouts",
  "arena_submitted_round_indices",
  "arena_submitted_rollouts",
  "phase_stops",
]);

function uniqueIds(rows: Rollout[], key: string, what: string): Set<number> {
  const seen = new Set<number>();
  const duplicates = new Set<number>();
  for (const row of rows) {
    const value = Number(row[key]);
    if (seen.has(value)) duplicates.add(value);
    seen.add(value);
  }
  if (duplicates.size > 0) {
    throw new Error(`${what} has duplicate ${key} values: ${[...duplicates].sort((a, b) => a - b)}`);
  }
  return seen;
}

function validateSummaryPolicyIds(payload: Payload): void {
  const summary = payload.summary as Rollout[];
  const rollouts = payload.rollouts as Rollout[];
  const summaryIds = uniqueIds(summary, "policy_id", "results summary");
  const rolloutIds = new Set(rollouts.map((r) => Number(r.policy_id)));
  const missingSummary = [...rolloutIds].filter((id) => !summaryIds.has(id)).sort((a, b) => a - b);
  const missingRollouts = [...summaryIds].filter((id) => !rolloutIds.has(id)).sort((a, b) => a - b);
  const errors: string[] = [];
  if (missingSummary.length) errors.push(`missing summary rows for rollout policy_id values ${missingSummary}`);
  if (missingRollouts.length) errors.push(`summary policy_id values with no rollouts ${missingRollouts}`);
  if (errors.length) {
    throw new Error("results.json summary policy ids disagree with rollouts: " + errors.join("; "));
  }
}

export function rolloutsByEpisode(payload: Payload): Map<number, Rollout> {
  const rollouts = payload.rollouts as Rollout[];
  uniqueIds(rollouts, "episode_index", "results rollouts");
  return new Map(rollouts.map((r) => [Number(r.episode_index), r]));
}

/**
 * live_subtask_frames_from_results: per-episode LIVE (eval-time) subtask marks
 * from a results.json payload. rollout_episode records the operator's live 'g'
 * press as `subtask_frames` on the rollout row and finalize_episode_data writes
 * the same reward=1.0 / done=0 spike apply writes, so an episode nobody has
 * reviewed yet legitimately carries a mid-episode spike. Older results files
 * have no key (no live marks).
 */
export function liveSubtaskFramesByEpisode(payload: Payload | null): Map<number, number[]> {
  const out = new Map<number, number[]>();
  if (payload === null) return out;
  for (const [ep, rollout] of rolloutsByEpisode(payload)) {
    const frames = (rollout.subtask_frames as unknown[] | undefined) ?? [];
    if (frames.length > 0) {
      out.set(ep, [...new Set(frames.map((f) => Number(f)))].sort((a, b) => a - b));
    }
  }
  return out;
}

/**
 * subtask_frames_for_validation: subtask frames the frame validator must
 * tolerate, per episode — live eval-time marks overridden per episode by the
 * review record. A reviewed episode (present in changed_episodes) is
 * authoritative even with no marks: apply zeroed its pre-outcome reward and
 * wrote exactly the recorded spikes. An UNREVIEWED episode keeps the live spike.
 * Without the live fallback a partial review of an in-progress eval could never
 * apply.
 */
export function subtaskFramesForValidation(
  progress: ProgressRecord,
  payload: Payload | null
): Map<number, number[]> {
  const out = liveSubtaskFramesByEpisode(payload);
  for (const [epStr, entry] of Object.entries(progress.changed_episodes)) {
    const epIdx = parseInt(epStr, 10);
    const frames = [...new Set((entry.subtask_frames ?? []).map((f) => Number(f)))].sort(
      (a, b) => a - b
    );
    if (frames.length > 0) out.set(epIdx, frames);
    else out.delete(epIdx);
  }
  return out;
}

export function recomputeSummaryFromRollouts(payload: Payload): void {
  validateSummaryPolicyIds(payload);
  const rollouts = payload.rollouts as Rollout[];
  for (const row of payload.summary as Rollout[]) {
    const policyId = Number(row.policy_id);
    const sub = rollouts.filter((r) => Number(r.policy_id) === policyId);
    const successes = sub.filter((r) => String(r.outcome) === "success").length;
    if ("num_rounds" in row) row.num_rounds = sub.length;
    row.successes = successes;
    row.failures = sub.length - successes;
    row.success_rate = sub.length ? successes / sub.length : 0.0;
  }
}

export function validateSummaryMatchesRollouts(payload: Payload): void {
  validateSummaryPolicyIds(payload);
  const rollouts = payload.rollouts as Rollout[];
  const errors: string[] = [];
  for (const row of payload.summary as Rollout[]) {
    const policyId = Number(row.policy_id);
    const sub = rollouts.filter((r) => Number(r.policy_id) === policyId);
    const successes = sub.filter((r) => String(r.outcome) === "success").length;
    if ("num_rounds" in row && Number(row.num_rounds) !== sub.length) {
      errors.push(`policy_id ${policyId}: summary num_rounds=${row.num_rounds} but rollouts=${sub.length}`);
    }
    if (Number(row.successes) !== successes) {
      errors.push(`policy_id ${policyId}: summary successes=${row.successes} but rollouts=${successes}`);
    }
    if (Number(row.failures) !== sub.length - successes) {
      errors.push(
        `policy_id ${policyId}: summary failures=${row.failures} but rollouts=${sub.length - successes}`
      );
    }
  }
  if (errors.length) {
    throw new Error("results.json summary disagrees with rollouts:\n" + errors.slice(0, 20).join("\n"));
  }
}

/** apply_outcome_edit_record: patch rollout outcomes/num_steps in place. */
export function applyOutcomeEditRecord(
  payload: Payload,
  record: ProgressRecord,
  overridesFilename: string
): Reconciliation {
  const existingReconciliation = payload._outcome_edit_reconciliation;
  const changed = new Map<number, Record<string, unknown>>(
    Object.entries(record.changed_episodes).map(([k, v]) => [parseInt(k, 10), v as never])
  );
  const byEp = rolloutsByEpisode(payload);
  const unknown = [...changed.keys()].filter((ep) => !byEp.has(ep)).sort((a, b) => a - b);
  if (unknown.length) {
    throw new Error(`${overridesFilename} references episodes not in results.json: ${unknown}`);
  }

  let classChanges = 0;
  let stepPatches = 0;
  const flips: Reconciliation["success_flips"] = [];
  for (const ep of [...changed.keys()].sort((a, b) => a - b)) {
    const entry = changed.get(ep)!;
    const newOutcome = String(entry.new_outcome);
    if (!(OUTCOME_NAMES as readonly string[]).includes(newOutcome)) {
      throw new Error(`episode ${ep}: unknown edited outcome ${JSON.stringify(newOutcome)}`);
    }
    const rollout = byEp.get(ep)!;
    const old = String(rollout.outcome);
    if (old !== newOutcome) {
      classChanges += 1;
      if ((old === "success") !== (newOutcome === "success")) {
        flips.push({ episode_index: ep, old, new: newOutcome });
      }
      rollout.outcome = newOutcome;
      if ("success" in rollout) rollout.success = newOutcome === "success";
    }
    const frame = entry.outcome_frame;
    if (
      frame !== undefined &&
      frame !== null &&
      (newOutcome === "success" || newOutcome === "failure" || Boolean(entry.soft_truncate))
    ) {
      const steps = Number(frame) + 1;
      if (steps < 1) throw new Error(`episode ${ep}: invalid outcome_frame ${frame}`);
      if (steps !== Number(rollout.num_steps)) {
        rollout.num_steps = steps;
        stepPatches += 1;
      }
    }
  }

  recomputeSummaryFromRollouts(payload);
  const reconciliation: Reconciliation = {
    overrides_file: overridesFilename,
    episodes_reviewed: changed.size,
    outcome_class_changes: classChanges,
    num_steps_patches: stepPatches,
    success_flips: flips,
  };
  if (existingReconciliation !== undefined && classChanges === 0 && stepPatches === 0) {
    payload._outcome_edit_reconciliation = existingReconciliation;
    return existingReconciliation as unknown as Reconciliation;
  }
  payload._outcome_edit_reconciliation = reconciliation as unknown as Json;
  return reconciliation;
}

export function validateResultsAgainstFrameOutcomes(
  payload: Payload,
  frameOutcomes: Map<number, FrameOutcome>
): void {
  const errors: string[] = [];
  const byEp = rolloutsByEpisode(payload);
  const resultEpisodes = new Set(byEp.keys());
  const frameEpisodes = new Set(frameOutcomes.keys());
  const missingFrames = [...resultEpisodes].filter((e) => !frameEpisodes.has(e)).sort((a, b) => a - b);
  const missingResults = [...frameEpisodes].filter((e) => !resultEpisodes.has(e)).sort((a, b) => a - b);
  if (missingFrames.length) {
    errors.push(`episodes present in results.json but missing from frame data: ${missingFrames.slice(0, 30)}`);
  }
  if (missingResults.length) {
    errors.push(`episodes present in frame data but missing from results.json: ${missingResults.slice(0, 30)}`);
  }
  for (const ep of [...byEp.keys()].sort((a, b) => a - b)) {
    const rollout = byEp.get(ep)!;
    const frame = frameOutcomes.get(ep);
    if (!frame) continue;
    const outcome = String(rollout.outcome);
    if (outcome !== frame.outcome) {
      errors.push(`episode ${ep}: results outcome=${JSON.stringify(outcome)} but frame/meta outcome=${JSON.stringify(frame.outcome)}`);
    }
    if (Number(rollout.num_steps) !== frame.expectedNumSteps) {
      errors.push(`episode ${ep}: results num_steps=${rollout.num_steps} but frame/meta expected_num_steps=${frame.expectedNumSteps}`);
    }
  }
  if (errors.length) {
    throw new Error(
      `results.json disagrees with frame/meta outcomes:\n${errors.slice(0, 30).join("\n")}` +
        (errors.length > 30 ? `\n... ${errors.length - 30} more` : "")
    );
  }
}

/**
 * _resumed_args_compatible: `args` may gain/lose CLI keys across a resume (the
 * eval CLI can change between the sessions of one eval); every key present in
 * BOTH must match. Non-object args fall back to strict equality.
 */
function resumedArgsCompatible(previous: unknown, current: unknown): boolean {
  const isObj = (v: unknown): v is Record<string, unknown> =>
    typeof v === "object" && v !== null && !Array.isArray(v);
  if (!isObj(previous) || !isObj(current)) {
    return dumpsSorted((previous ?? null) as Json) === dumpsSorted((current ?? null) as Json);
  }
  for (const key of Object.keys(previous)) {
    if (!(key in current)) continue;
    if (dumpsSorted(previous[key] as Json) !== dumpsSorted(current[key] as Json)) return false;
  }
  return true;
}

/** _is_resumed_eval_prefix: previous is an exact earlier checkpoint of current. */
export function isResumedEvalPrefix(previous: Payload, current: Payload): boolean {
  const identityKeys = ["dataset_name", "arena_session_id"];
  if (!identityKeys.some((k) => k in previous && k in current)) return false;
  if (identityKeys.some((k) => dumpsSorted((previous[k] ?? null) as Json) !== dumpsSorted((current[k] ?? null) as Json))) {
    return false;
  }
  const prevRollouts = previous.rollouts;
  const currRollouts = current.rollouts;
  if (!Array.isArray(prevRollouts) || !Array.isArray(currRollouts)) return false;
  if (prevRollouts.length >= currRollouts.length) return false;
  if (
    dumpsSorted(currRollouts.slice(0, prevRollouts.length) as Json) !==
    dumpsSorted(prevRollouts as Json)
  ) {
    return false;
  }
  for (const key of ["arena_submitted_round_indices", "arena_submitted_rollouts", "phase_stops"]) {
    const prevSubmitted = previous[key];
    const currSubmitted = current[key];
    if (prevSubmitted === undefined && currSubmitted === undefined) continue;
    if (!Array.isArray(prevSubmitted) || !Array.isArray(currSubmitted)) return false;
    if (
      dumpsSorted(currSubmitted.slice(0, prevSubmitted.length) as Json) !==
      dumpsSorted(prevSubmitted as Json)
    ) {
      return false;
    }
  }
  const stableKeys = new Set([...Object.keys(previous), ...Object.keys(current)]);
  for (const key of RESUMABLE_DYNAMIC_KEYS) stableKeys.delete(key);
  stableKeys.delete("args");
  for (const key of stableKeys) {
    if (dumpsSorted((previous[key] ?? null) as Json) !== dumpsSorted((current[key] ?? null) as Json)) {
      return false;
    }
  }
  return resumedArgsCompatible(previous.args, current.args);
}

export interface CanonicalizeFileResult {
  /** New results.json text, or null when nothing changed. */
  resultsText: string | null;
  /** New/updated results_eval_time.json backup text, or null. */
  backupText: string | null;
  reconciliation: Reconciliation | null;
}

/**
 * canonicalize_results_file on in-memory texts. `validateFrameOutcomes` is
 * injected so the caller can supply the (post-edit) frame classification.
 */
export function canonicalizeResultsTexts(args: {
  resultsText: string;
  progressRecord: ProgressRecord | null;
  existingBackupText: string | null;
  overridesFilename: string;
  frameOutcomes: Map<number, FrameOutcome>;
}): CanonicalizeFileResult {
  const payload = JSON.parse(args.resultsText) as Payload;
  const canonical = JSON.parse(args.resultsText) as Payload; // deep copy
  let reconciliation: Reconciliation | null = null;
  if (args.progressRecord !== null) {
    reconciliation = applyOutcomeEditRecord(canonical, args.progressRecord, args.overridesFilename);
  } else {
    recomputeSummaryFromRollouts(canonical);
  }
  validateSummaryMatchesRollouts(canonical);
  validateResultsAgainstFrameOutcomes(canonical, args.frameOutcomes);

  const originalText = dumpsIndent2Sorted(payload as Json) + "\n";
  const canonicalText = dumpsIndent2Sorted(canonical as Json) + "\n";
  const alreadyReconciled = "_outcome_edit_reconciliation" in payload;
  if (alreadyReconciled && args.existingBackupText === null) {
    throw new Error(
      "results.json is already outcome-reconciled but results_eval_time.json is missing; " +
        "refusing to invent eval-time provenance"
    );
  }
  if (canonicalText === originalText) {
    return { resultsText: null, backupText: null, reconciliation };
  }
  let backupText: string | null = null;
  if (args.existingBackupText !== null) {
    const normalizedBackup =
      dumpsIndent2Sorted(JSON.parse(args.existingBackupText) as Json) + "\n";
    if (!alreadyReconciled && normalizedBackup !== originalText) {
      const backupPayload = JSON.parse(args.existingBackupText) as Payload;
      if (isResumedEvalPrefix(backupPayload, payload)) {
        backupText = originalText;
      } else {
        throw new Error(
          "results_eval_time.json already exists and differs from the current raw results " +
            "payload; refusing to overwrite results.json with stale provenance"
        );
      }
    }
  } else {
    backupText = originalText;
  }
  return { resultsText: canonicalText, backupText, reconciliation };
}
