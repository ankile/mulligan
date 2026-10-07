/**
 * Progress-record and review-overlay semantics.
 *
 * Port of the mulligan/tools/outcome_review.py record primitives, plus the
 * review overlay. The `.outcome_edit_progress.json` record shape is shared with
 * the Python outcome editor:
 *   {"changed_episodes": {"<idx>": {new_outcome, outcome_frame, soft_truncate,
 *     subtask_frames?}}, "skipped_episodes": [idx, ...]}
 * subtask_frames key PRESENCE marks a record reviewed in subtask mode (may be
 * an empty list); records from 0-mark sessions never carry the key.
 */

import { dumpsIndent2 } from "./pyjson";

export const PROGRESS_FILENAME = ".outcome_edit_progress.json";
export const OUTCOME_NAMES = ["success", "failure", "timeout"] as const;
export type OutcomeName = (typeof OUTCOME_NAMES)[number];

export interface ChangedRecord {
  new_outcome: OutcomeName;
  outcome_frame: number;
  soft_truncate: boolean;
  subtask_frames?: number[];
}

export interface ProgressRecord {
  changed_episodes: Record<string, ChangedRecord>;
  skipped_episodes: number[];
}

export interface Overlay {
  changed_episodes: Record<string, ChangedRecord>;
  skipped_episodes: number[];
}

/** One folded (latest-per-episode, cleared-removed) Convex outcome review row. */
export interface ReviewRow {
  episode_index: number | bigint;
  status: string;
  new_outcome?: string;
  outcome_frame?: number | bigint;
  soft_truncate?: boolean;
  subtask_frames?: Array<number | bigint>;
  reviewer: string;
}

export function asInt(value: number | bigint): number {
  if (typeof value === "bigint") {
    if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(-Number.MAX_SAFE_INTEGER)) {
      throw new Error(`int64 value ${value} exceeds safe integer range`);
    }
    return Number(value);
  }
  if (!Number.isInteger(value)) throw new Error(`Expected integer, got ${value}`);
  return value;
}

function asOutcome(value: string | undefined): OutcomeName {
  if (value === undefined || !(OUTCOME_NAMES as readonly string[]).includes(value)) {
    throw new Error(`Unknown outcome ${JSON.stringify(value)}`);
  }
  return value as OutcomeName;
}

/**
 * Convert latest-per-episode Convex reviews into a progress-record overlay.
 * Overlay rules: confirmed →
 * changed_episodes entry (subtask_frames key present ⇔ the review carried
 * one, even empty); skipped → skipped_episodes.
 */
export function buildOverlay(rows: ReviewRow[]): Overlay {
  const changed: Record<string, ChangedRecord> = {};
  const skipped: number[] = [];
  for (const row of rows) {
    const epIdx = asInt(row.episode_index);
    if (row.status === "confirmed") {
      if (row.outcome_frame === undefined) {
        throw new Error(`Episode ${epIdx}: confirmed review missing outcome_frame`);
      }
      const record: ChangedRecord = {
        new_outcome: asOutcome(row.new_outcome),
        outcome_frame: asInt(row.outcome_frame),
        soft_truncate: Boolean(row.soft_truncate ?? false),
      };
      if (row.subtask_frames !== undefined && row.subtask_frames !== null) {
        // Deduplicated + sorted to match mark_episode_changed record semantics.
        record.subtask_frames = [...new Set(row.subtask_frames.map(asInt))].sort((a, b) => a - b);
      }
      changed[String(epIdx)] = record;
    } else if (row.status === "skipped") {
      skipped.push(epIdx);
    } else {
      throw new Error(`Unexpected review status for episode ${epIdx}: ${row.status}`);
    }
  }
  return { changed_episodes: changed, skipped_episodes: skipped };
}

/** load_progress for the `.outcome_edit_progress.json` record (null text: no
 * record yet). */
export function loadProgress(text: string | null): ProgressRecord {
  if (text === null) return { changed_episodes: {}, skipped_episodes: [] };
  const parsed = JSON.parse(text) as Partial<ProgressRecord>;
  return {
    changed_episodes: parsed.changed_episodes ?? {},
    skipped_episodes: parsed.skipped_episodes ?? [],
  };
}

export function serializeProgress(progress: ProgressRecord): string {
  return dumpsIndent2(progress as never);
}

function removeSkipped(progress: ProgressRecord, epIdx: number): void {
  progress.skipped_episodes = progress.skipped_episodes.filter((s) => asInt(s) !== epIdx);
}

/** mark_episode_changed: persists subtask_frames key ⇔ reviewed-or-nonempty. */
export function markEpisodeChanged(
  progress: ProgressRecord,
  epIdx: number,
  record: {
    new_outcome: OutcomeName;
    outcome_frame: number;
    soft_truncate: boolean;
    subtask_frames: number[];
    subtask_reviewed: boolean;
  }
): void {
  removeSkipped(progress, epIdx);
  const entry: ChangedRecord = {
    new_outcome: record.new_outcome,
    outcome_frame: record.outcome_frame,
    soft_truncate: record.soft_truncate,
  };
  if (record.subtask_reviewed || record.subtask_frames.length > 0) {
    entry.subtask_frames = [...new Set(record.subtask_frames)].sort((a, b) => a - b);
  }
  progress.changed_episodes[String(epIdx)] = entry;
}

export function markEpisodeSkipped(progress: ProgressRecord, epIdx: number): void {
  delete progress.changed_episodes[String(epIdx)];
  removeSkipped(progress, epIdx);
  progress.skipped_episodes.push(epIdx);
}

/** merge_overlay_into_progress: same primitives as the interactive editor. */
export function mergeOverlayIntoProgress(progress: ProgressRecord, overlay: Overlay): ProgressRecord {
  for (const [epStr, record] of Object.entries(overlay.changed_episodes)) {
    markEpisodeChanged(progress, parseInt(epStr, 10), {
      new_outcome: record.new_outcome,
      outcome_frame: record.outcome_frame,
      soft_truncate: record.soft_truncate ?? false,
      subtask_frames: record.subtask_frames ?? [],
      subtask_reviewed: "subtask_frames" in record,
    });
  }
  for (const epIdx of overlay.skipped_episodes) {
    markEpisodeSkipped(progress, asInt(epIdx));
  }
  return progress;
}

/** subtask_mark_count_error — shared success/prefix legality contract. */
export function subtaskMarkCountError(
  newOutcome: OutcomeName,
  nMarks: number,
  subtaskMarks: number
): string | null {
  if (subtaskMarks <= 0) return null;
  if (newOutcome === "success") {
    if (nMarks !== subtaskMarks) {
      return (
        `a SUCCESS episode must carry exactly ${subtaskMarks} subtask mark(s) ` +
        `(the mid-episode sub-goal(s); the final sub-goal is the terminal success), ` +
        `got ${nMarks}`
      );
    }
  } else if (nMarks > subtaskMarks) {
    return (
      `a ${newOutcome.toUpperCase()} episode may carry at most ${subtaskMarks} subtask ` +
      `mark(s), got ${nMarks}`
    );
  }
  return null;
}

/**
 * resolve_subtask_marks against the Convex taskSpecs export (RealTaskSpec is
 * the Python single source of truth; taskSpecs rows mirror task_name +
 * num_subtask_marks). No registered task → 0 (single terminal mark);
 * registered tasks disagreeing → loud failure.
 */
export function resolveSubtaskMarks(
  datasetTasks: string[],
  taskSpecs: Array<{ task_name: string; num_subtask_marks: number | bigint }>
): number {
  const byTaskName = new Map(taskSpecs.map((s) => [s.task_name, asInt(s.num_subtask_marks)]));
  const counts = new Set<number>();
  for (const name of datasetTasks) {
    const marks = byTaskName.get(name);
    if (marks !== undefined) counts.add(marks);
  }
  if (counts.size === 0) return 0;
  if (counts.size > 1) {
    throw new Error(
      `dataset mixes tasks with differing num_subtask_marks ${[...counts].sort()} ` +
        `across tasks ${[...datasetTasks].sort()}`
    );
  }
  return [...counts][0];
}
