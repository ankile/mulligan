/**
 * Scalar frame model + outcome-edit semantics.
 *
 * Port of the frame-level pieces of mulligan/tools/outcome_review.py
 * (apply_outcome_edits, normalize_outcome_frame, detect_episode_outcome) and
 * mulligan/real/eval/outcome_results.py (valid_prefix_length,
 * _detect_frame_outcome). Operates on mutable per-file column arrays extracted
 * from the data parquet files; only the four edit-affected columns are ever
 * rewritten (success, reward, done, is_valid — reward is float32 in storage,
 * held here in a Float32Array so written values are exact float32).
 */

import { subtaskMarkCountError, asInt } from "./progress";
import type { OutcomeName, ProgressRecord } from "./progress";

export const EDIT_AFFECTED_FEATURES = ["success", "reward", "done", "is_valid"] as const;

export const OUTCOME_TYPES: Record<OutcomeName, { success: number; reward: number; done: number }> = {
  success: { success: 1, reward: 1.0, done: 1 },
  failure: { success: 0, reward: 0.0, done: 1 },
  timeout: { success: 0, reward: 0.0, done: 0 },
};

/** Mutable scalar columns of one data parquet file, in row order. */
export interface FileFrameColumns {
  path: string; // repo-relative, e.g. data/chunk-000/file-000.parquet
  numRows: number;
  episodeIndex: Float64Array; // int64 values, exact in float64 range
  frameIndex: Float64Array;
  reward: Float32Array;
  done: Float64Array;
  success: Float64Array;
  isValid: Float64Array | null; // null when the dataset has no is_valid column
  /** Set when an edit touched this file; only dirty files are rewritten. */
  dirty: boolean;
}

/** Row locations of one episode within one file, sorted by frame_index. */
export interface EpisodeRows {
  episodeIndex: number;
  file: FileFrameColumns;
  rows: number[]; // row indices into the file arrays, ascending frame_index
}

export type EpisodeMap = Map<number, EpisodeRows>;

/** Group rows by episode across files; every episode must live in ONE file. */
export function buildEpisodeMap(files: FileFrameColumns[]): EpisodeMap {
  const map: EpisodeMap = new Map();
  for (const file of files) {
    const byEp = new Map<number, number[]>();
    for (let i = 0; i < file.numRows; i++) {
      const ep = file.episodeIndex[i];
      if (!Number.isInteger(ep)) throw new Error(`${file.path}: non-integer episode_index ${ep}`);
      const rows = byEp.get(ep);
      if (rows) rows.push(i);
      else byEp.set(ep, [i]);
    }
    for (const [ep, rows] of byEp) {
      if (map.has(ep)) {
        throw new Error(
          `episode ${ep} spans multiple data files (${map.get(ep)!.file.path}, ${file.path})`
        );
      }
      rows.sort((a, b) => file.frameIndex[a] - file.frameIndex[b]);
      map.set(ep, { episodeIndex: ep, file, rows });
    }
  }
  return map;
}

function frameRow(ep: EpisodeRows, frame: number): number | null {
  for (const r of ep.rows) if (ep.file.frameIndex[r] === frame) return r;
  return null;
}

/** normalize_outcome_frame: bounds-check, then terminal-padding fixup. */
export function normalizeOutcomeFrame(ep: EpisodeRows, outcomeFrame: number): number {
  const row = frameRow(ep, outcomeFrame);
  if (row === null) {
    throw new Error(`Episode ${ep.episodeIndex}: frame ${outcomeFrame} not found`);
  }
  const isValid = ep.file.isValid;
  if (isValid === null) return outcomeFrame;
  if (isValid[row] !== 0) return outcomeFrame;

  const validFrames = ep.rows.filter((r) => isValid[r] === 1).map((r) => ep.file.frameIndex[r]);
  if (validFrames.length === 0) {
    throw new Error(`Episode ${ep.episodeIndex}: no valid frames available for outcome placement`);
  }
  const lastValidFrame = Math.max(...validFrames);
  const lastFrame = Math.max(...ep.rows.map((r) => ep.file.frameIndex[r]));
  if (outcomeFrame === lastFrame && outcomeFrame > lastValidFrame) {
    // Invalid terminal padding: outcome lands on the last valid frame.
    return lastValidFrame;
  }
  // A non-terminal is_valid=0 frame is a previously soft-truncated tail frame;
  // the apply re-validates pre-terminal frames, so it is a legal new mark.
  return outcomeFrame;
}

function validateSubtaskFrame(
  ep: EpisodeRows,
  frame: number,
  outcomeFrame: number,
  newOutcome: OutcomeName
): void {
  if (frameRow(ep, frame) === null) {
    throw new Error(`Episode ${ep.episodeIndex}: subtask frame ${frame} not found`);
  }
  if (frame > outcomeFrame || (frame === outcomeFrame && newOutcome !== "timeout")) {
    throw new Error(
      `Episode ${ep.episodeIndex}: subtask frame ${frame} must be strictly before the ` +
        `outcome frame ${outcomeFrame} (equality is allowed only for timeout)`
    );
  }
}

/** apply_outcome_edits: rewrite the four edit columns for every changed episode. */
export function applyOutcomeEdits(
  episodes: EpisodeMap,
  progress: ProgressRecord,
  subtaskMarks: number
): boolean {
  const changed = progress.changed_episodes;
  const entries = Object.entries(changed);
  if (entries.length === 0) return false;

  for (const [epStr, info] of entries) {
    const epIdx = parseInt(epStr, 10);
    const ep = episodes.get(epIdx);
    if (!ep) throw new Error(`Episode ${epIdx} not found in dataset`);
    const file = ep.file;
    const newOutcome = info.new_outcome;
    const softTruncate = info.soft_truncate ?? true;
    const subtaskReviewed = "subtask_frames" in info;
    const subtaskFrames = [...new Set((info.subtask_frames ?? []).map(asInt))].sort((a, b) => a - b);
    const outcomeVals = OUTCOME_TYPES[newOutcome];
    if (!outcomeVals) throw new Error(`Episode ${epIdx}: unknown outcome ${newOutcome}`);

    const outcomeFrame = normalizeOutcomeFrame(ep, asInt(info.outcome_frame));
    const lastFrame = Math.max(...ep.rows.map((r) => file.frameIndex[r]));

    if (subtaskMarks > 0) {
      // Count contract binds only to REVIEWED records (subtask_frames key present).
      if (subtaskReviewed) {
        const err = subtaskMarkCountError(newOutcome, subtaskFrames.length, subtaskMarks);
        if (err) throw new Error(`Episode ${epIdx}: ${err} ([${subtaskFrames}])`);
      }
    } else if (subtaskFrames.length > 0) {
      throw new Error(
        `Episode ${epIdx}: progress record carries subtask marks [${subtaskFrames}] ` +
          `but the task requires 0 subtask marks`
      );
    }

    for (const frame of subtaskFrames) {
      validateSubtaskFrame(ep, frame, outcomeFrame, newOutcome);
    }
    const subtaskSet = new Set(subtaskFrames);
    for (const r of ep.rows) {
      const frame = file.frameIndex[r];
      const reward = subtaskSet.has(frame) ? 1.0 : frame < outcomeFrame ? 0.0 : outcomeVals.reward;
      const done = frame < outcomeFrame ? 0 : outcomeVals.done;
      const valid = frame < lastFrame && (!softTruncate || frame <= outcomeFrame) ? 1 : 0;
      // Re-applying an existing review must not re-encode and retain another
      // output parquet when its stored columns already match the decision.
      if (file.success[r] !== outcomeVals.success || file.reward[r] !== reward ||
          file.done[r] !== done || (file.isValid !== null && file.isValid[r] !== valid)) {
        file.dirty = true;
      }
      // success is an episode-level constant on ALL frames.
      file.success[r] = outcomeVals.success;
      file.reward[r] = reward;
      file.done[r] = done;
      if (file.isValid !== null) file.isValid[r] = valid;
    }
  }
  return true;
}

/** detect_episode_outcome: classify from the last is_valid==1 frame. */
export function detectEpisodeOutcome(ep: EpisodeRows): OutcomeName {
  const file = ep.file;
  if (file.isValid === null) {
    throw new Error(`Episode ${ep.episodeIndex}: dataset has no is_valid column`);
  }
  const validRows = ep.rows.filter((r) => file.isValid![r] === 1);
  if (validRows.length === 0) throw new Error(`Episode ${ep.episodeIndex} has no valid frames`);
  const last = validRows[validRows.length - 1];
  const reward = file.reward[last];
  const done = file.done[last];
  if (reward === 1.0 && done === 1) return "success";
  if (reward === 0.0 && done === 1) return "failure";
  if (reward === 0.0 && done === 0) return "timeout";
  throw new Error(`Unexpected reward=${reward}, done=${done} on last valid frame`);
}

export function episodeOutcomesByIndex(
  episodes: EpisodeMap,
  subtaskFramesByEpisode: Map<number, number[]> = new Map()
): Map<number, OutcomeName> {
  const out = new Map<number, OutcomeName>();
  for (const ep of [...episodes.keys()].sort((a, b) => a - b)) {
    out.set(
      ep,
      detectFrameOutcome(episodes.get(ep)!, subtaskFramesByEpisode.get(ep) ?? []).outcome
    );
  }
  return out;
}

/** valid_prefix_length: leading is_valid==1 run; any 1-after-0 is corruption. */
export function validPrefixLength(isValid: number[], episodeIndex: number): number {
  if (isValid.length === 0) throw new Error(`episode ${episodeIndex}: empty is_valid sequence`);
  let seenInvalid = false;
  let prefix = 0;
  for (const value of isValid) {
    if (value !== 0 && value !== 1) {
      throw new Error(`episode ${episodeIndex}: is_valid must be 0/1, got ${value}`);
    }
    if (value === 1) {
      if (seenInvalid) {
        throw new Error(
          `episode ${episodeIndex}: is_valid is not a valid prefix followed by padding`
        );
      }
      prefix += 1;
    } else {
      seenInvalid = true;
    }
  }
  if (prefix === 0) throw new Error(`episode ${episodeIndex}: no valid frames`);
  return prefix;
}

export interface FrameOutcome {
  outcome: OutcomeName;
  expectedNumSteps: number;
}

/**
 * _detect_frame_outcome: strict per-episode classifier used to validate
 * results.json against the edited frame data. Nonzero pre-terminal reward is
 * tolerated only at recorded subtask frames, and only with the exact 1.0
 * spike value.
 */
export function detectFrameOutcome(ep: EpisodeRows, subtaskFrames: number[]): FrameOutcome {
  const file = ep.file;
  const frames = ep.rows.map((r) => file.frameIndex[r]);
  const dupes = frames.filter((f, i) => i > 0 && frames[i - 1] === f);
  if (dupes.length > 0) {
    throw new Error(`episode ${ep.episodeIndex}: duplicate frame_index values ${[...new Set(dupes)]}`);
  }
  const subtaskSet = new Set(subtaskFrames.map(asInt));

  let validRows: number[];
  if (file.isValid !== null) {
    const prefix = validPrefixLength(
      ep.rows.map((r) => file.isValid![r]),
      ep.episodeIndex
    );
    validRows = ep.rows.slice(0, prefix);
  } else {
    validRows = ep.rows;
  }
  if (validRows.length === 0) throw new Error("episode has no valid frames");

  const checkSpikeValues = (rows: number[]) => {
    for (const r of rows) {
      if (subtaskSet.has(file.frameIndex[r]) && file.reward[r] !== 1.0) {
        throw new Error(
          `recorded subtask frame(s) [${file.frameIndex[r]}] have reward != 1.0; ` +
            `spike values must be exactly 1.0`
        );
      }
    }
  };

  const terminalPos = validRows.findIndex((r) => file.done[r] === 1);
  let outcome: OutcomeName;
  let expectedNumSteps: number;
  if (terminalPos === -1) {
    for (const r of validRows) {
      if (file.reward[r] !== 0.0 && !subtaskSet.has(file.frameIndex[r])) {
        throw new Error("timeout episode has nonzero reward before terminal");
      }
    }
    checkSpikeValues(validRows);
    outcome = "timeout";
    expectedNumSteps = validRows.length;
  } else {
    const pre = validRows.slice(0, terminalPos);
    const tail = validRows.slice(terminalPos);
    for (const r of pre) {
      if (file.done[r] !== 0) {
        throw new Error("terminal episode has done=1 before the outcome frame");
      }
      if (file.reward[r] !== 0.0 && !subtaskSet.has(file.frameIndex[r])) {
        throw new Error("terminal episode has nonzero reward before outcome frame");
      }
    }
    checkSpikeValues(pre);
    for (const r of tail) {
      if (file.done[r] !== 1) {
        throw new Error("terminal episode has done=0 after the outcome frame");
      }
    }
    const firstTerminalReward = file.reward[tail[0]];
    if (firstTerminalReward !== 0.0 && firstTerminalReward !== 1.0) {
      throw new Error(`unexpected terminal reward=${firstTerminalReward}`);
    }
    for (const r of tail) {
      if (file.reward[r] !== firstTerminalReward) {
        throw new Error("terminal episode changes reward after the outcome frame");
      }
    }
    outcome = firstTerminalReward === 1.0 ? "success" : "failure";
    expectedNumSteps = Math.min(...tail.map((r) => file.frameIndex[r])) + 1;
  }

  // success column must be episode-constant over ALL rows (padding included).
  const expectedSuccess = outcome === "success" ? 1 : 0;
  for (const r of ep.rows) {
    if (file.success[r] !== expectedSuccess) {
      throw new Error(
        `success column is not episode-constant ${expectedSuccess}; ` +
          `saw ${[...new Set(ep.rows.map((r2) => file.success[r2]))].sort()}`
      );
    }
  }
  return { outcome, expectedNumSteps };
}

/** Per-episode success + valid-frame counts for the arena resubmit step. */
export function episodeSuccessAndFrames(
  episodes: EpisodeMap
): Map<number, { success: boolean; numFrames: number }> {
  const out = new Map<number, { success: boolean; numFrames: number }>();
  for (const [epIdx, ep] of episodes) {
    const firstRow = ep.rows.find((r) => ep.file.frameIndex[r] === 0);
    if (firstRow === undefined) throw new Error(`episode ${epIdx}: no frame_index==0 row`);
    const numFrames =
      ep.file.isValid === null
        ? ep.rows.length
        : ep.rows.filter((r) => ep.file.isValid![r] === 1).length;
    out.set(epIdx, { success: ep.file.success[firstRow] === 1, numFrames });
  }
  return out;
}
