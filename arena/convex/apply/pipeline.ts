"use node";

/**
 * Headless outcome-review apply on an in-memory dataset snapshot.
 *
 * TS port of mulligan/tools/outcome_review.py headless_apply_and_push +
 * apply_progress_and_push: parquet rewrite of the four edit-affected columns,
 * per-episode + global stats refresh, ledger repair, results.json
 * canonicalization, progress-record merge, label-history append.
 *
 * Pure with respect to the network: callers supply a snapshot FileStore
 * (fetched at the pre-apply sha) and receive the changed files to commit.
 */

import type { Table } from "apache-arrow";
import {
  readArrowTable,
  writeArrowTable,
  readFrameColumns,
  rewriteEditColumnsStreaming,
  patchListColumns,
  numberColumn,
  stringListColumn,
  listCell,
} from "./parquetIO";
import {
  buildEpisodeMap,
  applyOutcomeEdits,
  normalizeOutcomeFrame,
  episodeOutcomesByIndex,
  detectFrameOutcome,
  episodeSuccessAndFrames,
  EDIT_AFFECTED_FEATURES,
} from "./frames";
import type { FileFrameColumns, EpisodeMap, FrameOutcome } from "./frames";
import { getFeatureStats1D, STAT_KEYS } from "./stats";
import type { StatKey } from "./stats";
import {
  PROGRESS_FILENAME,
  loadProgress,
  serializeProgress,
  mergeOverlayIntoProgress,
  resolveSubtaskMarks,
  asInt,
} from "./progress";
import type { Overlay } from "./progress";
import {
  HISTORY_FILENAME,
  buildOutcomeEvents,
  appendEvents,
  utcNowIso,
} from "./labelHistory";
import type { LabelSource } from "./labelHistory";
import { DEFAULT_LEDGER_NAMES, repairLedger } from "./ledgers";
import {
  canonicalizeResultsTexts,
  subtaskFramesForValidation,
} from "./results";
import { dumpsIndent4 } from "./pyjson";
import type { Json } from "./pyjson";

/** Snapshot of the repo at the pre-apply sha. */
export interface FileStore {
  /** All file paths in the repo at the snapshot revision. */
  paths: string[];
  /** Fetch one file's bytes (throws on missing). */
  fetch(path: string): Promise<Uint8Array>;
}

export interface ApplySummary {
  applied_episodes: number[];
  skipped_episodes: number[];
  updated_ledgers: string[];
  subtask_marks: number;
  overlay_changed: number[];
  overlay_skipped: number[];
  num_data_files_rewritten: number;
  episode_success: Array<{
    episode_index: number;
    success: boolean;
    num_frames: number;
    /** Sub-goal marks reached (review record, else live eval-time marks). */
    num_subtask_marks: number;
  }>;
  log: string[];
}

export interface ApplyResult {
  /** Repo-relative path → new content, ready for one atomic commit. */
  changedFiles: Map<string, Uint8Array | string>;
  summary: ApplySummary;
}

const RESULTS_FILENAME = "results.json";
const RESULTS_BACKUP_FILENAME = "results_eval_time.json";

async function fetchText(store: FileStore, path: string): Promise<string> {
  return new TextDecoder().decode(await store.fetch(path));
}

async function fetchOptionalText(store: FileStore, path: string): Promise<string | null> {
  return store.paths.includes(path) ? await fetchText(store, path) : null;
}

/** normalize_overlay_records: interactive-confirm record semantics. */
function normalizeOverlayRecords(overlay: Overlay, episodes: EpisodeMap, subtaskMarks: number): void {
  for (const [epStr, record] of Object.entries(overlay.changed_episodes)) {
    const epIdx = parseInt(epStr, 10);
    const ep = episodes.get(epIdx);
    if (!ep) throw new Error(`Overlay episode ${epIdx} not found in parquet data`);
    record.outcome_frame = normalizeOutcomeFrame(ep, asInt(record.outcome_frame));
    if (subtaskMarks === 0 && "subtask_frames" in record) {
      if ((record.subtask_frames ?? []).length > 0) {
        throw new Error(
          `Overlay episode ${epIdx} carries subtask_frames [${record.subtask_frames}] ` +
            `but the task requires 0 subtask marks`
        );
      }
      // Interactive semantics: a 0-mark task never stamps the key.
      delete record.subtask_frames;
    }
  }
}

interface MetaFile {
  path: string;
  table: Table;
  episodeOrder: number[];
}

/**
 * refresh_episode_stats: patch per-episode stats cells in meta/episodes
 * parquet + matching entries in meta/stats.json for the edit-affected
 * features. Returns rewritten meta tables and the new stats.json text.
 */
function refreshEpisodeStats(args: {
  metaFiles: MetaFile[];
  statsJsonText: string | null;
  episodes: EpisodeMap;
  dataFiles: FileFrameColumns[];
  changedEpisodes: Set<number>;
  log: string[];
}): { rewrittenMeta: Map<string, Table>; statsJsonText: string | null } {
  const hasIsValid = args.dataFiles.every((f) => f.isValid !== null);
  const features = EDIT_AFFECTED_FEATURES.filter((f) => f !== "is_valid" || hasIsValid);

  const featureValues = (file: FileFrameColumns, feat: string): ArrayLike<number> => {
    if (feat === "success") return file.success;
    if (feat === "reward") return file.reward;
    if (feat === "done") return file.done;
    if (feat === "is_valid") return file.isValid!;
    throw new Error(`unknown feature ${feat}`);
  };

  const episodeFeature = (epIdx: number, feat: string): number[] => {
    const ep = args.episodes.get(epIdx);
    if (!ep) throw new Error(`Episode ${epIdx} listed in meta has no frames in the data parquet`);
    const col = featureValues(ep.file, feat);
    return ep.rows.map((r) => col[r] as number);
  };

  const rewrittenMeta = new Map<string, Table>();
  for (const meta of args.metaFiles) {
    const patches = new Map<string, Map<number, number[]>>();
    for (const feat of features) {
      const statCols = meta.table.schema.fields
        .map((f) => f.name)
        .filter((n) => n.startsWith(`stats/${feat}/`));
      if (statCols.length === 0) continue;
      for (let rowPos = 0; rowPos < meta.episodeOrder.length; rowPos++) {
        const epIdx = meta.episodeOrder[rowPos];
        if (!args.changedEpisodes.has(epIdx)) continue;
        const stats = getFeatureStats1D(episodeFeature(epIdx, feat));
        for (const col of statCols) {
          const statKey = col.split("/").pop() as StatKey;
          if (!STAT_KEYS.includes(statKey)) {
            throw new Error(`${meta.path}: unknown stat column ${col}`);
          }
          const oldCell = listCell(meta.table, col, rowPos);
          if (oldCell.length !== 1) {
            throw new Error(`${meta.path}: ${col} cell is not length-1 (${oldCell.length})`);
          }
          const newValue = statKey === "count" ? Math.round(stats.count) : stats[statKey];
          if (oldCell[0] !== newValue) {
            let colPatch = patches.get(col);
            if (!colPatch) patches.set(col, (colPatch = new Map()));
            colPatch.set(rowPos, [newValue]);
          }
        }
      }
    }
    if (patches.size > 0) {
      rewrittenMeta.set(meta.path, patchListColumns(meta.table, patches));
    }
  }

  let newStatsJson: string | null = null;
  if (args.statsJsonText !== null) {
    const globalStats = JSON.parse(args.statsJsonText) as Record<string, Record<string, unknown>>;
    let globalChanged = false;
    for (const feat of features) {
      if (!(feat in globalStats)) continue;
      const all: number[] = [];
      for (const file of [...args.dataFiles].sort((a, b) => a.path.localeCompare(b.path))) {
        const col = featureValues(file, feat);
        for (let i = 0; i < file.numRows; i++) all.push(col[i] as number);
      }
      const stats = getFeatureStats1D(all);
      for (const statKey of Object.keys(globalStats[feat])) {
        if (!STAT_KEYS.includes(statKey as StatKey)) {
          throw new Error(`stats.json ${feat}: unknown stat key ${statKey}`);
        }
        const oldArr = globalStats[feat][statKey] as number[];
        if (!Array.isArray(oldArr) || oldArr.length !== 1) {
          throw new Error(`stats.json ${feat}/${statKey}: expected length-1 array`);
        }
        const newValue =
          statKey === "count" ? Math.round(stats.count) : stats[statKey as StatKey];
        // np.allclose(old, new, atol=1e-9) with numpy's DEFAULT rtol=1e-5 —
        // sub-1e-5-relative drift is deliberately left un-rewritten, matching
        // refresh_episode_stats in mulligan/tools/outcome_review.py.
        if (!(Math.abs(oldArr[0] - newValue) <= 1e-9 + 1e-5 * Math.abs(newValue))) {
          globalStats[feat][statKey] = [newValue];
          globalChanged = true;
        }
      }
    }
    if (globalChanged) newStatsJson = dumpsIndent4(globalStats as Json);
  }

  if (rewrittenMeta.size > 0) {
    args.log.push(`Refreshed ${rewrittenMeta.size + (newStatsJson ? 1 : 0)} stats metadata file(s).`);
  }
  return { rewrittenMeta, statsJsonText: newStatsJson };
}

export async function headlessApply(args: {
  store: FileStore;
  overlay: Overlay;
  provenance: { sourceByEpisode: Map<number, LabelSource>; evidence: Json } | null;
  preApplySha: string;
  taskSpecs: Array<{ task_name: string; num_subtask_marks: number | bigint }>;
  now?: Date;
  /** Validation-only footer provenance makes fresh LFS objects without
   * changing Arrow schemas/values, so upload tests cannot pass via dedup. */
  parquetCreatedBy?: string;
  onProgress?: (message: string) => void | Promise<void>;
}): Promise<ApplyResult> {
  const log: string[] = [];
  const { store, overlay } = args;

  // --- Load episode metadata → active data files + task names.
  const metaPaths = store.paths
    .filter((p) => p.startsWith("meta/episodes/") && p.endsWith(".parquet"))
    .sort();
  if (metaPaths.length === 0) throw new Error("no meta/episodes parquet files found");
  const metaFiles: MetaFile[] = [];
  for (const path of metaPaths) {
    const table = readArrowTable(await store.fetch(path));
    metaFiles.push({ path, table, episodeOrder: numberColumn(table, "episode_index") });
  }
  const allMetaEpisodes = metaFiles.flatMap((m) => m.episodeOrder);
  await args.onProgress?.(`Loaded metadata for ${allMetaEpisodes.length} episodes`);
  if (new Set(allMetaEpisodes).size !== allMetaEpisodes.length) {
    throw new Error("meta/episodes has duplicate episode_index values");
  }

  const dataFileKeys = new Set<string>();
  const datasetTasks = new Set<string>();
  for (const meta of metaFiles) {
    const chunks = numberColumn(meta.table, "data/chunk_index");
    const files = numberColumn(meta.table, "data/file_index");
    for (let i = 0; i < chunks.length; i++) {
      dataFileKeys.add(
        `data/chunk-${String(chunks[i]).padStart(3, "0")}/file-${String(files[i]).padStart(3, "0")}.parquet`
      );
    }
    for (const tasks of stringListColumn(meta.table, "tasks")) {
      for (const t of tasks) datasetTasks.add(t);
    }
  }

  const subtaskMarks = resolveSubtaskMarks([...datasetTasks], args.taskSpecs);

  // Keep only projected edit columns. Retaining every compressed input adds
  // dataset-sized memory to the action; dirty files are fetched again below.
  const dataFiles: FileFrameColumns[] = [];
  for (const path of [...dataFileKeys].sort()) {
    if (!store.paths.includes(path)) {
      throw new Error(`Current dataset metadata references missing parquet file: ${path}`);
    }
    const buf = await store.fetch(path);
    dataFiles.push(await readFrameColumns(path, buf));
    if (dataFiles.length % 100 === 0) await args.onProgress?.(`Read ${dataFiles.length}/${dataFileKeys.size} data files`);
  }
  const episodes = buildEpisodeMap(dataFiles);

  // --- Normalize overlay to interactive-confirm record semantics.
  normalizeOverlayRecords(overlay, episodes, subtaskMarks);

  // --- Progress record + label history (events vs the PRE-apply record).
  const progress = loadProgress(await fetchOptionalText(store, PROGRESS_FILENAME));
  const changedFiles = new Map<string, Uint8Array | string>();

  if (args.provenance !== null) {
    const events = buildOutcomeEvents({
      preProgress: progress,
      overlay,
      sourceByEpisode: args.provenance.sourceByEpisode,
      evidence: { ...(args.provenance.evidence as object), pre_sha: args.preApplySha } as Json,
      ts: utcNowIso(args.now),
    });
    if (events.length > 0) {
      const existing = await fetchOptionalText(store, HISTORY_FILENAME);
      changedFiles.set(HISTORY_FILENAME, appendEvents(existing, events));
      log.push(`Appended ${events.length} label-history event(s).`);
    }
  }

  mergeOverlayIntoProgress(progress, overlay);
  changedFiles.set(PROGRESS_FILENAME, serializeProgress(progress));

  // Subtask frames the validator tolerates: live eval-time marks from
  // results.json (unreviewed episodes still carry their live spike), overridden
  // per episode by the merged progress record.
  const resultsText = store.paths.includes(RESULTS_FILENAME)
    ? await fetchText(store, RESULTS_FILENAME)
    : null;
  const subtaskByEp = subtaskFramesForValidation(
    progress,
    resultsText === null ? null : (JSON.parse(resultsText) as Record<string, unknown>)
  );

  // --- Apply edits + refresh stats + repair ledgers.
  const changedEpisodes = new Set(Object.keys(progress.changed_episodes).map((s) => parseInt(s, 10)));
  const updatedLedgers: string[] = [];
  if (changedEpisodes.size > 0) {
    applyOutcomeEdits(episodes, progress, subtaskMarks);
    await args.onProgress?.(`Rewriting ${dataFiles.filter((file) => file.dirty).length} changed data files`);
    // Streamed rewrite, one dirty file at a time; no full data table is ever held.
    for (const file of dataFiles) {
      if (!file.dirty) continue;
      changedFiles.set(file.path, await rewriteEditColumnsStreaming(file.path, await store.fetch(file.path), file, args.parquetCreatedBy));
      if (changedFiles.size % 50 === 0) await args.onProgress?.(`Prepared ${changedFiles.size} output files`);
    }
    log.push("Parquet files updated.");
    await args.onProgress?.("Data rewrites complete; refreshing statistics");

    const statsRefresh = refreshEpisodeStats({
      metaFiles,
      statsJsonText: await fetchOptionalText(store, "meta/stats.json"),
      episodes,
      dataFiles,
      changedEpisodes,
      log,
    });
    for (const [path, table] of statsRefresh.rewrittenMeta) {
      changedFiles.set(path, writeArrowTable(table, args.parquetCreatedBy));
    }
    if (statsRefresh.statsJsonText !== null) {
      changedFiles.set("meta/stats.json", statsRefresh.statsJsonText);
    }
    await args.onProgress?.("Statistics refresh complete");

    const outcomes = episodeOutcomesByIndex(episodes, subtaskByEp);
    for (const ledgerPath of DEFAULT_LEDGER_NAMES) {
      if (!store.paths.includes(ledgerPath)) continue;
      const newText = repairLedger({
        path: ledgerPath,
        text: await fetchText(store, ledgerPath),
        outcomesByEpisode: outcomes,
        onlyEpisodes: changedEpisodes,
      });
      if (newText !== null) {
        changedFiles.set(ledgerPath, newText);
        updatedLedgers.push(ledgerPath);
      }
    }
    if (updatedLedgers.length > 0) log.push(`Updated ${updatedLedgers.length} ledger file(s).`);
  }

  // --- Canonicalize root results.json against the post-edit frame data.
  if (resultsText !== null) {
    const frameOutcomes = new Map<number, FrameOutcome>();
    for (const [epIdx, ep] of episodes) {
      frameOutcomes.set(epIdx, detectFrameOutcome(ep, subtaskByEp.get(epIdx) ?? []));
    }
    const canonical = canonicalizeResultsTexts({
      resultsText,
      progressRecord: changedEpisodes.size > 0 ? progress : null,
      existingBackupText: await fetchOptionalText(store, RESULTS_BACKUP_FILENAME),
      overridesFilename: PROGRESS_FILENAME,
      frameOutcomes,
    });
    if (canonical.resultsText !== null) {
      changedFiles.set(RESULTS_FILENAME, canonical.resultsText);
      if (canonical.backupText !== null) {
        changedFiles.set(RESULTS_BACKUP_FILENAME, canonical.backupText);
      }
      const rec = canonical.reconciliation;
      log.push(
        `Repaired results.json from outcome edits: reviewed=${rec?.episodes_reviewed} ` +
          `class_changes=${rec?.outcome_class_changes} success_flips=${rec?.success_flips.length}`
      );
    }
  }

  const successAndFrames = episodeSuccessAndFrames(episodes);
  return {
    changedFiles,
    summary: {
      applied_episodes: [...changedEpisodes].sort((a, b) => a - b),
      skipped_episodes: [...progress.skipped_episodes].map(asInt).sort((a, b) => a - b),
      updated_ledgers: updatedLedgers,
      subtask_marks: subtaskMarks,
      overlay_changed: Object.keys(overlay.changed_episodes)
        .map((s) => parseInt(s, 10))
        .sort((a, b) => a - b),
      overlay_skipped: [...overlay.skipped_episodes].map(asInt).sort((a, b) => a - b),
      num_data_files_rewritten: dataFiles.filter((f) => f.dirty).length,
      episode_success: [...successAndFrames.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([episode_index, v]) => ({
          episode_index,
          success: v.success,
          num_frames: v.numFrames,
          num_subtask_marks: subtaskByEp.get(episode_index)?.length ?? 0,
        })),
      log,
    },
  };
}
