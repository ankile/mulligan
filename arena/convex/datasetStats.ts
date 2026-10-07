"use node";

import type { Table } from "apache-arrow";
import { v } from "convex/values";
import { internalAction } from "./_generated/server";
import { internal } from "./_generated/api";
import {
  downloadRepoFile,
  downloadRepoText,
  listRepoFiles,
  revisionSha,
} from "./apply/hf";
import type { HfClient } from "./apply/hf";
import { listCell, numberColumn, readArrowTable } from "./apply/parquetIO";
import {
  DATASET_STATS_ALGORITHM_VERSION,
  episodeStatsFromFrames,
  summarizeDatasetStats,
} from "./datasetStatsLogic";
import type {
  DatasetStatsSummary,
  EpisodeStatsInput,
  FrameStatsInput,
} from "./datasetStatsLogic";

const DOWNLOAD_BATCH_SIZE = 8;

type DatasetInfo = {
  fps: number;
  total_episodes: number;
  total_frames: number;
};

function hasColumn(table: Table, name: string): boolean {
  return table.schema.fields.some((field) => field.name === name);
}

function singleStat(table: Table, name: string, row: number): number | null {
  const column = table.getChild(name);
  if (!column) throw new Error(`missing column ${name}`);
  const cell = column.get(row);
  if (cell === null) return null;

  let value: number;
  if (typeof cell === "number" || typeof cell === "bigint") {
    value = Number(cell);
  } else {
    const values = listCell(table, name, row);
    if (values.length !== 1) {
      throw new Error(`${name} row ${row} must contain one value, got ${values.length}`);
    }
    value = values[0];
  }
  if (!Number.isFinite(value)) {
    throw new Error(`${name} row ${row} must be finite, got ${value}`);
  }
  return value;
}

function binaryTrueCount(
  table: Table,
  feature: "done" | "is_valid" | "source",
  row: number,
  rawLength: number,
  required: boolean
): number | null | undefined {
  const requiredNames = ["min", "max", "mean"].map(
    (stat) => `stats/${feature}/${stat}`
  );
  const countName = `stats/${feature}/count`;
  const present = requiredNames.filter((name) => hasColumn(table, name));
  if (present.length === 0 && !required) return null;
  if (present.length !== requiredNames.length) {
    throw new Error(`Episode metadata has incomplete ${feature} statistics`);
  }

  const min = singleStat(table, requiredNames[0], row);
  const max = singleStat(table, requiredNames[1], row);
  const mean = singleStat(table, requiredNames[2], row);
  const count = hasColumn(table, countName)
    ? singleStat(table, countName, row)
    : rawLength;
  if (min === null || max === null || mean === null || count === null) {
    return undefined;
  }
  if (!Number.isInteger(count) || count !== rawLength) {
    throw new Error(
      `${feature} stats count ${count} does not match episode length ${rawLength}`
    );
  }
  if (![0, 1].includes(min) || ![0, 1].includes(max) || min > max) {
    throw new Error(`${feature} statistics are not binary`);
  }
  if (mean < 0 || mean > 1) {
    throw new Error(`${feature} mean must be in [0, 1], got ${mean}`);
  }

  const exactCount = mean * count;
  const trueCount = Math.round(exactCount);
  if (Math.abs(exactCount - trueCount) > 1e-6 * Math.max(1, count)) {
    throw new Error(`${feature} mean does not encode an integer count`);
  }
  if ((max === 0 && trueCount !== 0) || (min === 1 && trueCount !== count)) {
    throw new Error(`${feature} statistics are internally inconsistent`);
  }
  return trueCount;
}

function extractEpisodeRows(table: Table, path: string): EpisodeStatsInput[] | null {
  const episodeIndices = numberColumn(table, "episode_index");
  const lengths = numberColumn(table, "length");
  if (episodeIndices.length !== table.numRows || lengths.length !== table.numRows) {
    throw new Error(`${path}: scalar metadata columns do not match table length`);
  }

  const hasSource = hasColumn(table, "stats/source/mean");
  const rows: EpisodeStatsInput[] = [];
  for (let row = 0; row < episodeIndices.length; row++) {
    const episodeIndex = episodeIndices[row];
    const rawLength = lengths[row];
    if (!Number.isInteger(rawLength) || rawLength < 1) {
      throw new Error(`${path}: episode ${episodeIndex} has invalid length ${rawLength}`);
    }
    const success = singleStat(table, "stats/success/max", row);
    if (success === null) return null;
    if (success !== 0 && success !== 1) {
      throw new Error(`${path}: episode ${episodeIndex} success max must be 0 or 1`);
    }
    const validFrames = binaryTrueCount(table, "is_valid", row, rawLength, false);
    const doneFrames = binaryTrueCount(table, "done", row, rawLength, false);
    const humanFrames = hasSource
      ? binaryTrueCount(table, "source", row, rawLength, true)
      : null;
    if (
      validFrames === undefined ||
      doneFrames === undefined ||
      humanFrames === undefined
    ) {
      return null;
    }
    rows.push({
      episodeIndex,
      rawLength,
      success: success === 1,
      validFrames,
      doneFrames,
      humanFrames,
    });
  }
  return rows;
}

function strictNumberColumn(table: Table, name: string, path: string): number[] {
  const column = table.getChild(name);
  if (!column) throw new Error(`${path}: missing required column ${name}`);
  const values: number[] = [];
  for (let row = 0; row < table.numRows; row++) {
    const cell = column.get(row);
    if (cell === null) throw new Error(`${path}: ${name} is null at row ${row}`);
    const value = Number(cell);
    if (!Number.isFinite(value)) {
      throw new Error(`${path}: ${name} is not finite at row ${row}`);
    }
    values.push(value);
  }
  return values;
}

function extractFrameRows(table: Table, path: string): FrameStatsInput[] {
  const episodeIndices = strictNumberColumn(table, "episode_index", path);
  const success = strictNumberColumn(table, "success", path);
  const optional = (name: "is_valid" | "done" | "source"): number[] | null =>
    hasColumn(table, name) ? strictNumberColumn(table, name, path) : null;
  const isValid = optional("is_valid");
  const done = optional("done");
  const source = optional("source");
  return episodeIndices.map((episodeIndex, row) => ({
    episodeIndex,
    success: success[row],
    isValid: isValid?.[row] ?? null,
    done: done?.[row] ?? null,
    source: source?.[row] ?? null,
  }));
}

async function computeFrameFallback(
  client: HfClient,
  sha: string
): Promise<EpisodeStatsInput[]> {
  const paths = (await listRepoFiles(client, sha, "data"))
    .filter((path) => path.endsWith(".parquet"))
    .sort();
  if (paths.length === 0) {
    throw new Error(`${client.repoId}@${sha}: no data parquet files`);
  }
  const frames: FrameStatsInput[] = [];
  for (let start = 0; start < paths.length; start += DOWNLOAD_BATCH_SIZE) {
    const batchPaths = paths.slice(start, start + DOWNLOAD_BATCH_SIZE);
    const buffers = await Promise.all(
      batchPaths.map((path) => downloadRepoFile(client, path, sha))
    );
    for (let index = 0; index < batchPaths.length; index++) {
      const table = readArrowTable(buffers[index]);
      frames.push(...extractFrameRows(table, batchPaths[index]));
    }
  }
  return episodeStatsFromFrames(frames);
}

async function computeStats(
  client: HfClient,
  sha: string
): Promise<DatasetStatsSummary> {
  const info = JSON.parse(
    await downloadRepoText(client, "meta/info.json", sha)
  ) as DatasetInfo;
  const paths = (await listRepoFiles(client, sha, "meta/episodes"))
    .filter((path) => path.endsWith(".parquet"))
    .sort();
  if (paths.length === 0) {
    throw new Error(`${client.repoId}@${sha}: no meta/episodes parquet files`);
  }

  let rows: EpisodeStatsInput[] = [];
  let needsFrameFallback = false;
  for (let start = 0; start < paths.length; start += DOWNLOAD_BATCH_SIZE) {
    const batchPaths = paths.slice(start, start + DOWNLOAD_BATCH_SIZE);
    const buffers = await Promise.all(
      batchPaths.map((path) => downloadRepoFile(client, path, sha))
    );
    for (let index = 0; index < batchPaths.length; index++) {
      const table = readArrowTable(buffers[index]);
      const extracted = extractEpisodeRows(table, batchPaths[index]);
      if (extracted === null) {
        needsFrameFallback = true;
      } else {
        rows.push(...extracted);
      }
    }
  }
  if (needsFrameFallback) rows = await computeFrameFallback(client, sha);

  const summary = summarizeDatasetStats(rows, Number(info.fps));
  if (summary.numEpisodes !== Number(info.total_episodes)) {
    throw new Error(
      `Episode metadata has ${summary.numEpisodes} rows, meta/info.json declares ${info.total_episodes}`
    );
  }
  if (summary.totalFrames !== Number(info.total_frames)) {
    throw new Error(
      `Episode metadata has ${summary.totalFrames} frames, meta/info.json declares ${info.total_frames}`
    );
  }
  return summary;
}

export const refresh = internalAction({
  args: {
    repo_id: v.string(),
    requested_at: v.float64(),
  },
  handler: async (ctx, args) => {
    let sha: string | undefined;
    try {
      const token = process.env.HF_TOKEN;
      if (!token) throw new Error("HF_TOKEN deployment env var is not set");
      const client: HfClient = { repoId: args.repo_id, token };
      sha = await revisionSha(client);
      const cached = await ctx.runQuery(internal.datasets.getStatsStateInternal, {
        repo_id: args.repo_id,
      });
      if (cached === null) return;
      if (
        cached.stats_hf_sha === sha &&
        cached.stats_algorithm_version === DATASET_STATS_ALGORITHM_VERSION
      ) {
        await ctx.runMutation(internal.datasets.finishCachedStatsRefreshInternal, {
          repo_id: args.repo_id,
          requested_at: args.requested_at,
        });
        return;
      }

      const summary = await computeStats(client, sha);
      const currentSha = await revisionSha(client);
      if (currentSha !== sha) {
        throw new Error(
          `${args.repo_id} changed during stats refresh: ${sha.slice(0, 8)} -> ${currentSha.slice(0, 8)}`
        );
      }
      await ctx.runMutation(internal.datasets.finishStatsRefreshInternal, {
        repo_id: args.repo_id,
        requested_at: args.requested_at,
        hf_sha: sha,
        algorithm_version: DATASET_STATS_ALGORITHM_VERSION,
        num_episodes: summary.numEpisodes,
        total_duration_seconds: summary.totalDurationSeconds,
        num_success: summary.numSuccess,
        num_failure: summary.numFailure,
        num_human_frames: summary.numHumanFrames ?? undefined,
        num_policy_frames: summary.numPolicyFrames ?? undefined,
        num_autonomous_success: summary.numAutonomousSuccess ?? undefined,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await ctx.runMutation(internal.datasets.failStatsRefreshInternal, {
        repo_id: args.repo_id,
        requested_at: args.requested_at,
        error: message.slice(0, 1000),
      });
      throw error;
    }
  },
});
