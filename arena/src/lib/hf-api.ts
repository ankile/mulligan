import { asyncBufferFromUrl, parquetReadObjects, toJson } from "hyparquet";

export const FPS = 15;
// The wrist camera's right eye is recorded but no policy consumes it.
const HIDDEN_CAMERA_KEYS = new Set(["observation.images.wrist_right"]);
type EpisodeWithoutSuccess = Omit<EpisodeMetadata, "success">;
type ParquetCacheEntry = {
  episodes: EpisodeWithoutSuccess[];
  cameraKeys: string[];
  successMap: Map<number, boolean>;
  complete: boolean;
};

export function missingEpisodeIndices(
  episodes: readonly Pick<EpisodeMetadata, "episodeIndex">[],
  neededIndices: ReadonlySet<number>
): number[] {
  const present = new Set(episodes.map((episode) => episode.episodeIndex));
  return [...neededIndices]
    .filter((episodeIndex) => !present.has(episodeIndex))
    .sort((a, b) => a - b);
}

export function assertEpisodeCoverage(
  datasetId: string,
  episodes: readonly Pick<EpisodeMetadata, "episodeIndex">[],
  neededIndices: ReadonlySet<number>
): void {
  const missing = missingEpisodeIndices(episodes, neededIndices);
  if (missing.length > 0) {
    throw new Error(
      `${datasetId} does not contain requested episode indices: ${missing.join(", ")}`
    );
  }
}
type EpisodeFrameSummary = {
  effectiveLength: number;
  success: boolean | null;
};

export interface EpisodeMetadata {
  episodeIndex: number;
  numFrames: number;
  duration: number;
  success: boolean;
  cameraTimings?: Record<string, {videoFileIndex:number;fromTimestamp:number;toTimestamp:number}>;
  videoFileIndex: number;
  fromTimestamp: number;
  toTimestamp: number;
}

/** Set once before mounting a frozen public release; unknown repositories fail closed. */
let releasePins: ReadonlyMap<string, {revision:string;fps:number}> | null = null;
export function configureReleaseDatasets(pins: ReadonlyMap<string, {revision:string;fps:number}>) {
  if (parquetCache.size) throw new Error("Configure release pins before loading datasets");
  releasePins = pins;
}
function datasetRevision(datasetId:string): string {
  if (!releasePins) return 'main';
  const pin=releasePins.get(datasetId);
  if (!pin) throw new Error(`Dataset is outside the public release: ${datasetId}`);
  return pin.revision;
}
export function datasetFps(datasetId:string):number {return releasePins?.get(datasetId)?.fps ?? FPS;}

export function seedReleaseEpisodes(datasetId:string, entry: ParquetCacheEntry) {
  datasetRevision(datasetId);
  if (!releasePins) throw new Error('Release metadata requires pinned repositories');
  parquetCache.set(datasetId,entry);
}
export function episodeForCamera<T extends EpisodeWithoutSuccess>(episode:T, camera:string):T {
  const timing=episode.cameraTimings?.[camera];
  return timing ? {...episode,...timing,duration:timing.toTimestamp-timing.fromTimestamp} : episode;
}

function hfBase(datasetId: string): string {
  return `https://huggingface.co/datasets/${datasetId}/resolve/${datasetRevision(datasetId)}`;
}

async function listEpisodeMetadataFiles(datasetId: string): Promise<string[]> {
  const url =
    `https://huggingface.co/api/datasets/${datasetId}/tree/${datasetRevision(datasetId)}/meta/episodes` +
    "?recursive=true&expand=false&limit=1000";
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Hugging Face metadata listing returned ${response.status} for ${datasetId}`);
  }
  const entries = (await response.json()) as { path: string; type: string }[];
  const paths = entries
    .filter((entry) => entry.type === "file" && entry.path.endsWith(".parquet"))
    .map((entry) => entry.path)
    .sort();
  if (paths.length === 0) {
    throw new Error(`${datasetId} has no meta/episodes parquet files`);
  }
  return paths;
}

async function readParquet(url: string): Promise<Record<string, unknown>[]> {
  const file = await asyncBufferFromUrl({ url });
  const rows = await parquetReadObjects({ file });
  return rows.map((raw) => toJson(raw) as Record<string, unknown>);
}

export function getVideoUrl(
  cameraKey: string,
  fileIndex: number,
  datasetId: string
): string {
  return `${hfBase(datasetId)}/videos/${cameraKey}/chunk-000/file-${String(fileIndex).padStart(3, "0")}.mp4`;
}

/** Every camera key present in the parquet schema, including hidden ones. */
function discoverAllCameraKeys(columnNames: string[]): string[] {
  const prefix = "videos/";
  const suffix = "/file_index";
  return columnNames
    .filter((col) => col.startsWith(prefix) && col.endsWith(suffix))
    .map((col) => col.slice(prefix.length, -suffix.length))
    .sort();
}

/** Discover camera keys from parquet column names matching videos/<key>/file_index. */
function discoverCameraKeys(columnNames: string[]): string[] {
  return discoverAllCameraKeys(columnNames).filter(
    (key) => !HIDDEN_CAMERA_KEYS.has(key)
  );
}

export function visibleCameraKeys(cameraKeys: string[]): string[] {
  return cameraKeys.filter((key) => !HIDDEN_CAMERA_KEYS.has(key));
}

/**
 * Cameras shown by Data Explorer.
 *
 * Datasets that name cameras by ZED serial expose both eyes of each camera
 * (`<serial>_left`, `<serial>_right`); showing the left eye only avoids
 * duplicate stereo views. Datasets that name cameras by station role keep
 * every policy-facing view; `wrist_right` is excluded centrally because it is
 * not policy-facing.
 */
export function explorerCameraKeys(cameraKeys: string[]): string[] {
  const visibleKeys = visibleCameraKeys(cameraKeys);
  const legacyStereoRole = /^\d+_(left|right)$/;
  const isEntirelyLegacyStereo =
    visibleKeys.length > 0 &&
    visibleKeys.every((key) => legacyStereoRole.test(key.split(".").at(-1) ?? ""));

  if (!isEntirelyLegacyStereo) return visibleKeys;
  return visibleKeys.filter((key) => key.split(".").at(-1)?.endsWith("_left"));
}

export function selectPrimaryCameraKey(cameraKeys: string[]): string {
  const visibleKeys = visibleCameraKeys(cameraKeys);
  // Prefer the side_1 station view for previews when it exists.
  const sideOne = visibleKeys.find((key) => key.includes("side_1"));
  if (sideOne) return sideOne;
  return visibleKeys.length > 1
    ? visibleKeys[visibleKeys.length - 1]
    : (visibleKeys[0] ?? "");
}

function metadataScalar(row: Record<string, unknown>, key: string): number {
  const raw = row[key];
  const value = Array.isArray(raw) ? (raw.length === 1 ? raw[0] : NaN) : raw;
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`Episode metadata field ${key} must be a finite scalar, got ${JSON.stringify(raw)}`);
  }
  return number;
}

function optionalMetadataScalar(
  row: Record<string, unknown>,
  key: string
): number | null {
  return key in row ? metadataScalar(row, key) : null;
}

/** Recover the exact number of true values in a binary per-frame feature. */
export function binaryTrueCountFromEpisodeStats(
  row: Record<string, unknown>,
  feature: "done" | "is_valid",
  rawLength: number
): number | null {
  const prefix = `stats/${feature}`;
  const keys = [`${prefix}/min`, `${prefix}/max`, `${prefix}/mean`, `${prefix}/count`];
  const present = keys.filter((key) => key in row);
  if (present.length === 0) return null;
  if (present.length !== keys.length) {
    throw new Error(`Episode metadata has incomplete ${feature} statistics`);
  }

  const min = metadataScalar(row, `${prefix}/min`);
  const max = metadataScalar(row, `${prefix}/max`);
  const mean = metadataScalar(row, `${prefix}/mean`);
  const count = metadataScalar(row, `${prefix}/count`);
  if (!Number.isInteger(count) || count !== rawLength) {
    throw new Error(`${feature} stats count ${count} does not match episode length ${rawLength}`);
  }
  if (![0, 1].includes(min) || ![0, 1].includes(max) || min > max || mean < 0 || mean > 1) {
    throw new Error(`Episode metadata ${feature} statistics are not binary`);
  }

  const exactCount = mean * count;
  const trueCount = Math.round(exactCount);
  if (Math.abs(exactCount - trueCount) > 1e-6 * Math.max(1, count)) {
    throw new Error(`Episode metadata ${feature} mean does not encode an integer count`);
  }
  if ((max === 0 && trueCount !== 0) || (min === 1 && trueCount !== count)) {
    throw new Error(`Episode metadata ${feature} statistics are internally inconsistent`);
  }
  return trueCount;
}

/**
 * Effective task length: include the first done frame, but exclude the first
 * invalid frame and everything after either boundary.
 *
 * The real datasets enforce a binary done suffix and an is_valid prefix. Their
 * episode parquet statistics therefore encode both boundary positions exactly:
 * first_done = raw_length - sum(done), valid_length = sum(is_valid).
 */
export function effectiveEpisodeLengthFromStats(
  row: Record<string, unknown>
): number {
  const rawLength = metadataScalar(row, "length");
  if (!Number.isInteger(rawLength) || rawLength < 1) {
    throw new Error(`Episode length must be a positive integer, got ${rawLength}`);
  }

  const validCount = binaryTrueCountFromEpisodeStats(row, "is_valid", rawLength);
  const doneCount = binaryTrueCountFromEpisodeStats(row, "done", rawLength);
  const validLength = validCount ?? rawLength;
  const doneInclusiveLength =
    doneCount != null && doneCount > 0 ? rawLength - doneCount + 1 : rawLength;
  const effectiveLength = Math.min(validLength, doneInclusiveLength);
  if (effectiveLength < 1 || effectiveLength > rawLength) {
    throw new Error(
      `Invalid effective episode length ${effectiveLength} from raw length ${rawLength}`
    );
  }
  return effectiveLength;
}

export function successFromEpisodeStats(
  row: Record<string, unknown>
): boolean | null {
  const value = optionalMetadataScalar(row, "stats/success/max");
  if (value === null) return null;
  if (value !== 0 && value !== 1) {
    throw new Error(`Episode success statistic must be 0 or 1, got ${value}`);
  }
  return value === 1;
}

export function summarizeEpisodeFrames(
  rows: Record<string, unknown>[],
  episodeIndex: number
): EpisodeFrameSummary {
  if (rows.length === 0) {
    throw new Error(`Episode ${episodeIndex} has no frame rows`);
  }
  const sorted = [...rows].sort(
    (a, b) => metadataScalar(a, "frame_index") - metadataScalar(b, "frame_index")
  );
  for (let index = 0; index < sorted.length; index++) {
    if (metadataScalar(sorted[index], "frame_index") !== index) {
      throw new Error(`Episode ${episodeIndex} has non-contiguous frame indices`);
    }
  }

  let validLength = sorted.length;
  if ("is_valid" in sorted[0]) {
    let seenInvalid = false;
    validLength = 0;
    for (const row of sorted) {
      const value = metadataScalar(row, "is_valid");
      if (value !== 0 && value !== 1) {
        throw new Error(`Episode ${episodeIndex} has non-binary is_valid`);
      }
      if (value === 0) {
        seenInvalid = true;
      } else {
        if (seenInvalid) {
          throw new Error(`Episode ${episodeIndex} has a valid frame after an invalid frame`);
        }
        validLength += 1;
      }
    }
    if (validLength === 0) {
      throw new Error(`Episode ${episodeIndex} has no valid frames`);
    }
  }

  let effectiveLength = validLength;
  if ("done" in sorted[0]) {
    const validDone = sorted
      .slice(0, validLength)
      .map((row) => metadataScalar(row, "done"));
    if (validDone.some((value) => value !== 0 && value !== 1)) {
      throw new Error(`Episode ${episodeIndex} has non-binary done`);
    }
    const firstDone = validDone.indexOf(1);
    if (firstDone >= 0) {
      if (validDone.slice(firstDone).some((value) => value === 0)) {
        throw new Error(`Episode ${episodeIndex} done returns to zero after its terminal frame`);
      }
      effectiveLength = firstDone + 1;
    }
  }

  let success: boolean | null = null;
  if ("success" in sorted[0]) {
    const values = sorted.map((row) => metadataScalar(row, "success"));
    if (values.some((value) => value !== 0 && value !== 1)) {
      throw new Error(`Episode ${episodeIndex} has non-binary success`);
    }
    success = values.some((value) => value === 1);
  }
  return { effectiveLength, success };
}

/** Path of the data parquet file holding an episode's frame rows. */
function episodeDataPath(row: Record<string, unknown>): string {
  const chunkIndex = metadataScalar(row, "data/chunk_index");
  const fileIndex = metadataScalar(row, "data/file_index");
  return (
    `data/chunk-${String(chunkIndex).padStart(3, "0")}/` +
    `file-${String(fileIndex).padStart(3, "0")}.parquet`
  );
}

function needsFrameSummary(row: Record<string, unknown>): boolean {
  const hasOutcome = "stats/success/max" in row;
  const hasBoundary =
    "stats/is_valid/count" in row || "stats/done/count" in row;
  return !hasOutcome || !hasBoundary;
}

async function loadFrameSummaries(
  datasetId: string,
  episodeRows: Record<string, unknown>[]
): Promise<Map<number, EpisodeFrameSummary>> {
  const needed = episodeRows.filter(needsFrameSummary);
  if (needed.length === 0) return new Map();

  const episodesByPath = new Map<string, Set<number>>();
  for (const row of needed) {
    const episodeIndex = metadataScalar(row, "episode_index");
    const path = episodeDataPath(row);
    const episodeIndices = episodesByPath.get(path) ?? new Set<number>();
    episodeIndices.add(episodeIndex);
    episodesByPath.set(path, episodeIndices);
  }

  const summaries = new Map<number, EpisodeFrameSummary>();
  await Promise.all(
    [...episodesByPath].map(async ([path, episodeIndices]) => {
      const frameRows = await readParquet(`${hfBase(datasetId)}/${path}`);
      for (const episodeIndex of episodeIndices) {
        const rows = frameRows.filter(
          (row) => metadataScalar(row, "episode_index") === episodeIndex
        );
        summaries.set(episodeIndex, summarizeEpisodeFrames(rows, episodeIndex));
      }
    })
  );
  return summaries;
}

function parseEpisodeRows(
  rows: Record<string, unknown>[],
  videoColPrefix: string | null,
  frameSummaries: Map<number, EpisodeFrameSummary>,
  fps = FPS
): { episodes: EpisodeWithoutSuccess[]; successMap: Map<number, boolean> } {
  const successMap = new Map<number, boolean>();
  const episodes = rows.map((row) => {
    const episodeIndex = metadataScalar(row, "episode_index");
    const rawLength = metadataScalar(row, "length");
    const frameSummary = frameSummaries.get(episodeIndex);
    const numFrames =
      frameSummary?.effectiveLength ?? effectiveEpisodeLengthFromStats(row);
    const rawFromTimestamp = videoColPrefix
      ? metadataScalar(row, `${videoColPrefix}/from_timestamp`)
      : 0;
    const rawToTimestamp = videoColPrefix
      ? metadataScalar(row, `${videoColPrefix}/to_timestamp`)
      : rawLength / fps;
    const frameDuration =
      rawToTimestamp > rawFromTimestamp
        ? (rawToTimestamp - rawFromTimestamp) / rawLength
        : 1 / fps;
    const toTimestamp = rawFromTimestamp + numFrames * frameDuration;
    const success = successFromEpisodeStats(row) ?? frameSummary?.success ?? null;
    if (success !== null) successMap.set(episodeIndex, success);

    return {
      episodeIndex,
      numFrames,
      duration: numFrames * frameDuration,
      videoFileIndex: videoColPrefix
        ? metadataScalar(row, `${videoColPrefix}/file_index`)
        : 0,
      fromTimestamp: rawFromTimestamp,
      toTimestamp,
    };
  });
  return { episodes, successMap };
}

// ── Module-level cache for parquet metadata ──
const parquetCache = new Map<string, ParquetCacheEntry>();
const fullMetadataRequests = new Map<string, Promise<ParquetCacheEntry>>();

export function coalesceFullMetadataRequest(
  datasetId: string,
  load: () => Promise<ParquetCacheEntry>
): Promise<ParquetCacheEntry> {
  const existing = fullMetadataRequests.get(datasetId);
  if (existing) return existing;

  const request = load();
  fullMetadataRequests.set(datasetId, request);
  const clear = () => {
    if (fullMetadataRequests.get(datasetId) === request) {
      fullMetadataRequests.delete(datasetId);
    }
  };
  request.then(clear, clear);
  return request;
}

/**
 * Fetch only the parquet metadata needed for a specific set of episode indices.
 * Returns cached results instantly on hit. Fetches parquet files in batches of 4
 * with early termination once all needed episodes are found.
 */
export async function fetchEpisodeSubset(
  datasetId: string,
  neededIndices: Set<number>
): Promise<{ episodes: EpisodeWithoutSuccess[]; cameraKeys: string[] }> {
  // Return full cache hit
  const cached = parquetCache.get(datasetId);
  const missingFromCache = cached
    ? missingEpisodeIndices(cached.episodes, neededIndices)
    : [...neededIndices];
  if (cached && missingFromCache.length === 0) {
    return cached;
  }
  if (cached?.complete) {
    assertEpisodeCoverage(datasetId, cached.episodes, neededIndices);
  }

  // Read the first parquet file to discover camera keys and column schema
  const metadataFiles = await listEpisodeMetadataFiles(datasetId);
  const firstRows = await readParquet(`${hfBase(datasetId)}/${metadataFiles[0]}`);

  const cameraKeys =
    firstRows.length > 0 ? discoverCameraKeys(Object.keys(firstRows[0])) : [];
  const videoColPrefix =
    cameraKeys.length > 0 ? `videos/${cameraKeys[0]}` : null;

  const firstFrameSummaries = await loadFrameSummaries(datasetId, firstRows);
  const firstParsed = parseEpisodeRows(firstRows, videoColPrefix, firstFrameSummaries, datasetFps(datasetId));
  const allEpisodes = firstParsed.episodes;
  const successMap = firstParsed.successMap;
  const found = new Set(allEpisodes.map((e) => e.episodeIndex));
  let filesRead = 1;

  // Check if we already have all needed episodes
  const allFound = () => missingFromCache.every((idx) => found.has(idx));

  if (!allFound()) {
    // Fetch additional parquet files in batches of 4, with early termination.
    // Paths come from the Hub tree, so a network/read failure is not mistaken
    // for the end of a guessed numbered sequence.
    const BATCH_SIZE = 4;
    const remainingFiles = metadataFiles.slice(1);
    for (let batchStart = 0; batchStart < remainingFiles.length; batchStart += BATCH_SIZE) {
      const batchEnd = Math.min(batchStart + BATCH_SIZE, remainingFiles.length);
      const batchResults = await Promise.all(
        remainingFiles
          .slice(batchStart, batchEnd)
          .map((path) => readParquet(`${hfBase(datasetId)}/${path}`))
      );
      filesRead += batchResults.length;

      for (const rows of batchResults) {
        const frameSummaries = await loadFrameSummaries(datasetId, rows);
        const parsed = parseEpisodeRows(rows, videoColPrefix, frameSummaries, datasetFps(datasetId));
        allEpisodes.push(...parsed.episodes);
        for (const [episodeIndex, success] of parsed.successMap) {
          successMap.set(episodeIndex, success);
        }
        for (const ep of parsed.episodes) found.add(ep.episodeIndex);
      }

      if (allFound()) break;
    }
  }

  // Another request may have populated more of this repository while this one
  // was in flight. Never replace a complete entry with a subset, and merge
  // partial entries so later previews retain every episode already fetched.
  const latest = parquetCache.get(datasetId);
  if (latest?.complete) {
    assertEpisodeCoverage(datasetId, latest.episodes, neededIndices);
    return latest;
  }

  const mergedEpisodes = new Map<number, EpisodeWithoutSuccess>();
  for (const episode of latest?.episodes ?? []) {
    mergedEpisodes.set(episode.episodeIndex, episode);
  }
  for (const episode of allEpisodes) {
    mergedEpisodes.set(episode.episodeIndex, episode);
  }
  const mergedSuccessMap = new Map(latest?.successMap ?? []);
  for (const [episodeIndex, success] of successMap) {
    mergedSuccessMap.set(episodeIndex, success);
  }
  const result = {
    episodes: [...mergedEpisodes.values()].sort(
      (a, b) => a.episodeIndex - b.episodeIndex
    ),
    cameraKeys: cameraKeys.length > 0 ? cameraKeys : (latest?.cameraKeys ?? []),
    successMap: mergedSuccessMap,
    complete: filesRead === metadataFiles.length,
  };

  parquetCache.set(datasetId, result);
  // Cache a complete scan before reporting an absent requested episode. A
  // retry can then fail from the local index instead of downloading it again.
  assertEpisodeCoverage(datasetId, result.episodes, neededIndices);
  return result;
}

/** Get the module-level parquet cache (for initializing component state). */
export function getParquetCache(): ReadonlyMap<
  string,
  ParquetCacheEntry
> {
  return parquetCache;
}

async function loadParquetMetadata(
  datasetId: string
): Promise<ParquetCacheEntry> {
  // Read the first parquet file to discover camera keys and column schema
  const metadataFiles = await listEpisodeMetadataFiles(datasetId);
  const firstRows = await readParquet(`${hfBase(datasetId)}/${metadataFiles[0]}`);

  const cameraKeys =
    firstRows.length > 0 ? discoverCameraKeys(Object.keys(firstRows[0])) : [];

  // Fetch every file explicitly listed by the Hub. Do not cap the sequence:
  // older datasets often shard episode metadata across many small files.
  const additionalResults = await Promise.all(
    metadataFiles
      .slice(1)
      .map((path) => readParquet(`${hfBase(datasetId)}/${path}`))
  );

  const allRows = [...firstRows];
  for (const rows of additionalResults) {
    allRows.push(...rows);
  }

  // Use the first camera key for video timing metadata
  const videoColPrefix =
    cameraKeys.length > 0 ? `videos/${cameraKeys[0]}` : null;

  const frameSummaries = await loadFrameSummaries(datasetId, allRows);
  const parsed = parseEpisodeRows(allRows, videoColPrefix, frameSummaries, datasetFps(datasetId));

  const result = {
    episodes: parsed.episodes.sort((a, b) => a.episodeIndex - b.episodeIndex),
    cameraKeys,
    successMap: parsed.successMap,
    complete: true,
  };

  // Store in module-level cache
  parquetCache.set(datasetId, result);

  return result;
}

export function fetchParquetMetadata(
  datasetId: string
): Promise<{ episodes: EpisodeWithoutSuccess[]; cameraKeys: string[] }> {
  const cached = parquetCache.get(datasetId);
  if (cached?.complete) return Promise.resolve(cached);
  return coalesceFullMetadataRequest(datasetId, () =>
    loadParquetMetadata(datasetId)
  );
}

export async function fetchSuccessStatus(
  datasetId: string
): Promise<Map<number, boolean>> {
  if (!parquetCache.get(datasetId)?.complete) {
    await fetchParquetMetadata(datasetId);
  }
  const cached = parquetCache.get(datasetId)!;
  if (cached.successMap.size !== cached.episodes.length) {
    const missing = cached.episodes
      .filter((episode) => !cached.successMap.has(episode.episodeIndex))
      .map((episode) => episode.episodeIndex);
    throw new Error(
      `${datasetId} is missing episode-level stats/success/max for episodes ${missing.join(", ")}`
    );
  }
  return new Map(cached.successMap);
}

// ---------------------------------------------------------------------------
// Outcome review (web port of mulligan/tools/outcome_review.py)
//
// The review UI needs a strictly RAW view of every episode: the videos contain
// `length` frames, so the scrub range is [0, rawLength) — NOT the trimmed
// `numFrames` the explorer shows. It also needs per-camera video timing so all
// station views can be seeked frame-exactly, and the frame-level reward/done/
// is_valid signals the desktop editor reads out of the data parquet.
// ---------------------------------------------------------------------------

export interface ReviewCameraTiming {
  fileIndex: number;
  fromTimestamp: number;
  toTimestamp: number;
}

export interface ReviewEpisode {
  episodeIndex: number;
  /** RAW episode length: the number of frames actually present in the videos. */
  rawLength: number;
  /** Data parquet holding this episode's frame rows (reward/done/is_valid). */
  dataPath: string;
  perCamera: Record<string, ReviewCameraTiming>;
}

export type DetectedOutcome = "success" | "failure" | "timeout";

export interface EpisodeFrameSignals {
  detectedOutcome: DetectedOutcome;
  /** Last frame_index with is_valid==1 (rawLength-1 when there is no column). */
  lastValidFrame: number;
  /** Number of leading is_valid==1 frames. */
  validLength: number;
  /** First done==1 frame inside the valid prefix — the existing outcome mark. */
  doneOnsetFrame: number | null;
  /** Mid-episode reward spikes (reward>0.5 with done==0): subtask rewards. */
  rewardSpikeFrames: number[];
}

const reviewEpisodeCache = new Map<string, ReviewEpisode[]>();
const frameSignalCache = new Map<string, EpisodeFrameSignals>();
const frameSignalFileLoads = new Map<string, Promise<void>>();

function frameSignalKey(datasetId: string, episodeIndex: number): string {
  return `${datasetId}::${episodeIndex}`;
}

/**
 * Raw per-episode metadata for outcome review: raw length, the data parquet
 * path, and timing for EVERY camera (the explorer only reads camera 0).
 */
export async function fetchReviewEpisodes(
  datasetId: string
): Promise<ReviewEpisode[]> {
  const cached = reviewEpisodeCache.get(datasetId);
  if (cached) return cached;

  const metadataFiles = await listEpisodeMetadataFiles(datasetId);
  const perFileRows = await Promise.all(
    metadataFiles.map((path) => readParquet(`${hfBase(datasetId)}/${path}`))
  );
  const rows = perFileRows.flat();
  if (rows.length === 0) {
    throw new Error(`${datasetId} has no episode metadata rows`);
  }

  const cameraKeys = discoverAllCameraKeys(Object.keys(rows[0]));
  if (cameraKeys.length === 0) {
    throw new Error(`${datasetId} has no video columns to review`);
  }

  const episodes = rows.map((row) => {
    const rawLength = metadataScalar(row, "length");
    if (!Number.isInteger(rawLength) || rawLength < 1) {
      throw new Error(`Episode length must be a positive integer, got ${rawLength}`);
    }
    const perCamera: Record<string, ReviewCameraTiming> = {};
    for (const key of cameraKeys) {
      perCamera[key] = {
        fileIndex: metadataScalar(row, `videos/${key}/file_index`),
        fromTimestamp: metadataScalar(row, `videos/${key}/from_timestamp`),
        toTimestamp: metadataScalar(row, `videos/${key}/to_timestamp`),
      };
    }
    return {
      episodeIndex: metadataScalar(row, "episode_index"),
      rawLength,
      dataPath: episodeDataPath(row),
      perCamera,
    };
  });

  episodes.sort((a, b) => a.episodeIndex - b.episodeIndex);
  reviewEpisodeCache.set(datasetId, episodes);
  return episodes;
}

/**
 * Frame-level outcome signals for one episode.
 *
 * Mirrors `detect_episode_outcome` / `detect_existing_outcome_frame` /
 * `last_valid_frame_index` in mulligan/tools/outcome_review.py: the outcome is read
 * off the LAST is_valid==1 frame (reward>0.5 && done==1 -> success, done==1 ->
 * failure, otherwise timeout).
 */
function computeFrameSignals(
  rows: Record<string, unknown>[],
  episodeIndex: number
): EpisodeFrameSignals {
  if (rows.length === 0) {
    throw new Error(`Episode ${episodeIndex} has no frame rows`);
  }
  const sorted = [...rows].sort(
    (a, b) => metadataScalar(a, "frame_index") - metadataScalar(b, "frame_index")
  );
  for (let index = 0; index < sorted.length; index++) {
    if (metadataScalar(sorted[index], "frame_index") !== index) {
      throw new Error(`Episode ${episodeIndex} has non-contiguous frame indices`);
    }
  }

  const hasIsValid = "is_valid" in sorted[0];
  const hasDone = "done" in sorted[0];
  const hasReward = "reward" in sorted[0];

  let validLength = sorted.length;
  if (hasIsValid) {
    validLength = 0;
    let seenInvalid = false;
    for (const row of sorted) {
      const value = metadataScalar(row, "is_valid");
      if (value !== 0 && value !== 1) {
        throw new Error(`Episode ${episodeIndex} has non-binary is_valid`);
      }
      if (value === 0) {
        seenInvalid = true;
      } else {
        if (seenInvalid) {
          throw new Error(
            `Episode ${episodeIndex} has a valid frame after an invalid frame`
          );
        }
        validLength += 1;
      }
    }
    if (validLength === 0) {
      throw new Error(`Episode ${episodeIndex} has no valid frames`);
    }
  }

  const lastValidFrame = validLength - 1;
  const lastValid = sorted[lastValidFrame];
  const lastDone = hasDone ? metadataScalar(lastValid, "done") : 0;
  const lastReward = hasReward ? metadataScalar(lastValid, "reward") : 0;
  const detectedOutcome: DetectedOutcome =
    lastReward > 0.5 && lastDone === 1
      ? "success"
      : lastDone === 1
        ? "failure"
        : "timeout";

  let doneOnsetFrame: number | null = null;
  if (hasDone) {
    for (let index = 0; index < validLength; index++) {
      if (metadataScalar(sorted[index], "done") === 1) {
        doneOnsetFrame = index;
        break;
      }
    }
  }

  const rewardSpikeFrames: number[] = [];
  if (hasReward) {
    for (let index = 0; index < sorted.length; index++) {
      const reward = metadataScalar(sorted[index], "reward");
      const done = hasDone ? metadataScalar(sorted[index], "done") : 0;
      if (reward > 0.5 && done === 0) rewardSpikeFrames.push(index);
    }
  }

  return {
    detectedOutcome,
    lastValidFrame,
    validLength,
    doneOnsetFrame,
    rewardSpikeFrames,
  };
}

/**
 * Frame signals for one episode, cached per (datasetId, episodeIndex).
 *
 * One data parquet holds many episodes, so the file is read ONCE and every
 * episode it contains is cached; concurrent callers share the in-flight read.
 */
export async function fetchEpisodeFrameSignals(
  datasetId: string,
  dataPath: string,
  episodeIndex: number
): Promise<EpisodeFrameSignals> {
  const cached = frameSignalCache.get(frameSignalKey(datasetId, episodeIndex));
  if (cached) return cached;

  const fileKey = `${datasetId}::${dataPath}`;
  let load = frameSignalFileLoads.get(fileKey);
  if (!load) {
    load = (async () => {
      const frameRows = await readParquet(`${hfBase(datasetId)}/${dataPath}`);
      const byEpisode = new Map<number, Record<string, unknown>[]>();
      for (const row of frameRows) {
        const index = metadataScalar(row, "episode_index");
        const bucket = byEpisode.get(index);
        if (bucket) bucket.push(row);
        else byEpisode.set(index, [row]);
      }
      for (const [index, rows] of byEpisode) {
        frameSignalCache.set(
          frameSignalKey(datasetId, index),
          computeFrameSignals(rows, index)
        );
      }
    })().finally(() => {
      frameSignalFileLoads.delete(fileKey);
    });
    frameSignalFileLoads.set(fileKey, load);
  }
  await load;

  const signals = frameSignalCache.get(frameSignalKey(datasetId, episodeIndex));
  if (!signals) {
    throw new Error(
      `${dataPath} in ${datasetId} contains no rows for episode ${episodeIndex}`
    );
  }
  return signals;
}

// Known per-episode ledger sidecars, mirroring `_default_ledger_paths` in
// mulligan/tools/outcome_review.py. Absence of a ledger is genuinely optional
// (teleop-only repos have none); a present-but-malformed ledger throws.
const LEDGER_FILES = [
  "meta/blind_dagger_ledger.jsonl",
  "meta/protocol_quota_ledger.jsonl",
  "meta/teleop_manifest_ledger.jsonl",
];

// Arm label fields in --arm-key precedence order (build_episode_selector
// in mulligan/tools/outcome_review.py filters on any of these; for display we take
// the first present).
const ARM_LABEL_FIELDS = ["arm_key", "arm", "manifest_source", "manifest_sources"];

const ledgerArmCache = new Map<string, Map<number, string>>();

/** One episode's entry in the APPLIED (HF) outcome-edit record. */
export interface AppliedRecord {
  newOutcome: string;
  outcomeFrame: number;
  softTruncate: boolean;
  /** Present ⇔ reviewed in subtask mode (may be empty), like the progress file. */
  subtaskFrames: number[] | null;
}

export interface AppliedProgress {
  changed: Map<number, AppliedRecord>;
  skipped: Set<number>;
}

const appliedProgressCache = new Map<string, AppliedProgress | null>();

/** One event from the append-only label-provenance ledger (.label_history.jsonl).
 *  `payload` is taxonomy-blind: opaque here, interpreted by the renderer per
 *  label_kind (see mulligan/real/lifecycle/label_history.py). */
export interface LabelEvent {
  v: number;
  label_kind: string;
  episode_index: number;
  payload: Record<string, unknown>;
  prev?: Record<string, unknown>;
  source: { kind: string; agent: string | null; tool: string };
  evidence: Record<string, unknown>;
  ts: string;
  taxonomy?: string;
}

/**
 * The dataset's label-provenance ledger from HF main. Null when the dataset
 * has no ledger yet (404). Deliberately uncached: the ledger grows with every
 * apply, and provenance shown stale is worse than a small refetch.
 */
export async function fetchLabelHistory(
  datasetId: string
): Promise<LabelEvent[] | null> {
  const resp = await fetch(`${hfBase(datasetId)}/.label_history.jsonl`);
  if (resp.status === 404) return null;
  if (!resp.ok) {
    throw new Error(`Failed to fetch .label_history.jsonl: HTTP ${resp.status}`);
  }
  const text = await resp.text();
  const events: LabelEvent[] = [];
  for (const [i, line] of text.split("\n").entries()) {
    if (!line.trim()) continue;
    const event = JSON.parse(line) as LabelEvent;
    // Type-check, don't just presence-check: malformed values reach render
    // paths (Timeline .map, ts.slice) and would white-screen the session or
    // silently drop provenance (a float episode_index misses every Map get).
    const bad = (why: string): never => {
      throw new Error(`.label_history.jsonl line ${i + 1}: ${why}`);
    };
    if (typeof event.label_kind !== "string") bad("label_kind not a string");
    if (!Number.isInteger(event.episode_index)) bad("episode_index not an integer");
    if (typeof event.payload !== "object" || event.payload === null) bad("payload not an object");
    if (typeof event.ts !== "string") bad("ts not a string");
    if (
      typeof event.source !== "object" ||
      event.source === null ||
      typeof event.source.kind !== "string" ||
      typeof event.source.tool !== "string"
    ) {
      bad("source missing kind/tool strings");
    }
    events.push(event);
  }
  return events;
}

/**
 * The dataset's applied `.outcome_edit_progress.json` from HF — the record of
 * truth for treatment that already reached the Hub (apply worker or Python
 * editor). Null when the dataset has never been treated (404).
 */
export async function fetchAppliedProgress(
  datasetId: string
): Promise<AppliedProgress | null> {
  if (appliedProgressCache.has(datasetId)) {
    return appliedProgressCache.get(datasetId) ?? null;
  }
  const resp = await fetch(`${hfBase(datasetId)}/.outcome_edit_progress.json`);
  if (resp.status === 404) {
    appliedProgressCache.set(datasetId, null);
    return null;
  }
  if (!resp.ok) {
    throw new Error(`Failed to fetch .outcome_edit_progress.json: HTTP ${resp.status}`);
  }
  const raw = (await resp.json()) as {
    changed_episodes: Record<
      string,
      {
        new_outcome: string;
        outcome_frame: number;
        soft_truncate: boolean;
        subtask_frames?: number[];
      }
    >;
    skipped_episodes: number[];
  };
  const changed = new Map<number, AppliedRecord>();
  for (const [ep, record] of Object.entries(raw.changed_episodes)) {
    // These values feed pending state and Timeline rendering — a wrong TYPE
    // (string frame, non-array marks) white-screens the session, so fail
    // loud here where the banner is wired.
    if (typeof record.new_outcome !== "string" || typeof record.outcome_frame !== "number") {
      throw new Error(`.outcome_edit_progress.json ep ${ep}: malformed record`);
    }
    if (record.subtask_frames !== undefined && !Array.isArray(record.subtask_frames)) {
      throw new Error(`.outcome_edit_progress.json ep ${ep}: subtask_frames not an array`);
    }
    changed.set(Number(ep), {
      newOutcome: record.new_outcome,
      outcomeFrame: record.outcome_frame,
      softTruncate: record.soft_truncate,
      subtaskFrames: record.subtask_frames ?? null,
    });
  }
  const progress: AppliedProgress = {
    changed,
    skipped: new Set(raw.skipped_episodes.map(Number)),
  };
  appliedProgressCache.set(datasetId, progress);
  return progress;
}

/**
 * Per-episode arm labels from the dataset's HF ledger JSONLs. Episodes without
 * a ledger row (or repos without ledgers) are simply absent from the map.
 */
export async function fetchLedgerArms(
  datasetId: string
): Promise<Map<number, string>> {
  const cached = ledgerArmCache.get(datasetId);
  if (cached) return cached;

  const arms = new Map<number, string>();
  // Duplicate detection over ALL rows (labelled or not), mirroring the loud
  // _load_ledger_rows_by_episode failure in mulligan/tools/outcome_review.py.
  const seen = new Set<number>();
  for (const file of LEDGER_FILES) {
    const resp = await fetch(`${hfBase(datasetId)}/${file}`);
    if (resp.status === 404) continue;
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${file}: HTTP ${resp.status}`);
    }
    const text = await resp.text();
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      const row = JSON.parse(line) as Record<string, unknown>;
      if (row.episode_index == null) {
        throw new Error(`${file}: ledger row missing episode_index`);
      }
      const episodeIndex = Number(row.episode_index);
      if (seen.has(episodeIndex)) {
        throw new Error(
          `Duplicate episode_index ${episodeIndex} across ledger files`
        );
      }
      seen.add(episodeIndex);
      let label: string | null = null;
      for (const field of ARM_LABEL_FIELDS) {
        const value = row[field];
        if (value == null) continue;
        label = Array.isArray(value) ? value.map(String).join("+") : String(value);
        break;
      }
      if (label !== null) arms.set(episodeIndex, label);
    }
  }
  ledgerArmCache.set(datasetId, arms);
  return arms;
}
