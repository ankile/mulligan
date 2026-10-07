export const DATASET_STATS_ALGORITHM_VERSION = "episode-metadata-v1";

export type EpisodeStatsInput = {
  episodeIndex: number;
  rawLength: number;
  success: boolean;
  validFrames: number | null;
  doneFrames: number | null;
  humanFrames: number | null;
};

export type FrameStatsInput = {
  episodeIndex: number;
  success: number;
  isValid: number | null;
  done: number | null;
  source: number | null;
};

export type DatasetStatsSummary = {
  numEpisodes: number;
  totalFrames: number;
  totalDurationSeconds: number;
  numSuccess: number;
  numFailure: number;
  numHumanFrames: number | null;
  numPolicyFrames: number | null;
  numAutonomousSuccess: number | null;
};

function requireFrameCount(
  value: number,
  rawLength: number,
  name: string,
  episodeIndex: number
): void {
  if (!Number.isInteger(value) || value < 0 || value > rawLength) {
    throw new Error(
      `Episode ${episodeIndex} ${name} must be an integer in [0, ${rawLength}], got ${value}`
    );
  }
}

function requireBinary(value: number, name: string, episodeIndex: number): void {
  if (value !== 0 && value !== 1) {
    throw new Error(
      `Episode ${episodeIndex} ${name} must be 0 or 1, got ${value}`
    );
  }
}

export function episodeStatsFromFrames(
  frames: FrameStatsInput[]
): EpisodeStatsInput[] {
  if (frames.length === 0) throw new Error("Dataset has no frame rows");

  for (const feature of ["isValid", "done", "source"] as const) {
    const availability = new Set(frames.map((frame) => frame[feature] !== null));
    if (availability.size !== 1) {
      throw new Error(`Frame data has ${feature} for only part of the dataset`);
    }
  }

  const episodes = new Map<number, EpisodeStatsInput>();
  for (const frame of frames) {
    if (!Number.isInteger(frame.episodeIndex) || frame.episodeIndex < 0) {
      throw new Error(
        `Frame episode_index must be a non-negative integer, got ${frame.episodeIndex}`
      );
    }
    requireBinary(frame.success, "success", frame.episodeIndex);
    if (frame.isValid !== null) {
      requireBinary(frame.isValid, "is_valid", frame.episodeIndex);
    }
    if (frame.done !== null) requireBinary(frame.done, "done", frame.episodeIndex);
    if (frame.source !== null) {
      requireBinary(frame.source, "source", frame.episodeIndex);
    }

    const existing = episodes.get(frame.episodeIndex);
    if (existing === undefined) {
      episodes.set(frame.episodeIndex, {
        episodeIndex: frame.episodeIndex,
        rawLength: 1,
        success: frame.success === 1,
        validFrames: frame.isValid,
        doneFrames: frame.done,
        humanFrames: frame.source,
      });
      continue;
    }
    if (existing.success !== (frame.success === 1)) {
      throw new Error(
        `Episode ${frame.episodeIndex} has inconsistent frame-level success values`
      );
    }
    existing.rawLength += 1;
    if (existing.validFrames !== null) existing.validFrames += frame.isValid!;
    if (existing.doneFrames !== null) existing.doneFrames += frame.done!;
    if (existing.humanFrames !== null) existing.humanFrames += frame.source!;
  }

  return [...episodes.values()].sort((a, b) => a.episodeIndex - b.episodeIndex);
}

export function summarizeDatasetStats(
  rows: EpisodeStatsInput[],
  fps: number
): DatasetStatsSummary {
  if (rows.length === 0) throw new Error("Dataset has no episode metadata rows");
  if (!Number.isFinite(fps) || fps <= 0) {
    throw new Error(`Dataset fps must be positive, got ${fps}`);
  }

  const seenEpisodes = new Set<number>();
  const sourceAvailability = new Set(rows.map((row) => row.humanFrames !== null));
  if (sourceAvailability.size !== 1) {
    throw new Error("Episode metadata has source statistics for only part of the dataset");
  }
  const hasSourceStats = sourceAvailability.has(true);

  let totalFrames = 0;
  let effectiveFrames = 0;
  let numSuccess = 0;
  let numHumanFrames = 0;
  let numAutonomousSuccess = 0;

  for (const row of rows) {
    if (!Number.isInteger(row.episodeIndex) || row.episodeIndex < 0) {
      throw new Error(`Episode index must be a non-negative integer, got ${row.episodeIndex}`);
    }
    if (seenEpisodes.has(row.episodeIndex)) {
      throw new Error(`Duplicate episode_index ${row.episodeIndex}`);
    }
    seenEpisodes.add(row.episodeIndex);

    if (!Number.isInteger(row.rawLength) || row.rawLength < 1) {
      throw new Error(
        `Episode ${row.episodeIndex} length must be a positive integer, got ${row.rawLength}`
      );
    }
    totalFrames += row.rawLength;

    const validFrames = row.validFrames ?? row.rawLength;
    const doneFrames = row.doneFrames ?? 0;
    requireFrameCount(validFrames, row.rawLength, "valid frame count", row.episodeIndex);
    requireFrameCount(doneFrames, row.rawLength, "done frame count", row.episodeIndex);
    if (validFrames === 0) {
      throw new Error(`Episode ${row.episodeIndex} has no valid frames`);
    }
    const doneInclusiveLength =
      doneFrames > 0 ? row.rawLength - doneFrames + 1 : row.rawLength;
    const effectiveLength = Math.min(validFrames, doneInclusiveLength);
    if (effectiveLength < 1 || effectiveLength > row.rawLength) {
      throw new Error(
        `Episode ${row.episodeIndex} has invalid effective length ${effectiveLength}`
      );
    }
    effectiveFrames += effectiveLength;

    if (row.success) numSuccess += 1;
    if (hasSourceStats) {
      const humanFrames = row.humanFrames!;
      requireFrameCount(humanFrames, row.rawLength, "human frame count", row.episodeIndex);
      numHumanFrames += humanFrames;
      if (row.success && humanFrames === 0) numAutonomousSuccess += 1;
    }
  }

  return {
    numEpisodes: rows.length,
    totalFrames,
    totalDurationSeconds: effectiveFrames / fps,
    numSuccess,
    numFailure: rows.length - numSuccess,
    numHumanFrames: hasSourceStats ? numHumanFrames : null,
    numPolicyFrames: hasSourceStats ? totalFrames - numHumanFrames : null,
    numAutonomousSuccess: hasSourceStats ? numAutonomousSuccess : null,
  };
}
