import { describe, expect, test } from "bun:test";

import {
  assertEpisodeCoverage,
  binaryTrueCountFromEpisodeStats,
  coalesceFullMetadataRequest,
  effectiveEpisodeLengthFromStats,
  explorerCameraKeys,
  missingEpisodeIndices,
  summarizeEpisodeFrames,
  successFromEpisodeStats,
} from "./hf-api";

describe("episode subset coverage", () => {
  test("reports requested indices missing from a partial cache", () => {
    const cached = [{ episodeIndex: 1 }, { episodeIndex: 4 }];
    expect(missingEpisodeIndices(cached, new Set([1, 4, 9]))).toEqual([9]);
  });

  test("detects a newly requested episode after an earlier cache hit", () => {
    const cached = [{ episodeIndex: 2 }];
    expect(missingEpisodeIndices(cached, new Set([2]))).toEqual([]);
    expect(missingEpisodeIndices(cached, new Set([2, 7]))).toEqual([7]);
  });

  test("sorts missing indices for deterministic diagnostics", () => {
    expect(
      missingEpisodeIndices([{ episodeIndex: 3 }], new Set([9, 3, 1]))
    ).toEqual([1, 9]);
  });

  test("rejects requested indices absent from a complete scan", () => {
    expect(() =>
      assertEpisodeCoverage(
        "org/dataset",
        [{ episodeIndex: 3 }],
        new Set([3, 11, 7])
      )
    ).toThrow(
      "org/dataset does not contain requested episode indices: 7, 11"
    );
  });
});

describe("full metadata request coalescing", () => {
  test("shares one in-flight request per repository and clears it on success", async () => {
    let loads = 0;
    let resolveLoad: (() => void) | null = null;
    const load = async () => {
      loads += 1;
      await new Promise<void>((resolve) => {
        resolveLoad = resolve;
      });
      return {
        episodes: [],
        cameraKeys: [],
        successMap: new Map<number, boolean>(),
        complete: true,
      };
    };

    const first = coalesceFullMetadataRequest("test/coalesce-success", load);
    const second = coalesceFullMetadataRequest("test/coalesce-success", load);
    expect(first).toBe(second);
    expect(loads).toBe(1);
    resolveLoad?.();
    await first;

    await coalesceFullMetadataRequest("test/coalesce-success", async () => {
      loads += 1;
      return {
        episodes: [],
        cameraKeys: [],
        successMap: new Map<number, boolean>(),
        complete: true,
      };
    });
    expect(loads).toBe(2);
  });

  test("clears a failed request so callers can retry", async () => {
    const failed = coalesceFullMetadataRequest(
      "test/coalesce-failure",
      async () => {
        throw new Error("network failed");
      }
    );
    await expect(failed).rejects.toThrow("network failed");

    const recovered = await coalesceFullMetadataRequest(
      "test/coalesce-failure",
      async () => ({
        episodes: [],
        cameraKeys: [],
        successMap: new Map<number, boolean>(),
        complete: true,
      })
    );
    expect(recovered.complete).toBe(true);
  });
});

function binaryStats(feature: "done" | "is_valid", values: number[]) {
  const count = values.length;
  const sum = values.reduce((total, value) => total + value, 0);
  return {
    [`stats/${feature}/min`]: [Math.min(...values)],
    [`stats/${feature}/max`]: [Math.max(...values)],
    [`stats/${feature}/mean`]: [sum / count],
    [`stats/${feature}/count`]: [count],
  };
}

describe("episode metadata outcomes", () => {
  test("reads episode success from stats/success/max", () => {
    expect(successFromEpisodeStats({ "stats/success/max": [1] })).toBe(true);
    expect(successFromEpisodeStats({ "stats/success/max": [0] })).toBe(false);
    expect(successFromEpisodeStats({})).toBeNull();
  });

  test("rejects non-binary success metadata", () => {
    expect(() => successFromEpisodeStats({ "stats/success/max": [0.5] })).toThrow(
      "must be 0 or 1"
    );
  });
});

describe("Data Explorer cameras", () => {
  test("keeps modern policy-facing roles and omits wrist_right", () => {
    expect(
      explorerCameraKeys([
        "observation.images.wrist_left",
        "observation.images.wrist_right",
        "observation.images.side_1",
        "observation.images.side_2",
      ])
    ).toEqual([
      "observation.images.wrist_left",
      "observation.images.side_1",
      "observation.images.side_2",
    ]);
  });

  test("keeps only the left eye for serial-named stereo cameras", () => {
    expect(
      explorerCameraKeys([
        "observation.images.10000001_left",
        "observation.images.10000001_right",
        "observation.images.20000002_left",
        "observation.images.20000002_right",
      ])
    ).toEqual([
      "observation.images.10000001_left",
      "observation.images.20000002_left",
    ]);
  });

  test("does not partially filter a mixed or unfamiliar camera contract", () => {
    expect(
      explorerCameraKeys([
        "observation.images.10000001_left",
        "observation.images.overhead",
      ])
    ).toEqual([
      "observation.images.10000001_left",
      "observation.images.overhead",
    ]);
  });
});

describe("effective episode length", () => {
  test("includes the first done frame and drops its retained tail", () => {
    const row = {
      length: 6,
      ...binaryStats("done", [0, 0, 1, 1, 1, 1]),
      ...binaryStats("is_valid", [1, 1, 1, 1, 1, 0]),
    };
    expect(effectiveEpisodeLengthFromStats(row)).toBe(3);
  });

  test("ends at the last valid frame when there is no done frame", () => {
    const row = {
      length: 5,
      ...binaryStats("done", [0, 0, 0, 0, 0]),
      ...binaryStats("is_valid", [1, 1, 1, 0, 0]),
    };
    expect(effectiveEpisodeLengthFromStats(row)).toBe(3);
  });

  test("uses raw length for legacy metadata without terminal features", () => {
    expect(effectiveEpisodeLengthFromStats({ length: 7 })).toBe(7);
  });

  test("recovers exact binary counts from floating-point means", () => {
    const row = {
      ...binaryStats("is_valid", [1, 1, 1, 0, 0]),
    };
    expect(binaryTrueCountFromEpisodeStats(row, "is_valid", 5)).toBe(3);
  });

  test("fails on partial or inconsistent terminal metadata", () => {
    expect(() =>
      effectiveEpisodeLengthFromStats({
        length: 5,
        "stats/is_valid/mean": [0.8],
      })
    ).toThrow("incomplete is_valid statistics");

    expect(() =>
      effectiveEpisodeLengthFromStats({
        length: 5,
        ...binaryStats("done", [0, 0, 1, 1]),
      })
    ).toThrow("does not match episode length");
  });
});

describe("legacy per-frame fallback", () => {
  test("uses first done or last valid frame and reads success", () => {
    const rows = [
      { frame_index: 0, done: 0, is_valid: 1, success: 1 },
      { frame_index: 1, done: 1, is_valid: 1, success: 1 },
      { frame_index: 2, done: 1, is_valid: 0, success: 1 },
      { frame_index: 3, done: 1, is_valid: 0, success: 1 },
    ];
    expect(summarizeEpisodeFrames(rows, 7)).toEqual({
      effectiveLength: 2,
      success: true,
    });
  });

  test("fails on a non-prefix validity mask", () => {
    const rows = [
      { frame_index: 0, done: 0, is_valid: 1, success: 0 },
      { frame_index: 1, done: 0, is_valid: 0, success: 0 },
      { frame_index: 2, done: 0, is_valid: 1, success: 0 },
    ];
    expect(() => summarizeEpisodeFrames(rows, 9)).toThrow(
      "valid frame after an invalid frame"
    );
  });
});
