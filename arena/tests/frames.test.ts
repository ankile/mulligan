import { describe, expect, test } from "bun:test";

import {
  applyOutcomeEdits,
  buildEpisodeMap,
  detectFrameOutcome,
  episodeOutcomesByIndex,
  type FileFrameColumns,
} from "../convex/apply/frames";

function episodeWithTerminalPadding(): FileFrameColumns {
  return {
    path: "data/chunk-000/file-000.parquet",
    numRows: 6,
    episodeIndex: new Float64Array([0, 0, 0, 0, 0, 0]),
    frameIndex: new Float64Array([0, 1, 2, 3, 4, 5]),
    reward: new Float32Array(6),
    done: new Float64Array(6),
    success: new Float64Array(6),
    isValid: new Float64Array([1, 1, 1, 1, 1, 0]),
    dirty: false,
  };
}

describe("outcome frame and subtask frame boundaries", () => {
  test("repeat applies do not dirty unchanged columns, but repair differing stored labels", () => {
    const file = episodeWithTerminalPadding();
    const episodes = buildEpisodeMap([file]);
    const progress = {
      changed_episodes: {
        "0": { new_outcome: "success" as const, outcome_frame: 3, soft_truncate: true, subtask_frames: [1] },
      },
      skipped_episodes: [],
    };
    applyOutcomeEdits(episodes, progress, 1);
    expect(file.dirty).toBe(true);
    const expected = structuredClone(file);
    file.dirty = false;
    applyOutcomeEdits(episodes, progress, 1);
    expect(file.dirty).toBe(false);
    for (const column of ["success", "reward", "done", "isValid"] as const) {
      file[column]![1] = 99;
      applyOutcomeEdits(episodes, progress, 1);
      expect(file.dirty).toBe(true);
      expect(file[column]).toEqual(expected[column]);
      file.dirty = false;
    }
  });
  test("allows a timeout subtask on the final valid frame after padding normalization", () => {
    const file = episodeWithTerminalPadding();
    const episodes = buildEpisodeMap([file]);

    expect(
      applyOutcomeEdits(
        episodes,
        {
          changed_episodes: {
            "0": {
              new_outcome: "timeout",
              outcome_frame: 5,
              soft_truncate: false,
              subtask_frames: [4],
            },
          },
          skipped_episodes: [],
        },
        1
      )
    ).toBe(true);

    expect(Array.from(file.reward)).toEqual([0, 0, 0, 0, 1, 0]);
    expect(Array.from(file.done)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(Array.from(file.isValid!)).toEqual([1, 1, 1, 1, 1, 0]);
    expect(detectFrameOutcome(episodes.get(0)!, [4])).toEqual({
      outcome: "timeout",
      expectedNumSteps: 5,
    });
    expect(episodeOutcomesByIndex(episodes, new Map([[0, [4]]])).get(0)).toBe("timeout");
  });

  test("still rejects a terminal failure and subtask on the same frame", () => {
    const file = episodeWithTerminalPadding();
    const episodes = buildEpisodeMap([file]);

    expect(() =>
      applyOutcomeEdits(
        episodes,
        {
          changed_episodes: {
            "0": {
              new_outcome: "failure",
              outcome_frame: 4,
              soft_truncate: false,
              subtask_frames: [4],
            },
          },
          skipped_episodes: [],
        },
        1
      )
    ).toThrow("equality is allowed only for timeout");
  });
});
