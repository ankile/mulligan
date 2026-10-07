import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";

import type { Id } from "../release/api";
import RolloutSection, {
  type RolloutDataSource,
  type RolloutResult,
} from "./RolloutSection";

GlobalRegistrator.register();
const { act, cleanup, fireEvent, render } = await import("@testing-library/react");

afterEach(() => cleanup());
afterAll(() => GlobalRegistrator.unregister());

function episode(episodeIndex: number) {
  return {
    episodeIndex,
    numFrames: 15,
    duration: 1,
    videoFileIndex: 0,
    fromTimestamp: episodeIndex,
    toTimestamp: episodeIndex + 1,
  };
}

function result(repo: string, episodeIndex: number): RolloutResult {
  return {
    session_id: `session-${repo}` as Id<"evalSessions">,
    dataset_repo: repo,
    round_index: 0,
    episode_index: episodeIndex,
    success: true,
    num_frames: 15,
    session_creation_time: 0,
  };
}

function cacheEntry(indices: number[], complete: boolean) {
  return {
    episodes: indices.map(episode),
    cameraKeys: ["observation.images.side_1"],
    successMap: new Map(indices.map((index) => [index, true])),
    complete,
  };
}

describe("RolloutSection loading", () => {
  test("fetches when results add an episode absent from a partial cache", async () => {
    const cache = new Map([["org/repo", cacheEntry([1], false)]]);
    const requested: number[][] = [];
    const dataSource: RolloutDataSource = {
      getParquetCache: () => cache,
      fetchEpisodeSubset: async (_repo, indices) => {
        requested.push([...indices].sort((a, b) => a - b));
        return {
          episodes: [episode(1), episode(9)],
          cameraKeys: ["observation.images.side_1"],
        };
      },
    };

    const view = render(
      <RolloutSection
        results={[result("org/repo", 1)]}
        isOpen
        dataSource={dataSource}
      />
    );
    expect(requested).toEqual([]);

    await act(async () => {
      view.rerender(
        <RolloutSection
          results={[result("org/repo", 1), result("org/repo", 9)]}
          isOpen
          dataSource={dataSource}
        />
      );
    });

    expect(requested).toEqual([[1, 9]]);
    expect(view.container.querySelectorAll("video")).toHaveLength(2);
  });

  test("clears loading when an in-flight request is replaced by cached results", async () => {
    const cache = new Map([["org/cached", cacheEntry([2], true)]]);
    let resolvePending: ((value: {
      episodes: ReturnType<typeof episode>[];
      cameraKeys: string[];
    }) => void) | null = null;
    const dataSource: RolloutDataSource = {
      getParquetCache: () => cache,
      fetchEpisodeSubset: async () =>
        new Promise((resolve) => {
          resolvePending = resolve;
        }),
    };

    const view = render(
      <RolloutSection
        results={[result("org/pending", 1)]}
        isOpen
        dataSource={dataSource}
      />
    );
    expect(view.container.textContent).toContain("Loading video data...");

    await act(async () => {
      view.rerender(
        <RolloutSection
          results={[result("org/cached", 2)]}
          isOpen
          dataSource={dataSource}
        />
      );
    });
    expect(view.container.textContent).not.toContain("Loading video data...");

    await act(async () => {
      resolvePending?.({
        episodes: [episode(1)],
        cameraKeys: ["observation.images.side_1"],
      });
    });
  });

  test("keeps successful repositories visible and retries failed ones", async () => {
    const cache = new Map<string, ReturnType<typeof cacheEntry>>();
    let badAttempts = 0;
    const dataSource: RolloutDataSource = {
      getParquetCache: () => cache,
      fetchEpisodeSubset: async (repo) => {
        if (repo === "org/bad" && badAttempts++ === 0) {
          throw new Error("metadata unavailable");
        }
        const index = repo === "org/good" ? 1 : 2;
        return {
          episodes: [episode(index)],
          cameraKeys: ["observation.images.side_1"],
        };
      },
    };

    const view = render(
      <RolloutSection
        results={[result("org/good", 1), result("org/bad", 2)]}
        isOpen
        dataSource={dataSource}
      />
    );
    await act(async () => {});

    expect(view.container.querySelectorAll("video")).toHaveLength(1);
    expect(view.container.textContent).toContain(
      "org/bad: metadata unavailable"
    );
    expect(view.container.textContent).toContain("Video unavailable");

    await act(async () => {
      fireEvent.click(view.getByRole("button", { name: "Retry" }));
    });

    expect(view.container.querySelectorAll("video")).toHaveLength(2);
    expect(view.container.textContent).not.toContain("metadata unavailable");
    expect(view.container.textContent).not.toContain("Video unavailable");
  });
});
