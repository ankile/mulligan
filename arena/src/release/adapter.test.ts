import { describe, expect, test } from "bun:test";
import { createReleaseAdapter } from "./adapter";
import type { UISnapshot } from "./adapter";
import type { Release } from "./types";
const release: Release = {
  schemaVersion: 1,
  version: "release-1",
  state: "public_verified",
  selectionSha256: "frozen",
  sourceSha256: {},
  datasets: [
    {
      id: "mulligan/eval",
      task: "real-marker-d2",
      role: "evaluation",
      variant: null,
      episodes: 2,
      frames: 100,
      fps: 15,
      cameras: [],
      parent: null,
      source: "source/eval",
      revision: "a".repeat(40),
      tier: "mainline",
    },
  ],
  tasks: [
    {
      id: "marker_d2",
      title: "Marker",
      domain: "real",
      datasetTask: "real-marker-d2",
      metric: "full_success",
      policies: ["a", "b"].map((id, i) => ({
        id,
        round: "R0",
        method: id,
        arm: id,
        modelId: id,
        rate: 1 - i,
        lo: 0,
        hi: 1,
        successes: 1 - i,
        episodes: 1,
        provenance: "pinned",
        blocks: ["block"],
      })),
      blocks: [
        {
          id: "block",
          round: "R0",
          dataset: "mulligan/eval",
          source: "source/eval",
          revision: "a".repeat(40),
          fps: 15,
          cameras: [],
          reviewedEpisodes: null,
          sourcePath: "pinned",
          starts: [
            {
              index: 7,
              results: ["a", "b"].map((policyId, i) => ({
                policyId,
                episode: 10 + i,
                success: i === 0,
                outcome: i === 0 ? "success" : "failure",
                steps: 20 + i,
                frames: 50,
                score: 1 - i,
                marks: null,
                videos: {},
              })),
            },
          ],
        },
      ],
    },
  ],
};
const ui: UISnapshot = {
  selectionSha256: "frozen",
  datasets: [{ repo: "mulligan/eval", created: 123, sourceType: "eval" }],
  blocks: [{ id: "block", created: 456, mode: "manual" }],
  coverage: [],
};
describe("original Arena release adapter", () => {
  test("rejects mismatched snapshots and unknown queries or repositories", () => {
    expect(() =>
      createReleaseAdapter(release, { ...ui, selectionSha256: "changed" }),
    ).toThrow("mismatch");
    const a = createReleaseAdapter(release, ui);
    expect(() =>
      a.resolve("datasets:getByRepo", { repo_id: "source/pilot" }),
    ).toThrow("Not in release");
    expect(() => a.resolve("reviews:latestForRepo", {})).toThrow(
      "not available",
    );
    expect(a.resolve("users:viewer")).toBeNull();
  });
  test("preserves paired initial-state identity, selected episodes, and corrected step counts", () => {
    const a = createReleaseAdapter(release, ui);
    expect(a.resolve("evalSessions:getDetail", { id: "block" })).toMatchObject({
      dataset_repo: "mulligan/eval",
      rounds: [
        {
          index: 7,
          results: [
            {
              policy_id: "a",
              episode_index: 10,
              success: true,
              num_frames: 20,
            },
            {
              policy_id: "b",
              episode_index: 11,
              success: false,
              num_frames: 21,
            },
          ],
        },
      ],
    });
    expect(a.resolve("ratings:sessionOutcomes")).toMatchObject([
      {
        perPolicy: [
          { policy_id: "a", rollouts: 1, successes: 1, successFramesSum: 20 },
          { policy_id: "b", rollouts: 1, successes: 0 },
        ],
      },
    ]);
    expect(
      a.resolve("pairings:listRounds", { policyIdA: "b", policyIdB: "a" }),
    ).toMatchObject([
      { roundIndex: 7, results: [{ policyId: "b" }, { policyId: "a" }] },
    ]);
    expect(a.resolve("pairings:listRounds", { policyIdA: "outside" })).toEqual(
      [],
    );
  });
  test("memoizes query values for the original effect and joined-view dependencies", () => {
    const a = createReleaseAdapter(release, ui);
    expect(a.resolve("evalSessions:getDetail", { id: "block" })).toBe(
      a.resolve("evalSessions:getDetail", { id: "block" }),
    );
  });
});

test("simulation cannot silently omit computed statistics", () => {
  const simRelease = {...release, tasks: [{...release.tasks[0], domain: "sim" as const}]};
  expect(() => createReleaseAdapter(simRelease, ui)).toThrow("simulation statistics");
  expect(() => createReleaseAdapter(simRelease, ui, {
    schemaVersion: 1, selectionSha256: "frozen", sessions: [], evidence: [],
  })).toThrow("Incomplete simulation statistics");
});

test("round-dataset sessions keep start IDs for joins and number rows by the session's Sobol block", () => {
  const block = release.tasks[0].blocks[0];
  const withSession: Release = {
    ...release,
    tasks: [
      {
        ...release.tasks[0],
        blocks: [
          {
            ...block,
            session: "b03",
            roundDataset: "mulligan/eval",
            label: "Session b03 · Jul 3 · starts 51–100 · pooled",
            startRange: [51, 100],
            starts: [
              {
                ...block.starts[0],
                index: 107,
                manifestIndex: 7,
                results: block.starts[0].results.map((r) => ({ ...r, visit: "20260907T144237-d212838a" })),
              },
            ],
          },
        ],
      },
    ],
  };
  const a = createReleaseAdapter(withSession, ui);
  expect(a.resolve("evalSessions:getDetail", { id: "block" })).toMatchObject({
    label: "Session b03 · Jul 3 · starts 51–100 · pooled",
    round_dataset: "mulligan/eval",
    rounds: [
      {
        index: 107,
        label: "Round 58",
        results: [{ visit_id: "20260907T144237-d212838a" }, { visit_id: "20260907T144237-d212838a" }],
      },
    ],
  });
  expect(a.resolve("pairings:listRounds", { policyIdA: "a" })).toMatchObject([
    { roundIndex: 107, label: "Round 58", sessionLabel: "Session b03 · Jul 3 · starts 51–100 · pooled" },
  ]);
});
