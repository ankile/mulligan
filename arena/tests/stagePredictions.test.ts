import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { convexTest, type TestConvex } from "convex-test";

import { api } from "../convex/_generated/api";
import type { Id } from "../convex/_generated/dataModel";
import schema from "../convex/schema";
import { manifestDigest, predictionDigest } from "../convex/stagePredictionContract";
import fixtures from "./fixtures/stage-consistency-fixtures.json";

// The module map is explicit because Bun has no Vite import.meta.glob transform.
// All writes below use convex-test's private, in-memory database. No deployment
// URL or credentials are loaded, and no HTTP request leaves the test process.
const modules = {
  "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
  "../convex/stagePredictions.ts": () => import("../convex/stagePredictions"),
  "../convex/stageTaskSpecs.ts": () => import("../convex/stageTaskSpecs"),
  "../convex/stageReviews.ts": () => import("../convex/stageReviews"),
  "../convex/stageCoverage.ts": () => import("../convex/stageCoverage"),
};

const SERVICE_TOKEN = "stage-prediction-test-only-bridge-token";
const EDITOR_SUB = "stage-prediction-test-editor";
const REPO = "test/routing-immutable-predictions";
const TASK = "routing_d2";
const fixture = fixtures["routing_d2@s10_v1"];
const spec = fixture.spec;
const cleanLabel = fixture.fixtures.find((entry) => entry.name === "clean_success")!.row;
const pipeline = { name: "test-pipeline", version: "v1", git_commit: "a".repeat(40) };
const service = { serviceToken: SERVICE_TOKEN };

interface Prediction {
  episode_index: bigint;
  label: Record<string, unknown>;
  episode_duration_s: number;
  evidence: Record<string, unknown>;
  canonical_response: Record<string, unknown>;
  source_revision: string;
}

function prediction(episode: number, maxStage = 10): Prediction {
  return {
    episode_index: BigInt(episode),
    label: { ...cleanLabel, max_stage: maxStage },
    episode_duration_s: 20,
    evidence: { artifact_uri: "s3://test-only/predictions.jsonl", ordinal: episode },
    canonical_response: { max_stage: { stage_index: maxStage }, schema_version: "test/v1" },
    source_revision: "b".repeat(40),
  };
}

type RunId = Id<"stagePredictionRuns">;
type Store = TestConvex<typeof schema>;
let t: Store;
let oldServiceToken: string | undefined;
let oldEditorSubs: string | undefined;

async function runArguments(rows: Prediction[], runKey = "run-v1") {
  return {
    ...service,
    run_key: runKey,
    dataset_repo: REPO,
    task: TASK,
    taxonomy_version: spec.taxonomy_version,
    taxonomy_hash: spec.taxonomy_hash,
    pipeline,
    expected_count: rows.length,
    manifest_sha256: await manifestDigest(await Promise.all(rows.map(async (row) => ({
      episode_index: row.episode_index,
      content_sha256: await predictionDigest(row),
    })))),
    source: "tests/stagePredictions.test.ts",
    provenance: { campaign: "test-only", dataset_revision: "b".repeat(40) },
  };
}

async function begin(rows = [prediction(0)], key = "run-v1") {
  return t.mutation(api.stagePredictions.begin, await runArguments(rows, key));
}

async function append(runId: RunId, rows: Prediction[]) {
  return t.mutation(api.stagePredictions.appendBatch, { ...service, run_id: runId, rows });
}

async function publish(rows = [prediction(0)], key = "run-v1") {
  const runId = await begin(rows, key);
  await append(runId, rows);
  await t.mutation(api.stagePredictions.publish, { ...service, run_id: runId });
  return runId;
}

async function page(runId: RunId, cursor: string | null = null, numItems = 50) {
  return t.query(api.stagePredictions.forRun, {
    run_id: runId,
    paginationOpts: { cursor, numItems },
  });
}

async function list() {
  return t.query(api.stagePredictions.listForRepo, {
    dataset_repo: REPO,
    taxonomy_version: spec.taxonomy_version,
  });
}

async function storeSnapshot() {
  return t.run(async (ctx) => ({
    specs: await ctx.db.query("stageTaskSpecs").collect(),
    reviews: await ctx.db.query("stageReviews").collect(),
    outcomes: await ctx.db.query("outcomeReviews").collect(),
    applyJobs: await ctx.db.query("applyJobs").collect(),
  }));
}

async function reviewArgs(runId: RunId) {
  const predictionRow = (await page(runId)).page[0];
  return {
    ...service,
    task: TASK,
    dataset_repo: REPO,
    episode_index: predictionRow.episode_index,
    taxonomy_version: spec.taxonomy_version,
    status: "confirmed",
    label: cleanLabel,
    prediction_id: predictionRow._id,
    prediction_sha256: predictionRow.content_sha256,
    reviewer_override: "test-reviewer",
  };
}

async function editor() {
  const userId = await t.run(async (ctx) => {
    const id = await ctx.db.insert("users", { username: "test-human" });
    await ctx.db.insert("authAccounts", {
      userId: id, provider: "huggingface", providerAccountId: EDITOR_SUB,
    });
    return id;
  });
  return { userId, client: t.withIdentity({ subject: userId }) };
}

beforeEach(async () => {
  oldServiceToken = process.env.ARENA_SERVICE_TOKEN;
  oldEditorSubs = process.env.ARENA_EDITOR_SUBS;
  process.env.ARENA_SERVICE_TOKEN = SERVICE_TOKEN;
  process.env.ARENA_EDITOR_SUBS = EDITOR_SUB;
  t = convexTest({ schema, modules, transactionLimits: true });
  await t.run(async (ctx) => {
    await ctx.db.insert("stageTaskSpecs", {
      task: TASK,
      taxonomy_version: spec.taxonomy_version,
      taxonomy_hash: spec.taxonomy_hash,
      live: true,
      spec,
      exported_at: 1,
      source: "test-fixture",
    });
    await ctx.db.insert("datasets", {
      repo_id: REPO,
      name: "Test routing dataset",
      task: TASK,
      source_type: "eval",
      environment: TASK,
      num_episodes: 100n,
    });
  });
});

afterEach(() => {
  if (oldServiceToken === undefined) delete process.env.ARENA_SERVICE_TOKEN;
  else process.env.ARENA_SERVICE_TOKEN = oldServiceToken;
  if (oldEditorSubs === undefined) delete process.env.ARENA_EDITOR_SUBS;
  else process.env.ARENA_EDITOR_SUBS = oldEditorSubs;
});

describe("immutable prediction ingestion", () => {
  test("raw canonical output and evidence round-trip without dropping fields or changing values", async () => {
    const row = {
      ...prediction(0),
      canonical_response: { nested: [{ text: "é机器人", score: 0.125, absent: null }], flag: false },
      evidence: { attempts: [1, 2, 3], model_prompt: "test prompt", artifact: "s3://test/input" },
    };
    const runId = await publish([row]);
    const stored = (await page(runId)).page[0];
    expect(stored).toMatchObject(row);
    expect(stored.content_sha256).toBe(await predictionDigest(row));
  });

  test("unknown flat-label fields are rejected because the review form cannot edit them", async () => {
    const row = { ...prediction(0), label: { ...cleanLabel, foreign_pipeline_field: "unreviewable" } };
    const runId = await begin([row]);
    await expect(append(runId, [row])).rejects.toThrow();
    expect((await page(runId)).page).toHaveLength(0);
  });

  test("invalid known fields remain immutable and flagged, and cannot be confirmed without correction", async () => {
    const invalid = prediction(0, 99);
    const runId = await publish([invalid]);
    const stored = (await page(runId)).page[0];
    expect(stored.label).toEqual(invalid.label);
    expect(stored.content_sha256).toBe(await predictionDigest(invalid));
    expect(stored.validation_codes).toContain("stage_out_of_range");
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: runId, expected_active_run_id: null,
    });
    expect((await t.query(api.stageCoverage.forTask, { task: TASK }))?.repos[0].n_flagged).toBe(1);
    const args = await reviewArgs(runId);
    await expect(t.mutation(api.stageReviews.save, { ...args, label: invalid.label }))
      .rejects.toThrow(/stage_out_of_range/);
    await t.mutation(api.stageReviews.save, { ...args, label: cleanLabel });
    expect((await page(runId)).page[0].label).toEqual(invalid.label);
  });

  test("staging is absent from ordinary selection; publication does not activate", async () => {
    const rows = [prediction(0), prediction(1)];
    const runId = await begin(rows);
    await append(runId, rows.slice(0, 1));
    expect((await list()).runs).toEqual([]);
    expect((await list()).active_run_id).toBeNull();
    expect((await t.query(api.stagePredictions.getRun, { run_id: runId }))?.status).toBe("uploading");
    await expect(t.mutation(api.stagePredictions.publish, { ...service, run_id: runId })).rejects.toThrow();
    expect((await list()).runs).toEqual([]);
    await append(runId, rows.slice(1));
    await t.mutation(api.stagePredictions.publish, { ...service, run_id: runId });
    expect((await list()).runs.map((run) => run._id)).toEqual([runId]);
    expect((await list()).active_run_id).toBeNull();
  });

  test("identical begin, append, and publication retries preserve stored bytes", async () => {
    const rows = [prediction(0), prediction(1)];
    const args = await runArguments(rows);
    const runId = await t.mutation(api.stagePredictions.begin, args);
    expect(await t.mutation(api.stagePredictions.begin, args)).toBe(runId);
    expect(await append(runId, rows)).toMatchObject({ inserted: 2, unchanged: 0 });
    const before = await page(runId);
    expect(await append(runId, [...rows].reverse())).toMatchObject({ inserted: 0, unchanged: 2 });
    expect(await page(runId)).toEqual(before);
    const storedBefore = await t.run(async (ctx) => ctx.db.query("stagePredictions").collect());
    await t.mutation(api.stagePredictions.publish, { ...service, run_id: runId });
    const published = await t.query(api.stagePredictions.getRun, { run_id: runId });
    const publishedPage = await page(runId);
    await t.mutation(api.stagePredictions.publish, { ...service, run_id: runId });
    expect(await t.query(api.stagePredictions.getRun, { run_id: runId })).toEqual(published);
    expect(await append(runId, rows)).toMatchObject({ inserted: 0, unchanged: 2 });
    expect(await page(runId)).toEqual(publishedPage);
    expect(await t.run(async (ctx) => ctx.db.query("stagePredictions").collect())).toEqual(storedBefore);
  });

  test("same run key cannot change provenance, manifest, pipeline, or schema", async () => {
    const args = await runArguments([prediction(0)]);
    const runId = await t.mutation(api.stagePredictions.begin, args);
    const before = await t.query(api.stagePredictions.getRun, { run_id: runId });
    for (const changed of [
      { provenance: { campaign: "different" } },
      { source: "different source" },
      { manifest_sha256: "f".repeat(64) },
      { pipeline: { ...pipeline, version: "v2" } },
      { taxonomy_hash: "f".repeat(64) },
      { expected_count: 2 },
    ]) {
      await expect(t.mutation(api.stagePredictions.begin, { ...args, ...changed })).rejects.toThrow();
    }
    expect(await t.query(api.stagePredictions.getRun, { run_id: runId })).toEqual(before);
  });

  test("a conflicting later row rolls back earlier inserts in the same batch", async () => {
    const runId = await begin([prediction(0), prediction(1)]);
    await append(runId, [prediction(0)]);
    const before = await page(runId);
    await expect(append(runId, [prediction(1), prediction(0, 3)])).rejects.toThrow();
    expect(await page(runId)).toEqual(before);
  });

  test("a duplicate episode with conflicting content aborts the batch", async () => {
    const runId = await begin([prediction(0)]);
    await expect(append(runId, [prediction(0), prediction(0, 2)])).rejects.toThrow();
    expect((await page(runId)).page).toHaveLength(0);
  });

  test("published rows reject both a changed prediction and an extra episode", async () => {
    const runId = await publish();
    const before = await page(runId);
    await expect(append(runId, [prediction(0, 3)])).rejects.toThrow();
    await expect(append(runId, [prediction(1)])).rejects.toThrow();
    expect(await page(runId)).toEqual(before);
  });

  test("equal count with the wrong episode identities cannot publish", async () => {
    const runId = await begin([prediction(0), prediction(1)]);
    await append(runId, [prediction(0), prediction(2)]);
    await expect(t.mutation(api.stagePredictions.publish, { ...service, run_id: runId })).rejects.toThrow();
    expect((await list()).runs).toEqual([]);
  });

  test("equal episode identities with changed raw evidence cannot publish", async () => {
    const rows = [prediction(0)];
    const runId = await begin(rows);
    await append(runId, [{ ...rows[0], evidence: { replaced: true } }]);
    await expect(t.mutation(api.stagePredictions.publish, { ...service, run_id: runId })).rejects.toThrow();
    expect((await list()).runs).toEqual([]);
  });

  test("negative episodes, nonfinite duration, empty batch, and oversized rows fail without writes", async () => {
    const runId = await begin([prediction(0)]);
    for (const rows of [
      [],
      [prediction(-1)],
      [{ ...prediction(0), episode_duration_s: NaN }],
      [{ ...prediction(0), episode_duration_s: -1 }],
      [{ ...prediction(0), evidence: { huge: "x".repeat(128 * 1024) } }],
    ]) {
      await expect(append(runId, rows)).rejects.toThrow();
      expect((await page(runId)).page).toHaveLength(0);
    }
  });

  test("batch and declared run-size limits reject oversized requests", async () => {
    const runId = await begin([prediction(0)]);
    await expect(append(runId, Array.from({ length: 51 }, (_, i) => prediction(i)))).rejects.toThrow();
    const args = await runArguments([prediction(0)], "too-large");
    await expect(t.mutation(api.stagePredictions.begin, { ...args, expected_count: 10001 })).rejects.toThrow();
    await expect(t.mutation(api.stagePredictions.begin, { ...args, expected_count: 0 })).rejects.toThrow();
    await expect(t.mutation(api.stagePredictions.begin, { ...args, expected_count: 1.5 })).rejects.toThrow();
    expect((await page(runId)).page).toHaveLength(0);
  });

  test("a run cannot claim a task conflicting with the registered dataset", async () => {
    const args = await runArguments([prediction(0)]);
    await t.run(async (ctx) => {
      await ctx.db.insert("stageTaskSpecs", {
        task: "other_task", taxonomy_version: spec.taxonomy_version,
        taxonomy_hash: spec.taxonomy_hash, live: true,
        spec: { ...spec, task: "other_task" }, exported_at: 1, source: "fixture",
      });
    });
    await expect(t.mutation(api.stagePredictions.begin, { ...args, task: "other_task" })).rejects.toThrow();
    expect((await list()).runs).toEqual([]);
  });

  test("source revisions must be complete pinned hashes", async () => {
    const runId = await begin();
    for (const revision of ["main", "a".repeat(41), "a".repeat(63)]) {
      await expect(append(runId, [{ ...prediction(0), source_revision: revision }])).rejects.toThrow();
    }
    expect((await page(runId)).page).toHaveLength(0);
  });

  test("pagination returns each prediction once, including nonconsecutive episodes", async () => {
    const runId = await publish([prediction(2), prediction(5), prediction(11)]);
    const first = await page(runId, null, 2);
    expect(first.isDone).toBe(false);
    const second = await page(runId, first.continueCursor, 2);
    expect(second.isDone).toBe(true);
    expect([...first.page, ...second.page].map((row) => row.episode_index).sort((a, b) => Number(a - b)))
      .toEqual([2n, 5n, 11n]);
  });
});

describe("selection integrity", () => {
  test("coverage counts one selected version and hides uploading runs", async () => {
    const baseline = await t.query(api.stageCoverage.forTask, { task: TASK });
    expect(baseline?.repos).toEqual([]);
    const firstId = await publish([prediction(0, 4), prediction(1, 4)], "first");
    const secondId = await publish([prediction(0, 8), prediction(1, 8)], "second");
    const pendingId = await begin([prediction(2, 3)], "uploading");
    await append(pendingId, [prediction(2, 3)]);
    expect((await t.query(api.stageCoverage.forTask, { task: TASK }))?.repos).toEqual(baseline?.repos);
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: firstId, expected_active_run_id: null,
    });
    const first = await t.query(api.stageCoverage.forTask, { task: TASK });
    expect(first?.repos[0].n_prefill).toBe(2);
    expect(first?.stage_hist_current[4]).toBe(2);
    expect(first?.stage_hist_current.reduce((sum, n) => sum + n, 0)).toBe(2);
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: secondId, expected_active_run_id: firstId,
    });
    const second = await t.query(api.stageCoverage.forTask, { task: TASK });
    expect(second?.repos[0].n_prefill).toBe(2);
    expect(second?.stage_hist_current[8]).toBe(2);
    expect(second?.stage_hist_current[4]).toBe(0);
    expect((await list()).runs).toHaveLength(2);
  });

  test("selection appends history; retries do not duplicate events", async () => {
    const firstId = await publish([prediction(0)], "first");
    const secondId = await publish([prediction(0, 4)], "second");
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: firstId, expected_active_run_id: null,
    });
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: firstId, expected_active_run_id: null,
    });
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: secondId, expected_active_run_id: firstId,
    });
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: secondId, expected_active_run_id: firstId,
    });
    expect((await list()).active_run_id).toBe(secondId);
    const history = await t.run(async (ctx) => ctx.db.query("stagePredictionSelectionHistory").collect());
    expect(history.map((row) => [row.generation, row.previous_run_id, row.run_id])).toEqual([
      [1, null, firstId], [2, firstId, secondId],
    ]);
    expect((await list()).runs).toHaveLength(2);
  });

  test("episode history includes both published versions, excluding partial uploads", async () => {
    const firstId = await publish([prediction(0)], "first");
    const secondId = await publish([prediction(0, 4)], "second");
    const pendingId = await begin([prediction(0)], "pending");
    await append(pendingId, [prediction(0)]);
    const history = await t.query(api.stagePredictions.historyForEpisode, {
      dataset_repo: REPO, taxonomy_version: spec.taxonomy_version, episode_index: 0n,
    });
    expect(history.predictions.map((row) => row.run_id).sort()).toEqual([firstId, secondId].sort());
  });

  test("activation requires a published run and the expected current selection", async () => {
    const pendingId = await begin([prediction(0)], "pending");
    await expect(t.mutation(api.stagePredictions.activate, {
      ...service, run_id: pendingId, expected_active_run_id: null,
    })).rejects.toThrow();
    const firstId = await publish([prediction(0)], "first");
    const secondId = await publish([prediction(0, 4)], "second");
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: firstId, expected_active_run_id: null,
    });
    await expect(t.mutation(api.stagePredictions.activate, {
      ...service, run_id: secondId, expected_active_run_id: null,
    })).rejects.toThrow();
    expect((await list()).active_run_id).toBe(firstId);
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: secondId, expected_active_run_id: firstId,
    });
    expect((await list()).active_run_id).toBe(secondId);
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: firstId, expected_active_run_id: secondId,
    });
    expect((await list()).active_run_id).toBe(firstId);
    expect((await list()).runs).toHaveLength(2);
  });

  test("upload and activation preserve every review, outcome, and apply job", async () => {
    await t.run(async (ctx) => {
      await ctx.db.insert("stageReviews", {
        task: TASK, dataset_repo: REPO, episode_index: 0n,
        taxonomy_version: spec.taxonomy_version, status: "confirmed",
        label: cleanLabel, reviewer: "earlier-reviewer", saved_at: 100,
        prefill_pushed_at: 77, episode_duration_s: 17,
      });
      await ctx.db.insert("outcomeReviews", {
        dataset_repo: REPO, episode_index: 0n, status: "confirmed",
        new_outcome: "success", reviewer: "earlier-reviewer", saved_at: 80,
      });
    });
    const before = await storeSnapshot();
    const runId = await publish();
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: runId, expected_active_run_id: null,
    });
    expect(await storeSnapshot()).toEqual(before);
  });

  test("schema bytes cannot change under an existing version, even with the same supplied hash", async () => {
    const before = await storeSnapshot();
    const args = {
      ...service, task: TASK, taxonomy_version: spec.taxonomy_version,
      taxonomy_hash: spec.taxonomy_hash, live: true, spec,
      source: "test re-export",
    };
    await expect(t.mutation(api.stageTaskSpecs.upsert, {
      ...args, spec: { ...spec, fps: spec.fps + 1 },
    })).rejects.toThrow();
    await expect(t.mutation(api.stageTaskSpecs.upsert, {
      ...args, taxonomy_hash: "f".repeat(64),
      spec: { ...spec, taxonomy_hash: "f".repeat(64) },
    })).rejects.toThrow();
    expect(await storeSnapshot()).toEqual(before);
    await expect(t.mutation(api.stageTaskSpecs.upsert, args)).resolves.toBeDefined();
  });
});

describe("review provenance", () => {
  test("copying a human label preserves the referenced review and its exact prediction origin", async () => {
    const runId = await publish();
    const originalId = await t.mutation(api.stageReviews.save, await reviewArgs(runId));
    const original = (await t.run(async (ctx) => ctx.db.get(originalId)))!;
    const copyArgs = {
      ...(await reviewArgs(runId)),
      reviewer_override: "adjudicator",
      copied_from_review_id: originalId,
      prefill_pushed_at: original.prefill_pushed_at,
      episode_duration_s: 999,
    };
    const copiedId = await t.mutation(api.stageReviews.save, copyArgs);
    const copied = (await t.run(async (ctx) => ctx.db.get(copiedId)))!;
    expect(copied.copied_from_review_id).toBe(originalId);
    expect(copied.prediction_id).toBe(original.prediction_id);
    expect(copied.prediction_sha256).toBe(original.prediction_sha256);
    expect(copied.prediction_run_id).toBe(original.prediction_run_id);
    expect(copied.prefill_pushed_at).toBe(original.prefill_pushed_at);
    expect(copied.episode_duration_s).toBe(original.episode_duration_s);
    expect(await t.run(async (ctx) => ctx.db.get(originalId))).toEqual(original);
    const differentRun = await publish([prediction(0, 4)], "different");
    const different = await reviewArgs(differentRun);
    for (const changes of [
      { prediction_id: different.prediction_id, prediction_sha256: different.prediction_sha256 },
      { prefill_pushed_at: original.prefill_pushed_at! + 1 },
      { prediction_sha256: "f".repeat(64) },
    ]) {
      await expect(t.mutation(api.stageReviews.save, { ...copyArgs, ...changes }))
        .rejects.toThrow(/provenance mismatch/);
    }
    expect((await storeSnapshot()).reviews).toHaveLength(2);
  });

  test("copied human reviews must match episode, task, dataset, and taxonomy and cannot be mutable drafts", async () => {
    const runId = await publish();
    const args = await reviewArgs(runId);
    for (const change of [
      { episode_index: 1n },
      { task: "foreign_task" },
      { dataset_repo: "test/foreign" },
      { taxonomy_version: "foreign_schema" },
      { status: "draft" },
      { status: "cleared", label: undefined },
    ]) {
      const copiedId = await t.run(async (ctx) => ctx.db.insert("stageReviews", {
        task: TASK, dataset_repo: REPO, episode_index: 0n,
        taxonomy_version: spec.taxonomy_version, status: "confirmed", label: cleanLabel,
        reviewer: "original-reviewer", saved_at: 10,
        ...change,
      }));
      await expect(t.mutation(api.stageReviews.save, { ...args, copied_from_review_id: copiedId }))
        .rejects.toThrow(/identity mismatch/);
    }
    // The only rows are the synthetic originals; none of the copies committed.
    expect((await storeSnapshot()).reviews).toHaveLength(6);
  });

  test("copying a source-free human review preserves its stored duration too", async () => {
    const human = await editor();
    const originalId = await t.run(async (ctx) => ctx.db.insert("stageReviews", {
      task: TASK, dataset_repo: REPO, episode_index: 0n,
      taxonomy_version: spec.taxonomy_version, status: "confirmed", label: cleanLabel,
      reviewer: "other-reviewer", saved_at: 10, episode_duration_s: 12,
    }));
    const id = await human.client.mutation(api.stageReviews.save, {
      task: TASK, dataset_repo: REPO, episode_index: 0n,
      taxonomy_version: spec.taxonomy_version, status: "confirmed", label: cleanLabel,
      copied_from_review_id: originalId, episode_duration_s: 999,
    });
    const saved = (await t.run(async (ctx) => ctx.db.get(id)))!;
    expect(saved.episode_duration_s).toBe(12);
    expect(saved.prediction_id).toBeUndefined();
    expect(saved.prefill_pushed_at).toBeUndefined();
  });

  test("a prediction timestamp without its prediction is rejected", async () => {
    const human = await editor();
    await expect(human.client.mutation(api.stageReviews.save, {
      task: TASK, dataset_repo: REPO, episode_index: 0n,
      taxonomy_version: spec.taxonomy_version, status: "confirmed", label: cleanLabel,
      prefill_pushed_at: 77, episode_duration_s: 999,
    })).rejects.toThrow(/requires prediction_id/);
    expect((await storeSnapshot()).reviews).toHaveLength(0);
  });

  test("saving against an earlier version pins its ID, hash, and duration after another version activates", async () => {
    const first = await publish([{ ...prediction(0), episode_duration_s: 17 }], "first");
    const args = await reviewArgs(first);
    const second = await publish([{ ...prediction(0), episode_duration_s: 35 }], "second");
    await t.mutation(api.stagePredictions.activate, {
      ...service, run_id: second, expected_active_run_id: null,
    });
    const id = await t.mutation(api.stageReviews.save, { ...args, episode_duration_s: 999 });
    const saved = await t.run(async (ctx) => ctx.db.get(id));
    expect(saved).toMatchObject({
      prediction_id: args.prediction_id,
      prediction_sha256: args.prediction_sha256,
      episode_duration_s: 17,
    });
  });

  test("wrong hash, wrong episode and wrong repo references fail", async () => {
    const runId = await publish();
    const args = await reviewArgs(runId);
    for (const changed of [
      { prediction_sha256: "f".repeat(64) },
      { episode_index: 1n },
      { dataset_repo: "test/foreign-repo" },
    ]) {
      await expect(t.mutation(api.stageReviews.save, { ...args, ...changed })).rejects.toThrow();
    }
    expect((await storeSnapshot()).reviews).toHaveLength(0);
  });

  test("unpublished predictions cannot be referenced by a human review", async () => {
    const runId = await begin();
    await append(runId, [prediction(0)]);
    const args = await reviewArgs(runId);
    await expect(t.mutation(api.stageReviews.save, args)).rejects.toThrow();
    expect((await storeSnapshot()).reviews).toHaveLength(0);
  });

  test("human-label folding remains one current judgment across prediction versions", async () => {
    const first = await publish([prediction(0)], "first");
    const second = await publish([prediction(0, 4)], "second");
    await t.mutation(api.stageReviews.save, { ...(await reviewArgs(first)), saved_at_override: 100 });
    await t.mutation(api.stageReviews.save, { ...(await reviewArgs(second)), saved_at_override: 200 });
    const latest = await t.query(api.stageReviews.latestForRepo, {
      dataset_repo: REPO, taxonomy_version: spec.taxonomy_version,
    });
    const history = await t.query(api.stageReviews.historyForEpisode, {
      dataset_repo: REPO, episode_index: 0n,
    });
    expect(latest.episodes).toHaveLength(1);
    expect(history).toHaveLength(2);
    expect(history[0].prediction_id).not.toBe(history[1].prediction_id);
    expect(latest.episodes[0].prediction_id).toBe(history[1].prediction_id);
  });
});

describe("authorization", () => {
  test("anonymous and invalid-service callers cannot begin a run", async () => {
    const { serviceToken, ...args } = await runArguments([prediction(0)]);
    void serviceToken;
    await expect(t.mutation(api.stagePredictions.begin, args)).rejects.toThrow();
    await expect(t.mutation(api.stagePredictions.begin, {
      ...args, serviceToken: "wrong-token",
    })).rejects.toThrow();
    expect((await list()).runs).toEqual([]);
  });

  test("an allowlisted human cannot write machine predictions", async () => {
    const human = await editor();
    const { serviceToken, ...args } = await runArguments([prediction(0)]);
    void serviceToken;
    await expect(human.client.mutation(api.stagePredictions.begin, args)).rejects.toThrow();
    expect((await list()).runs).toEqual([]);
  });

  test("anonymous append, publication, and activation leave the run untouched", async () => {
    const runId = await begin();
    const before = await t.query(api.stagePredictions.getRun, { run_id: runId });
    await expect(t.mutation(api.stagePredictions.appendBatch, {
      run_id: runId, rows: [prediction(0)],
    })).rejects.toThrow();
    await expect(t.mutation(api.stagePredictions.publish, { run_id: runId })).rejects.toThrow();
    await expect(t.mutation(api.stagePredictions.activate, {
      run_id: runId, expected_active_run_id: null,
    })).rejects.toThrow();
    expect(await t.query(api.stagePredictions.getRun, { run_id: runId })).toEqual(before);
    expect((await page(runId)).page).toHaveLength(0);
  });
});
