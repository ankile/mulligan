import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { convexTest } from "convex-test";
import { api } from "../convex/_generated/api";
import schema from "../convex/schema";
import { EXCLUDED_REVIEW_FIELDS, stageReviewCoverage, STRUCTURED_REVIEW_FIELDS } from "../convex/stageReviewCoverage";
import { manifestDigest, predictionDigest } from "../convex/stagePredictionContract";
import { trajectoryFromReview } from "../convex/trajectoryReview";
import { validateStageLabel, type ExportedStageSpec } from "../convex/stageConsistency";
import fixtures from "./fixtures/trajectory-review-fixtures.json";
import legacyFixtures from "./fixtures/stage-consistency-fixtures.json";

const modules = {
  "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
  "../convex/stageReviews.ts": () => import("../convex/stageReviews"),
  "../convex/stageTaskSpecs.ts": () => import("../convex/stageTaskSpecs"),
  "../convex/stagePredictions.ts": () => import("../convex/stagePredictions"),
};
const source = fixtures.synthetic.tasks.find((task) => task.spec.task === "routing_d2")!;
const fixture = source.cases.find((row) => row.name === "valid_success")!;
const repo = "test/structured-review";
const service = { serviceToken: "review-coverage-local-only" };
let t: ReturnType<typeof convexTest<typeof schema>>;
let oldToken: string | undefined;
beforeEach(async () => {
  oldToken = process.env.ARENA_SERVICE_TOKEN;
  process.env.ARENA_SERVICE_TOKEN = service.serviceToken;
  t = convexTest({ schema, modules, transactionLimits: true });
  await t.mutation(api.stageTaskSpecs.upsert, { ...service, task: source.spec.task,
    taxonomy_version: source.spec.taxonomy_version, taxonomy_hash: source.spec.taxonomy_hash,
    live: true, spec: source.spec, source: "test-only" });
});
afterEach(() => {
  if (oldToken === undefined) delete process.env.ARENA_SERVICE_TOKEN;
  else process.env.ARENA_SERVICE_TOKEN = oldToken;
});
function args() {
  const label = structuredClone(fixture.review_label!);
  label.trajectory_identity.sample_id = `${repo}#episode=0`;
  // The synthetic prediction fixture assigns timestamps independently; the
  // human review explicitly reconciles this known equivalent event.
  label.key_action_observations[0].first_time_s = 2;
  label.key_action_observations[0].occurrences[0].time_s = 2;
  return { ...service, reviewer_override: "test-reviewer", task: source.spec.task,
    dataset_repo: repo, taxonomy_version: source.spec.taxonomy_version, episode_index: 0n,
    status: "confirmed", label, episode_duration_s: fixture.duration_s };
}

describe("structured review coverage", () => {
  test("completed coverage is fixed, explicit, and never claims source prose or confidence", () => {
    for (const status of ["confirmed", "corrected"]) {
      const coverage = stageReviewCoverage("structured-v1", status, true)!;
      expect(coverage.reviewed_fields).toEqual([...STRUCTURED_REVIEW_FIELDS]);
      expect(coverage.excluded_fields).toEqual([...EXCLUDED_REVIEW_FIELDS]);
      expect(coverage.reviewed_fields.some((path) => /evidence|confidence|notes|review_reasons|needs_human_review/.test(path))).toBe(false);
      coverage.reviewed_fields.pop();
      expect(stageReviewCoverage("structured-v1", status, true)!.reviewed_fields).toHaveLength(STRUCTURED_REVIEW_FIELDS.length);
    }
    expect(stageReviewCoverage(undefined, "confirmed", true)).toBeUndefined();
  });

  test("legacy save followed by protocol review appends without rewriting old labels or inventing historical coverage", async () => {
    const originalArgs = args();
    const originalId = await t.mutation(api.stageReviews.save, originalArgs);
    const original = await t.run((ctx) => ctx.db.get(originalId));
    expect(original!.review_coverage).toBeUndefined();
    const humanNotes = "Human observation, separate from retained model notes.";
    const id = await t.mutation(api.stageReviews.save, { ...originalArgs,
      review_protocol: "structured-v1", notes: humanNotes });
    const saved = await t.run((ctx) => ctx.db.get(id));
    expect(id).not.toBe(originalId);
    expect(saved!.notes).toBe(humanNotes);
    expect(saved!.label).toEqual(originalArgs.label);
    expect(saved!.label!.notes).toBe(originalArgs.label.notes);
    expect(saved!.review_coverage).toEqual(stageReviewCoverage("structured-v1", "confirmed", true));
    expect(await t.run((ctx) => ctx.db.get(originalId))).toEqual(original);
    expect(await t.query(api.stageReviews.historyForEpisode, { dataset_repo: repo, episode_index: 0n })).toHaveLength(2);
  });

  test("draft and uncertain retain contradictions losslessly without completed coverage", async () => {
    for (const status of ["draft", "uncertain"]) {
      const input = args();
      input.label.key_action_observations[0].first_time_s = 2.5;
      input.label.key_action_observations[0].occurrences[0].time_s = 2.5;
      const id = await t.mutation(api.stageReviews.save, { ...input, status, review_protocol: "structured-v1" });
      const row = await t.run((ctx) => ctx.db.get(id));
      expect(row!.label).toEqual(input.label);
      expect(row!.review_coverage!.reviewed_fields).toEqual([]);
      expect(row!.review_coverage!.excluded_fields).toContain("notes");
    }
  });

  test("new confirmations and corrections reject cross-timeline conflicts even from old clients", async () => {
    const input = args();
    input.label.key_action_observations[0].first_time_s = 2.5;
    input.label.key_action_observations[0].occurrences[0].time_s = 2.5;
    expect(validateStageLabel(source.spec as ExportedStageSpec, input.label, input.episode_duration_s)).toEqual([]);
    for (const status of ["confirmed", "corrected"]) {
      await expect(t.mutation(api.stageReviews.save, { ...input, status })).rejects.toThrow("label timeline is inconsistent");
      await expect(t.mutation(api.stageReviews.save, { ...input, status, review_protocol: "structured-v1" })).rejects.toThrow("label timeline is inconsistent");
    }
    expect(await t.run((ctx) => ctx.db.query("stageReviews").collect())).toEqual([]);
  });

  test("protocol is rejected for clear while old clear clients continue to work", async () => {
    const input = { ...args(), label: undefined };
    await expect(t.mutation(api.stageReviews.save, { ...input, status: "cleared", review_protocol: "structured-v1" })).rejects.toThrow("cleared review");
    const id = await t.mutation(api.stageReviews.save, { ...input, status: "cleared" });
    expect((await t.run((ctx) => ctx.db.get(id)))!.review_coverage).toBeUndefined();
  });

  test("legacy schemas still save without protocol and reject invented structured coverage", async () => {
    const legacy = legacyFixtures["routing_d2@s10_v1"];
    await t.mutation(api.stageTaskSpecs.upsert, { ...service, task: "routing_d2",
      taxonomy_version: legacy.spec.taxonomy_version, taxonomy_hash: legacy.spec.taxonomy_hash,
      live: true, spec: legacy.spec, source: "test-only" });
    const input = { ...args(), taxonomy_version: legacy.spec.taxonomy_version,
      label: legacy.fixtures.find((entry) => entry.name === "clean_success")!.row };
    const id = await t.mutation(api.stageReviews.save, input);
    expect((await t.run((ctx) => ctx.db.get(id)))!.review_coverage).toBeUndefined();
    await expect(t.mutation(api.stageReviews.save, { ...input, review_protocol: "structured-v1" })).rejects.toThrow("requires a trajectory schema");
  });

  test("prediction import preserves a conflicting timeline and its original model metadata", async () => {
    const input = args();
    input.label.key_action_observations[0].first_time_s = 2.5;
    input.label.key_action_observations[0].occurrences[0].time_s = 2.5;
    await t.run((ctx) => ctx.db.insert("datasets", { repo_id: repo, name: "Test", task: source.spec.task,
      source_type: "eval", environment: source.spec.task, num_episodes: 1n }));
    const prediction = { episode_index: 0n, label: input.label,
      canonical_response: trajectoryFromReview(input.label), episode_duration_s: fixture.duration_s,
      evidence: {}, source_revision: "b".repeat(40) };
    const sha = await predictionDigest(prediction);
    const runId = await t.mutation(api.stagePredictions.begin, { ...service,
      task: source.spec.task, taxonomy_version: source.spec.taxonomy_version,
      taxonomy_hash: source.spec.taxonomy_hash, dataset_repo: repo, run_key: "test-preserved-conflict",
      pipeline: { name: "test", version: "v1", git_commit: "a".repeat(40) }, expected_count: 1,
      manifest_sha256: await manifestDigest([{ episode_index: 0n, content_sha256: sha }]), source: "test-only", provenance: {} });
    await t.mutation(api.stagePredictions.appendBatch, { ...service, run_id: runId, rows: [prediction] });
    await t.mutation(api.stagePredictions.publish, { ...service, run_id: runId });
    const stored = (await t.query(api.stagePredictions.forRun, { run_id: runId, paginationOpts: { numItems: 10, cursor: null } })).page[0];
    expect(stored.label).toEqual(input.label);
    expect(stored.content_sha256).toBe(sha);
    const originalPrediction = await t.run((ctx) => ctx.db.get(stored._id));
    await expect(t.mutation(api.stageReviews.save, { ...input, review_protocol: "structured-v1",
      prediction_id: stored._id, prediction_sha256: sha })).rejects.toThrow("label timeline is inconsistent");
    expect(await t.run((ctx) => ctx.db.get(stored._id))).toEqual(originalPrediction);
  });
});
