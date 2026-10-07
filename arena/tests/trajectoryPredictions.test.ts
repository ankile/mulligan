import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { convexTest } from "convex-test";
import { api } from "../convex/_generated/api";
import schema from "../convex/schema";
import { manifestDigest, predictionDigest } from "../convex/stagePredictionContract";
import fixtures from "./fixtures/trajectory-review-fixtures.json";
import { blankTrajectoryReview } from "../convex/trajectoryReview";
import { normalizeStageSpec } from "../src/lib/stage-spec";

const modules = {
  "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
  "../convex/stagePredictions.ts": () => import("../convex/stagePredictions"),
  "../convex/stageTaskSpecs.ts": () => import("../convex/stageTaskSpecs"),
  "../convex/stageReviews.ts": () => import("../convex/stageReviews"),
};
const service = { serviceToken: "trajectory-test-token" };
const repo = "test/generic-trajectory";
const source = fixtures.synthetic.tasks[0];
const fixture = source.cases.find((item) => item.name === "historical_high_stage_final_failure")!;
const pipeline = { name: "test", version: "v1", git_commit: "a".repeat(40) };
let t: ReturnType<typeof convexTest<typeof schema>>;
let oldToken: string | undefined;
beforeEach(async () => {
  oldToken = process.env.ARENA_SERVICE_TOKEN;
  process.env.ARENA_SERVICE_TOKEN = service.serviceToken;
  t = convexTest({ schema, modules, transactionLimits: true });
  await t.run(async (ctx) => { await ctx.db.insert("datasets", { repo_id: repo, name: "Test", task: source.spec.task,
    source_type: "eval", environment: source.spec.task, num_episodes: 100n }); });
});
afterEach(() => { if (oldToken === undefined) delete process.env.ARENA_SERVICE_TOKEN; else process.env.ARENA_SERVICE_TOKEN = oldToken; });
async function register(spec = source.spec, live = true) {
  return t.mutation(api.stageTaskSpecs.upsert, { ...service, task: spec.task, taxonomy_version: spec.taxonomy_version,
    taxonomy_hash: spec.taxonomy_hash, live, spec, source: "test-only" });
}
function prediction() {
  return { episode_index: 0n, label: structuredClone(fixture.review_label!), canonical_response: fixture.canonical,
    episode_duration_s: fixture.duration_s, evidence: {}, source_revision: "b".repeat(40) };
}
async function begin(spec = source.spec, row = prediction()) {
  return t.mutation(api.stagePredictions.begin, { ...service, task: spec.task, taxonomy_version: spec.taxonomy_version,
    taxonomy_hash: spec.taxonomy_hash, dataset_repo: repo, run_key: spec.taxonomy_version, pipeline, expected_count: 1,
    manifest_sha256: await manifestDigest([{ episode_index: row.episode_index, content_sha256: await predictionDigest(row) }]),
    source: "test-only", provenance: {} });
}
async function publish() {
  await register(); const row = prediction(); const run_id = await begin();
  await t.mutation(api.stagePredictions.appendBatch, { ...service, run_id, rows: [row] });
  await t.mutation(api.stagePredictions.publish, { ...service, run_id });
  const stored = (await t.query(api.stagePredictions.forRun, { run_id, paginationOpts: { numItems: 50, cursor: null } })).page[0];
  return { run_id, row, stored };
}
function review(stored: Awaited<ReturnType<typeof publish>>["stored"]) {
  return { ...service, reviewer_override: "test-reviewer", dataset_repo: repo, task: source.spec.task,
    taxonomy_version: source.spec.taxonomy_version, episode_index: 0n, status: "confirmed",
    label: structuredClone(fixture.review_label!), prediction_id: stored._id, prediction_sha256: stored.content_sha256 };
}

describe("generic immutable prediction and review storage", () => {
  test("registration verifies embedded definition and complete spec content hashes", async () => {
    await register();
    const definitionTamper = structuredClone(source.spec);
    definitionTamper.trajectory.task_definition.objective += " changed";
    await expect(register(definitionTamper)).rejects.toThrow("SHA-256 pin");
    const specTamper = structuredClone(source.spec); specTamper.ladder.header += " changed";
    await expect(register(specTamper)).rejects.toThrow("taxonomy_hash");
  });
  test("historical achievement with final failure saves without the legacy highest-stage success gate", async () => {
    const { stored, row } = await publish();
    expect(stored.label).toEqual(row.label);
    expect(stored.canonical_response).toEqual(row.canonical_response);
    expect(stored.validation_codes).toEqual([]);
    // The immutable fixture predates the review timeline gate: its synthetic
    // action times come after stage entry. Supply an explicitly corrected
    // human timeline while preserving the historical maximum/final failure.
    const human = review(stored);
    human.label.stage_transitions.forEach((event, i) => { event.time_s = 13 + i; });
    const id = await t.mutation(api.stageReviews.save, human);
    const saved = await t.run((ctx) => ctx.db.get(id));
    expect(saved!.label).toEqual(human.label);
    expect(saved!.episode_duration_s).toBe(row.episode_duration_s);
    const corrected = structuredClone(human); corrected.label.notes = "Human correction with full history retained";
    await t.mutation(api.stageReviews.save, { ...corrected, status: "corrected" });
    expect((await t.run((ctx) => ctx.db.get(id)))!.label).toEqual(human.label);
    expect((await t.run((ctx) => ctx.db.get(stored._id)))!.label).toEqual(row.label);
  });
  test("every review status pins identity and confirmed reviews enforce full event semantics", async () => {
    const { stored } = await publish();
    const wrongIdentity = review(stored); wrongIdentity.label.trajectory_identity.sample_id = "another-episode";
    await expect(t.mutation(api.stageReviews.save, { ...wrongIdentity, status: "draft" })).rejects.toThrow("trajectory identity");
    const lateEvent = review(stored); lateEvent.label.failure_events[0].time_s = 31;
    await expect(t.mutation(api.stageReviews.save, lateEvent)).rejects.toThrow("internally inconsistent");
    await t.mutation(api.stageReviews.save, { ...lateEvent, status: "draft" });
  });
  test("source-free drafts use exact dataset episode identity and preserve unset outcomes", async () => {
    await register();
    const label = blankTrajectoryReview(normalizeStageSpec(source.spec).trajectory!, repo, 12);
    const args = { ...service, reviewer_override: "test-reviewer", task: source.spec.task, dataset_repo: repo,
      taxonomy_version: source.spec.taxonomy_version, episode_index: 12n, status: "draft", label, episode_duration_s: 30 };
    const id = await t.mutation(api.stageReviews.save, args);
    expect((await t.run((ctx) => ctx.db.get(id)))!.label!.task_success).toBeNull();
    await expect(t.mutation(api.stageReviews.save, { ...args, status: "confirmed" })).rejects.toThrow("internally inconsistent");
  });
  test("contradictory canonical and editable representations fail before any row is inserted", async () => {
    await register(); const row = prediction(); row.label.notes = "contradiction";
    const run_id = await begin(source.spec, row);
    await expect(t.mutation(api.stagePredictions.appendBatch, { ...service, run_id, rows: [row] })).rejects.toThrow("losslessly match");
    expect(await t.run((ctx) => ctx.db.query("stagePredictions").collect())).toEqual([]);
  });
  test("other-schema links are published and episode-specific without switching active selection", async () => {
    await publish();
    const args = { dataset_repo: repo, task: source.spec.task, taxonomy_version: "another-schema", episode_index: 0n };
    const choices = await t.query(api.stagePredictions.otherSchemasForEpisode, args);
    expect(choices).toHaveLength(1); expect(choices[0].taxonomy_version).toBe(source.spec.taxonomy_version);
    expect(await t.query(api.stagePredictions.otherSchemasForEpisode, { ...args, episode_index: 1n })).toEqual([]);
    expect(await t.query(api.stagePredictions.otherSchemasForEpisode, { ...args, taxonomy_version: source.spec.taxonomy_version })).toEqual([]);
    expect(await t.run((ctx) => ctx.db.query("stagePredictionSelections").collect())).toEqual([]);
  });
});


test("immutable generic identity errors fail before insertion, while supported semantic errors are retained", async () => {
  await register();
  const bad = prediction(); bad.label.trajectory_identity.task_id = "other-task";
  bad.canonical_response = { ...bad.canonical_response, task_id: "other-task" };
  const badRun = await begin(source.spec, bad);
  await expect(t.mutation(api.stagePredictions.appendBatch, { ...service, run_id: badRun, rows: [bad] })).rejects.toThrow("trajectory identity");
  expect(await t.run((ctx) => ctx.db.query("stagePredictions").collect())).toEqual([]);
});

test("nested unknown fields block committed gold while draft preserves the original content", async () => {
  const { stored } = await publish();
  const args = review(stored);
  (args.label.key_action_observations[0] as Record<string, unknown>).extra_unreviewable_field = "unsupported";
  await expect(t.mutation(api.stageReviews.save, args)).rejects.toThrow("trajectory_shape");
  const id = await t.mutation(api.stageReviews.save, { ...args, status: "draft" });
  expect((await t.run((ctx) => ctx.db.get(id)))!.label).toEqual(args.label);
});

test("source-free confirmed reviews reject reset-tail events beyond the validated policy duration", async () => {
  await register();
  const label = structuredClone(source.cases.find((item) => item.name === "valid_failure")!.review_label!);
  label.trajectory_identity.sample_id = `${repo}#episode=12`;
  const args = { ...service, reviewer_override: "test-reviewer", task: source.spec.task, dataset_repo: repo,
    taxonomy_version: source.spec.taxonomy_version, episode_index: 12n, status: "confirmed", label, episode_duration_s: 8 };
  await t.mutation(api.stageReviews.save, args);
  label.primary_failure_time_s = 9;
  label.failure_events[0].time_s = 9;
  await expect(t.mutation(api.stageReviews.save, args)).rejects.toThrow("trajectory_time_bounds");
});
