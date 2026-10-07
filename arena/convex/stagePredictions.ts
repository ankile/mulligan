/** Immutable, dataset-scoped prediction runs. Publication and selection are separate. */
import { paginationOptsValidator } from "convex/server";
import { v } from "convex/values";
import { mutation, query, type MutationCtx } from "./_generated/server";
import type { Id } from "./_generated/dataModel";
import { requireEditorOrService } from "./access";
import { trajectoryFromReview, trajectoryShapeViolations, validTrajectoryIdentity } from "./trajectoryReview";
import { canonicalizeStageLabel, validateStageLabel, type ExportedStageSpec } from "./stageConsistency";
import {
  CONTENT_PROTOCOL, canonicalDigest, canonicalEncoding, hash, manifestDigest, pipelineValidator,
  predictionContent, predictionDigest, predictionValidator,
} from "./stagePredictionContract";

async function requirePipeline(ctx: MutationCtx, token: string | undefined) {
  if (await requireEditorOrService(ctx, token) !== "service") {
    throw new Error("Prediction publication requires an ingest machine credential");
  }
}

const beginFields = {
  run_key: v.string(), dataset_repo: v.string(), task: v.string(),
  taxonomy_version: v.string(), taxonomy_hash: v.string(),
  pipeline: pipelineValidator, expected_count: v.float64(),
  manifest_sha256: v.string(), source: v.string(), provenance: v.any(),
};

export const begin = mutation({
  args: { serviceToken: v.optional(v.string()), ...beginFields },
  handler: async (ctx, args) => {
    await requirePipeline(ctx, args.serviceToken);
    const identity = { ...args };
    delete identity.serviceToken;
    for (const key of ["run_key", "dataset_repo", "task", "taxonomy_version", "source"] as const) {
      if (!args[key].trim() || args[key].length > 512) throw new Error(`invalid ${key}`);
    }
    for (const value of Object.values(args.pipeline)) {
      if (!value.trim() || value.length > 512) throw new Error("invalid pipeline identity");
    }
    if (!Number.isInteger(args.expected_count) || args.expected_count < 1 || args.expected_count > 10000) {
      throw new Error("expected_count must be an integer in 1..10000");
    }
    hash(args.taxonomy_hash, "taxonomy_hash");
    hash(args.manifest_sha256, "manifest_sha256");
    if (new TextEncoder().encode(canonicalEncoding(identity)).length > 128 * 1024) throw new Error("run metadata exceeds 128 KiB");
    const spec = await ctx.db.query("stageTaskSpecs").withIndex("by_task_version", (q) =>
      q.eq("task", args.task).eq("taxonomy_version", args.taxonomy_version)
    ).unique();
    if (!spec || spec.taxonomy_hash !== args.taxonomy_hash) throw new Error("export the matching task spec and taxonomy hash first");
    const dataset = await ctx.db.query("datasets").withIndex("by_repo", (q) => q.eq("repo_id", args.dataset_repo)).unique();
    if (dataset && dataset.task !== args.task) throw new Error("prediction task does not match registered dataset");
    const specDigest = await canonicalDigest(spec.spec);
    const identityDigest = await canonicalDigest(identity);
    const existing = await ctx.db.query("stagePredictionRuns").withIndex("by_key", (q) => q.eq("run_key", args.run_key)).unique();
    if (existing) {
      if (existing.identity_sha256 !== identityDigest || existing.spec_content_sha256 !== specDigest) {
        throw new Error("immutable run identity conflict; use a new run_key");
      }
      return existing._id;
    }
    return ctx.db.insert("stagePredictionRuns", {
      ...identity, content_protocol: CONTENT_PROTOCOL, identity_sha256: identityDigest, spec_content_sha256: specDigest,
      status: "uploading", received_count: 0, created_at: Date.now(),
    });
  },
});

export const appendBatch = mutation({
  args: { serviceToken: v.optional(v.string()), run_id: v.id("stagePredictionRuns"), rows: v.array(predictionValidator) },
  handler: async (ctx, args) => {
    await requirePipeline(ctx, args.serviceToken);
    if (args.rows.length < 1 || args.rows.length > 50) throw new Error("appendBatch requires 1..50 rows");
    const run = await ctx.db.get(args.run_id);
    if (!run) throw new Error("prediction run not found");
    const spec = await ctx.db.query("stageTaskSpecs").withIndex("by_task_version", (q) =>
      q.eq("task", run.task).eq("taxonomy_version", run.taxonomy_version)
    ).unique();
    if (!spec || await canonicalDigest(spec.spec) !== run.spec_content_sha256) throw new Error("run schema content changed");
    const seen = new Set<string>();
    let inserted = 0, unchanged = 0, batchBytes = 0;
    for (const row of args.rows) {
      const episode = row.episode_index.toString();
      if (seen.has(episode)) throw new Error("duplicate episode in append batch");
      seen.add(episode);
      if (!Number.isFinite(row.episode_duration_s) || row.episode_duration_s <= 0) throw new Error("episode_duration_s must be finite and positive");
      if (row.source_revision !== undefined && !/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(row.source_revision)) throw new Error("source_revision must be a pinned content revision");
      const bytes = new TextEncoder().encode(canonicalEncoding(predictionContent(row))).length;
      batchBytes += bytes;
      if (bytes > 128 * 1024 || batchBytes > 1024 * 1024) throw new Error("prediction payload exceeds row/batch limit; reduce batch size");
      const digest = await predictionDigest(row);
      const existing = await ctx.db.query("stagePredictions").withIndex("by_run_episode", (q) =>
        q.eq("run_id", run._id).eq("episode_index", row.episode_index)
      ).unique();
      if (existing) {
        if (existing.content_sha256 !== digest) throw new Error(`immutable prediction conflict at episode ${episode}`);
        unchanged++;
        continue;
      }
      if (run.status !== "uploading") throw new Error("published runs cannot accept new episodes");
      if (run.received_count + inserted >= run.expected_count) throw new Error("run exceeds expected episode count");
      const labelSpec = spec.spec as ExportedStageSpec;
      const normalized = canonicalizeStageLabel(labelSpec, row.label);
      if (normalized.unknownKeys.length) throw new Error(`label contains unknown editable fields: ${normalized.unknownKeys.join(", ")}; put full provider output in canonical_response`);
      if (labelSpec.trajectory && !validTrajectoryIdentity(labelSpec.trajectory, row.label.trajectory_identity)) {
        throw new Error("trajectory identity must match the registered source task, taxonomy, and schema with a nonempty sample ID");
      }
      if (labelSpec.trajectory && (row.canonical_response === undefined ||
          await canonicalDigest(trajectoryFromReview(row.label)) !== await canonicalDigest(row.canonical_response))) {
        throw new Error("trajectory review label must losslessly match canonical_response");
      }
      if (labelSpec.trajectory && trajectoryShapeViolations(labelSpec.trajectory, row.canonical_response).length) {
        throw new Error("trajectory canonical_response contains missing or unmapped structural fields");
      }
      const validationCodes = [...new Set(
        validateStageLabel(labelSpec, row.label, row.episode_duration_s).map((v) => v.code)
      )].sort();
      const predictionId = await ctx.db.insert("stagePredictions", {
        ...row, run_id: run._id, content_sha256: digest, validation_codes: validationCodes,
      });
      const stageValue = row.label[spec.spec.stage_field];
      const stage = typeof stageValue === "number" && Number.isInteger(stageValue) ? stageValue : undefined;
      await ctx.db.insert("stagePredictionMembers", {
        run_id: run._id, prediction_id: predictionId, episode_index: row.episode_index,
        content_sha256: digest, ...(stage === undefined ? {} : { stage }),
        flagged: Boolean(row.review_reason || row.violation_codes?.length || validationCodes.length),
      });
      inserted++;
    }
    if (inserted) await ctx.db.patch(run._id, { received_count: run.received_count + inserted });
    return { inserted, unchanged };
  },
});

export const publish = mutation({
  args: { serviceToken: v.optional(v.string()), run_id: v.id("stagePredictionRuns") },
  handler: async (ctx, args) => {
    await requirePipeline(ctx, args.serviceToken);
    const run = await ctx.db.get(args.run_id);
    if (!run) throw new Error("prediction run not found");
    if (run.status === "published") return run._id;
    const members = await ctx.db.query("stagePredictionMembers").withIndex("by_run_episode", (q) => q.eq("run_id", run._id)).collect();
    if (members.length !== run.expected_count || run.received_count !== run.expected_count) throw new Error("prediction run is incomplete");
    if (await manifestDigest(members) !== run.manifest_sha256) throw new Error("prediction manifest digest mismatch");
    const publishedAt = Date.now();
    await ctx.db.patch(run._id, { status: "published", published_at: publishedAt });
    await ctx.db.insert("stagePredictionCatalog", {
      run_id: run._id, run_key: run.run_key, dataset_repo: run.dataset_repo,
      task: run.task, taxonomy_version: run.taxonomy_version, pipeline: run.pipeline,
      expected_count: run.expected_count, published_at: publishedAt,
    });
    return run._id;
  },
});

async function select(
  ctx: MutationCtx, repo: string, task: string, taxonomy: string,
  runId: Id<"stagePredictionRuns">, expected: Id<"stagePredictionRuns"> | null,
) {
  const current = await ctx.db.query("stagePredictionSelections").withIndex("by_repo_taxonomy", (q) => q.eq("dataset_repo", repo).eq("taxonomy_version", taxonomy)).unique();
  const previous = current?.run_id ?? null;
  if (previous === runId) return runId;
  if (previous !== expected) throw new Error("active prediction changed; refresh before selecting a version");
  const generation = (current?.generation ?? 0) + 1;
  const selection = { dataset_repo: repo, task, taxonomy_version: taxonomy, run_id: runId, generation };
  if (current) await ctx.db.replace(current._id, selection);
  else await ctx.db.insert("stagePredictionSelections", selection);
  await ctx.db.insert("stagePredictionSelectionHistory", {
    dataset_repo: repo, taxonomy_version: taxonomy, previous_run_id: previous,
    run_id: runId, generation, selected_at: Date.now(),
  });
  return runId;
}

export const activate = mutation({
  args: { serviceToken: v.optional(v.string()), run_id: v.id("stagePredictionRuns"), expected_active_run_id: v.union(v.id("stagePredictionRuns"), v.null()) },
  handler: async (ctx, args) => {
    await requirePipeline(ctx, args.serviceToken);
    const run = await ctx.db.get(args.run_id);
    if (!run || run.status !== "published") throw new Error("only a published run can be selected");
    return select(ctx, run.dataset_repo, run.task, run.taxonomy_version, run._id, args.expected_active_run_id);
  },
});

export const getRun = query({
  args: { run_id: v.id("stagePredictionRuns") },
  handler: async (ctx, args) => ctx.db.get(args.run_id),
});

export const listForRepo = query({
  args: { dataset_repo: v.string(), taxonomy_version: v.string() },
  handler: async (ctx, args) => {
    const runs = await ctx.db.query("stagePredictionCatalog").withIndex("by_repo_taxonomy", (q) => q.eq("dataset_repo", args.dataset_repo).eq("taxonomy_version", args.taxonomy_version)).collect();
    const selection = await ctx.db.query("stagePredictionSelections").withIndex("by_repo_taxonomy", (q) => q.eq("dataset_repo", args.dataset_repo).eq("taxonomy_version", args.taxonomy_version)).unique();
    return {
      runs: runs.sort((a, b) => b.published_at - a.published_at).map((r) => ({ ...r, _id: r.run_id })),
      active_run_id: selection?.run_id ?? null,
    };
  },
});

/** Explicit run reads include uploading runs for importer readback, never defaults. */
export const forRun = query({
  args: { run_id: v.id("stagePredictionRuns"), paginationOpts: paginationOptsValidator },
  handler: async (ctx, args) => {
    if (args.paginationOpts.numItems > 50) throw new Error("prediction pages are limited to 50 rows");
    const run = await ctx.db.get(args.run_id);
    if (!run) throw new Error("prediction run not found");
    const result = await ctx.db.query("stagePredictions").withIndex("by_run_episode", (q) => q.eq("run_id", run._id)).paginate(args.paginationOpts);
    return { ...result, page: result.page.map((p) => ({ ...p,
      task: run.task, dataset_repo: run.dataset_repo, taxonomy_version: run.taxonomy_version,
      pipeline: run.pipeline, source: run.source, pushed_at: run.published_at ?? run.created_at,
    })) };
  },
});

export const historyForEpisode = query({
  args: { dataset_repo: v.string(), taxonomy_version: v.string(), episode_index: v.int64() },
  handler: async (ctx, args) => {
    const runs = await ctx.db.query("stagePredictionCatalog").withIndex("by_repo_taxonomy", (q) => q.eq("dataset_repo", args.dataset_repo).eq("taxonomy_version", args.taxonomy_version)).collect();
    const predictions = [];
    for (const run of runs) {
      const member = await ctx.db.query("stagePredictionMembers").withIndex("by_run_episode", (q) => q.eq("run_id", run.run_id).eq("episode_index", args.episode_index)).unique();
      if (member) predictions.push({ ...member, run_key: run.run_key, pipeline: run.pipeline, published_at: run.published_at });
    }
    return { predictions };
  },
});

/** Read one immutable prediction after resolving its identity through history. */
export const getPrediction = query({
  args: { prediction_id: v.id("stagePredictions") },
  handler: async (ctx, args) => ctx.db.get(args.prediction_id),
});

export const selectionHistory = query({
  args: { dataset_repo: v.string(), taxonomy_version: v.string(), paginationOpts: paginationOptsValidator },
  handler: async (ctx, args) => ctx.db.query("stagePredictionSelectionHistory")
    .withIndex("by_repo_taxonomy", (q) => q.eq("dataset_repo", args.dataset_repo).eq("taxonomy_version", args.taxonomy_version))
    .order("desc").paginate(args.paginationOpts),
});

/** Other schemas remain separate; expose only published per-episode coverage. */
export const otherSchemasForEpisode = query({
  args: { dataset_repo: v.string(), task: v.string(), taxonomy_version: v.string(), episode_index: v.int64() },
  handler: async (ctx, args) => {
    const catalog = await ctx.db.query("stagePredictionCatalog").withIndex("by_repo_taxonomy", (q) =>
      q.eq("dataset_repo", args.dataset_repo)).collect();
    const available = new Map<string, { taxonomy_version: string; run_id: Id<"stagePredictionRuns">; expected_count: number; published_at: number }>();
    for (const run of catalog.sort((a, b) => b.published_at - a.published_at)) {
      if (run.task !== args.task || run.taxonomy_version === args.taxonomy_version || available.has(run.taxonomy_version)) continue;
      const member = await ctx.db.query("stagePredictionMembers").withIndex("by_run_episode", (q) =>
        q.eq("run_id", run.run_id).eq("episode_index", args.episode_index)).unique();
      if (member) available.set(run.taxonomy_version, { taxonomy_version: run.taxonomy_version,
        run_id: run.run_id, expected_count: run.expected_count, published_at: run.published_at });
    }
    return [...available.values()];
  },
});
