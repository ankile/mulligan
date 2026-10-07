import { query, mutation, internalMutation, internalQuery } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import { requireEditorOrService } from "./access";
import { loadTaskStatusMap } from "./statuses";
import { effectiveStatus, statusOrInheritValidator } from "./statusShared";

export const register = mutation({
  args: {
    serviceToken: v.optional(v.string()),
    repo_id: v.string(),
    name: v.string(),
    task: v.string(),
    source_type: v.string(),
    dataset_role: v.optional(v.string()),
    trainable: v.optional(v.boolean()),
    environment: v.string(),
    num_episodes: v.optional(v.int64()),
    model_id: v.optional(v.string()),
    model_url: v.optional(v.string()),
    parent_repo_id: v.optional(v.string()),
    derived_repo_ids: v.optional(v.array(v.string())),
    mutually_exclusive_with: v.optional(v.array(v.string())),
    view_family_id: v.optional(v.string()),
    view_id: v.optional(v.string()),
    producer_model_ids: v.optional(v.array(v.string())),
    target_model_id: v.optional(v.string()),
    target_arm_key: v.optional(v.string()),
    notes: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { serviceToken, ...fields } = args;
    await requireEditorOrService(ctx, serviceToken);
    const requestedAt = Date.now();
    const existing = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", fields.repo_id))
      .unique();
    const refreshFields = {
      stats_status: "pending" as const,
      stats_error: undefined,
      stats_refresh_requested_at: requestedAt,
    };
    let datasetId;
    if (existing) {
      await ctx.db.patch(existing._id, { ...fields, ...refreshFields });
      datasetId = existing._id;
    } else {
      datasetId = await ctx.db.insert("datasets", { ...fields, ...refreshFields });
    }
    await ctx.scheduler.runAfter(0, internal.datasetStats.refresh, {
      repo_id: fields.repo_id,
      requested_at: requestedAt,
    });
    return datasetId;
  },
});

export const refreshStats = mutation({
  args: {
    repo_id: v.string(),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) throw new Error(`Dataset not found: ${args.repo_id}`);
    const requestedAt = Date.now();
    await ctx.db.patch(dataset._id, {
      stats_status: "pending",
      stats_error: undefined,
      stats_refresh_requested_at: requestedAt,
    });
    await ctx.scheduler.runAfter(0, internal.datasetStats.refresh, {
      repo_id: args.repo_id,
      requested_at: requestedAt,
    });
    return requestedAt;
  },
});

export const enqueueStatsRefreshInternal = internalMutation({
  args: { repo_id: v.string() },
  handler: async (ctx, args) => {
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) return false;
    const requestedAt = Date.now();
    await ctx.db.patch(dataset._id, {
      stats_status: "pending",
      stats_error: undefined,
      stats_refresh_requested_at: requestedAt,
    });
    await ctx.scheduler.runAfter(0, internal.datasetStats.refresh, {
      repo_id: args.repo_id,
      requested_at: requestedAt,
    });
    return true;
  },
});

export const getStatsStateInternal = internalQuery({
  args: { repo_id: v.string() },
  handler: async (ctx, args) => {
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) return null;
    return {
      stats_hf_sha: dataset.stats_hf_sha,
      stats_algorithm_version: dataset.stats_algorithm_version,
    };
  },
});

export const finishCachedStatsRefreshInternal = internalMutation({
  args: {
    repo_id: v.string(),
    requested_at: v.float64(),
  },
  handler: async (ctx, args) => {
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset || dataset.stats_refresh_requested_at !== args.requested_at) return false;
    await ctx.db.patch(dataset._id, {
      stats_status: "ready",
      stats_error: undefined,
      stats_refresh_requested_at: undefined,
    });
    return true;
  },
});

export const finishStatsRefreshInternal = internalMutation({
  args: {
    repo_id: v.string(),
    requested_at: v.float64(),
    hf_sha: v.string(),
    algorithm_version: v.string(),
    num_episodes: v.number(),
    total_duration_seconds: v.number(),
    num_success: v.number(),
    num_failure: v.number(),
    num_human_frames: v.optional(v.number()),
    num_policy_frames: v.optional(v.number()),
    num_autonomous_success: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset || dataset.stats_refresh_requested_at !== args.requested_at) return false;
    await ctx.db.patch(dataset._id, {
      num_episodes: BigInt(args.num_episodes),
      total_duration_seconds: args.total_duration_seconds,
      num_success: BigInt(args.num_success),
      num_failure: BigInt(args.num_failure),
      num_human_frames:
        args.num_human_frames === undefined ? undefined : BigInt(args.num_human_frames),
      num_policy_frames:
        args.num_policy_frames === undefined ? undefined : BigInt(args.num_policy_frames),
      num_autonomous_success:
        args.num_autonomous_success === undefined
          ? undefined
          : BigInt(args.num_autonomous_success),
      stats_status: "ready",
      stats_hf_sha: args.hf_sha,
      stats_computed_at: Date.now(),
      stats_algorithm_version: args.algorithm_version,
      stats_error: undefined,
      stats_refresh_requested_at: undefined,
    });
    return true;
  },
});

export const failStatsRefreshInternal = internalMutation({
  args: {
    repo_id: v.string(),
    requested_at: v.float64(),
    error: v.string(),
  },
  handler: async (ctx, args) => {
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset || dataset.stats_refresh_requested_at !== args.requested_at) return false;
    await ctx.db.patch(dataset._id, {
      stats_status: "error",
      stats_error: args.error,
      stats_refresh_requested_at: undefined,
    });
    return true;
  },
});

export const updateStats = mutation({
  args: {
    serviceToken: v.optional(v.string()),
    repo_id: v.string(),
    num_episodes: v.number(),
    total_duration_seconds: v.number(),
    num_success: v.optional(v.number()),
    num_failure: v.optional(v.number()),
    num_human_frames: v.optional(v.number()),
    num_policy_frames: v.optional(v.number()),
    num_autonomous_success: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) return;
    const patch: Record<string, unknown> = {};

    const count = BigInt(args.num_episodes);
    if (dataset.num_episodes !== count) patch.num_episodes = count;
    if (dataset.total_duration_seconds !== args.total_duration_seconds)
      patch.total_duration_seconds = args.total_duration_seconds;

    for (const field of [
      "num_success",
      "num_failure",
      "num_human_frames",
      "num_policy_frames",
      "num_autonomous_success",
    ] as const) {
      const val = args[field];
      if (val != null) {
        const bigVal = BigInt(val);
        if (dataset[field] !== bigVal) patch[field] = bigVal;
      }
    }

    if (Object.keys(patch).length > 0) {
      await ctx.db.patch(dataset._id, patch);
    }
  },
});

export const deleteByRepo = mutation({
  args: { repo_id: v.string(), serviceToken: v.optional(v.string()) },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) return null;
    await ctx.db.delete(dataset._id);
    return dataset._id;
  },
});

export const getByRepo = query({
  args: { repo_id: v.string() },
  handler: async (ctx, args) => {
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) return null;
    const taskStatuses = await loadTaskStatusMap(ctx);
    return {
      ...dataset,
      effective_status: effectiveStatus(
        dataset.status,
        dataset.task,
        taskStatuses
      ),
    };
  },
});

export const updateTask = mutation({
  args: {
    repo_id: v.string(),
    task: v.string(),
    environment: v.string(),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) {
      throw new Error(`Dataset not found: ${args.repo_id}`);
    }
    await ctx.db.patch(dataset._id, {
      task: args.task,
      environment: args.environment,
    });
    return dataset._id;
  },
});

export const updateClassification = mutation({
  args: {
    repo_id: v.string(),
    dataset_role: v.string(),
    trainable: v.boolean(),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) {
      throw new Error(`Dataset not found: ${args.repo_id}`);
    }
    await ctx.db.patch(dataset._id, {
      dataset_role: args.dataset_role,
      trainable: args.trainable,
    });
    return dataset._id;
  },
});

export const list = query({
  args: {
    source_type: v.optional(v.string()),
    task: v.optional(v.string()),
    dataset_role: v.optional(v.string()),
    trainable: v.optional(v.boolean()),
  },
  handler: async (ctx, args) => {
    let datasets;
    if (args.dataset_role) {
      datasets = await ctx.db
        .query("datasets")
        .withIndex("by_dataset_role", (q) =>
          q.eq("dataset_role", args.dataset_role!)
        )
        .collect();
    } else if (args.source_type) {
      datasets = await ctx.db
        .query("datasets")
        .withIndex("by_source_type", (q) =>
          q.eq("source_type", args.source_type!)
        )
        .collect();
    } else {
      datasets = await ctx.db.query("datasets").collect();
    }
    if (args.task) {
      datasets = datasets.filter((d) => d.task === args.task);
    }
    if (args.source_type && args.dataset_role) {
      datasets = datasets.filter((d) => d.source_type === args.source_type);
    }
    if (args.trainable !== undefined) {
      datasets = datasets.filter((d) => d.trainable === args.trainable);
    }
    const taskStatuses = await loadTaskStatusMap(ctx);
    return datasets
      .map((d) => ({
        ...d,
        effective_status: effectiveStatus(d.status, d.task, taskStatuses),
      }))
      .sort((a, b) => b._creationTime - a._creationTime);
  },
});

export const setStatus = mutation({
  args: {
    repo_id: v.string(),
    status: statusOrInheritValidator,
    status_reason: v.optional(v.string()),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.repo_id))
      .unique();
    if (!dataset) {
      throw new Error(`Dataset not found: ${args.repo_id}`);
    }
    await ctx.db.patch(dataset._id, {
      status: args.status === "inherit" ? undefined : args.status,
      status_reason: args.status === "inherit" ? undefined : args.status_reason,
    });
    return dataset._id;
  },
});
