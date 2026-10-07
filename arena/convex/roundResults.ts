import { query } from "./_generated/server";
import { v } from "convex/values";
import type { Id } from "./_generated/dataModel";

export const getFailuresByPolicy = query({
  args: { policy_id: v.id("policies") },
  handler: async (ctx, args) => {
    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_policy", (q) => q.eq("policy_id", args.policy_id))
      .order("desc")
      .collect();

    const failures = results.filter((r) => !r.success).slice(0, 20);

    return Promise.all(
      failures.map(async (r) => {
        const session = await ctx.db.get(r.session_id);
        return {
          session_id: r.session_id,
          dataset_repo: session?.dataset_repo ?? "",
          round_index: Number(r.round_index),
          episode_index: Number(r.episode_index),
          success: r.success,
          num_frames: r.num_frames != null ? Number(r.num_frames) : null,
          session_creation_time: session?._creationTime ?? 0,
        };
      })
    );
  },
});

export const getSuccessRateHistory = query({
  args: { policy_id: v.id("policies") },
  handler: async (ctx, args) => {
    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_policy", (q) => q.eq("policy_id", args.policy_id))
      .collect();

    const bySession = new Map<
      string,
      { successes: number; total: number; session_id: Id<"evalSessions"> }
    >();
    for (const r of results) {
      const key = r.session_id as string;
      const entry = bySession.get(key) ?? {
        successes: 0,
        total: 0,
        session_id: r.session_id,
      };
      entry.total++;
      if (r.success) entry.successes++;
      bySession.set(key, entry);
    }

    const entries = await Promise.all(
      [...bySession.values()].map(async ({ successes, total, session_id }) => {
        const session = await ctx.db.get(session_id);
        return {
          successRate: successes / total,
          successes,
          total,
          datasetRepo: session?.dataset_repo ?? "",
          sessionCreationTime: session?._creationTime ?? 0,
        };
      })
    );

    return entries.sort((a, b) => a.sessionCreationTime - b.sessionCreationTime);
  },
});

export const getRecentByPolicy = query({
  args: { policy_id: v.id("policies") },
  handler: async (ctx, args) => {
    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_policy", (q) => q.eq("policy_id", args.policy_id))
      .order("desc")
      .take(20);

    return Promise.all(
      results.map(async (r) => {
        const session = await ctx.db.get(r.session_id);
        return {
          session_id: r.session_id,
          dataset_repo: session?.dataset_repo ?? "",
          round_index: Number(r.round_index),
          episode_index: Number(r.episode_index),
          success: r.success,
          num_frames: r.num_frames != null ? Number(r.num_frames) : null,
          session_creation_time: session?._creationTime ?? 0,
        };
      })
    );
  },
});
