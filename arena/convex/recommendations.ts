import { query } from "./_generated/server";
import { v } from "convex/values";
import { loadTaskStatusMap } from "./statuses";
import { effectiveStatus } from "./statusShared";
import {
  RATING_ANCHOR,
  fitBradleyTerry,
  mergePairOutcomes,
  pairOutcomesFromRounds,
  type PairOutcome,
} from "./bradleyTerry";

export const getPairCounts = query({
  args: {
    environment: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    // Get policies (optionally filtered by environment)
    let policies;
    if (args.environment) {
      policies = await ctx.db
        .query("policies")
        .withIndex("by_environment", (q) => q.eq("environment", args.environment!))
        .collect();
    } else {
      policies = await ctx.db.query("policies").collect();
    }
    // The eval planner consumes this — retired/ablation/testing policies must
    // not be proposed as opponents.
    const taskStatuses = await loadTaskStatusMap(ctx);
    policies = policies.filter(
      (p) => effectiveStatus(p.status, p.environment, taskStatuses) === "mainline"
    );

    // Build policy_id -> model_id map
    const idToModelId = new Map<string, string>();
    const policyIds = new Set<string>();
    for (const p of policies) {
      idToModelId.set(p._id as string, p.model_id);
      policyIds.add(p._id as string);
    }

    // Collect all round results for these policies, grouped by (session_id, round_index)
    const roundGroups = new Map<string, string[]>(); // "session_id|round_index" -> [model_id, ...]
    for (const p of policies) {
      const results = await ctx.db
        .query("roundResults")
        .withIndex("by_policy", (q) => q.eq("policy_id", p._id))
        .collect();
      for (const r of results) {
        const key = `${r.session_id}|${r.round_index}`;
        const modelId = idToModelId.get(r.policy_id as string);
        if (!modelId) continue;
        if (!roundGroups.has(key)) {
          roundGroups.set(key, []);
        }
        roundGroups.get(key)!.push(modelId);
      }
    }

    // Count pairwise co-occurrences
    const counts: Record<string, Record<string, number>> = {};
    for (const modelIds of roundGroups.values()) {
      // All pairs within this round group
      for (let i = 0; i < modelIds.length; i++) {
        for (let j = i + 1; j < modelIds.length; j++) {
          const a = modelIds[i];
          const b = modelIds[j];
          if (!counts[a]) counts[a] = {};
          if (!counts[b]) counts[b] = {};
          counts[a][b] = (counts[a][b] || 0) + 1;
          counts[b][a] = (counts[b][a] || 0) + 1;
        }
      }
    }

    return counts;
  },
});

export const getOpponents = query({
  args: {
    environment: v.optional(v.string()),
    exclude_model_ids: v.optional(v.array(v.string())),
  },
  handler: async (ctx, args) => {
    let policies = await ctx.db.query("policies").collect();

    // Only mainline policies are eligible opponents (see getPairCounts note).
    const taskStatuses = await loadTaskStatusMap(ctx);
    policies = policies.filter(
      (p) => effectiveStatus(p.status, p.environment, taskStatuses) === "mainline"
    );

    // Filter by environment if specified
    if (args.environment) {
      policies = policies.filter((p) => p.environment === args.environment);
    }

    // Exclude specific model_ids (e.g. the focus policy in calibrate mode)
    if (args.exclude_model_ids) {
      policies = policies.filter(
        (p) => !args.exclude_model_ids!.includes(p.model_id)
      );
    }

    // Bradley-Terry ratings fit over effectively-mainline sessions — the same
    // set the UI's default lens rates on, returned under the `elo` key;
    // policies without pair games sit at the anchor.
    const sessions = await ctx.db.query("evalSessions").collect();
    const perSessionPairs: PairOutcome[][] = [];
    for (const session of sessions) {
      const dataset = await ctx.db
        .query("datasets")
        .withIndex("by_repo", (q) => q.eq("repo_id", session.dataset_repo))
        .unique();
      const status = effectiveStatus(
        session.status,
        dataset?.task ?? null,
        taskStatuses
      );
      if (status !== "mainline") continue;
      if (session.policy_ids.length < 2) continue; // rollout sessions: no pairs
      const results = await ctx.db
        .query("roundResults")
        .withIndex("by_session", (q) => q.eq("session_id", session._id))
        .collect();
      const rounds = new Map<number, Array<{ id: string; success: boolean }>>();
      for (const r of results) {
        const roundIdx = Number(r.round_index);
        if (!rounds.has(roundIdx)) rounds.set(roundIdx, []);
        rounds.get(roundIdx)!.push({
          id: r.policy_id as string,
          success: r.success,
        });
      }
      perSessionPairs.push(pairOutcomesFromRounds([...rounds.values()]));
    }
    const ratings = fitBradleyTerry(mergePairOutcomes(perSessionPairs));

    const rated = policies.map((p) => ({
      model_id: p.model_id,
      name: p.name,
      elo: ratings.get(p._id as string) ?? RATING_ANCHOR,
    }));
    return rated.sort((a, b) => b.elo - a.elo);
  },
});
