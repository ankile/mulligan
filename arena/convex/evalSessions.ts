import { query, mutation, internalMutation } from "./_generated/server";
import type { MutationCtx } from "./_generated/server";
import { ConvexError, v } from "convex/values";
import type { Id } from "./_generated/dataModel";
import { requireEditorOrService } from "./access";
import { loadTaskStatusMap } from "./statuses";
import {
  effectiveStatus,
  statusValidator,
  statusOrInheritValidator,
} from "./statusShared";

function uniqueRoundIndexes(
  rounds: Array<{ round_index: bigint }>,
): Set<number> {
  const indexes = new Set<number>();
  for (const round of rounds) {
    const index = Number(round.round_index);
    if (indexes.has(index)) {
      throw new Error(`Duplicate round_index ${index} in submitted rounds`);
    }
    indexes.add(index);
  }
  return indexes;
}

async function requireKnownOperator(
  ctx: MutationCtx,
  operator: string,
): Promise<string> {
  const username = operator.trim();
  const row = await ctx.db
    .query("operators")
    .withIndex("by_username", (q) => q.eq("hf_username", username))
    .unique();
  if (!row) {
    throw new Error(
      `Unknown operator ${JSON.stringify(username)} — add them first via operators:add`,
    );
  }
  return username;
}

export const submit = mutation({
  args: {
    serviceToken: v.optional(v.string()),
    dataset_repo: v.string(),
    submission_id: v.optional(v.string()),
    submission_fingerprint: v.optional(v.string()),
    notes: v.optional(v.string()),
    session_mode: v.optional(v.string()),
    status: v.optional(statusValidator), // e.g. tag an ablation eval at submit time
    // HF username of the human who ran the eval (validated against the
    // `operators` registry; see schema.ts for the auth-join rationale).
    operator: v.optional(v.string()),
    policies: v.array(
      v.object({
        name: v.string(),
        model_id: v.string(),
        model_url: v.optional(v.string()),
        training_url: v.optional(v.string()),
        environment: v.string(),
      }),
    ),
    rounds: v.array(
      v.object({
        round_index: v.int64(),
        results: v.array(
          v.object({
            model_id: v.string(),
            success: v.boolean(),
            episode_index: v.int64(),
            num_frames: v.optional(v.int64()),
            num_subtask_marks: v.optional(v.int64()),
          }),
        ),
      }),
    ),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    if (
      (args.submission_id === undefined) !==
      (args.submission_fingerprint === undefined)
    ) {
      throw new Error(
        "submission_id and submission_fingerprint must be provided together",
      );
    }
    if (args.submission_id !== undefined) {
      const existingSession = await ctx.db
        .query("evalSessions")
        .withIndex("by_submission_id", (q) =>
          q.eq("submission_id", args.submission_id),
        )
        .unique();
      if (existingSession !== null) {
        if (
          existingSession.submission_fingerprint !== args.submission_fingerprint
        ) {
          throw new ConvexError({ code: "idempotency_conflict" });
        }
        return existingSession._id;
      }
    }
    const operator =
      args.operator !== undefined
        ? await requireKnownOperator(ctx, args.operator)
        : undefined;
    // 1. Register/upsert all policies
    const modelIdToPolicy = new Map<string, Id<"policies">>();
    for (const p of args.policies) {
      const existing = await ctx.db
        .query("policies")
        .withIndex("by_model_id", (q) => q.eq("model_id", p.model_id))
        .unique();

      if (existing) {
        await ctx.db.patch(existing._id, {
          name: p.name,
          model_url: p.model_url,
          training_url: p.training_url,
          environment: p.environment,
        });
        modelIdToPolicy.set(p.model_id, existing._id);
      } else {
        const id = await ctx.db.insert("policies", {
          name: p.name,
          model_id: p.model_id,
          model_url: p.model_url,
          training_url: p.training_url,
          environment: p.environment,
        });
        modelIdToPolicy.set(p.model_id, id);
      }
    }

    const policyIds = args.policies.map(
      (p) => modelIdToPolicy.get(p.model_id)!,
    );

    const roundIndexes = uniqueRoundIndexes(args.rounds);

    // 2. Create eval session
    const sessionId = await ctx.db.insert("evalSessions", {
      dataset_repo: args.dataset_repo,
      num_rounds: BigInt(roundIndexes.size),
      policy_ids: policyIds,
      submission_id: args.submission_id,
      submission_fingerprint: args.submission_fingerprint,
      notes: args.notes,
      session_mode: args.session_mode,
      status: args.status,
      operator,
    });

    // 3. Insert round results
    for (const round of args.rounds) {
      for (const result of round.results) {
        const policyId = modelIdToPolicy.get(result.model_id)!;
        await ctx.db.insert("roundResults", {
          session_id: sessionId,
          round_index: BigInt(Number(round.round_index)),
          policy_id: policyId,
          success: result.success,
          episode_index: BigInt(Number(result.episode_index)),
          ...(result.num_frames != null
            ? { num_frames: BigInt(Number(result.num_frames)) }
            : {}),
          ...(result.num_subtask_marks != null
            ? { num_subtask_marks: BigInt(Number(result.num_subtask_marks)) }
            : {}),
        });
      }
    }

    // Ratings are Bradley-Terry, fit on read (convex/bradleyTerry.ts) — no
    // per-submit rating writes.
    return sessionId;
  },
});

export const list = query({
  args: {},
  handler: async (ctx) => {
    const sessions = await ctx.db.query("evalSessions").order("desc").collect();
    const taskStatuses = await loadTaskStatusMap(ctx);

    return Promise.all(
      sessions.map(async (session) => {
        const policyNames = await Promise.all(
          session.policy_ids.map(async (id) => {
            const policy = await ctx.db.get(id);
            return policy?.name ?? "Unknown";
          }),
        );
        const dataset = await ctx.db
          .query("datasets")
          .withIndex("by_repo", (q) => q.eq("repo_id", session.dataset_repo))
          .unique();
        const task = dataset?.task ?? null;
        return {
          ...session,
          policyNames,
          task,
          effective_status: effectiveStatus(session.status, task, taskStatuses),
          derivedDatasetRepos: dataset?.derived_repo_ids ?? [],
        };
      }),
    );
  },
});

export const getDetail = query({
  args: { id: v.id("evalSessions") },
  handler: async (ctx, args) => {
    const session = await ctx.db.get(args.id);
    if (!session) return null;

    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_session", (q) => q.eq("session_id", args.id))
      .collect();

    // Group results by round
    const roundsMap = new Map<
      number,
      Array<{
        policy_id: string;
        policyName: string;
        success: boolean;
        episode_index: number;
        num_subtask_marks: number | null;
        num_frames: number | null;
      }>
    >();

    for (const r of results) {
      const roundIdx = Number(r.round_index);
      if (!roundsMap.has(roundIdx)) roundsMap.set(roundIdx, []);
      const policy = await ctx.db.get(r.policy_id);
      roundsMap.get(roundIdx)!.push({
        policy_id: r.policy_id,
        policyName: policy?.name ?? "Unknown",
        success: r.success,
        episode_index: Number(r.episode_index),
        num_subtask_marks:
          r.num_subtask_marks === undefined ? null : Number(r.num_subtask_marks),
        num_frames: r.num_frames === undefined ? null : Number(r.num_frames),
      });
    }

    // Sub-goal budget of the session's task (0 = binary task): the dataset
    // registration names the task, taskSpecs carries RealTaskSpec.num_subtask_marks.
    const dataset = await ctx.db
      .query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", session.dataset_repo))
      .unique();
    const taskSpec = dataset
      ? await ctx.db
          .query("taskSpecs")
          .withIndex("by_task", (q) => q.eq("task", dataset.task))
          .unique()
      : null;
    const maxSubtaskMarks = taskSpec ? Number(taskSpec.num_subtask_marks) : 0;

    const policies = await Promise.all(
      session.policy_ids.map(async (id) => {
        const policy = await ctx.db.get(id);
        return policy!;
      }),
    );

    // Sort each round's results to match session.policy_ids order
    const policyIdOrder = session.policy_ids.map(String);
    for (const [, roundResults] of roundsMap) {
      roundResults.sort(
        (a, b) =>
          policyIdOrder.indexOf(a.policy_id) -
          policyIdOrder.indexOf(b.policy_id),
      );
    }

    return {
      ...session,
      policies,
      max_subtask_marks: maxSubtaskMarks,
      rounds: Array.from(roundsMap.entries())
        .sort(([a], [b]) => a - b)
        .map(([index, results]) => ({ index, results })),
    };
  },
});

export const deleteSession = mutation({
  args: { id: v.id("evalSessions"), serviceToken: v.optional(v.string()) },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");

    // Delete round results, then the session. Ratings are fit on read, so
    // no recompute/replay is needed.
    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_session", (q) => q.eq("session_id", args.id))
      .collect();
    for (const r of results) {
      await ctx.db.delete(r._id);
    }
    await ctx.db.delete(args.id);

    return { deleted: args.id, deleted_results: results.length };
  },
});

export const removePolicyFromSession = mutation({
  args: {
    id: v.id("evalSessions"),
    model_id: v.string(),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");

    // 1. Look up the policy by model_id
    const policy = await ctx.db
      .query("policies")
      .withIndex("by_model_id", (q) => q.eq("model_id", args.model_id))
      .unique();
    if (!policy) throw new Error(`Policy not found: ${args.model_id}`);

    if (!session.policy_ids.some((id) => id === policy._id)) {
      throw new Error(`Policy ${args.model_id} is not in session ${args.id}`);
    }

    // 2. Delete round results for this policy in this session
    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_session", (q) => q.eq("session_id", args.id))
      .collect();
    let deletedCount = 0;
    for (const r of results) {
      if (r.policy_id === policy._id) {
        await ctx.db.delete(r._id);
        deletedCount++;
      }
    }

    // 3. Remove policy from session's policy_ids. Ratings are fit on read,
    // so no recompute/replay is needed.
    const updatedPolicyIds = session.policy_ids.filter(
      (id) => id !== policy._id,
    );
    await ctx.db.patch(args.id, { policy_ids: updatedPolicyIds });

    return {
      session_id: args.id,
      removed_model_id: args.model_id,
      removed_policy_id: policy._id,
      deleted_results: deletedCount,
      remaining_policies: updatedPolicyIds.length,
    };
  },
});

export const addRounds = mutation({
  args: {
    id: v.id("evalSessions"),
    serviceToken: v.optional(v.string()),
    policies: v.array(
      v.object({
        name: v.string(),
        model_id: v.string(),
        model_url: v.optional(v.string()),
        training_url: v.optional(v.string()),
        environment: v.string(),
      }),
    ),
    rounds: v.array(
      v.object({
        round_index: v.int64(),
        results: v.array(
          v.object({
            model_id: v.string(),
            success: v.boolean(),
            episode_index: v.int64(),
            num_frames: v.optional(v.int64()),
            num_subtask_marks: v.optional(v.int64()),
          }),
        ),
      }),
    ),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");
    const incomingRoundIndexes = uniqueRoundIndexes(args.rounds);

    const existingResults = await ctx.db
      .query("roundResults")
      .withIndex("by_session", (q) => q.eq("session_id", args.id))
      .collect();
    const existingRoundIndexes = new Set(
      existingResults.map((r) => Number(r.round_index)),
    );
    // Dedup at (round_index, policy) granularity, not per round: a phased eval
    // can submit some arms' rollouts for a round and append the other arms'
    // rollouts to the SAME round later. A result that already exists for its
    // (round, policy) is rejected, so a double submission fails loud.
    const existingPairs = new Set(
      existingResults.map((r) => `${Number(r.round_index)}:${String(r.policy_id)}`),
    );

    // 1. Register/upsert all policies
    const modelIdToPolicy = new Map<string, Id<"policies">>();
    for (const p of args.policies) {
      const existing = await ctx.db
        .query("policies")
        .withIndex("by_model_id", (q) => q.eq("model_id", p.model_id))
        .unique();

      if (existing) {
        await ctx.db.patch(existing._id, {
          name: p.name,
          model_url: p.model_url,
          training_url: p.training_url,
          environment: p.environment,
        });
        modelIdToPolicy.set(p.model_id, existing._id);
      } else {
        const id = await ctx.db.insert("policies", {
          name: p.name,
          model_id: p.model_id,
          model_url: p.model_url,
          training_url: p.training_url,
          environment: p.environment,
        });
        modelIdToPolicy.set(p.model_id, id);
      }
    }

    // 2. Expand session's policy_ids if new policies appeared
    const existingPolicyIds = new Set(session.policy_ids.map(String));
    const updatedPolicyIds = [...session.policy_ids];
    for (const [, id] of modelIdToPolicy) {
      if (!existingPolicyIds.has(String(id))) {
        updatedPolicyIds.push(id);
      }
    }

    // 3. Insert round results (fail loud on an existing (round, policy) pair)
    for (const round of args.rounds) {
      const roundIndex = Number(round.round_index);
      for (const result of round.results) {
        const policyId = modelIdToPolicy.get(result.model_id)!;
        const pairKey = `${roundIndex}:${String(policyId)}`;
        if (existingPairs.has(pairKey)) {
          throw new Error(
            `Result already exists in session ${args.id} for round ${roundIndex}, policy ${result.model_id}`,
          );
        }
        existingPairs.add(pairKey);
        await ctx.db.insert("roundResults", {
          session_id: args.id,
          round_index: BigInt(Number(round.round_index)),
          policy_id: policyId,
          success: result.success,
          episode_index: BigInt(Number(result.episode_index)),
          ...(result.num_frames != null
            ? { num_frames: BigInt(Number(result.num_frames)) }
            : {}),
          ...(result.num_subtask_marks != null
            ? { num_subtask_marks: BigInt(Number(result.num_subtask_marks)) }
            : {}),
        });
      }
    }

    // 4. Update session metadata
    const newNumRounds = BigInt(
      new Set([...existingRoundIndexes, ...incomingRoundIndexes]).size,
    );
    await ctx.db.patch(args.id, {
      num_rounds: newNumRounds,
      policy_ids: updatedPolicyIds,
      notes: `Eval: ${updatedPolicyIds.length} policies, ${newNumRounds} rounds`,
    });

    // Ratings are Bradley-Terry, fit on read — no per-submit rating writes.
    return args.id;
  },
});

export const setStatus = mutation({
  args: {
    id: v.id("evalSessions"),
    status: statusOrInheritValidator,
    status_reason: v.optional(v.string()),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");
    await ctx.db.patch(args.id, {
      status: args.status === "inherit" ? undefined : args.status,
      status_reason: args.status === "inherit" ? undefined : args.status_reason,
    });
    return args.id;
  },
});

export const setOperator = mutation({
  args: {
    id: v.id("evalSessions"),
    operator: v.string(),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");
    const operator = await requireKnownOperator(ctx, args.operator);
    await ctx.db.patch(args.id, { operator });
    return args.id;
  },
});

export const updateNotes = mutation({
  args: {
    id: v.id("evalSessions"),
    notes: v.string(),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");
    await ctx.db.patch(args.id, { notes: args.notes });
    return args.id;
  },
});

export const getByDatasetRepo = query({
  args: {
    dataset_repo: v.string(),
    session_mode: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const sessions = await ctx.db.query("evalSessions").order("desc").collect();

    const matches = sessions.filter(
      (s) =>
        s.dataset_repo === args.dataset_repo &&
        (args.session_mode == null || s.session_mode === args.session_mode),
    );

    return matches.length > 0 ? matches[0] : null;
  },
});

export const getByPolicy = query({
  args: { policy_id: v.id("policies") },
  handler: async (ctx, args) => {
    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_policy", (q) => q.eq("policy_id", args.policy_id))
      .collect();

    // Get unique session IDs
    const sessionIds = [
      ...new Set(results.map((r) => r.session_id)),
    ] as Id<"evalSessions">[];
    const sessions = await Promise.all(
      sessionIds.map(async (id) => {
        const session = await ctx.db.get(id);
        return session!;
      }),
    );

    return sessions.sort((a, b) => b._creationTime - a._creationTime);
  },
});

/**
 * Sync a session's roundResults with corrected episode outcomes after an
 * outcome-review apply (applyWorker). Ratings are fit on read
 * (bradleyTerry.ts), so patching success/num_frames in place keeps the
 * session's identity and history.
 */
const correctedOutcomesArgs = {
  dataset_repo: v.string(),
  corrections: v.array(
    v.object({
      episode_index: v.int64(),
      success: v.boolean(),
      num_frames: v.int64(),
      num_subtask_marks: v.optional(v.int64()),
    }),
  ),
};

async function correctOutcomesForRepo(
  ctx: MutationCtx,
  args: {
    dataset_repo: string;
    corrections: Array<{
      episode_index: bigint;
      success: boolean;
      num_frames: bigint;
      num_subtask_marks?: bigint;
    }>;
  },
) {
    const sessions = await ctx.db.query("evalSessions").order("desc").collect();
    const session = sessions.find((s) => s.dataset_repo === args.dataset_repo);
    if (!session) return { session_found: false, updated: 0 };

    const byEpisode = new Map(
      args.corrections.map((c) => [Number(c.episode_index), c]),
    );
    const results = await ctx.db
      .query("roundResults")
      .withIndex("by_session", (q) => q.eq("session_id", session._id))
      .collect();
    let updated = 0;
    for (const result of results) {
      const corrected = byEpisode.get(Number(result.episode_index));
      if (corrected === undefined) {
        throw new Error(
          `Episode ${result.episode_index} in session ${session._id} has no ` +
            `corrected outcome from the dataset parquet`,
        );
      }
      const patch: {
        success?: boolean;
        num_frames?: bigint;
        num_subtask_marks?: bigint;
      } = {};
      if (result.success !== corrected.success)
        patch.success = corrected.success;
      if (
        result.num_frames === undefined ||
        Number(result.num_frames) !== Number(corrected.num_frames)
      ) {
        patch.num_frames = corrected.num_frames;
      }
      if (
        corrected.num_subtask_marks !== undefined &&
        (result.num_subtask_marks === undefined ||
          Number(result.num_subtask_marks) !== Number(corrected.num_subtask_marks))
      ) {
        patch.num_subtask_marks = corrected.num_subtask_marks;
      }
      if (Object.keys(patch).length > 0) {
        await ctx.db.patch(result._id, patch);
        updated += 1;
      }
    }
    return { session_found: true, updated };
}

export const correctOutcomes = internalMutation({
  args: correctedOutcomesArgs,
  handler: correctOutcomesForRepo,
});
