import { mutation, query, type MutationCtx, type QueryCtx } from "./_generated/server";
import type { Id } from "./_generated/dataModel";
import { v } from "convex/values";
import { requireEditorOrService } from "./access";

async function getSessionRoundCount(
  ctx: QueryCtx | MutationCtx,
  sessionId: Id<"evalSessions">
): Promise<number> {
  const results = await ctx.db
    .query("roundResults")
    .withIndex("by_session", (q) => q.eq("session_id", sessionId))
    .collect();
  return new Set(results.map((r) => Number(r.round_index))).size;
}

// W/D/L counters are not stored (they are derived on read from
// ratings:sessionOutcomes), so audit/repair covers only the session's own
// derived fields: num_rounds and the auto-generated notes string.
export const auditSessionDerivedData = query({
  args: { id: v.id("evalSessions") },
  handler: async (ctx, args) => {
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");
    const actualRoundCount = await getSessionRoundCount(ctx, args.id);

    return {
      session: {
        id: session._id,
        storedNumRounds: Number(session.num_rounds),
        actualRoundCount,
        notes: session.notes,
      },
    };
  },
});

export const repairSessionDerivedData = mutation({
  args: { id: v.id("evalSessions"), serviceToken: v.optional(v.string()) },
  handler: async (ctx, args) => {
    await requireEditorOrService(ctx, args.serviceToken);
    const session = await ctx.db.get(args.id);
    if (!session) throw new Error("Session not found");

    const actualRoundCount = await getSessionRoundCount(ctx, args.id);
    const sessionPatch: {
      num_rounds: bigint;
      notes?: string;
    } = { num_rounds: BigInt(actualRoundCount) };

    if (/^Eval: \d+ policies, \d+ rounds$/.test(session.notes ?? "")) {
      sessionPatch.notes = `Eval: ${session.policy_ids.length} policies, ${actualRoundCount} rounds`;
    }
    await ctx.db.patch(args.id, sessionPatch);

    return {
      sessionId: args.id,
      numRounds: actualRoundCount,
    };
  },
});
