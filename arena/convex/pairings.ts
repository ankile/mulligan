import { query } from "./_generated/server";
import { v } from "convex/values";

export const listRounds = query({
  args: {
    policyIdA: v.id("policies"),
    policyIdB: v.optional(v.id("policies")),
  },
  handler: async (ctx, args) => {
    // Load policy A (required)
    const policyA = await ctx.db.get(args.policyIdA);
    if (!policyA) return [];

    // Load all sessions and filter to those containing policy A
    const allSessions = await ctx.db.query("evalSessions").order("desc").collect();
    const sessionsWithA = allSessions.filter((s) =>
      s.policy_ids.some((id) => id === args.policyIdA)
    );

    // If policyIdB specified, further filter to sessions containing both
    const filteredSessions = args.policyIdB
      ? sessionsWithA.filter((s) =>
          s.policy_ids.some((id) => id === args.policyIdB)
        )
      : sessionsWithA;

    // Pre-load all policies for name lookups
    const allPolicies = await ctx.db.query("policies").collect();
    const policyMap = new Map<string, string>();
    for (const p of allPolicies) {
      policyMap.set(p._id as string, p.name);
    }

    // Build flat rounds array
    const rounds: Array<{
      sessionId: string;
      sessionCreationTime: number;
      datasetRepo: string;
      sessionMode: string;
      roundIndex: number;
      results: Array<{
        policyId: string;
        policyName: string;
        success: boolean;
        episodeIndex: number;
      }>;
    }> = [];

    for (const session of filteredSessions) {
      const results = await ctx.db
        .query("roundResults")
        .withIndex("by_session", (q) => q.eq("session_id", session._id))
        .collect();

      // Group by round_index
      const roundsMap = new Map<
        number,
        Array<{
          policyId: string;
          policyName: string;
          success: boolean;
          episodeIndex: number;
        }>
      >();

      for (const r of results) {
        const idx = Number(r.round_index);
        if (!roundsMap.has(idx)) roundsMap.set(idx, []);
        roundsMap.get(idx)!.push({
          policyId: r.policy_id as string,
          policyName: policyMap.get(r.policy_id as string) ?? "Unknown",
          success: r.success,
          episodeIndex: Number(r.episode_index),
        });
      }

      for (const [roundIndex, roundResults] of roundsMap) {
        // Only include rounds where policy A appears with at least one other policy
        const hasA = roundResults.some((r) => r.policyId === (args.policyIdA as string));
        const hasOther = roundResults.some((r) => r.policyId !== (args.policyIdA as string));
        if (!hasA || !hasOther) continue;

        // If policyIdB specified, only include rounds where B also appears
        if (args.policyIdB) {
          const hasB = roundResults.some((r) => r.policyId === (args.policyIdB as string));
          if (!hasB) continue;
        }

        rounds.push({
          sessionId: session._id as string,
          sessionCreationTime: session._creationTime,
          datasetRepo: session.dataset_repo,
          sessionMode: session.session_mode ?? "manual",
          roundIndex,
          results: roundResults.sort((a, b) => {
            // Policy A first, then policy B (if specified), then others
            const aIsA = a.policyId === (args.policyIdA as string);
            const bIsA = b.policyId === (args.policyIdA as string);
            if (aIsA !== bIsA) return aIsA ? -1 : 1;
            if (args.policyIdB) {
              const aIsB = a.policyId === (args.policyIdB as string);
              const bIsB = b.policyId === (args.policyIdB as string);
              if (aIsB !== bIsB) return aIsB ? -1 : 1;
            }
            return 0;
          }),
        });
      }
    }

    // Sort by session creation time desc, then round_index asc
    rounds.sort((a, b) => {
      if (a.sessionCreationTime !== b.sessionCreationTime) {
        return b.sessionCreationTime - a.sessionCreationTime;
      }
      return a.roundIndex - b.roundIndex;
    });

    return rounds;
  },
});
