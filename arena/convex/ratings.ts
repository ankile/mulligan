import { query } from "./_generated/server";
import { pairOutcomesFromRounds } from "./bradleyTerry";
import { loadTaskStatusMap } from "./statuses";
import { effectiveStatus } from "./statusShared";

/**
 * Compact per-session outcome aggregates for CLIENT-SIDE rating fits: the
 * frontend filters sessions by the current lens (mainline/all), merges the
 * pair outcomes, and runs fitBradleyTerry — so ratings, W/D/L, and success
 * rates are always self-consistent with whatever the view shows. ~122
 * sessions with a handful of pairs each; the heavy roundResults rows never
 * leave the server.
 */
export const sessionOutcomes = query({
  args: {},
  handler: async (ctx) => {
    const sessions = await ctx.db.query("evalSessions").order("asc").collect();
    const taskStatuses = await loadTaskStatusMap(ctx);

    return Promise.all(
      sessions.map(async (session) => {
        const results = await ctx.db
          .query("roundResults")
          .withIndex("by_session", (q) => q.eq("session_id", session._id))
          .collect();

        const rounds = new Map<
          number,
          Array<{ id: string; success: boolean }>
        >();
        const perPolicy = new Map<
          string,
          {
            rollouts: number;
            successes: number;
            successFramesSum: number;
            successFramesCount: number;
          }
        >();
        for (const r of results) {
          const roundIdx = Number(r.round_index);
          if (!rounds.has(roundIdx)) rounds.set(roundIdx, []);
          rounds.get(roundIdx)!.push({
            id: r.policy_id as string,
            success: r.success,
          });
          let agg = perPolicy.get(r.policy_id as string);
          if (!agg) {
            agg = {
              rollouts: 0,
              successes: 0,
              successFramesSum: 0,
              successFramesCount: 0,
            };
            perPolicy.set(r.policy_id as string, agg);
          }
          agg.rollouts++;
          if (r.success) {
            agg.successes++;
            if (r.num_frames != null) {
              agg.successFramesSum += Number(r.num_frames);
              agg.successFramesCount++;
            }
          }
        }

        const dataset = await ctx.db
          .query("datasets")
          .withIndex("by_repo", (q) => q.eq("repo_id", session.dataset_repo))
          .unique();
        const task = dataset?.task ?? null;

        return {
          session_id: session._id,
          creation_time: session._creationTime,
          session_mode: session.session_mode ?? "manual",
          task,
          effective_status: effectiveStatus(session.status, task, taskStatuses),
          pairs: pairOutcomesFromRounds([...rounds.values()]),
          perPolicy: [...perPolicy.entries()].map(([policy_id, agg]) => ({
            policy_id,
            ...agg,
          })),
        };
      })
    );
  },
});
