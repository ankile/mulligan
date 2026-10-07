import {
  fitBradleyTerry,
  mergePairOutcomes,
  type PairOutcome,
} from "../../convex/bradleyTerry";

export type SessionOutcome = {
  session_id: string;
  creation_time: number;
  session_mode: string;
  task: string | null;
  rating_group?: string;
  effective_status: string;
  pairs: PairOutcome[];
  perPolicy: Array<{
    policy_id: string;
    rollouts: number;
    successes: number;
    successFramesSum: number;
    successFramesCount: number;
  }>;
};

export type PolicyArenaStats = {
  /** Bradley-Terry rating; absent when the policy has no pair games in view. */
  ratings: Map<string, number>;
  wdl: Map<string, { wins: number; draws: number; losses: number }>;
  success: Map<
    string,
    { rollouts: number; successes: number; avgSuccessSteps: number | null }
  >;
};

export function visibleSessions(
  sessions: SessionOutcome[],
  showAll: boolean
): SessionOutcome[] {
  return showAll
    ? sessions
    : sessions.filter((s) => s.effective_status === "mainline");
}

/** Ratings + W/D/L + success stats over exactly the given sessions. */
export function computeArenaStats(sessions: SessionOutcome[], separateGroups = false): PolicyArenaStats {
  const merged = mergePairOutcomes(sessions.map((s) => s.pairs));
  const ratings = separateGroups
    ? new Map([...new Set(sessions.map(s => s.rating_group ?? s.task))].flatMap(group =>
        [...fitBradleyTerry(mergePairOutcomes(sessions.filter(s => (s.rating_group ?? s.task) === group).map(s => s.pairs)))]))
    : fitBradleyTerry(merged);

  const wdl = new Map<string, { wins: number; draws: number; losses: number }>();
  const ensureWdl = (id: string) => {
    let w = wdl.get(id);
    if (!w) {
      w = { wins: 0, draws: 0, losses: 0 };
      wdl.set(id, w);
    }
    return w;
  };
  for (const p of merged) {
    const a = ensureWdl(p.a);
    const b = ensureWdl(p.b);
    a.wins += p.winsA;
    a.losses += p.winsB;
    a.draws += p.draws;
    b.wins += p.winsB;
    b.losses += p.winsA;
    b.draws += p.draws;
  }

  const successAgg = new Map<
    string,
    { rollouts: number; successes: number; framesSum: number; framesCount: number }
  >();
  for (const s of sessions) {
    for (const pp of s.perPolicy) {
      let agg = successAgg.get(pp.policy_id);
      if (!agg) {
        agg = { rollouts: 0, successes: 0, framesSum: 0, framesCount: 0 };
        successAgg.set(pp.policy_id, agg);
      }
      agg.rollouts += pp.rollouts;
      agg.successes += pp.successes;
      agg.framesSum += pp.successFramesSum;
      agg.framesCount += pp.successFramesCount;
    }
  }
  const success = new Map<
    string,
    { rollouts: number; successes: number; avgSuccessSteps: number | null }
  >();
  for (const [id, agg] of successAgg) {
    success.set(id, {
      rollouts: agg.rollouts,
      successes: agg.successes,
      avgSuccessSteps:
        agg.framesCount > 0 ? Math.round(agg.framesSum / agg.framesCount) : null,
    });
  }

  return { ratings, wdl, success };
}

/**
 * Rating-over-time for one policy: refit on the chronological session prefix
 * at each session the policy has pair games in. Order matters only in the
 * legitimate "rating as of then" sense; every point is itself an
 * order-independent fit.
 */
export function ratingTrajectory(
  sessions: SessionOutcome[],
  policyId: string
): Array<{ sessionId: string; creationTime: number; rating: number }> {
  const sorted = [...sessions].sort((a, b) => a.creation_time - b.creation_time);
  const out: Array<{ sessionId: string; creationTime: number; rating: number }> =
    [];
  for (let i = 0; i < sorted.length; i++) {
    const involves = sorted[i].pairs.some(
      (p) => p.a === policyId || p.b === policyId
    );
    if (!involves) continue;
    const prefix = mergePairOutcomes(
      sorted.slice(0, i + 1).map((s) => s.pairs)
    );
    const rating = fitBradleyTerry(prefix).get(policyId);
    if (rating === undefined) continue;
    out.push({
      sessionId: sorted[i].session_id,
      creationTime: sorted[i].creation_time,
      rating,
    });
  }
  return out;
}
