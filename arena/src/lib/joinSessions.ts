// Pure helpers for the "join sessions" view: N eval sessions picked by the
// user, aligned round-by-round on round_index so every round's rollouts from
// every session can be shown side by side. No React, no network — unit-tested
// in joinSessions.test.ts.

import type { Arm } from "./armStats";

export interface JoinRoundResult {
  policy_id: string;
  policyName: string;
  success: boolean;
  episode_index: number;
  /** Sub-goal marks reached; null/absent on rounds submitted without them. */
  num_subtask_marks?: number | null;
  /** Episode length in control steps; null/absent on rounds submitted without it. */
  num_frames?: number | null;
}

export interface JoinSide {
  sessionId: string;
  rounds: Array<{ index: number; results: JoinRoundResult[] }>;
}

export interface JoinedRound {
  index: number;
  // One entry per side, in the order the sides were given. `null` when that
  // session has no round with this index (sessions may differ in length).
  perSide: Array<JoinRoundResult[] | null>;
}

// Session letters shown on chips/badges: A, B, C, ...
export function sessionLetter(position: number): string {
  if (position < 0 || position >= 26) {
    throw new Error(`Session position ${position} out of letter range`);
  }
  return String.fromCharCode(65 + position);
}

/**
 * Align rounds across sides by round index: the union of every side's round
 * indices, ascending, with a per-side slot that is null where a side lacks
 * that round.
 */
export function alignRounds(sides: JoinSide[]): JoinedRound[] {
  const indices = new Set<number>();
  const bySide = sides.map((side) => {
    const map = new Map<number, JoinRoundResult[]>();
    for (const round of side.rounds) {
      if (map.has(round.index)) {
        throw new Error(
          `Session ${side.sessionId} has duplicate round index ${round.index}`,
        );
      }
      map.set(round.index, round.results);
      indices.add(round.index);
    }
    return map;
  });
  return [...indices]
    .sort((a, b) => a - b)
    .map((index) => ({
      index,
      perSide: bySide.map((map) => map.get(index) ?? null),
    }));
}

export interface PolicySuccessSummary {
  policy_id: string;
  policyName: string;
  successes: number;
  rounds: number;
}

/** Per-policy success counts within one side, in first-seen policy order. */
export function sideSuccessSummary(side: JoinSide): PolicySuccessSummary[] {
  const byPolicy = new Map<string, PolicySuccessSummary>();
  for (const round of side.rounds) {
    for (const result of round.results) {
      let entry = byPolicy.get(result.policy_id);
      if (!entry) {
        entry = {
          policy_id: result.policy_id,
          policyName: result.policyName,
          successes: 0,
          rounds: 0,
        };
        byPolicy.set(result.policy_id, entry);
      }
      entry.rounds += 1;
      if (result.success) entry.successes += 1;
    }
  }
  return [...byPolicy.values()];
}

/** Human summary of how the sides overlap, e.g. "40 aligned · 10 A-only". */
export function alignmentSummary(rounds: JoinedRound[]): string {
  const sideCount = rounds[0]?.perSide.length ?? 0;
  let aligned = 0;
  const onlyCounts = new Array<number>(sideCount).fill(0);
  for (const round of rounds) {
    const present = round.perSide
      .map((results, i) => (results ? i : -1))
      .filter((i) => i >= 0);
    if (present.length === sideCount) {
      aligned += 1;
    } else if (present.length === 1) {
      onlyCounts[present[0]] += 1;
    }
  }
  const parts = [`${aligned} aligned`];
  onlyCounts.forEach((count, i) => {
    if (count > 0) parts.push(`${count} ${sessionLetter(i)}-only`);
  });
  return parts.join(" · ");
}

export interface JoinedPolicy {
  policy_id: string;
  policyName: string;
}

/** Union of policies across all sides, in first-seen (side, round) order. */
export function joinedPolicies(sides: JoinSide[]): JoinedPolicy[] {
  const seen = new Map<string, JoinedPolicy>();
  for (const side of sides) {
    for (const round of side.rounds) {
      for (const result of round.results) {
        if (!seen.has(result.policy_id)) {
          seen.set(result.policy_id, {
            policy_id: result.policy_id,
            policyName: result.policyName,
          });
        }
      }
    }
  }
  return [...seen.values()];
}

/** Arm key for (session position, policy): the same policy in two sessions stays two arms. */
export function joinedArmKey(sideIdx: number, policyId: string): string {
  return `${sessionLetter(sideIdx)}:${policyId}`;
}

/**
 * One arm per (session, policy), in side order then each side's first-seen
 * policy order. Policy numbers are assigned over the UNION of policies across
 * all sides (first-seen order, hidden policies included so numbers do not
 * shift when a policy is toggled), and the label is session letter + number:
 * the same model in sessions A and B reads `A3` and `B3`.
 */
export function joinedArms(sides: JoinSide[], hiddenPolicyIds: ReadonlySet<string>): Arm[] {
  const numberOf = new Map(joinedPolicies(sides).map((p, i) => [p.policy_id, i + 1]));
  return sides.flatMap((side, i) =>
    sideSuccessSummary(side)
      .filter((p) => !hiddenPolicyIds.has(p.policy_id))
      .map((p) => {
        const policyNumber = numberOf.get(p.policy_id)!;
        return {
          key: joinedArmKey(i, p.policy_id),
          name: p.policyName,
          policyNumber,
          label: `${sessionLetter(i)}${policyNumber}`,
          session: sessionLetter(i),
        };
      }),
  );
}

/**
 * Drop every result whose policy is hidden. A side that HAS the round but
 * whose rollouts are all hidden becomes `[]` (distinct from `null` = the
 * session never ran that round), so the UI can say "all hidden" rather than
 * "no round k". Rounds are never dropped: alignment is about sessions.
 */
export function hidePolicies(
  rounds: JoinedRound[],
  hiddenPolicyIds: ReadonlySet<string>,
): JoinedRound[] {
  if (hiddenPolicyIds.size === 0) return rounds;
  return rounds.map((round) => ({
    index: round.index,
    perSide: round.perSide.map((results) =>
      results === null
        ? null
        : results.filter((r) => !hiddenPolicyIds.has(r.policy_id)),
    ),
  }));
}

// URL id lists: `?join=<sessionId>,<sessionId>` (order-preserving; order
// defines the A/B/C letters) and `?hide=<policyId>,<policyId>` (policies
// hidden from the side-by-side rounds).
export function parseIdList(raw: string | null): string[] {
  if (raw === null || raw === "") return [];
  const ids = raw.split(",").filter((id) => id.length > 0);
  return [...new Set(ids)];
}

export function formatIdList(ids: string[]): string | null {
  return ids.length === 0 ? null : ids.join(",");
}

export function toggleId(ids: string[], id: string): string[] {
  return ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id];
}
