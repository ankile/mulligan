/**
 * Bradley-Terry ratings fit from a SET of pairwise outcomes — order-
 * independent and deterministic, unlike sequential ELO. Shared by the
 * frontend (per-view live fits over whatever the current filter shows) and
 * Convex functions (opponent recommendations). Pure module: no convex/react
 * imports.
 *
 * Model: P(a beats b) = p_a / (p_a + p_b). Draws count as half a win for
 * each side. Fit by minorization-maximization (Hunter 2004) with a small
 * pseudo-game prior against a fixed virtual opponent (strength 1), which
 * keeps all-win/all-loss players finite and pulls unplayed players toward
 * the anchor. Displayed on a chess-like scale:
 *   rating = 1500 + 400 * log10(p)   with mean(log p) normalized to 0.
 */

export type PairOutcome = {
  a: string;
  b: string;
  winsA: number;
  winsB: number;
  draws: number;
};

export const RATING_ANCHOR = 1500;
export const RATING_SCALE = 400;

/**
 * Pairwise outcomes from one session's rounds: within each round, every
 * policy pair yields a win (success vs failure) or a draw (same outcome).
 */
export function pairOutcomesFromRounds(
  rounds: Array<Array<{ id: string; success: boolean }>>
): PairOutcome[] {
  const byKey = new Map<string, PairOutcome>();
  for (const results of rounds) {
    for (let i = 0; i < results.length; i++) {
      for (let j = i + 1; j < results.length; j++) {
        const [x, y] =
          results[i].id < results[j].id
            ? [results[i], results[j]]
            : [results[j], results[i]];
        const key = `${x.id}|${y.id}`;
        let pair = byKey.get(key);
        if (!pair) {
          pair = { a: x.id, b: y.id, winsA: 0, winsB: 0, draws: 0 };
          byKey.set(key, pair);
        }
        if (x.success && !y.success) pair.winsA++;
        else if (!x.success && y.success) pair.winsB++;
        else pair.draws++;
      }
    }
  }
  return [...byKey.values()];
}

/** Merge outcome lists (e.g. one per session) into canonical a<b pairs. */
export function mergePairOutcomes(lists: PairOutcome[][]): PairOutcome[] {
  const byKey = new Map<string, PairOutcome>();
  for (const list of lists) {
    for (const p of list) {
      const [a, b, winsA, winsB] =
        p.a < p.b ? [p.a, p.b, p.winsA, p.winsB] : [p.b, p.a, p.winsB, p.winsA];
      const key = `${a}|${b}`;
      let merged = byKey.get(key);
      if (!merged) {
        merged = { a, b, winsA: 0, winsB: 0, draws: 0 };
        byKey.set(key, merged);
      }
      merged.winsA += winsA;
      merged.winsB += winsB;
      merged.draws += p.draws;
    }
  }
  return [...byKey.values()];
}

/**
 * Fit ratings for every player appearing in `pairs`. Players with no pair
 * games simply don't appear in the result (callers display "—").
 */
export function fitBradleyTerry(
  pairs: PairOutcome[],
  opts: { priorGames?: number; maxIter?: number; tol?: number } = {}
): Map<string, number> {
  const priorGames = opts.priorGames ?? 1;
  const maxIter = opts.maxIter ?? 1000;
  const tol = opts.tol ?? 1e-9;

  const players: string[] = [];
  const index = new Map<string, number>();
  const idx = (id: string) => {
    let i = index.get(id);
    if (i === undefined) {
      i = players.length;
      players.push(id);
      index.set(id, i);
    }
    return i;
  };

  // Per-player effective wins and per-pair game counts.
  type Edge = { other: number; games: number };
  const wins: number[] = [];
  const edges: Edge[][] = [];
  const ensure = (i: number) => {
    while (wins.length <= i) {
      wins.push(0);
      edges.push([]);
    }
  };
  for (const p of pairs) {
    const ia = idx(p.a);
    const ib = idx(p.b);
    ensure(Math.max(ia, ib));
    const games = p.winsA + p.winsB + p.draws;
    if (games === 0) continue;
    wins[ia] += p.winsA + p.draws / 2;
    wins[ib] += p.winsB + p.draws / 2;
    edges[ia].push({ other: ib, games });
    edges[ib].push({ other: ia, games });
  }

  const n = players.length;
  if (n === 0) return new Map();

  let p = new Array<number>(n).fill(1);
  for (let iter = 0; iter < maxIter; iter++) {
    const next = new Array<number>(n);
    let maxDelta = 0;
    for (let i = 0; i < n; i++) {
      let denom = priorGames / (p[i] + 1); // virtual anchor at strength 1
      for (const e of edges[i]) {
        denom += e.games / (p[i] + p[e.other]);
      }
      next[i] = (wins[i] + priorGames / 2) / denom;
      maxDelta = Math.max(maxDelta, Math.abs(Math.log(next[i] / p[i])));
    }
    p = next;
    if (maxDelta < tol) break;
  }

  // Normalize mean(log p) to 0 so the view's mean rating is RATING_ANCHOR.
  const meanLog = p.reduce((s, v) => s + Math.log(v), 0) / n;
  const ratings = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    ratings.set(
      players[i],
      RATING_ANCHOR + (RATING_SCALE * (Math.log(p[i]) - meanLog)) / Math.LN10
    );
  }
  return ratings;
}
