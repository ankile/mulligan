// Paired per-round comparison statistics between two policies of one eval
// session. Mirrors mulligan/real/lifecycle/stats.py (exact_mcnemar_pvalue,
// bootstrap_paired_delta, signflip_permutation_pvalue) in method; the
// Monte-Carlo numbers are not bit-identical to the Python ones because the
// PRNG differs, but both use a pinned seed so a page re-render is stable.
// Pure functions, no React — unit-tested in pairedStats.test.ts.

import { episodeScore } from "./outcomeScore";

export interface PairedComparison {
  /** Paired rounds (both arms present, both scored for the graded metric). */
  n: number;
  meanA: number;
  meanB: number;
  /** mean(A) - mean(B) over paired rounds. */
  delta: number;
  /** Paired bootstrap 95% CI on `delta` (resamples rounds with replacement). */
  ciLo: number;
  ciHi: number;
  /** A's record: rounds where A > B / A == B / A < B. */
  wins: number;
  draws: number;
  losses: number;
  /**
   * Exact two-sided sign test on the discordant rounds (draws ignored). For
   * binary success this IS the exact McNemar test.
   */
  signPValue: number;
  /**
   * Two-sided sign-flip permutation p-value for mean(delta) == 0. Uses the
   * magnitude of each round's difference, so a 2-point graded swing counts
   * more than a 1-point one; the sign test only counts direction.
   */
  signFlipPValue: number;
}

const N_RESAMPLES = 20_000;
const SEED = 0;

/** mulberry32: small deterministic PRNG, uniform in [0, 1). */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Exact two-sided sign test / McNemar p-value from the discordant counts. */
export function exactSignTestPValue(aOnly: number, bOnly: number): number {
  const n = aOnly + bOnly;
  if (n === 0) return 1;
  const k = Math.min(aOnly, bOnly);
  // Binomial(n, 1/2) lower tail up to k, via the iterative pmf recurrence.
  let pmf = Math.pow(0.5, n);
  let tail = pmf;
  for (let i = 1; i <= k; i++) {
    pmf = (pmf * (n - i + 1)) / i;
    tail += pmf;
  }
  return Math.min(1, 2 * tail);
}

/** numpy.quantile default (linear interpolation) on a sorted array. */
function quantileSorted(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function bootstrapPairedDeltaCI(
  deltas: number[],
  nBoot = N_RESAMPLES,
  seed = SEED,
): [number, number] {
  if (deltas.length === 0) throw new Error("bootstrapPairedDeltaCI: empty deltas");
  const rand = seededRandom(seed);
  const n = deltas.length;
  const means = new Array<number>(nBoot);
  for (let b = 0; b < nBoot; b++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += deltas[Math.floor(rand() * n)];
    means[b] = sum / n;
  }
  means.sort((x, y) => x - y);
  return [quantileSorted(means, 0.025), quantileSorted(means, 0.975)];
}

export function signFlipPermutationPValue(
  deltas: number[],
  nPerm = N_RESAMPLES,
  seed = SEED,
): number {
  if (deltas.length === 0) throw new Error("signFlipPermutationPValue: empty deltas");
  if (deltas.every((d) => d === 0)) return 1;
  const n = deltas.length;
  const observed = Math.abs(deltas.reduce((s, d) => s + d, 0) / n);
  const rand = seededRandom(seed);
  let extreme = 0;
  for (let p = 0; p < nPerm; p++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += rand() < 0.5 ? -deltas[i] : deltas[i];
    // 1e-12: |obs| itself is one of the permutations up to float roundoff.
    if (Math.abs(sum / n) >= observed - 1e-12) extreme += 1;
  }
  return (1 + extreme) / (1 + nPerm);
}

/** Full paired comparison of two aligned outcome vectors (A[i] vs B[i]). */
export function pairedComparison(a: number[], b: number[]): PairedComparison {
  if (a.length !== b.length) {
    throw new Error(`pairedComparison: length mismatch ${a.length} vs ${b.length}`);
  }
  if (a.length === 0) throw new Error("pairedComparison: no paired rounds");
  const n = a.length;
  const deltas = a.map((x, i) => x - b[i]);
  let wins = 0;
  let draws = 0;
  let losses = 0;
  for (const d of deltas) {
    if (d > 0) wins += 1;
    else if (d < 0) losses += 1;
    else draws += 1;
  }
  const meanA = a.reduce((s, x) => s + x, 0) / n;
  const meanB = b.reduce((s, x) => s + x, 0) / n;
  const [ciLo, ciHi] = bootstrapPairedDeltaCI(deltas);
  return {
    n,
    meanA,
    meanB,
    delta: meanA - meanB,
    ciLo,
    ciHi,
    wins,
    draws,
    losses,
    signPValue: exactSignTestPValue(wins, losses),
    signFlipPValue: signFlipPermutationPValue(deltas),
  };
}

export interface SessionRoundResult {
  policy_id: string;
  success: boolean;
  num_subtask_marks: number | null;
}

export interface PairedRow {
  policyA: string;
  policyB: string;
  metric: "success" | "score";
  /** Rounds where both arms ran but at least one lacks a mark count. */
  droppedUnscored: number;
  stats: PairedComparison | null;
}

/**
 * One row per unordered policy pair (in `policyIds` order) and metric: binary
 * success always; the graded score when the task has sub-goal marks, over
 * rounds where BOTH arms carry a mark count (same rule as the Python
 * pairwise summary's "jointly reviewed graded rounds").
 */
export function pairedRows(
  policyIds: string[],
  rounds: Array<{ results: SessionRoundResult[] }>,
  maxMarks: number,
): PairedRow[] {
  const rows: PairedRow[] = [];
  for (let i = 0; i < policyIds.length; i++) {
    for (let j = i + 1; j < policyIds.length; j++) {
      const idA = policyIds[i];
      const idB = policyIds[j];
      const successA: number[] = [];
      const successB: number[] = [];
      const scoreA: number[] = [];
      const scoreB: number[] = [];
      let droppedUnscored = 0;
      for (const round of rounds) {
        const ra = round.results.find((r) => r.policy_id === idA);
        const rb = round.results.find((r) => r.policy_id === idB);
        if (!ra || !rb) continue;
        successA.push(ra.success ? 1 : 0);
        successB.push(rb.success ? 1 : 0);
        if (ra.num_subtask_marks === null || rb.num_subtask_marks === null) {
          droppedUnscored += 1;
          continue;
        }
        scoreA.push(episodeScore(ra.success, ra.num_subtask_marks));
        scoreB.push(episodeScore(rb.success, rb.num_subtask_marks));
      }
      rows.push({
        policyA: idA,
        policyB: idB,
        metric: "success",
        droppedUnscored: 0,
        stats: successA.length > 0 ? pairedComparison(successA, successB) : null,
      });
      if (maxMarks > 0) {
        rows.push({
          policyA: idA,
          policyB: idB,
          metric: "score",
          droppedUnscored,
          stats: scoreA.length > 0 ? pairedComparison(scoreA, scoreB) : null,
        });
      }
    }
  }
  return rows;
}

export function formatPValue(p: number): string {
  if (p < 0.001) return "<0.001";
  return p.toFixed(3);
}
