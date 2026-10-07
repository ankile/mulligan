/**
 * Port of lerobot.datasets.compute_stats.get_feature_stats for the 1-D
 * scalar features the outcome editor rewrites (success/reward/done/is_valid).
 *
 * Semantics mirror the Python implementation for a SINGLE update batch (the
 * refresh path always feeds one episode's — or the whole dataset's — values in
 * one call): min/max/mean/std from running moments, quantiles from a 5000-bin
 * histogram over linspace(min-1e-10, max+1e-10) edges with linear in-bin
 * interpolation. With < 2 samples, quantiles collapse to the mean
 * (_compute_basic_stats).
 *
 * Arithmetic is float64 throughout. The Python path computes float32-sourced
 * features (reward) in float32, so last-ulp differences (~1e-8 relative) are
 * expected and acceptable — this port targets value correctness, not bit
 * parity.
 */

export const STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"] as const;
export type StatKey = (typeof STAT_KEYS)[number];

const QUANTILES: Array<[StatKey, number]> = [
  ["q01", 0.01],
  ["q10", 0.1],
  ["q50", 0.5],
  ["q90", 0.9],
  ["q99", 0.99],
];
const NUM_BINS = 5000;

export type FeatureStats = Record<StatKey, number>;

/** np.linspace(start, stop, num): step-based with exact endpoint. */
function linspace(start: number, stop: number, num: number): Float64Array {
  const out = new Float64Array(num);
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) out[i] = start + i * step;
  out[num - 1] = stop;
  return out;
}

/** np.searchsorted(a, v, side="right"): first index where a[i] > v. */
function searchSortedRight(a: ArrayLike<number>, v: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** np.searchsorted(a, v, side="left"): first index where a[i] >= v. */
function searchSortedLeft(a: ArrayLike<number>, v: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** np.histogram(values, bins=edges) for monotonically increasing edges. */
function histogram(values: ArrayLike<number>, edges: Float64Array): Float64Array {
  const hist = new Float64Array(edges.length - 1);
  const lastEdge = edges[edges.length - 1];
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    if (x < edges[0] || x > lastEdge) continue;
    // side="right" bin placement; values equal to the last edge fall in the
    // final bin (np.histogram's closed right boundary on the last bin).
    let idx = searchSortedRight(edges, x) - 1;
    if (idx === hist.length) idx = hist.length - 1;
    hist[idx] += 1;
  }
  return hist;
}

function computeSingleQuantile(cumsum: Float64Array, edges: Float64Array, targetCount: number): number {
  const idx = searchSortedLeft(cumsum, targetCount);
  if (idx === 0) return edges[0];
  if (idx >= cumsum.length) return edges[edges.length - 1];
  const countBefore = cumsum[idx - 1];
  const countInBin = cumsum[idx] - countBefore;
  if (countInBin === 0) return edges[idx];
  const fraction = (targetCount - countBefore) / countInBin;
  return edges[idx] + fraction * (edges[idx + 1] - edges[idx]);
}

/** get_feature_stats(values, axis=0, keepdims=True) for a 1-D feature. */
export function getFeatureStats1D(values: ArrayLike<number>): FeatureStats {
  const n = values.length;
  if (n === 0) throw new Error("Cannot compute stats of an empty feature array");

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const x = values[i];
    if (!Number.isFinite(x)) throw new Error(`Non-finite feature value ${x} at row ${i}`);
    if (x < min) min = x;
    if (x > max) max = x;
    sum += x;
    sumSq += x * x;
  }
  const mean = sum / n;
  const meanOfSquares = sumSq / n;

  if (n < 2) {
    // _compute_basic_stats: quantiles collapse to the mean; std of one value is 0.
    return {
      min,
      max,
      mean,
      std: 0,
      count: n,
      q01: mean,
      q10: mean,
      q50: mean,
      q90: mean,
      q99: mean,
    };
  }

  const variance = meanOfSquares - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  const edges = linspace(min - 1e-10, max + 1e-10, NUM_BINS + 1);
  const hist = histogram(values, edges);
  const cumsum = new Float64Array(NUM_BINS);
  let acc = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    acc += hist[i];
    cumsum[i] = acc;
  }

  const stats: FeatureStats = { min, max, mean, std, count: n } as FeatureStats;
  for (const [key, q] of QUANTILES) {
    stats[key] = computeSingleQuantile(cumsum, edges, q * n);
  }
  return stats;
}
