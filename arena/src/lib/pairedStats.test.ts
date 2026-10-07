import { describe, expect, test } from "bun:test";

import {
  bootstrapPairedDeltaCI,
  exactSignTestPValue,
  formatPValue,
  pairedComparison,
  pairedRows,
  seededRandom,
  signFlipPermutationPValue,
} from "./pairedStats";

describe("exact sign test / McNemar", () => {
  test("no discordant pairs is p = 1", () => {
    expect(exactSignTestPValue(0, 0)).toBe(1);
  });

  test("matches scipy binomtest(k, n, 0.5) two-sided", () => {
    // binomtest(4, 12) = 0.3877, binomtest(0, 5) = 0.0625, binomtest(3, 3) = 1
    expect(exactSignTestPValue(8, 4)).toBeCloseTo(0.38770, 5);
    expect(exactSignTestPValue(5, 0)).toBeCloseTo(0.0625, 10);
    expect(exactSignTestPValue(3, 3)).toBe(1);
  });

  test("a 12W/4L graded record is p = 0.077", () => {
    expect(exactSignTestPValue(12, 4)).toBeCloseTo(0.0768, 4);
  });
});

describe("seeded PRNG", () => {
  test("is deterministic and in [0, 1)", () => {
    const a = seededRandom(7);
    const b = seededRandom(7);
    for (let i = 0; i < 100; i++) {
      const x = a();
      expect(x).toBe(b());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});

describe("sign-flip permutation", () => {
  test("all-zero deltas is p = 1", () => {
    expect(signFlipPermutationPValue([0, 0, 0])).toBe(1);
  });

  test("fully one-sided deltas are near the exact 2 / 2^n floor", () => {
    // 10 identical positive deltas: exact two-sided p = 2/1024 = 0.00195.
    const p = signFlipPermutationPValue(new Array(10).fill(1));
    expect(p).toBeGreaterThan(0.0005);
    expect(p).toBeLessThan(0.006);
  });

  test("matches exact enumeration of all sign flips on a small sample", () => {
    const deltas = [2, 1, 1, -1, 1, 2, -1, 1, 0, 2];
    const n = deltas.length;
    const observed = Math.abs(deltas.reduce((s, d) => s + d, 0) / n);
    let extreme = 0;
    for (let mask = 0; mask < 1 << n; mask++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += mask & (1 << i) ? -deltas[i] : deltas[i];
      if (Math.abs(sum / n) >= observed - 1e-12) extreme += 1;
    }
    const exact = extreme / (1 << n);
    expect(signFlipPermutationPValue(deltas)).toBeCloseTo(exact, 2);
  });
});

describe("bootstrap CI", () => {
  test("constant deltas give a degenerate interval", () => {
    expect(bootstrapPairedDeltaCI([2, 2, 2, 2])).toEqual([2, 2]);
  });

  test("brackets the point estimate and is seed-stable", () => {
    const deltas = [1, 0, 1, -1, 0, 1, 1, 0, -1, 1];
    const [lo, hi] = bootstrapPairedDeltaCI(deltas);
    expect(lo).toBeLessThan(0.3);
    expect(hi).toBeGreaterThan(0.3);
    expect(bootstrapPairedDeltaCI(deltas)).toEqual([lo, hi]);
  });
});

describe("pairedComparison", () => {
  test("record, means and delta", () => {
    const c = pairedComparison([1, 1, 0, 0, 1], [0, 1, 0, 1, 0]);
    expect(c.n).toBe(5);
    expect(c.wins).toBe(2);
    expect(c.draws).toBe(2);
    expect(c.losses).toBe(1);
    expect(c.meanA).toBeCloseTo(0.6);
    expect(c.meanB).toBeCloseTo(0.4);
    expect(c.delta).toBeCloseTo(0.2);
    expect(c.signPValue).toBe(1);
  });

  test("rejects unequal or empty inputs", () => {
    expect(() => pairedComparison([1], [1, 0])).toThrow("length mismatch");
    expect(() => pairedComparison([], [])).toThrow("no paired rounds");
  });
});

describe("pairedRows", () => {
  const rounds = [
    { results: [r("a", true, 1), r("b", false, 1), r("c", false, 0)] },
    { results: [r("a", false, 1), r("b", false, 0)] },
    { results: [r("a", true, null), r("b", true, 1), r("c", true, 1)] },
  ];

  function r(policy_id: string, success: boolean, marks: number | null) {
    return { policy_id, success, num_subtask_marks: marks };
  }

  test("binary task: one success row per unordered pair in policy order", () => {
    const rows = pairedRows(["a", "b", "c"], rounds, 0);
    expect(rows.map((x) => [x.policyA, x.policyB, x.metric])).toEqual([
      ["a", "b", "success"],
      ["a", "c", "success"],
      ["b", "c", "success"],
    ]);
    expect(rows[0].stats!.n).toBe(3);
    expect(rows[1].stats!.n).toBe(2); // c missing in round 2
  });

  test("graded task: score rows use jointly scored rounds only", () => {
    const rows = pairedRows(["a", "b"], rounds, 1);
    expect(rows.map((x) => x.metric)).toEqual(["success", "score"]);
    const score = rows[1];
    expect(score.stats!.n).toBe(2);
    expect(score.droppedUnscored).toBe(1);
    // Round 1: a=2 vs b=1 (win); round 2: a=1 vs b=0 (win).
    expect(score.stats!.wins).toBe(2);
    expect(score.stats!.delta).toBeCloseTo(1);
  });

  test("a pair with no shared rounds has null stats", () => {
    const rows = pairedRows(["a", "z"], rounds, 0);
    expect(rows[0].stats).toBeNull();
  });
});

describe("formatPValue", () => {
  test("three decimals with a floor", () => {
    expect(formatPValue(0.0768)).toBe("0.077");
    expect(formatPValue(0.0002)).toBe("<0.001");
    expect(formatPValue(1)).toBe("1.000");
  });
});
