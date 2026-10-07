import { describe, expect, test } from "bun:test";
import {
  RATING_ANCHOR,
  fitBradleyTerry,
  mergePairOutcomes,
  pairOutcomesFromRounds,
  type PairOutcome,
} from "../../convex/bradleyTerry";

describe("pairOutcomesFromRounds", () => {
  test("each round yields a win, loss or draw per policy pair", () => {
    const rounds = [
      [
        { id: "A", success: true },
        { id: "B", success: false },
      ],
      [
        { id: "A", success: true },
        { id: "B", success: true },
      ],
      [
        { id: "A", success: false },
        { id: "B", success: false },
      ],
      [
        { id: "B", success: true },
        { id: "A", success: false },
      ],
    ];
    const pairs = pairOutcomesFromRounds(rounds);
    expect(pairs).toEqual([{ a: "A", b: "B", winsA: 1, winsB: 1, draws: 2 }]);
  });

  test("three-arm rounds produce all pairwise outcomes", () => {
    const pairs = pairOutcomesFromRounds([
      [
        { id: "A", success: true },
        { id: "B", success: false },
        { id: "C", success: true },
      ],
    ]);
    expect(pairs).toHaveLength(3);
    const ab = pairs.find((p) => p.a === "A" && p.b === "B")!;
    const ac = pairs.find((p) => p.a === "A" && p.b === "C")!;
    const bc = pairs.find((p) => p.a === "B" && p.b === "C")!;
    expect(ab).toMatchObject({ winsA: 1, winsB: 0, draws: 0 });
    expect(ac).toMatchObject({ winsA: 0, winsB: 0, draws: 1 });
    expect(bc).toMatchObject({ winsA: 0, winsB: 1, draws: 0 });
  });
});

describe("mergePairOutcomes", () => {
  test("canonicalizes orientation and sums counts", () => {
    const merged = mergePairOutcomes([
      [{ a: "A", b: "B", winsA: 2, winsB: 1, draws: 1 }],
      [{ a: "B", b: "A", winsA: 3, winsB: 1, draws: 0 }],
    ]);
    expect(merged).toEqual([{ a: "A", b: "B", winsA: 3, winsB: 4, draws: 1 }]);
  });
});

describe("fitBradleyTerry", () => {
  test("dominant player rates higher; symmetric data rates equal", () => {
    const r = fitBradleyTerry([
      { a: "A", b: "B", winsA: 8, winsB: 2, draws: 0 },
    ]);
    expect(r.get("A")!).toBeGreaterThan(r.get("B")!);

    const even = fitBradleyTerry([
      { a: "A", b: "B", winsA: 5, winsB: 5, draws: 3 },
    ]);
    expect(even.get("A")!).toBeCloseTo(even.get("B")!, 6);
    expect(even.get("A")!).toBeCloseTo(RATING_ANCHOR, 6);
  });

  test("order-independent: shuffled and split inputs give identical ratings", () => {
    const pairs: PairOutcome[] = [
      { a: "A", b: "B", winsA: 7, winsB: 3, draws: 2 },
      { a: "B", b: "C", winsA: 6, winsB: 4, draws: 0 },
      { a: "A", b: "C", winsA: 9, winsB: 1, draws: 1 },
    ];
    const forward = fitBradleyTerry(pairs);
    const reversed = fitBradleyTerry([...pairs].reverse());
    // Same data split across "sessions" and merged.
    const split = fitBradleyTerry(
      mergePairOutcomes([
        [{ a: "B", b: "A", winsA: 1, winsB: 4, draws: 2 }],
        [{ a: "A", b: "B", winsA: 3, winsB: 2, draws: 0 }],
        [{ a: "B", b: "C", winsA: 6, winsB: 4, draws: 0 }],
        [{ a: "C", b: "A", winsA: 1, winsB: 9, draws: 1 }],
      ])
    );
    for (const id of ["A", "B", "C"]) {
      expect(reversed.get(id)!).toBeCloseTo(forward.get(id)!, 6);
      expect(split.get(id)!).toBeCloseTo(forward.get(id)!, 6);
    }
  });

  test("transitive dominance orders A > B > C", () => {
    const r = fitBradleyTerry([
      { a: "A", b: "B", winsA: 7, winsB: 3, draws: 0 },
      { a: "B", b: "C", winsA: 7, winsB: 3, draws: 0 },
    ]);
    expect(r.get("A")!).toBeGreaterThan(r.get("B")!);
    expect(r.get("B")!).toBeGreaterThan(r.get("C")!);
  });

  test("all-win player stays finite thanks to the prior", () => {
    const r = fitBradleyTerry([
      { a: "A", b: "B", winsA: 10, winsB: 0, draws: 0 },
    ]);
    expect(Number.isFinite(r.get("A")!)).toBe(true);
    expect(r.get("A")!).toBeGreaterThan(r.get("B")!);
    // More evidence of dominance widens the gap.
    const r2 = fitBradleyTerry([
      { a: "A", b: "B", winsA: 100, winsB: 0, draws: 0 },
    ]);
    expect(r2.get("A")! - r2.get("B")!).toBeGreaterThan(
      r.get("A")! - r.get("B")!
    );
  });

  test("removing a subset of sessions changes ratings deterministically (view-dependence)", () => {
    const all = fitBradleyTerry(
      mergePairOutcomes([
        [{ a: "A", b: "B", winsA: 5, winsB: 5, draws: 0 }],
        [{ a: "A", b: "B", winsA: 10, winsB: 0, draws: 0 }],
      ])
    );
    const subset = fitBradleyTerry([
      { a: "A", b: "B", winsA: 5, winsB: 5, draws: 0 },
    ]);
    expect(all.get("A")!).toBeGreaterThan(all.get("B")!);
    expect(subset.get("A")!).toBeCloseTo(subset.get("B")!, 6);
  });

  test("players without games are absent; empty input yields empty map", () => {
    expect(fitBradleyTerry([]).size).toBe(0);
    const r = fitBradleyTerry([{ a: "A", b: "B", winsA: 1, winsB: 0, draws: 0 }]);
    expect(r.has("C")).toBe(false);
  });
});
