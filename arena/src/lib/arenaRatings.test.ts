import { expect, test } from "bun:test";
import { computeArenaStats, type SessionOutcome } from "./arenaRatings";

const grid: SessionOutcome = {
  session_id: "grid-a", creation_time: 0, session_mode: "fixed_grid",
  task: "square", rating_group: "square/grid-a", effective_status: "mainline",
  pairs: [{a: "a", b: "b", winsA: 2, winsB: 1, draws: 2}],
  perPolicy: [
    {policy_id: "a", rollouts: 5, successes: 3, successFramesSum: 301, successFramesCount: 3},
    {policy_id: "b", rollouts: 5, successes: 2, successFramesSum: 240, successFramesCount: 2},
  ],
};
test("grid comparisons count draws and average successful episodes only", () => {
  const stats = computeArenaStats([grid], true);
  expect(stats.wdl.get("a")).toEqual({wins: 2, losses: 1, draws: 2});
  expect(stats.success.get("a")).toEqual({rollouts: 5, successes: 3, avgSuccessSteps: 100});
  expect(stats.ratings.get("a")!).toBeGreaterThan(stats.ratings.get("b")!);
});
test("a separate grid cannot change existing ratings or add cross-grid games", () => {
  const other = {...grid, session_id: "grid-b", rating_group: "square/grid-b",
    pairs: [{a: "c", b: "d", winsA: 99, winsB: 0, draws: 1}], perPolicy: []};
  const before = computeArenaStats([grid], true);
  const after = computeArenaStats([grid, other], true);
  expect(after.ratings.get("a")).toBe(before.ratings.get("a"));
  expect(after.wdl.get("a")).toEqual(before.wdl.get("a"));
});
