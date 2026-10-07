import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { convexTest, type TestConvex } from "convex-test";
import { api } from "../convex/_generated/api";
import schema from "../convex/schema";

// Phased eval contract: addRounds dedups per (round, policy), so a
// later phase can append the un-dropped arms' rollouts to rounds the session
// already holds, while a duplicate (round, policy) result is still rejected.
const modules = {
  "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
  "../convex/evalSessions.ts": () => import("../convex/evalSessions"),
  "../convex/access.ts": () => import("../convex/access"),
  "../convex/statuses.ts": () => import("../convex/statuses"),
  "../convex/statusShared.ts": () => import("../convex/statusShared"),
};
const SERVICE_TOKEN = "add-rounds-test-only-bridge-token";
const service = { serviceToken: SERVICE_TOKEN };
const ENV = "routing_d2";
const policy = (id: string) => ({ name: id, model_id: `test://${id}`, environment: ENV });
const result = (id: string, episode: number, success = true) => ({
  model_id: `test://${id}`,
  success,
  episode_index: BigInt(episode),
  num_frames: BigInt(100),
});

let t: TestConvex<typeof schema>;
let oldServiceToken: string | undefined;

beforeEach(() => {
  oldServiceToken = process.env.ARENA_SERVICE_TOKEN;
  process.env.ARENA_SERVICE_TOKEN = SERVICE_TOKEN;
  t = convexTest(schema, modules);
});
afterEach(() => {
  if (oldServiceToken === undefined) delete process.env.ARENA_SERVICE_TOKEN;
  else process.env.ARENA_SERVICE_TOKEN = oldServiceToken;
});

async function roundResults(sessionId: string) {
  return await t.run(async (ctx) => {
    const rows = await ctx.db
      .query("roundResults")
      .withIndex("by_session", (q) => q.eq("session_id", sessionId as never))
      .collect();
    return rows.map((r) => [Number(r.round_index), String(r.policy_id)] as const);
  });
}

describe("evalSessions.addRounds phased append", () => {
  test("appends new arms to existing rounds and adds new rounds; rejects duplicate pairs", async () => {
    // Phase 1: rounds 0-1 with arms A, B (C retired).
    const sessionId = await t.mutation(api.evalSessions.submit, {
      ...service,
      dataset_repo: "test/phased-append",
      session_mode: "manual",
      policies: [policy("A"), policy("B")],
      rounds: [
        { round_index: BigInt(0), results: [result("A", 0), result("B", 1, false)] },
        { round_index: BigInt(1), results: [result("A", 2), result("B", 3)] },
      ],
    });
    expect((await roundResults(sessionId)).length).toBe(4);

    // Phase 2: arm C catches up on rounds 0-1 AND round 2 runs all three arms.
    await t.mutation(api.evalSessions.addRounds, {
      ...service,
      id: sessionId,
      policies: [policy("A"), policy("B"), policy("C")],
      rounds: [
        { round_index: BigInt(0), results: [result("C", 4)] },
        { round_index: BigInt(1), results: [result("C", 5, false)] },
        { round_index: BigInt(2), results: [result("A", 6), result("B", 7), result("C", 8)] },
      ],
    });
    const rows = await roundResults(sessionId);
    expect(rows.length).toBe(9);
    const perRound = new Map<number, number>();
    for (const [idx] of rows) perRound.set(idx, (perRound.get(idx) ?? 0) + 1);
    expect([...perRound.entries()].sort()).toEqual([[0, 3], [1, 3], [2, 3]]);
    const session = await t.run(async (ctx) => await ctx.db.get(sessionId as never));
    expect(Number((session as { num_rounds: bigint }).num_rounds)).toBe(3);
    expect((session as { policy_ids: unknown[] }).policy_ids.length).toBe(3);

    // A genuine duplicate (round 0, arm C again) is rejected and nothing is written.
    await expect(
      t.mutation(api.evalSessions.addRounds, {
        ...service,
        id: sessionId,
        policies: [policy("C")],
        rounds: [{ round_index: BigInt(0), results: [result("C", 9)] }],
      }),
    ).rejects.toThrow(/already exists/);
    expect((await roundResults(sessionId)).length).toBe(9);
  });

  test("still rejects a duplicate round index inside one submission", async () => {
    const sessionId = await t.mutation(api.evalSessions.submit, {
      ...service,
      dataset_repo: "test/phased-append-2",
      session_mode: "manual",
      policies: [policy("A")],
      rounds: [{ round_index: BigInt(0), results: [result("A", 0)] }],
    });
    await expect(
      t.mutation(api.evalSessions.addRounds, {
        ...service,
        id: sessionId,
        policies: [policy("B")],
        rounds: [
          { round_index: BigInt(1), results: [result("B", 1)] },
          { round_index: BigInt(1), results: [result("B", 2)] },
        ],
      }),
    ).rejects.toThrow(/Duplicate round_index/);
  });
});
