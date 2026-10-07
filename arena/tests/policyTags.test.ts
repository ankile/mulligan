import { afterEach, beforeEach, expect, test } from "bun:test";
import { convexTest } from "convex-test";
import { api } from "../convex/_generated/api";
import schema from "../convex/schema";
import { matchesPolicyTags, normalizePolicyTags, policyTagOptions } from "../convex/policyTags";

const modules = {
  "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
  "../convex/policies.ts": () => import("../convex/policies"),
};
let previous: string | undefined;
beforeEach(() => { previous = process.env.ARENA_EDITOR_SUBS; process.env.ARENA_EDITOR_SUBS = "tag-editor"; });
afterEach(() => {
  if (previous === undefined) delete process.env.ARENA_EDITOR_SUBS;
  else process.env.ARENA_EDITOR_SUBS = previous;
});

test("edit, query, re-register and clear policy tags without changing identity or status", async () => {
  const t = convexTest(schema, modules);
  const user = await t.run(async (ctx) => {
    const userId = await ctx.db.insert("users", { username: "editor" });
    await ctx.db.insert("authAccounts", { userId, provider: "huggingface", providerAccountId: "tag-editor" });
    return userId;
  });
  const editor = t.withIdentity({ subject: user });
  const identity = { model_id: "hf://test/policy", name: "Policy", environment: "marker_d2" };
  const id = await editor.mutation(api.policies.register, identity);
  await editor.mutation(api.policies.setStatus, { model_id: identity.model_id, status: "ablation" });
  const tags = { model_id: identity.model_id, round: 0, method: " HG-DAgger mulligan ", tags: [" relative action ", "relative action", ""] };
  await expect(t.mutation(api.policies.setTags, tags)).rejects.toThrow("Not signed in");
  await expect(t.withIdentity({ subject: "reader" }).mutation(api.policies.setTags, tags)).rejects.toThrow();
  await editor.mutation(api.policies.setTags, tags);
  await editor.mutation(api.policies.register, { ...identity, name: "Renamed" });
  expect(await t.query(api.policies.get, { id })).toMatchObject({ round: 0, method: "HG-DAgger mulligan", tags: ["relative action"], status: "ablation", name: "Renamed" });
  expect((await t.query(api.policies.leaderboard, {}))[0]).toMatchObject({ round: 0, method: "HG-DAgger mulligan" });
  expect(await t.query(api.policies.tagOptions, {})).toEqual({ rounds: [0], methods: ["HG-DAgger mulligan"], tags: ["relative action"] });
  await expect(editor.mutation(api.policies.setTags, { ...tags, round: -1 })).rejects.toThrow("nonnegative integer");
  await expect(editor.mutation(api.policies.setTags, { ...tags, model_id: "missing" })).rejects.toThrow("Policy not found");
  await editor.mutation(api.policies.setTags, { model_id: identity.model_id, round: null, method: null, tags: [] });
  const cleared = await t.query(api.policies.get, { id });
  expect(cleared?.round).toBeUndefined();
  expect(cleared?.method).toBeUndefined();
  expect(cleared?.tags).toEqual([]);
  expect(cleared?.status).toBe("ablation");
});

test("invalid round values and excessive labels fail loudly", () => {
  for (const round of [-1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    expect(() => normalizePolicyTags({ round, method: null, tags: [] })).toThrow();
  }
  expect(() => normalizePolicyTags({ round: 1, method: "x".repeat(101), tags: [] })).toThrow();
  expect(() => normalizePolicyTags({ round: null, method: null, tags: ["x".repeat(81)] })).toThrow();
});

test("filters compose, include Round 0, and distinguish unassigned metadata", () => {
  const policy = { round: 0, method: "HG-DAgger baseline", tags: ["relative action"] };
  expect(matchesPolicyTags(policy, "0", policy.method, "relative action")).toBe(true);
  expect(matchesPolicyTags(policy, "1", "", "")).toBe(false);
  expect(matchesPolicyTags(policy, "0", "HiL-IDQL mulligan", "")).toBe(false);
  expect(matchesPolicyTags(policy, "", "", "unknown")).toBe(false);
  expect(matchesPolicyTags({}, "untagged", "untagged", "")).toBe(true);
  expect(matchesPolicyTags(policy, "untagged", "", "")).toBe(false);
  expect(policyTagOptions([policy, { round: 10 }, { round: 2 }, {}]).rounds).toEqual([0, 2, 10]);
});
