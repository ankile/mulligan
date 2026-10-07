import { afterEach, beforeEach, expect, test } from "bun:test";
import { convexTest } from "convex-test";
import { api } from "../convex/_generated/api";
import schema from "../convex/schema";

const modules = {
  "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
  "../convex/reviews.ts": () => import("../convex/reviews"),
};
let oldAllowlist: string | undefined;
beforeEach(() => {
  oldAllowlist = process.env.ARENA_EDITOR_SUBS;
  process.env.ARENA_EDITOR_SUBS = "notes-editor";
});
afterEach(() => {
  if (oldAllowlist === undefined) delete process.env.ARENA_EDITOR_SUBS;
  else process.env.ARENA_EDITOR_SUBS = oldAllowlist;
});

test("notes persist independently per dataset and episode, can be edited and cleared", async () => {
  const t = convexTest(schema, modules);
  const user = await t.run(async (ctx) => {
    const userId = await ctx.db.insert("users", { username: "editor" });
    await ctx.db.insert("authAccounts", {
      userId, provider: "huggingface", providerAccountId: "notes-editor",
    });
    return userId;
  });
  const editor = t.withIdentity({ subject: user });
  const episode = { dataset_repo: "test/notes", episode_index: BigInt(0) };
  expect(await t.query(api.reviews.episodeNotes, episode)).toBeNull();
  await editor.mutation(api.reviews.saveNotes, { ...episode, notes: "Revisit grasp\nCheck wrist view." });
  expect((await t.query(api.reviews.episodeNotes, episode))?.notes).toBe("Revisit grasp\nCheck wrist view.");
  expect(await t.query(api.reviews.episodeNotes, { ...episode, episode_index: BigInt(1) })).toBeNull();
  expect(await t.query(api.reviews.episodeNotes, { ...episode, dataset_repo: "test/other" })).toBeNull();
  await editor.mutation(api.reviews.saveNotes, { ...episode, notes: "Updated" });
  expect((await t.query(api.reviews.episodeNotes, episode))?.notes).toBe("Updated");
  await editor.mutation(api.reviews.saveNotes, { ...episode, notes: "" });
  expect((await t.query(api.reviews.episodeNotes, episode))?.notes).toBe("");
  expect(await t.query(api.reviews.latestForRepo, { dataset_repo: episode.dataset_repo })).toEqual({
    episodes: [], num_confirmed: 0, num_skipped: 0,
  });
  expect(await t.run(async (ctx) => (await ctx.db.query("applyJobs").collect()).length)).toBe(0);
  expect(await t.run(async (ctx) => (await ctx.db.query("episodeNotes").collect()).length)).toBe(1);
});

test("anonymous and non-editor users cannot write notes", async () => {
  const t = convexTest(schema, modules);
  const args = { dataset_repo: "test/notes", episode_index: BigInt(0), notes: "No" };
  await expect(t.mutation(api.reviews.saveNotes, args)).rejects.toThrow("Not signed in");
  const user = await t.run(async (ctx) => {
    const userId = await ctx.db.insert("users", { username: "reader" });
    await ctx.db.insert("authAccounts", { userId, provider: "huggingface", providerAccountId: "reader" });
    return userId;
  });
  await expect(t.withIdentity({ subject: user }).mutation(api.reviews.saveNotes, args)).rejects.toThrow("not an allowlisted editor");
});

test("suggestions rank distinct same-task notes by recency and episode usage", async () => {
  const t = convexTest(schema, modules);
  await t.run(async (ctx) => {
    for (const [repo_id, task] of [["test/current", "routing"], ["test/prior", "routing"], ["test/other", "square"]]) {
      await ctx.db.insert("datasets", { repo_id, task, name: repo_id, source_type: "eval", environment: "real" });
    }
    for (const [dataset_repo, episode_index, notes, updated_at] of [
      ["test/current", 0n, "Current episode only", 100],
      ["test/current", 1n, "Common", 10],
      ["test/prior", 0n, " Common ", 20],
      ["test/prior", 1n, "Recent", 30],
      ["test/prior", 2n, "   ", 40],
      ["test/other", 0n, "Wrong task", 200],
    ] as const) {
      await ctx.db.insert("episodeNotes", { dataset_repo, episode_index, notes, updated_at, updated_by: "editor" });
    }
  });
  const result = await t.query(api.reviews.noteSuggestions, { dataset_repo: "test/current", episode_index: 0n });
  expect(result.recent).toEqual([
    { notes: "Recent", count: 1, lastUsed: 30 },
    { notes: "Common", count: 2, lastUsed: 20 },
  ]);
  expect(result.mostUsed.map((row) => row.notes)).toEqual(["Common", "Recent"]);
  expect(await t.query(api.reviews.noteSuggestions, { dataset_repo: "test/unregistered", episode_index: 0n }))
    .toEqual({ recent: [], mostUsed: [] });
});

test("suggestions cap each ordering at six notes", async () => {
  const t = convexTest(schema, modules);
  await t.run(async (ctx) => {
    await ctx.db.insert("datasets", { repo_id: "test/many", task: "routing", name: "Many", source_type: "eval", environment: "real" });
    for (let index = 1; index <= 8; index++) {
      await ctx.db.insert("episodeNotes", { dataset_repo: "test/many", episode_index: BigInt(index), notes: `Note ${index}`, updated_at: index, updated_by: "editor" });
    }
  });
  const result = await t.query(api.reviews.noteSuggestions, { dataset_repo: "test/many", episode_index: 0n });
  expect(result.recent.map((row) => row.notes)).toEqual(["Note 8", "Note 7", "Note 6", "Note 5", "Note 4", "Note 3"]);
  expect(result.mostUsed).toEqual(result.recent);
});
