import { query, mutation, internalMutation } from "./_generated/server";
import { v } from "convex/values";
import { requireEditorOrService } from "./access";

/**
 * Registry of known eval operators — the humans who physically run eval
 * sessions on the robot. Entries are HF USERNAMES so an operator can be
 * recorded on sessions before ever signing in; once they authenticate via HF
 * OAuth, `users.username` connects the operator string to their account.
 */

export const list = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("operators").collect();
    return rows.sort((a, b) => a.hf_username.localeCompare(b.hf_username));
  },
});

export const add = mutation({
  args: { serviceToken: v.optional(v.string()), hf_username: v.string() },
  handler: async (ctx, args) => {
    const principal = await requireEditorOrService(ctx, args.serviceToken);
    const username = args.hf_username.trim();
    if (!username) throw new Error("hf_username must be non-empty");
    const existing = await ctx.db
      .query("operators")
      .withIndex("by_username", (q) => q.eq("hf_username", username))
      .unique();
    if (existing) return existing._id;
    return await ctx.db.insert("operators", {
      hf_username: username,
      added_at: Date.now(),
      added_by: principal,
    });
  },
});

/** Idempotent seed (run via `npx convex run operators:seed '{"usernames": [...]}'`). */
export const seed = internalMutation({
  args: { usernames: v.array(v.string()) },
  handler: async (ctx, args) => {
    let inserted = 0;
    for (const raw of args.usernames) {
      const username = raw.trim();
      if (!username) throw new Error("empty username in seed list");
      const existing = await ctx.db
        .query("operators")
        .withIndex("by_username", (q) => q.eq("hf_username", username))
        .unique();
      if (!existing) {
        await ctx.db.insert("operators", {
          hf_username: username,
          added_at: Date.now(),
          added_by: "seed",
        });
        inserted += 1;
      }
    }
    return { inserted };
  },
});
