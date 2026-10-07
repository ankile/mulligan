import { internalMutation } from "./_generated/server";

// Deliberately internal: wiping the arena must not be reachable from the
// public API. Run via `npx convex run seed:clearAll` with deploy credentials.
export const clearAll = internalMutation({
  args: {},
  handler: async (ctx) => {
    const tables = [
      "roundResults",
      "evalSessions",
      "policies",
    ] as const;

    let totalDeleted = 0;
    for (const table of tables) {
      const docs = await ctx.db.query(table).collect();
      for (const doc of docs) {
        await ctx.db.delete(doc._id);
      }
      totalDeleted += docs.length;
    }

    return `Cleared ${totalDeleted} documents across ${tables.length} tables`;
  },
});
