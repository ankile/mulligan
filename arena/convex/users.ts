import { getAuthUserId } from "@convex-dev/auth/server";
import { query } from "./_generated/server";
import { viewerIsEditor } from "./access";

/** Current signed-in user (or null) plus whether they may edit. */
export const viewer = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    const user = await ctx.db.get(userId);
    if (!user) return null;
    return {
      userId,
      name: user.name ?? user.username ?? "unknown",
      username: user.username ?? null,
      image: user.image ?? null,
      isEditor: await viewerIsEditor(ctx),
    };
  },
});
