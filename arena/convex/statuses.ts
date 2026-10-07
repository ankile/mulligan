import { query, mutation } from "./_generated/server";
import type { QueryCtx } from "./_generated/server";
import type { Doc } from "./_generated/dataModel";
import { v } from "convex/values";
import { requireEditorOrService } from "./access";
import { statusValidator } from "./statusShared";

export async function loadTaskStatusMap(
  ctx: QueryCtx
): Promise<Map<string, Doc<"taskStatuses">>> {
  const rows = await ctx.db.query("taskStatuses").collect();
  return new Map(rows.map((r) => [r.task, r]));
}

export const listTaskStatuses = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("taskStatuses").collect();
    return rows.sort((a, b) => a.task.localeCompare(b.task));
  },
});

/**
 * Declarative upsert of a task's status row: omitted reason/superseded_by
 * CLEAR those fields (callers editing one field pass the others through).
 */
export const setTaskStatus = mutation({
  args: {
    task: v.string(),
    status: statusValidator,
    reason: v.optional(v.string()),
    superseded_by: v.optional(v.string()),
    serviceToken: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const principal = await requireEditorOrService(ctx, args.serviceToken);
    const fields = {
      status: args.status,
      reason: args.reason,
      superseded_by: args.superseded_by,
      updated_at: Date.now(),
      updated_by: principal,
    };
    const existing = await ctx.db
      .query("taskStatuses")
      .withIndex("by_task", (q) => q.eq("task", args.task))
      .unique();
    if (existing) {
      await ctx.db.patch(existing._id, fields);
      return existing._id;
    }
    return await ctx.db.insert("taskStatuses", { task: args.task, ...fields });
  },
});

