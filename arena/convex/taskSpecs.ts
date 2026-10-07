import { query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { requireEditorOrService } from "./access";

/**
 * Task specs are EXPORTED DATA, not editable state: the Python task registry
 * (mulligan/real/lifecycle/tasks.py RealTaskSpec + the station camera config in
 * mulligan/real/robot/cameras.py) is the single source of truth, serialized to
 * this table by mulligan/tools/export_arena_task_specs.py. The UI consumes them
 * for station display crops and the subtask-mark legality hint; the apply
 * worker re-validates every record, so a stale row here can only mis-guide the
 * UI, never write a bad label.
 */

export const upsert = mutation({
  args: {
    serviceToken: v.optional(v.string()),
    task: v.string(),
    task_name: v.string(),
    num_subtask_marks: v.int64(),
    stored_frame_hw: v.array(v.int64()),
    camera_keys_by_role: v.record(v.string(), v.string()),
    crop_boxes: v.record(v.string(), v.array(v.int64())),
    review_camera_roles: v.array(v.string()),
    source: v.string(),
  },
  handler: async (ctx, args) => {
    const principal = await requireEditorOrService(ctx, args.serviceToken);
    if (principal !== "service") {
      throw new Error("Task specs are exported by the Python registry (service principal) only");
    }
    if (args.stored_frame_hw.length !== 2) {
      throw new Error(`stored_frame_hw must be [H, W], got ${args.stored_frame_hw}`);
    }
    const [h, w] = args.stored_frame_hw.map(Number);
    for (const [role, box] of Object.entries(args.crop_boxes)) {
      if (box.length !== 4) {
        throw new Error(`crop box for ${role} must be [x0, y0, x1, y1], got ${box}`);
      }
      const [x0, y0, x1, y1] = box.map(Number);
      if (!(0 <= x0 && x0 < x1 && x1 <= w && 0 <= y0 && y0 < y1 && y1 <= h)) {
        throw new Error(
          `crop box for ${role} out of bounds for ${w}x${h} stored frames: ${box}`
        );
      }
      if (!(role in args.camera_keys_by_role)) {
        throw new Error(`crop box role ${role} is not a station role`);
      }
    }
    for (const role of args.review_camera_roles) {
      if (!(role in args.camera_keys_by_role)) {
        throw new Error(`review camera role ${role} is not a station role`);
      }
    }
    const row = {
      task: args.task,
      task_name: args.task_name,
      num_subtask_marks: args.num_subtask_marks,
      stored_frame_hw: args.stored_frame_hw,
      camera_keys_by_role: args.camera_keys_by_role,
      crop_boxes: args.crop_boxes,
      review_camera_roles: args.review_camera_roles,
      exported_at: Date.now(),
      source: args.source,
    };
    const existing = await ctx.db
      .query("taskSpecs")
      .withIndex("by_task", (q) => q.eq("task", args.task))
      .unique();
    if (existing) {
      await ctx.db.replace(existing._id, row);
      return existing._id;
    }
    return await ctx.db.insert("taskSpecs", row);
  },
});

export const forTask = query({
  args: { task: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("taskSpecs")
      .withIndex("by_task", (q) => q.eq("task", args.task))
      .unique();
  },
});

export const all = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("taskSpecs").collect();
  },
});
