import { query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { requireEditorOrService } from "./access";
import { normalizeTrajectoryReviewSpec, trajectoryTaskDefinitionDigest } from "./trajectoryReview";
import { canonicalDigest } from "./stagePredictionContract";

/**
 * Stage-label task specs are EXPORTED DATA: the Python stage-labeling registry
 * (mulligan/real/stage_specs/tasks.py StageLabelTaskSpec, serialized by
 * mulligan/tools/export_arena_task_specs.py) is the single source of truth. Rows
 * are keyed (task, taxonomy_version) so a live taxonomy and a candidate one
 * (a proposed stage split under evaluation) coexist; exactly one version per
 * task carries `live: true`. The UI renders the stage-review form and its
 * instant consistency feedback from `spec`, and the server re-runs the same
 * checks on every confirmed save.
 */

// Keys the UI's spec normalizer requires; guards against a partial or foreign
// payload reaching the UI.
const REQUIRED_SPEC_KEYS = [
  "task",
  "lifecycle_task",
  "taxonomy_version",
  "taxonomy_hash",
  "released_field",
  "ladder",
  "failure_modes",
  "final_states",
  "success_final_state",
  "stage_field",
  "final_state_field",
  "failure_mode_field",
  "event_fields",
  "bool_fields",
  "time_fields",
  "editable_fields",
  "constraints",
  "fps",
] as const;

export const upsert = mutation({
  args: {
    serviceToken: v.optional(v.string()),
    task: v.string(),
    taxonomy_version: v.string(),
    taxonomy_hash: v.string(),
    live: v.boolean(),
    spec: v.any(),
    source: v.string(),
  },
  handler: async (ctx, args) => {
    const principal = await requireEditorOrService(ctx, args.serviceToken);
    if (principal !== "service") {
      throw new Error(
        "Stage task specs are exported by the Python registry (service principal) only"
      );
    }
    if (typeof args.spec !== "object" || args.spec === null || Array.isArray(args.spec)) {
      throw new Error("spec payload must be a serialized stage-spec object");
    }
    const spec = args.spec as Record<string, unknown>;
    for (const key of REQUIRED_SPEC_KEYS) {
      if (!(key in spec)) {
        throw new Error(`spec payload missing required key "${key}"`);
      }
    }
    if (
      spec.task !== args.task ||
      spec.taxonomy_version !== args.taxonomy_version ||
      spec.taxonomy_hash !== args.taxonomy_hash
    ) {
      throw new Error(
        "spec payload task/taxonomy_version/taxonomy_hash must match the row keys"
      );
    }
    if (spec.trajectory !== undefined) {
      const trajectory = normalizeTrajectoryReviewSpec(spec.trajectory, args.task, args.taxonomy_version);
      if (await trajectoryTaskDefinitionDigest(trajectory.task_definition) !== trajectory.task_definition_sha256) {
        throw new Error("trajectory task definition content does not match its SHA-256 pin");
      }
      const hashInput = { ...spec }; delete hashInput.taxonomy_hash;
      if (await canonicalDigest(hashInput) !== args.taxonomy_hash) {
        throw new Error("trajectory review spec content does not match taxonomy_hash");
      }
    }
    const existing = await ctx.db
      .query("stageTaskSpecs")
      .withIndex("by_task_version", (q) =>
        q.eq("task", args.task).eq("taxonomy_version", args.taxonomy_version)
      )
      .unique();
    // Versions are immutable, including when only drafts or predictions refer
    // to them. Compare actual content; a caller-supplied hash cannot authorize
    // changing the meaning of previously stored labels.
    if (existing && (
      existing.taxonomy_hash !== args.taxonomy_hash ||
      await canonicalDigest(existing.spec) !== await canonicalDigest(args.spec)
    )) {
      throw new Error("stage taxonomy content is immutable; bump taxonomy_version instead");
    }
    const siblings = await ctx.db
      .query("stageTaskSpecs")
      .withIndex("by_task", (q) => q.eq("task", args.task))
      .collect();
    if (args.live) {
      for (const sib of siblings) {
        if (sib._id !== existing?._id && sib.live) {
          await ctx.db.patch(sib._id, { live: false });
        }
      }
    } else if (existing?.live) {
      // Demoting the ONLY live version would strand the task with no live
      // spec (the review UI's entry point) — a candidate export must carry a
      // NEW taxonomy_version, not repurpose the live one.
      const otherLive = siblings.some((sib) => sib._id !== existing._id && sib.live);
      if (!otherLive) {
        throw new Error(
          `${args.task}@${args.taxonomy_version} is the only LIVE version; ` +
            "exporting it as a candidate (live=false) would leave the task " +
            "with no live spec. Export the candidate under a new taxonomy_version."
        );
      }
    }
    const row = {
      task: args.task,
      taxonomy_version: args.taxonomy_version,
      taxonomy_hash: args.taxonomy_hash,
      live: args.live,
      spec: args.spec,
      exported_at: Date.now(),
      source: args.source,
    };
    if (existing) {
      await ctx.db.replace(existing._id, row);
      return existing._id;
    }
    return await ctx.db.insert("stageTaskSpecs", row);
  },
});

/** Every exported taxonomy version for a task (live + candidates). */
export const forTask = query({
  args: { task: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("stageTaskSpecs")
      .withIndex("by_task", (q) => q.eq("task", args.task))
      .collect();
  },
});

/** The single live taxonomy version for a task, or null when none exists. */
export const liveForTask = query({
  args: { task: v.string() },
  handler: async (ctx, args) => {
    const rows = await ctx.db
      .query("stageTaskSpecs")
      .withIndex("by_task", (q) => q.eq("task", args.task))
      .collect();
    const live = rows.filter((r) => r.live);
    if (live.length > 1) {
      throw new Error(
        `multiple live stage specs for ${args.task}: ` +
          live.map((r) => r.taxonomy_version).join(", ")
      );
    }
    return live[0] ?? null;
  },
});

export const all = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("stageTaskSpecs").collect();
  },
});
