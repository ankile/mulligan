import { query, mutation } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { requireEditor, requireEditorOrService } from "./access";

export const episodeNotes = query({
  args: { dataset_repo: v.string(), episode_index: v.int64() },
  handler: async (ctx, args) => await ctx.db
    .query("episodeNotes")
    .withIndex("by_repo_episode", (q) =>
      q.eq("dataset_repo", args.dataset_repo).eq("episode_index", args.episode_index)
    )
    .unique(),
});

/** Reusable notes from other episodes across datasets for the same task. */
export const noteSuggestions = query({
  args: { dataset_repo: v.string(), episode_index: v.int64() },
  handler: async (ctx, args) => {
    const dataset = await ctx.db.query("datasets")
      .withIndex("by_repo", (q) => q.eq("repo_id", args.dataset_repo)).unique();
    if (!dataset || !dataset.task.trim()) return { recent: [], mostUsed: [] };
    const datasets = await ctx.db.query("datasets")
      .withIndex("by_task", (q) => q.eq("task", dataset.task)).collect();
    const groups = await Promise.all(datasets.map((row) => ctx.db.query("episodeNotes")
      .withIndex("by_repo_episode", (q) => q.eq("dataset_repo", row.repo_id)).collect()));
    const counts = new Map<string, { notes: string; count: number; lastUsed: number }>();
    for (const row of groups.flat()) {
      if (row.dataset_repo === args.dataset_repo && row.episode_index === args.episode_index) continue;
      const notes = row.notes.trim();
      if (!notes) continue;
      const previous = counts.get(notes);
      counts.set(notes, {
        notes,
        count: (previous?.count ?? 0) + 1,
        lastUsed: Math.max(previous?.lastUsed ?? 0, row.updated_at),
      });
    }
    const suggestions = [...counts.values()];
    const byRecent = (a: typeof suggestions[number], b: typeof suggestions[number]) =>
      b.lastUsed - a.lastUsed || a.notes.localeCompare(b.notes);
    return {
      recent: [...suggestions].sort(byRecent).slice(0, 6),
      mostUsed: suggestions.sort((a, b) => b.count - a.count || byRecent(a, b)).slice(0, 6),
    };
  },
});

export const saveNotes = mutation({
  args: { dataset_repo: v.string(), episode_index: v.int64(), notes: v.string() },
  handler: async (ctx, args) => {
    const editor = await requireEditor(ctx);
    if (args.episode_index < BigInt(0)) throw new Error("Episode index must be non-negative");
    const previous = await ctx.db
      .query("episodeNotes")
      .withIndex("by_repo_episode", (q) =>
        q.eq("dataset_repo", args.dataset_repo).eq("episode_index", args.episode_index)
      )
      .unique();
    const fields = { ...args, updated_by: editor, updated_at: Date.now() };
    if (previous) await ctx.db.patch(previous._id, fields);
    else await ctx.db.insert("episodeNotes", fields);
  },
});

const OUTCOMES = ["success", "failure", "timeout"] as const;
const STATUSES = ["confirmed", "skipped", "cleared"] as const;

/**
 * Outcome reviews are APPEND-ONLY: every save inserts a row, and readers fold
 * to the latest row per (dataset_repo, episode_index). This preserves a full
 * audit trail; the applied record of truth on HF remains
 * `.outcome_edit_progress.json`, materialized by the apply worker.
 *
 * Record semantics mirror mulligan/tools/outcome_review.py exactly:
 *  - confirmed: {new_outcome, outcome_frame, soft_truncate, subtask_frames?}
 *    (subtask_frames key present ⇔ reviewed in subtask mode, may be empty)
 *  - skipped: reviewed, no change
 *  - cleared: undo a previous review (episode returns to unreviewed)
 */
export const save = mutation({
  args: {
    serviceToken: v.optional(v.string()),
    dataset_repo: v.string(),
    episode_index: v.int64(),
    status: v.string(),
    new_outcome: v.optional(v.string()),
    outcome_frame: v.optional(v.int64()),
    soft_truncate: v.optional(v.boolean()),
    subtask_frames: v.optional(v.array(v.int64())),
  },
  handler: async (ctx, args) => {
    const principal = await requireEditorOrService(ctx, args.serviceToken);
    if (!(STATUSES as readonly string[]).includes(args.status)) {
      throw new Error(`Invalid review status: ${args.status}`);
    }
    if (args.status === "confirmed") {
      if (
        args.new_outcome === undefined ||
        !(OUTCOMES as readonly string[]).includes(args.new_outcome)
      ) {
        throw new Error(
          `Confirmed review requires new_outcome in ${OUTCOMES.join("/")}, got ${args.new_outcome}`
        );
      }
      if (args.outcome_frame === undefined || args.outcome_frame < BigInt(0)) {
        throw new Error("Confirmed review requires a non-negative outcome_frame");
      }
    } else {
      if (
        args.new_outcome !== undefined ||
        args.outcome_frame !== undefined ||
        args.subtask_frames !== undefined
      ) {
        throw new Error(`A ${args.status} review must not carry outcome fields`);
      }
    }
    if (args.status === "cleared") {
      // Clearing an ALREADY-APPLIED decision is a trap: the HF edit stays in
      // place, the episode folds out of latestForRepo, and the next apply
      // would re-record lower counts that satisfy the freshness gate over an
      // un-reverted dataset. Applied decisions are corrected by RE-REVIEWING.
      const latest = await ctx.db
        .query("outcomeReviews")
        .withIndex("by_repo_episode", (q) =>
          q.eq("dataset_repo", args.dataset_repo).eq("episode_index", args.episode_index)
        )
        .collect();
      const newest = latest
        .filter((row) => row.status !== "cleared")
        .sort((a, b) => b.saved_at - a.saved_at)[0];
      if (newest !== undefined) {
        const jobs = await ctx.db
          .query("applyJobs")
          .withIndex("by_repo", (q) => q.eq("dataset_repo", args.dataset_repo))
          .collect();
        const appliedAfter = jobs.some(
          (job) =>
            job.status === "applied" &&
            job.started_at !== undefined &&
            job.started_at >= newest.saved_at
        );
        if (appliedAfter) {
          throw new Error(
            "This decision was already applied to HuggingFace — clearing cannot " +
              "revert it. Re-review the episode (confirm the corrected outcome) " +
              "and commit again instead."
          );
        }
      }
    }
    // Humans are attributed from their sign-in; machine saves as "service".
    const reviewer = principal;
    const userId = await getAuthUserId(ctx);
    return await ctx.db.insert("outcomeReviews", {
      dataset_repo: args.dataset_repo,
      episode_index: args.episode_index,
      status: args.status,
      new_outcome: args.new_outcome,
      outcome_frame: args.outcome_frame,
      soft_truncate: args.status === "confirmed" ? (args.soft_truncate ?? false) : undefined,
      subtask_frames: args.subtask_frames,
      reviewer,
      reviewer_user_id: userId ?? undefined,
      saved_at: Date.now(),
    });
  },
});

/** Latest review per episode for a repo (cleared rows fold to "no review"). */
export const latestForRepo = query({
  args: { dataset_repo: v.string() },
  handler: async (ctx, args) => {
    const rows = await ctx.db
      .query("outcomeReviews")
      .withIndex("by_repo", (q) => q.eq("dataset_repo", args.dataset_repo))
      .collect();
    const latest = new Map<string, (typeof rows)[number]>();
    for (const row of rows) {
      // _creationTime is monotonically assigned; later insert wins.
      const key = row.episode_index.toString();
      const prev = latest.get(key);
      if (!prev || row._creationTime > prev._creationTime) latest.set(key, row);
    }
    const episodes = [...latest.values()]
      .filter((row) => row.status !== "cleared")
      .sort((a, b) => Number(a.episode_index - b.episode_index));
    return {
      episodes,
      num_confirmed: episodes.filter((e) => e.status === "confirmed").length,
      num_skipped: episodes.filter((e) => e.status === "skipped").length,
    };
  },
});

/** Full append-only history for one episode (audit surface). */
export const historyForEpisode = query({
  args: { dataset_repo: v.string(), episode_index: v.int64() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("outcomeReviews")
      .withIndex("by_repo_episode", (q) =>
        q.eq("dataset_repo", args.dataset_repo).eq("episode_index", args.episode_index)
      )
      .collect();
  },
});
