import { query, mutation } from "./_generated/server";
import type { Id } from "./_generated/dataModel";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";
import { canonicalDigest } from "./stagePredictionContract";
import { blankTrajectoryReview } from "./trajectoryReview";
import { requireEditorOrService } from "./access";
import { reviewProtocolValidator, stageReviewCoverage } from "./stageReviewCoverage";
import { analyzeTrajectoryTimeline } from "./trajectoryTimeline";
import { eventLinksValidator, validateTrajectoryEventLinks } from "./trajectoryEventLinks";
import {
  canonicalizeStageLabel,
  validateStageLabel,
  type ExportedStageSpec,
} from "./stageConsistency";

/**
 * Human stage-label reviews are APPEND-ONLY (with one deliberate exception:
 * consecutive drafts collapse — see below): every save inserts a row; readers
 * fold to the latest row per (dataset_repo, episode_index, taxonomy_version,
 * reviewer). Per-reviewer folding is what makes blinded double-labeling
 * representable — two reviewers each hold a live latest row and disagreement
 * is a query, not a conflict.
 *
 * Statuses:
 *  - confirmed:  the episode's review is complete (gold-eligible within its
 *    recorded review_coverage; a row without coverage counts as unknown).
 *    For structured-v1, source prose/confidence are not human gold. Whether the
 *    reviewer edited the prediction is not encoded here — it is derivable
 *    from the row's label vs the prediction it was shown (prediction_id,
 *    prefill_pushed_at) and from the HF ledger's vlm/human event chain.
 *  - corrected:  gold-eligible, same as confirmed. The web form saves
 *    confirmed; machine saves may still send corrected.
 *  - uncertain:  reviewed but not gold-eligible; violations allowed
 *  - draft:      lossless autosave of in-progress edits; violations allowed.
 *    A draft REPLACES the same reviewer's previous draft (working state, not
 *    audit trail) so autosave cannot grow the table unboundedly.
 *  - cleared:    undo — the reviewer's row folds out entirely
 *
 * Every stored label is canonicalized against the spec's field kinds before
 * insert (string-typed machine saves would otherwise manufacture false
 * disagreements against web-typed rows), and confirmed/corrected saves are
 * gated by the SAME TS rule interpreter the form uses (stageConsistency.ts,
 * pinned by tests/fixtures/stage-consistency-fixtures.json).
 */

const STATUSES = ["confirmed", "corrected", "uncertain", "draft", "cleared"] as const;
const COMMITTED = ["confirmed", "corrected"] as const;

/** Stable per-reviewer fold key: auth user id for humans, name for machine saves. */
export function reviewerKey(row: { reviewer: string; reviewer_user_id?: string | null }): string {
  return row.reviewer_user_id ?? `svc:${row.reviewer}`;
}

/** Later-wins ordering with a deterministic tiebreak: a replay re-run with an
 * identical saved_at_override must supersede the earlier row, not lose to it. */
export function isNewer(
  a: { saved_at: number; _creationTime: number },
  b: { saved_at: number; _creationTime: number }
): boolean {
  return a.saved_at > b.saved_at || (a.saved_at === b.saved_at && a._creationTime > b._creationTime);
}

export const save = mutation({
  args: {
    serviceToken: v.optional(v.string()),
    task: v.string(),
    dataset_repo: v.string(),
    episode_index: v.int64(),
    taxonomy_version: v.string(),
    status: v.string(),
    label: v.optional(v.record(v.string(), v.any())),
    notes: v.optional(v.string()),
    review_protocol: v.optional(reviewProtocolValidator),
    event_links: v.optional(eventLinksValidator),
    prefill_pushed_at: v.optional(v.float64()),
    prediction_id: v.optional(v.id("stagePredictions")),
    prediction_sha256: v.optional(v.string()),
    copied_from_review_id: v.optional(v.id("stageReviews")),
    blind: v.optional(v.boolean()),
    episode_duration_s: v.optional(v.float64()),
    // Reviewer name for machine saves of labels a person made elsewhere
    // (service principal only, and required there).
    reviewer_override: v.optional(v.string()),
    // Decision time for those machine saves; human saves are stamped
    // server-side and may not set it.
    saved_at_override: v.optional(v.float64()),
  },
  handler: async (ctx, args) => {
    const principal = await requireEditorOrService(ctx, args.serviceToken);
    if (!(STATUSES as readonly string[]).includes(args.status)) {
      throw new Error(`Invalid stage review status: ${args.status}`);
    }
    if (args.episode_index < BigInt(0)) {
      throw new Error(`episode_index must be non-negative, got ${args.episode_index}`);
    }
    const specRow = await ctx.db
      .query("stageTaskSpecs")
      .withIndex("by_task_version", (q) =>
        q.eq("task", args.task).eq("taxonomy_version", args.taxonomy_version)
      )
      .unique();
    if (specRow === null) {
      throw new Error(
        `no stageTaskSpecs row for ${args.task}@${args.taxonomy_version} — ` +
          "a review must reference an exported taxonomy version"
      );
    }
    const spec = specRow.spec as ExportedStageSpec;
    if (args.event_links !== undefined && (!spec.trajectory || args.status === "cleared")) {
      throw new Error("event_links require a non-cleared trajectory review");
    }
    const reviewCoverage = stageReviewCoverage(args.review_protocol, args.status, !!spec.trajectory);

    // Pin exactly the prediction the reviewer saw. An active-run change must
    // never alter the validation duration or attribution of an in-flight form.
    if ((args.prediction_id === undefined) !== (args.prediction_sha256 === undefined)) {
      throw new Error("prediction_id and prediction_sha256 must be supplied together");
    }
    const copiedReview = args.copied_from_review_id === undefined ? null : await ctx.db.get(args.copied_from_review_id);
    if (args.copied_from_review_id !== undefined && (!copiedReview || !copiedReview.label ||
        !["confirmed", "corrected", "uncertain"].includes(copiedReview.status) ||
        copiedReview.task !== args.task || copiedReview.dataset_repo !== args.dataset_repo ||
        copiedReview.episode_index !== args.episode_index || copiedReview.taxonomy_version !== args.taxonomy_version)) {
      throw new Error("copied human review identity mismatch");
    }
    if (copiedReview && (copiedReview.prediction_id !== args.prediction_id ||
        copiedReview.prediction_sha256 !== args.prediction_sha256 ||
        copiedReview.prefill_pushed_at !== args.prefill_pushed_at)) {
      throw new Error("copied human review prediction provenance mismatch");
    }
    let trajectoryIdentity = spec.trajectory
      ? blankTrajectoryReview(spec.trajectory, args.dataset_repo, args.episode_index).trajectory_identity
      : undefined;
    let predictionRunId: Id<"stagePredictionRuns"> | undefined;
    let resolvedDuration = copiedReview?.episode_duration_s ?? args.episode_duration_s;
    let resolvedPushedAt = args.prefill_pushed_at;
    if (args.prediction_id !== undefined) {
      const prediction = await ctx.db.get(args.prediction_id);
      if (!prediction || prediction.content_sha256 !== args.prediction_sha256) {
        throw new Error("prediction reference is missing or its hash does not match");
      }
      const run = await ctx.db.get(prediction.run_id);
      if (!run || run.status !== "published" || run.task !== args.task ||
          run.dataset_repo !== args.dataset_repo || run.taxonomy_version !== args.taxonomy_version ||
          prediction.episode_index !== args.episode_index || run.taxonomy_hash !== specRow.taxonomy_hash) {
        throw new Error("prediction reference does not match this episode and schema, or run is unpublished");
      }
      if (spec.trajectory) trajectoryIdentity = prediction.label.trajectory_identity;
      predictionRunId = run._id;
      resolvedDuration = prediction.episode_duration_s;
      resolvedPushedAt = run.published_at;
    } else if (args.prefill_pushed_at !== undefined) {
      throw new Error("prefill_pushed_at requires prediction_id");
    }
    if (resolvedDuration !== undefined && (!Number.isFinite(resolvedDuration) || resolvedDuration <= 0)) {
      throw new Error("episode_duration_s must be finite and positive");
    }

    let label = args.label;
    if (args.status === "cleared") {
      if (label !== undefined) {
        throw new Error("A cleared review must not carry a label");
      }
    } else {
      if (label === undefined) {
        throw new Error(`A ${args.status} review requires the full label row`);
      }
      const canonical = canonicalizeStageLabel(spec, label);
      if (canonical.unknownKeys.length > 0) {
        throw new Error(
          `label carries keys outside the spec's editable fields: ` +
            canonical.unknownKeys.join(", ")
        );
      }
      label = canonical.label;
      if (spec.trajectory && (trajectoryIdentity === undefined || label.trajectory_identity === undefined ||
          await canonicalDigest(label.trajectory_identity) !== await canonicalDigest(trajectoryIdentity))) {
        throw new Error("trajectory identity must match the exact prediction source or source-free episode identity");
      }
      if ((COMMITTED as readonly string[]).includes(args.status)) {
        const linkErrors = validateTrajectoryEventLinks(label, args.event_links ?? []);
        if (linkErrors.length > 0) throw new Error(linkErrors.join("; "));
        const violations = validateStageLabel(spec, label, resolvedDuration);
        if (violations.length > 0) {
          throw new Error(
            `label is internally inconsistent (${violations.length} violation(s)): ` +
              violations.map((viol) => `[${viol.code}] ${viol.message}`).join("; ")
          );
        }
        // A review-only gate: immutable imported predictions and earlier
        // saved labels remain untouched, including their original conflicts.
        if (spec.trajectory) {
          const timelineIssues = analyzeTrajectoryTimeline(spec.trajectory, label);
          if (timelineIssues.length > 0) {
            throw new Error("label timeline is inconsistent: " + timelineIssues.map((issue) => issue.message).join("; "));
          }
        }
      }
    }

    let reviewer: string;
    let reviewerUserId: Id<"users"> | undefined;
    if (principal === "service") {
      if (args.reviewer_override === undefined) {
        throw new Error("service saves must carry reviewer_override for attribution");
      }
      reviewer = args.reviewer_override;
      // Attach the stable user id when the overridden reviewer has a known
      // account, so a machine save of a person's labels folds
      // under the SAME identity as their live web reviews (otherwise one
      // person holds two fold keys and double-counts / self-disagrees).
      const users = await ctx.db.query("users").collect();
      const matches = users.filter((u) => u.username === reviewer);
      reviewerUserId = matches.length === 1 ? matches[0]._id : undefined;
    } else {
      if (args.reviewer_override !== undefined || args.saved_at_override !== undefined) {
        throw new Error("reviewer/saved_at overrides are reserved for the service principal");
      }
      reviewer = principal;
      reviewerUserId = (await getAuthUserId(ctx)) ?? undefined;
    }

    const doc = {
      task: args.task,
      dataset_repo: args.dataset_repo,
      episode_index: args.episode_index,
      taxonomy_version: args.taxonomy_version,
      status: args.status,
      label,
      notes: args.notes,
      review_coverage: reviewCoverage,
      event_links: args.event_links,
      prefill_pushed_at: resolvedPushedAt,
      prediction_id: args.prediction_id,
      prediction_sha256: args.prediction_sha256,
      prediction_run_id: predictionRunId,
      copied_from_review_id: args.copied_from_review_id,
      taxonomy_hash: specRow.taxonomy_hash,
      blind: args.blind,
      // Persisted so a stored row stays re-validatable on its own (the bounds
      // context rides the row, independent of later prediction selections).
      episode_duration_s: resolvedDuration,
      reviewer,
      reviewer_user_id: reviewerUserId,
      saved_at: principal === "service" ? (args.saved_at_override ?? Date.now()) : Date.now(),
    };

    if (args.status === "draft") {
      const rows = (
        await ctx.db
          .query("stageReviews")
          .withIndex("by_repo_episode", (q) =>
            q.eq("dataset_repo", args.dataset_repo).eq("episode_index", args.episode_index)
          )
          .collect()
      ).filter(
        (r) =>
          r.taxonomy_version === args.taxonomy_version &&
          reviewerKey(r) === (reviewerUserId ?? `svc:${reviewer}`)
      );
      let latest: (typeof rows)[number] | undefined;
      for (const row of rows) {
        if (latest === undefined || isNewer(row, latest)) latest = row;
      }
      // A draft must never DEMOTE a committed verdict out of the latest-wins
      // fold (a stray keypress + navigation would silently pull the episode
      // back into the unreviewed queue and out of gold). Committed decisions
      // change only by re-verdicting.
      if (latest !== undefined && (COMMITTED as readonly string[]).includes(latest.status)) {
        throw new Error(
          `episode ${args.episode_index} already has a ${latest.status} review — ` +
            "drafts cannot supersede a committed verdict; re-verdict to change it"
        );
      }
      // Collapse consecutive drafts: replace this reviewer's latest row when
      // it is itself a draft (autosave is working state, not audit trail).
      if (latest !== undefined && latest.status === "draft") {
        await ctx.db.replace(latest._id, doc);
        return latest._id;
      }
    }
    return await ctx.db.insert("stageReviews", doc);
  },
});

/**
 * Latest non-cleared row per (episode_index, taxonomy_version, reviewer), plus
 * summary counts. `taxonomy_version` narrows to one schema when given.
 */
export const latestForRepo = query({
  args: {
    dataset_repo: v.string(),
    taxonomy_version: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    let rows = await ctx.db
      .query("stageReviews")
      .withIndex("by_repo", (q) => q.eq("dataset_repo", args.dataset_repo))
      .collect();
    if (args.taxonomy_version !== undefined) {
      rows = rows.filter((r) => r.taxonomy_version === args.taxonomy_version);
    }
    const latest = new Map<string, (typeof rows)[number]>();
    for (const row of rows) {
      const key = `${row.episode_index}|${row.taxonomy_version}|${reviewerKey(row)}`;
      const prev = latest.get(key);
      if (prev === undefined || isNewer(row, prev)) {
        latest.set(key, row);
      }
    }
    const folded = [...latest.values()].filter((r) => r.status !== "cleared");
    const count = (status: string) => folded.filter((r) => r.status === status).length;
    return {
      episodes: folded,
      num_confirmed: count("confirmed"),
      num_corrected: count("corrected"),
      num_uncertain: count("uncertain"),
      num_draft: count("draft"),
    };
  },
});

/** Distinct dataset repos holding stage reviews for a task — the gold
 *  consolidator's discovery surface (no hand-maintained repo lists). */
export const reposForTask = query({
  args: { task: v.string() },
  handler: async (ctx, args) => {
    const rows = await ctx.db
      .query("stageReviews")
      .withIndex("by_task", (q) => q.eq("task", args.task))
      .collect();
    return [...new Set(rows.map((r) => r.dataset_repo))].sort();
  },
});

/** Full audit trail for one episode, oldest first. */
export const historyForEpisode = query({
  args: { dataset_repo: v.string(), episode_index: v.int64() },
  handler: async (ctx, args) => {
    const rows = await ctx.db
      .query("stageReviews")
      .withIndex("by_repo_episode", (q) =>
        q.eq("dataset_repo", args.dataset_repo).eq("episode_index", args.episode_index)
      )
      .collect();
    return rows.sort((a, b) => (isNewer(a, b) ? 1 : -1));
  },
});

/**
 * Episodes whose COMMITTED latest rows differ across reviewers on the core
 * triple (stage, failure mode, final state) under one taxonomy version — the
 * blinded-double-labeling disagreement queue.
 */
export const disagreementsForRepo = query({
  args: {
    dataset_repo: v.string(),
    task: v.string(),
    taxonomy_version: v.string(),
  },
  handler: async (ctx, args) => {
    const specRow = await ctx.db
      .query("stageTaskSpecs")
      .withIndex("by_task_version", (q) =>
        q.eq("task", args.task).eq("taxonomy_version", args.taxonomy_version)
      )
      .unique();
    if (specRow === null) {
      throw new Error(`no stageTaskSpecs row for ${args.task}@${args.taxonomy_version}`);
    }
    const spec = specRow.spec as ExportedStageSpec;
    const rows = (
      await ctx.db
        .query("stageReviews")
        .withIndex("by_repo", (q) => q.eq("dataset_repo", args.dataset_repo))
        .collect()
    ).filter((r) => r.taxonomy_version === args.taxonomy_version);
    const latest = new Map<string, (typeof rows)[number]>();
    for (const row of rows) {
      const key = `${row.episode_index}|${reviewerKey(row)}`;
      const prev = latest.get(key);
      if (prev === undefined || isNewer(row, prev)) {
        latest.set(key, row);
      }
    }
    const byEpisode = new Map<string, (typeof rows)[number][]>();
    for (const row of latest.values()) {
      if (!(COMMITTED as readonly string[]).includes(row.status)) continue;
      const key = String(row.episode_index);
      byEpisode.set(key, [...(byEpisode.get(key) ?? []), row]);
    }
    const triple = (row: (typeof rows)[number]) => {
      const label = row.label ?? {};
      return JSON.stringify([
        label[spec.stage_field] ?? null,
        label[spec.failure_mode_field] ?? null,
        label[spec.final_state_field] ?? null,
      ]);
    };
    const disagreements = [];
    for (const [episode, reviewRows] of byEpisode) {
      if (reviewRows.length < 2) continue;
      const triples = new Set(reviewRows.map(triple));
      if (triples.size > 1) {
        disagreements.push({ episode_index: Number(episode), reviews: reviewRows });
      }
    }
    return disagreements.sort((a, b) => a.episode_index - b.episode_index);
  },
});
