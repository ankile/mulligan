/**
 * Live stage-label coverage aggregates — the data behind the Coverage tab.
 *
 * Everything is computed from the reactive tables (stageTaskSpecs, the
 * selected stagePredictions run per dataset, stageReviews, datasets), so the
 * dashboard reflects every prediction selection and review save immediately.
 * Per (episode, schema) the "current label" is the latest COMMITTED review
 * (confirmed | corrected, across reviewers), else the selected prediction.
 *
 * Aggregation is per task over the by_task indexes (one query per task
 * section), which keeps each query well inside Convex's per-transaction read
 * budget.
 */
import { v } from "convex/values";
import { query } from "./_generated/server";
import { isNewer, reviewerKey } from "./stageReviews";
import { effectiveStatus } from "./statusShared";
import { loadTaskStatusMap } from "./statuses";

function emptyHist(nStages: number): number[] {
  return Array.from({ length: nStages }, () => 0);
}

function stageOf(label: Record<string, unknown> | undefined, stageField: string): number | null {
  // Label numbers arrive as float64 through the service client; only integral
  // values are valid stages.
  const raw = label?.[stageField];
  return typeof raw === "number" && Number.isInteger(raw) ? raw : null;
}

/** The live task list (light — used by the dashboard to fan out per-task queries). */
export const tasks = query({
  args: {},
  handler: async (ctx) => {
    const taskStatuses = await loadTaskStatusMap(ctx);
    const live = (await ctx.db.query("stageTaskSpecs").collect()).filter((s) => s.live);
    return live
      .map((s) => ({
        task: s.task,
        taxonomy_version: s.taxonomy_version,
        status: effectiveStatus(undefined, s.task, taskStatuses),
      }))
      .sort((a, b) => a.task.localeCompare(b.task));
  },
});

export const forTask = query({
  args: { task: v.string(), includeAll: v.optional(v.boolean()) },
  handler: async (ctx, args) => {
    const specRow = (
      await ctx.db
        .query("stageTaskSpecs")
        .withIndex("by_task", (q) => q.eq("task", args.task))
        .collect()
    ).find((s) => s.live);
    if (specRow === undefined) return null;
    const spec = specRow.spec as {
      stage_field: string;
      ladder: { max_stage: number };
    };
    const stageField = spec.stage_field;
    const nStages = spec.ladder.max_stage + 1;
    const tax = specRow.taxonomy_version;

    const datasets = await ctx.db.query("datasets").collect();
    const numEpisodes = new Map<string, number | null>(
      datasets.map((d) => [d.repo_id, d.num_episodes == null ? null : Number(d.num_episodes)])
    );
    // Mainline lens: drop repos whose effective dataset status is not
    // mainline BEFORE aggregating, so the repo table, stage histograms,
    // pipeline volumes, and reviewer counts all describe the same visible
    // set. Repos with predictions/reviews but no datasets row inherit the
    // task-level status (same resolution as everywhere else).
    const taskStatuses = await loadTaskStatusMap(ctx);
    const statusByRepo = new Map(
      datasets.map((d) => [d.repo_id, effectiveStatus(d.status, d.task, taskStatuses)])
    );
    const repoVisible = (repo: string) =>
      args.includeAll === true ||
      (statusByRepo.get(repo) ?? effectiveStatus(undefined, args.task, taskStatuses)) ===
        "mainline";
    const selections = (await ctx.db.query("stagePredictionSelections")
      .withIndex("by_task", (q) => q.eq("task", args.task)).collect())
      .filter((s) => s.taxonomy_version === tax && repoVisible(s.dataset_repo));
    type CoveragePrediction = {
      dataset_repo: string; episode_index: bigint; label: Record<string, unknown>;
      violation_codes?: string[]; pipeline: { name: string; version: string; git_commit: string };
      evidence: unknown;
    };
    const prefills: CoveragePrediction[] = [];
    for (const selected of selections) {
      const run = await ctx.db.get(selected.run_id);
      if (!run || run.status !== "published") throw new Error("coverage selection references an unpublished or missing run");
      const members = await ctx.db.query("stagePredictionMembers")
        .withIndex("by_run_episode", (q) => q.eq("run_id", run._id)).collect();
      for (const member of members) prefills.push({
        dataset_repo: run.dataset_repo, episode_index: member.episode_index,
        label: member.stage === undefined ? {} : { [stageField]: member.stage },
        violation_codes: member.flagged ? ["review_required"] : [],
        pipeline: run.pipeline, evidence: { model: run.provenance?.model ?? "unknown" },
      });
    }
    const reviews = (
      await ctx.db
        .query("stageReviews")
        .withIndex("by_task", (q) => q.eq("task", args.task))
        .collect()
    ).filter((r) => r.taxonomy_version === tax && repoVisible(r.dataset_repo));

    // Latest review per (repo, episode, reviewer); cleared folds out.
    const latestPerReviewer = new Map<string, (typeof reviews)[number]>();
    for (const row of reviews) {
      const key = `${row.dataset_repo}|${row.episode_index}|${reviewerKey(row)}`;
      const prev = latestPerReviewer.get(key);
      if (prev === undefined || isNewer(row, prev)) latestPerReviewer.set(key, row);
    }
    const folded = [...latestPerReviewer.values()].filter((r) => r.status !== "cleared");

    // Latest committed row per (repo, episode) across reviewers — the human
    // side of the current-label fold.
    const committedByEp = new Map<string, (typeof reviews)[number]>();
    for (const row of folded) {
      if (row.status !== "confirmed" && row.status !== "corrected") continue;
      const key = `${row.dataset_repo}|${row.episode_index}`;
      const prev = committedByEp.get(key);
      if (prev === undefined || isNewer(row, prev)) committedByEp.set(key, row);
    }

    const prefillByEp = new Map<string, (typeof prefills)[number]>();
    for (const p of prefills) prefillByEp.set(`${p.dataset_repo}|${p.episode_index}`, p);

    // Per-repo coverage rows.
    const repoNames = [
      ...new Set([...prefills.map((p) => p.dataset_repo), ...folded.map((r) => r.dataset_repo)]),
    ].sort();
    const repos = repoNames.map((repo) => {
      const rPrefills = prefills.filter((p) => p.dataset_repo === repo);
      const rFolded = folded.filter((r) => r.dataset_repo === repo);
      const committedEps = new Set(
        rFolded
          .filter((r) => r.status === "confirmed" || r.status === "corrected")
          .map((r) => String(r.episode_index))
      );
      const prefillEps = new Set(rPrefills.map((p) => String(p.episode_index)));
      // DISTINCT EPISODES per status (not reviewer-rows): two reviewers both
      // uncertain on one episode is a 1-episode backlog, not 2.
      const epsWith = (status: string) =>
        new Set(rFolded.filter((r) => r.status === status).map((r) => String(r.episode_index)))
          .size;
      return {
        repo,
        num_episodes: numEpisodes.get(repo) ?? null,
        n_prefill: rPrefills.length,
        n_flagged: rPrefills.filter((p) => (p.violation_codes ?? []).length > 0).length,
        n_committed: committedEps.size,
        n_uncertain: epsWith("uncertain"),
        n_draft: epsWith("draft"),
        n_vlm_only: [...prefillEps].filter((e) => !committedEps.has(e)).length,
        n_human_only: [...committedEps].filter((e) => !prefillEps.has(e)).length,
      };
    });

    // Stage histograms: current labels (committed > prefill) + committed-only.
    const currentHist = emptyHist(nStages);
    const committedHist = emptyHist(nStages);
    let unknownStage = 0;
    const allEps = new Set([...prefillByEp.keys(), ...committedByEp.keys()]);
    for (const key of allEps) {
      const committed = committedByEp.get(key);
      // A committed row wins outright — never fall through to the prefill
      // when a committed row exists but lacks a label (save() forbids that
      // state today; if it ever appears, count it as unknown, not VLM).
      const label = (committed !== undefined ? committed.label : prefillByEp.get(key)?.label) as
        | Record<string, unknown>
        | undefined;
      const s = stageOf(label, stageField);
      if (s === null || s < 0 || s >= nStages) {
        unknownStage += 1;
        continue;
      }
      currentHist[s] += 1;
      if (committed !== undefined) committedHist[s] += 1;
    }

    // Pipeline volume (prefills are already one operative row per episode).
    const pipelineCounts = new Map<
      string,
      { name: string; version: string; model: string; n: number }
    >();
    for (const p of prefills) {
      const model = String((p.evidence as { model?: unknown })?.model ?? "unknown");
      const key = `${p.pipeline.name}@${p.pipeline.version}|${model}`;
      const entry = pipelineCounts.get(key) ?? {
        name: p.pipeline.name,
        version: p.pipeline.version,
        model,
        n: 0,
      };
      entry.n += 1;
      pipelineCounts.set(key, entry);
    }

    // Reviewer activity over the folded (latest, non-cleared) rows, keyed by
    // the FOLD identity (reviewer_user_id, else svc:name) so a service replay
    // and a signed-in session sharing a display name never merge.
    const reviewerCounts = new Map<
      string,
      { reviewer: string; committed: number; uncertain: number; draft: number }
    >();
    for (const r of folded) {
      const key = reviewerKey(r);
      const display = key.startsWith("svc:") ? `${r.reviewer} (service)` : r.reviewer;
      const entry = reviewerCounts.get(key) ?? {
        reviewer: display,
        committed: 0,
        uncertain: 0,
        draft: 0,
      };
      if (r.status === "confirmed" || r.status === "corrected") entry.committed += 1;
      else if (r.status === "uncertain") entry.uncertain += 1;
      else if (r.status === "draft") entry.draft += 1;
      reviewerCounts.set(key, entry);
    }

    return {
      task: specRow.task,
      taxonomy_version: tax,
      stage_field: stageField,
      n_stages: nStages,
      repos,
      pipelines: [...pipelineCounts.values()].sort((a, b) => b.n - a.n),
      reviewers: [...reviewerCounts.values()].sort((a, b) => b.committed - a.committed),
      stage_hist_current: currentHist,
      stage_hist_committed: committedHist,
      n_unknown_stage: unknownStage,
      generated_at: Date.now(),
    };
  },
});
