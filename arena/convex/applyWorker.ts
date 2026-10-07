"use node";

/**
 * Outcome-review apply worker, run as a Convex action.
 *
 * Scheduled by applyJobs:enqueue (no polling): claims the job, materializes
 * the review overlay, runs the TS apply pipeline (convex/apply/) on a
 * snapshot fetched at the pre-apply sha, pushes ONE atomic HF commit pinned
 * on that sha (a concurrent push to main fails the commit loudly), advances
 * the v3.0 tag, patches the arena session's roundResults in place, and
 * reports status + shas back to the job row.
 *
 * Requires the HF_TOKEN deployment env var (write access to the datasets).
 * A dry-run may also upload to a pre-created apply-validation/ branch at
 * dryRunRevision, to exercise the real transport without moving main/v3.0
 * or changing Arena results.
 */

import { v } from "convex/values";
import { internalAction } from "./_generated/server";
import { internal, api } from "./_generated/api";
import { buildOverlay } from "./apply/progress";
import type { ReviewRow } from "./apply/progress";
import type { LabelSource } from "./apply/labelHistory";
import { headlessApply } from "./apply/pipeline";
import {
  revisionSha,
  listRepoFiles,
  downloadRepoFile,
  commitFiles,
  advanceLerobotVersionTag,
} from "./apply/hf";
import type { HfClient } from "./apply/hf";

const LOG_TAIL_CHARS = 2000;

export function resolveUploadBranch(dryRun: boolean, revision?: string, branch?: string): string | null {
  if (!dryRun && (revision !== undefined || branch !== undefined)) {
    throw new Error("Validation revision/branch require a dry-run job");
  }
  if (branch !== undefined && (!revision || !branch.startsWith("apply-validation/"))) {
    throw new Error("Validation uploads require a pinned revision and an apply-validation/ branch");
  }
  return dryRun ? (branch ?? null) : "main";
}

export const run = internalAction({
  args: {
    jobId: v.id("applyJobs"),
    dryRunRevision: v.optional(v.string()),
    dryRunUploadBranch: v.optional(v.string()),
  },
  handler: async (ctx, { jobId, dryRunRevision, dryRunUploadBranch }) => {
    const job = await ctx.runMutation(internal.applyJobs.claimById, { id: jobId });
    if (job === null) return; // claimed by another worker or cancelled first
    const repoId = job.dataset_repo;
    const log: string[] = [];
    let preSha: string | undefined;
    let postSha: string | undefined;
    const reportProgress = async (message: string) => {
      const memory = process.memoryUsage();
      const line = `${new Date().toISOString()} ${message} (RSS ${Math.round(memory.rss / 1048576)} MiB, external ${Math.round(memory.external / 1048576)} MiB)`;
      console.info(`[apply ${jobId}] ${message}`, memory);
      log.push(line);
      await ctx.runMutation(internal.applyJobs.recordProgressInternal, {
        id: jobId, message: line, pre_apply_sha: preSha, hf_commit_sha: postSha,
      });
    };
    try {
      const uploadBranch = resolveUploadBranch(Boolean(job.dry_run), dryRunRevision, dryRunUploadBranch);
      const token = process.env.HF_TOKEN;
      if (!token) throw new Error("HF_TOKEN deployment env var is not set");
      const client: HfClient = { repoId, token };

      const reviews = await ctx.runQuery(api.reviews.latestForRepo, { dataset_repo: repoId });
      const overlay = buildOverlay(reviews.episodes as unknown as ReviewRow[]);
      const numConfirmed = Object.keys(overlay.changed_episodes).length;
      const numSkipped = overlay.skipped_episodes.length;
      if (numConfirmed === 0 && numSkipped === 0) {
        throw new Error(`No confirmed or skipped reviews recorded for ${repoId}`);
      }
      // Label-history provenance: who made each decision and which apply job
      // carried it. Side-channel, NOT overlay fields — decision records keep
      // the shape mulligan/tools/outcome_review.py writes.
      const sourceByEpisode = new Map<number, LabelSource>(
        (reviews.episodes as unknown as ReviewRow[]).map((row) => [
          Number(row.episode_index),
          { kind: "human", agent: String(row.reviewer), tool: "web-review" },
        ])
      );

      const taskSpecs = (await ctx.runQuery(api.taskSpecs.all, {})) as Array<{
        task_name: string;
        num_subtask_marks: number | bigint;
      }>;

      preSha = await revisionSha(client, dryRunRevision ?? "main");
      await reportProgress(`Reading snapshot ${preSha}`);
      const paths = await listRepoFiles(client, preSha);
      const result = await headlessApply({
        store: { paths, fetch: (p) => downloadRepoFile(client, p, preSha!) },
        overlay,
        provenance: { sourceByEpisode, evidence: { apply_job: String(jobId) } },
        preApplySha: preSha,
        taskSpecs,
        parquetCreatedBy: dryRunUploadBranch === undefined ? undefined : `policy-arena validation ${jobId}`,
        onProgress: reportProgress,
      });
      log.push(...result.summary.log);

      if (job.dry_run) {
        if (uploadBranch !== null) {
          await reportProgress(`Validating upload of ${result.changedFiles.size} files to ${uploadBranch}`);
          const validationSha = await commitFiles({
            client, files: result.changedFiles, message: `Validate outcome apply ${jobId}`,
            parentCommit: preSha, branch: uploadBranch, onProgress: reportProgress,
          });
          await reportProgress(`Validation branch committed @ ${validationSha}`);
        }
        // Main was not changed, so the validation job must not read as applied.
        await ctx.runMutation(internal.applyJobs.finishInternal, {
          id: jobId,
          ok: false,
          error: uploadBranch === null
            ? "dry-run: apply computed in-memory, nothing pushed"
            : `dry-run: upload verified on ${uploadBranch}; main and v3.0 unchanged`,
          pre_apply_sha: preSha,
          log_tail: log.join("\n").slice(-LOG_TAIL_CHARS),
          num_confirmed: BigInt(numConfirmed),
          num_skipped: BigInt(numSkipped),
        });
        return;
      }

      await reportProgress(`Uploading ${result.changedFiles.size} files`);
      postSha = await commitFiles({
        client,
        files: result.changedFiles,
        message:
          `Apply ${numConfirmed} outcome review(s) from Policy Arena ` +
          `(job ${String(jobId)})`,
        parentCommit: preSha,
        onProgress: reportProgress,
      });
      await reportProgress(`Committed ${result.changedFiles.size} file(s) @ ${postSha}`);
      await advanceLerobotVersionTag(client, postSha);
      log.push(`Moved v3.0 -> ${postSha.slice(0, 8)}`);

      const corrected = await ctx.runMutation(internal.evalSessions.correctOutcomes, {
        dataset_repo: repoId,
        corrections: result.summary.episode_success.map((e) => ({
          episode_index: BigInt(e.episode_index),
          success: e.success,
          num_frames: BigInt(e.num_frames),
          num_subtask_marks: BigInt(e.num_subtask_marks),
        })),
      });
      log.push(
        corrected.session_found
          ? `Arena session synced in place: ${corrected.updated} round result(s) corrected.`
          : "(no arena session registered for this repo; sync skipped)"
      );

      const statsRefreshQueued = await ctx.runMutation(
        internal.datasets.enqueueStatsRefreshInternal,
        { repo_id: repoId }
      );
      log.push(
        statsRefreshQueued
          ? "Dataset summary refresh queued."
          : "(no arena dataset registered for this repo; summary refresh skipped)"
      );

      await ctx.runMutation(internal.applyJobs.finishInternal, {
        id: jobId,
        ok: true,
        hf_commit_sha: postSha,
        pre_apply_sha: preSha,
        log_tail: log.join("\n").slice(-LOG_TAIL_CHARS),
        num_confirmed: BigInt(numConfirmed),
        num_skipped: BigInt(numSkipped),
      });
    } catch (error) {
      const err = error instanceof Error ? (error.stack ?? error.message) : String(error);
      await ctx.runMutation(internal.applyJobs.finishInternal, {
        id: jobId,
        ok: false,
        error: err.slice(-LOG_TAIL_CHARS),
        log_tail: log.join("\n").slice(-LOG_TAIL_CHARS),
        // If HF was already mutated the pre-state sha is the rollback anchor;
        // record it on the failed job too.
        pre_apply_sha: preSha,
        hf_commit_sha: postSha,
      });
    }
  },
});
