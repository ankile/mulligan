import { useEffect, useState } from "react";
import { useMutation, useQuery } from "../../lib/arenaClient";
import { api } from "../../release/api";
import { formatClock } from "./format";

// ---------------------------------------------------------------------------
// Commit panel for the outcome review: queues an apply job that writes the
// reviewed outcomes to the Hugging Face dataset, and lists the repo's recent
// jobs with their status, commit and error.
// ---------------------------------------------------------------------------

export function CommitPanel({
  repoId,
  numConfirmed,
  numSkipped,
  numEpisodes,
}: {
  repoId: string;
  numConfirmed: number;
  numSkipped: number;
  numEpisodes: number;
}) {
  const jobs = useQuery(api.applyJobs.forRepo, { dataset_repo: repoId });
  const enqueue = useMutation(api.applyJobs.enqueue);
  const cancelJob = useMutation(api.applyJobs.cancel);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 5000);
    return () => clearInterval(timer);
  }, []);

  const activeJob = jobs?.find(
    (job) => job.status === "pending" || job.status === "applying"
  );
  const totalReviews = numConfirmed + numSkipped;

  const statusChip: Record<string, string> = {
    pending: "bg-gold-light text-gold",
    applying: "bg-gold-light text-gold",
    applied: "bg-teal-light text-teal",
    failed: "bg-coral-light text-coral",
    cancelled: "bg-warm-100 text-ink-muted",
  };

  return (
    <div className="border-t border-warm-200 bg-warm-50 px-6 py-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="font-mono text-xs text-ink">
          {numConfirmed} confirmed · {numSkipped} skipped · {numEpisodes} episodes
        </span>
        <div className="flex-1" />
        <button
          disabled={busy || totalReviews === 0 || Boolean(activeJob)}
          onClick={() => {
            setError(null);
            setBusy(true);
            enqueue({ dataset_repo: repoId })
              .catch((err: Error) => setError(err.message))
              .finally(() => setBusy(false));
          }}
          className={`px-4 py-2 rounded-lg text-xs font-medium transition-colors ${
            busy || totalReviews === 0 || activeJob
              ? "bg-warm-100 text-ink-muted/50 cursor-not-allowed"
              : "bg-teal text-white hover:bg-teal/90 cursor-pointer"
          }`}
        >
          {activeJob ? `Job ${activeJob.status}…` : "Commit to HuggingFace"}
        </button>
      </div>

      {error && (
        <div className="mt-3 rounded-lg border border-coral/30 bg-coral-light px-3 py-2 text-xs text-coral font-mono">
          {error}
        </div>
      )}

      {jobs && jobs.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {jobs.map((job) => (
            <div
              key={job._id}
              className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-ink-muted"
            >
              <span
                className={`px-1.5 py-0.5 rounded ${statusChip[job.status] ?? "bg-warm-100 text-ink-muted"}`}
              >
                {job.status}
              </span>
              <span>{formatClock(job.requested_at)}</span>
              <span>{job.requested_by}</span>
              {job.num_confirmed != null && (
                <span>
                  {Number(job.num_confirmed)} confirmed ·{" "}
                  {Number(job.num_skipped ?? BigInt(0))} skipped
                </span>
              )}
              {job.hf_commit_sha && (
                <a
                  href={`https://huggingface.co/datasets/${repoId}/tree/${job.hf_commit_sha}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-teal hover:underline"
                >
                  {job.hf_commit_sha.slice(0, 8)} →
                </a>
              )}
              {job.error && (
                <button
                  onClick={() => setExpanded(expanded === job._id ? null : job._id)}
                  className="text-coral hover:underline cursor-pointer"
                >
                  {expanded === job._id ? "hide error" : "show error"}
                </button>
              )}
              {(job.status === "pending" ||
                (job.status === "applying" &&
                  job.started_at !== undefined &&
                  now - Number(job.started_at) > 10 * 60 * 1000)) && (
                <button
                  onClick={() => {
                    setError(null);
                    cancelJob({ id: job._id }).catch((err: Error) =>
                      setError(err.message)
                    );
                  }}
                  className="text-ink-muted hover:text-coral cursor-pointer"
                  title={
                    job.status === "applying"
                      ? "Worker claim went stale — reclaim the stuck job, then re-commit"
                      : undefined
                  }
                >
                  {job.status === "applying" ? "reclaim stuck job" : "cancel"}
                </button>
              )}
              {expanded === job._id && (
                <pre className="w-full whitespace-pre-wrap rounded bg-white border border-warm-200 p-2 text-[10px] text-coral">
                  {job.error}
                  {job.log_tail ? `\n\n${job.log_tail}` : ""}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
