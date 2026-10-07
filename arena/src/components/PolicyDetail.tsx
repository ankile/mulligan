import type { Policy } from "../release/types";
import { useMemo, useState } from "react";
import { useQuery, useMutation } from "../lib/arenaClient";
import { api } from "../release/api";
import type { Id } from "../release/api";
import RolloutSection from "./RolloutSection";
import { StatusBadge, StatusSelect } from "./StatusBadge";
import { PolicyTagBadges, PolicyTagEditor } from "./PolicyTags";
import { computeArenaStats, ratingTrajectory, visibleSessions } from "../lib/arenaRatings";
import { useSearchParam, useSearchParamNullable } from "../lib/useSearchParam";

function CollapsibleSection({
  title,
  count,
  countColor,
  isOpen,
  onToggle,
  children,
}: {
  title: string;
  count: number | undefined;
  countColor: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div>
      <button
        onClick={onToggle}
        className="flex items-center gap-2 w-full text-left cursor-pointer group"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="currentColor"
          className={`text-ink-muted transition-transform ${isOpen ? "rotate-90" : ""}`}
        >
          <polygon points="8,4 20,12 8,20" />
        </svg>
        <span className="text-sm font-medium text-ink group-hover:text-teal transition-colors">
          {title}
        </span>
        {count !== undefined && (
          <span
            className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${countColor}`}
          >
            {count}
          </span>
        )}
      </button>
      {children}
    </div>
  );
}

export default function PolicyDetail({
  policyId, published, metric,
}: {
  policyId: Id<"policies">;
  published?: Policy;
  metric?: string;
}) {
  const policy = useQuery(api.policies.get, { id: policyId });
  const sessions = useQuery(api.evalSessions.getByPolicy, {
    policy_id: policyId,
  });
  const sessionOutcomes = useQuery(api.ratings.sessionOutcomes);
  const recentResults = useQuery(api.roundResults.getRecentByPolicy, {
    policy_id: policyId,
  });
  const failureResults = useQuery(api.roundResults.getFailuresByPolicy, {
    policy_id: policyId,
  });
  const successRateHistory = useQuery(
    api.roundResults.getSuccessRateHistory,
    { policy_id: policyId }
  );
  const viewer = useQuery(api.users.viewer);
  const setPolicyStatus = useMutation(api.policies.setStatus);
  const [showParam] = useSearchParam("show", "mainline");
  const showAll = showParam === "all";

  // Rating-as-of-then: one order-independent fit per chronological prefix,
  // over the same session set the current lens rates on.
  const trajectory = useMemo(
    () =>
      sessionOutcomes && !published?.seeds
        ? ratingTrajectory(
            visibleSessions(sessionOutcomes, showAll).filter(s => !published || s.perPolicy.some(p => p.policy_id === policyId) || s.task === policy?.environment),
            policyId as string
          )
        : [],
    [sessionOutcomes, showAll, policyId, published, policy?.environment]
  );

  const [rolloutsParam, setRolloutsParam] = useSearchParamNullable("rollouts");
  const rolloutsOpen = rolloutsParam !== null;
  const [failuresOpen, setFailuresOpen] = useState(false);

  const simStats = published?.seeds && sessionOutcomes ? computeArenaStats(sessionOutcomes, true) : null;
  const simWdl = simStats?.wdl.get(policyId);
  const simSuccess = simStats?.success.get(policyId);
  if (!policy) return null;

  return (
    <div className="px-6 py-5 bg-warm-50/30">
      {published && <div className="mb-5 text-sm text-ink"><h3 className="font-medium">Published {metric === "task_progress" ? "task progress" : "success rate"}: {(100 * published.rate).toFixed(1)}%</h3><p className="text-ink-muted">Interval {(100 * published.lo).toFixed(1)}–{(100 * published.hi).toFixed(1)}%. {published.seeds ? "Student-t 95% across five seeds." : "One standard error, matching the paper."}</p><p className="mt-2">{published.provenance}</p>{published.seeds && <ul className="mt-2 space-y-1">{published.seeds.map(s => <li key={s.seed}>Seed {s.seed}: {(100 * s.rate).toFixed(1)}% {s.dataUrl && <a className="text-teal hover:underline" href={s.dataUrl}>Per-state evidence ↗</a>}</li>)}</ul>}</div>}
      {simWdl && simSuccess && <p className="mb-5 text-sm text-ink-muted">
        {simSuccess.successes.toLocaleString()} successes in {simSuccess.rollouts.toLocaleString()} evaluations.
        {" "}{simWdl.wins.toLocaleString()} wins / {simWdl.draws.toLocaleString()} draws / {simWdl.losses.toLocaleString()} losses against all selected policies on the same grid and training seed.
        {" "}R0 aliases reuse the archived baseline outcomes. These comparisons are descriptive; the five-seed confidence interval above remains the reported uncertainty.
        {" "}<a className="text-teal hover:underline" href="/data/sim-statistics.json">Computed statistics and source checksums ↗</a>
      </p>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: info */}
        <div>
          <h3 className="text-sm font-medium text-ink mb-3">Details</h3>
          <dl className="space-y-2 text-xs">
            <div className="flex gap-2">
              <dt className="text-ink-muted w-24 shrink-0">Model ID</dt>
              <dd className="font-mono text-ink break-all">
                {policy.model_url ? (
                  <a
                    href={policy.model_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-teal hover:underline"
                  >
                    {policy.model_id}
                  </a>
                ) : (
                  policy.model_id
                )}
              </dd>
            </div>
            {policy.training_url && (
              <div className="flex gap-2">
                <dt className="text-ink-muted w-24 shrink-0">Training Run</dt>
                <dd>
                  <a
                    href={policy.training_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-teal hover:underline font-mono"
                  >
                    View run &rarr;
                  </a>
                </dd>
              </div>
            )}
            <div className="flex gap-2 items-center">
              <dt className="text-ink-muted w-24 shrink-0">Status</dt>
              <dd className="flex items-center gap-2">
                {policy.effective_status === "mainline" ? (
                  <span className="text-ink-muted">mainline</span>
                ) : (
                  <StatusBadge
                    status={policy.effective_status}
                    reason={policy.status_reason}
                  />
                )}
                {viewer?.isEditor && (
                  <StatusSelect
                    value={policy.status ?? "inherit"}
                    onChange={(v) =>
                      setPolicyStatus({ model_id: policy.model_id, status: v })
                    }
                  />
                )}
              </dd>
            </div>
          </dl>
          <div className="mt-3"><PolicyTagBadges policy={policy} /></div>
          {viewer?.isEditor && <PolicyTagEditor policy={policy} />}

          {/* Rating History (Bradley-Terry fit per chronological prefix) */}
          {trajectory.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium text-ink mb-2">
                Rating History
              </h3>
              <div className="flex items-end gap-1 h-12">
                {trajectory.map((entry) => {
                  const min = Math.min(...trajectory.map((e) => e.rating));
                  const max = Math.max(...trajectory.map((e) => e.rating));
                  const range = max - min || 1;
                  const height = 20 + ((entry.rating - min) / range) * 80;
                  return (
                    <div
                      key={entry.sessionId}
                      className="flex-1 bg-teal/30 rounded-t hover:bg-teal/50 transition-colors"
                      style={{ height: `${height}%` }}
                      title={`Rating ${Math.round(entry.rating)} — ${new Date(entry.creationTime).toLocaleDateString()}`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Success Rate History */}
          {successRateHistory && successRateHistory.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium text-ink mb-2">
                Success Rate History
              </h3>
              <div className="flex items-end gap-1 h-12">
                {successRateHistory.map((entry, i) => {
                  const pct = Math.round(entry.successRate * 100);
                  const height = 20 + entry.successRate * 80;
                  const repo = entry.datasetRepo.split("/").pop() ?? entry.datasetRepo;
                  return (
                    <div
                      key={i}
                      className={`flex-1 rounded-t transition-colors ${
                        entry.successRate >= 0.5
                          ? "bg-emerald-bar/30 hover:bg-emerald-bar/50"
                          : "bg-rose-bar/30 hover:bg-rose-bar/50"
                      }`}
                      style={{ height: `${height}%` }}
                      title={`${pct}% (${entry.successes}/${entry.total}) — ${repo}`}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right: recent sessions */}
        <div>
          <h3 className="text-sm font-medium text-ink mb-3">
            Recent Sessions
          </h3>
          {sessions === undefined ? (
            <div className="text-xs text-ink-muted">Loading...</div>
          ) : sessions.length === 0 ? (
            <div className="text-xs text-ink-muted">{published?.seeds ? "Simulation evaluation uses fixed grids. Per-seed outcomes are linked above; evaluation videos were not recorded." : "No sessions yet"}</div>
          ) : (
            <div className="space-y-2">
              {sessions.slice(0, 5).map((session) => (
                <a
                  key={session._id}
                  href={`?tab=sessions&session=${session._id}`}
                  className="block px-3 py-2 rounded-lg bg-white border border-warm-200 text-xs hover:border-teal/40 hover:shadow-sm transition-all"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-ink-muted">
                      {new Date(session._creationTime).toLocaleDateString()}
                    </span>
                    <div className="flex items-center gap-2">
                      <SessionModeTag mode={session.session_mode ?? "manual"} />
                      <span className="text-ink-muted font-mono">
                        {Number(session.num_rounds)} rounds
                      </span>
                    </div>
                  </div>
                  <span className="text-teal font-mono text-[11px]">
                    {session.dataset_repo}
                  </span>
                </a>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Rollout sections */}
      <div className="mt-5 pt-5 border-t border-warm-200 space-y-3">
        <CollapsibleSection
          title="Recent Rollouts"
          count={recentResults?.length}
          countColor="bg-teal-light text-teal"
          isOpen={rolloutsOpen}
          onToggle={() => setRolloutsParam(rolloutsOpen ? null : "1")}
        >
          <RolloutSection
            results={recentResults ?? []}
            isOpen={rolloutsOpen}
          />
        </CollapsibleSection>

        <CollapsibleSection
          title="Failures"
          count={failureResults?.length}
          countColor="bg-coral-light text-coral"
          isOpen={failuresOpen}
          onToggle={() => setFailuresOpen((o) => !o)}
        >
          <RolloutSection
            results={failureResults ?? []}
            isOpen={failuresOpen}
          />
        </CollapsibleSection>
      </div>
    </div>
  );
}

function SessionModeTag({ mode }: { mode: string }) {
  const styles: Record<string, string> = {
    manual: "bg-warm-100 text-ink-muted",
    "pool-sample": "bg-teal-light text-teal",
    calibrate: "bg-gold-light text-gold",
  };
  return (
    <span
      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${styles[mode] ?? "bg-warm-100 text-ink-muted"}`}
    >
      {mode}
    </span>
  );
}
