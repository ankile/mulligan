import type { Release } from "./release/types";
import { useMemo, useState } from "react";
import { useQuery } from "./lib/arenaClient";
import { api } from "./release/api";
import { computeArenaStats, visibleSessions } from "./lib/arenaRatings";

import ReleaseLinks from "./release/ReleaseLinks";
import { AppTabNavigation } from "./components/AppTabNavigation";
import DataExplorer from "./components/DataExplorer";
import CoverageDashboard from "./components/CoverageDashboard";
import EvalSessions from "./components/EvalSessions";
import Pairings from "./components/Pairings";
import PolicyDetail from "./components/PolicyDetail";
import { PolicyTagBadges } from "./components/PolicyTags";
import { matchesPolicyTags, policyTagOptions } from "../convex/policyTags";
import { StatusBadge } from "./components/StatusBadge";
import TaskStatusManager from "./components/TaskStatusManager";
import { useSearchParam, useSearchParamNullable, clearSearchParams } from "./lib/useSearchParam";

function winRate(wins: number, losses: number): number {
  const total = wins + losses;
  if (total === 0) return 0;
  return Math.round((wins / total) * 100);
}

const medalColors = [
  "from-amber-400 to-yellow-500",
  "from-gray-300 to-gray-400",
  "from-orange-400 to-amber-600",
];

function RankBadge({ rank }: { rank: number }) {
  if (rank <= 3) {
    return (
      <div
        className={`w-8 h-8 rounded-full bg-gradient-to-br ${medalColors[rank - 1]} flex items-center justify-center text-white font-body font-semibold text-sm shadow-sm`}
      >
        {rank}
      </div>
    );
  }
  return (
    <div className="w-8 h-8 rounded-full bg-warm-100 flex items-center justify-center text-ink-muted font-mono text-sm">
      {rank}
    </div>
  );
}

function WinRateBar({ wins, losses }: { wins: number; losses: number }) {
  const rate = winRate(wins, losses);
  return (
    <div data-testid="policy-winrate" className="flex items-center gap-3">
      <div className="w-20 h-1.5 rounded-full bg-warm-100 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${rate}%`,
            backgroundColor:
              rate >= 50 ? "var(--color-emerald-bar)" : "var(--color-rose-bar)",
          }}
        />
      </div>
      <span className="font-mono text-xs text-ink-muted w-8">{rate}%</span>
    </div>
  );
}

function EnvironmentTag({ env }: { env: string }) {
  const colors: Record<string, string> = {
    franka_pick_cube: "bg-teal-light text-teal",
    PushT: "bg-teal-light text-teal",
    BimanualInsertion: "bg-coral-light text-coral",
    PickAndPlace: "bg-gold-light text-gold",
  };
  return (
    <span
      className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-body font-medium ${colors[env] ?? "bg-warm-100 text-ink-muted"}`}
    >
      {env}
    </span>
  );
}

type SortKey = "elo" | "success" | "winRate" | "avgSuccessSteps" | "roundMethod";
function App({ release }: { release?: Release }) {
  const [activeTab] = useSearchParam("tab", "leaderboard");
  const [explorerView] = useSearchParam("view", "explorer");
  const [selectedEnv, setSelectedEnv] = useSearchParam("env", "all");
  const [expandedPolicy, setExpandedPolicy] = useSearchParamNullable("policy");
  const [sortBy, setSortBy] = useSearchParam("sort", release ? "roundMethod" : "elo") as [SortKey, (v: string) => void];
  // Global mainline/all lens — deliberately NOT cleared on tab switch.
  const [showParam, setShowParam] = useSearchParam("show", "mainline");
  const showAll = !release && showParam === "all";
  const [managerOpen, setManagerOpen] = useState(false);
  const [roundFilter, setRoundFilter] = useSearchParam("round", "");
  const [methodFilter, setMethodFilter] = useSearchParam("method", "");
  const [tagFilter, setTagFilter] = useSearchParam("tag", "");

  const viewer = useQuery(api.users.viewer);
  const envList = useQuery(api.policies.environmentsDetailed);
  const policies = useQuery(
    api.policies.leaderboard,
    selectedEnv === "all" ? {} : { environment: selectedEnv },
  );
  const sessionOutcomes = useQuery(api.ratings.sessionOutcomes);

  // Bradley-Terry ratings + W/D/L + success stats fit LIVE over exactly the
  // sessions the current lens shows — retagging a session re-rates instantly.
  const arenaStats = useMemo(
    () =>
      sessionOutcomes
        ? computeArenaStats(visibleSessions(sessionOutcomes, showAll), !!release)
        : null,
    [sessionOutcomes, showAll, release],
  );


  const visibleEnvs = (envList ?? []).filter(
    (e) => showAll || e.status === "mainline"
  );
  const lensPolicies = (policies ?? []).filter((p) => showAll || p.effective_status === "mainline");
  const tagOptions = policyTagOptions(lensPolicies);
  const visiblePolicies = lensPolicies
    .filter((p) => matchesPolicyTags(p, roundFilter, methodFilter, tagFilter))
    .map((p) => {
      const id = p._id as string;
      const rating = arenaStats?.ratings.get(id) ?? null;
      const wdl = arenaStats?.wdl.get(id) ?? { wins: 0, draws: 0, losses: 0 };
      const succ = arenaStats?.success.get(id) ?? null;
      const published = release?.tasks.flatMap(t => t.policies).find(p => p.id === id);
      const metric = release?.tasks.find(t => t.policies.some(p => p.id === id))?.metric;
      return {
        ...p,
        published, metric,
        rating,
        wdl,
        successRate:
          published ? published.rate : succ && succ.rollouts > 0 ? succ.successes / succ.rollouts : null,
        totalRollouts: succ?.rollouts ?? 0,
        totalSuccesses: succ?.successes ?? 0,
        avgSuccessSteps: succ?.avgSuccessSteps ?? null,
      };
    });

  const sortedPolicies = [...visiblePolicies].sort((a, b) => {
    if (sortBy === "roundMethod") {
      if (a.round !== b.round) {
        return (a.round ?? Infinity) - (b.round ?? Infinity);
      }
      if (a.method !== b.method) {
        if (!a.method) return 1;
        if (!b.method) return -1;
        return a.method.localeCompare(b.method);
      }
      return a.name.localeCompare(b.name, undefined, { numeric: true });
    }
    if (sortBy === "success") {
      return (b.successRate ?? -1) - (a.successRate ?? -1);
    }
    if (sortBy === "winRate") {
      return winRate(b.wdl.wins, b.wdl.losses) - winRate(a.wdl.wins, a.wdl.losses);
    }
    if (sortBy === "avgSuccessSteps") {
      const aSteps = a.avgSuccessSteps ?? Number.POSITIVE_INFINITY;
      const bSteps = b.avgSuccessSteps ?? Number.POSITIVE_INFINITY;
      return aSteps - bSteps;
    }
    return (b.rating ?? -Infinity) - (a.rating ?? -Infinity);
  });
  const maxRating = sortedPolicies.reduce<number | null>(
    (best, p) => (p.rating != null && (best === null || p.rating > best) ? p.rating : best),
    null,
  );


  return (
    <div className="min-h-screen bg-cream font-body text-ink">
      {/* Subtle top accent line */}
      <div className="h-1 bg-gradient-to-r from-teal via-gold to-coral" />

      <div className={`${activeTab === "explorer" && explorerView === "stage" ? "max-w-[1800px]" : "max-w-6xl"} mx-auto px-6 py-16`}>
        {/* Header */}
        <header
          className="mb-14"
          style={{ animation: "fade-up 0.6s ease-out both" }}
        >
          <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
            <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-teal to-teal/70 flex items-center justify-center">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 8V4H8" />
                <rect x="4" y="8" width="16" height="12" rx="2" />
                <path d="M2 14h2" />
                <path d="M20 14h2" />
                <path d="M9 13v2" />
                <path d="M15 13v2" />
              </svg>
            </div>
            <h1 className="font-display text-4xl text-ink tracking-tight">
              Policy Arena
            </h1>
            </div>
            <ReleaseLinks />
          </div>
          <p className="text-ink-muted font-body text-lg ml-[52px]">
            {release ? "Mulligan · Mainline policies, evaluations, and datasets" : "Track, compare, and rank robot learning policies and datasets"}
          </p>
        </header>

        {/* Tab navigation + mainline/all lens */}
        <div
          className="flex items-center justify-between gap-3 mb-8 flex-wrap"
          style={{ animation: "fade-up 0.6s ease-out 0.1s both" }}
        >
          <AppTabNavigation activeTab={activeTab} />
          <div className="flex items-center gap-2">
            {!release && <div className="flex gap-1 bg-warm-100 rounded-xl p-1">
              {(
                [
                  { id: "mainline", label: "Mainline" },
                  { id: "all", label: "All" },
                ] as const
              ).map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setShowParam(opt.id)}
                  className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 cursor-pointer ${
                    showParam === opt.id
                      ? "bg-white text-ink shadow-sm"
                      : "text-ink-muted hover:text-ink"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>}
            {viewer?.isEditor && (
              <button
                onClick={() => setManagerOpen((o) => !o)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer ${
                  managerOpen
                    ? "bg-teal text-white border-teal shadow-sm"
                    : "bg-white text-ink-muted border-warm-200 hover:border-teal/40 hover:text-ink"
                }`}
              >
                Task statuses
              </button>
            )}
          </div>
        </div>

        {managerOpen && viewer?.isEditor && <TaskStatusManager />}

        {activeTab === "leaderboard" && (
          <>
            {/* Environment filter */}
            {visibleEnvs.length > 1 && (
              <div
                className="flex gap-2 mb-6 flex-wrap"
                style={{ animation: "fade-up 0.6s ease-out 0.12s both" }}
              >
                {["all", ...visibleEnvs.map((e) => e.environment)].map((env) => (
                  <button
                    key={env}
                    onClick={() => setSelectedEnv(env)}
                    className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-150 cursor-pointer border ${
                      selectedEnv === env
                        ? "bg-teal text-white border-teal shadow-sm"
                        : "bg-white text-ink-muted border-warm-200 hover:border-teal/40 hover:text-ink"
                    }`}
                  >
                    {env === "all" ? "All Tasks" : release?.tasks.find(t => t.id === env)?.title ?? env}
                  </button>
                ))}
              </div>
            )}

            <div className="flex flex-wrap items-end gap-3 mb-6">
              {[
                { label: "Round", value: roundFilter, set: setRoundFilter, options: tagOptions.rounds.map((r) => ({ value: String(r), label: `Round ${r}` })), missing: true },
                { label: "Method", value: methodFilter, set: setMethodFilter, options: tagOptions.methods.map((m) => ({ value: m, label: m })), missing: true },
                { label: "Tag", value: tagFilter, set: setTagFilter, options: tagOptions.tags.map((t) => ({ value: t, label: t })), missing: false },
              ].map((filter) => <label key={filter.label} className="text-xs text-ink-muted">
                {filter.label}
                <select aria-label={filter.label} value={filter.value} onChange={(e) => filter.set(e.target.value)} className="block mt-1 rounded-lg border border-warm-200 bg-white px-3 py-2 text-sm text-ink max-w-72">
                  <option value="">All {filter.label.toLowerCase()}s</option>
                  {filter.missing && <option value="untagged">Unassigned</option>}
                  {filter.value && filter.value !== "untagged" && !filter.options.some((o) => o.value === filter.value) && <option value={filter.value}>{filter.value}</option>}
                  {filter.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </label>)}
              <label className="text-xs text-ink-muted">
                Sort by
                <select aria-label="Sort by" value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="block mt-1 rounded-lg border border-warm-200 bg-white px-3 py-2 text-sm text-ink">
                  <option value="elo">Rating</option>
                  <option value="roundMethod">Round, then method</option>
                  <option value="success">Success rate</option>
                  <option value="winRate">Win rate</option>
                  <option value="avgSuccessSteps">Average steps</option>
                </select>
              </label>
              {(roundFilter || methodFilter || tagFilter) && <button onClick={() => clearSearchParams("round", "method", "tag")} className="text-xs text-teal py-2 cursor-pointer">Clear tag filters</button>}
              <p className="w-full text-xs text-ink-muted">{release ? "Frozen mainline release. Ratings compare full success within each task and matching evaluation grid. Separate groups are not a common ranking. Square-Narrow R0 uses a different grid from R1–R3. W/D/L compares all selected opponents on the same grid, including other rounds; win rate excludes draws. Avg steps counts successful episodes only. Routing reports task progress; simulation reports five-seed means. Filters select rows, not opponents." : "Tag filters select policies. Ratings use all comparisons in the current Mainline or All view."}</p>
            </div>

            {/* Stats summary */}
            <div
              className="grid grid-cols-3 gap-4 mb-10"
              style={{ animation: "fade-up 0.6s ease-out 0.15s both" }}
            >
              {[
                {
                  label: "Policies",
                  value: sortedPolicies.length.toString(),
                },
                {
                  label: "Top Rating",
                  value: maxRating != null ? Math.round(maxRating).toString() : "—",
                },
                {
                  label: "Comparisons",
                  value: Math.round(lensPolicies
                    .reduce(
                      (a, p) => { const wdl = arenaStats?.wdl.get(p._id); return a + (wdl ? wdl.wins + wdl.losses + wdl.draws : 0); },
                      0
                    ) / 2).toLocaleString(),
                },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="bg-white rounded-xl border border-warm-200 px-5 py-4"
                >
                  <div className="text-xs uppercase tracking-widest text-ink-muted font-medium mb-1">
                    {stat.label}
                  </div>
                  <div className="font-display text-2xl text-ink">
                    {stat.value}
                  </div>
                </div>
              ))}
            </div>

            {/* Leaderboard */}
            {policies === undefined || sessionOutcomes === undefined ? (
              <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
                <div className="flex items-center justify-center gap-3 text-ink-muted">
                  <div className="w-5 h-5 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
                  <span className="font-body">Loading leaderboard...</span>
                </div>
              </div>
            ) : sortedPolicies.length === 0 ? (
              <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted">
                {lensPolicies.length ? "No policies match these tags. Clear the tag filters to see all policies." : "No policies registered yet. Submit an eval session to get started."}
              </div>
            ) : (
              <div
                className="bg-white rounded-2xl border border-warm-200 shadow-sm overflow-x-auto"
                style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
              >
                {/* Table header */}
                <div className="min-w-[950px] grid grid-cols-[56px_1fr_80px_170px_140px_100px_90px] px-6 py-3.5 border-b border-warm-100 bg-warm-50">
                  <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
                    #
                  </span>
                  <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
                    Policy
                  </span>
                  <button
                    onClick={() => setSortBy("elo")}
                    className={`text-[11px] uppercase tracking-widest font-medium cursor-pointer ${sortBy === "elo" ? "text-teal" : "text-ink-muted hover:text-ink"}`}
                  >
                    Rating {sortBy === "elo" && "▼"}
                  </button>
                  <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
                    W / D / L
                  </span>
                  <button
                    onClick={() => setSortBy("winRate")}
                    className={`text-[11px] uppercase tracking-widest font-medium cursor-pointer ${sortBy === "winRate" ? "text-teal" : "text-ink-muted hover:text-ink"}`}
                  >
                    Win Rate {sortBy === "winRate" && "▼"}
                  </button>
                  <button
                    onClick={() => setSortBy("success")}
                    className={`text-[11px] uppercase tracking-widest font-medium cursor-pointer ${sortBy === "success" ? "text-teal" : "text-ink-muted hover:text-ink"}`}
                  >
                    {release ? "Outcome" : "Success"} {sortBy === "success" && "▼"}
                  </button>
                  <button
                    onClick={() => setSortBy("avgSuccessSteps")}
                    className={`text-[11px] uppercase tracking-widest font-medium cursor-pointer ${sortBy === "avgSuccessSteps" ? "text-teal" : "text-ink-muted hover:text-ink"}`}
                  >
                    Avg Steps {sortBy === "avgSuccessSteps" && "▲"}
                  </button>
                </div>

                {/* Rows */}
                {sortedPolicies.map(
                  (policy, i) => (
                    <div key={policy._id}>
                      <div
                        data-policy-id={policy._id}
                        className={`min-w-[950px] grid grid-cols-[56px_1fr_80px_170px_140px_100px_90px] items-center px-6 py-4 transition-colors duration-150 hover:bg-warm-50 cursor-pointer ${
                          i < sortedPolicies.length - 1 &&
                          expandedPolicy !== (policy._id as string)
                            ? "border-b border-warm-100"
                            : ""
                        }`}
                        style={{
                          animation: `slide-in-right 0.4s ease-out ${0.35 + i * 0.04}s both`,
                        }}
                        onClick={() => {
                          clearSearchParams("rollouts");
                          setExpandedPolicy(
                            expandedPolicy === (policy._id as string)
                              ? null
                              : (policy._id as string)
                          );
                        }}
                      >
                        {/* Rank */}
                        <div>
                          <RankBadge rank={i + 1} />
                        </div>

                        {/* Name + Environment */}
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="min-w-0 flex-1">
                            <div className="font-body font-semibold text-ink text-[15px] truncate" title={policy.name}>
                              {policy.name}
                            </div>
                            <div className="mt-1 flex flex-wrap items-center gap-1.5">
                              <EnvironmentTag env={policy.environment} />
                              {showAll && (
                                <StatusBadge
                                  status={policy.effective_status}
                                  reason={policy.status_reason}
                                />
                              )}
                            </div>
                            <div className="mt-1.5"><PolicyTagBadges policy={policy} /></div>
                          </div>
                          <svg
                            width="12"
                            height="12"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            className={`text-ink-muted/50 transition-transform duration-200 flex-shrink-0 ${
                              expandedPolicy === (policy._id as string)
                                ? "rotate-90"
                                : ""
                            }`}
                          >
                            <polyline points="9 18 15 12 9 6" />
                          </svg>
                        </div>

                        {/* Rating */}
                        <div data-testid="policy-rating" className="font-mono text-sm font-medium text-ink">
                          {policy.rating != null ? Math.round(policy.rating) : "—"}
                        </div>

                        {/* W / D / L */}
                        <div data-testid="policy-wdl" title={`${policy.wdl.wins.toLocaleString()} wins / ${policy.wdl.draws.toLocaleString()} draws / ${policy.wdl.losses.toLocaleString()} losses`} className="font-mono text-xs text-ink-muted whitespace-nowrap">
                          <span className="text-emerald-bar">
                            {new Intl.NumberFormat("en", {notation: "compact", maximumFractionDigits: 1}).format(policy.wdl.wins)}
                          </span>
                          <span className="text-warm-300 mx-1">/</span>
                          <span className="text-ink-muted">
                            {new Intl.NumberFormat("en", {notation: "compact", maximumFractionDigits: 1}).format(policy.wdl.draws)}
                          </span>
                          <span className="text-warm-300 mx-1">/</span>
                          <span className="text-rose-bar">
                            {new Intl.NumberFormat("en", {notation: "compact", maximumFractionDigits: 1}).format(policy.wdl.losses)}
                          </span>
                        </div>

                        {/* Win Rate */}
                        <WinRateBar
                          wins={policy.wdl.wins}
                          losses={policy.wdl.losses}
                        />

                        {/* Success Rate */}
                        <div
                          className="font-mono text-sm text-ink-muted"
                          title={policy.successRate != null
                            ? `${policy.totalSuccesses} of ${policy.totalRollouts} rollouts succeeded`
                            : undefined}
                        >
                          {policy.successRate != null
                            ? <>
                                {release ? (policy.successRate * 100).toFixed(1) : Math.round(policy.successRate * 100)}%
                                <span className="text-xs text-ink-muted/60 ml-1">
                                  ({policy.published?.seeds ? "5 seeds" : policy.totalRollouts})
                                </span>
                              </>
                            : "—"}
                        </div>

                        {/* Avg Steps */}
                        <div data-testid="policy-steps" title="Mean control steps in successful episodes" className="font-mono text-sm text-ink-muted">
                          {policy.avgSuccessSteps != null
                            ? policy.avgSuccessSteps
                            : "—"}
                        </div>
                      </div>

                      {/* Expanded detail */}
                      {expandedPolicy === (policy._id as string) && (
                        <div className="border-b border-warm-100">
                          <PolicyDetail policyId={policy._id} published={policy.published} metric={policy.metric} />
                        </div>
                      )}
                    </div>
                  )
                )}
              </div>
            )}
          </>
        )}

        {activeTab === "sessions" && <EvalSessions />}

        {activeTab === "pairings" && <Pairings gridOutcomes={release ? sessionOutcomes?.filter(s => s.session_mode === "fixed_grid") : undefined} />}

        {activeTab === "explorer" && <DataExplorer readOnly={Boolean(release)} />}
        {activeTab === "coverage" && <CoverageDashboard readOnly={Boolean(release)} />}

        {/* Footer */}
        <footer
          className="mt-8 text-center text-xs text-ink-muted/60"
          style={{ animation: "fade-up 0.6s ease-out 0.9s both" }}
        >
          {release ? "Mulligan release · Read-only · Outcomes and dataset revisions are frozen" : "Bradley-Terry ratings fit live from pairwise policy evaluations in the current view"}
        </footer>
      </div>
    </div>
  );
}

export default App;
