import { releaseLabel } from "../lib/releaseLabels";
import type { SessionOutcome } from "../lib/arenaRatings";
import { useState, useEffect } from "react";
import { useQuery } from "../lib/arenaClient";
import { api } from "../release/api";
import type { Id } from "../release/api";
import {
  fetchEpisodeSubset,
  getParquetCache,
  selectPrimaryCameraKey,
  type EpisodeMetadata,
} from "../lib/hf-api";
import {
  useSearchParam,
  useSearchParamNullable,
  clearSearchParams,
} from "../lib/useSearchParam";
import { RoundVideos } from "./RoundVideos";
import { roundVideoSpecs } from "../lib/roundVideoSpecs";
import { roundNumber } from "../lib/roundNumber";

function formatDate(timestamp: number): string {
  return new Date(timestamp).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
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

export default function Pairings({ gridOutcomes = [] }: { gridOutcomes?: SessionOutcome[] }) {
  const [selectedEnv, setSelectedEnv] = useSearchParam("env", "all");
  const [policyA, setPolicyARaw] = useSearchParam("policyA", "all");
  const [policyB, setPolicyBRaw] = useSearchParam("policyB", "all");
  const [expandedRound, setExpandedRound] = useSearchParamNullable("pRound");
  const [showParam] = useSearchParam("show", "mainline");
  const showAll = showParam === "all";

  const setPolicyA = (v: string) => {
    clearSearchParams("pRound");
    setPolicyARaw(v);
  };
  const setPolicyB = (v: string) => {
    clearSearchParams("pRound");
    setPolicyBRaw(v);
  };

  const envList = useQuery(api.policies.environmentsDetailed);
  const policyNames = useQuery(api.policies.listNames);

  const visibleEnvs = (envList ?? []).filter(
    (e) => showAll || e.status === "mainline"
  );
  const filteredPolicies = (policyNames ?? []).filter(
    (p) =>
      (showAll || p.effective_status === "mainline") &&
      (selectedEnv === "all" || p.environment === selectedEnv)
  );

  // Query rounds only when a specific policy A is selected
  const rounds = useQuery(
    api.pairings.listRounds,
    policyA !== "all"
      ? {
          policyIdA: policyA as Id<"policies">,
          ...(policyB !== "all"
            ? { policyIdB: policyB as Id<"policies"> }
            : {}),
        }
      : "skip"
  );

  const grid = gridOutcomes.find(s => s.perPolicy.some(p => p.policy_id === policyA));
  const gridPairs = grid?.pairs.filter(p => (p.a === policyA || p.b === policyA) &&
    (policyB === "all" || p.a === policyB || p.b === policyB));

  // Cache dataset info by repo
  type DsCacheEntry =
    | { status: "loading" }
    | { status: "loaded"; episodeMap: Map<number, Omit<EpisodeMetadata, "success">>; cameraKey: string }
    | { status: "error" };
  const [datasetCache, setDatasetCache] = useState<Map<string, DsCacheEntry>>(() => {
    // Initialize from module-level parquet cache
    const initial = new Map<string, DsCacheEntry>();
    for (const [repo, cached] of getParquetCache()) {
      const episodeMap = new Map<number, Omit<EpisodeMetadata, "success">>();
      for (const ep of cached.episodes) episodeMap.set(ep.episodeIndex, ep);
      const cameraKey = selectPrimaryCameraKey(cached.cameraKeys);
      initial.set(repo, { status: "loaded", episodeMap, cameraKey });
    }
    return initial;
  });

  // Load dataset info for all unique repos in the current rounds
  const uniqueRepos = rounds
    ? [...new Set(rounds.map((r) => r.datasetRepo))]
    : [];

  useEffect(() => {
    if (!rounds) return;
    for (const repo of uniqueRepos) {
      if (datasetCache.has(repo)) continue;
      // Publish the in-flight cache entry before starting its async load.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDatasetCache((prev) => new Map(prev).set(repo, { status: "loading" }));
      const neededIndices = new Set(
        rounds.filter((r) => r.datasetRepo === repo)
          .flatMap((r) => r.results.map((res) => res.episodeIndex))
      );
      fetchEpisodeSubset(repo, neededIndices)
        .then((info) => {
          const episodeMap = new Map<number, Omit<EpisodeMetadata, "success">>();
          for (const ep of info.episodes) {
            episodeMap.set(ep.episodeIndex, ep);
          }
          const cameraKey = selectPrimaryCameraKey(info.cameraKeys);
          setDatasetCache((prev) =>
            new Map(prev).set(repo, { status: "loaded", episodeMap, cameraKey })
          );
        })
        .catch(() => {
          setDatasetCache((prev) => new Map(prev).set(repo, { status: "error" }));
        });
    }
  }, [uniqueRepos.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      className="space-y-4"
      style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
    >
      {/* Filter bar */}
      <div className="flex items-center gap-4 flex-wrap">
        {/* Environment pills */}
        {visibleEnvs.length > 1 && (
          <div className="flex gap-2 flex-wrap">
            {["all", ...visibleEnvs.map((e) => e.environment)].map((env) => (
              <button
                key={env}
                onClick={() => {
                  setSelectedEnv(env);
                  setPolicyA("all");
                  setPolicyB("all");
                }}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-150 cursor-pointer border ${
                  selectedEnv === env
                    ? "bg-teal text-white border-teal shadow-sm"
                    : "bg-white text-ink-muted border-warm-200 hover:border-teal/40 hover:text-ink"
                }`}
              >
                {env === "all" ? "All Tasks" : env}
              </button>
            ))}
          </div>
        )}

        {/* Policy A dropdown (required) */}
        <select
          value={policyA}
          onChange={(e) => setPolicyA(e.target.value)}
          className="px-3 py-1.5 rounded-lg border border-warm-200 bg-white text-sm text-ink font-body cursor-pointer hover:border-warm-300 transition-colors"
        >
          <option value="all">Policy A</option>
          {filteredPolicies.map((p) => (
            <option key={p._id} value={p._id}>
              {p.name}
            </option>
          ))}
        </select>

        {/* Policy B dropdown (optional) */}
        <select
          value={policyB}
          onChange={(e) => setPolicyB(e.target.value)}
          className="px-3 py-1.5 rounded-lg border border-warm-200 bg-white text-sm text-ink font-body cursor-pointer hover:border-warm-300 transition-colors"
        >
          <option value="all">Any opponent</option>
          {filteredPolicies
            .filter((p) => (p._id as string) !== policyA)
            .map((p) => (
              <option key={p._id} value={p._id}>
                {p.name}
              </option>
            ))}
        </select>
      </div>

      {/* Content */}
      {policyA === "all" ? (
        <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted">
          Select a policy to view its head-to-head rounds.
        </div>
      ) : grid ? (
        <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-6 overflow-x-auto">
          <h2 className="font-display text-xl mb-2">Fixed-grid comparisons</h2>
          <p className="text-sm text-ink-muted mb-4">Binary success at matching initial states and training seeds, pooled over five seeds. Counts are from Policy A's perspective. Reused R0 baselines are aliases; these counts do not represent independent training runs. Simulation evaluation videos were not recorded.</p>
          {gridPairs?.length ? <table className="w-full text-sm text-left">
            <thead><tr><th className="pb-3">Opponent</th><th>Wins</th><th>Draws</th><th>Losses</th><th>Matched evaluations</th></tr></thead>
            <tbody>{gridPairs.map(p => {
              const opponent = p.a === policyA ? p.b : p.a;
              return <tr key={opponent} className="border-t border-warm-100">
                <td className="py-3 pr-4">{policyNames?.find(n => n._id === opponent)?.name}</td>
                <td>{(p.a === policyA ? p.winsA : p.winsB).toLocaleString()}</td>
                <td>{p.draws.toLocaleString()}</td>
                <td>{(p.a === policyA ? p.winsB : p.winsA).toLocaleString()}</td>
                <td>{(p.winsA + p.winsB + p.draws).toLocaleString()}</td>
              </tr>;
            })}</tbody>
          </table> : <p>No matched grid: these policies were evaluated on different initial-state grids or tasks.</p>}
          <a className="inline-block mt-4 text-sm text-teal" href="/data/sim-statistics.json">Computed statistics and source checksums ↗</a>
        </div>
      ) : rounds === undefined ? (
        <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
          <div className="flex items-center justify-center gap-3 text-ink-muted">
            <div className="w-5 h-5 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
            <span className="font-body">Loading rounds...</span>
          </div>
        </div>
      ) : rounds.length === 0 ? (
        <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted">
          No rounds found for the selected policies.
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-warm-200 shadow-sm overflow-hidden">
          {/* Header */}
          <div className="px-6 py-3 border-b border-warm-100 bg-warm-50 flex items-center justify-between">
            <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
              Rounds
            </span>
            <span className="font-mono text-xs text-ink-muted">
              {rounds.length} total
            </span>
          </div>

          {/* Rounds list */}
          {rounds.map((round, i) => {
            const roundKey = `${round.sessionId}:${round.roundIndex}`;
            const isExpanded = expandedRound === roundKey;
            const dsInfo = datasetCache.get(round.datasetRepo);
            const dsLoaded = dsInfo?.status === "loaded" ? dsInfo : null;

            return (
              <div
                key={roundKey}
                className={
                  i < rounds.length - 1 && !isExpanded
                    ? "border-b border-warm-100"
                    : ""
                }
              >
                <button
                  onClick={() =>
                    setExpandedRound(isExpanded ? null : roundKey)
                  }
                  className="w-full flex items-center gap-3 px-6 py-3 hover:bg-warm-50 transition-colors cursor-pointer text-left"
                  style={{
                    animation: `slide-in-right 0.4s ease-out ${0.35 + i * 0.02}s both`,
                  }}
                >
                  {/* Round index */}
                  <span className="text-xs font-mono text-ink-muted w-16 shrink-0">
                    {releaseLabel(round) ?? `Round ${roundNumber(round.roundIndex)}`}
                  </span>

                  {/* Pass/fail pills */}
                  <div className="flex gap-2 flex-wrap flex-1">
                    {round.results.map((result, j) => (
                      <span
                        key={j}
                        className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium ${
                          result.success
                            ? "bg-teal-light text-teal"
                            : "bg-coral-light text-coral"
                        }`}
                      >
                        {result.policyName}
                        <span className="text-[10px]">
                          {result.success ? "PASS" : "FAIL"}
                        </span>
                      </span>
                    ))}
                  </div>

                  {/* Session metadata */}
                  <span className="text-[11px] text-ink-muted/70 shrink-0">
                    {(round as { sessionLabel?: string }).sessionLabel ??
                      formatDate(round.sessionCreationTime)}
                  </span>
                  <SessionModeTag mode={round.sessionMode} />
                  <a
                    href={`https://huggingface.co/datasets/${round.datasetRepo}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-teal transition-colors font-mono text-[11px] text-ink-muted/60 shrink-0"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {round.datasetRepo.split("/").pop()}
                  </a>

                  {/* Chevron */}
                  {dsLoaded && (
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      className={`text-ink-muted/50 transition-transform duration-200 shrink-0 ${
                        isExpanded ? "rotate-90" : ""
                      }`}
                    >
                      <polyline points="9 18 15 12 9 6" />
                    </svg>
                  )}
                </button>

                {/* Expanded video */}
                {isExpanded && dsLoaded && (
                  <div className="px-6 pb-4">
                    <RoundVideos
                      videos={roundVideoSpecs(
                        round.results.map((r) => ({
                          policy_id: r.policyId,
                          policyName: r.policyName,
                          success: r.success,
                          episode_index: r.episodeIndex,
                        })),
                        round.datasetRepo,
                        dsLoaded.episodeMap,
                        dsLoaded.cameraKey,
                        null,
                      )}
                    />
                  </div>
                )}

                {isExpanded && dsInfo?.status === "error" && (
                  <p className="text-xs text-ink-muted px-6 pb-3">
                    Video previews unavailable (dataset not found on HuggingFace).
                  </p>
                )}

                {isExpanded && dsInfo?.status === "loading" && (
                  <div className="px-6 pb-3 flex items-center gap-2 text-ink-muted text-xs">
                    <div className="w-3 h-3 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
                    Loading video data...
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
