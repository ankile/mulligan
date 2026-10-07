import { releaseLabel } from "../lib/releaseLabels";
import { useState, useEffect, useMemo } from "react";
import { useQuery, useMutation } from "../lib/arenaClient";
import { api } from "../release/api";
import { StatusBadge, StatusSelect } from "./StatusBadge";
import type { Id } from "../release/api";
import {
  fetchEpisodeSubset,
  getParquetCache,
  selectPrimaryCameraKey,
  type EpisodeMetadata,
} from "../lib/hf-api";
import { useSearchParam, useSearchParamNullable, useSearchParamNumber, clearSearchParams } from "../lib/useSearchParam";
import {
  formatIdList,
  parseIdList,
  sessionLetter,
  toggleId,
} from "../lib/joinSessions";
import JoinedSessions, { type SessionListRow } from "./JoinedSessions";
import { TaskFilterChips } from "./TaskFilterChips";
import { RoundVideos } from "./RoundVideos";
import { roundVideoSpecs } from "../lib/roundVideoSpecs";
import {
  ArmLegend,
  ExpandAllButton,
  PairedTestsDrawer,
  PolicyStatCards,
  RoundsGrid,
} from "./RoundOutcomes";
import { armLabels, armsFromPolicies } from "../lib/armStats";

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
    rollout: "bg-purple-100 text-purple-700",
  };
  return (
    <span
      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${styles[mode] ?? "bg-warm-100 text-ink-muted"}`}
    >
      {mode}
    </span>
  );
}

function SessionDetail({ sessionId }: { sessionId: Id<"evalSessions"> }) {
  const detail = useQuery(api.evalSessions.getDetail, { id: sessionId });
  const [expandedRound, setExpandedRound] = useSearchParamNumber("round");
  const [expandAll, setExpandAll] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState<{
    episodeMap: Map<number, Omit<EpisodeMetadata, "success">>;
    cameraKey: string;
  } | null>(() => {
    // Initialize from module-level cache if available
    if (!detail) return null;
    const cached = getParquetCache().get(detail.dataset_repo);
    if (!cached) return null;
    const episodeMap = new Map<number, Omit<EpisodeMetadata, "success">>();
    for (const ep of cached.episodes) episodeMap.set(ep.episodeIndex, ep);
    const cameraKey = selectPrimaryCameraKey(cached.cameraKeys);
    return { episodeMap, cameraKey };
  });
  const [datasetError, setDatasetError] = useState(false);

  useEffect(() => {
    if (!detail || datasetInfo) return;
    const neededIndices = new Set(
      detail.rounds.flatMap((r) => r.results.map((res) => res.episode_index))
    );
    fetchEpisodeSubset(detail.dataset_repo, neededIndices)
      .then((info) => {
        const episodeMap = new Map<number, Omit<EpisodeMetadata, "success">>();
        for (const ep of info.episodes) {
          episodeMap.set(ep.episodeIndex, ep);
        }
        const cameraKey = selectPrimaryCameraKey(info.cameraKeys);
        setDatasetInfo({ episodeMap, cameraKey });
      })
      .catch(() => {
        setDatasetError(true);
      });
  }, [detail?.dataset_repo]); // eslint-disable-line react-hooks/exhaustive-deps

  // Stable arm list so the paired-tests drawer only recomputes on new data.
  const arms = useMemo(
    () => (detail ? armsFromPolicies(detail.policies, detail.rounds) : []),
    [detail],
  );

  if (!detail) {
    return (
      <div className="px-6 py-4">
        <div className="flex items-center gap-2 text-ink-muted text-sm">
          <div className="w-4 h-4 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
          Loading round details...
        </div>
      </div>
    );
  }

  return (
    <div className="px-6 pb-5">
      <ArmLegend arms={arms} />
      <PolicyStatCards arms={arms} rounds={detail.rounds} maxMarks={detail.max_subtask_marks} />
      <PairedTestsDrawer arms={arms} rounds={detail.rounds} maxMarks={detail.max_subtask_marks} />

      <div className="flex items-center justify-between gap-3 mb-2">
        <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
          Rounds
        </span>
        {datasetInfo && (
          <ExpandAllButton expandAll={expandAll} onToggle={() => setExpandAll((v) => !v)} />
        )}
      </div>
      <RoundsGrid
        arms={arms}
        rounds={detail.rounds}
        maxMarks={detail.max_subtask_marks}
        expandedRound={expandedRound}
        expandAll={expandAll}
        canExpand={datasetInfo !== null}
        onToggleRound={setExpandedRound}
        renderExpanded={(round) =>
          datasetInfo && (
            <RoundVideos
              videos={roundVideoSpecs(
                round.results,
                detail.dataset_repo,
                datasetInfo.episodeMap,
                datasetInfo.cameraKey,
                armLabels(arms),
                detail.max_subtask_marks,
              )}
            />
          )
        }
      />

      {datasetError && (
        <p className="text-xs text-ink-muted mt-3">
          Video previews unavailable (dataset not found on HuggingFace).
        </p>
      )}
    </div>
  );
}

type SessionModeFilter = "all" | "manual" | "pool-sample" | "calibrate" | "rollout";

const SESSION_MODE_FILTERS: { id: SessionModeFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "manual", label: "Manual" },
  { id: "pool-sample", label: "Pool Sample" },
  { id: "calibrate", label: "Calibrate" },
  { id: "rollout", label: "Rollout" },
];

export default function EvalSessions() {
  const sessions = useQuery(api.evalSessions.list);
  const viewer = useQuery(api.users.viewer);
  const setSessionStatus = useMutation(api.evalSessions.setStatus);
  const setSessionOperator = useMutation(api.evalSessions.setOperator);
  const operators = useQuery(api.operators.list);
  const [expandedSession, setExpandedSessionRaw] = useSearchParamNullable("session");
  const [modeFilter, setModeFilter] = useSearchParam("mode", "all");
  const [taskFilter, setTaskFilter] = useSearchParam("task", "all");
  const [showParam] = useSearchParam("show", "mainline");
  const showAll = showParam === "all";

  // "Join" selection: an ordered list of session ids (order = A/B/C letters)
  // in ?join=; ?view=join swaps the list for the round-aligned joined view.
  const [joinParam, setJoinParam] = useSearchParamNullable("join");
  const [view, setView] = useSearchParam("view", "list");
  const joinIds = parseIdList(joinParam);
  const joinedView = view === "join" && joinIds.length >= 2;
  const toggleJoin = (id: string) => {
    const next = toggleId(joinIds, id);
    if (next.length < 2 && view === "join") setView("list");
    setJoinParam(formatIdList(next));
  };
  const clearJoin = () => {
    if (view === "join") setView("list");
    clearSearchParams("hide");
    setJoinParam(null);
  };

  const setExpandedSession = (id: string | null) => {
    if (id === null) clearSearchParams("round");
    setExpandedSessionRaw(id);
  };

  if (sessions === undefined) {
    return (
      <div
        className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8"
        style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
      >
        <div className="flex items-center justify-center gap-3 text-ink-muted">
          <div className="w-5 h-5 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
          <span className="font-body">Loading eval sessions...</span>
        </div>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div
        className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted"
        style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
      >
        No eval sessions yet. Submit one via the Python client to get started.
      </div>
    );
  }

  const sessionById = new Map<string, SessionListRow>(
    sessions.map((s) => [s._id as string, s]),
  );

  // Mainline/all lens applies before every other filter and count.
  const statusVisible = showAll
    ? sessions
    : sessions.filter((s) => s.effective_status === "mainline");

  const modeFiltered =
    modeFilter === "all"
      ? statusVisible
      : statusVisible.filter((s) => (s.session_mode ?? "manual") === modeFilter);

  const filteredSessions =
    taskFilter === "all"
      ? modeFiltered
      : modeFiltered.filter((s) => s.task === taskFilter);

  // Count sessions per mode for the filter badges
  const modeCounts = new Map<string, number>();
  for (const s of statusVisible) {
    const mode = s.session_mode ?? "manual";
    modeCounts.set(mode, (modeCounts.get(mode) ?? 0) + 1);
  }

  // Unique tasks for filter pills (from mode-filtered sessions); ordering is
  // done by TaskFilterChips (mainline first).
  const allTasks = modeFiltered.map((s) => s.task).filter(Boolean) as string[];

  return (
    <div
      className="space-y-4"
      style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
    >
      {/* Join bar: selected sessions (A, B, ...) + enter/leave the joined view */}
      {joinIds.length > 0 && (
        <div className="bg-white rounded-2xl border border-gold/40 shadow-sm px-5 py-3 flex items-center gap-3 flex-wrap">
          <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
            Join sessions
          </span>
          {joinIds.map((id, i) => {
            const row = sessionById.get(id);
            return (
              <span
                key={id}
                className="inline-flex items-center gap-1.5 rounded-lg bg-warm-100 px-2 py-1 text-xs"
              >
                <span className="w-4 h-4 rounded bg-gold text-white text-[10px] font-mono font-semibold flex items-center justify-center">
                  {sessionLetter(i)}
                </span>
                <span className="font-mono text-ink-light">
                  {row ? (releaseLabel(row) ?? row.policyNames.join(" vs ")) : id}
                </span>
                {row && (
                  <span className="text-ink-muted">{formatDate(row._creationTime)}</span>
                )}
                <button
                  onClick={() => toggleJoin(id)}
                  className="text-ink-muted hover:text-coral cursor-pointer ml-1 font-mono"
                  title="Remove from join"
                >
                  ×
                </button>
              </span>
            );
          })}
          {joinIds.length < 2 ? (
            <span className="text-xs text-ink-muted">
              Pick at least one more session to join.
            </span>
          ) : (
            <button
              onClick={() => setView(joinedView ? "list" : "join")}
              className="px-3 py-1.5 rounded-lg bg-teal text-white text-xs font-medium cursor-pointer hover:bg-teal/90 transition-colors"
            >
              {joinedView ? "← Back to session list" : "View joined rollouts →"}
            </button>
          )}
          <button
            onClick={clearJoin}
            className="text-xs text-ink-muted hover:text-ink cursor-pointer ml-auto"
          >
            Clear
          </button>
        </div>
      )}

      {joinedView ? (
        <JoinedSessions sessionIds={joinIds} sessionRows={sessionById} />
      ) : (
        <>
      {/* Filter bars */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Mode filter */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium mr-1">
            Mode
          </span>
          {SESSION_MODE_FILTERS.map((filter) => {
            const count =
              filter.id === "all"
                ? statusVisible.length
                : modeCounts.get(filter.id) ?? 0;
            const isActive = modeFilter === filter.id;
            return (
              <button
                key={filter.id}
                onClick={() => setModeFilter(filter.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                  isActive
                    ? "bg-teal text-white shadow-sm"
                    : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300 hover:text-ink"
                }`}
              >
                {filter.label}
                <span
                  className={`font-mono text-[10px] ${
                    isActive ? "text-white/70" : "text-ink-muted/60"
                  }`}
                >
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        {/* Task filter */}
        <TaskFilterChips tasks={allTasks} value={taskFilter} onChange={setTaskFilter} />
      </div>

      {filteredSessions.length === 0 ? (
        <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted">
          No {modeFilter} sessions found.
        </div>
      ) : (
        filteredSessions.map((session) => (
        <div
          key={session._id}
          className="bg-white rounded-2xl border border-warm-200 shadow-sm overflow-hidden"
        >
          {/* Session header */}
          <button
            onClick={() =>
              setExpandedSession(
                expandedSession === (session._id as string) ? null : (session._id as string)
              )
            }
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-warm-50/50 transition-colors cursor-pointer text-left"
          >
            <div className="min-w-0">
              <div className="flex flex-wrap gap-1.5 mb-1.5">
                {session.policyNames.map((name, i) => (
                  <span
                    key={i}
                    className="rounded bg-warm-100 px-2 py-0.5 text-xs font-mono text-ink-light"
                  >
                    {name}
                  </span>
                ))}
              </div>
              <div className="flex items-center gap-3 text-xs text-ink-muted">
                {releaseLabel(session) && (
                  <span className="font-medium text-ink-light whitespace-nowrap">{releaseLabel(session)}</span>
                )}
                <span>{formatDate(session._creationTime)}</span>
                <span className="inline-block px-2 py-0.5 rounded-full bg-warm-100 text-ink-muted text-[11px] font-mono">
                  {Number(session.num_rounds)} rounds
                </span>
                <SessionModeTag mode={session.session_mode ?? "manual"} />
                <StatusBadge
                  status={session.effective_status}
                  reason={session.status_reason}
                />
                {session.operator && (
                  <span
                    className="inline-block px-2 py-0.5 rounded-full bg-warm-100 text-ink-muted text-[11px] font-mono"
                    title="Operator — who physically ran this eval"
                  >
                    op: {session.operator}
                  </span>
                )}
                {(() => {
                  const pos = joinIds.indexOf(session._id as string);
                  const joined = pos >= 0;
                  return (
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleJoin(session._id as string);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          e.stopPropagation();
                          toggleJoin(session._id as string);
                        }
                      }}
                      className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-mono transition-colors cursor-pointer ${
                        joined
                          ? "bg-gold text-white"
                          : "bg-warm-100 text-ink-muted hover:bg-gold-light hover:text-gold"
                      }`}
                      title="Join this session with another to view their rollouts side by side, round by round"
                    >
                      {joined ? `joined · ${sessionLetter(pos)}` : "+ join"}
                    </span>
                  );
                })()}
                <a
                  href={`?tab=explorer&dataset=${encodeURIComponent(session.dataset_repo)}`}
                  className="hover:text-teal transition-colors font-mono"
                  onClick={(e) => e.stopPropagation()}
                >
                  Data Explorer &rarr;
                </a>
                <a
                  href={`https://huggingface.co/datasets/${session.dataset_repo}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-teal transition-colors font-mono"
                  onClick={(e) => e.stopPropagation()}
                >
                  HuggingFace &rarr;
                </a>
                {session.derivedDatasetRepos?.map((repo) => (
                  <a
                    key={repo}
                    href={`?tab=explorer&dataset=${encodeURIComponent(repo)}`}
                    className="hover:text-teal transition-colors font-mono"
                    onClick={(e) => e.stopPropagation()}
                  >
                    Training view &rarr;
                  </a>
                ))}
              </div>
            </div>
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className={`text-ink-muted transition-transform duration-200 ${
                expandedSession === (session._id as string) ? "rotate-90" : ""
              }`}
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>

          {/* Notes */}
          {session.notes && expandedSession === (session._id as string) && (
            <div className="px-6 pb-2">
              <p className="text-xs text-ink-muted italic">{session.notes}</p>
            </div>
          )}

          {/* Editor status override */}
          {viewer?.isEditor && expandedSession === (session._id as string) && (
            <div className="px-6 pb-3 flex items-center gap-2 text-xs text-ink-muted">
              <span className="uppercase tracking-widest text-[10px] font-medium">
                Status
              </span>
              <StatusSelect
                value={session.status ?? "inherit"}
                onChange={(v) =>
                  setSessionStatus({ id: session._id, status: v })
                }
              />
              <StatusBadge
                status={session.effective_status}
                reason={session.status_reason}
              />
              <span className="uppercase tracking-widest text-[10px] font-medium ml-4">
                Operator
              </span>
              <select
                value={session.operator ?? ""}
                onChange={(e) =>
                  setSessionOperator({ id: session._id, operator: e.target.value })
                }
                className="rounded-lg border border-warm-200 bg-white px-2 py-1 text-xs text-ink cursor-pointer"
              >
                {!session.operator && <option value="">unset</option>}
                {(operators ?? []).map((op) => (
                  <option key={op.hf_username} value={op.hf_username}>
                    {op.hf_username}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Expanded detail */}
          {expandedSession === (session._id as string) && (
            <SessionDetail sessionId={session._id} />
          )}
        </div>
      ))
      )}
        </>
      )}
    </div>
  );
}
