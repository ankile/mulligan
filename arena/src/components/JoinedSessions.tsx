import { releaseLabel } from "../lib/releaseLabels";
import { useEffect, useMemo, useState } from "react";
import { useQueries } from "../lib/arenaClient";
import type { FunctionReturnType } from "convex/server";
import { api } from "../release/api";
import type { Id } from "../release/api";
import {
  fetchEpisodeSubset,
  getParquetCache,
  selectPrimaryCameraKey,
  type EpisodeMetadata,
} from "../lib/hf-api";
import { useSearchParamNullable, useSearchParamNumber } from "../lib/useSearchParam";
import {
  alignRounds,
  alignmentSummary,
  formatIdList,
  hidePolicies,
  joinedArmKey,
  joinedArms,
  joinedPolicies,
  parseIdList,
  sessionLetter,
  sideSuccessSummary,
  toggleId,
  type JoinSide,
  type JoinedRound,
} from "../lib/joinSessions";
import { RoundVideos } from "./RoundVideos";
import { roundVideoSpecs, type RoundVideoSpec } from "../lib/roundVideoSpecs";
import { StatusBadge } from "./StatusBadge";
import {
  ArmBadge,
  ArmLegend,
  ExpandAllButton,
  PairedTestsDrawer,
  PolicyStatCards,
  RoundsGrid,
  type ArmRound,
} from "./RoundOutcomes";

type SessionDetail = NonNullable<
  FunctionReturnType<typeof api.evalSessions.getDetail>
>;
export type SessionListRow = FunctionReturnType<
  typeof api.evalSessions.list
>[number];

type DsCacheEntry =
  | { status: "loading" }
  | {
      status: "loaded";
      episodeMap: Map<number, Omit<EpisodeMetadata, "success">>;
      cameraKey: string;
    }
  | { status: "error"; message: string };

function formatDate(timestamp: number): string {
  return new Date(timestamp).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// A round's tiles share one row when there are at most this many; beyond
// that each session gets its own column-aligned row.
const SINGLE_ROW_MAX_TILES = 4;

const SIDE_GRID: Record<number, string> = {
  1: "grid-cols-1",
  2: "grid-cols-2",
  3: "grid-cols-3",
};

/**
 * Side-by-side view of N eval sessions aligned on round index: round k of
 * session A next to round k of session B (and C, ...). Every arm (session
 * letter × policy) gets the same summary cards, paired tests and outcome
 * grid as a single session, computed over the aligned rounds; each round
 * expands into one synchronized video grid holding every rollout from every
 * session.
 */
export default function JoinedSessions({
  sessionIds,
  sessionRows,
}: {
  sessionIds: string[];
  sessionRows: Map<string, SessionListRow>;
}) {
  const joinKey = sessionIds.join(",");
  const detailQueries = useMemo(() => {
    const queries: Record<
      string,
      { query: typeof api.evalSessions.getDetail; args: { id: Id<"evalSessions"> } }
    > = {};
    for (const id of sessionIds) {
      queries[id] = {
        query: api.evalSessions.getDetail,
        args: { id: id as Id<"evalSessions"> },
      };
    }
    return queries;
  }, [joinKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const details = useQueries(detailQueries) as Record<
    string,
    SessionDetail | null | undefined | Error
  >;

  const [expandedRound, setExpandedRound] = useSearchParamNumber("round");
  const [expandAll, setExpandAll] = useState(false);

  // Policies hidden from the joined surface (cards, tests, pills, videos), by
  // policy id, so a joined eval can be narrowed to the models being compared.
  // Lives in ?hide= so a filtered view is shareable.
  const [hideParam, setHideParam] = useSearchParamNullable("hide");
  const hiddenPolicyIds = new Set(parseIdList(hideParam));
  const toggleHidden = (policyId: string) =>
    setHideParam(formatIdList(toggleId(parseIdList(hideParam), policyId)));

  // Per-dataset episode metadata (sessions may live in different repos).
  const [datasetCache, setDatasetCache] = useState<Map<string, DsCacheEntry>>(
    () => {
      const initial = new Map<string, DsCacheEntry>();
      for (const [repo, cached] of getParquetCache()) {
        const episodeMap = new Map<number, Omit<EpisodeMetadata, "success">>();
        for (const ep of cached.episodes) episodeMap.set(ep.episodeIndex, ep);
        initial.set(repo, {
          status: "loaded",
          episodeMap,
          cameraKey: selectPrimaryCameraKey(cached.cameraKeys),
        });
      }
      return initial;
    },
  );

  const loadedDetails = sessionIds
    .map((id) => details[id])
    .filter((d): d is SessionDetail => d !== undefined && d !== null && !(d instanceof Error));
  const uniqueRepos = [...new Set(loadedDetails.map((d) => d.dataset_repo))];

  useEffect(() => {
    for (const repo of uniqueRepos) {
      if (datasetCache.has(repo)) continue;
      // Publish the in-flight cache entry before starting its async load.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDatasetCache((prev) => new Map(prev).set(repo, { status: "loading" }));
      const neededIndices = new Set(
        loadedDetails
          .filter((d) => d.dataset_repo === repo)
          .flatMap((d) => d.rounds.flatMap((r) => r.results.map((res) => res.episode_index))),
      );
      fetchEpisodeSubset(repo, neededIndices)
        .then((info) => {
          const episodeMap = new Map<number, Omit<EpisodeMetadata, "success">>();
          for (const ep of info.episodes) episodeMap.set(ep.episodeIndex, ep);
          setDatasetCache((prev) =>
            new Map(prev).set(repo, {
              status: "loaded",
              episodeMap,
              cameraKey: selectPrimaryCameraKey(info.cameraKeys),
            }),
          );
        })
        .catch((err: unknown) => {
          setDatasetCache((prev) =>
            new Map(prev).set(repo, { status: "error", message: String(err) }),
          );
        });
    }
  }, [uniqueRepos.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  // Content signature of the loaded sessions: the joined arms/rounds (and the
  // 20k-resample paired tests downstream) only recompute when a session's
  // outcomes actually change, not on every render.
  const detailSig = loadedDetails
    .map(
      (d) =>
        `${d._id}:${d.rounds
          .map((r) =>
            r.results
              .map(
                (x) =>
                  `${x.policy_id}${x.success ? 1 : 0}${x.num_subtask_marks ?? "n"}:${x.num_frames ?? "n"}`,
              )
              .join(""),
          )
          .join("/")}`,
    )
    .join("|");
  const joined = useMemo(() => {
    const sides: JoinSide[] = loadedDetails.map((d) => ({
      sessionId: d._id,
      rounds: d.rounds,
    }));
    const hidden = new Set(parseIdList(hideParam));
    const aligned = hidePolicies(alignRounds(sides), hidden);
    const arms = joinedArms(sides, hidden);
    // Policy number per policy id (hidden ones included, so numbers are
    // stable across toggles) for the side-card chips.
    const policyNumbers = new Map(joinedPolicies(sides).map((p, i) => [p.policy_id, i + 1]));
    // Release rounds carry display labels (start IDs are the alignment key).
    const labels = new Map<number, string>();
    for (const d of loadedDetails)
      for (const r of d.rounds) {
        const label = releaseLabel(r);
        if (label !== undefined && !labels.has(r.index)) labels.set(r.index, label);
      }
    const rounds: ArmRound[] = aligned.map((round) => ({
      index: round.index,
      label: labels.get(round.index),
      results: round.perSide.flatMap((results, i) =>
        (results ?? []).map((r) => ({
          policy_id: joinedArmKey(i, r.policy_id),
          policyName: r.policyName,
          success: r.success,
          episode_index: r.episode_index,
          num_subtask_marks: r.num_subtask_marks ?? null,
          num_frames: r.num_frames ?? null,
          visit_id: (r as { visit_id?: string }).visit_id,
        })),
      ),
    }));
    const alignedByIndex = new Map(aligned.map((r) => [r.index, r]));
    const maxMarks = Math.max(0, ...loadedDetails.map((d) => d.max_subtask_marks));
    return { sides, aligned, alignedByIndex, arms, policyNumbers, rounds, maxMarks };
  }, [detailSig, hideParam]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Loading / error gates ──
  const pending = sessionIds.some((id) => details[id] === undefined);
  if (pending) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
        <div className="flex items-center justify-center gap-3 text-ink-muted">
          <div className="w-5 h-5 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
          <span className="font-body">Loading joined sessions...</span>
        </div>
      </div>
    );
  }
  const problems = sessionIds.flatMap((id, i) => {
    const d = details[id];
    if (d instanceof Error) return [`Session ${sessionLetter(i)} (${id}): ${d.message}`];
    if (d === null) return [`Session ${sessionLetter(i)} (${id}) no longer exists — remove it from the join.`];
    return [];
  });
  if (problems.length > 0) {
    return (
      <div className="bg-white rounded-2xl border border-coral/40 shadow-sm p-6 space-y-1">
        {problems.map((p) => (
          <p key={p} className="text-sm text-coral font-mono">
            {p}
          </p>
        ))}
      </div>
    );
  }

  const { sides, aligned, alignedByIndex, arms, policyNumbers, rounds, maxMarks } = joined;
  const hiddenCount = joinedPolicies(sides).filter((p) => hiddenPolicyIds.has(p.policy_id)).length;
  const anyVideo = uniqueRepos.some((repo) => datasetCache.get(repo)?.status === "loaded");

  // Expanded round: one synchronized grid over every side's rollouts. When
  // every visible tile fits in one row, pack them all (the arm label badge,
  // e.g. B3, tells them apart); otherwise one row per session, padded to the
  // widest side so A's rollouts sit above B's, column-aligned.
  const renderExpanded = (round: JoinedRound) => {
    const sideSpecs: RoundVideoSpec[][] = [];
    const notes: string[] = [];
    round.perSide.forEach((results, sideIdx) => {
      if (!results || results.length === 0) return;
      const detail = loadedDetails[sideIdx];
      const ds = datasetCache.get(detail.dataset_repo);
      if (ds?.status === "loaded") {
        sideSpecs.push(
          roundVideoSpecs(
            results,
            detail.dataset_repo,
            ds.episodeMap,
            ds.cameraKey,
            new Map(results.map((r) => [r.policy_id, `${sessionLetter(sideIdx)}${policyNumbers.get(r.policy_id)!}`])),
            detail.max_subtask_marks,
          ),
        );
      } else if (ds?.status === "error") {
        notes.push(`${sessionLetter(sideIdx)}: video metadata failed — ${ds.message}`);
      } else {
        notes.push(`${sessionLetter(sideIdx)}: loading video metadata…`);
      }
    });
    const total = sideSpecs.reduce((n, specs) => n + specs.length, 0);
    const singleRow = total <= SINGLE_ROW_MAX_TILES;
    const columns = singleRow
      ? Math.max(1, total)
      : Math.max(1, ...sideSpecs.map((specs) => specs.length));
    const videos: (RoundVideoSpec | null)[] = [];
    for (const specs of sideSpecs) {
      videos.push(...specs);
      if (!singleRow) {
        for (let k = specs.length; k < columns; k++) videos.push(null);
      }
    }
    return (
      <>
        {videos.some(Boolean) && <RoundVideos videos={videos} columns={columns} />}
        {notes.map((note) => (
          <p key={note} className="text-xs text-ink-muted mt-2 font-mono">
            {note}
          </p>
        ))}
      </>
    );
  };

  return (
    <div className="space-y-4">
      {/* Side identity cards: letter, provenance, and the policy chips that
          toggle a policy in/out of the whole joined surface. */}
      <div className={`grid gap-3 ${SIDE_GRID[loadedDetails.length] ?? "grid-cols-4"}`}>
        {loadedDetails.map((detail, i) => {
          const row = sessionRows.get(detail._id);
          return (
            <div
              key={detail._id}
              className="bg-white rounded-2xl border border-warm-200 shadow-sm px-5 py-4"
            >
              <div className="flex items-center gap-2 mb-2 flex-wrap">
                <span className="w-6 h-6 rounded-md bg-gold text-white text-xs font-mono font-semibold flex items-center justify-center">
                  {sessionLetter(i)}
                </span>
                {releaseLabel(detail) && (
                  <span className="text-xs font-medium text-ink-light whitespace-nowrap">{releaseLabel(detail)}</span>
                )}
                <span className="text-xs text-ink-muted">
                  {formatDate(detail._creationTime)}
                </span>
                <span className="inline-block px-2 py-0.5 rounded-full bg-warm-100 text-ink-muted text-[11px] font-mono">
                  {detail.rounds.length} rounds
                </span>
                {row && (
                  <StatusBadge status={row.effective_status} reason={row.status_reason} />
                )}
                {detail.operator && (
                  <span className="inline-block px-2 py-0.5 rounded-full bg-warm-100 text-ink-muted text-[11px] font-mono">
                    op: {detail.operator}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5 flex-wrap mb-2">
                {sideSuccessSummary(sides[i]).map((p) => {
                  const hidden = hiddenPolicyIds.has(p.policy_id);
                  return (
                    <button
                      key={p.policy_id}
                      onClick={() => toggleHidden(p.policy_id)}
                      title={hidden ? "Show this policy" : "Hide this policy from the joined view"}
                      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-mono border transition-all cursor-pointer max-w-full ${
                        hidden
                          ? "bg-white text-ink-muted/60 border-warm-200 border-dashed"
                          : "bg-warm-100 text-ink border-warm-200 hover:border-teal/40"
                      }`}
                    >
                      <ArmBadge label={`${sessionLetter(i)}${policyNumbers.get(p.policy_id)!}`} />
                      <span className={`truncate ${hidden ? "line-through" : ""}`}>{p.policyName}</span>
                    </button>
                  );
                })}
              </div>
              <a
                href={`https://huggingface.co/datasets/${detail.dataset_repo}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[11px] font-mono text-ink-muted hover:text-teal transition-colors break-all"
              >
                {detail.dataset_repo} &rarr;
              </a>
              {detail.notes && (
                <p className="text-xs text-ink-muted italic mt-1">{detail.notes}</p>
              )}
            </div>
          );
        })}
      </div>

      {/* Joined analysis: same cards / tests / grid as a single session, over
          the aligned rounds with one arm per (session, policy). */}
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm px-6 pt-5 pb-5">
        <div className="flex items-center gap-3 mb-4 flex-wrap">
          <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
            Rounds aligned by index
          </span>
          <span className="font-mono text-xs text-ink-muted">
            {aligned.length} total · {alignmentSummary(aligned)}
          </span>
          {hiddenCount > 0 && (
            <button
              onClick={() => setHideParam(null)}
              className="text-[11px] text-teal hover:underline cursor-pointer"
            >
              show all ({hiddenCount} hidden)
            </button>
          )}
        </div>

        <ArmLegend arms={arms} />
        <PolicyStatCards arms={arms} rounds={rounds} maxMarks={maxMarks} />
        <PairedTestsDrawer arms={arms} rounds={rounds} maxMarks={maxMarks} />

        <div className="flex items-center justify-between gap-3 mb-2">
          <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium">
            Rounds
          </span>
          {anyVideo && (
            <ExpandAllButton expandAll={expandAll} onToggle={() => setExpandAll((v) => !v)} />
          )}
        </div>
        <RoundsGrid
          arms={arms}
          rounds={rounds}
          maxMarks={maxMarks}
          expandedRound={expandedRound}
          expandAll={expandAll}
          canExpand={anyVideo}
          onToggleRound={setExpandedRound}
          renderExpanded={(round) => renderExpanded(alignedByIndex.get(round.index)!)}
        />
      </div>
    </div>
  );
}
