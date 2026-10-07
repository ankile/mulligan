import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { useSearchParam, useSearchParamNullable, useSearchParamNumber, clearSearchParams } from "../lib/useSearchParam";
import { useQuery, useMutation } from "../lib/arenaClient";
import { api } from "../release/api";
import {
  fetchParquetMetadata,
  fetchSuccessStatus,
  getParquetCache,
  getVideoUrl,
  datasetFps,
  episodeForCamera,
  explorerCameraKeys,
  type EpisodeMetadata,
} from "../lib/hf-api";
import {
  datasetRoleLabel,
  resolvedDatasetRole,
} from "../lib/dataset-classification";
import OutcomeReview from "./OutcomeReview";
import StageReview from "./StageReview";
import { navigateToDataset } from "../lib/appNavigation";
import { cameraLabel, orderCameraKeys } from "./review/format";
import { StatusBadge, StatusSelect } from "./StatusBadge";
import { TaskFilterChips } from "./TaskFilterChips";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SourceTypeFilter = "all" | "teleop" | "rollout" | "dagger" | "eval";
type DatasetRoleFilter =
  | "all"
  | "aggregate_parent"
  | "training_view"
  | "eval_session"
  | "rollout";
type TrainableFilter = "all" | "true" | "false";

type EpisodeWithOptionalSuccess = Omit<EpisodeMetadata, "success"> & { success: boolean | null };

const SOURCE_TYPE_FILTERS: { id: SourceTypeFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "teleop", label: "Teleop" },
  { id: "rollout", label: "Rollout" },
  { id: "dagger", label: "DAgger" },
  { id: "eval", label: "Eval" },
];

const ROLE_FILTERS: { id: DatasetRoleFilter; label: string }[] = [
  { id: "all", label: "All roles" },
  { id: "training_view", label: "Training views" },
  { id: "aggregate_parent", label: "Parents" },
  { id: "eval_session", label: "Eval sessions" },
  { id: "rollout", label: "Rollouts" },
];

const TRAINABLE_FILTERS: { id: TrainableFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "true", label: "Trainable" },
  { id: "false", label: "Not trainable" },
];


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Datasets name their video features by station camera role
// (`observation.images.side_1`); other keys are shown as stored.
const CAMERA_ROLE_LABELS: Record<string, string> = {
  side_1: "Side 1",
  side_2: "Side 2",
  wrist_left: "Wrist",
  wrist_right: "Wrist (right eye)",
};

function cameraDisplayName(key: string): string {
  const name = cameraLabel(key);
  return CAMERA_ROLE_LABELS[name] ?? name;
}

function formatDuration(seconds: number): string {
  const s = Math.round(seconds * 10) / 10;
  return `${s.toFixed(1)}s`;
}

function formatDurationLong(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  return `${(seconds / 3600).toFixed(1)} hrs`;
}

function formatFrameCount(count: number): string {
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
  return count.toString();
}

function SourceTypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    teleop: "bg-blue-100 text-blue-700",
    rollout: "bg-purple-100 text-purple-700",
    dagger: "bg-amber-100 text-amber-700",
    eval: "bg-teal-light text-teal",
  };
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-medium capitalize ${
        colors[type] ?? "bg-warm-100 text-ink-muted"
      }`}
    >
      {type === "dagger" ? "DAgger" : type}
    </span>
  );
}

function RoleBadge({ role, sourceType }: { role?: string; sourceType: string }) {
  const resolvedRole = resolvedDatasetRole(role, sourceType);
  const colors: Record<string, string> = {
    aggregate_parent: "bg-warm-100 text-ink-muted",
    training_view: "bg-teal-light text-teal",
    eval_session: "bg-purple-100 text-purple-700",
    rollout: "bg-blue-100 text-blue-700",
  };
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-medium ${
        resolvedRole
          ? (colors[resolvedRole] ?? "bg-warm-100 text-ink-muted")
          : "bg-warm-100 text-ink-muted"
      }`}
    >
      {datasetRoleLabel(role, sourceType)}
    </span>
  );
}

function TrainableBadge({ trainable }: { trainable?: boolean }) {
  if (trainable === undefined) return null;
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-medium ${
        trainable ? "bg-teal-light text-teal" : "bg-warm-100 text-ink-muted"
      }`}
    >
      {trainable ? "Trainable" : "Not trainable"}
    </span>
  );
}

function EnvironmentTag({ env }: { env: string }) {
  const colors: Record<string, string> = {
    franka_pick_cube: "bg-teal-light text-teal",
    franka_nut_assembly_square: "bg-coral-light text-coral",
    franka_stack_two_blocks: "bg-gold-light text-gold",
  };
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-medium ${
        colors[env] ?? "bg-warm-100 text-ink-muted"
      }`}
    >
      {env}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Video Grid (synced multi-camera playback)
// ---------------------------------------------------------------------------

function VideoGrid({
  episode,
  playing,
  onTogglePlay,
  cameraKeys,
  datasetId,
}: {
  episode: EpisodeWithOptionalSuccess;
  playing: boolean;
  onTogglePlay: () => void;
  cameraKeys: string[];
  datasetId: string;
}) {
  const videoRefs = useRef<(HTMLVideoElement | null)[]>([]);
  const primaryRef = useRef<HTMLVideoElement | null>(null);
  const animFrameRef = useRef<number>(0);
  const [videoReady, setVideoReady] = useState<Record<string, boolean>>({});

  const setVideoRef = useCallback(
    (index: number) => (el: HTMLVideoElement | null) => {
      videoRefs.current[index] = el;
      if (index === 0) primaryRef.current = el;
    },
    []
  );

  useEffect(() => {
    const videos = videoRefs.current.filter(Boolean) as HTMLVideoElement[];
    videos.forEach((video,i) => {
      video.pause();
      video.currentTime = episodeForCamera(episode,cameraKeys[i]).fromTimestamp;
    });
  }, [episode, cameraKeys]);

  useEffect(() => {
    const primary = primaryRef.current;
    if (!primary) return;
    const videos = videoRefs.current.filter(Boolean) as HTMLVideoElement[];

    if (playing) {
      for (const v of videos) v.play();
      const sync = () => {
        const first = episodeForCamera(episode,cameraKeys[0]);
        const t = primary.currentTime;
        const elapsed = t - first.fromTimestamp;
        for (let i = 1; i < videos.length; i++) {
          const target = elapsed + episodeForCamera(episode,cameraKeys[i]).fromTimestamp;
          if (Math.abs(videos[i].currentTime - target) > 0.1) {
            videos[i].currentTime = target;
          }
        }
        if (t >= first.toTimestamp - 0.05) {
          videos.forEach((v,i) => {
            v.pause();
            v.currentTime = episodeForCamera(episode,cameraKeys[i]).fromTimestamp;
          });
          onTogglePlay();
          return;
        }
        animFrameRef.current = requestAnimationFrame(sync);
      };
      animFrameRef.current = requestAnimationFrame(sync);
    } else {
      for (const v of videos) v.pause();
    }
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [playing, episode, onTogglePlay, cameraKeys]);

  const gridCols = cameraKeys.length === 1 ? "grid-cols-1" : "grid-cols-2";

  return (
    <div>
      <div className={`grid ${gridCols} gap-3`}>
        {cameraKeys.map((key, i) => (
          <div key={key} className="relative">
            <video
              ref={setVideoRef(i)}
              src={getVideoUrl(key, episodeForCamera(episode,key).videoFileIndex, datasetId)}
              className="w-full rounded-lg bg-warm-100"
              muted
              playsInline
              preload="auto"
              onLoadedMetadata={(e) => {
                (e.target as HTMLVideoElement).currentTime =
                  episodeForCamera(episode,key).fromTimestamp;
                setVideoReady((prev) => ({ ...prev, [key]: true }));
              }}
            />
            {!videoReady[key] && (
              <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-warm-100">
                <div className="w-5 h-5 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
              </div>
            )}
            <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 text-white text-[11px] font-mono">
              {cameraDisplayName(key)}
            </span>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4 mt-4">
        <button
          onClick={onTogglePlay}
          className="flex items-center gap-2 px-5 py-2 rounded-lg bg-teal text-white font-body font-medium text-sm hover:bg-teal/90 transition-colors cursor-pointer"
        >
          {playing ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="6,4 20,12 6,20" />
            </svg>
          )}
          {playing ? "Pause" : "Play"}
        </button>

        <button
          onClick={() => {
            const videos = videoRefs.current.filter(
              Boolean
            ) as HTMLVideoElement[];
            for (const v of videos) {
              v.currentTime = episode.fromTimestamp;
            }
            if (playing) onTogglePlay();
          }}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-warm-200 text-ink-muted font-body text-sm hover:bg-warm-50 transition-colors cursor-pointer"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
          </svg>
          Reset
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Episode Card
// ---------------------------------------------------------------------------

function EpisodeCard({
  episode,
  selected,
  onClick,
}: {
  episode: EpisodeWithOptionalSuccess;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex-shrink-0 px-4 py-3 rounded-xl border transition-all duration-150 text-left cursor-pointer ${
        selected
          ? "bg-teal/10 border-teal shadow-sm"
          : "bg-white border-warm-200 hover:border-warm-300 hover:shadow-sm"
      }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-sm font-medium text-ink">
          Ep {episode.episodeIndex}
        </span>
        {episode.success === null ? (
          <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide bg-warm-100 text-ink-muted animate-pulse">
            ...
          </span>
        ) : (
          <span
            className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${
              episode.success
                ? "bg-teal-light text-teal"
                : "bg-coral-light text-coral"
            }`}
          >
            {episode.success ? "Success" : "Fail"}
          </span>
        )}
      </div>
      <div className="text-xs text-ink-muted font-mono">
        {formatDuration(episode.duration)} &middot; {episode.numFrames}f
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Dataset Detail Panel
// ---------------------------------------------------------------------------

/** Resolve the model link from explicit model_url. */
function resolveModelLink(dataset: { model_url?: string }): {
  url: string;
  label: string;
} | null {
  if (!dataset.model_url) return null;
  const url = dataset.model_url;
  if (url.includes("wandb.ai")) return { url, label: "W&B" };
  if (url.includes("huggingface.co")) return { url, label: "HF Model" };
  return { url, label: "Model" };
}

function DatasetDetail({
  repoId,
  onBack, readOnly,
}: {
  repoId: string;
  onBack: () => void;
  readOnly: boolean;
}) {
  const dataset = useQuery(api.datasets.getByRepo, { repo_id: repoId });
  const viewer = useQuery(api.users.viewer);
  // Stage review is offered only for tasks with an exported live stage spec.
  const stageSpecs = useQuery(
    api.stageTaskSpecs.forTask,
    dataset?.task ? { task: dataset.task } : "skip"
  );
  const hasStageSpec = Boolean(stageSpecs?.some((row) => row.live));
  const [selectedIndex, setSelectedIndex] = useSearchParamNumber("episode");
  const [playing, setPlaying] = useState(false);
  const [episodeFilter, setEpisodeFilter] = useSearchParam("outcome", "all");
  // "view" (not "mode") because EvalSessions already owns the "mode" param.
  const [requestedView, setView] = useSearchParam("view", "explorer");
  const view = readOnly ? "explorer" : requestedView;
  const setDatasetStatus = useMutation(api.datasets.setStatus);

  // -- Staged state --
  const [baseEpisodes, setBaseEpisodes] = useState<Omit<EpisodeMetadata, "success">[]>([]);
  const [cameraKeys, setCameraKeys] = useState<string[]>([]);
  const [successMap, setSuccessMap] = useState<Map<number, boolean> | null>(null);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [successLoading, setSuccessLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Derive episodes with optional success
  const episodes: EpisodeWithOptionalSuccess[] = useMemo(
    () =>
      baseEpisodes.map((ep) => ({
        ...ep,
        success: successMap ? (successMap.get(ep.episodeIndex) ?? false) : null,
      })),
    [baseEpisodes, successMap]
  );

  // Initialize from cache on mount / dataset switch
  const prevRepoId = useRef(repoId);
  /* eslint-disable react-hooks/set-state-in-effect -- a dataset identity change
     resets working UI state and hydrates the new dataset from the shared cache. */
  useEffect(() => {
    if (prevRepoId.current !== repoId) {
      prevRepoId.current = repoId;
      setSelectedIndex(null);
      // Reset UI state when navigating between dataset identities.
      setPlaying(false);
      setEpisodeFilter("all");
    }

    // Pre-populate from cache
    const cached = getParquetCache().get(repoId);
    if (cached?.complete) {
      setBaseEpisodes(cached.episodes);
      setCameraKeys(orderCameraKeys(explorerCameraKeys(cached.cameraKeys)));
      setEpisodesLoading(false);
    } else {
      setBaseEpisodes([]);
      setCameraKeys([]);
      setEpisodesLoading(true);
    }
    setSuccessMap(null);
    setSuccessLoading(true);
    setError(null);
  }, [repoId, setEpisodeFilter, setSelectedIndex]);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Effect 1: Parquet fetch
  useEffect(() => {
    let cancelled = false;
    // A rollout preview may have populated only a subset of the episodes.
    // Skip this full fetch only when the cache itself says it is complete.
    if (getParquetCache().get(repoId)?.complete) return;

    fetchParquetMetadata(repoId)
      .then((result) => {
        if (cancelled) return;
        setBaseEpisodes(result.episodes);
        setCameraKeys(orderCameraKeys(explorerCameraKeys(result.cameraKeys)));
        setEpisodesLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setEpisodesLoading(false);
      });

    return () => { cancelled = true; };
  }, [repoId]);

  // Effect 2: Success status (retry once on transient HF server errors)
  useEffect(() => {
    let cancelled = false;
    const attempt = (retriesLeft: number) => {
      fetchSuccessStatus(repoId)
        .then((map) => {
          if (cancelled) return;
          setSuccessMap(map);
          setSuccessLoading(false);
        })
        .catch((err) => {
          if (cancelled) return;
          if (retriesLeft > 0) {
            setTimeout(() => attempt(retriesLeft - 1), 1500);
          } else {
            setSuccessMap(null);
            setSuccessLoading(false);
            setError(`Failed to load episode outcomes: ${err.message}`);
          }
        });
    };
    attempt(1);

    return () => { cancelled = true; };
  }, [repoId]);

  const handleTogglePlay = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  const filteredEpisodes =
    episodeFilter === "all"
      ? episodes
      : episodeFilter === "success"
        ? episodes.filter((e) => e.success === true)
        : episodes.filter((e) => e.success === false);

  const selectedEpisode =
    selectedIndex !== null ? episodes[selectedIndex] : null;

  const successCount = successMap ? [...successMap.values()].filter(Boolean).length : null;

  // Outcome review replaces the detail view entirely. It fetches its own raw
  // episode metadata, so it must not wait on the explorer's parquet load.
  if (view === "review") {
    return (
      <OutcomeReview
        repoId={repoId}
        task={dataset?.task}
        onExit={() => {
          // The "episode" param means an array position in the explorer and an
          // episode_index in review mode; drop it when crossing the boundary.
          setSelectedIndex(null);
          setView("explorer");
        }}
      />
    );
  }

  // Stage-label review — same boundary semantics as outcome review.
  if (view === "stage") {
    return (
      <StageReview
        repoId={repoId}
        task={dataset?.task}
        onExit={() => {
          setSelectedIndex(null);
          clearSearchParams("sstatus", "sconf", "sflag", "sarm", "schema", "prediction", "blind");
          setView("explorer");
        }}
        // The "episode" param means episode_index in BOTH review views, so it
        // carries over — the outcome editor opens on the gated episode.
        onOpenOutcomeReview={() => setView("review")}
      />
    );
  }

  // Only show full-page spinner on cache miss with no data
  if (episodesLoading && baseEpisodes.length === 0) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
        <button
          onClick={onBack}
          className="text-xs text-ink-muted hover:text-teal mb-4 cursor-pointer"
        >
          &larr; Back to datasets
        </button>
        <div className="flex items-center justify-center gap-3 text-ink-muted">
          <div className="w-5 h-5 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
          <span className="font-body">Loading episodes from HuggingFace...</span>
        </div>
      </div>
    );
  }

  if (error && baseEpisodes.length === 0) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
        <button
          onClick={onBack}
          className="text-xs text-ink-muted hover:text-teal mb-4 cursor-pointer"
        >
          &larr; Back to datasets
        </button>
        <p className="text-coral font-body text-center">
          Failed to load episodes: {error}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl border border-warm-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-warm-100 bg-warm-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="text-ink-muted hover:text-teal transition-colors cursor-pointer"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="15 18 9 12 15 6" />
              </svg>
            </button>
            <div>
              <h2 className="font-display text-xl text-ink">
                {repoId.split("/").pop()}
              </h2>
              <p className="text-xs text-ink-muted mt-0.5 font-mono">
                {repoId}
              </p>
              {dataset && (
                <div className="flex flex-wrap items-center gap-1.5 mt-2">
                  <SourceTypeBadge type={dataset.source_type} />
                  <RoleBadge
                    role={dataset.dataset_role}
                    sourceType={dataset.source_type}
                  />
                  <TrainableBadge trainable={dataset.trainable} />
                  <EnvironmentTag env={dataset.environment} />
                  <StatusBadge
                    status={dataset.effective_status}
                    reason={dataset.status_reason}
                  />
                  {viewer?.isEditor && (
                    <StatusSelect
                      value={dataset.status ?? "inherit"}
                      onChange={(v) =>
                        setDatasetStatus({ repo_id: repoId, status: v })
                      }
                    />
                  )}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            {dataset && (() => {
              const link = resolveModelLink(dataset);
              return link ? (
                <a
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-ink-muted hover:text-teal transition-colors font-mono"
                >
                  {link.label} &rarr;
                </a>
              ) : null;
            })()}
            <a
              href={`https://huggingface.co/datasets/${repoId}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-ink-muted hover:text-teal transition-colors font-mono"
            >
              HuggingFace &rarr;
            </a>
          </div>
        </div>
      </div>

      <div className="p-6">
        {error && (
          <div className="mb-5 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral">
            {error}
          </div>
        )}
        {/* Summary stats */}
        {dataset && (
          <div className="mb-5 flex flex-wrap items-center gap-2 text-xs font-mono">
            {dataset.parent_repo_id && (
              <a
                href={`?tab=explorer&dataset=${encodeURIComponent(dataset.parent_repo_id)}`}
                className="px-2 py-1 rounded bg-warm-100 text-ink-muted hover:text-teal transition-colors"
              >
                Parent dataset &rarr;
              </a>
            )}
            {dataset.derived_repo_ids?.map((repo) => (
              <a
                key={repo}
                href={`?tab=explorer&dataset=${encodeURIComponent(repo)}`}
                className="px-2 py-1 rounded bg-teal-light text-teal hover:opacity-80 transition-opacity"
              >
                {repo.split("/").pop()} &rarr;
              </a>
            ))}
            {dataset.mutually_exclusive_with?.map((repo) => (
              <a
                key={repo}
                href={`?tab=explorer&dataset=${encodeURIComponent(repo)}`}
                className="px-2 py-1 rounded bg-coral-light text-coral hover:opacity-80 transition-opacity"
              >
                Exclusive with {repo.split("/").pop()} &rarr;
              </a>
            ))}
          </div>
        )}

        {/* Summary stats */}
        {dataset && (() => {
          const total = episodes.length;
          const successPct = successCount != null && total > 0 ? Math.round((successCount / total) * 100) : null;
          const statsReady = dataset.stats_status === "ready";
          const statsPending = dataset.stats_status == null || dataset.stats_status === "pending";
          const statsFailed = dataset.stats_status === "error";
          const cachedSuccess = dataset.num_success == null ? null : Number(dataset.num_success);
          const cachedFailure = dataset.num_failure == null ? null : Number(dataset.num_failure);
          const cachedOutcomeTotal =
            cachedSuccess != null && cachedFailure != null ? cachedSuccess + cachedFailure : null;
          const autonomousCount =
            statsReady && dataset.num_autonomous_success != null
              ? Number(dataset.num_autonomous_success)
              : null;
          const autonomousPct = autonomousCount != null && cachedOutcomeTotal != null && cachedOutcomeTotal > 0
            ? Math.round((autonomousCount / cachedOutcomeTotal) * 100)
            : null;
          const humanFrames =
            statsReady && dataset.num_human_frames != null
              ? Number(dataset.num_human_frames)
              : null;
          const policyFrames =
            statsReady && dataset.num_policy_frames != null
              ? Number(dataset.num_policy_frames)
              : null;
          const totalFrames =
            humanFrames != null && policyFrames != null ? humanFrames + policyFrames : null;
          const humanPct = totalFrames != null && totalFrames > 0
            ? Math.round((humanFrames! / totalFrames) * 100)
            : null;

          const stats: { label: string; value: string; color?: string; loading?: boolean; title?: string }[] = [
            { label: "Episodes", value: total.toString() },
            { label: "Duration", value: formatDurationLong(episodes.reduce((s, e) => s + e.duration, 0)) },
            {
              label: "Success Rate",
              value: successCount != null ? `${successCount}/${total} (${successPct}%)` : "...",
              color: successCount != null ? "text-teal" : undefined,
              loading: successLoading,
            },
          ];
          if (autonomousCount != null) {
            stats.push({ label: "Autonomous Success", value: `${autonomousCount}/${cachedOutcomeTotal} (${autonomousPct}%)`, color: "text-teal" });
          } else if (statsPending) {
            stats.push({ label: "Autonomous Success", value: "Pending", loading: true });
          } else if (statsFailed) {
            stats.push({ label: "Autonomous Success", value: "Error", color: "text-coral", title: dataset.stats_error });
          } else {
            stats.push({ label: "Autonomous Success", value: "N/A" });
          }
          if (humanFrames != null && policyFrames != null) {
            stats.push({
              label: "Human Frames",
              value: `${formatFrameCount(humanFrames)} (${humanPct}%)`,
            });
            stats.push({
              label: "Policy Frames",
              value: `${formatFrameCount(policyFrames)} (${100 - (humanPct ?? 0)}%)`,
            });
          } else if (statsPending) {
            stats.push({ label: "Human Frames", value: "Pending", loading: true });
            stats.push({ label: "Policy Frames", value: "Pending", loading: true });
          } else if (statsFailed) {
            stats.push({ label: "Human Frames", value: "Error", color: "text-coral", title: dataset.stats_error });
            stats.push({ label: "Policy Frames", value: "Error", color: "text-coral", title: dataset.stats_error });
          } else {
            stats.push({ label: "Human Frames", value: "N/A" });
            stats.push({ label: "Policy Frames", value: "N/A" });
          }
          stats.push({ label: "Cameras", value: cameraKeys.length.toString() });

          return (
            <div className="flex flex-wrap gap-4 mb-6">
              {stats.map((stat) => (
                <div
                  key={stat.label}
                  title={stat.title}
                  className="bg-warm-50 rounded-lg px-4 py-2.5 border border-warm-100"
                >
                  <div className="text-[10px] uppercase tracking-widest text-ink-muted font-medium mb-0.5">
                    {stat.label}
                  </div>
                  <div className={`font-mono text-lg font-medium ${stat.color ?? "text-ink"} ${stat.loading ? "animate-pulse" : ""}`}>
                    {stat.value}
                  </div>
                </div>
              ))}
            </div>
          );
        })()}

        {/* Episode filter pills */}
        <div className="flex items-center gap-2 mb-4">
          {(
            [
              { id: "all" as const, label: "All", count: episodes.length },
              {
                id: "success" as const,
                label: "Success",
                count: successCount,
              },
              {
                id: "failure" as const,
                label: "Failure",
                count: successCount != null ? episodes.length - successCount : null,
              },
            ] as const
          ).map((filter) => {
            const isActive = episodeFilter === filter.id;
            const isDisabled = filter.id !== "all" && successLoading;
            return (
              <button
                key={filter.id}
                onClick={() => !isDisabled && setEpisodeFilter(filter.id)}
                disabled={isDisabled}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isDisabled
                    ? "bg-white border border-warm-200 text-ink-muted/40 cursor-not-allowed"
                    : isActive
                      ? "bg-teal text-white shadow-sm cursor-pointer"
                      : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300 hover:text-ink cursor-pointer"
                }`}
              >
                {filter.label}
                <span
                  className={`font-mono text-[10px] ${
                    isDisabled
                      ? "text-ink-muted/30 animate-pulse"
                      : isActive ? "text-white/70" : "text-ink-muted/60"
                  }`}
                >
                  {filter.count != null ? filter.count : "..."}
                </span>
              </button>
            );
          })}
          {viewer?.isEditor && (
            <>
              <div className="flex-1" />
              <button
                onClick={() => {
                  setPlaying(false);
                  setSelectedIndex(null);
                  setView("review");
                }}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white border border-teal text-teal hover:bg-teal-light transition-all cursor-pointer"
              >
                Review episodes &rarr;
              </button>
              {hasStageSpec && (
                <button
                  onClick={() => {
                    setPlaying(false);
                    setSelectedIndex(null);
                    setView("stage");
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white border border-gold text-gold hover:bg-gold-light transition-all cursor-pointer"
                >
                  Stage review &rarr;
                </button>
              )}
            </>
          )}
        </div>

        {/* Episode strip */}
        <div className="flex gap-2 overflow-x-auto pb-3 -mx-1 px-1">
          {filteredEpisodes.map((ep) => {
            const globalIndex = episodes.indexOf(ep);
            return (
              <EpisodeCard
                key={ep.episodeIndex}
                episode={ep}
                selected={selectedIndex === globalIndex}
                onClick={() => {
                  setPlaying(false);
                  setSelectedIndex(globalIndex);
                }}
              />
            );
          })}
        </div>

        {/* Video grid */}
        {selectedEpisode ? (
          <div className="mt-5">
            <div className="flex items-center gap-3 mb-4">
              <span className="font-display text-lg text-ink">
                Episode {selectedEpisode.episodeIndex}
              </span>
              {selectedEpisode.success === null ? (
                <span className="inline-block px-2 py-0.5 rounded-full text-xs font-medium bg-warm-100 text-ink-muted animate-pulse">
                  ...
                </span>
              ) : (
                <span
                  className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${
                    selectedEpisode.success
                      ? "bg-teal-light text-teal"
                      : "bg-coral-light text-coral"
                  }`}
                >
                  {selectedEpisode.success ? "Success" : "Failure"}
                </span>
              )}
              <span className="text-xs text-ink-muted font-mono">
                {selectedEpisode.numFrames} frames &middot;{" "}
                {formatDuration(selectedEpisode.duration)} &middot; {datasetFps(repoId)} FPS
              </span>
            </div>
            {cameraKeys.length === 0 ? <p className="text-sm text-ink-muted">This dataset contains state observations without camera recordings. <a className="text-teal" href={`https://huggingface.co/datasets/${repoId}`}>Browse the data on Hugging Face ↗</a></p> : <VideoGrid
              episode={selectedEpisode}
              playing={playing}
              onTogglePlay={handleTogglePlay}
              cameraKeys={cameraKeys}
              datasetId={repoId}
            />}
          </div>
        ) : (
          <div className="mt-5 py-12 text-center text-ink-muted font-body">
            Select an episode above to view camera recordings
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main DataExplorer component
// ---------------------------------------------------------------------------

export default function DataExplorer({readOnly = false}: {readOnly?: boolean}) {
  const [sourceFilter, setSourceFilter] = useSearchParam("source", "all");
  const [roleFilter, setRoleFilter] = useSearchParam("role", "all");
  const [trainableFilter, setTrainableFilter] = useSearchParam("trainable", "all");
  const [taskFilter, setTaskFilter] = useSearchParam("task", "all");
  const [showParam] = useSearchParam("show", "mainline");
  const showAll = showParam === "all";
  const [selectedRepoId] = useSearchParamNullable("dataset");
  const setSelectedRepoId = navigateToDataset;

  const datasetQueryArgs: {
    source_type?: string;
    dataset_role?: string;
    trainable?: boolean;
  } = {};
  if (sourceFilter !== "all") datasetQueryArgs.source_type = sourceFilter;
  if (roleFilter !== "all") datasetQueryArgs.dataset_role = roleFilter;
  if (trainableFilter !== "all") datasetQueryArgs.trainable = trainableFilter === "true";

  const datasets = useQuery(api.datasets.list, datasetQueryArgs);

  // Check selectedRepoId FIRST — DatasetDetail fetches its own data from
  // HuggingFace, so it doesn't need the Convex datasets list to be ready.
  // This lets deep links render immediately without waiting for the unrelated query.
  if (selectedRepoId) {
    return (
      <div style={{ animation: "fade-up 0.6s ease-out 0.1s both" }}>
        <DatasetDetail
          // Defense in depth: a repoId swap without unmount would carry one
          // dataset's signals/pending into another's same-numbered episode.
          key={selectedRepoId}
          repoId={selectedRepoId}
          readOnly={readOnly}
          onBack={() => setSelectedRepoId(null)}
        />
      </div>
    );
  }

  if (datasets === undefined) {
    return (
      <div
        className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8"
        style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
      >
        <div className="flex items-center justify-center gap-3 text-ink-muted">
          <div className="w-5 h-5 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
          <span className="font-body">Loading datasets...</span>
        </div>
      </div>
    );
  }

  if (datasets.length === 0 && sourceFilter === "all") {
    return (
      <div
        className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted"
        style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
      >
        No datasets registered yet. Use <code className="font-mono text-xs bg-warm-100 px-1.5 py-0.5 rounded">--arena-url</code> when pushing datasets to register them.
      </div>
    );
  }

  // Mainline/all lens applies before every other filter and count.
  const visibleDatasets = showAll
    ? datasets
    : datasets.filter((d) => d.effective_status === "mainline");

  // Tasks for filter pills; TaskFilterChips dedupes and orders them (mainline first).
  const allTasks = visibleDatasets.map((d) => d.task);

  // Compute source type counts and episode counts (from all datasets before task filter)
  const sourceTypeCounts = new Map<string, number>();
  const sourceTypeEpisodeCounts = new Map<string, number>();
  for (const d of visibleDatasets) {
    sourceTypeCounts.set(d.source_type, (sourceTypeCounts.get(d.source_type) ?? 0) + 1);
    const eps = d.num_episodes != null ? Number(d.num_episodes) : 0;
    sourceTypeEpisodeCounts.set(d.source_type, (sourceTypeEpisodeCounts.get(d.source_type) ?? 0) + eps);
  }

  // Apply task filter
  const filteredDatasets =
    taskFilter === "all"
      ? visibleDatasets
      : visibleDatasets.filter((d) => d.task === taskFilter);

  return (
    <div
      className="space-y-4"
      style={{ animation: "fade-up 0.6s ease-out 0.3s both" }}
    >
      {/* Filter bars */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Source type filters */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium mr-1">
            Source
          </span>
          {SOURCE_TYPE_FILTERS.map((filter) => {
            const count =
              filter.id === "all"
                ? visibleDatasets.length
                : sourceTypeCounts.get(filter.id) ?? 0;
            const epCount =
              filter.id === "all"
                ? visibleDatasets.reduce((s, d) => s + (d.num_episodes != null ? Number(d.num_episodes) : 0), 0)
                : sourceTypeEpisodeCounts.get(filter.id) ?? 0;
            const isActive = sourceFilter === filter.id;
            return (
              <button
                key={filter.id}
                onClick={() => setSourceFilter(filter.id)}
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
                  {count} · {epCount} ep
                </span>
              </button>
            );
          })}
        </div>

        {/* Role filters */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium mr-1">
            Role
          </span>
          {ROLE_FILTERS.map((filter) => {
            const isActive = roleFilter === filter.id;
            return (
              <button
                key={filter.id}
                onClick={() => setRoleFilter(filter.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                  isActive
                    ? "bg-teal text-white shadow-sm"
                    : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300 hover:text-ink"
                }`}
              >
                {filter.label}
              </button>
            );
          })}
        </div>

        {/* Trainable filters */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium mr-1">
            Use
          </span>
          {TRAINABLE_FILTERS.map((filter) => {
            const isActive = trainableFilter === filter.id;
            return (
              <button
                key={filter.id}
                onClick={() => setTrainableFilter(filter.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                  isActive
                    ? "bg-teal text-white shadow-sm"
                    : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300 hover:text-ink"
                }`}
              >
                {filter.label}
              </button>
            );
          })}
        </div>

        {/* Task filter */}
        <TaskFilterChips tasks={allTasks} value={taskFilter} onChange={setTaskFilter} />
      </div>

      {readOnly && <p className="text-xs text-ink-muted">Raw recordings and training views can contain the same episodes; these totals count repository entries. Selected evaluation playback uses the frozen mainline policies. Some source recordings are included because later policies trained on them.</p>}
      {/* Aggregate summary */}
      {filteredDatasets.length > 0 && (() => {
        const totalEpisodes = filteredDatasets.reduce(
          (sum, d) => sum + (d.num_episodes != null ? Number(d.num_episodes) : 0),
          0
        );
        const totalDuration = filteredDatasets.reduce(
          (sum, d) => sum + (d.total_duration_seconds ?? 0),
          0
        );
        const totalSuccess = filteredDatasets.reduce(
          (sum, d) => sum + (d.num_success != null ? Number(d.num_success) : 0),
          0
        );
        const totalFailure = filteredDatasets.reduce(
          (sum, d) => sum + (d.num_failure != null ? Number(d.num_failure) : 0),
          0
        );
        const totalOutcomes = totalSuccess + totalFailure;
        const totalAutonomous = filteredDatasets.reduce(
          (sum, d) =>
            sum +
            (d.stats_status === "ready" && d.num_autonomous_success != null
              ? Number(d.num_autonomous_success)
              : 0),
          0
        );
        const totalHuman = filteredDatasets.reduce(
          (sum, d) =>
            sum +
            (d.stats_status === "ready" && d.num_human_frames != null
              ? Number(d.num_human_frames)
              : 0),
          0
        );
        const totalPolicy = filteredDatasets.reduce(
          (sum, d) =>
            sum +
            (d.stats_status === "ready" && d.num_policy_frames != null
              ? Number(d.num_policy_frames)
              : 0),
          0
        );
        const hasFrameStats = filteredDatasets.some(
          (d) => d.stats_status === "ready" && d.num_human_frames != null
        );
        const totalFrames = totalHuman + totalPolicy;
        return (
          <div className="text-xs font-mono flex items-center gap-3 flex-wrap">
            <span className="text-ink-muted">
              {filteredDatasets.length} datasets · {totalEpisodes} episodes
              {totalDuration > 0 && ` · ${formatDurationLong(totalDuration)}`}
            </span>
            {totalOutcomes > 0 && (
              <span className="text-teal">
                {totalSuccess}/{totalOutcomes} success ({Math.round((totalSuccess / totalOutcomes) * 100)}%)
              </span>
            )}
            {totalAutonomous > 0 && totalOutcomes > 0 && (
              <span className="text-teal/60">
                {totalAutonomous} autonomous ({Math.round((totalAutonomous / totalOutcomes) * 100)}%)
              </span>
            )}
            {hasFrameStats && totalFrames > 0 && (<>
              <span className="w-px h-3 bg-warm-200" />
              <span className="text-ink-muted">
                {formatFrameCount(totalHuman)} human · {formatFrameCount(totalPolicy)} policy frames ({Math.round((totalHuman / totalFrames) * 100)}% human)
              </span>
            </>)}
          </div>
        );
      })()}

      {/* Dataset list */}
      {filteredDatasets.length === 0 ? (
        <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted">
          No datasets match the current filters.
        </div>
      ) : (
        <div className="grid gap-3">
          {filteredDatasets.map((dataset) => (
            <button
              key={dataset._id}
              onClick={() => setSelectedRepoId(dataset.repo_id)}
              className="bg-white rounded-xl border border-warm-200 shadow-sm px-5 py-4 text-left hover:border-warm-300 hover:shadow-md transition-all cursor-pointer"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="font-body font-semibold text-ink text-[15px]">
                      {dataset.name}
                    </span>
                    <SourceTypeBadge type={dataset.source_type} />
                    <RoleBadge
                      role={dataset.dataset_role}
                      sourceType={dataset.source_type}
                    />
                    <TrainableBadge trainable={dataset.trainable} />
                    <EnvironmentTag env={dataset.environment} />
                    {showAll && (
                      <StatusBadge
                        status={dataset.effective_status}
                        reason={dataset.status_reason}
                      />
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs text-ink-muted">
                    {dataset.num_episodes != null && Number(dataset.num_episodes) > 0 && (
                      <span className="font-mono">
                        {Number(dataset.num_episodes)} episodes
                        {dataset.total_duration_seconds != null &&
                          ` · ${formatDurationLong(dataset.total_duration_seconds)}`}
                      </span>
                    )}
                    {(() => {
                      const link = resolveModelLink(dataset);
                      return link ? (
                        <a
                          href={link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-mono truncate max-w-[300px] hover:text-teal transition-colors"
                          onClick={(e) => e.stopPropagation()}
                          title={dataset.model_url}
                        >
                          {link.label} &rarr;
                        </a>
                      ) : null;
                    })()}
                    <span>
                      {new Date(dataset._creationTime).toLocaleDateString(
                        "en-US",
                        { month: "short", day: "numeric", year: "numeric" }
                      )}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted mt-1 font-mono">
                    {dataset.parent_repo_id && (
                      <a
                        href={`?tab=explorer&dataset=${encodeURIComponent(dataset.parent_repo_id)}`}
                        className="hover:text-teal transition-colors"
                        onClick={(e) => e.stopPropagation()}
                      >
                        parent &rarr;
                      </a>
                    )}
                    {dataset.derived_repo_ids && dataset.derived_repo_ids.length > 0 && (
                      <span>{dataset.derived_repo_ids.length} derived view(s)</span>
                    )}
                    {dataset.mutually_exclusive_with &&
                      dataset.mutually_exclusive_with.length > 0 && (
                        <span className="text-coral">
                          mutually exclusive with {dataset.mutually_exclusive_with.length}
                        </span>
                      )}
                  </div>
                  {dataset.num_success != null && (() => {
                    const nSuccess = Number(dataset.num_success);
                    const total = nSuccess + Number(dataset.num_failure);
                    const successPct = total > 0 ? Math.round((nSuccess / total) * 100) : 0;
                    const statsReady = dataset.stats_status === "ready";
                    const nAutonomous = statsReady && dataset.num_autonomous_success != null ? Number(dataset.num_autonomous_success) : 0;
                    const autonomousPct = total > 0 ? Math.round((nAutonomous / total) * 100) : 0;
                    const nHuman = statsReady && dataset.num_human_frames != null ? Number(dataset.num_human_frames) : null;
                    const nPolicy = statsReady && dataset.num_policy_frames != null ? Number(dataset.num_policy_frames) : null;
                    const humanPct = nHuman != null && nPolicy != null && (nHuman + nPolicy) > 0
                      ? Math.round((nHuman / (nHuman + nPolicy)) * 100) : null;
                    return (
                      <div className="flex items-center gap-3 text-xs mt-1 font-mono">
                        <span className="text-teal">
                          {nSuccess}/{total} success ({successPct}%)
                        </span>
                        {nAutonomous > 0 && (
                          <span className="text-teal/60">
                            {nAutonomous} autonomous ({autonomousPct}%)
                          </span>
                        )}
                        {nHuman != null && nPolicy != null && (<>
                          <span className="w-px h-3 bg-warm-200" />
                          <span className="text-ink-muted">
                            {formatFrameCount(nHuman)} human · {formatFrameCount(nPolicy)} policy frames ({humanPct}% human)
                          </span>
                        </>)}
                      </div>
                    );
                  })()}
                </div>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-ink-muted/40"
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
