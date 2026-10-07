import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "../lib/arenaClient";
import { api } from "../release/api";
import { useSearchParam, useSearchParamNumber } from "../lib/useSearchParam";
import {
  FPS,
  explorerCameraKeys,
  fetchAppliedProgress,
  fetchEpisodeFrameSignals,
  fetchLabelHistory,
  fetchLedgerArms,
  fetchReviewEpisodes,
  selectPrimaryCameraKey,
  type AppliedProgress,
  type EpisodeFrameSignals,
  type LabelEvent,
  type ReviewEpisode,
} from "../lib/hf-api";
import { placeSubtaskMark } from "../lib/subtaskMarks";
import { CommitPanel } from "./review/CommitPanel";
import { EpisodeNotes } from "./review/EpisodeNotes";
import { HelpOverlay } from "./review/HelpOverlay";
import { LabelHistoryPanel } from "./review/LabelHistoryPanel";
import { describeLabelPayload, sourceLabel } from "./review/labelHistory";
import { ReviewViewer, type ViewerControls } from "./review/ReviewViewer";
import {
  cameraRoleForVideoKey,
  clamp,
  formatClock,
  orderCameraKeys,
  type CropBox,
} from "./review/format";
import { isTypingTarget, useWindowKeydown } from "./review/useWindowKeydown";

// ---------------------------------------------------------------------------
// Contract notes
//
// This is the operator decision-capture half of mulligan/tools/outcome_review.py.
// It ONLY writes decisions into Convex (api.reviews.save) and enqueues apply
// jobs (api.applyJobs.enqueue). Materializing `.outcome_edit_progress.json`,
// rewriting reward/done/is_valid and pushing to HuggingFace all happen in the
// Python apply worker — nothing here touches the dataset.
// ---------------------------------------------------------------------------

type Outcome = "success" | "failure" | "timeout";
type QueueFilter = "all" | "failure" | "success" | "timeout";

const OUTCOMES: Outcome[] = ["success", "failure", "timeout"];

/**
 * Required mid-episode subtask marks, keyed by the Convex dataset `task` —
 * FALLBACK ONLY, used while the exported `taskSpecs` row is loading or for a
 * task the exporter has not pushed yet.
 *
 * AUTHORITY IS PYTHON: `resolve_subtask_marks` in mulligan/tools/outcome_review.py
 * reads `RealTaskSpec.num_subtask_marks` (exported to Convex by
 * mulligan/tools/export_arena_task_specs.py), and the apply worker RE-VALIDATES every
 * confirmed record with `subtask_mark_count_error` before touching HuggingFace.
 * This table only makes the gate visible to the operator while they review; a
 * stale entry here cannot write a bad label — it can only mis-guide the UI, and
 * the worker will reject the job loudly.
 */
const REVIEW_SUBTASK_MARKS: Record<string, number> = {
  routing_d2: 1,
};

const OUTCOME_CHIP: Record<Outcome, string> = {
  success: "bg-teal-light text-teal",
  failure: "bg-coral-light text-coral",
  timeout: "bg-gold-light text-gold",
};

// Palette hex values mirror the @theme block in src/index.css (teal / coral / gold).
const OUTCOME_HEX: Record<Outcome, string> = {
  success: "#0B6E6E",
  failure: "#D4654A",
  timeout: "#C4961A",
};

const QUEUE_FILTERS: { id: QueueFilter; label: string }[] = [
  { id: "failure", label: "Failures" },
  { id: "success", label: "Successes" },
  { id: "timeout", label: "Timeouts" },
  { id: "all", label: "All" },
];

const SIGNAL_CONCURRENCY = 3;

/**
 * Legality of `n` subtask marks for `outcome` — mirrors
 * `subtask_mark_count_error` in mulligan/real/eval/outcome_results.py.
 */
function subtaskMarkCountError(
  outcome: Outcome,
  n: number,
  required: number
): string | null {
  if (required <= 0) return null;
  if (outcome === "success") {
    if (n !== required) {
      return `a SUCCESS episode must carry exactly ${required} subtask mark(s), got ${n}`;
    }
    return null;
  }
  if (n > required) {
    return `a ${outcome.toUpperCase()} episode may carry at most ${required} subtask mark(s), got ${n}`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Review records (Convex rows, BigInt → number)
// ---------------------------------------------------------------------------

interface ReviewRecord {
  episodeIndex: number;
  status: string;
  newOutcome: Outcome | null;
  outcomeFrame: number | null;
  softTruncate: boolean;
  subtaskFrames: number[] | null;
  reviewer: string;
  savedAt: number;
}

interface PendingReview {
  outcome: Outcome | null;
  markedFrame: number | null;
  subtaskFrames: number[];
  softTruncate: boolean;
}

const EMPTY_PENDING: PendingReview = {
  outcome: null,
  markedFrame: null,
  subtaskFrames: [],
  softTruncate: false,
};

// ---------------------------------------------------------------------------
// Work queue row
// ---------------------------------------------------------------------------

function QueueRow({
  episode,
  signals,
  signalError,
  review,
  applied,
  selected,
  onSelect,
  arm,
}: {
  episode: ReviewEpisode;
  signals: EpisodeFrameSignals | null;
  signalError: string | null;
  review: ReviewRecord | null;
  /** Applied HF-record state when no web review exists: outcome, or "skip". */
  applied: string | null;
  selected: boolean;
  onSelect: () => void;
  arm: string | null;
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-3 py-2 rounded-lg border transition-all cursor-pointer ${
        selected
          ? "bg-teal/10 border-teal shadow-sm"
          : "bg-white border-warm-200 hover:border-warm-300"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs font-medium text-ink">
          Ep {episode.episodeIndex}
        </span>
        {signalError ? (
          <span
            className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-coral-light text-coral"
            title={signalError}
          >
            error
          </span>
        ) : signals ? (
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${OUTCOME_CHIP[signals.detectedOutcome]}`}
          >
            {signals.detectedOutcome}
          </span>
        ) : (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-warm-100 text-ink-muted animate-pulse">
            …
          </span>
        )}
      </div>
      <div className="flex items-center justify-between gap-2 mt-1">
        <span className="text-[10px] font-mono text-ink-muted truncate">
          {episode.rawLength}f
          {arm ? ` · ${arm}` : ""}
        </span>
        {review === null && applied !== null ? (
          <span
            className="text-[10px] font-mono text-teal/70"
            title="Already treated on HuggingFace (applied outcome-edit record)"
          >
            {applied === "skip" ? "skipped" : `✓ ${applied}`} ·applied
          </span>
        ) : review === null ? (
          <span className="text-[10px] font-mono text-ink-muted/60">
            unreviewed
          </span>
        ) : review.status === "confirmed" ? (
          <span
            className="text-[10px] font-mono text-teal"
            title={`${review.reviewer} · ${formatClock(review.savedAt)}`}
          >
            ✓ {review.newOutcome}
            {review.softTruncate ? " ·trunc" : ""}
          </span>
        ) : (
          <span
            className="text-[10px] font-mono text-ink-muted"
            title={`${review.reviewer} · ${formatClock(review.savedAt)}`}
          >
            skipped
          </span>
        )}
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Help overlay
// ---------------------------------------------------------------------------

const HELP_KEYS: [string, string][] = [
  ["← / →", "step 1 frame (shift: 10)"],
  ["[ / ]", "step 10 frames"],
  ["Home / End", "first / last frame"],
  ["space", "play / pause"],
  ["s / f / t", "set outcome success / failure / timeout + mark here"],
  ["m", "move the outcome mark to this frame"],
  ["g", "subtask mark here (moves the nearest one when full; on a marked frame: remove)"],
  ["x", "toggle soft truncation"],
  ["u", "unmark: reset to the detected outcome"],
  ["c", "confirm + save, advance to the next episode"],
  ["n", "skip (twice when subtask marks are unsaved)"],
  ["p / b", "previous episode in the queue"],
  ["q / Esc", "exit review mode"],
  ["?", "toggle this help"],
];

// ---------------------------------------------------------------------------
// Outcome review (main)
// ---------------------------------------------------------------------------

export default function OutcomeReview({
  repoId,
  task,
  onExit,
}: {
  repoId: string;
  task?: string;
  onExit: () => void;
}) {
  const viewer = useQuery(api.users.viewer);
  const reviews = useQuery(api.reviews.latestForRepo, { dataset_repo: repoId });
  const taskSpec = useQuery(api.taskSpecs.forTask, task ? { task } : "skip");
  const saveReview = useMutation(api.reviews.save);

  const [episodes, setEpisodes] = useState<ReviewEpisode[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [ledgerArms, setLedgerArms] = useState<Map<number, string> | null>(null);
  const [ledgerError, setLedgerError] = useState<string | null>(null);
  const [armFilter, setArmFilter] = useSearchParam("arm", "all");
  // The APPLIED record on HF (written by the apply worker or the Python editor) — the
  // record of truth for treatment that already reached the Hub. Without it the
  // queue calls fully-treated datasets "unreviewed". Tri-state: undefined =
  // still loading, null = dataset has no record (never treated).
  const [applied, setApplied] = useState<AppliedProgress | null | undefined>(undefined);
  const [appliedError, setAppliedError] = useState<string | null>(null);
  // First-time flow: only never-addressed episodes (no web review, no applied
  // record entry). Switch to "all" to revisit treated episodes.
  const [statusFilter, setStatusFilter] = useSearchParam("status", "unaddressed");
  // Append-only label-provenance ledger (who/when/how for every label change).
  // Supplementary: absence or load failure never blocks reviewing.
  const [labelHistory, setLabelHistory] = useState<LabelEvent[] | null | undefined>(undefined);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [signals, setSignals] = useState<Map<number, EpisodeFrameSignals>>(
    () => new Map()
  );
  const [signalErrors, setSignalErrors] = useState<Map<number, string>>(
    () => new Map()
  );
  // Default queue is every outcome; the status filter ("unaddressed") already
  // narrows the first-time flow to episodes nobody has treated yet.
  const [filter, setFilter] = useSearchParam("queue", "all");
  const [selectedEpisode, setSelectedEpisode] = useSearchParamNumber("episode");
  const [frame, setFrame] = useState(0);
  const [pending, setPending] = useState<PendingReview>(EMPTY_PENDING);
  const [viewerDrift, setViewerDrift] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [skipArmed, setSkipArmed] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const controlsRef = useRef<ViewerControls | null>(null);
  const queueScrollRef = useRef<HTMLDivElement | null>(null);

  // Exported registry row wins; hardcoded map only bridges MISSING rows.
  // Convex useQuery is undefined WHILE LOADING and null for a missing row —
  // only the latter may fall back. Treating "loading" as 0 marks hides
  // subtask work and lets structurally-invalid labels through, so actions
  // and the unaddressed filter gate on specReady.
  const specReady = !task || taskSpec !== undefined;
  const subtaskMarksRequired = task
    ? taskSpec != null
      ? Number(taskSpec.num_subtask_marks)
      : (REVIEW_SUBTASK_MARKS[task] ?? 0)
    : 0;

  // -- Episode metadata --------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setEpisodes(null);
    setLoadError(null);
    fetchReviewEpisodes(repoId)
      .then((result) => {
        if (!cancelled) setEpisodes(result);
      })
      .catch((err: Error) => {
        if (!cancelled) setLoadError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId]);

  // -- Applied outcome-edit record from HF (absence = never treated) --------
  useEffect(() => {
    let cancelled = false;
    setApplied(undefined);
    setAppliedError(null);
    fetchAppliedProgress(repoId)
      .then((progress) => {
        if (!cancelled) setApplied(progress);
      })
      .catch((err: Error) => {
        // A present-but-unreadable record must be loud: silently treating a
        // fully-reviewed dataset as unaddressed invites duplicate review work.
        if (!cancelled) setAppliedError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId]);

  // -- Label-history ledger (provenance; absence is fine) -------------------
  useEffect(() => {
    let cancelled = false;
    setLabelHistory(undefined);
    setHistoryError(null);
    fetchLabelHistory(repoId)
      .then((events) => {
        if (!cancelled) setLabelHistory(events);
      })
      .catch((err: Error) => {
        if (!cancelled) setHistoryError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId]);

  const historyByEpisode = useMemo(() => {
    const byEpisode = new Map<number, LabelEvent[]>();
    for (const event of labelHistory ?? []) {
      const list = byEpisode.get(event.episode_index);
      if (list) list.push(event);
      else byEpisode.set(event.episode_index, [event]);
    }
    // "Current" = last by TIMESTAMP, not file order — a bootstrap appended
    // around live events must not misreport the current label source.
    for (const list of byEpisode.values()) {
      list.sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts));
    }
    return byEpisode;
  }, [labelHistory]);

  // -- Ledger arm labels (optional sidecars; absence is fine) --------------
  useEffect(() => {
    let cancelled = false;
    setLedgerArms(null);
    setLedgerError(null);
    fetchLedgerArms(repoId)
      .then((arms) => {
        if (!cancelled) setLedgerArms(arms);
      })
      .catch((err: Error) => {
        // A malformed ledger must be visible, not silently unfiltered.
        if (!cancelled) setLedgerError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId]);

  // -- Detected outcomes, progressively in the background -----------------
  useEffect(() => {
    if (!episodes) return;
    let cancelled = false;
    const queue = [...episodes];
    const worker = async () => {
      while (!cancelled) {
        const episode = queue.shift();
        if (!episode) return;
        try {
          const result = await fetchEpisodeFrameSignals(
            repoId,
            episode.dataPath,
            episode.episodeIndex
          );
          if (cancelled) return;
          setSignals((prev) => new Map(prev).set(episode.episodeIndex, result));
        } catch (err) {
          if (cancelled) return;
          // Surfaced in the queue row + a banner; never defaulted to an outcome.
          setSignalErrors((prev) =>
            new Map(prev).set(episode.episodeIndex, (err as Error).message)
          );
        }
      }
    };
    void Promise.all(
      Array.from({ length: SIGNAL_CONCURRENCY }, () => worker())
    );
    return () => {
      cancelled = true;
    };
  }, [episodes, repoId]);

  // -- Convex reviews ----------------------------------------------------
  const reviewByEpisode = useMemo(() => {
    const map = new Map<number, ReviewRecord>();
    for (const row of reviews?.episodes ?? []) {
      map.set(Number(row.episode_index), {
        episodeIndex: Number(row.episode_index),
        status: row.status,
        newOutcome: (row.new_outcome as Outcome | undefined) ?? null,
        outcomeFrame:
          row.outcome_frame != null ? Number(row.outcome_frame) : null,
        softTruncate: row.soft_truncate ?? false,
        subtaskFrames: row.subtask_frames
          ? row.subtask_frames.map((value) => Number(value))
          : null,
        reviewer: row.reviewer,
        savedAt: row.saved_at,
      });
    }
    return map;
  }, [reviews]);

  const effectiveOutcome = useCallback(
    (episodeIndex: number): Outcome | null => {
      const review = reviewByEpisode.get(episodeIndex);
      if (review?.status === "confirmed" && review.newOutcome) {
        return review.newOutcome;
      }
      return signals.get(episodeIndex)?.detectedOutcome ?? null;
    },
    [reviewByEpisode, signals]
  );

  const armOptions = useMemo(
    () => (ledgerArms ? [...new Set(ledgerArms.values())].sort() : []),
    [ledgerArms]
  );

  // Addressed = a web review exists OR the applied HF record already carries
  // the episode (changed or skipped — an applied skip is a deliberate
  // keep-labels-as-is decision, not an omission). On a subtask task a changed
  // record WITHOUT the subtask_frames key was not reviewed in subtask mode and
  // must re-queue, exactly like episode_fully_processed in
  // mulligan/tools/outcome_review.py.
  const isAddressed = useCallback(
    (episodeIndex: number): boolean => {
      if (reviewByEpisode.has(episodeIndex)) return true;
      if (applied == null) return false;
      if (applied.skipped.has(episodeIndex)) return true;
      const record = applied.changed.get(episodeIndex);
      if (record === undefined) return false;
      return subtaskMarksRequired <= 0 || record.subtaskFrames !== null;
    },
    [reviewByEpisode, applied, subtaskMarksRequired]
  );

  const filteredEpisodes = useMemo(() => {
    if (!episodes) return [];
    let result = episodes;
    if (statusFilter === "unaddressed") {
      // Addressed-ness is unknown until the applied record AND the task spec
      // settle (isAddressed's subtask-key rule needs subtaskMarksRequired) —
      // an early queue here would flash fully-treated episodes. A load ERROR
      // falls through unfiltered (the loud banner explains treated episodes
      // may show as unaddressed) rather than blanking the queue forever.
      if ((applied === undefined && appliedError === null) || !specReady) return [];
      if (applied !== undefined) {
        result = result.filter((episode) => !isAddressed(episode.episodeIndex));
      }
    }
    if (filter !== "all") {
      result = result.filter(
        (episode) => effectiveOutcome(episode.episodeIndex) === filter
      );
    }
    // A stale ?arm= from another dataset must not silently empty the queue:
    // only apply the filter when the value exists in THIS dataset's ledger.
    if (armFilter !== "all" && ledgerArms && armOptions.includes(armFilter)) {
      result = result.filter(
        (episode) => ledgerArms.get(episode.episodeIndex) === armFilter
      );
    }
    return result;
  }, [
    episodes,
    statusFilter,
    applied,
    appliedError,
    specReady,
    isAddressed,
    filter,
    effectiveOutcome,
    armFilter,
    ledgerArms,
    armOptions,
  ]);

  const currentEpisode = useMemo(
    () =>
      episodes?.find((episode) => episode.episodeIndex === selectedEpisode) ??
      null,
    [episodes, selectedEpisode]
  );
  const currentSignals =
    selectedEpisode !== null ? (signals.get(selectedEpisode) ?? null) : null;
  const currentSignalError =
    selectedEpisode !== null ? (signalErrors.get(selectedEpisode) ?? null) : null;

  // Pull the selected episode's signals to the front of the queue. Keyed on a
  // boolean, not the signals map, so unrelated background arrivals do not
  // re-fire this fetch on every update.
  const currentSignalsMissing =
    currentEpisode !== null && !signals.has(currentEpisode.episodeIndex);
  useEffect(() => {
    if (!currentEpisode || !currentSignalsMissing) return;
    let cancelled = false;
    fetchEpisodeFrameSignals(
      repoId,
      currentEpisode.dataPath,
      currentEpisode.episodeIndex
    )
      .then((result) => {
        if (!cancelled) {
          setSignals((prev) =>
            new Map(prev).set(currentEpisode.episodeIndex, result)
          );
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setSignalErrors((prev) =>
            new Map(prev).set(currentEpisode.episodeIndex, err.message)
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [currentEpisode, currentSignalsMissing, repoId]);

  // Auto-select the head of the queue when nothing is selected yet.
  useEffect(() => {
    if (selectedEpisode !== null) return;
    const head = filteredEpisodes[0];
    if (head) setSelectedEpisode(head.episodeIndex);
  }, [selectedEpisode, filteredEpisodes, setSelectedEpisode]);

  // Keep keyboard-driven queue navigation visible without moving the page.
  // scrollIntoView would also jump vertically to the queue on episode changes.
  useEffect(() => {
    const container = queueScrollRef.current;
    if (container === null || selectedEpisode === null) return;
    const selected = container.querySelector<HTMLElement>(
      `[data-episode-index="${selectedEpisode}"]`
    );
    if (selected === null) return;
    const itemLeft = selected.offsetLeft;
    const itemRight = itemLeft + selected.offsetWidth;
    const visibleLeft = container.scrollLeft;
    const visibleRight = visibleLeft + container.clientWidth;
    if (itemLeft < visibleLeft) {
      container.scrollTo({ left: itemLeft, behavior: "smooth" });
    } else if (itemRight > visibleRight) {
      container.scrollTo({
        left: itemRight - container.clientWidth,
        behavior: "smooth",
      });
    }
  }, [
    selectedEpisode,
    statusFilter,
    filter,
    armFilter,
    filteredEpisodes.length,
  ]);

  // -- Prefill on episode open -------------------------------------------
  const prefilledFor = useRef<number | null>(null);

  // Selection change: IMMEDIATELY clear the previous episode's decision state.
  // The prefill below waits on async loads (signals, applied record); without
  // this reset the old episode's tint/chips/marks render over the NEW
  // episode's video for seconds, and edits made in that gap are discarded.
  useEffect(() => {
    setPending(EMPTY_PENDING);
    setFrame(0);
    setDirty(false);
    setSkipArmed(false);
    setViewerDrift(null);
  }, [selectedEpisode]);

  useEffect(() => {
    if (selectedEpisode === null || !currentSignals) return;
    // The applied HF record outranks detected signals in the prefill — wait
    // for it to settle (a load error falls through, with its loud banner).
    if (applied === undefined && appliedError === null) return;
    if (prefilledFor.current === selectedEpisode) return;
    prefilledFor.current = selectedEpisode;
    // Opening an episode resets the operator's working state to the prefill:
    // web review > applied HF record (as the Python editor resumes) > detected signals.
    const review = reviewByEpisode.get(selectedEpisode);
    const appliedRecord = applied?.changed.get(selectedEpisode);
    if (review?.status === "confirmed" && review.newOutcome) {
      setPending({
        outcome: review.newOutcome,
        markedFrame: review.outcomeFrame,
        subtaskFrames: review.subtaskFrames ?? [],
        softTruncate: review.softTruncate,
      });
    } else if (
      // A web SKIP means keep-labels-as-is: the as-is state IS the applied
      // record when one exists (falling to detected signals here would show
      // the pre-edit outcome and a stray confirm could revert an applied edit).
      (review === undefined || review.status === "skipped") &&
      appliedRecord !== undefined
    ) {
      setPending({
        outcome: appliedRecord.newOutcome as Outcome,
        markedFrame: appliedRecord.outcomeFrame,
        subtaskFrames: appliedRecord.subtaskFrames ?? [],
        softTruncate: appliedRecord.softTruncate,
      });
    } else {
      // No review and no applied record: any reward spike in the parquet is a
      // LIVE operator sub-goal mark (eval-time 'g' / numpad '3' press, written by
      // finalize_episode_data). Prefill it so review is confirm + optional nudge;
      // a confirm with an empty list would have the apply erase the spike.
      setPending({
        outcome: currentSignals.detectedOutcome,
        markedFrame:
          currentSignals.doneOnsetFrame ?? currentSignals.lastValidFrame,
        subtaskFrames: [...currentSignals.rewardSpikeFrames],
        softTruncate: false,
      });
    }
    setFrame(0);
    setDirty(false);
    setSkipArmed(false);
    setActionError(null);
  }, [selectedEpisode, currentSignals, reviewByEpisode, applied, appliedError]);

  // -- Actions -----------------------------------------------------------
  const selectEpisode = useCallback(
    (episodeIndex: number) => {
      setSelectedEpisode(episodeIndex);
    },
    [setSelectedEpisode]
  );

  const advance = useCallback(
    (fromIndex: number) => {
      const position = filteredEpisodes.findIndex(
        (episode) => episode.episodeIndex === fromIndex
      );
      // Out-of-queue (e.g. a deep-linked addressed episode under the
      // unaddressed filter): advancing would silently jump to the queue HEAD
      // and walk already-done work. Stay put instead.
      if (position === -1) return;
      const next = filteredEpisodes[position + 1];
      if (next) selectEpisode(next.episodeIndex);
    },
    [filteredEpisodes, selectEpisode]
  );

  const confirm = useCallback(async () => {
    if (selectedEpisode === null || saving) return;
    // The pending state still holds the PREVIOUS episode's decision until this
    // episode's signals arrive and the prefill effect runs — confirming in
    // that window would record the old outcome/frame onto the new episode.
    if (prefilledFor.current !== selectedEpisode) {
      setActionError(
        `Episode ${selectedEpisode} is still loading — wait for the outcome prefill.`
      );
      return;
    }
    if (!specReady) {
      // Confirming against the fallback mark count could save a structurally
      // invalid label (e.g. a subtask success with no marks, key omitted).
      setActionError("Task spec still loading — wait before confirming.");
      return;
    }
    if (viewerDrift !== null) {
      setActionError(
        `Video/frame-counter drift detected (${viewerDrift}) — reload before confirming; ` +
          "the displayed frame may not be the counted frame."
      );
      return;
    }
    if (!pending.outcome) {
      setActionError("Set an outcome (s / f / t) before confirming.");
      return;
    }
    if (pending.markedFrame === null) {
      setActionError("Mark the outcome frame (m) before confirming.");
      return;
    }
    if (subtaskMarksRequired > 0) {
      const countError = subtaskMarkCountError(
        pending.outcome,
        pending.subtaskFrames.length,
        subtaskMarksRequired
      );
      if (countError) {
        setActionError(
          `Cannot confirm: ${countError}; press g to place a subtask mark.`
        );
        return;
      }
      const late = pending.subtaskFrames.filter((mark) => {
        const outcomeFrame = pending.markedFrame as number;
        return mark > outcomeFrame || (mark === outcomeFrame && pending.outcome !== "timeout");
      });
      if (late.length > 0) {
        setActionError(
          `Subtask mark(s) ${late.join(", ")} must be before the outcome frame ` +
            `${pending.markedFrame} (equality is allowed only for timeout); ` +
            `move them earlier (g to toggle).`
        );
        return;
      }
    } else if (pending.subtaskFrames.length > 0) {
      setActionError(
        `Task ${task ?? "(unknown)"} takes no subtask marks; remove them with g.`
      );
      return;
    }

    setSaving(true);
    setActionError(null);
    try {
      await saveReview({
        dataset_repo: repoId,
        episode_index: BigInt(selectedEpisode),
        status: "confirmed",
        new_outcome: pending.outcome,
        outcome_frame: BigInt(pending.markedFrame),
        soft_truncate: pending.softTruncate,
        // The key's PRESENCE marks this as reviewed in subtask mode, exactly
        // like the desktop editor's progress record. Omit it when N == 0.
        subtask_frames:
          subtaskMarksRequired > 0
            ? pending.subtaskFrames.map((mark) => BigInt(mark))
            : undefined,
      });
      setDirty(false);
      advance(selectedEpisode);
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }, [
    advance,
    pending,
    repoId,
    saveReview,
    saving,
    selectedEpisode,
    specReady,
    subtaskMarksRequired,
    task,
    viewerDrift,
  ]);

  const skip = useCallback(async () => {
    if (selectedEpisode === null || saving) return;
    // Same prefill-race guard as confirm: pending may still be the previous
    // episode's state (its subtask marks would trigger a bogus double-n arm).
    if (prefilledFor.current !== selectedEpisode) {
      setActionError(
        `Episode ${selectedEpisode} is still loading — wait for the outcome prefill.`
      );
      return;
    }
    if (!specReady) {
      setActionError("Task spec still loading — wait before skipping.");
      return;
    }
    if (pending.subtaskFrames.length > 0 && !skipArmed) {
      // Same double-n guard as the Python outcome editor: skipping would silently drop
      // marks the operator already placed.
      setSkipArmed(true);
      setActionError(
        `Episode ${selectedEpisode} has ${pending.subtaskFrames.length} unsaved subtask ` +
          `mark(s) at ${pending.subtaskFrames.join(", ")}. Press c to SAVE, or n again to DISCARD.`
      );
      return;
    }
    setSaving(true);
    setActionError(null);
    try {
      await saveReview({
        dataset_repo: repoId,
        episode_index: BigInt(selectedEpisode),
        status: "skipped",
      });
      setDirty(false);
      setSkipArmed(false);
      advance(selectedEpisode);
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }, [
    advance,
    pending.subtaskFrames,
    repoId,
    saveReview,
    saving,
    selectedEpisode,
    skipArmed,
    specReady,
  ]);

  // Stepping while playing would desync the frame counter from the video, so a
  // navigation key stops playback first (the Python outcome editor does the same).
  const stepFrame = useCallback(
    (delta: number) => {
      if (!currentEpisode) return;
      controlsRef.current?.pause();
      setFrame((prev) => clamp(prev + delta, 0, currentEpisode.rawLength - 1));
    },
    [currentEpisode]
  );

  const jumpToFrame = useCallback((next: number) => {
    controlsRef.current?.pause();
    setFrame(next);
  }, []);

  const updatePending = useCallback((update: Partial<PendingReview>) => {
    setPending((prev) => ({ ...prev, ...update }));
    setDirty(true);
  }, []);

  // -- Keyboard ----------------------------------------------------------
  function handleKey(event: KeyboardEvent) {
    if (isTypingTarget(event)) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    const key = event.key;
    if (key !== "n") setSkipArmed(false);

    if (showHelp && (key === "Escape" || key === "q" || key === "?")) {
      event.preventDefault();
      setShowHelp(false);
      return;
    }
    if (key === "Escape" || key === "q") {
      event.preventDefault();
      onExit();
      return;
    }
    if (key === "?") {
      event.preventDefault();
      setShowHelp(true);
      return;
    }
    if (!currentEpisode) return;

    switch (key) {
      case "ArrowLeft":
        event.preventDefault();
        stepFrame(event.shiftKey ? -10 : -1);
        return;
      case "ArrowRight":
        event.preventDefault();
        stepFrame(event.shiftKey ? 10 : 1);
        return;
      case "[":
        event.preventDefault();
        stepFrame(-10);
        return;
      case "]":
        event.preventDefault();
        stepFrame(10);
        return;
      case "Home":
        event.preventDefault();
        jumpToFrame(0);
        return;
      case "End":
        event.preventDefault();
        jumpToFrame(currentEpisode.rawLength - 1);
        return;
      case " ":
        event.preventDefault();
        controlsRef.current?.togglePlay();
        return;
      case "s":
      case "f":
      case "t": {
        event.preventDefault();
        // During playback the `frame` state is FROZEN at the play-start value;
        // pause() returns the actually-displayed frame — mark there.
        const markAt = controlsRef.current?.pause() ?? frame;
        const outcome: Outcome =
          key === "s" ? "success" : key === "f" ? "failure" : "timeout";
        updatePending({ outcome, markedFrame: markAt });
        setActionError(null);
        return;
      }
      case "m":
      case "Enter": {
        event.preventDefault();
        const markAt = controlsRef.current?.pause() ?? frame;
        updatePending({ markedFrame: markAt });
        setActionError(null);
        return;
      }
      case "g": {
        event.preventDefault();
        if (!specReady) {
          setActionError("Task spec still loading — wait before placing subtask marks.");
          return;
        }
        if (subtaskMarksRequired <= 0) {
          setActionError(
            `Task ${task ?? "(unknown)"} defines 0 subtask marks (RealTaskSpec.num_subtask_marks).`
          );
          return;
        }
        const markAt = controlsRef.current?.pause() ?? frame;
        // Add below the cap; at the cap, MOVE the nearest mark here; on a
        // marked frame, remove it (rule + tests in lib/subtaskMarks.ts).
        updatePending({
          subtaskFrames: placeSubtaskMark(pending.subtaskFrames, markAt, subtaskMarksRequired),
        });
        setActionError(null);
        return;
      }
      case "x":
        event.preventDefault();
        updatePending({ softTruncate: !pending.softTruncate });
        return;
      case "u":
        event.preventDefault();
        if (!currentSignals) return;
        updatePending({
          outcome: currentSignals.detectedOutcome,
          markedFrame: null,
          subtaskFrames: [],
          softTruncate: false,
        });
        setActionError(null);
        return;
      case "c":
        event.preventDefault();
        void confirm();
        return;
      case "n":
        event.preventDefault();
        void skip();
        return;
      case "p":
      case "b": {
        event.preventDefault();
        const position = filteredEpisodes.findIndex(
          (episode) => episode.episodeIndex === currentEpisode.episodeIndex
        );
        if (position === -1) {
          setActionError(
            "This episode is not in the current queue; pick one from the list."
          );
          return;
        }
        const previous = filteredEpisodes[position - 1];
        if (previous) selectEpisode(previous.episodeIndex);
        else setActionError("Already at the first episode in the queue.");
        return;
      }
      default:
        return;
    }
  }

  useWindowKeydown(handleKey);

  // -- Render ------------------------------------------------------------
  const cameraKeys = useMemo(() => {
    if (!currentEpisode) return [];
    const all = orderCameraKeys(
      explorerCameraKeys(Object.keys(currentEpisode.perCamera))
    );
    // Task-default review cameras (RealTaskSpec.consumed_camera_roles, e.g.
    // marker_d2 -> side_1 + wrist_left), in the spec's display order. Fall
    // back to every stream when the spec has no default or nothing matches
    // (e.g. serial-named camera keys).
    const roles = taskSpec?.review_camera_roles;
    if (roles == null || roles.length === 0) return all;
    const keysByRole = taskSpec!.camera_keys_by_role;
    const selected = roles
      .map((role) =>
        all.find((key) => cameraRoleForVideoKey(key, keysByRole) === role)
      )
      .filter((key): key is string => key !== undefined);
    return selected.length > 0 ? selected : all;
  }, [currentEpisode, taskSpec]);

  // A PARTIAL role match must not pass silently: the task registry says the
  // review needs these views, and judging outcomes from fewer is a banner-
  // worthy degradation (e.g. a role hidden by the explorer's key filter).
  const missingCameraRoles = useMemo(() => {
    const roles = taskSpec?.review_camera_roles;
    if (!currentEpisode || roles == null || roles.length === 0) return [];
    const keysByRole = taskSpec!.camera_keys_by_role;
    const all = orderCameraKeys(
      explorerCameraKeys(Object.keys(currentEpisode.perCamera))
    );
    return roles.filter(
      (role) => !all.some((key) => cameraRoleForVideoKey(key, keysByRole) === role)
    );
  }, [currentEpisode, taskSpec]);
  const primaryKey = useMemo(
    () => (cameraKeys.length > 0 ? selectPrimaryCameraKey(cameraKeys) : ""),
    [cameraKeys]
  );

  // Station display crops from the exported task spec, resolved per video key
  // (role- or serial-named), in stored-frame pixel space.
  const storedFrameHW = useMemo<[number, number] | null>(
    () =>
      taskSpec != null
        ? [Number(taskSpec.stored_frame_hw[0]), Number(taskSpec.stored_frame_hw[1])]
        : null,
    [taskSpec]
  );
  const cropByCameraKey = useMemo<Record<string, CropBox> | null>(() => {
    if (taskSpec == null) return null;
    const keysByRole = taskSpec.camera_keys_by_role;
    const map: Record<string, CropBox> = {};
    for (const key of cameraKeys) {
      const role = cameraRoleForVideoKey(key, keysByRole);
      const box = role !== null ? taskSpec.crop_boxes[role] : undefined;
      if (box !== undefined) {
        map[key] = box.map(Number) as CropBox;
      }
    }
    return map;
  }, [taskSpec, cameraKeys]);

  // Decision overlays (as in the Python outcome editor), injected into ReviewViewer
  // (which calls them only while paused): frames at/after the pending outcome
  // frame get a translucent outcome tint (denser strictly-after the mark when
  // soft-truncate is on); a pending subtask frame gets a solid border + tag.
  const renderVideoOverlay = (overlayFrame: number) => {
    const decisionTint =
      pending.outcome !== null &&
      pending.markedFrame !== null &&
      overlayFrame >= pending.markedFrame
        ? {
            color: OUTCOME_HEX[pending.outcome],
            alpha:
              pending.softTruncate && overlayFrame > pending.markedFrame ? 0.4 : 0.2,
          }
        : null;
    const isSubtaskFrame = pending.subtaskFrames.includes(overlayFrame);
    return (
      <>
        {decisionTint && (
          <div
            className="absolute inset-0 pointer-events-none rounded-lg"
            style={{
              backgroundColor: decisionTint.color,
              opacity: decisionTint.alpha,
            }}
          />
        )}
        {isSubtaskFrame && (
          <div className="absolute inset-0 pointer-events-none rounded-lg border-[6px] border-purple-600">
            <span className="absolute bottom-2 right-2 px-1.5 py-0.5 rounded bg-purple-600 text-white text-[10px] font-mono font-medium">
              SUBTASK REWARD
            </span>
          </div>
        )}
      </>
    );
  };

  // Outcome-specific timeline markers (soft-trunc hatch, done onset, reward
  // spikes, subtask dots, outcome marker) — same nodes the pre-extraction
  // Timeline rendered inline.
  const renderTimelineOverlays = (pct: (value: number) => string) => {
    if (!currentEpisode) return null;
    const rawLength = currentEpisode.rawLength;
    const truncStart =
      pending.softTruncate && pending.markedFrame !== null
        ? pending.markedFrame + 1
        : null;
    return (
      <>
        {/* Region a confirmed soft truncation would invalidate */}
        {truncStart !== null && truncStart < rawLength && (
          <div
            className="absolute top-0 bottom-0"
            style={{
              left: pct(truncStart),
              right: 0,
              backgroundImage:
                "repeating-linear-gradient(-45deg, rgba(212,101,74,0.35) 0 5px, transparent 5px 10px)",
            }}
            title={`soft truncation would drop frames ${truncStart}..${rawLength - 1}`}
          />
        )}

        {/* Existing done==1 onset */}
        {currentSignals?.doneOnsetFrame != null && (
          <div
            className="absolute top-0 bottom-0 border-l border-dashed border-ink-muted"
            style={{ left: pct(currentSignals.doneOnsetFrame) }}
            title={`existing outcome frame (done==1 onset) ${currentSignals.doneOnsetFrame}`}
          />
        )}

        {/* Existing mid-episode reward spikes (hollow) */}
        {currentSignals?.rewardSpikeFrames.map((spike) => (
          <div
            key={`spike-${spike}`}
            className="absolute bottom-1 w-2 h-2 rounded-full border border-ink-muted"
            style={{ left: pct(spike), transform: "translateX(-50%)" }}
            title={`existing reward spike at frame ${spike}`}
          />
        ))}

        {/* Pending subtask marks */}
        {pending.subtaskFrames.map((mark) => (
          <div
            key={`mark-${mark}`}
            className="absolute bottom-1 w-2.5 h-2.5 rounded-full bg-purple-600"
            style={{ left: pct(mark), transform: "translateX(-50%)" }}
            title={`subtask mark at frame ${mark}`}
          />
        ))}

        {/* Pending outcome marker */}
        {pending.markedFrame !== null && pending.outcome !== null && (
          <>
            <div
              className="absolute top-0 bottom-0 w-px"
              style={{
                left: pct(pending.markedFrame),
                backgroundColor: OUTCOME_HEX[pending.outcome],
              }}
            />
            <div
              className="absolute top-0"
              style={{
                left: pct(pending.markedFrame),
                transform: "translateX(-50%)",
                width: 0,
                height: 0,
                borderLeft: "6px solid transparent",
                borderRight: "6px solid transparent",
                borderTop: `9px solid ${OUTCOME_HEX[pending.outcome]}`,
              }}
              title={`${pending.outcome} @ frame ${pending.markedFrame}`}
            />
          </>
        )}
      </>
    );
  };

  const numConfirmed = reviews?.num_confirmed ?? 0;
  const numSkipped = reviews?.num_skipped ?? 0;
  const numLoadedSignals = signals.size;
  const numAddressed = useMemo(
    () =>
      episodes === null || (applied === undefined && appliedError === null) || !specReady
        ? null
        : episodes.filter((episode) => isAddressed(episode.episodeIndex)).length,
    [episodes, applied, appliedError, isAddressed, specReady]
  );

  if (!viewer?.isEditor) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
        <button
          onClick={onExit}
          className="text-xs text-ink-muted hover:text-teal mb-4 cursor-pointer"
        >
          &larr; Back to explorer
        </button>
        <p className="font-body text-ink-muted text-center">
          {viewer === undefined
            ? "Checking permissions…"
            : "Outcome review is limited to allowlisted editors."}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl border border-warm-200 shadow-sm overflow-hidden">
      {showHelp && (
        <HelpOverlay
          title="Review shortcuts"
          keys={HELP_KEYS}
          onClose={() => setShowHelp(false)}
        />
      )}

      {/* Header */}
      <div className="px-6 py-4 border-b border-warm-100 bg-warm-50 flex items-center justify-between gap-4">
        <div>
          <button
            onClick={onExit}
            className="text-xs text-ink-muted hover:text-teal cursor-pointer"
          >
            &larr; Back to explorer
          </button>
          <h2 className="font-display text-xl text-ink mt-1">Outcome review</h2>
          <p className="text-xs text-ink-muted font-mono mt-0.5">
            {repoId}
            {task ? ` · ${task}` : ""} ·{" "}
            {subtaskMarksRequired > 0
              ? `${subtaskMarksRequired} subtask mark(s) required`
              : "no subtask marks"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-ink-muted">
            {dirty
              ? "unsaved changes"
              : selectedEpisode !== null && reviewByEpisode.has(selectedEpisode)
                ? "saved ✓"
                : "no web review yet"}
          </span>
          <button
            onClick={() => setShowHelp(true)}
            className="w-7 h-7 rounded-full border border-warm-200 text-ink-muted hover:text-teal hover:border-teal text-sm cursor-pointer"
            title="Keyboard shortcuts"
          >
            ?
          </button>
        </div>
      </div>

      {loadError && (
        <div className="mx-6 mt-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral font-mono">
          Failed to load episode metadata: {loadError}
        </div>
      )}
      {appliedError && (
        <div className="mx-6 mt-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-xs text-coral font-mono">
          Failed to load the applied outcome-edit record — treated episodes may
          show as unaddressed: {appliedError}
        </div>
      )}
      {historyError && (
        <div className="mx-6 mt-4 rounded-lg border border-gold/40 bg-gold-light px-4 py-3 text-xs text-ink font-mono">
          Label-history ledger failed to load (provenance hidden, reviewing
          unaffected): {historyError}
        </div>
      )}
      {missingCameraRoles.length > 0 && (
        <div className="mx-6 mt-4 rounded-lg border border-gold/40 bg-gold-light px-4 py-3 text-xs text-ink font-mono">
          Task expects review camera(s) {missingCameraRoles.join(", ")} but no
          matching stream was found in this dataset — judging from the
          remaining views only.
        </div>
      )}
      {signalErrors.size > 0 && (
        <div className="mx-6 mt-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-xs text-coral font-mono">
          {signalErrors.size} episode(s) failed frame-signal parsing:{" "}
          {[...signalErrors.entries()]
            .slice(0, 3)
            .map(([index, message]) => `ep ${index}: ${message}`)
            .join(" · ")}
        </div>
      )}

      <div>
        {/* Viewer */}
        <div className="p-5">
          {currentEpisode === null ? (
            <div className="py-16 text-center text-ink-muted font-body">
              Select an episode from the queue to review it.
            </div>
          ) : currentSignalError ? (
            <div className="rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral font-mono">
              Episode {currentEpisode.episodeIndex} frame signals failed to
              parse: {currentSignalError}
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <span className="font-display text-lg text-ink">
                  Episode {currentEpisode.episodeIndex}
                </span>
                {currentSignals ? (
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-medium ${OUTCOME_CHIP[currentSignals.detectedOutcome]}`}
                  >
                    detected {currentSignals.detectedOutcome}
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded-full text-xs bg-warm-100 text-ink-muted animate-pulse">
                    detecting…
                  </span>
                )}
                {(() => {
                  const chain = historyByEpisode.get(currentEpisode.episodeIndex);
                  const last = chain?.[chain.length - 1];
                  if (!last) return null;
                  return (
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-mono bg-warm-100 text-ink-muted"
                      title={`Current label source (${chain!.length} event${chain!.length === 1 ? "" : "s"} in ledger): ${describeLabelPayload(last.label_kind, last.payload)} at ${last.ts}`}
                    >
                      {sourceLabel(last)}
                    </span>
                  );
                })()}
                {reviewByEpisode.get(currentEpisode.episodeIndex) === undefined &&
                  applied != null &&
                  (applied.changed.has(currentEpisode.episodeIndex) ||
                    applied.skipped.has(currentEpisode.episodeIndex)) && (
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-medium bg-teal/10 text-teal"
                      title="This episode's decisions are already applied on HuggingFace; confirming records a NEW review on top"
                    >
                      already applied on HF
                      {applied.skipped.has(currentEpisode.episodeIndex)
                        ? " (skip)"
                        : ""}
                    </span>
                  )}
                {pending === EMPTY_PENDING && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-warm-100 text-ink-muted animate-pulse">
                    loading decision…
                  </span>
                )}
                {pending.outcome && (
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-medium ${OUTCOME_CHIP[pending.outcome]}`}
                  >
                    pending {pending.outcome}
                    {pending.markedFrame !== null
                      ? ` @ ${pending.markedFrame}`
                      : " (unmarked)"}
                  </span>
                )}
                {pending.softTruncate && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-coral-light text-coral">
                    soft-truncate
                  </span>
                )}
                {subtaskMarksRequired > 0 && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-purple-100 text-purple-700">
                    subtask {pending.subtaskFrames.length}/{subtaskMarksRequired}
                    {pending.subtaskFrames.length > 0
                      ? ` @ ${pending.subtaskFrames.join(", ")}`
                      : ""}
                  </span>
                )}
                {currentSignals && (
                  <span className="text-[11px] font-mono text-ink-muted">
                    valid {currentSignals.validLength}/{currentEpisode.rawLength}
                    {currentSignals.doneOnsetFrame != null
                      ? ` · done@${currentSignals.doneOnsetFrame}`
                      : ""}
                  </span>
                )}
              </div>

              {actionError && (
                <div className="mb-3 rounded-lg border border-coral/30 bg-coral-light px-3 py-2 text-xs text-coral font-mono">
                  {actionError}
                </div>
              )}

              {cameraKeys.length === 0 ? (
                <div className="rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral font-mono">
                  Episode {currentEpisode.episodeIndex} exposes no reviewable
                  camera streams.
                </div>
              ) : (
                <ReviewViewer
                  datasetId={repoId}
                  episode={currentEpisode}
                  cameraKeys={cameraKeys}
                  primaryKey={primaryKey}
                  fps={FPS}
                  frame={frame}
                  onFrame={setFrame}
                  lastValidFrame={currentSignals?.lastValidFrame ?? null}
                  controlsRef={controlsRef}
                  cropByCameraKey={cropByCameraKey}
                  storedFrameHW={storedFrameHW}
                  onDrift={setViewerDrift}
                  renderVideoOverlay={renderVideoOverlay}
                  renderTimelineOverlays={renderTimelineOverlays}
                />
              )}

              <EpisodeNotes repoId={repoId} episodeIndex={currentEpisode.episodeIndex} />

              <LabelHistoryPanel
                chain={historyByEpisode.get(currentEpisode.episodeIndex) ?? []}
                rawLength={currentEpisode.rawLength}
                onSeek={jumpToFrame}
              />

              <div className="mt-4 flex flex-wrap items-center gap-2">
                {OUTCOMES.map((outcome) => (
                  <button
                    key={outcome}
                    onClick={() => {
                      updatePending({ outcome, markedFrame: frame });
                      setActionError(null);
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                      pending.outcome === outcome
                        ? OUTCOME_CHIP[outcome]
                        : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300"
                    }`}
                  >
                    {outcome}
                    <span className="ml-1.5 font-mono text-[10px] opacity-60">
                      {outcome[0]}
                    </span>
                  </button>
                ))}
                <button
                  onClick={() => updatePending({ softTruncate: !pending.softTruncate })}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                    pending.softTruncate
                      ? "bg-coral-light text-coral"
                      : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300"
                  }`}
                >
                  soft-truncate
                  <span className="ml-1.5 font-mono text-[10px] opacity-60">x</span>
                </button>
                <div className="flex-1" />
                <button
                  disabled={saving}
                  onClick={() => void skip()}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium border border-warm-200 text-ink-muted hover:border-warm-300 cursor-pointer"
                >
                  {skipArmed ? "skip (discard marks)" : "skip"}
                  <span className="ml-1.5 font-mono text-[10px] opacity-60">n</span>
                </button>
                <button
                  disabled={saving}
                  onClick={() => void confirm()}
                  className={`px-4 py-1.5 rounded-lg text-xs font-medium ${
                    saving
                      ? "bg-warm-100 text-ink-muted/50 cursor-not-allowed"
                      : "bg-teal text-white hover:bg-teal/90 cursor-pointer"
                  }`}
                >
                  confirm
                  <span className="ml-1.5 font-mono text-[10px] opacity-70">c</span>
                </button>
              </div>
            </>
          )}
        </div>

        {/* Work queue */}
        <div className="border-t border-warm-100 p-4 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                e.currentTarget.blur();
              }}
              className="min-w-40 rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink cursor-pointer"
              title="Unaddressed = no web review and not in the applied HF record"
            >
              <option value="unaddressed">Unaddressed only</option>
              <option value="all">All statuses</option>
            </select>
            <select
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value as QueueFilter);
                // A focused select swallows the review shortcuts and letter keys
                // drive its native typeahead — release focus after each change.
                e.currentTarget.blur();
              }}
              className="min-w-32 rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink cursor-pointer"
            >
              {QUEUE_FILTERS.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
            {armOptions.length > 0 && (
              <select
                value={armFilter}
                onChange={(e) => {
                  setArmFilter(e.target.value);
                  e.currentTarget.blur();
                }}
                className="min-w-32 rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink cursor-pointer"
                title="Filter by ledger arm (blind_dagger / protocol_quota / teleop_manifest ledgers)"
              >
                <option value="all">All arms</option>
                {armOptions.map((arm) => (
                  <option key={arm} value={arm}>
                    {arm}
                  </option>
                ))}
              </select>
            )}
            <div className="ml-auto text-[11px] font-mono text-ink-muted">
              {numAddressed ?? "…"} of {episodes?.length ?? 0} addressed ·{" "}
              {reviewByEpisode.size} this web ledger ·{" "}
              {filteredEpisodes.length} in queue
              {episodes && numLoadedSignals < episodes.length && (
                <span className="ml-2 animate-pulse">
                  outcomes {numLoadedSignals}/{episodes.length}…
                </span>
              )}
            </div>
          </div>
          {ledgerError && (
            <div className="rounded-lg border border-coral/30 bg-coral-light px-2 py-1.5 text-[10px] text-coral font-mono">
              arm filter unavailable — ledger parse failed: {ledgerError}
            </div>
          )}
          <div
            ref={queueScrollRef}
            className="relative flex gap-1.5 overflow-x-auto pb-1"
          >
            {episodes === null && !loadError && (
              <div className="flex items-center gap-2 text-xs text-ink-muted">
                <div className="w-4 h-4 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
                Loading episodes…
              </div>
            )}
            {filteredEpisodes.map((episode) => (
              <div
                key={episode.episodeIndex}
                data-episode-index={episode.episodeIndex}
                className="w-44 shrink-0"
              >
                <QueueRow
                  episode={episode}
                  signals={signals.get(episode.episodeIndex) ?? null}
                  signalError={signalErrors.get(episode.episodeIndex) ?? null}
                  review={reviewByEpisode.get(episode.episodeIndex) ?? null}
                  applied={
                    applied?.changed.get(episode.episodeIndex)?.newOutcome ??
                    (applied?.skipped.has(episode.episodeIndex) ? "skip" : null)
                  }
                  selected={selectedEpisode === episode.episodeIndex}
                  onSelect={() => selectEpisode(episode.episodeIndex)}
                  arm={ledgerArms?.get(episode.episodeIndex) ?? null}
                />
              </div>
            ))}
            {episodes !== null && filteredEpisodes.length === 0 && (
              <div className="text-xs text-ink-muted font-body">
                {statusFilter === "unaddressed" &&
                applied === undefined &&
                appliedError === null
                  ? "Checking the applied HF record…"
                  : statusFilter === "unaddressed" &&
                      episodes.length > 0 &&
                      episodes.every((episode) => isAddressed(episode.episodeIndex))
                    ? `All ${episodes.length} episodes are already addressed. ` +
                      `Switch to "All statuses" to revisit them.`
                    : "No episodes match this filter yet."}
              </div>
            )}
          </div>
        </div>
      </div>

      <CommitPanel
        repoId={repoId}
        numConfirmed={numConfirmed}
        numSkipped={numSkipped}
        numEpisodes={episodes?.length ?? 0}
      />
    </div>
  );
}
