import { validateTrajectoryEventLinks, type TrajectoryEventLink } from "../../convex/trajectoryEventLinks";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../release/api";
import { blankTrajectoryReview } from "../../convex/trajectoryReview";
import type { Id } from "../release/api";
import {
  validateStageLabel,
  type ExportedStageSpec,
  type StageLabelRow,
} from "../../convex/stageConsistency";
import {
  explorerCameraKeys,
  selectPrimaryCameraKey,
  type AppliedProgress,
  type EpisodeFrameSignals,
  type LabelEvent,
  type ReviewEpisode,
} from "../lib/hf-api";
import { normalizeStageSpec } from "../lib/stage-spec";
import {
  attributionDescription,
  NO_PREDICTIONS,
  resolvePredictionSelection,
  seedStageReview,
  stageDisplay,
  type PredictionAttribution,
} from "../lib/stagePredictionReview";
import { useStageReviewDraft } from "../lib/useStageReviewDraft";
import { analyzeTrajectoryTimeline } from "../../convex/trajectoryTimeline";
import { stageReviewDataSource, type StageReviewDataSource } from "../lib/stageReviewDataSource";
import { useSearchParam, useSearchParamNumber, useSearchParamNavigationGuard, setSearchParams } from "../lib/useSearchParam";
import { EvidencePanel, type StagePrefillView } from "./review/EvidencePanel";
import { HelpOverlay } from "./review/HelpOverlay";
import { LabelHistoryPanel } from "./review/LabelHistoryPanel";
import { ReviewViewer, type ViewerControls } from "./review/ReviewViewer";
import { StageLabelForm } from "./review/StageLabelForm";
import {
  cameraRoleForVideoKey,
  clamp,
  formatClock,
  orderCameraKeys,
  type CropBox,
} from "./review/format";
import { isTypingTarget, useWindowKeydown } from "./review/useWindowKeydown";

// ---------------------------------------------------------------------------
// Stage-label review: per (episode, taxonomy_version) the surface shows the
// CURRENT label — the reviewer's own saved row if one exists, else the pinned
// immutable prediction version (or an empty form with no predictions). Verdicts capture their source in the
// append-only stageReviews table. The form, its instant consistency feedback,
// and the fps math are all driven by the EXPORTED spec (stageTaskSpecs); no
// task names or field names appear below.
//
// Blind mode is ON by default: the stage-bucket protocol forbids policy/arm
// identity from influencing labels, so arm data is not even FETCHED while
// blind (single data-layer choke point, not per-render discipline).
// ---------------------------------------------------------------------------

// Stable empty row list for "No predictions", so memoized views do not rebuild every render.
const NO_PREDICTION_ROWS: never[] = [];

type StatusFilter = "unreviewed" | "draft" | "confirmed" | "corrected" | "uncertain" | "all";

interface StageReviewRecord {
  id: Id<"stageReviews">;
  episodeIndex: number;
  status: string;
  label: StageLabelRow | null;
  humanNotes?: string;
  eventLinks?: TrajectoryEventLink[];
  reviewer: string;
  savedAt: number;
  blind: boolean;
  attribution: PredictionAttribution;
}

const STATUS_GLYPH: Record<string, string> = {
  confirmed: "✓",
  corrected: "✎",
  uncertain: "?",
  draft: "…",
};

const CONFIDENCE_DOT: Record<string, string> = {
  high: "bg-teal",
  medium: "bg-gold",
  low: "bg-coral",
};

const OUTCOME_CHIP: Record<string, string> = {
  success: "bg-teal-light text-teal",
  failure: "bg-coral-light text-coral",
  timeout: "bg-gold-light text-gold",
};

/**
 * The outcome-editor decision for an episode. Stage labeling is GATED on the
 * outcome flow having run first: `null` outcome under source "detected" means
 * the decision was keep-as-is (a skip) and the dataset's own frame signals are
 * still loading for the selected episode.
 */
interface ResolvedOutcome {
  outcome: string | null;
  source: "review" | "applied" | "detected";
}

const HELP_KEYS: [string, string][] = [
  ["← / →", "step 1 frame (shift: 10)"],
  ["[ / ]", "step 30 frames"],
  ["Home / End", "first / last frame"],
  ["space", "play / pause"],
  ["0–9", "set the stage rung"],
  ["- / =", "decrement / increment the stage (covers S10)"],
  ["c", "confirm: reviewed labels and event times (gold-eligible) + advance"],
  ["u", "uncertain: reviewed but not gold-eligible + advance"],
  ["e", "toggle the model-evidence rail"],
  ["n", "next episode (drafts unsaved edits)"],
  ["p / b", "previous episode in the queue"],
  ["q / Esc", "exit stage review"],
  ["?", "toggle this help"],
];

export default function StageReview({
  repoId,
  task,
  onExit,
  onOpenOutcomeReview,
  dataSource = stageReviewDataSource,
}: {
  repoId: string;
  task?: string;
  onExit: () => void;
  /** Jump to outcome review for this dataset (episode param carries over). */
  onOpenOutcomeReview: () => void;
  dataSource?: StageReviewDataSource;
}) {
  const { useQuery, useMutation, usePaginatedQuery, fetchAppliedProgress,
    fetchEpisodeFrameSignals, fetchLabelHistory, fetchLedgerArms, fetchReviewEpisodes } = dataSource;
  const viewer = useQuery(api.users.viewer);
  const specRows = useQuery(api.stageTaskSpecs.forTask, task ? { task } : "skip");
  const taskSpec = useQuery(api.taskSpecs.forTask, task ? { task } : "skip");
  const saveReview = useMutation(api.stageReviews.save);
  // Stage labeling is gated on the outcome-editor flow: every episode must
  // carry an outcome decision (web review, or the applied HF record) before it
  // may be stage-labeled. Only forms without predictions inherit successful outcomes.
  const outcomeReviews = useQuery(api.reviews.latestForRepo, { dataset_repo: repoId });

  // -- Taxonomy (schema) selection: live by default, candidates addressable --
  const [schemaParam] = useSearchParam("schema", "");
  const liveRow = specRows?.find((row) => row.live) ?? null;
  // A stale/typo'd ?schema= must not strand the surface on a loading card
  // with no selector on screen — fall back to the live row, loudly.
  const schemaFellBack = Boolean(
    schemaParam && specRows !== undefined &&
      !specRows.some((row) => row.taxonomy_version === schemaParam)
  );
  const specRow =
    (schemaParam && !schemaFellBack
      ? specRows?.find((row) => row.taxonomy_version === schemaParam)
      : liveRow) ?? null;
  // A malformed spec payload must render as a banner, not a render-throw that
  // white-screens the app (there is no ErrorBoundary above us).
  const specResult = useMemo<{ spec: ExportedStageSpec | null; error: string | null }>(() => {
    if (!specRow) return { spec: null, error: null };
    try {
      return { spec: normalizeStageSpec(specRow.spec), error: null };
    } catch (err) {
      return { spec: null, error: (err as Error).message };
    }
  }, [specRow]);
  const spec = specResult.spec;
  const taxonomyVersion = spec?.taxonomy_version ?? null;

  const reviews = useQuery(
    api.stageReviews.latestForRepo,
    taxonomyVersion ? { dataset_repo: repoId, taxonomy_version: taxonomyVersion } : "skip"
  );
  const [predictionParam, setPredictionParam] = useSearchParam("prediction", "");

  const predictionVersions = useQuery(
    api.stagePredictions.listForRepo,
    taxonomyVersion ? { dataset_repo: repoId, taxonomy_version: taxonomyVersion } : "skip"
  );
  // Capture the default once by writing its concrete identity into the URL.
  // A publication in another session must not switch an in-progress form.
  useEffect(() => {
    if (predictionParam || predictionVersions === undefined) return;
    setPredictionParam(predictionVersions.active_run_id ?? NO_PREDICTIONS);
  }, [predictionParam, predictionVersions, setPredictionParam]);
  const predictionSelection = predictionVersions !== undefined && predictionParam
    ? resolvePredictionSelection(predictionParam, predictionVersions.runs)
    : null;
  const selectedRun = predictionVersions?.runs.find((run) => run._id === predictionParam);
  const predictionPages = usePaginatedQuery(
    api.stagePredictions.forRun,
    predictionSelection?.runId ? { run_id: predictionSelection.runId } : "skip",
    { initialNumItems: 50 }
  );
  const { status: predictionPageStatus, loadMore: loadMorePredictions } = predictionPages;
  useEffect(() => {
    if (predictionSelection?.runId && predictionPageStatus === "CanLoadMore") {
      loadMorePredictions(50);
    }
  }, [predictionSelection?.runId, predictionPageStatus, loadMorePredictions]);
  const prefillRows = predictionParam === NO_PREDICTIONS
    ? NO_PREDICTION_ROWS
    : predictionSelection?.runId && predictionPages.status === "Exhausted"
      ? predictionPages.results
      : undefined;
  const predictionError = predictionSelection?.error ?? (
    selectedRun && predictionPages.status === "Exhausted" &&
    predictionPages.results.length !== selectedRun.expected_count
      ? `Prediction version ${selectedRun._id} has ${predictionPages.results.length} episodes; its published manifest requires ${selectedRun.expected_count}.`
      : null
  );
  const predictionsReady = predictionSelection !== null &&
    predictionError === null && prefillRows !== undefined;

  // -- Blind mode (default ON): policy/arm identity hidden AND unfetched -----
  const [blindParam, setBlindParam] = useSearchParam("blind", "1");
  const blind = blindParam !== "0";

  // -- Queue filters (stage-prefixed params; REPLACE_KEYS members) ----------
  const [statusFilter, setStatusFilter] = useSearchParam("sstatus", "unreviewed");
  const [confFilter, setConfFilter] = useSearchParam("sconf", "all");
  const [flagFilter, setFlagFilter] = useSearchParam("sflag", "all");
  const [armFilter, setArmFilter] = useSearchParam("sarm", "all");
  const [selectedEpisode, setSelectedEpisode] = useSearchParamNumber("episode");
  const otherSchemaPredictions = useQuery(api.stagePredictions.otherSchemasForEpisode,
    task && taxonomyVersion && selectedEpisode !== null && Number.isSafeInteger(selectedEpisode) && selectedEpisode >= 0
      ? { dataset_repo: repoId, task, taxonomy_version: taxonomyVersion, episode_index: BigInt(selectedEpisode) } : "skip");

  // -- HF loads ---------------------------------------------------------------
  const [episodes, setEpisodes] = useState<ReviewEpisode[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [labelHistory, setLabelHistory] = useState<LabelEvent[] | null | undefined>(undefined);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [ledgerArms, setLedgerArms] = useState<Map<number, string> | null>(null);
  const [ledgerError, setLedgerError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect -- imperative reset of
       fetch state on repo change, not derived render state (same pattern as
       OutcomeReview). */
    setEpisodes(null);
    setLoadError(null);
    /* eslint-enable react-hooks/set-state-in-effect */
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
  }, [repoId, fetchReviewEpisodes]);

  useEffect(() => {
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect -- imperative fetch-state reset. */
    setLabelHistory(undefined);
    setHistoryError(null);
    /* eslint-enable react-hooks/set-state-in-effect */
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
  }, [repoId, fetchLabelHistory]);

  useEffect(() => {
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect -- imperative fetch-state reset. */
    setLedgerArms(null);
    setLedgerError(null);
    /* eslint-enable react-hooks/set-state-in-effect */
    if (blind) return; // data-layer redaction: never even fetch arm identity
    fetchLedgerArms(repoId)
      .then((arms) => {
        if (!cancelled) setLedgerArms(arms);
      })
      .catch((err: Error) => {
        if (!cancelled) setLedgerError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId, blind, fetchLedgerArms]);

  // -- Outcome gate: the applied HF outcome-edit record (apply worker or
  // Python editor) complements the web outcome-review ledger. Tri-state like
  // OutcomeReview: undefined = loading, null = never treated.
  const [applied, setApplied] = useState<AppliedProgress | null | undefined>(undefined);
  const [appliedError, setAppliedError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect -- imperative fetch-state reset. */
    setApplied(undefined);
    setAppliedError(null);
    /* eslint-enable react-hooks/set-state-in-effect */
    fetchAppliedProgress(repoId)
      .then((progress) => {
        if (!cancelled) setApplied(progress);
      })
      .catch((err: Error) => {
        // Fail CLOSED but loud: episodes whose only outcome decision lives in
        // the unreadable applied record stay gated until it loads.
        if (!cancelled) setAppliedError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId, fetchAppliedProgress]);

  // Frame signals, fetched lazily for the SELECTED episode only — needed to
  // resolve the outcome of a keep-as-is (skip) decision, whose recorded
  // outcome is whatever the dataset's own reward/done signals say.
  const [outcomeSignals, setOutcomeSignals] = useState<Map<string, EpisodeFrameSignals>>(
    () => new Map()
  );
  const [outcomeSignalErrors, setOutcomeSignalErrors] = useState<Map<string, string>>(
    () => new Map()
  );

  const signalKey = useCallback((episodeIndex: number) => `${repoId}::${episodeIndex}`, [repoId]);

  const outcomeByEpisode = useMemo(() => {
    const map = new Map<number, { status: string; newOutcome: string | null }>();
    for (const row of outcomeReviews?.episodes ?? []) {
      map.set(Number(row.episode_index), {
        status: row.status,
        newOutcome: (row.new_outcome as string | undefined) ?? null,
      });
    }
    return map;
  }, [outcomeReviews]);

  // Ready once both outcome sources settled (an applied-record load ERROR
  // counts as settled: the gate then fails closed with a loud banner).
  const outcomeGateReady =
    outcomeReviews !== undefined && (applied !== undefined || appliedError !== null);

  // The episode's outcome decision, or null when the outcome flow has not been
  // run on it yet (⇒ stage labeling is blocked for that episode).
  const resolveOutcome = useCallback(
    (episodeIndex: number): ResolvedOutcome | null => {
      const web = outcomeByEpisode.get(episodeIndex);
      if (web?.status === "confirmed" && web.newOutcome !== null) {
        return { outcome: web.newOutcome, source: "review" };
      }
      const appliedRecord = applied?.changed.get(episodeIndex);
      if (appliedRecord !== undefined) {
        return { outcome: appliedRecord.newOutcome, source: "applied" };
      }
      if (web !== undefined || applied?.skipped.has(episodeIndex)) {
        // keep-as-is decision — the outcome is the dataset's detected signals
        return {
          outcome: outcomeSignals.get(signalKey(episodeIndex))?.detectedOutcome ?? null,
          source: "detected",
        };
      }
      return null;
    },
    [outcomeByEpisode, applied, outcomeSignals, signalKey]
  );

  // -- Convex rows -> typed views ---------------------------------------------
  const prefillByEpisode = useMemo(() => {
    const map = new Map<number, StagePrefillView>();
    for (const row of prefillRows ?? []) {
      const violationCodes = [...new Set([...(row.violation_codes ?? []), ...row.validation_codes])];
      const reasons = [...new Set([
        ...(row.review_reason ?? "").split(";").map((reason) => reason.trim()).filter(Boolean),
        ...violationCodes.map((code) => `consistency:${code}`),
      ])];
      map.set(Number(row.episode_index), {
        label: row.label as Record<string, unknown>,
        reviewReason: reasons.length ? reasons.join(";") : null,
        violationCodes,
        confidence: row.confidence == null ? null
          : Object.hasOwn(CONFIDENCE_DOT, row.confidence) ? row.confidence : "invalid",
        voteSummary: (row.vote_summary as Record<string, unknown> | undefined) ?? null,
        episodeDurationS: row.episode_duration_s ?? null,
        pipeline: row.pipeline,
        evidence: row.evidence as Record<string, unknown>,
        pushedAt: row.pushed_at,
        attribution: {
          prediction_id: row._id,
          prediction_sha256: row.content_sha256,
          episode_duration_s: row.episode_duration_s,
        },
        canonicalResponse: row.canonical_response,
        sourceRevision: row.source_revision,
      });
    }
    return map;
  }, [prefillRows]);

  // Only the signed-in reviewer's OWN rows prefill/queue — blinded double
  // labeling means another reviewer's decision must never leak into the form.
  const ownReviewByEpisode = useMemo(() => {
    const map = new Map<number, StageReviewRecord>();
    const userId = viewer?.userId;
    if (!userId) return map;
    for (const row of reviews?.episodes ?? []) {
      if (row.reviewer_user_id !== userId) continue;
      map.set(Number(row.episode_index), {
        id: row._id,
        episodeIndex: Number(row.episode_index),
        status: row.status,
        label: (row.label as StageLabelRow | undefined) ?? null,
        humanNotes: row.notes,
        eventLinks: row.event_links,
        reviewer: row.reviewer,
        savedAt: row.saved_at,
        blind: row.blind ?? false,
        attribution: {
          copied_from_review_id: row.copied_from_review_id,
          prediction_id: row.prediction_id,
          prediction_sha256: row.prediction_sha256,
          prefill_pushed_at: row.prefill_pushed_at,
          episode_duration_s: row.episode_duration_s,
        },
      });
    }
    return map;
  }, [reviews, viewer]);

  // Other reviewers' latest rows — surfaced ONLY under the "Needs
  // adjudication" filter (an annotator presses uncertain to hand an episode
  // off; the adjudicator deliberately unblinds to their progress + notes).
  const otherReviewsByEpisode = useMemo(() => {
    const map = new Map<number, StageReviewRecord[]>();
    const userId = viewer?.userId;
    for (const row of reviews?.episodes ?? []) {
      if (userId && row.reviewer_user_id === userId) continue;
      const ep = Number(row.episode_index);
      map.set(ep, [
        ...(map.get(ep) ?? []),
        {
          id: row._id,
          episodeIndex: ep,
          status: row.status,
          label: (row.label as StageLabelRow | undefined) ?? null,
          humanNotes: row.notes,
          eventLinks: row.event_links,
          reviewer: row.reviewer,
          savedAt: row.saved_at,
          blind: row.blind ?? false,
          attribution: {
            copied_from_review_id: row.copied_from_review_id,
            prediction_id: row.prediction_id,
            prediction_sha256: row.prediction_sha256,
            prefill_pushed_at: row.prefill_pushed_at,
            episode_duration_s: row.episode_duration_s,
          },
        },
      ]);
    }
    return map;
  }, [reviews, viewer]);

  const historyByEpisode = useMemo(() => {
    const byEpisode = new Map<number, LabelEvent[]>();
    for (const event of labelHistory ?? []) {
      const list = byEpisode.get(event.episode_index);
      if (list) list.push(event);
      else byEpisode.set(event.episode_index, [event]);
    }
    for (const list of byEpisode.values()) {
      list.sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts));
    }
    return byEpisode;
  }, [labelHistory]);

  // -- Queue -------------------------------------------------------------------
  const armOptions = useMemo(
    () => (ledgerArms ? [...new Set(ledgerArms.values())].sort() : []),
    [ledgerArms]
  );

  const flagCount = useCallback(
    (episodeIndex: number): number => {
      const reason = prefillByEpisode.get(episodeIndex)?.reviewReason;
      return reason ? reason.split(";").filter((r) => r.trim()).length : 0;
    },
    [prefillByEpisode]
  );

  // Episodes the outcome flow has not addressed yet — gated out of the queue.
  const awaitingOutcome = useMemo(() => {
    if (!episodes || !outcomeGateReady) return [];
    return episodes
      .filter((episode) => resolveOutcome(episode.episodeIndex) === null)
      .map((episode) => episode.episodeIndex);
  }, [episodes, outcomeGateReady, resolveOutcome]);

  const filteredEpisodes = useMemo(() => {
    if (!episodes) return [];
    // The outcome gate must settle before the queue exists at all — an
    // ungated flash would let a verdict land on an episode that later turns
    // out to lack its outcome decision.
    if (!outcomeGateReady) return [];
    let result = episodes.filter(
      (episode) => resolveOutcome(episode.episodeIndex) !== null
    );
    if (statusFilter === "adjudicate") {
      result = result.filter((episode) =>
        (otherReviewsByEpisode.get(episode.episodeIndex) ?? []).some(
          (r) => r.status === "uncertain"
        )
      );
    } else if (statusFilter !== "all") {
      result = result.filter((episode) => {
        const own = ownReviewByEpisode.get(episode.episodeIndex);
        if (statusFilter === "unreviewed") return own === undefined || own.status === "draft";
        return own?.status === statusFilter;
      });
    }
    if (confFilter !== "all") {
      result = result.filter(
        (episode) => prefillByEpisode.get(episode.episodeIndex)?.confidence === confFilter
      );
    }
    if (flagFilter === "flagged") {
      result = result.filter((episode) => flagCount(episode.episodeIndex) > 0);
    } else if (flagFilter === "unflagged") {
      result = result.filter((episode) => flagCount(episode.episodeIndex) === 0);
    }
    if (!blind && armFilter !== "all" && ledgerArms && armOptions.includes(armFilter)) {
      result = result.filter(
        (episode) => ledgerArms.get(episode.episodeIndex) === armFilter
      );
    }
    // Flagged-first (by flag count), then low confidence first, then index.
    const confRank: Record<string, number> = { low: 0, medium: 1, high: 2 };
    return [...result].sort((a, b) => {
      const flagDelta = flagCount(b.episodeIndex) - flagCount(a.episodeIndex);
      if (flagDelta !== 0) return flagDelta;
      const confA = confRank[prefillByEpisode.get(a.episodeIndex)?.confidence ?? ""] ?? 3;
      const confB = confRank[prefillByEpisode.get(b.episodeIndex)?.confidence ?? ""] ?? 3;
      if (confA !== confB) return confA - confB;
      return a.episodeIndex - b.episodeIndex;
    });
  }, [
    episodes,
    outcomeGateReady,
    resolveOutcome,
    statusFilter,
    confFilter,
    flagFilter,
    armFilter,
    blind,
    ledgerArms,
    armOptions,
    ownReviewByEpisode,
    otherReviewsByEpisode,
    prefillByEpisode,
    flagCount,
  ]);

  const currentEpisode = useMemo(
    () => episodes?.find((episode) => episode.episodeIndex === selectedEpisode) ?? null,
    [episodes, selectedEpisode]
  );
  const currentPrefill =
    selectedEpisode !== null ? (prefillByEpisode.get(selectedEpisode) ?? null) : null;

  // undefined = outcome gate still loading; null = outcome flow never ran on
  // this episode (stage labeling blocked).
  const currentResolvedOutcome =
    selectedEpisode !== null && outcomeGateReady
      ? resolveOutcome(selectedEpisode)
      : undefined;
  const currentOutcomeSignalError =
    selectedEpisode !== null ? (outcomeSignalErrors.get(signalKey(selectedEpisode)) ?? null) : null;

  const selectedOwn = selectedEpisode !== null ? ownReviewByEpisode.get(selectedEpisode) : undefined;
  const selectedSourceDuration = selectedOwn?.label != null
    ? selectedOwn.attribution.episode_duration_s : currentPrefill?.episodeDurationS;
  const needsPolicyDuration = Boolean(spec?.trajectory && selectedSourceDuration == null);
  const selectedSignals = selectedEpisode !== null ? outcomeSignals.get(signalKey(selectedEpisode)) : undefined;
  const policyDurationS = spec && selectedSignals ? selectedSignals.validLength / spec.fps : null;

  // The same lazily fetched frame signals resolve keep-as-is outcomes and
  // supply the validated policy prefix for a new source-free trajectory.
  useEffect(() => {
    const needsOutcome = currentResolvedOutcome?.source === "detected" && currentResolvedOutcome.outcome === null;
    if (!currentEpisode || (!needsOutcome && !needsPolicyDuration) || selectedSignals ||
        outcomeSignalErrors.has(signalKey(currentEpisode.episodeIndex))) return;
    let cancelled = false;
    fetchEpisodeFrameSignals(repoId, currentEpisode.dataPath, currentEpisode.episodeIndex)
      .then((result) => {
        if (!cancelled) {
          setOutcomeSignals((prev) => new Map(prev).set(signalKey(currentEpisode.episodeIndex), result));
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setOutcomeSignalErrors((prev) =>
            new Map(prev).set(signalKey(currentEpisode.episodeIndex), err.message)
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [currentEpisode, currentResolvedOutcome, outcomeSignalErrors, repoId, fetchEpisodeFrameSignals,
      needsPolicyDuration, selectedSignals, signalKey]);

  useEffect(() => {
    if (selectedEpisode !== null) return;
    const head = filteredEpisodes[0];
    if (head) setSelectedEpisode(head.episodeIndex);
  }, [selectedEpisode, filteredEpisodes, setSelectedEpisode]);

  // -- Cameras (same resolution as OutcomeReview, from the lifecycle spec) -----
  const cameraKeys = useMemo(() => {
    if (!currentEpisode) return [];
    const all = orderCameraKeys(explorerCameraKeys(Object.keys(currentEpisode.perCamera)));
    const roles = taskSpec?.review_camera_roles;
    if (roles == null || roles.length === 0) return all;
    const keysByRole = taskSpec!.camera_keys_by_role;
    const selected = roles
      .map((role) => all.find((key) => cameraRoleForVideoKey(key, keysByRole) === role))
      .filter((key): key is string => key !== undefined);
    return selected.length > 0 ? selected : all;
  }, [currentEpisode, taskSpec]);
  const primaryKey = useMemo(
    () => (cameraKeys.length > 0 ? selectPrimaryCameraKey(cameraKeys) : ""),
    [cameraKeys]
  );
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
      if (box !== undefined) map[key] = box.map(Number) as CropBox;
    }
    return map;
  }, [taskSpec, cameraKeys]);


  // -- Working state ------------------------------------------------------------
  const [frame, setFrame] = useState(0);
  const [viewerDrift, setViewerDrift] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [unverifiable, setUnverifiable] = useState(false);
  const [draftError, setDraftError] = useState<string | null>(null);
  const saveInFlight = useRef(false);
  const allowSavedNavigation = useRef(false);
  const pendingInputs = useRef(new Set<string>());
  const [pendingInputCount, setPendingInputCount] = useState(0);
  const onPendingInputChange = useCallback((id: string, pending: boolean) => {
    if (pending) pendingInputs.current.add(id); else pendingInputs.current.delete(id);
    setPendingInputCount(pendingInputs.current.size);
  }, []);
  useEffect(() => {
    if (!pendingInputCount) return;
    const guard = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [pendingInputCount]);

  // Once arm identity has been revealed, saved reviews cannot claim blindness.
  const everUnblindedRef = useRef(false);
  useEffect(() => {
    if (!blind) everUnblindedRef.current = true;
  }, [blind]);
  const unblind = () => {
    everUnblindedRef.current = true;
    setBlindParam("0");
  };
  const controlsRef = useRef<ViewerControls | null>(null);
  const sourceKey = selectedEpisode !== null && taxonomyVersion && predictionParam
    ? JSON.stringify([repoId, taxonomyVersion, predictionParam, selectedEpisode, viewer?.userId])
    : null;
  const seed = useMemo(() => {
    if (selectedEpisode === null || !spec || !currentEpisode || !predictionsReady ||
        reviews === undefined || viewer === undefined || !outcomeGateReady ||
        (needsPolicyDuration && policyDurationS === null)) return null;
    const resolved = resolveOutcome(selectedEpisode);
    if (resolved === null) return null;
    if (resolved.source === "detected" && resolved.outcome === null &&
        !outcomeSignalErrors.has(signalKey(selectedEpisode))) return null;
    const own = ownReviewByEpisode.get(selectedEpisode);
    const prediction = prefillByEpisode.get(selectedEpisode);
    const seeded = seedStageReview({
      own,
      prediction,
      outcome: resolved.outcome,
      spec,
      sourceFree: predictionParam === NO_PREDICTIONS,
      emptyLabel: spec.trajectory ? blankTrajectoryReview(spec.trajectory, repoId, selectedEpisode) : {},
    });
    if (spec.trajectory && seeded.attribution.episode_duration_s === undefined && policyDurationS !== null) {
      seeded.attribution.episode_duration_s = policyDurationS;
    }
    return seeded;
  }, [selectedEpisode, spec, currentEpisode, predictionsReady, reviews, viewer,
      outcomeGateReady, resolveOutcome, outcomeSignalErrors, ownReviewByEpisode,
      prefillByEpisode, predictionParam, repoId, needsPolicyDuration, policyDurationS, signalKey]);
  const { draft, edit: editDraft, editHumanNotes, editEventLinks, replaceLabel, markSaved, unsavedCount } =
    useStageReviewDraft(sourceKey, seed);
  const pending = predictionsReady ? draft?.label ?? null : null;
  const dirty = draft?.dirty ?? false;
  const inheritedSuccess = draft?.inheritedSuccess ?? false;
  const shownAttribution = draft?.attribution;
  const formDisabled = saving || pending === null || !predictionsReady;

  useSearchParamNavigationGuard(useCallback((current, next) => {
    if (pendingInputs.current.size > 0 && ["tab", "dataset", "view", "episode", "prediction", "schema"].some((key) => current.get(key) !== next.get(key))) {
      setDraftError("A timestamp contains unfinished or invalid text. Correct it or clear its input before saving or leaving.");
      return false;
    }
    const leavesReview = ["tab", "dataset", "view"].some((key) => current.get(key) !== next.get(key));
    if (!leavesReview || allowSavedNavigation.current) return true;
    if (saving || saveInFlight.current) {
      setDraftError("A stage review is being saved. Stay on this page until the save finishes.");
      return false;
    }
    if (unsavedCount > 0) {
      setDraftError("Unsaved stage edits are still in this review. Save them before changing tabs or datasets.");
      return false;
    }
    return true;
  }, [saving, unsavedCount]));

  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect -- reset video controls on source change */
    setFrame(0);
    setViewerDrift(null);
    setActionError(null);
    /* eslint-enable react-hooks/set-state-in-effect */
    controlsRef.current?.pause();
  }, [sourceKey]);

  // Bounds belong to the original source of the human draft, even if another
  // model is selected for inspection. New trajectory bounds come from parsed
  // policy-phase signals; flat (non-trajectory) specs use the raw metadata length.
  const episodeDurationS = shownAttribution?.episode_duration_s ??
    (spec?.trajectory ? policyDurationS : spec && currentEpisode ? currentEpisode.rawLength / spec.fps : null);
  const violations = useMemo(
    () => (spec && pending ? [
      ...validateStageLabel(spec, pending, episodeDurationS),
      ...validateTrajectoryEventLinks(pending, draft?.eventLinks ?? []).map((message) => ({
        code: "trajectory_event_links", message, fields: ["stage_transitions", "key_action_observations"],
      })),
      ...(spec.trajectory ? analyzeTrajectoryTimeline(spec.trajectory, pending).map((issue) => ({
        code: "trajectory_timeline", message: issue.message, fields: ["stage_transitions", "key_action_observations", "failure_events"],
      })) : []),
    ] : []),
    [spec, pending, episodeDurationS, draft?.eventLinks]
  );
  const edit = useCallback((patch: StageLabelRow) => {
    if (formDisabled || saveInFlight.current) return;
    const next = { ...patch };
    if (spec?.trajectory && typeof next[spec.stage_field] === "number") {
      const selected = spec.trajectory.task_definition.stages.find((stage) => stage.index === next[spec.stage_field]);
      if (!selected) throw new Error("Stage selection is outside the trajectory definition");
      next.max_stage_id = selected.id;
    }
    editDraft(next);
    setActionError(null);
  }, [editDraft, formDisabled, spec]);

  const jumpToFrame = useCallback((next: number) => {
    controlsRef.current?.pause();
    setFrame(next);
  }, []);

  const seekTime = useCallback(
    (timeS: number) => {
      if (!currentEpisode || !spec) return;
      jumpToFrame(clamp(Math.round(timeS * spec.fps), 0, currentEpisode.rawLength - 1));
    },
    [currentEpisode, spec, jumpToFrame]
  );

  const stepFrame = useCallback(
    (delta: number) => {
      if (!currentEpisode) return;
      controlsRef.current?.pause();
      setFrame((prev) => clamp(prev + delta, 0, currentEpisode.rawLength - 1));
    },
    [currentEpisode]
  );

  // A mark taken while the displayed frame is unverified/
  // drifted becomes a wrong gold timestamp that survives after the drift
  // banner clears — refuse the mark itself, not just the later verdict.
  const markFrame = useCallback((): number | null => {
    if (formDisabled || saveInFlight.current) return null;
    if (viewerDrift !== null) {
      setActionError(
        "Frame drift detected — re-seek (arrow keys) until the banner clears before marking."
      );
      return null;
    }
    return controlsRef.current?.pause() ?? frame;
  }, [frame, viewerDrift, formDisabled]);

  // -- Save flow ------------------------------------------------------------------
  const doSave = useCallback(
    async (
      episodeIndex: number,
      status: string,
      label: StageLabelRow | null,
      { isDraft = false }: { isDraft?: boolean } = {}
    ): Promise<boolean> => {
      if (!spec || !task || !draft || !predictionsReady) return false;
      if (pendingInputs.current.size > 0) {
        setDraftError("A timestamp contains unfinished or invalid text. Correct it or clear its input before saving or leaving.");
        return false;
      }
      try {
        await saveReview({
          task,
          dataset_repo: repoId,
          episode_index: BigInt(episodeIndex),
          taxonomy_version: spec.taxonomy_version,
          status,
          label: label ?? undefined,
          ...(spec.trajectory && status !== "cleared" ? {
            review_protocol: "structured-v1" as const,
            notes: draft.humanNotes ?? "",
            event_links: draft.eventLinks ?? [],
          } : {}),
          ...draft.attribution,
          blind: blind && !everUnblindedRef.current,
          episode_duration_s: episodeDurationS ?? undefined,
        });
        return true;
      } catch (err) {
        const message = `Episode ${episodeIndex}: ${(err as Error).message}`;
        if (isDraft) setDraftError(message);
        else setActionError(message);
        return false;
      }
    },
    [spec, task, saveReview, repoId, draft, predictionsReady, blind, episodeDurationS]
  );

  const advance = useCallback(
    (fromIndex: number) => {
      const position = filteredEpisodes.findIndex(
        (episode) => episode.episodeIndex === fromIndex
      );
      if (position === -1) return;
      const next = filteredEpisodes[position + 1];
      if (next) setSelectedEpisode(next.episodeIndex);
    },
    [filteredEpisodes, setSelectedEpisode]
  );

  const verdict = useCallback(
    async (status: "confirmed" | "uncertain") => {
      if (selectedEpisode === null || formDisabled || saveInFlight.current || !spec) return;
      if (!outcomeGateReady || resolveOutcome(selectedEpisode) === null) {
        setActionError(
          `Episode ${selectedEpisode} has no outcome decision yet — the outcome ` +
            "review flow must run before stage labeling."
        );
        return;
      }
      if (pending === null || draft?.key !== sourceKey) {
        setActionError(`Episode ${selectedEpisode} is still loading — wait for the prefill.`);
        return;
      }
      if (viewerDrift !== null) {
        setActionError(
          `Video/frame-counter drift detected (${viewerDrift}) — re-seek before a verdict; ` +
            "event times derive from the displayed frame."
        );
        return;
      }
      if (status !== "uncertain" && unverifiable) {
        setActionError(
          "This browser cannot verify seek landings (no requestVideoFrameCallback) — " +
            "committed verdicts need Chrome/Edge; you may still save as uncertain."
        );
        return;
      }
      if (status !== "uncertain" && cameraKeys.length === 0) {
        setActionError(
          "No reviewable camera streams — a committed verdict needs the video; " +
            "save as uncertain instead."
        );
        return;
      }
      // Committed verdicts are validator-gated (uncertain included). The escape
      // hatch for "I cannot make this consistent" is the draft autosave.
      if (violations.length > 0 && status !== "uncertain") {
        setActionError(
          `Cannot ${status}: ${violations.length} consistency violation(s) — fix them or save as uncertain.`
        );
        return;
      }
      saveInFlight.current = true;
      setSaving(true);
      setActionError(null);
      const savedKey = draft.key;
      const ok = await doSave(selectedEpisode, status, pending);
      saveInFlight.current = false;
      setSaving(false);
      if (ok) {
        markSaved(savedKey);
        advance(selectedEpisode);
      }
    },
    [
      selectedEpisode,
      formDisabled,
      draft,
      sourceKey,
      markSaved,
      spec,
      outcomeGateReady,
      resolveOutcome,
      pending,
      viewerDrift,
      unverifiable,
      cameraKeys,
      violations,
      doSave,
      advance,
    ]
  );

  // Source/episode changes and exits await autosave. A failed save keeps the
  // form open. Committed reviews can only be changed by an explicit verdict.
  const leaveForm = useCallback(async (next: () => void, exiting = false) => {
    if (saveInFlight.current || saving) return;
    if (pendingInputs.current.size > 0) {
      setDraftError("A timestamp contains unfinished or invalid text. Correct it or clear its input before saving or leaving.");
      return;
    }
    if (exiting && unsavedCount > (dirty ? 1 : 0)) {
      setDraftError("Save the retained edits in the other episode or prediction version before leaving stage review.");
      return;
    }
    if (dirty && pending !== null && selectedEpisode !== null && draft !== null) {
      const own = ownReviewByEpisode.get(selectedEpisode);
      if (own && (own.status === "confirmed" || own.status === "corrected")) {
        setDraftError(`Episode ${selectedEpisode}: confirm or mark uncertain before leaving; ` +
          `unsaved edits to your ${own.status} review are still in the form.`);
        return;
      }
      saveInFlight.current = true;
      setSaving(true);
      const ok = await doSave(selectedEpisode, "draft", pending, { isDraft: true });
      saveInFlight.current = false;
      setSaving(false);
      if (!ok) return;
      markSaved(draft.key);
    }
    setDraftError(null);
    allowSavedNavigation.current = true;
    try { next(); } finally { allowSavedNavigation.current = false; }
  }, [saving, dirty, pending, selectedEpisode, draft, ownReviewByEpisode, doSave, markSaved, unsavedCount]);
  const navigateTo = useCallback((episodeIndex: number) => {
    if (episodeIndex === selectedEpisode) return;
    void leaveForm(() => setSelectedEpisode(episodeIndex));
  }, [leaveForm, selectedEpisode, setSelectedEpisode]);
  const exitReview = () => void leaveForm(onExit, true);
  const openOutcomeReview = () => void leaveForm(onOpenOutcomeReview, true);

  // -- Keyboard ----------------------------------------------------------------
  function handleKey(event: KeyboardEvent) {
    if (isTypingTarget(event)) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key;

    if (showHelp && (key === "Escape" || key === "q" || key === "?")) {
      event.preventDefault();
      setShowHelp(false);
      return;
    }
    if (key === "Escape" || key === "q") {
      event.preventDefault();
      exitReview();
      return;
    }
    if (key === "?") {
      event.preventDefault();
      setShowHelp(true);
      return;
    }
    if (!currentEpisode || !spec || formDisabled || saveInFlight.current) return;

    if (/^[0-9]$/.test(key)) {
      event.preventDefault();
      // A keystroke before the prefill lands would seed pending with a bare
      // stage and permanently suppress the prefill (its guard sees a non-null
      // pending) — the reviewer would label from a blank form unknowingly.
      if (pending === null) return;
      const stage = parseInt(key, 10);
      if (stage > spec.ladder.max_stage) {
        setActionError(`S${stage} is beyond this ladder (max S${spec.ladder.max_stage}).`);
      } else {
        edit({ [spec.stage_field]: stage });
      }
      return;
    }

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
        stepFrame(-30);
        return;
      case "]":
        event.preventDefault();
        stepFrame(30);
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
        // Preserve native keyboard activation of focused form controls.
        if (event.target instanceof Element && event.target.closest("button, summary, a[href]")) return;
        event.preventDefault();
        controlsRef.current?.togglePlay();
        return;
      case "-":
      case "=": {
        event.preventDefault();
        if (pending === null) return; // same prefill-suppression guard as digits
        const current =
          typeof pending[spec.stage_field] === "number"
            ? (pending[spec.stage_field] as number)
            : null;
        if (current === null) {
          setActionError("No stage set yet — pick a rung (0-9 or the buttons) first.");
          return;
        }
        const next = clamp(current + (key === "=" ? 1 : -1), 0, spec.ladder.max_stage);
        edit({ [spec.stage_field]: next });
        return;
      }
      case "e":
        event.preventDefault();
        setShowEvidence((show) => !show);
        return;
      case "c":
        event.preventDefault();
        void verdict("confirmed");
        return;
      case "u":
        event.preventDefault();
        void verdict("uncertain");
        return;
      case "n": {
        event.preventDefault();
        const position = filteredEpisodes.findIndex(
          (episode) => episode.episodeIndex === currentEpisode.episodeIndex
        );
        const next = position === -1 ? undefined : filteredEpisodes[position + 1];
        if (next) navigateTo(next.episodeIndex);
        else setActionError("Already at the last episode in the queue.");
        return;
      }
      case "p":
      case "b": {
        event.preventDefault();
        if (saving) return;
        const position = filteredEpisodes.findIndex(
          (episode) => episode.episodeIndex === currentEpisode.episodeIndex
        );
        if (position === -1) {
          setActionError("This episode is not in the current queue; pick one from the list.");
          return;
        }
        const previous = filteredEpisodes[position - 1];
        if (previous) navigateTo(previous.episodeIndex);
        else setActionError("Already at the first episode in the queue.");
        return;
      }
      default:
        return;
    }
  }
  useWindowKeydown(handleKey);

  // Stage timeline markers: the policy-phase end (episodes keep recording
  // through the physical reset; times beyond it are invalid) + set event times.
  const renderTimelineOverlays = useCallback(
    (pct: (value: number) => string) => {
      if (!spec || !currentEpisode) return null;
      const policyEnd =
        episodeDurationS !== null ? Math.round(episodeDurationS * spec.fps) : null;
      return (
        <>
          {policyEnd !== null && policyEnd < currentEpisode.rawLength && (
            <div
              className="absolute top-0 bottom-0 bg-warm-300/50"
              style={{
                left: pct(policyEnd),
                right: 0,
                backgroundImage:
                  "repeating-linear-gradient(45deg, rgba(138,127,114,0.3) 0 5px, transparent 5px 10px)",
              }}
              title={`policy phase ends at frame ${policyEnd} (reset tail beyond)`}
            />
          )}
          {spec.time_fields.map((tf) => {
            const t = pending && typeof pending[tf] === "number" ? (pending[tf] as number) : null;
            if (t === null) return null;
            const dotFrame = Math.min(
              Math.round(t * spec.fps),
              currentEpisode.rawLength - 1
            );
            return (
              <div
                key={tf}
                className="absolute bottom-1 w-2 h-2 rounded-full bg-teal"
                style={{ left: pct(dotFrame), transform: "translateX(-50%)" }}
                title={`${tf} = ${t}s`}
              />
            );
          })}
        </>
      );
    },
    [spec, currentEpisode, episodeDurationS, pending]
  );

  // -- Gates --------------------------------------------------------------------
  if (!viewer?.isEditor) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
        <button
          onClick={exitReview}
          className="text-xs text-ink-muted hover:text-teal mb-4 cursor-pointer"
        >
          &larr; Back to explorer
        </button>
        <p className="font-body text-ink-muted text-center">
          {viewer === undefined
            ? "Checking permissions…"
            : "Stage review is limited to allowlisted editors."}
        </p>
      </div>
    );
  }
  if (task && specRows !== undefined && specRows.length === 0) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
        <button
          onClick={exitReview}
          className="text-xs text-ink-muted hover:text-teal mb-4 cursor-pointer"
        >
          &larr; Back to explorer
        </button>
        <p className="font-body text-ink-muted text-center">
          No stage-label spec is registered for task {task}: register
          one with the Python client (upsert_stage_task_spec) first.
        </p>
      </div>
    );
  }
  if (specResult.error !== null) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
        <button
          onClick={exitReview}
          className="text-xs text-ink-muted hover:text-teal mb-4 cursor-pointer"
        >
          &larr; Back to explorer
        </button>
        <div className="rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral font-mono">
          Stage spec for {task}@{specRow?.taxonomy_version} is malformed — re-run the
          exporter: {specResult.error}
        </div>
      </div>
    );
  }
  if (!task || spec === null) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted font-body">
        {task ? "Loading stage spec…" : "This dataset has no task — stage review needs one."}
      </div>
    );
  }

  const currentOwn = selectedEpisode !== null ? ownReviewByEpisode.get(selectedEpisode) : undefined;
  // Blinded double-labeling: ANOTHER reviewer's stage decision in the ledger
  // would anchor this reviewer far more directly than a prefill — filter
  // stage-kind events not authored by the signed-in reviewer. Outcome events
  // stay (outcome is legitimate stage evidence, e.g. success <=> top rung).
  const fullChain = selectedEpisode !== null ? (historyByEpisode.get(selectedEpisode) ?? []) : [];
  const currentChain = fullChain.filter(
    (event) => event.label_kind !== "stage" || event.source.agent === viewer?.username
  );

  return (
    <div className="bg-white rounded-2xl border border-warm-200 shadow-sm overflow-clip">
      {showHelp && (
        <HelpOverlay
          title="Stage review shortcuts"
          keys={HELP_KEYS}
          onClose={() => setShowHelp(false)}
        />
      )}

      {/* Header */}
      <div className="px-6 py-4 border-b border-warm-100 bg-warm-50 flex flex-wrap items-center justify-between gap-4">
        <div>
          <button
            onClick={exitReview}
            className="text-xs text-ink-muted hover:text-teal cursor-pointer"
          >
            &larr; Back to explorer
          </button>
          <h2 className="font-display text-xl text-ink mt-1">Stage review</h2>
          <p className="text-xs text-ink-muted font-mono mt-0.5">
            {repoId} · {task} · S0–S{spec.ladder.max_stage}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 min-w-0">
          {specRows && specRows.length > 1 && (
            <select
              value={spec.taxonomy_version}
              onChange={(e) => {
                const next = e.target.value;
                void leaveForm(() => {
                  setSearchParams({ schema: next === liveRow?.taxonomy_version ? null : next, prediction: null });
                });
                e.currentTarget.blur();
              }}
              className="max-w-full rounded-lg border border-warm-200 bg-white px-2 py-1 text-xs font-mono text-ink cursor-pointer"
              disabled={saving}
              aria-label="Taxonomy version"
              title="Taxonomy (schema) version — candidates coexist with the live one"
            >
              {specRows.map((row) => (
                <option key={row.taxonomy_version} value={row.taxonomy_version}>
                  {row.taxonomy_version}
                  {row.live ? " (live)" : " (candidate)"}
                </option>
              ))}
            </select>
          )}
          <label className="flex flex-col gap-1 text-[10px] font-mono text-ink-muted">
            Prediction version
            <select
              aria-label="Prediction version"
              value={predictionParam}
              disabled={saving || predictionVersions === undefined}
              onChange={(event) => {
                const next = event.target.value;
                void leaveForm(() => setPredictionParam(next));
                event.currentTarget.blur();
              }}
              className="max-w-[300px] rounded-lg border border-warm-200 bg-white px-2 py-1 text-xs text-ink"
            >
              {!predictionParam && <option value="">Loading versions…</option>}
              {predictionSelection?.error && <option value={predictionParam}>Unavailable version</option>}
              <option value={NO_PREDICTIONS}>No predictions</option>
              {predictionVersions?.runs.map((run) => (
                <option key={run._id} value={run._id}>
                  {blind ? run._id : run.run_key}{run._id === predictionVersions.active_run_id ? " (active)" : ""}
                </option>
              ))}
            </select>
          </label>
          {!specRow?.live && (
            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-gold-light text-gold">
              candidate taxonomy
            </span>
          )}
          <button
            onClick={() => { if (blind) unblind(); else setBlindParam("1"); }}
            className={`px-2 py-1 rounded-lg text-xs font-medium cursor-pointer ${
              blind
                ? "bg-teal/10 text-teal border border-teal/40"
                : "bg-coral-light text-coral border border-coral/40"
            }`}
            title="Blind mode hides (and never fetches) policy/arm identity — the protocol forbids it from influencing labels"
          >
            {blind ? "blind" : "⚠ unblinded"}
          </button>
          <span className="text-xs font-mono text-ink-muted">
            {dirty
              ? "unsaved edits"
              : currentOwn
                ? `${currentOwn.status} ✓`
                : "no review yet"}
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

      {otherSchemaPredictions && otherSchemaPredictions.length > 0 && <div className="px-6 py-2 border-b border-warm-200 text-xs text-ink-muted">
        This episode also has predictions under a separate taxonomy:
        {otherSchemaPredictions.map((available) => <button key={available.taxonomy_version}
          className="ml-2 text-teal underline disabled:opacity-40" disabled={saving}
          onClick={() => void leaveForm(() => setSearchParams({ schema: available.taxonomy_version, prediction: available.run_id }))}>
          Open {available.taxonomy_version} ({available.expected_count} episodes in this run)
        </button>)}
      </div>}

      {loadError && (
        <div className="mx-6 mt-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral font-mono">
          Failed to load episode metadata: {loadError}
        </div>
      )}
      {draftError && (
        <div className="mx-6 mt-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-xs text-coral font-mono flex items-start justify-between gap-3">
          <span>{draftError}</span>
          <button
            onClick={() => setDraftError(null)}
            className="shrink-0 underline cursor-pointer"
          >
            dismiss
          </button>
        </div>
      )}
      {schemaFellBack && (
        <div className="mx-6 mt-4 rounded-lg border border-gold/40 bg-gold-light px-4 py-3 text-xs text-ink font-mono">
          Unknown taxonomy version "{schemaParam}" in the URL — showing the live
          taxonomy instead.
        </div>
      )}
      {historyError && (
        <div className="mx-6 mt-4 rounded-lg border border-gold/40 bg-gold-light px-4 py-3 text-xs text-ink font-mono">
          Label-history ledger failed to load (provenance hidden, reviewing
          unaffected): {historyError}
        </div>
      )}
      {ledgerError && !blind && (
        <div className="mx-6 mt-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-xs text-coral font-mono">
          arm filter unavailable — ledger parse failed: {ledgerError}
        </div>
      )}
      {appliedError && (
        <div className="mx-6 mt-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-xs text-coral font-mono">
          Failed to load the applied outcome-edit record — episodes whose only
          outcome decision lives there stay gated until it loads: {appliedError}
        </div>
      )}
      {awaitingOutcome.length > 0 && (
        <div className="mx-6 mt-4 rounded-lg border border-gold/40 bg-gold-light px-4 py-3 text-xs text-ink font-mono flex items-center justify-between gap-3">
          <span>
            {awaitingOutcome.length} episode(s) have no outcome decision yet and
            are hidden from the stage queue — outcome review runs first (ep{" "}
            {awaitingOutcome.slice(0, 8).join(", ")}
            {awaitingOutcome.length > 8 ? ", …" : ""}).
          </span>
          <button
            onClick={openOutcomeReview}
            className="shrink-0 px-2 py-1 rounded-lg text-xs font-medium bg-white border border-gold text-gold hover:bg-gold/10 cursor-pointer"
          >
            Open outcome review &rarr;
          </button>
        </div>
      )}

      {predictionError && (
        <div role="alert" className="m-4 rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral">
          {predictionError} The review form is disabled.
        </div>
      )}
      {!predictionsReady && !predictionError && (
        <div role="status" className="m-4 text-sm text-ink-muted">
          Loading prediction version{predictionSelection?.runId
            ? `: ${predictionPages.results.length} of ${selectedRun?.expected_count ?? "?"} episodes`
            : "…"}. The review form will open after the complete version loads.
        </div>
      )}
      {selectedRun && (
        <details className="px-6 py-2 border-b border-warm-100 text-[11px] font-mono text-ink-muted break-all">
          <summary className="cursor-pointer">Prediction version details</summary>
          {selectedRun._id === predictionVersions?.active_run_id ? "Active" : "Inactive"} prediction version: {blind ? selectedRun._id : selectedRun.run_key}
          {!blind && <>{" · "}{selectedRun.pipeline.name}@{selectedRun.pipeline.version}</>}
          {" · published "}{new Date(selectedRun.published_at).toISOString()}
          {" · "}{selectedRun._id}
        </details>
      )}
      {pendingInputCount > 0 && <p role="alert" className="px-6 py-2 text-xs text-coral">A timestamp contains unfinished or invalid text. Correct it or clear its input before saving or leaving.</p>}
      {unsavedCount > (dirty ? 1 : 0) && (
        <div role="alert" className="m-4 rounded-lg border border-gold/40 bg-gold-light px-4 py-3 text-sm text-ink">
          Unsaved edits are retained in this tab for another episode or prediction version.
          Return to that selection to save them before closing the tab.
        </div>
      )}

      <div
        className={`grid gap-0 ${
          showEvidence ? "lg:grid-cols-[190px_minmax(0,1fr)] 2xl:grid-cols-[190px_minmax(0,1fr)_280px]" : "lg:grid-cols-[190px_minmax(0,1fr)]"
        }`}
      >
        {/* Queue */}
        <div className="border-r border-warm-100 p-4 flex flex-col gap-2 max-h-[32vh] lg:max-h-[85vh] lg:sticky lg:top-4">
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value as StatusFilter);
              e.currentTarget.blur();
            }}
            className="w-full rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink cursor-pointer"
          >
            <option value="unreviewed">Unreviewed (+ drafts)</option>
            <option value="draft">Drafts</option>
            <option value="confirmed">Confirmed</option>
            <option value="corrected">Corrected</option>
            <option value="uncertain">Uncertain (mine)</option>
            <option value="adjudicate">Needs adjudication (others&apos; uncertain)</option>
            <option value="all">All</option>
          </select>
          <div className="flex gap-2">
            <select
              value={flagFilter}
              onChange={(e) => {
                setFlagFilter(e.target.value);
                e.currentTarget.blur();
              }}
              className="flex-1 rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink cursor-pointer"
              title="Review-reason flags from the labeling pipeline"
            >
              <option value="all">All flags</option>
              <option value="flagged">Flagged</option>
              <option value="unflagged">Unflagged</option>
            </select>
            <select
              value={confFilter}
              onChange={(e) => {
                setConfFilter(e.target.value);
                e.currentTarget.blur();
              }}
              className="flex-1 rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink cursor-pointer"
            >
              <option value="all">All conf</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
          {!blind && armOptions.length > 0 && (
            <select
              value={armFilter}
              onChange={(e) => {
                setArmFilter(e.target.value);
                e.currentTarget.blur();
              }}
              className="w-full rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink cursor-pointer"
            >
              <option value="all">All arms</option>
              {armOptions.map((arm) => (
                <option key={arm} value={arm}>
                  {arm}
                </option>
              ))}
            </select>
          )}
          <div className="text-[11px] font-mono text-ink-muted">
            {Number(reviews?.num_confirmed ?? 0) + Number(reviews?.num_corrected ?? 0)} committed ·{" "}
            {prefillByEpisode.size} predictions · {filteredEpisodes.length} in queue
            {awaitingOutcome.length > 0 && ` · ${awaitingOutcome.length} outcome-gated`}
          </div>
          <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
            {episodes === null && !loadError && (
              <div className="flex items-center gap-2 text-xs text-ink-muted">
                <div className="w-4 h-4 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
                Loading episodes…
              </div>
            )}
            {episodes !== null && !outcomeGateReady && (
              <div className="flex items-center gap-2 text-xs text-ink-muted">
                <div className="w-4 h-4 border-2 border-teal/30 border-t-teal rounded-full animate-spin" />
                Checking outcome-review coverage…
              </div>
            )}
            {filteredEpisodes.map((episode) => {
              const prefill = prefillByEpisode.get(episode.episodeIndex);
              const own = ownReviewByEpisode.get(episode.episodeIndex);
              const stage = prefill?.label[spec.stage_field];
              const flags = flagCount(episode.episodeIndex);
              const outcome = resolveOutcome(episode.episodeIndex)?.outcome ?? null;
              return (
                <button
                  key={episode.episodeIndex}
                  onClick={() => navigateTo(episode.episodeIndex)}
                  className={`w-full text-left px-3 py-2 rounded-lg border transition-all cursor-pointer ${
                    selectedEpisode === episode.episodeIndex
                      ? "bg-teal/10 border-teal shadow-sm"
                      : "bg-white border-warm-200 hover:border-warm-300"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs font-medium text-ink">
                      Ep {episode.episodeIndex}
                    </span>
                    <span className="flex items-center gap-1.5">
                      {prefill?.confidence && (
                        <span
                          className={`w-2 h-2 rounded-full ${
                            CONFIDENCE_DOT[prefill.confidence] ?? "bg-warm-300"
                          }`}
                          title={`${prefill.confidence} confidence`}
                        />
                      )}
                      {outcome !== null && (
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            OUTCOME_CHIP[outcome] ?? "bg-warm-100 text-ink-muted"
                          }`}
                          title={`outcome decision: ${outcome}`}
                        >
                          {outcome[0]}
                        </span>
                      )}
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-warm-100 text-ink">
                        {stageDisplay(stage)}
                      </span>
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2 mt-1">
                    <span className="text-[10px] font-mono text-ink-muted truncate">
                      {flags > 0 ? `⚑ ${flags}` : ""}
                      {!blind && ledgerArms?.get(episode.episodeIndex)
                        ? ` · ${ledgerArms.get(episode.episodeIndex)}`
                        : ""}
                    </span>
                    {statusFilter === "adjudicate" ? (
                      <span className="text-[10px] font-mono text-gold">
                        ?{" "}
                        {(otherReviewsByEpisode.get(episode.episodeIndex) ?? [])
                          .filter((r) => r.status === "uncertain")
                          .map((r) => r.reviewer)
                          .join(", ")}
                      </span>
                    ) : own ? (
                      <span
                        className={`text-[10px] font-mono ${
                          own.status === "confirmed" || own.status === "corrected"
                            ? "text-teal"
                            : "text-ink-muted"
                        }`}
                        title={`${own.reviewer} · ${formatClock(own.savedAt)}`}
                      >
                        {STATUS_GLYPH[own.status] ?? ""} {own.status}
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono text-ink-muted/60">unreviewed</span>
                    )}
                  </div>
                </button>
              );
            })}
            {episodes !== null && outcomeGateReady && filteredEpisodes.length === 0 && (
              <div className="text-xs text-ink-muted font-body">
                {awaitingOutcome.length === episodes.length && episodes.length > 0
                  ? "Every episode is awaiting outcome review — run it first."
                  : "No episodes match this filter."}
              </div>
            )}
          </div>
        </div>

        {/* Viewer + form */}
        <div className="p-4 min-w-0">
          {currentEpisode === null ? (
            <div className="py-16 text-center text-ink-muted font-body">
              Select an episode from the queue to review it.
            </div>
          ) : currentResolvedOutcome === null ? (
            // Outcome gate: this episode has never been through the outcome
            // editor flow — stage labeling is blocked until it has.
            <div className="py-16 text-center font-body">
              <p className="text-ink">
                Episode {currentEpisode.episodeIndex} has no outcome decision yet.
              </p>
              <p className="mt-1 text-sm text-ink-muted">
                The outcome review flow must run before stage labeling.
              </p>
              <button
                onClick={openOutcomeReview}
                className="mt-4 px-4 py-1.5 rounded-lg text-xs font-medium bg-teal text-white hover:bg-teal/90 cursor-pointer"
              >
                Open outcome review &rarr;
              </button>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <span className="font-display text-lg text-ink">
                  Episode {currentEpisode.episodeIndex}
                </span>
                {pending === null && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-warm-100 text-ink-muted animate-pulse">
                    loading label…
                  </span>
                )}
                {currentResolvedOutcome !== undefined &&
                  (currentResolvedOutcome.outcome !== null ? (
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        OUTCOME_CHIP[currentResolvedOutcome.outcome] ??
                        "bg-warm-100 text-ink-muted"
                      }`}
                      title={`outcome decision (source: ${currentResolvedOutcome.source})`}
                    >
                      outcome {currentResolvedOutcome.outcome}
                    </span>
                  ) : currentOutcomeSignalError === null ? (
                    <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-warm-100 text-ink-muted animate-pulse">
                      resolving outcome…
                    </span>
                  ) : null)}
                {inheritedSuccess && spec && (
                  <span
                    className="px-2 py-0.5 rounded-full text-xs font-medium bg-teal/10 text-teal"
                    title="Stage, final state, and failure mode were seeded from the SUCCESS outcome decision — mark the event timings, then confirm."
                  >
                    S{spec.ladder.success_level} inherited from outcome
                  </span>
                )}
                {currentOwn && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-teal/10 text-teal">
                    your {currentOwn.status} · {formatClock(currentOwn.savedAt)}
                  </span>
                )}
                {currentPrefill && !currentOwn && !blind && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-warm-100 text-ink-muted">
                    selected model {currentPrefill.pipeline.name}@{currentPrefill.pipeline.version}
                  </span>
                )}
                {episodeDurationS !== null && (
                  <span className="text-[11px] font-mono text-ink-muted">
                    policy {episodeDurationS.toFixed(1)}s / raw {currentEpisode.rawLength}f
                  </span>
                )}
                <div className="flex-1" />
                <button
                  onClick={() => setShowEvidence((show) => !show)}
                  className="px-2 py-1 rounded-lg text-xs font-medium border border-warm-200 text-ink-muted hover:border-warm-300 cursor-pointer"
                >
                  {showEvidence ? "hide evidence" : "model evidence"}
                  <span className="ml-1 font-mono text-[10px] opacity-60">e</span>
                </button>
              </div>

              {draft && (
                <div className="mb-3 rounded-lg border border-warm-200 px-3 py-2 text-[11px] text-ink-muted">
                  <p>{draft.fromOwnReview
                    ? "Your saved label is in the form. The selected model prediction is shown in model evidence."
                    : draft.attribution.copied_from_review_id
                      ? "The form was copied from another human review. Its original prediction source is preserved."
                      : currentPrefill
                      ? "The form started from the selected prediction; edits are your review."
                      : "No model prediction seeded this form."}</p>
                  <details className="mt-1"><summary className="cursor-pointer text-teal">Saved label provenance</summary>
                    <p className="font-mono break-all">Review source: {attributionDescription(draft.attribution)}</p>
                    <p className="mt-1">Human labels can be scored against other predictions using compatible labeling definitions. The prediction shown during annotation is recorded for the audit.</p>
                  </details>
                  {inheritedSuccess && <p className="mt-1">Without a prediction, the form starts from the human success outcome.</p>}
                </div>
              )}

              {needsPolicyDuration && policyDurationS === null && <p role="alert" className="mb-3 rounded-lg border border-gold/40 bg-gold-light p-3 text-xs text-ink">
                {currentOutcomeSignalError ? `Cannot determine policy-phase duration: ${currentOutcomeSignalError}. New annotation is blocked.`
                  : "Loading the validated policy-phase duration before starting a new annotation…"}
              </p>}

              {currentOutcomeSignalError !== null && !needsPolicyDuration && (
                <div className="mb-3 rounded-lg border border-gold/40 bg-gold-light px-3 py-2 text-xs text-ink font-mono">
                  This keep-as-is outcome could not be resolved from frame
                  signals — labeling proceeds without the success prefill:{" "}
                  {currentOutcomeSignalError}
                </div>
              )}
              {actionError && (
                <div className="mb-3 rounded-lg border border-coral/30 bg-coral-light px-3 py-2 text-xs text-coral font-mono">
                  {actionError}
                </div>
              )}

              <div className={spec.trajectory ? "grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.9fr)] items-start" : ""}>
              <div className={spec.trajectory ? "min-w-0 xl:sticky xl:top-4" : ""}>
              {cameraKeys.length === 0 ? (
                <div className="rounded-lg border border-coral/30 bg-coral-light px-4 py-3 text-sm text-coral font-mono">
                  Episode {currentEpisode.episodeIndex} exposes no reviewable camera
                  streams.
                </div>
              ) : (
                <ReviewViewer
                  datasetId={repoId}
                  episode={currentEpisode}
                  cameraKeys={cameraKeys}
                  primaryKey={primaryKey}
                  fps={spec.fps}
                  frame={frame}
                  onFrame={setFrame}
                  lastValidFrame={null}
                  controlsRef={controlsRef}
                  cropByCameraKey={cropByCameraKey}
                  storedFrameHW={storedFrameHW}
                  onDrift={setViewerDrift}
                  onUnverifiable={setUnverifiable}
                  renderTimelineOverlays={renderTimelineOverlays}
                />
              )}
              </div>

              <div className={spec.trajectory ? "min-w-0" : ""}>
              {pending !== null && (
                <StageLabelForm
                  key={sourceKey}
                  onPendingInputChange={onPendingInputChange}
                  hasPendingInput={pendingInputCount > 0}
                  spec={spec}
                  row={pending}
                  eventLinks={draft?.eventLinks ?? []}
                  onEventLinksChange={(links) => {
                    if (!formDisabled && !saveInFlight.current) editEventLinks(links);
                  }}
                  humanNotes={draft?.humanNotes ?? ""}
                  onHumanNotesChange={(notes) => {
                    if (!formDisabled && !saveInFlight.current) editHumanNotes(notes);
                  }}
                  violations={violations}
                  frame={frame}
                  markFrame={markFrame}
                  markDisabled={viewerDrift !== null || formDisabled}
                  onEdit={edit}
                  onSeekTime={seekTime}
                  disabled={formDisabled}
                  blind={blind}
                />
              )}
              </div>
              </div>

              {blind && (
                <button onClick={unblind} className="mt-3 text-xs text-teal hover:underline cursor-pointer">
                  Show provenance and unblind
                </button>
              )}

              {statusFilter === "adjudicate" &&
                selectedEpisode !== null &&
                (otherReviewsByEpisode.get(selectedEpisode) ?? []).map((other) => (
                  <div
                    key={other.id}
                    className="mt-3 rounded-lg border border-gold/40 bg-gold-light px-3 py-2"
                  >
                    <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-ink">
                      <span className="font-medium">
                        {other.reviewer}: {other.status}
                      </span>
                      <span className="text-ink-muted">{formatClock(other.savedAt)}</span>
                      {other.label && spec && (
                        <span className="text-ink-muted">
                          {stageDisplay(other.label[spec.stage_field])} ·{" "}
                          {blind && !spec.failure_modes.includes(String(other.label[spec.failure_mode_field]))
                            ? "invalid failure value" : String(other.label[spec.failure_mode_field] ?? "—")} →{" "}
                          {blind && !spec.final_states.includes(String(other.label[spec.final_state_field]))
                            ? "invalid final state" : String(other.label[spec.final_state_field] ?? "—")}
                        </span>
                      )}
                      <div className="flex-1" />
                      {other.label && (
                        <button
                          onClick={() => {
                            if (!formDisabled && !saveInFlight.current) replaceLabel(other.label!, {
                              ...other.attribution, copied_from_review_id: other.id,
                            }, other.eventLinks);
                          }}
                          className="px-2 py-0.5 rounded text-[10px] font-mono bg-white border border-gold text-gold hover:bg-gold/10 cursor-pointer"
                          disabled={formDisabled}
                          title={`load ${other.reviewer}'s label into the form as your starting point`}
                        >
                          load their label
                        </button>
                      )}
                    </div>
                    {!blind && other.humanNotes && <p className="mt-1 text-xs text-ink whitespace-pre-wrap">Reviewer notes: {other.humanNotes}</p>}
                    {!blind && typeof other.label?.notes === "string" && other.label.notes && <details className="mt-1 text-xs text-ink-muted">
                      <summary className="cursor-pointer">Retained source notes · not reviewer notes</summary>
                      <p className="whitespace-pre-wrap">{other.label.notes}</p>
                    </details>}
                  </div>
                ))}

              <LabelHistoryPanel
                chain={currentChain}
                rawLength={currentEpisode.rawLength}
                onSeek={jumpToFrame}
              />

              {/* Verdict bar */}
              <div className="sticky bottom-0 z-20 mt-4 flex flex-wrap items-center gap-2 border-t border-warm-200 bg-white px-2 py-3">
                <span className="text-xs text-ink-muted">{dirty ? "Unsaved changes" : currentOwn ? "Your saved label" : "Reviewing prediction"}</span>
                <button
                  disabled={formDisabled}
                  onClick={() => void verdict("uncertain")}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium border border-warm-200 text-ink-muted hover:border-warm-300 cursor-pointer"
                >
                  uncertain
                  <span className="ml-1.5 font-mono text-[10px] opacity-60">u</span>
                </button>
                <div className="flex-1" />
                <button
                  disabled={formDisabled}
                  onClick={() => void verdict("confirmed")}
                  className={`px-4 py-1.5 rounded-lg text-xs font-medium ${
                    saving
                      ? "bg-warm-100 text-ink-muted/50 cursor-not-allowed"
                      : "bg-teal text-white hover:bg-teal/90 cursor-pointer"
                  }`}
                >
                  {spec.trajectory ? "confirm — labels reviewed" : "confirm — fully annotated"}
                  <span className="ml-1.5 font-mono text-[10px] opacity-70">c</span>
                </button>
              </div>
            </>
          )}
        </div>

        {/* Evidence rail */}
        {showEvidence && (
          <div className="border-l border-warm-100 p-4 max-h-[85vh] overflow-y-auto lg:col-start-2 2xl:col-start-auto">
            <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-2">
              Model evidence
            </div>
            <EvidencePanel spec={spec} prefill={currentPrefill} onSeekTime={seekTime} blind={blind} onUnblind={unblind} />
          </div>
        )}
      </div>
    </div>
  );
}
