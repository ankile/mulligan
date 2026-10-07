import type { TrajectoryEventLink } from "../../convex/trajectoryEventLinks";
import type { Id } from "../release/api";
import type { ExportedStageSpec, StageLabelRow } from "../../convex/stageConsistency";

/** Prediction-version selector value for reviewing without a prediction run. */
export const NO_PREDICTIONS = "none";

/** Captured when the form is seeded, never inferred from the current selector. */
export interface PredictionAttribution {
  prediction_id?: Id<"stagePredictions">;
  prediction_sha256?: string;
  prefill_pushed_at?: number;
  episode_duration_s?: number;
  copied_from_review_id?: Id<"stageReviews">;
}

export interface ReviewSeed {
  label: StageLabelRow;
  /** Human-authored review notes; never seeded from prediction label.notes. */
  humanNotes?: string;
  eventLinks?: TrajectoryEventLink[];
  attribution: PredictionAttribution;
  inheritedSuccess: boolean;
  fromOwnReview: boolean;
}

export function seedStageReview({
  own,
  prediction,
  outcome,
  spec,
  sourceFree,
  emptyLabel = {},
}: {
  own?: { label: StageLabelRow | null; attribution: PredictionAttribution; humanNotes?: string; eventLinks?: TrajectoryEventLink[] };
  prediction?: { label: StageLabelRow; attribution: PredictionAttribution };
  outcome: string | null;
  spec: ExportedStageSpec;
  sourceFree: boolean;
  emptyLabel?: StageLabelRow;
}): ReviewSeed {
  const fromOwnReview = own?.label != null;
  const label = { ...(fromOwnReview ? own.label : prediction?.label ?? emptyLabel) };
  // A form without a prediction source starts from a successful outcome. An
  // immutable prediction is shown exactly as registered, including
  // disagreement with human outcomes.
  const inheritedSuccess = !fromOwnReview && sourceFree && !spec.trajectory && outcome === "success";
  if (inheritedSuccess) {
    label[spec.stage_field] = spec.ladder.success_level;
    label[spec.final_state_field] = spec.success_final_state;
    label[spec.failure_mode_field] = "none";
  }
  return {
    label,
    ...(fromOwnReview && own.humanNotes !== undefined ? { humanNotes: own.humanNotes } : {}),
    ...(fromOwnReview && own.eventLinks !== undefined ? { eventLinks: own.eventLinks.map((link) => ({ ...link })) } : {}),
    attribution: { ...(fromOwnReview ? own.attribution : prediction?.attribution ?? {}) },
    inheritedSuccess,
    fromOwnReview,
  };
}

export function attributionDescription(source: PredictionAttribution): string {
  const copied = source.copied_from_review_id
    ? `Copied from human review ${source.copied_from_review_id}. Original source: ` : "";
  if (source.prediction_id) {
    return `${copied}prediction ${source.prediction_id} · SHA-256 ${source.prediction_sha256}`;
  }
  return `${copied}No prediction source recorded`;
}

/** Validate against the query list BEFORE sending an ID to Convex. */
export function resolvePredictionSelection(
  selected: string,
  runs: ReadonlyArray<{ _id: string }>,
): { runId: Id<"stagePredictionRuns"> | null; error: string | null } {
  if (selected === NO_PREDICTIONS) return { runId: null, error: null };
  const run = runs.find((candidate) => candidate._id === selected);
  if (!run) {
    return {
      runId: null,
      error: `Prediction version ${selected} is not published for this dataset and taxonomy. Choose a listed version.`,
    };
  }
  return { runId: run._id as Id<"stagePredictionRuns">, error: null };
}

/** Raw invalid label values may contain policy identity; keep blind displays numeric. */
export function stageDisplay(value: unknown): string {
  if (value === null || value === undefined) return "—";
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? `S${value}` : "invalid stage";
}
