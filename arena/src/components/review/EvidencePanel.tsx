import { attributionDescription, stageDisplay, type PredictionAttribution } from "../../lib/stagePredictionReview";
import type { ExportedStageSpec } from "../../../convex/stageConsistency";

// ---------------------------------------------------------------------------
// Model-evidence rail for the stage review: the prediction's pipeline identity,
// review-reason flags, confidence, sample-vote summary, and click-to-seek
// telemetry. Collapsed by default (form an opinion before seeing the model's).
// Raw model text/provenance can contain arm identity and is rendered only after
// explicit unblinding; structured prediction fields remain visible.
// ---------------------------------------------------------------------------

export interface StagePrefillView {
  label: Record<string, unknown>;
  reviewReason: string | null;
  violationCodes: string[];
  confidence: string | null;
  voteSummary: Record<string, unknown> | null;
  episodeDurationS: number | null;
  pipeline: { name: string; version: string; git_commit: string };
  evidence: Record<string, unknown>;
  pushedAt: number;
  attribution: PredictionAttribution;
  canonicalResponse?: unknown;
  sourceRevision?: string;
}

const CONFIDENCE_CHIP: Record<string, string> = {
  high: "bg-teal-light text-teal",
  medium: "bg-gold-light text-gold",
  low: "bg-coral-light text-coral",
};

export function EvidencePanel({
  spec,
  prefill,
  onSeekTime,
  blind,
  onUnblind,
}: {
  spec: ExportedStageSpec;
  prefill: StagePrefillView | null;
  onSeekTime: (timeS: number) => void;
  blind: boolean;
  onUnblind: () => void;
}) {
  if (prefill === null) {
    return (
      <div className="text-xs text-ink-muted font-body px-1">
        No pipeline prediction for this episode under taxonomy{" "}
        {spec.taxonomy_version}.
      </div>
    );
  }
  const reasons = (blind ? "" : prefill.reviewReason ?? "")
    .split(";")
    .map((r) => r.trim())
    .filter(Boolean);
  const label = prefill.label;
  const predictedTimes = spec.time_fields
    .map((tf) => [tf, label[tf]] as const)
    .filter(([, v]) => typeof v === "number");

  return (
    <div className="space-y-3 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        {!blind && <span
          className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-warm-100 text-ink"
          title={`git ${prefill.pipeline.git_commit} · pushed ${new Date(prefill.pushedAt).toLocaleString()}`}
        >
          {prefill.pipeline.name}@{prefill.pipeline.version}
        </span>}
        {prefill.confidence && (
          <span
            className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
              CONFIDENCE_CHIP[prefill.confidence] ?? "bg-warm-100 text-ink-muted"
            }`}
          >
            {prefill.confidence} confidence
          </span>
        )}
      </div>

      <div className="space-y-1 text-[11px] font-mono text-ink-muted">
        <p>Prediction duration: {prefill.episodeDurationS ?? "unknown"}s</p>
        {prefill.sourceRevision && <p className="break-all">Prediction source revision: {blind && !/^[a-f0-9]{40,64}$/.test(prefill.sourceRevision)
          ? "Unblind to inspect source identifier" : prefill.sourceRevision}</p>}
        <p>Outcome decisions shown in the review queue come from the current outcome records.</p>
      </div>

      {reasons.length > 0 && (
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">
            Review flags
          </div>
          <div className="flex flex-wrap gap-1">
            {reasons.map((reason) => (
              <span
                key={reason}
                className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                  reason.startsWith("consistency:")
                    ? "bg-coral-light text-coral"
                    : "bg-gold-light text-gold"
                }`}
              >
                {reason}
              </span>
            ))}
          </div>
        </div>
      )}

      <div>
        <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">
          Prediction
        </div>
        <div className="space-y-0.5 font-mono text-[11px] text-ink">
          <p>
            {spec.stage_field}: {stageDisplay(label[spec.stage_field])}
          </p>
          <p className="text-ink-muted">{blind && !spec.failure_modes.includes(String(label[spec.failure_mode_field]))
            ? "Invalid failure value; unblind to inspect" : String(label[spec.failure_mode_field] ?? "—")}</p>
          <p className="text-ink-muted">→ {blind && !spec.final_states.includes(String(label[spec.final_state_field]))
            ? "Invalid final state; unblind to inspect" : String(label[spec.final_state_field] ?? "—")}</p>
        </div>
      </div>

      {!blind && prefill.voteSummary && (
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">
            Sample votes
          </div>
          <div className="space-y-0.5 font-mono text-[11px] text-ink-muted">
            {Object.entries(prefill.voteSummary).map(([key, value]) => (
              <p key={key}>
                {key}: {String(value)}
              </p>
            ))}
          </div>
        </div>
      )}

      {predictedTimes.length > 0 && (
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">
            Predicted event times
          </div>
          <div className="space-y-0.5">
            {predictedTimes.map(([tf, v]) => (
              <button
                key={tf}
                onClick={() => onSeekTime(v as number)}
                className="block w-full text-left font-mono text-[11px] text-teal hover:underline cursor-pointer"
                title={`seek to ${v}s`}
              >
                {tf}: {String(v)}s → f{Math.round((v as number) * spec.fps)}
              </button>
            ))}
          </div>
        </div>
      )}

      {blind ? (
        <div className="text-[11px] text-ink-muted">
          <p>Pipeline identity, free-text notes, and raw evidence are hidden while blind.</p>
          <button onClick={onUnblind} className="mt-2 text-teal hover:underline cursor-pointer">
            Show provenance and unblind
          </button>
        </div>
      ) : <details className="text-[11px] font-mono text-ink-muted">
        <summary className="cursor-pointer">Full prediction and provenance</summary>
        <p className="mt-2 break-all">{attributionDescription(prefill.attribution)}</p>
        <pre className="mt-2 whitespace-pre-wrap break-all">{JSON.stringify({
          label: prefill.label,
          canonical_response: prefill.canonicalResponse,
          pipeline: prefill.pipeline,
          evidence: prefill.evidence,
        }, null, 2)}</pre>
      </details>}

      {!blind && typeof prefill.evidence.run_name === "string" && (
        <p className="text-[10px] font-mono text-ink-muted/70 break-all">
          run {prefill.evidence.run_name}
          {typeof prefill.evidence.model === "string" ? ` · ${prefill.evidence.model}` : ""}
        </p>
      )}
    </div>
  );
}
