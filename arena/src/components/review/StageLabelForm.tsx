import type { TrajectoryEventLink } from "../../../convex/trajectoryEventLinks";
import { useMemo, useState } from "react";
import { TimeControls } from "./ReviewTimeControls";
import { TrajectoryLabelForm } from "./TrajectoryLabelForm";
import type { ExportedStageSpec, StageLabelRow, Violation } from "../../../convex/stageConsistency";

// ---------------------------------------------------------------------------
// Spec-driven stage-label form. NOTHING here is task-specific: field names,
// ladder rungs, enums, and event fields all come from the exported spec, so a
// taxonomy change (split/removed stages, renamed fields) never touches this
// file. Values live in a Record keyed by the spec's own field names (the same
// row shape the validator, Convex, and gold consolidation consume).
// ---------------------------------------------------------------------------

function violationFields(violations: Violation[]): Set<string> {
  return new Set(violations.flatMap((v) => v.fields));
}

export interface StageLabelFormProps {
  spec: ExportedStageSpec;
  blind?: boolean;
  row: StageLabelRow;
  humanNotes?: string;
  eventLinks?: TrajectoryEventLink[];
  onEventLinksChange?: (links: TrajectoryEventLink[]) => void;
  onHumanNotesChange?: (notes: string) => void;
  violations: Violation[];
  /** Current playhead frame (display only). */
  frame: number;
  /** Snap-and-return the DISPLAYED frame (ViewerControls.pause() ?? frame);
   *  null when the display is unverified (drift) — the mark is refused. */
  markFrame: () => number | null;
  /** Frame verification failed — marking is disabled until it clears. */
  markDisabled: boolean;
  onEdit: (patch: StageLabelRow) => void;
  onSeekTime: (timeS: number) => void;
  disabled: boolean;
  onPendingInputChange?: (id: string, pending: boolean) => void;
  /** Do not reorder index-keyed event editors while local timestamp text is unfinished. */
  hasPendingInput?: boolean;
}

export function StageLabelForm(props: StageLabelFormProps) {
  return props.spec.trajectory ? <TrajectoryLabelForm {...props} /> : <FlatStageLabelForm {...props} />;
}

function FlatStageLabelForm({
  spec,
  row,
  violations,
  frame,
  markFrame,
  markDisabled,
  onEdit,
  onSeekTime,
  disabled,
  blind = false,
  onPendingInputChange,
}: StageLabelFormProps) {
  const flagged = violationFields(violations);
  const stage = typeof row[spec.stage_field] === "number" ? (row[spec.stage_field] as number) : null;
  const boolSet = new Set(spec.bool_fields);

  // DISPLAY pairing of bools to their event time. The validator's bool<=>time
  // contract only couples strict `<bool>_time_s` names (the consistency-rule
  // convention); semantically-paired names like approach_reached /
  // approach_time_s are joined here for the FORM only, by longest-prefix match
  // (convention pairs claimed first, each time claimed at most once; checked
  // against the field descriptions of every exported spec). No validation,
  // storage, or export changes ride on this map.
  const { pairedTime, claimedTimes } = useMemo(() => {
    const timeSet = new Set(spec.time_fields);
    const paired = new Map<string, string>();
    const claimed = new Set<string>();
    for (const bf of spec.bool_fields) {
      const tf = `${bf}_time_s`;
      if (timeSet.has(tf)) {
        paired.set(bf, tf);
        claimed.add(tf);
      }
    }
    for (const bf of spec.bool_fields) {
      if (paired.has(bf)) continue;
      const parts = bf.split("_");
      for (let k = parts.length - 1; k >= 1; k--) {
        const tf = `${parts.slice(0, k).join("_")}_time_s`;
        if (timeSet.has(tf) && !claimed.has(tf)) {
          paired.set(bf, tf);
          claimed.add(tf);
          break;
        }
      }
    }
    return { pairedTime: paired, claimedTimes: claimed };
  }, [spec]);
  const selectedLevel =
    stage !== null ? spec.ladder.levels.find((lvl) => lvl.sid === stage) : undefined;
  const [showFullLadder, setShowFullLadder] = useState(false);

  const timeOf = (tf: string): number | null =>
    typeof row[tf] === "number" ? (row[tf] as number) : null;

  const enumSelect = (field: string, options: string[]) => {
    const value = typeof row[field] === "string" ? (row[field] as string) : "";
    const unknown = value !== "" && !options.includes(value);
    return (
      <select
        value={blind && unknown ? "__invalid_prediction__" : value}
        disabled={disabled}
        onChange={(e) => {
          onEdit({ [field]: e.target.value || undefined });
          e.currentTarget.blur();
        }}
        className={`w-full rounded-lg border bg-white px-2 py-1.5 text-xs font-mono text-ink cursor-pointer ${
          flagged.has(field) || unknown ? "border-coral ring-1 ring-coral/40" : "border-warm-200"
        }`}
      >
        <option value="">— unset —</option>
        {unknown && (
          <option value={blind ? "__invalid_prediction__" : value} disabled>
            ⚠ {blind ? "Invalid value; unblind to inspect" : value} (not in taxonomy {spec.taxonomy_version})
          </option>
        )}
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    );
  };

  return (
    <div className="mt-4 space-y-3">
      {/* Stage ladder buttons */}
      <div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mr-1">
            {spec.stage_field}
          </span>
          {spec.ladder.levels.map((lvl) => (
            <button
              key={lvl.sid}
              disabled={disabled}
              onClick={() => onEdit({ [spec.stage_field]: lvl.sid })}
              title={lvl.text}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium cursor-pointer transition-all ${
                stage === lvl.sid
                  ? "bg-teal text-white"
                  : flagged.has(spec.stage_field)
                    ? "bg-white border border-coral/50 text-ink-muted"
                    : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300"
              }`}
            >
              S{lvl.sid}
              {(lvl.gate_field || lvl.gate_any_of.length > 0 || lvl.gate_all_of.length > 0) && (
                <span className="ml-0.5 opacity-60" title="gated rung">
                  ⚿
                </span>
              )}
            </button>
          ))}
          <button
            onClick={() => setShowFullLadder((show) => !show)}
            className="ml-1 text-[10px] font-mono text-ink-muted hover:text-teal cursor-pointer"
          >
            {showFullLadder ? "▾ ladder" : "▸ ladder"}
          </button>
        </div>
        {selectedLevel && !showFullLadder && (
          <p className="mt-1 text-[11px] text-ink-muted font-body">
            S{selectedLevel.sid}: {selectedLevel.text}
          </p>
        )}
        {showFullLadder && (
          <div className="mt-1 space-y-0.5">
            {spec.ladder.levels.map((lvl) => (
              <p
                key={lvl.sid}
                className={`text-[11px] font-body ${
                  stage === lvl.sid ? "text-ink font-medium" : "text-ink-muted"
                }`}
              >
                S{lvl.sid}: {lvl.text}
              </p>
            ))}
          </div>
        )}
      </div>

      {/* Enums */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">
            {spec.final_state_field}
          </div>
          {enumSelect(spec.final_state_field, spec.final_states)}
        </div>
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">
            {spec.failure_mode_field}
          </div>
          {enumSelect(spec.failure_mode_field, spec.failure_modes)}
        </div>
      </div>

      {/* Event fields, in spec order: bools (with their paired time when the
          ${bf}_time_s convention names one) and standalone times (e.g.
          approach_time_s pairing semantically with approach_reached) — every
          time field gets mark/seek/clear. */}
      <div className="space-y-1">
        {spec.event_fields.map((f) => {
          if (boolSet.has(f.name)) {
            const bf = f.name;
            const tf = pairedTime.get(bf) ?? `${bf}_time_s`;
            const hasTime = pairedTime.has(bf);
            const boolVal = row[bf] === true;
            const t = hasTime ? timeOf(tf) : null;
            const rowFlagged = flagged.has(bf) || (hasTime && flagged.has(tf));
            return (
              <div
                key={bf}
                className={`flex items-center gap-2 rounded-lg px-2 py-1 ${
                  rowFlagged ? "ring-1 ring-coral/50 bg-coral-light/30" : ""
                }`}
              >
                <label
                  className="flex items-center gap-1.5 text-[11px] font-mono text-ink cursor-pointer flex-1 min-w-0"
                  title={f.description}
                >
                  <input
                    type="checkbox"
                    checked={boolVal}
                    disabled={disabled}
                    onChange={(e) => onEdit({ [bf]: e.target.checked })}
                  />
                  <span className="truncate">{bf}</span>
                </label>
                {hasTime && (
                  <TimeControls
                    onPendingInputChange={onPendingInputChange}
                    t={t}
                    fps={spec.fps}
                    frame={frame}
                    flagged={flagged.has(tf)}
                    disabled={disabled}
                    markDisabled={markDisabled}
                    markTitle={`set ${bf} + time from the displayed frame`}
                    onCommit={(next) => onEdit({ [tf]: next ?? undefined })}
                    onMark={() => {
                      const at = markFrame();
                      if (at === null) return false; // drifted display — refused
                      onEdit({
                        [bf]: true,
                        [tf]: Math.round((at / spec.fps) * 100) / 100,
                      });
                      return true;
                    }}
                    onSeek={onSeekTime}
                    canClear={row[bf] !== undefined || t !== null}
                    onClear={() => onEdit({ [bf]: false, [tf]: undefined })}
                    clearTitle="clear (bool=false, time absent)"
                  />
                )}
              </div>
            );
          }
          // A standalone time field (no bool claims it, by convention or pairing).
          const tf = f.name;
          if (claimedTimes.has(tf)) {
            return null; // rendered inline with its bool above
          }
          const t = timeOf(tf);
          return (
            <div
              key={tf}
              className={`flex items-center gap-2 rounded-lg px-2 py-1 ${
                flagged.has(tf) ? "ring-1 ring-coral/50 bg-coral-light/30" : ""
              }`}
            >
              <span
                className="text-[11px] font-mono text-ink flex-1 min-w-0 truncate"
                title={f.description}
              >
                {tf}
              </span>
              <TimeControls
                    onPendingInputChange={onPendingInputChange}
                t={t}
                fps={spec.fps}
                frame={frame}
                flagged={flagged.has(tf)}
                disabled={disabled}
                markDisabled={markDisabled}
                markTitle={`set ${tf} from the displayed frame`}
                onCommit={(next) => onEdit({ [tf]: next ?? undefined })}
                onMark={() => {
                  const at = markFrame();
                  if (at === null) return false; // drifted display — refused
                  onEdit({ [tf]: Math.round((at / spec.fps) * 100) / 100 });
                  return true;
                }}
                onSeek={onSeekTime}
                canClear={t !== null}
                onClear={() => onEdit({ [tf]: undefined })}
                clearTitle="clear (time absent)"
              />
            </div>
          );
        })}
      </div>

      {/* Notes */}
      {blind ? (
        <p className="mt-3 text-xs text-ink-muted">Free-text notes are hidden while blind; their stored content is preserved.</p>
      ) : <textarea
        value={typeof row.notes === "string" ? (row.notes as string) : ""}
        disabled={disabled}
        onChange={(e) => onEdit({ notes: e.target.value || undefined })}
        placeholder="notes…"
        rows={2}
        className="w-full rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs font-body text-ink"
      />}

      {/* Live violations */}
      {violations.length > 0 && (
        <div className="rounded-lg border border-coral/30 bg-coral-light px-3 py-2 space-y-0.5">
          {violations.map((v, i) => (
            <p key={i} className="text-[11px] font-mono text-coral">
              [{v.code}] {blind ? `${v.fields.join(", ")} needs review. Unblind to inspect raw details.` : v.message}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
