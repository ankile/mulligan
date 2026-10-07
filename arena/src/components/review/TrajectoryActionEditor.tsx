import { useState } from "react";
import type { StageLabelRow } from "../../../convex/stageConsistency";
import type { KeyActionDefinition } from "../../../convex/trajectoryContract";
import type { StageLabelFormProps } from "./StageLabelForm";
import { TimeControls } from "./ReviewTimeControls";
import { setActionOccurrences, patchActionOccurrence, removeActionOccurrence, clearActionOccurrences, sortActionOccurrencesByTime } from "../../lib/trajectoryActionEdits";

const buttonClass = "rounded-lg border border-warm-200 px-2 py-1.5 text-xs text-teal hover:bg-warm-50 cursor-pointer disabled:opacity-40";
const inputClass = "rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-sm min-w-0 disabled:opacity-40";
const isRecord = (value: unknown): value is StageLabelRow => value !== null && typeof value === "object" && !Array.isArray(value);
const validTime = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;

/** Occurrences are the sole editable event facts. Redundant wire fields are
 * synchronized only by explicit edits; existing conflicting drafts stay intact. */
export function TrajectoryActionEditor({ action, index, definition, onChange, ...props }: StageLabelFormProps & {
  action: StageLabelRow;
  index: number;
  definition?: KeyActionDefinition;
  onChange: (action: StageLabelRow) => void;
}) {
  const [undo, setUndo] = useState<StageLabelRow | null>(null);
  const name = `Action ${index + 1}`;
  const structureDisabled = props.disabled || props.hasPendingInput;
  const events = Array.isArray(action.occurrences) && action.occurrences.every(isRecord) ? action.occurrences : null;
  const times = events?.map((event) => event.time_s).filter(validTime) ?? [];
  const earliest = times.length ? Math.min(...times) : null;
  // Match trajectoryContract.ts::timesClose without normalizing source bytes.
  const summaryMatches = action.first_time_s === earliest || (validTime(action.first_time_s) && earliest !== null && Math.abs(action.first_time_s - earliest) <= 1e-6);
  const inconsistent = events !== null && (action.occurred !== (events.length > 0) || !summaryMatches);
  const outOfOrder = events !== null && events.every((event) => validTime(event.time_s)) && events.some((event, i) => i > 0 && Number(events[i - 1].time_s) > Number(event.time_s));
  const change = (next: StageLabelRow) => { setUndo(null); onChange(next); };
  const clear = () => { setUndo(action); onChange(clearActionOccurrences(action)); };
  const append = () => {
    if (!events) return;
    change(setActionOccurrences(action, [...events, { attempt_index: 1, time_s: null, confidence: "medium", evidence: "" }]));
  };
  const applySummaryTime = () => {
    if (!events || !validTime(action.first_time_s)) return;
    if (!events.length) {
      change(setActionOccurrences(action, [{ attempt_index: 1, time_s: action.first_time_s, confidence: "medium", evidence: "" }]));
    } else {
      const first = earliest === null ? 0 : events.findIndex((event) => event.time_s === earliest);
      change(patchActionOccurrence(action, first, { time_s: action.first_time_s }));
    }
  };
  return <section className="rounded-lg border border-warm-200 p-3 space-y-3" aria-label={`${name}: ${definition?.name ?? "Invalid action"}`}>
    <div className="flex items-center justify-between gap-2">
      <strong className="text-sm"><span className="text-ink-muted font-normal mr-1">{index + 1}.</span> {definition?.name ?? "Invalid action"}</strong>
      <label className="flex items-center gap-2 text-xs">Occurred
        <select aria-label={`${name} occurred`} className={inputClass} disabled={structureDisabled || !events} value={typeof action.occurred === "boolean" ? String(action.occurred) : "unset"}
          onChange={(event) => {
            if (event.target.value === "false") clear();
            else if (events?.length) change(setActionOccurrences(action, events));
            else append();
          }}>
          {typeof action.occurred !== "boolean" && <option value="unset" disabled>Unset</option>}
          <option value="true">Yes</option><option value="false">No</option>
        </select>
      </label>
    </div>
    {definition && <p className="text-xs text-ink-muted">{definition.description}</p>}
    {undo && <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">Event times removed.
      <button className={buttonClass} disabled={structureDisabled} onClick={() => { onChange(undo); setUndo(null); }}>Undo {name.toLowerCase()} removal</button>
    </div>}
    {inconsistent && <div role="alert" className="rounded-lg border border-gold/40 bg-gold-light p-2 space-y-2 text-xs">
      <p>This saved draft has conflicting action fields. Its earlier summary says {action.occurred === true ? "Yes" : action.occurred === false ? "No" : "unset"}, first time {validTime(action.first_time_s) ? `${action.first_time_s}s` : "unset"}. Choose which observation to keep.</p>
      <div className="flex flex-wrap gap-2">
        <button className={buttonClass} disabled={structureDisabled} onClick={() => change(setActionOccurrences(action, events!))}>Use listed events</button>
        {action.occurred === false && <button className={buttonClass} disabled={structureDisabled} onClick={clear}>Keep No and clear events</button>}
        {validTime(action.first_time_s) && !summaryMatches && <button className={buttonClass} disabled={structureDisabled} onClick={applySummaryTime}>Use summary time {action.first_time_s}s</button>}
      </div>
    </div>}
    {events ? <>
      {events.length === 0 && <p className="text-xs text-ink-muted">No event recorded. Choose Yes to add a time.</p>}
      {events.map((event, eventIndex) => {
        const eventName = `${name} occurrence ${eventIndex + 1}`;
        const t = typeof event.time_s === "number" && Number.isFinite(event.time_s) ? event.time_s : null;
        const edit = (patch: StageLabelRow) => change(patchActionOccurrence(action, eventIndex, patch));
        const validConfidence = ["low", "medium", "high"].includes(String(event.confidence));
        return <div key={eventIndex} className="space-y-2 border-t border-warm-100 pt-2">
          <div role="group" aria-label={`${eventName} time`} className="flex flex-wrap items-center gap-1">
            <span className="text-xs text-ink-muted mr-1">{events.length === 1 ? "Time" : `Time ${eventIndex + 1}`}</span>
            <TimeControls comfortable preservePrecision onPendingInputChange={props.onPendingInputChange} t={t} fps={props.spec.fps} frame={props.frame} flagged={t === null} disabled={props.disabled}
              markDisabled={props.markDisabled} markTitle="Set event time from displayed frame" onCommit={(time_s) => edit({ time_s })}
              onMark={() => { const frame = props.markFrame(); if (frame === null) return false; edit({ time_s: frame / props.spec.fps }); return true; }}
              onSeek={props.onSeekTime} canClear={event.time_s !== null} onClear={() => edit({ time_s: null })} clearTitle={`Clear ${eventName} time`} />
            <button className={buttonClass} aria-label={`Remove ${eventName}`} disabled={structureDisabled} onClick={() => { setUndo(action); onChange(removeActionOccurrence(action, eventIndex)); }}>Remove</button>
          </div>
          {t === null && <p className="text-xs text-coral">Mark or enter this event's time before confirming.</p>}
          <details className="text-xs">
            <summary className="cursor-pointer text-teal">Event details · attempt {typeof event.attempt_index === "number" && Number.isFinite(event.attempt_index) ? event.attempt_index : "unset"}</summary>
            <div className="flex flex-wrap items-center gap-2 mt-2">
              <label className="flex items-center gap-2">Attempt<input aria-label={`${eventName} attempt`} className={`${inputClass} w-16`} type="number" min="1" step="1" disabled={props.disabled}
                value={typeof event.attempt_index === "number" && Number.isFinite(event.attempt_index) ? event.attempt_index : ""} onChange={(e) => edit({ attempt_index: e.target.value === "" ? null : Number(e.target.value) })} /></label>
              <label className="flex items-center gap-2">Source confidence (not reviewed)<select aria-label={`${eventName} confidence`} className={inputClass} disabled={props.disabled} value={validConfidence ? String(event.confidence) : "unset"} onChange={(e) => edit({ confidence: e.target.value })}>
                {!validConfidence && <option value="unset" disabled>Unset or invalid</option>}
                {["low", "medium", "high"].map((value) => <option key={value}>{value}</option>)}
              </select></label>
            </div>
            {!props.blind && <label className="block mt-2 text-ink-muted">Retained source evidence (not human notes)<textarea aria-label={`${eventName} evidence`} className={`${inputClass} block w-full mt-1`} rows={2} readOnly value={typeof event.evidence === "string" ? event.evidence : ""} /></label>}
          </details>
        </div>;
      })}
      {(events.length > 0 || action.occurred === true) && <button className={buttonClass} disabled={structureDisabled} onClick={append}>{events.length ? "Add another" : "Add"} {name.toLowerCase()} time</button>}
      {outOfOrder && <div className="flex flex-wrap items-center gap-2 text-xs text-coral">Event times are out of order.
        <button className={buttonClass} disabled={structureDisabled} onClick={() => change(sortActionOccurrencesByTime(action))}>Sort {name.toLowerCase()} times</button>
      </div>}
    </> : <p role="alert" className="text-xs text-coral">This action has an invalid event list. Inspect the original source before replacing it.</p>}
  </section>;
}
