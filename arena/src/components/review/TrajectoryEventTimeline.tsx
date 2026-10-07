import { useEffect, useId, useState, type ReactNode } from "react";
import type { StageLabelRow } from "../../../convex/stageConsistency";
import type { TrajectoryEventLink } from "../../../convex/trajectoryEventLinks";
import { analyzeTrajectoryTimeline } from "../../../convex/trajectoryTimeline";
import { getUnifiedEventCandidates, setUnifiedEventTime, getPrimaryFailureIndex, selectPrimaryFailure, patchFailureEvent, removeFailureEvent } from "../../lib/trajectoryUnifiedEvents";
import { patchActionOccurrence, removeActionOccurrence, setActionOccurrences } from "../../lib/trajectoryActionEdits";
import { TrajectoryActionEditor } from "./TrajectoryActionEditor";
import { TimeControls } from "./ReviewTimeControls";
import type { StageLabelFormProps } from "./StageLabelForm";

const buttonClass = "rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-xs text-teal hover:bg-warm-50 cursor-pointer disabled:opacity-40";
const inputClass = "rounded-lg border border-warm-200 bg-white px-2 py-1.5 text-sm min-w-0 max-w-full disabled:opacity-40";
const object = (value: unknown): value is StageLabelRow => value !== null && typeof value === "object" && !Array.isArray(value);
const records = (value: unknown): value is StageLabelRow[] => Array.isArray(value) && Array.from(value).every(object);
const timeValue = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const readable = (value: string) => value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());

/** Keep incomplete attempt text local, just like timestamp text, so clearing
 * an input cannot split a shared event or lose the user's draft on navigation. */
function AttemptInput({ value, disabled, name, onCommit, onPending }: {
  value: unknown; disabled: boolean; name: string; onCommit: (attempt: number) => void;
  onPending: StageLabelFormProps["onPendingInputChange"];
}) {
  const id = useId();
  const shown = typeof value === "number" && Number.isFinite(value) ? String(value) : "";
  const [text, setText] = useState(shown);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- controlled bridge preserves incomplete local text.
    setText(shown);
  }, [shown]);
  useEffect(() => () => onPending?.(id, false), [id, onPending]);
  const valid = text.trim() !== "" && Number.isSafeInteger(Number(text)) && Number(text) > 0;
  return <input className={`${inputClass} w-16`} aria-label={name} type="text" inputMode="numeric" aria-invalid={!valid}
    disabled={disabled} value={text} onKeyDown={(e) => e.stopPropagation()} onChange={(e) => {
      const next = e.target.value; setText(next);
      const validNext = next.trim() !== "" && Number.isSafeInteger(Number(next)) && Number(next) > 0;
      onPending?.(id, !validNext);
      if (validNext) onCommit(Number(next));
    }} />;
}

/** One visible timestamp per shared physical event. The per-field wire values are
 * materialized only on explicit edits; rendering never normalizes source data. */
export function TrajectoryEventTimeline(props: StageLabelFormProps) {
  const { row, spec, disabled } = props;
  const tag = spec.trajectory!;
  const task = tag.task_definition;
  const links = props.eventLinks ?? [];
  const [undo, setUndo] = useState<{ before: StageLabelRow; links: TrajectoryEventLink[]; after: string } | null>(null);
  const [order, setOrder] = useState<"chronological" | "recorded">("chronological");
  const malformed = props.violations.some((v) => v.code === "trajectory_shape");
  const structuralDisabled = disabled || props.hasPendingInput || malformed;
  const actions = records(row.key_action_observations) ? row.key_action_observations : null;
  const transitions = records(row.stage_transitions) ? row.stage_transitions : null;
  const failures = records(row.failure_events) ? row.failure_events : null;
  const groups = malformed ? [] : getUnifiedEventCandidates(tag, row);
  const issues = analyzeTrajectoryTimeline(tag, row);
  const linkFor = (group: typeof groups[number]) => links.find((link) => link.action_id === group.actionId && link.stage_id === group.stageId && link.attempt_index === group.attemptIndex);
  const sameRef = (link: TrajectoryEventLink, group: typeof groups[number]) => link.action_id === group.actionId && link.stage_id === group.stageId && link.attempt_index === group.attemptIndex;
  const validLinks = links.filter((link, index) => groups.some((group) => sameRef(link, group)) && links.findIndex((other) => other.action_id === link.action_id && other.stage_id === link.stage_id && other.attempt_index === link.attempt_index) === index);
  const paired = groups.filter((group) => linkFor(group)?.relation !== "distinct");
  const primaryIndex = failures ? getPrimaryFailureIndex(row) : null;
  const noFailure = row.failure_mode === task.successDefinition.noFailureModeId && row.primary_failure_time_s === null;
  const signature = (label: StageLabelRow, eventLinks: TrajectoryEventLink[]) => JSON.stringify([label, eventLinks]);
  const change = (next: StageLabelRow, nextLinks = links, remember = false) => {
    if (signature(next, nextLinks) === signature(row, links)) return;
    setUndo(remember ? { before: row, links, after: signature(next, nextLinks) } : null);
    props.onEdit(next);
    props.onEventLinksChange?.(nextLinks);
  };
  const withoutGroup = (group: typeof groups[number]) => links.filter((link) => !sameRef(link, group));
  const withoutGroupOrLinks = (group: typeof groups[number] | undefined) => group ? withoutGroup(group) : links;
  const sharedTime = (group: typeof groups[number], time: number | null, remember = false) => {
    const next = setUnifiedEventTime(tag, row, group.id, time);
    change(next, [...withoutGroup(group), { action_id: group.actionId, stage_id: group.stageId, attempt_index: group.attemptIndex, relation: "shared" }], remember);
  };
  const separate = (group: typeof groups[number]) => change(row, [...withoutGroup(group), {
    action_id: group.actionId, stage_id: group.stageId, attempt_index: group.attemptIndex, relation: "distinct",
  }], true);
  const retainRefs = (next: StageLabelRow) => {
    const candidates = getUnifiedEventCandidates(tag, next);
    return links.filter((link) => candidates.some((group) => sameRef(link, group)));
  };
  const structuralChange = (next: StageLabelRow) => change(next, retainRefs(next), true);
  const stageTitle = (id: unknown) => {
    const stage = task.stages.find((value) => value.id === id);
    return stage ? `S${stage.index}: ${stage.name === `S${stage.index}` ? readable(stage.id) : stage.name}` : "Unset stage";
  };
  const eventTime = (name: string, value: unknown, edit: (time: number | null) => void, extraDisabled = false) => <div role="group" aria-label={name} className="flex flex-wrap items-center gap-1.5">
    <span className="text-xs text-ink-muted">Time</span>
    <TimeControls comfortable preservePrecision t={timeValue(value)} fps={spec.fps} frame={props.frame} flagged={timeValue(value) === null}
      disabled={disabled || extraDisabled} markDisabled={props.markDisabled || extraDisabled} onPendingInputChange={props.onPendingInputChange}
      onCommit={edit} onMark={() => { const frame = props.markFrame(); if (frame === null) return false; edit(frame / spec.fps); return true; }}
      markTitle="Set event time from displayed frame" onSeek={props.onSeekTime} canClear={value !== null} onClear={() => edit(null)} clearTitle={`Clear ${name}`} />
  </div>;
  const details = (name: string, event: StageLabelRow, edit: (patch: StageLabelRow) => void, children?: ReactNode, shared = false) => <details className="text-xs">
    <summary className="cursor-pointer text-teal">Event details · attempt {(typeof event.attempt_index === "number" && Number.isFinite(event.attempt_index) ? event.attempt_index : "unset")}</summary>
    <div className="mt-3 space-y-3">
      {children}
      <label className="flex flex-wrap items-center gap-2">Attempt<AttemptInput name={`${name} attempt`} value={event.attempt_index}
        disabled={disabled || malformed || shared} onPending={props.onPendingInputChange} onCommit={(attempt_index) => edit({ attempt_index })} /></label>
      {shared && <p className="text-ink-muted">Separate the events before changing their attempt or stage identity.</p>}
      <label className="flex flex-wrap items-center gap-2 text-ink-muted">Source confidence, excluded from review
        <select aria-label={`${name} confidence`} className={inputClass} disabled={disabled} value={['low', 'medium', 'high'].includes(String(event.confidence)) ? String(event.confidence) : ""} onChange={(e) => edit({ confidence: e.target.value })}>
          <option value="" disabled>Unset or invalid</option>{['low', 'medium', 'high'].map((value) => <option key={value}>{value}</option>)}
        </select>
      </label>
      {!props.blind && <p className="whitespace-pre-wrap text-ink-muted">Retained source evidence: {typeof event.evidence === "string" ? event.evidence : ""}</p>}
    </div>
  </details>;
  const issueNotes = (kind: "transition" | "failure", index: number, representedAction?: number) => issues.filter((issue) => issue.dependent.kind === kind && issue.dependent.index === index && issue.prerequisite.actionIndex !== representedAction).map((issue) => <div key={issue.id} role="alert" className="rounded-lg bg-gold-light p-2 text-xs space-y-2">
    <p>{issue.message}</p><button className={buttonClass} disabled={disabled} onClick={() => props.onSeekTime(issue.prerequisite.timeS)}>Watch {issue.prerequisite.label} at {issue.prerequisite.timeS.toFixed(2)} s</button>
  </div>);
  const stageSelect = (name: string, event: StageLabelRow, prefix: "from" | "to", edit: (patch: StageLabelRow) => void) => <label className="flex flex-col gap-1">{prefix === "from" ? "Previous stage" : "Stage reached"}
    <select aria-label={`${name} ${prefix} stage`} className={inputClass} disabled={structuralDisabled} value={typeof event[`${prefix}_stage_id`] === "string" ? String(event[`${prefix}_stage_id`]) : ""}
      onChange={(e) => { const stage = task.stages.find((s) => s.id === e.target.value)!; edit({ [`${prefix}_stage_id`]: stage.id, [`${prefix}_stage_index`]: stage.index }); }}>
      <option value="" disabled>Choose a stage</option>{task.stages.map((s) => <option key={s.id} value={s.id}>{stageTitle(s.id)}</option>)}
    </select></label>;
  const entries: { key: string; time: number | null; node: ReactNode }[] = [];
  actions?.forEach((action, actionIndex) => {
    if (!records(action.occurrences)) return;
    const occurrenceList = action.occurrences;
    const definition = task.keyActions.find((a) => a.id === action.action_id);
    occurrenceList.forEach((event, occurrenceIndex) => {
      const group = paired.find((g) => g.actionIndex === actionIndex && g.occurrenceIndex === occurrenceIndex);
      const linked = !!group && ((linkFor(group)?.relation === "shared" && (group.sameTime || (group.actionTimeS === null && group.stageTimeS === null))) || (group.relation === "equivalent" && group.sameTime));
      const name = `Action ${actionIndex + 1} occurrence ${occurrenceIndex + 1}`;
      const edit = (patch: StageLabelRow) => {
        const next = { ...row, key_action_observations: actions.map((a, i) => i === actionIndex ? patchActionOccurrence(a, occurrenceIndex, patch) : a) };
        change(next, retainRefs(next));
      };
      const remove = () => {
        let next: StageLabelRow = { ...row, key_action_observations: actions.map((a, i) => i === actionIndex ? removeActionOccurrence(a, occurrenceIndex) : a) };
        if (group && linked) next = { ...next, stage_transitions: transitions!.filter((_, i) => i !== group.transitionIndex) };
        structuralChange(next);
      };
      const times = occurrenceList.map((e) => timeValue(e.time_s)).filter((t): t is number => t !== null);
      const earliest = times.length ? Math.min(...times) : null;
      const conflict = action.occurred !== true || (action.first_time_s !== earliest && !(typeof action.first_time_s === "number" && earliest !== null && Math.abs(action.first_time_s - earliest) <= 1e-6));
      entries.push({ key: `action:${actionIndex}:${occurrenceIndex}`, time: timeValue(event.time_s), node: <article aria-label={name} className="rounded-xl border border-warm-200 bg-white p-4 space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-2"><div>
          <h4 className="text-sm font-medium">{definition?.name ?? "Unset action"}{occurrenceList.length > 1 ? ` · occurrence ${occurrenceIndex + 1}` : ""}</h4>
          {group && <p className="mt-1 text-xs text-teal">{stageTitle(group.stageId)}{linked ? " · shared event" : " · related stage"}</p>}
        </div><button className={buttonClass} disabled={structuralDisabled} aria-label={`Remove ${name}`} onClick={remove}>{linked ? "Remove shared event" : "Remove"}</button></div>
        {definition && <p className="text-xs text-ink-muted">{definition.description}</p>}
        {group && !linked && <div role="group" aria-label={`Choose time for ${definition?.name ?? name}`} className="rounded-lg bg-gold-light p-3 text-xs space-y-2">
          <p>{group.sameTime ? "These observations have the same time. Link them if they describe one event." : "Two estimates for a related event. Watch them, then choose one shared time or keep separate events."}</p>
          <p className="text-ink-muted">{group.explanation}</p>
          <div className="flex flex-wrap gap-2">
            <button className={buttonClass} disabled={disabled || group.actionTimeS === null} onClick={() => { if (group.actionTimeS !== null) props.onSeekTime(group.actionTimeS); }}>Watch action · {(group.actionTimeS?.toFixed(2) ?? "unset")} s</button>
            <button className={buttonClass} disabled={disabled || group.stageTimeS === null} onClick={() => { if (group.stageTimeS !== null) props.onSeekTime(group.stageTimeS); }}>Watch stage · {(group.stageTimeS?.toFixed(2) ?? "unset")} s</button>
          </div>
          <div className="flex flex-wrap gap-2">
            <button className={buttonClass} disabled={structuralDisabled || group.actionTimeS === null} onClick={() => sharedTime(group, group.actionTimeS, true)}>Use {(group.actionTimeS?.toFixed(2) ?? "unset")} s for this event</button>
            {!group.sameTime && <button className={buttonClass} disabled={structuralDisabled || group.stageTimeS === null} onClick={() => sharedTime(group, group.stageTimeS, true)}>Use {(group.stageTimeS?.toFixed(2) ?? "unset")} s for this event</button>}
          </div>
        </div>}
        {eventTime(`${name} time`, event.time_s, (time) => group ? sharedTime(group, time, true) : edit({ time_s: time }), !!group && malformed)}
        {group && <p className="text-xs text-ink-muted">{linked ? "Editing this time updates the action and stage together." : "Entering or marking a time above uses it for both observations."}</p>}
        {group && group.relation !== "equivalent" && <button className={buttonClass} disabled={structuralDisabled} onClick={() => separate(group)}>Keep as separate events</button>}
        {group && issueNotes("transition", group.transitionIndex, !linked ? group.actionIndex : undefined)}
        {group && linked && <details className="text-xs"><summary className="cursor-pointer text-teal">Stage definition and retained evidence</summary>
          <p className="mt-2">{task.stages.find((s) => s.id === group.stageId)?.description}</p>
          {!props.blind && <p className="mt-2 text-ink-muted">Retained source evidence: {String(transitions![group.transitionIndex].evidence ?? "")}</p>}
        </details>}
        {conflict && <div role="alert" className="rounded-lg bg-gold-light p-2 text-xs space-y-2"><p>The saved action summary disagrees with its occurrences. Earlier summary: {action.occurred === true ? "Yes" : action.occurred === false ? "No" : "unset"}, first time {timeValue(action.first_time_s) ?? "unset"} s. Review the candidate observations before choosing.</p>
          <button className={buttonClass} disabled={structuralDisabled} onClick={() => change({ ...row, key_action_observations: actions.map((a, i) => i === actionIndex ? setActionOccurrences(a, occurrenceList) : a) }, links, true)}>Use listed events for action {actionIndex + 1}</button>
          {timeValue(action.first_time_s) !== null && <button className={buttonClass} disabled={structuralDisabled} onClick={() => {
            const firstIndex = occurrenceList.findIndex((e) => timeValue(e.time_s) === earliest);
            const next = { ...row, key_action_observations: actions.map((a, i) => i === actionIndex ? patchActionOccurrence(a, firstIndex < 0 ? 0 : firstIndex, { time_s: action.first_time_s }) : a) };
            // The old summary is only a candidate action observation; a stage
            // disagreement remains visible until the reviewer explicitly links it.
            change(next, withoutGroupOrLinks(group), true);
          }}>Use summary time {timeValue(action.first_time_s)} s</button>}
          {action.occurred === false && <button className={buttonClass} disabled={structuralDisabled} onClick={() => structuralChange({ ...row, key_action_observations: actions.map((a, i) => i === actionIndex ? setActionOccurrences(a, []) : a) })}>Keep No and clear events</button>}
        </div>}
        {details(name, event, (patch) => {
          if (group && 'attempt_index' in patch) {
            const next = { ...row,
              key_action_observations: actions.map((a, i) => i === actionIndex ? patchActionOccurrence(a, occurrenceIndex, patch) : a),
              stage_transitions: transitions!.map((stage, i) => i === group.transitionIndex ? { ...stage, attempt_index: patch.attempt_index } : stage),
            };
            const nextLinks = links.map((link) => sameRef(link, group) ? { ...link, attempt_index: Number(patch.attempt_index) } : link);
            change(next, nextLinks, true);
          } else edit(patch);
        }, group && <div className="grid gap-3">
          <p className="text-ink-muted">Stage fields describe this event. Changing the stage reached can separate it from the action.</p>
          {stageSelect(`Transition ${group.transitionIndex + 1}`, transitions![group.transitionIndex], "from", (patch) => {
            const next = { ...row, stage_transitions: transitions!.map((stage, i) => i === group.transitionIndex ? { ...stage, ...patch } : stage) };
            structuralChange(next);
          })}
          {stageSelect(`Transition ${group.transitionIndex + 1}`, transitions![group.transitionIndex], "to", (patch) => {
            const next = { ...row, stage_transitions: transitions!.map((stage, i) => i === group.transitionIndex ? { ...stage, ...patch } : stage) };
            structuralChange(next);
          })}
        </div>)}
      </article> });
    });
  });
  transitions?.forEach((event, index) => {
    if (paired.some((g) => g.transitionIndex === index)) return;
    const name = `Transition ${index + 1}`;
    const edit = (patch: StageLabelRow) => {
      const next = { ...row, stage_transitions: transitions.map((e, i) => i === index ? { ...e, ...patch } : e) };
      change(next, retainRefs(next));
    };
    const candidate = groups.find((g) => g.transitionIndex === index);
    entries.push({ key: `transition:${index}`, time: timeValue(event.time_s), node: <article aria-label={name} className="rounded-xl border border-warm-200 bg-white p-4 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2"><h4 className="text-sm font-medium">{stageTitle(event.to_stage_id)}</h4>
        <button className={buttonClass} disabled={structuralDisabled} aria-label={`Remove ${name}`} onClick={() => structuralChange({ ...row, stage_transitions: transitions.filter((_, i) => i !== index) })}>Remove</button></div>
      <p className="text-xs text-ink-muted">{task.stages.find((s) => s.id === event.to_stage_id)?.description}</p>
      {issueNotes("transition", index)}
      {eventTime(`${name} time`, event.time_s, (time_s) => edit({ time_s }))}
      {candidate && <button className={buttonClass} disabled={structuralDisabled || candidate.actionTimeS === null} onClick={() => sharedTime(candidate, candidate.actionTimeS, true)}>Link to action at {(candidate.actionTimeS?.toFixed(2) ?? "unset")} s</button>}
      {details(name, event, edit, <div className="grid gap-3">{stageSelect(name, event, "from", edit)}{stageSelect(name, event, "to", edit)}</div>)}
    </article> });
  });
  failures?.forEach((event, index) => {
    const name = `Failure ${index + 1}`;
    const edit = (patch: StageLabelRow) => { const next = patchFailureEvent(row, index, patch); change(next, retainRefs(next)); };
    entries.push({ key: `failure:${index}`, time: timeValue(event.time_s), node: <article aria-label={name} className="rounded-xl border border-warm-200 bg-white p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-sm font-medium">Failure event {index + 1}</h4>
        <label className="flex items-center gap-2 text-xs text-teal"><input type="radio" name="primary-failure-event" checked={primaryIndex === index} disabled={structuralDisabled} onChange={() => change(selectPrimaryFailure(row, index), links, true)} />Primary failure</label>
      </div>
      <label className="flex flex-col gap-1 text-xs">Failure mode<select className={inputClass} aria-label={`${name} mode`} disabled={disabled} value={typeof event.failure_mode_id === "string" ? event.failure_mode_id : ""} onChange={(e) => edit({ failure_mode_id: e.target.value })}>
        <option value="" disabled>Choose a failure</option>{task.failureModes.filter((f) => f.id !== task.successDefinition.noFailureModeId).map((f) => <option key={f.id} value={f.id}>{readable(f.id)}</option>)}
      </select></label>
      {issueNotes("failure", index)}
      {eventTime(`${name} time`, event.time_s, (time_s) => edit({ time_s }))}
      {details(name, event, edit)}
      <button className={buttonClass} disabled={structuralDisabled} aria-label={`Remove ${name}`} onClick={() => structuralChange(removeFailureEvent(row, index))}>Remove</button>
    </article> });
  });
  if (order === "chronological") entries.sort((a, b) => (a.time ?? Infinity) - (b.time ?? Infinity));
  const occurrenceArrays = [transitions, failures, ...(actions?.map((a) => records(a.occurrences) ? a.occurrences : null) ?? [])];
  const needsSort = occurrenceArrays.some((events) => events?.some((event, i) => i > 0 && timeValue(event.time_s) !== null && timeValue(events[i - 1].time_s) !== null && Number(event.time_s) < Number(events[i - 1].time_s)));
  const canSort = occurrenceArrays.every((events) => events !== null && events.every((event) => timeValue(event.time_s) !== null));
  const sortEvents = (events: StageLabelRow[]) => [...events].sort((a, b) => Number(a.time_s) - Number(b.time_s));
  const newOccurrence = () => ({ attempt_index: 1, time_s: null, confidence: "medium", evidence: "" });
  return <section aria-label="Episode events" className="space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-base font-medium">Episode events</h3>
      <select className={inputClass} aria-label="Event display order" disabled={structuralDisabled} value={order} onChange={(e) => setOrder(e.target.value as typeof order)}><option value="chronological">Time order</option><option value="recorded">Recorded order</option></select>
    </div>
    <p className="text-xs text-ink-muted">Review each physical event once. Shared action and stage times update together. Event details hold attempts and retained source evidence.</p>
    {undo?.after === signature(row, links) && <button className={buttonClass} disabled={structuralDisabled} onClick={() => { props.onEdit(undo.before); props.onEventLinksChange?.(undo.links); setUndo(null); }}>Undo last event edit</button>}
    {validLinks.length !== links.length && <div role="alert" className="rounded-lg bg-gold-light p-3 text-xs space-y-2"><p>Some saved event associations no longer identify a unique pair. Review these events separately before confirming.</p><button className={buttonClass} disabled={structuralDisabled} onClick={() => change(row, validLinks, true)}>Clear unmatched event associations</button></div>}
    <div className="rounded-lg border border-warm-200 bg-warm-50 p-3 text-sm space-y-2" aria-label="Primary failure summary">
      <p>Primary failure: {noFailure ? "None" : primaryIndex !== null && failures ? `${task.failureModes.some((f) => f.id === failures[primaryIndex].failure_mode_id) ? readable(String(failures[primaryIndex].failure_mode_id)) : "Unset or invalid failure"} at ${timeValue(failures[primaryIndex].time_s)?.toFixed(2) ?? "unset"} s` : "Choose a failure event below"}</p>
      {!noFailure && primaryIndex === null && !!row.failure_mode && <p className="text-xs text-ink-muted">Earlier summary: {task.failureModes.some((f) => f.id === row.failure_mode) ? readable(String(row.failure_mode)) : "unset or invalid"}, {timeValue(row.primary_failure_time_s)?.toFixed(2) ?? "unset"} s. Select the matching event to resolve it.</p>}
      <label className="flex items-center gap-2 text-xs"><input type="radio" name="primary-failure-event" checked={noFailure} disabled={structuralDisabled} onChange={() => change({ ...row, failure_mode: task.successDefinition.noFailureModeId, primary_failure_time_s: null }, links, true)} />No primary failure, successful episode</label>
    </div>
    {needsSort && <div role="alert" className="rounded-lg bg-gold-light p-3 text-xs space-y-2">
      <p>The corrected times changed the recorded event order. Update that order before confirming; no timestamps will change.</p>
      <button className={buttonClass} disabled={structuralDisabled || !canSort} onClick={() => change({ ...row,
        stage_transitions: sortEvents(transitions!), failure_events: sortEvents(failures!),
        key_action_observations: actions!.map((action) => ({ ...action, occurrences: sortEvents(action.occurrences as StageLabelRow[]) })),
      }, links, true)}>Update recorded event order</button>
      {!canSort && <p>Finish every event time before updating the order.</p>}
    </div>}
    {entries.map((entry) => <div key={entry.key}>{entry.node}</div>)}
    {(!actions || !transitions || !failures) && <p role="alert" className="text-sm text-coral">An event list has an invalid structure. Its contents are preserved; inspect the source before replacing it.</p>}
    <div className="flex flex-wrap gap-2">
      {transitions && <button className={buttonClass} disabled={structuralDisabled} onClick={() => structuralChange({ ...row, stage_transitions: [...transitions, { ...newOccurrence(), from_stage_id: task.stages[0].id, from_stage_index: task.stages[0].index, to_stage_id: task.stages[0].id, to_stage_index: task.stages[0].index }] })}>Add transition</button>}
      {failures && <button className={buttonClass} disabled={structuralDisabled} onClick={() => {
        const event = { ...newOccurrence(), failure_mode_id: task.failureModes.find((f) => f.id !== task.successDefinition.noFailureModeId)!.id };
        let next: StageLabelRow = { ...row, failure_events: [...failures, event] };
        if (!failures.length) next = selectPrimaryFailure(next, 0);
        structuralChange(next);
      }}>Add failure event</button>}
    </div>
    <details className="rounded-lg border border-warm-200 p-3 space-y-3"><summary className="cursor-pointer text-sm text-teal">Add an action or another occurrence</summary>
      {actions?.map((action, index) => <div key={index} className="pt-2">
        {!records(action.occurrences) || action.occurrences.length === 0 ? <TrajectoryActionEditor {...props} action={action} index={index} definition={task.keyActions.find((a) => a.id === action.action_id)} onChange={(next) => structuralChange({ ...row, key_action_observations: actions.map((a, i) => i === index ? next : a) })} />
          : <button className={buttonClass} disabled={structuralDisabled} onClick={() => structuralChange({ ...row, key_action_observations: actions.map((a, i) => i === index ? setActionOccurrences(a, [...action.occurrences as StageLabelRow[], newOccurrence()]) : a) })}>Add another {task.keyActions.find((a) => a.id === action.action_id)?.name ?? `action ${index + 1}`} occurrence</button>}
      </div>)}
    </details>
  </section>;
}
