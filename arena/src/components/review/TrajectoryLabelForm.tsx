import { TrajectoryEventTimeline } from "./TrajectoryEventTimeline";
import type { StageLabelFormProps } from "./StageLabelForm";
import { trajectoryReviewMessage } from "../../lib/trajectoryReviewMessages";

const inputClass = "rounded-lg border border-warm-200 bg-white px-2 py-2 text-sm text-ink min-w-0 max-w-full";
const confidences = ["low", "medium", "high"];
function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** The trajectory contract has repeated occurrences and independent summary
 * decisions. Explicit action edits synchronize redundant occurrence summaries. */
export function TrajectoryLabelForm(props: StageLabelFormProps) {
  const { spec, row, disabled, blind = false } = props;
  const violations = props.violations.filter((violation) => violation.code !== "trajectory_timeline");
  const onEdit = props.onEdit;
  if (!spec.trajectory) throw new Error("Trajectory form requires a trajectory spec");
  const task = spec.trajectory.task_definition;
  const readable = (id: string) => id.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
  const options = (items: Array<{ id: string; name?: string; description: string }>) => items.map((item) => ({ value: item.id, text: item.name || readable(item.id), title: item.description }));
  const stageName = (stage: typeof task.stages[number]) => stage.name === `S${stage.index}` ? readable(stage.id) : stage.name;
  const selectedStage = task.stages.find((stage) => stage.id === row.max_stage_id && stage.index === row.max_stage);
  const select = (name: string, value: unknown, choices: Array<{ value: string; text: string; title?: string }>, change: (value: string) => void) => {
    const valid = typeof value === "string" && choices.some((choice) => choice.value === value);
    return <select aria-label={name} className={inputClass} disabled={disabled} value={valid ? value : "__invalid__"}
      onChange={(event) => change(event.target.value)}>
      {!valid && <option value="__invalid__" disabled>{blind ? "Unset or invalid value" : `Unset or invalid: ${String(value ?? "")}`}</option>}
      {choices.map((choice) => <option key={choice.value} value={choice.value} title={choice.title}>{choice.text}</option>)}
    </select>;
  };
  const confidence = (name: string, value: unknown, change: (value: string) => void) => select(name, value, confidences.map((value) => ({ value, text: value })), change);
  const boolean = (name: string, value: unknown, change: (value: boolean) => void) => select(name,
    typeof value === "boolean" ? String(value) : value,
    [{ value: "true", text: "Yes" }, { value: "false", text: "No" }], (value) => change(value === "true"));
  const integer = (name: string, value: unknown, change: (value: number | null) => void) => <input
    aria-label={name} className={`${inputClass} w-16`} type="number" step="1" min="1" disabled={disabled}
    value={number(value) ?? ""} onChange={(event) => change(event.target.value === "" ? null : Number(event.target.value))} />;
  const sourceText = (name: string, value: unknown) => blind ? null : <label className="block text-xs text-ink-muted">{name} · retained source text, not human notes
    <textarea aria-label={name} readOnly className={`${inputClass} block w-full bg-warm-50`} rows={2}
      value={typeof value === "string" ? value : ""} />
  </label>;

  return <div className="space-y-5" data-testid="trajectory-form">
    <div>
      <h3 className="text-base font-medium text-ink">Episode label</h3>
      <p className="mt-1 text-xs text-ink-muted">Check the summary and event times against the video. Confirmation covers these structured judgments; retained source text and confidence are excluded.</p>
    </div>
    {violations.length > 0 && <div role="alert" className="rounded-lg border border-coral/30 bg-coral-light p-3 space-y-1">
      <p className="text-sm font-medium text-coral">Resolve before confirming</p>
      {violations.map((violation, index) => <p key={index} className="text-xs text-coral">{trajectoryReviewMessage(violation, spec, blind)}</p>)}
    </div>}
    <fieldset>
      <legend className="mb-2 text-sm font-medium">Furthest stage reached</legend>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Maximum stage">
        {task.stages.map((stage) => <button key={stage.id} disabled={disabled} aria-pressed={selectedStage?.id === stage.id}
          aria-label={`S${stage.index}: ${stageName(stage)}`} title={stage.description}
          onClick={() => onEdit({ max_stage: stage.index, max_stage_id: stage.id })}
          className={`rounded-lg px-3 py-2 text-sm font-mono cursor-pointer border ${selectedStage?.id === stage.id ? "bg-teal border-teal text-white" : "border-warm-200 text-ink-muted hover:border-teal"} disabled:opacity-40`}>S{stage.index}</button>)}
      </div>
      {selectedStage ? <p className="mt-2 text-sm text-ink-muted"><strong className="text-ink">{stageName(selectedStage)}.</strong> {selectedStage.description}</p>
        : <p role="alert" className="mt-2 text-sm text-coral">Choose a valid maximum stage.</p>}
      <p className="mt-1 text-xs text-ink-muted">Keep earlier progress even if the episode ends in failure.</p>
    </fieldset>
    <details className="text-xs"><summary className="cursor-pointer text-teal">Task definition and decision rules</summary>
      <p className="mt-2">{task.objective}</p>
      <p className="mt-2">Success requires a stage in {task.successDefinition.successfulStageIds.join(", ")};
        final state in {task.successDefinition.successfulFinalStateIds.join(", ")};
        actions {task.successDefinition.requiredKeyActionIds.join(", ") || "(none required)"};
        primary failure {task.successDefinition.noFailureModeId}.</p>
      {task.stages.map((stage) => <p key={stage.id} className="mt-2"><strong>S{stage.index}: {stage.name}</strong> — {stage.description}<br />{stage.entryCriteria.join(" ")}{stage.exclusions?.length ? ` Exclusions: ${stage.exclusions.join(" ")}` : ""}</p>)}
      {task.decisionRules.map((rule) => <p key={rule.id} className="mt-2">{rule.rule}</p>)}
    </details>
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <label className="text-xs flex flex-col gap-1">Task success{boolean("Task success", row.task_success, (task_success) => onEdit({ task_success }))}</label>
      <label className="text-xs flex flex-col gap-1">Final state{select("Final state", row.final_state, options(task.finalStates), (final_state) => onEdit({ final_state }))}</label>
      <label className="text-xs flex flex-col gap-1">Attempt count{integer("Attempt count", row.attempt_count, (attempt_count) => onEdit({ attempt_count }))}</label>
    </div>
    <TrajectoryEventTimeline {...props} />
    <label className="block text-sm font-medium">Your review notes
      <textarea aria-label="Your review notes" disabled={disabled || !props.onHumanNotesChange} className={`${inputClass} block w-full mt-2`} rows={3}
        placeholder="Optional observations or uncertainty from your review" value={props.humanNotes ?? ""}
        onChange={(event) => props.onHumanNotesChange?.(event.target.value)} />
      <span className="block mt-1 text-xs font-normal text-ink-muted">Saved separately from the prediction. Available while policy identity stays hidden.</span>
    </label>
    <details className="rounded-lg border border-warm-200 p-3 space-y-3">
      <summary className="cursor-pointer text-xs text-ink-muted">Retained source text and confidence · excluded from human review</summary>
      <p className="text-xs text-ink-muted">These fields come from the starting label and may describe events you have corrected. They are preserved for compatibility, not treated as your reviewed explanations or confidence.</p>
      <label className="text-xs flex flex-col gap-1">Source confidence{confidence("Overall confidence", row.confidence, (confidence) => onEdit({ confidence }))}</label>
      <label className="text-xs flex gap-2 items-center">Source requested further review{boolean("Needs further human review", row.needs_human_review, (needs_human_review) => onEdit({ needs_human_review }))}</label>
      {blind ? <p className="text-xs text-ink-muted">Source free text is hidden because it may reveal policy identity. Use your review notes above without unblinding.</p> : <>
        {sourceText("Source notes", row.notes)}
        <p className="text-xs text-ink-muted">Source review reasons</p>
        <pre className="text-xs whitespace-pre-wrap break-words">{JSON.stringify(row.review_reasons, null, 2)}</pre>
      </>}
    </details>
  </div>;
}
