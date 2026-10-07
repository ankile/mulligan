import type { ExportedStageSpec, Violation } from "../../convex/stageConsistency";

const redacted = "An annotation field needs review. Unblind to inspect raw details.";

// These are the exact static messages emitted by validateTrajectoryLabel.
// Never pass an arbitrary message through merely because its code is familiar.
const contractMessages: Record<string, string> = {
  "Response is not a JSON object": "The annotation must be an object.",
  "schema_version must be trajectory-label/v1": "The annotation must use the supported trajectory schema.",
  "response is missing or mistypes one or more required trajectory-label/v1 fields": "One or more required annotation fields are missing or have the wrong type.",
  "attempt_count must be an integer greater than or equal to 1": "Attempt count must be a whole number of at least 1.",
  "max_stage.stage_id is undeclared": "Choose a maximum stage from this task's stage ladder.",
  "max_stage ID and index do not match the task definition": "The maximum stage name and number must match.",
  "max_stage does not equal the greatest recorded transition": "Maximum stage must match the highest destination in the transition history.",
  "key_action_observations must cover every declared action in order": "Keep exactly one row for each declared key action, in task order.",
  "primary_failure uses an undeclared failure mode": "Choose a primary failure from this task's declared failure modes.",
  "primary_failure time must be null for the no-failure ID": "Clear the primary failure time when there is no failure.",
  "primary_failure no-failure ID requires verified success semantics": "No primary failure requires the task's success conditions to be satisfied.",
  "primary_failure must match a declared failure event mode and onset time": "Primary failure must match a failure event's mode and onset time.",
  "final_state_id is undeclared": "Choose a final state from this task's declared states.",
  "task_success must be a boolean": "Choose Yes or No for task success.",
  "task_success conflicts with the task success definition": "Task success must agree with the required stage, final state, actions, and primary failure.",
  "a successful trajectory must use the no-failure ID": "A successful trajectory must have no primary failure.",
  "an unsuccessful trajectory cannot use the no-failure ID": "An unsuccessful trajectory must name its primary failure.",
};

/** Only fixed contract paths and numeric positions become visible labels. */
function fieldName(path: string): string | null {
  if (path.length > 200) return null;
  const names: Record<string, string> = {
    label: "Annotation",
    trajectory: "Annotation",
    trajectory_identity: "Annotation identity",
    stage_transitions: "Stage transitions",
    key_action_observations: "Key actions",
    failure_events: "Failure events",
    primary_failure: "Primary failure",
    "primary_failure.time_s": "Primary failure time",
    primary_failure_time_s: "Primary failure time",
    max_stage: "Maximum stage",
    review_reasons: "Review reasons",
  };
  const clean = path.startsWith("label.") ? path.slice(6) : path;
  if (Object.hasOwn(names, clean)) return names[clean];
  const match = /^(stage_transitions|failure_events|key_action_observations)\[(\d+)\](?:\.occurrences(?:\[(\d+)\])?)?(?:\.(time_s|first_time_s|attempt_index))?$/.exec(clean);
  if (!match) return null;
  if (clean.includes(".occurrences") && match[1] !== "key_action_observations") return null;
  const index = Number(match[2]);
  const occurrence = match[3] === undefined ? null : Number(match[3]);
  if (!Number.isSafeInteger(index) || index >= Number.MAX_SAFE_INTEGER ||
      (occurrence !== null && (!Number.isSafeInteger(occurrence) || occurrence >= Number.MAX_SAFE_INTEGER))) return null;
  let name = `${match[1] === "stage_transitions" ? "Transition" : match[1] === "failure_events" ? "Failure" : "Action"} ${index + 1}`;
  if (occurrence !== null) name += ` occurrence ${occurrence + 1}`;
  else if (clean.includes(".occurrences")) name += " occurrences";
  if (match[4]) name += { time_s: " time", first_time_s: " first time", attempt_index: " attempt" }[match[4]];
  return name;
}

const pathRules: Array<[RegExp, string]> = [
  [/^(\S+) must be a finite number greater than or equal to zero$/, "must be a finite time of at least zero."],
  [/^(\S+) must be an integer$/, "must be a whole number."],
  [/^(\S+) must be an integer greater than or equal to 1$/, "must be a whole number of at least 1."],
  [/^(\S+) must be between 1 and attempt_count \(\d+(?:e\+\d+)?\)$/, "must be between 1 and the declared attempt count."],
  [/^(\S+) must be chronologically nondecreasing by time_s$/, "must be in chronological order."],
  [/^(\S+) is not an object$/, "has an invalid structure."],
  [/^(\S+) has mismatched stage IDs\/indices$/, "must use matching stage names and numbers."],
  [/^(\S+) is not forward$/, "must move to a higher stage."],
  [/^(\S+) absent-action fields conflict$/, "is marked No but still has a first time or occurrences. Review those fields together."],
  [/^(\S+) first_time_s is inconsistent$/, "first time must match its earliest occurrence. An action marked Yes needs an occurrence."],
  [/^(\S+) uses an undeclared failure mode$/, "must use a declared failure mode."],
  [/^(\S+) cannot record the no-failure ID$/, "must name an actual failure; use the summary for no primary failure."],
];

/** Safe explanation for policy-blind review. Does not change validation or labels.
 * Callers must not prepend unfiltered codes, fields, or raw values while blind.
 */
export function trajectoryReviewMessage(
  violation: Violation,
  spec: ExportedStageSpec,
  blind: boolean,
): string {
  if (!blind) return violation.message;
  if (!spec.trajectory) return redacted;
  const { code, message } = violation;
  if (code === "trajectory_duration" && message === "A trajectory review requires a positive episode duration") {
    return "A verified episode duration is required before confirming this annotation.";
  }
  if (code === "trajectory_identity" && message === "trajectory_identity must contain exactly schema_version, task_id, taxonomy_version, and sample_id") {
    return "The annotation identity has missing or unrecognized fields.";
  }
  if (code === "trajectory_shape") {
    // Shape messages enumerate untrusted extra keys. Use only a known field
    // path, never the message or an arbitrary field supplied alongside it.
    const field = violation.fields.length === 1 ? fieldName(violation.fields[0]) : null;
    return field ? `${field} has an invalid structure. Unblind to inspect raw details.` : redacted;
  }
  if (code === "trajectory_time_bounds") {
    const match = /^(\S+) must be within \[0, (\d+\.\d{6})\] seconds \(up to 0\.010s positive endpoint representation rounding is allowed\)$/.exec(message);
    const field = match ? fieldName(match[1]) : null;
    return field ? `${field} must be within the episode duration (${match![2]} seconds).` : redacted;
  }
  if (code !== "trajectory_contract") return redacted;
  if (Object.hasOwn(contractMessages, message)) return contractMessages[message];
  const task = spec.trajectory.task_definition;
  if (message === `task_id must be ${task.taskId}` || message === `taxonomy_version must be ${task.taxonomyVersion}`) {
    return "The annotation identity must match the selected task and schema.";
  }
  for (const [pattern, explanation] of pathRules) {
    const match = pattern.exec(message);
    const field = match ? fieldName(match[1]) : null;
    if (field) return `${field} ${explanation}`;
  }
  for (const [index, action] of task.keyActions.entries()) {
    for (const link of action.stageLinks) {
      if (link.relation !== "required_for_entry") continue;
      const stage = task.stages.find((stage) => stage.id === link.stageId);
      if (stage && message === `key action ${action.id} is required for reached stage ${link.stageId}`) {
        return `Action ${index + 1} must be observed for the recorded stage S${stage.index}.`;
      }
    }
  }
  return redacted;
}
