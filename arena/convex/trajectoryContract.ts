/**
 * Generic trajectory types, parser and task validator. The label semantics are
 * pinned by tests/fixtures/trajectory-review-fixtures.json (tests/trajectoryReview.test.ts).
 */
export const TRAJECTORY_LABEL_SCHEMA_VERSION = "trajectory-label/v1";

export type TaskDefinitionItem = {
  id: string;
  description: string;
};

export type StageDefinition = TaskDefinitionItem & {
  index: number;
  name: string;
  entryCriteria: string[];
  exclusions?: string[];
};

export type KeyActionDefinition = TaskDefinitionItem & {
  name: string;
  stageRelation: string;
  stageLinks: Array<{
    stageId: string;
    relation: "supports_entry" | "required_for_entry" | "may_occur_within";
  }>;
};

export type TrajectoryTaskDefinition = {
  definitionSchemaVersion: "trajectory-task/v1";
  taskId: string;
  taxonomyVersion: string;
  displayName: string;
  objective: string;
  successCriteria: string[];
  successDefinition: {
    successfulStageIds: string[];
    successfulFinalStateIds: string[];
    requiredKeyActionIds: string[];
    noFailureModeId: string;
  };
  entities: TaskDefinitionItem[];
  evidenceViews: Array<TaskDefinitionItem & { caveats: string[] }>;
  stages: StageDefinition[];
  keyActions: KeyActionDefinition[];
  failureModes: TaskDefinitionItem[];
  finalStates: TaskDefinitionItem[];
  decisionRules: Array<{ id: string; rule: string }>;
  auditProcedure: string[];
};

export type TrajectoryLabelSummary = {
  schemaVersion: string;
  stageIndex?: number;
  stageId?: string;
  finalState?: string;
  failureMode?: string;
};

export type EvidenceConfidence = "low" | "medium" | "high";

export type TrajectoryEventOccurrence = {
  attempt_index: number;
  time_s: number;
  confidence: EvidenceConfidence;
  evidence: string;
};

export type StageTransitionObservation = TrajectoryEventOccurrence & {
  from_stage_id: string;
  from_stage_index: number;
  to_stage_id: string;
  to_stage_index: number;
};

export type KeyActionObservation = {
  action_id: string;
  occurred: boolean;
  first_time_s: number | null;
  occurrences: TrajectoryEventOccurrence[];
};

export type FailureEventObservation = TrajectoryEventOccurrence & {
  failure_mode_id: string;
};

export type CanonicalTrajectoryLabel = {
  schema_version: typeof TRAJECTORY_LABEL_SCHEMA_VERSION;
  task_id: string;
  taxonomy_version: string;
  sample_id: string;
  attempt_count: number;
  task_success: boolean;
  max_stage: { stage_id: string; stage_index: number };
  stage_transitions: StageTransitionObservation[];
  key_action_observations: KeyActionObservation[];
  failure_events: FailureEventObservation[];
  primary_failure: { failure_mode_id: string; time_s: number | null };
  final_state_id: string;
  confidence: EvidenceConfidence;
  needs_human_review: boolean;
  review_reasons: string[];
  notes: string;
};

type UnknownRecord = Record<string, unknown>;

function asRecord(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isInteger(value: unknown): value is number {
  return isFiniteNumber(value) && Number.isInteger(value);
}

function isConfidence(value: unknown): value is EvidenceConfidence {
  return value === "low" || value === "medium" || value === "high";
}

function isEventOccurrence(value: unknown): value is TrajectoryEventOccurrence {
  const event = asRecord(value);
  return Boolean(event && hasEventOccurrenceFields(event));
}

function hasEventOccurrenceFields(event: UnknownRecord): boolean {
  return (
    isInteger(event.attempt_index) &&
    isFiniteNumber(event.time_s) &&
    isConfidence(event.confidence) &&
    isString(event.evidence)
  );
}

function isStageTransition(value: unknown): value is StageTransitionObservation {
  const transition = asRecord(value);
  return Boolean(
    transition &&
      hasEventOccurrenceFields(transition) &&
      isString(transition.from_stage_id) &&
      isInteger(transition.from_stage_index) &&
      isString(transition.to_stage_id) &&
      isInteger(transition.to_stage_index),
  );
}

function isKeyActionObservation(value: unknown): value is KeyActionObservation {
  const observation = asRecord(value);
  return Boolean(
    observation &&
      isString(observation.action_id) &&
      typeof observation.occurred === "boolean" &&
      (observation.first_time_s === null || isFiniteNumber(observation.first_time_s)) &&
      Array.isArray(observation.occurrences) &&
      observation.occurrences.every(isEventOccurrence),
  );
}

function isFailureEvent(value: unknown): value is FailureEventObservation {
  const failure = asRecord(value);
  return Boolean(
    failure &&
      hasEventOccurrenceFields(failure) &&
      isString(failure.failure_mode_id),
  );
}

export function parseCanonicalTrajectoryLabel(
  value: unknown,
): CanonicalTrajectoryLabel | null {
  const root = asRecord(value);
  const maxStage = asRecord(root?.max_stage);
  const primaryFailure = asRecord(root?.primary_failure);
  if (
    root?.schema_version !== TRAJECTORY_LABEL_SCHEMA_VERSION ||
    !isString(root.task_id) ||
    !isString(root.taxonomy_version) ||
    !isString(root.sample_id) ||
    !isInteger(root.attempt_count) ||
    typeof root.task_success !== "boolean" ||
    !maxStage ||
    !isString(maxStage.stage_id) ||
    !isInteger(maxStage.stage_index) ||
    !Array.isArray(root.stage_transitions) ||
    !root.stage_transitions.every(isStageTransition) ||
    !Array.isArray(root.key_action_observations) ||
    !root.key_action_observations.every(isKeyActionObservation) ||
    !Array.isArray(root.failure_events) ||
    !root.failure_events.every(isFailureEvent) ||
    !primaryFailure ||
    !isString(primaryFailure.failure_mode_id) ||
    !(primaryFailure.time_s === null || isFiniteNumber(primaryFailure.time_s)) ||
    !isString(root.final_state_id) ||
    !isConfidence(root.confidence) ||
    typeof root.needs_human_review !== "boolean" ||
    !Array.isArray(root.review_reasons) ||
    !root.review_reasons.every(isString) ||
    !isString(root.notes)
  ) {
    return null;
  }
  return root as unknown as CanonicalTrajectoryLabel;
}

export function validateTrajectoryTaskDefinition(value: unknown): string[] {
  const errors: string[] = [];
  const root = asRecord(value);
  if (!root) return ["Task definition is not a JSON object"];
  if (root.definitionSchemaVersion !== "trajectory-task/v1") {
    errors.push("definitionSchemaVersion must be trajectory-task/v1");
  }
  for (const key of ["taskId", "taxonomyVersion", "displayName", "objective"] as const) {
    if (typeof root[key] !== "string" || !String(root[key]).trim()) {
      errors.push(key + " must be a non-empty string");
    }
  }

  const collectionIds = new Map<string, string[]>();
  for (const key of [
    "entities",
    "evidenceViews",
    "stages",
    "keyActions",
    "failureModes",
    "finalStates",
    "decisionRules",
  ] as const) {
    const collection = root[key];
    if (!Array.isArray(collection) || collection.length === 0) {
      errors.push(key + " must be a non-empty array");
      collectionIds.set(key, []);
      continue;
    }
    const ids = collection.map((item) => {
      const record = asRecord(item);
      return typeof record?.id === "string" ? record.id : "";
    });
    if (ids.some((id) => !id)) errors.push(key + " contains an item without an ID");
    if (new Set(ids).size !== ids.length) errors.push(key + " contains duplicate IDs");
    collectionIds.set(key, ids);
  }

  const stages = Array.isArray(root.stages) ? root.stages : [];
  for (const [position, item] of stages.entries()) {
    const stage = asRecord(item);
    if (stage?.index !== position) {
      errors.push("stages must have consecutive indices beginning at zero");
      break;
    }
    if (!Array.isArray(stage?.entryCriteria) || stage.entryCriteria.length === 0) {
      errors.push("stages[" + position + "] must define entryCriteria");
    }
  }

  const actionIds = new Set(collectionIds.get("keyActions") ?? []);
  const stageIds = new Set(collectionIds.get("stages") ?? []);
  const failureIds = new Set(collectionIds.get("failureModes") ?? []);
  const finalStateIds = new Set(collectionIds.get("finalStates") ?? []);
  const success = asRecord(root.successDefinition);

  const keyActions = Array.isArray(root.keyActions) ? root.keyActions : [];
  for (const [position, item] of keyActions.entries()) {
    const action = asRecord(item);
    const links = action?.stageLinks;
    if (!Array.isArray(links)) {
      errors.push("keyActions[" + position + "].stageLinks must be an array");
      continue;
    }
    for (const link of links) {
      const record = asRecord(link);
      if (!record || !stageIds.has(String(record.stageId))) {
        errors.push("keyActions[" + position + "] contains an undeclared stage link");
      }
      if (
        !["supports_entry", "required_for_entry", "may_occur_within"].includes(
          String(record?.relation),
        )
      ) {
        errors.push("keyActions[" + position + "] contains an invalid stage relation");
      }
    }
  }

  if (!success) {
    errors.push("successDefinition must be an object");
  } else {
    const referenceChecks: Array<[string, Set<string>, boolean]> = [
      ["successfulStageIds", stageIds, false],
      ["successfulFinalStateIds", finalStateIds, false],
      ["requiredKeyActionIds", actionIds, true],
    ];
    for (const [key, allowed, allowEmpty] of referenceChecks) {
      const references = success[key];
      if (!Array.isArray(references) || (!allowEmpty && references.length === 0)) {
        errors.push(
          "successDefinition." + key +
            (allowEmpty ? " must be an array" : " must be a non-empty array"),
        );
      } else if (references.some((id) => typeof id !== "string" || !allowed.has(id))) {
        errors.push("successDefinition." + key + " contains an undeclared ID");
      }
    }
    if (
      typeof success.noFailureModeId !== "string" ||
      !failureIds.has(success.noFailureModeId)
    ) {
      errors.push("successDefinition.noFailureModeId must name a declared failure mode");
    }
  }
  return errors;
}


function nonnegativeTime(value: unknown, field: string, errors: string[]): boolean {
  if (!isFiniteNumber(value) || value < 0) {
    errors.push(`${field} must be a finite number greater than or equal to zero`);
    return false;
  }
  return true;
}

function attemptIndex(value: unknown, field: string, count: number | null, errors: string[]) {
  if (!isInteger(value)) errors.push(`${field} must be an integer`);
  else if (count !== null && !(1 <= value && value <= count)) {
    errors.push(`${field} must be between 1 and attempt_count (${count})`);
  } else if (value < 1) errors.push(`${field} must be an integer greater than or equal to 1`);
}

function chronologicalTimes(values: unknown[], field: string, errors: string[]) {
  let previous: number | null = null;
  for (const value of values) {
    if (!isFiniteNumber(value) || value < 0) continue;
    if (previous !== null && value < previous) {
      errors.push(`${field} must be chronologically nondecreasing by time_s`);
      return;
    }
    previous = value;
  }
}

function timesClose(left: unknown, right: unknown): boolean {
  return isFiniteNumber(left) && isFiniteNumber(right) && Math.abs(left - right) <= 1e-6;
}

/** Semantic parity with Python validate_trajectory_label; no terminal-stage shortcuts. */
export function validateTrajectoryLabel(value: unknown, task: TrajectoryTaskDefinition): string[] {
  const root = asRecord(value);
  if (!root) return ["Response is not a JSON object"];
  const errors: string[] = [];
  if (root.schema_version !== TRAJECTORY_LABEL_SCHEMA_VERSION) {
    errors.push(`schema_version must be ${TRAJECTORY_LABEL_SCHEMA_VERSION}`);
  } else if (!parseCanonicalTrajectoryLabel(root)) {
    errors.push("response is missing or mistypes one or more required trajectory-label/v1 fields");
  }
  if (root.task_id !== task.taskId) errors.push(`task_id must be ${task.taskId}`);
  if (root.taxonomy_version !== task.taxonomyVersion) errors.push(`taxonomy_version must be ${task.taxonomyVersion}`);
  const count = isInteger(root.attempt_count) && root.attempt_count >= 1 ? root.attempt_count : null;
  if (count === null) errors.push("attempt_count must be an integer greater than or equal to 1");
  const stages = new Map(task.stages.map((stage) => [stage.id, stage.index]));
  const maxStage = asRecord(root.max_stage);
  const maxId = typeof maxStage?.stage_id === "string" ? maxStage.stage_id : "";
  const maxIndex = maxStage?.stage_index;
  if (!stages.has(maxId)) errors.push("max_stage.stage_id is undeclared");
  if (stages.get(maxId) !== maxIndex) errors.push("max_stage ID and index do not match the task definition");

  const transitions = Array.isArray(root.stage_transitions) ? root.stage_transitions : [];
  let greatest = 0;
  const transitionTimes: unknown[] = [];
  for (const [index, value] of transitions.entries()) {
    const item = asRecord(value);
    if (!item) { errors.push(`stage_transitions[${index}] is not an object`); continue; }
    const path = `stage_transitions[${index}]`;
    attemptIndex(item.attempt_index, `${path}.attempt_index`, count, errors);
    nonnegativeTime(item.time_s, `${path}.time_s`, errors);
    transitionTimes.push(item.time_s);
    const fromId = typeof item.from_stage_id === "string" ? item.from_stage_id : "";
    const toId = typeof item.to_stage_id === "string" ? item.to_stage_id : "";
    if (stages.get(fromId) !== item.from_stage_index || stages.get(toId) !== item.to_stage_index) {
      errors.push(`${path} has mismatched stage IDs/indices`);
    }
    if (!isFiniteNumber(item.from_stage_index) || !isFiniteNumber(item.to_stage_index) || item.to_stage_index <= item.from_stage_index) {
      errors.push(`${path} is not forward`);
    } else greatest = Math.max(greatest, item.to_stage_index);
  }
  chronologicalTimes(transitionTimes, "stage_transitions", errors);
  if (isFiniteNumber(maxIndex) && greatest !== maxIndex) errors.push("max_stage does not equal the greatest recorded transition");

  const actions = Array.isArray(root.key_action_observations) ? root.key_action_observations : [];
  const expectedIds = task.keyActions.map((action) => action.id);
  const observedIds = actions.map((value) => {
    const item = asRecord(value);
    return typeof item?.action_id === "string" ? item.action_id : "";
  });
  if (JSON.stringify(expectedIds) !== JSON.stringify(observedIds)) {
    errors.push("key_action_observations must cover every declared action in order");
  }
  for (const [index, value] of actions.entries()) {
    const item = asRecord(value);
    if (!item) continue;
    const path = `key_action_observations[${index}]`;
    const occurrences = Array.isArray(item.occurrences) ? item.occurrences : [];
    const times: unknown[] = [];
    for (const [occurrenceIndex, value] of occurrences.entries()) {
      const occurrence = asRecord(value);
      if (!occurrence) continue;
      const occurrencePath = `${path}.occurrences[${occurrenceIndex}]`;
      attemptIndex(occurrence.attempt_index, `${occurrencePath}.attempt_index`, count, errors);
      nonnegativeTime(occurrence.time_s, `${occurrencePath}.time_s`, errors);
      times.push(occurrence.time_s);
    }
    chronologicalTimes(times, `${path}.occurrences`, errors);
    const firstTime = item.first_time_s ?? null;
    if (firstTime !== null) nonnegativeTime(firstTime, `${path}.first_time_s`, errors);
    if (item.occurred === false) {
      if (firstTime !== null || occurrences.length > 0) errors.push(`${path} absent-action fields conflict`);
    } else if (item.occurred === true) {
      const validTimes = times.filter((time): time is number => isFiniteNumber(time) && time >= 0);
      if (validTimes.length === 0 || !timesClose(firstTime, Math.min(...validTimes))) {
        errors.push(`${path} first_time_s is inconsistent`);
      }
    }
  }
  const reached = new Set(transitions.map((value) => asRecord(value)?.to_stage_id).filter((id): id is string => typeof id === "string"));
  if (maxId) reached.add(maxId);
  for (const [index, action] of task.keyActions.entries()) {
    const required = action.stageLinks.find((link) => link.relation === "required_for_entry" && reached.has(link.stageId));
    if (required && asRecord(actions[index])?.occurred !== true) {
      errors.push(`key action ${action.id} is required for reached stage ${required.stageId}`);
    }
  }

  const noFailure = task.successDefinition.noFailureModeId;
  const failureIds = new Set(task.failureModes.map((mode) => mode.id));
  const failures = Array.isArray(root.failure_events) ? root.failure_events : [];
  const failureTimes: unknown[] = [];
  for (const [index, value] of failures.entries()) {
    const item = asRecord(value);
    const path = `failure_events[${index}]`;
    if (!item) { errors.push(`${path} uses an undeclared failure mode`); continue; }
    attemptIndex(item.attempt_index, `${path}.attempt_index`, count, errors);
    nonnegativeTime(item.time_s, `${path}.time_s`, errors);
    failureTimes.push(item.time_s);
    if (!failureIds.has(String(item.failure_mode_id))) errors.push(`${path} uses an undeclared failure mode`);
    else if (item.failure_mode_id === noFailure) errors.push(`${path} cannot record the no-failure ID`);
  }
  chronologicalTimes(failureTimes, "failure_events", errors);
  const primary = asRecord(root.primary_failure);
  if (!primary || !failureIds.has(String(primary.failure_mode_id))) {
    errors.push("primary_failure uses an undeclared failure mode");
  } else if (primary.failure_mode_id === noFailure) {
    if (primary.time_s != null) {
      nonnegativeTime(primary.time_s, "primary_failure.time_s", errors);
      errors.push("primary_failure time must be null for the no-failure ID");
    }
  } else nonnegativeTime(primary.time_s, "primary_failure.time_s", errors);
  const finalIds = new Set(task.finalStates.map((state) => state.id));
  if (!finalIds.has(String(root.final_state_id))) errors.push("final_state_id is undeclared");
  const occurredIds = new Set(actions.filter((value) => asRecord(value)?.occurred === true).map((value) => String(asRecord(value)?.action_id)));
  const success = task.successDefinition;
  const verifiedSuccess = success.successfulStageIds.includes(maxId) &&
    success.successfulFinalStateIds.includes(String(root.final_state_id)) &&
    success.requiredKeyActionIds.every((id) => occurredIds.has(id));
  const primaryId = primary?.failure_mode_id;
  const primaryNoFailure = primaryId === noFailure;
  if (failureIds.has(String(primaryId))) {
    if (primaryNoFailure) {
      if (!verifiedSuccess) errors.push("primary_failure no-failure ID requires verified success semantics");
    } else if (!failures.some((value) => {
      const event = asRecord(value);
      return event?.failure_mode_id === primaryId && timesClose(event?.time_s, primary?.time_s);
    })) errors.push("primary_failure must match a declared failure event mode and onset time");
  }
  if (typeof root.task_success !== "boolean") errors.push("task_success must be a boolean");
  else {
    if (root.task_success !== (verifiedSuccess && primaryNoFailure)) errors.push("task_success conflicts with the task success definition");
    if (root.task_success === true && !primaryNoFailure) errors.push("a successful trajectory must use the no-failure ID");
    else if (root.task_success === false && primaryNoFailure) errors.push("an unsuccessful trajectory cannot use the no-failure ID");
  }
  return errors;
}
