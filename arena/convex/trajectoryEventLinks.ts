import { v } from "convex/values";
import type { StageLabelRow } from "./stageConsistency";

/** Human review associations, separate from immutable prediction/schema fields.
 * Unique semantic refs survive list reordering. Repeated or removed events make
 * a ref stale; consumers must never guess which occurrence the reviewer meant.
 */
export interface TrajectoryEventLink {
  action_id: string;
  stage_id: string;
  attempt_index: number;
  relation: "shared" | "distinct";
}

export const eventLinksValidator = v.array(v.object({
  action_id: v.string(), stage_id: v.string(), attempt_index: v.number(),
  relation: v.union(v.literal("shared"), v.literal("distinct")),
}));

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> =>
    item !== null && typeof item === "object" && !Array.isArray(item)) : [];
}

/** Drafts may retain stale associations for recovery, but cannot confirm them. */
export function validateTrajectoryEventLinks(row: StageLabelRow, links: readonly TrajectoryEventLink[]): string[] {
  const errors: string[] = [];
  const seen = new Set<string>();
  const actions = records(row.key_action_observations);
  const transitions = records(row.stage_transitions);
  for (const link of links) {
    const id = `${link.action_id}:${link.stage_id}:${link.attempt_index}`;
    if (seen.has(id)) errors.push(`Duplicate event association ${id}.`);
    seen.add(id);
    const action = actions.filter((item) => item.action_id === link.action_id);
    const stage = transitions.filter((item) => item.to_stage_id === link.stage_id);
    const occurrences = action.length === 1 ? records(action[0].occurrences) : [];
    if (!Number.isSafeInteger(link.attempt_index) || link.attempt_index < 1 ||
        action.length !== 1 || action[0].occurred !== true || occurrences.length !== 1 || stage.length !== 1 ||
        occurrences[0].attempt_index !== link.attempt_index || stage[0].attempt_index !== link.attempt_index) {
      errors.push(`Event association ${id} no longer identifies a unique action and stage in the same attempt. Review or remove the association.`);
      continue;
    }
    if (link.relation === "shared" && (typeof occurrences[0].time_s !== "number" ||
        !Number.isFinite(occurrences[0].time_s) || occurrences[0].time_s !== stage[0].time_s)) {
      errors.push(`Shared event ${id} must have one timestamp for its action and stage.`);
    }
  }
  return errors;
}
