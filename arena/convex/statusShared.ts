import { v } from "convex/values";

/**
 * Entity lifecycle status, shared by schema, Convex functions, and the
 * frontend (src imports this module directly).
 *
 * - mainline: the default view — active lines and their policies/sessions.
 * - retired: superseded lines/policies kept for provenance (see
 *   taskStatuses.superseded_by).
 * - ablation: variation studies that should not clutter the default view.
 * - testing: harness shakeouts / smoke runs.
 *
 * Resolution: per-entity override ?? task-level status ?? "mainline".
 * Unknown tasks default to mainline so a new line is never silently hidden.
 */
export const STATUS_VALUES = ["mainline", "retired", "ablation", "testing"] as const;
export type EntityStatus = (typeof STATUS_VALUES)[number];

export const statusValidator = v.union(
  v.literal("mainline"),
  v.literal("retired"),
  v.literal("ablation"),
  v.literal("testing")
);

/** "inherit" clears a per-entity override so the task-level status applies. */
export const statusOrInheritValidator = v.union(
  statusValidator,
  v.literal("inherit")
);

export function effectiveStatus(
  override: EntityStatus | undefined,
  task: string | null | undefined,
  taskStatuses: Map<string, { status: string }>
): EntityStatus {
  if (override) return override;
  if (task) {
    const row = taskStatuses.get(task);
    if (row) return row.status as EntityStatus;
  }
  return "mainline";
}
