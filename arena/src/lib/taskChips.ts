import type { EntityStatus } from "../../convex/statusShared";

/**
 * Ordering for task filter chips. Mainline (active) lines come first, then
 * ablation, testing, and finally retired lines; alphabetical within a group.
 * Unknown tasks resolve to mainline (same rule as `effectiveStatus`).
 */
export const TASK_STATUS_ORDER: readonly EntityStatus[] = [
  "mainline",
  "ablation",
  "testing",
  "retired",
];

export type TaskChip = { task: string; status: EntityStatus };

export function taskStatusRank(status: EntityStatus): number {
  const rank = TASK_STATUS_ORDER.indexOf(status);
  if (rank < 0) throw new Error(`unknown task status: ${status}`);
  return rank;
}

export function orderTaskChips(
  tasks: Iterable<string>,
  taskStatuses: ReadonlyMap<string, EntityStatus>,
): TaskChip[] {
  const unique = [...new Set(tasks)];
  const chips = unique.map((task) => ({
    task,
    status: taskStatuses.get(task) ?? ("mainline" as EntityStatus),
  }));
  return chips.sort(
    (a, b) =>
      taskStatusRank(a.status) - taskStatusRank(b.status) ||
      a.task.localeCompare(b.task),
  );
}
