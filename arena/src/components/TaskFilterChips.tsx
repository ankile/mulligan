import { useQuery } from "../lib/arenaClient";
import { api } from "../release/api";
import type { EntityStatus } from "../../convex/statusShared";
import { orderTaskChips } from "../lib/taskChips";

const CHIP_ACTIVE = "bg-teal text-white shadow-sm";
const CHIP_IDLE =
  "bg-white border border-warm-200 text-ink-muted hover:border-warm-300 hover:text-ink";
// Non-mainline lines are visually demoted so the active lines read first.
const CHIP_IDLE_DEMOTED =
  "bg-warm-50 border border-dashed border-warm-200 text-ink-muted/70 hover:border-warm-300 hover:text-ink";

/**
 * Task filter strip shared by Eval Sessions and Data Explorer. Chips wrap
 * onto multiple rows, are ordered mainline → ablation → testing → retired
 * (alphabetical within), and non-mainline groups are separated by a thin
 * divider and drawn demoted. In the Mainline lens only mainline tasks are
 * present, so the strip degenerates to the plain alphabetical row.
 */
export function TaskFilterChips({
  tasks,
  value,
  onChange,
}: {
  tasks: Iterable<string>;
  value: string;
  onChange: (task: string) => void;
}) {
  const taskRows = useQuery(api.statuses.listTaskStatuses);
  if (taskRows === undefined) return null;
  const statusByTask = new Map<string, EntityStatus>(
    taskRows.map((r) => [r.task, r.status]),
  );
  const chips = orderTaskChips(tasks, statusByTask);
  if (chips.length <= 1) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 min-w-0">
      <span className="text-[11px] uppercase tracking-widest text-ink-muted font-medium mr-1">
        Task
      </span>
      <button
        onClick={() => onChange("all")}
        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
          value === "all" ? CHIP_ACTIVE : CHIP_IDLE
        }`}
      >
        All
      </button>
      {chips.map((chip, i) => {
        const isActive = value === chip.task;
        const demoted = chip.status !== "mainline";
        const groupStart = i > 0 && chips[i - 1].status !== chip.status;
        return (
          <span key={chip.task} className="contents">
            {groupStart && (
              <span
                aria-hidden
                className="self-stretch w-px bg-warm-200 mx-1"
                title={`${chip.status} lines`}
              />
            )}
            <button
              onClick={() => onChange(chip.task)}
              title={demoted ? `${chip.status} line` : undefined}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                isActive ? CHIP_ACTIVE : demoted ? CHIP_IDLE_DEMOTED : CHIP_IDLE
              }`}
            >
              {chip.task}
            </button>
          </span>
        );
      })}
    </div>
  );
}
