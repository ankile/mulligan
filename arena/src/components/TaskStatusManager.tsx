import { useState } from "react";
import { useQuery, useMutation } from "../lib/arenaClient";
import { api } from "../release/api";
import { STATUS_VALUES, type EntityStatus } from "../../convex/statusShared";

type TaskRow = {
  task: string;
  status: EntityStatus;
  reason?: string;
  superseded_by?: string;
  updated_at?: number;
  updated_by?: string;
};

function ManagerRow({
  row,
  onSave,
}: {
  row: TaskRow;
  onSave: (fields: {
    status: EntityStatus;
    reason?: string;
    superseded_by?: string;
  }) => void;
}) {
  const [reason, setReason] = useState(row.reason ?? "");
  const [supersededBy, setSupersededBy] = useState(row.superseded_by ?? "");

  const save = (status: EntityStatus) =>
    onSave({
      status,
      reason: reason.trim() || undefined,
      superseded_by: supersededBy.trim() || undefined,
    });

  return (
    <div className="grid grid-cols-[1fr_110px_1fr_180px] items-center gap-3 px-4 py-2 border-b border-warm-100 last:border-b-0">
      <span className="font-mono text-xs text-ink truncate" title={row.task}>
        {row.task}
      </span>
      <select
        value={row.status}
        onChange={(e) => save(e.target.value as EntityStatus)}
        className="px-2 py-1 rounded-lg border border-warm-200 bg-white text-xs text-ink font-body cursor-pointer hover:border-warm-300 transition-colors"
      >
        {STATUS_VALUES.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      <input
        type="text"
        value={reason}
        placeholder="reason"
        onChange={(e) => setReason(e.target.value)}
        onBlur={() => {
          if ((reason.trim() || undefined) !== row.reason) save(row.status);
        }}
        className="px-2 py-1 rounded-lg border border-warm-200 bg-white text-xs text-ink font-body"
      />
      <input
        type="text"
        value={supersededBy}
        placeholder="superseded by"
        onChange={(e) => setSupersededBy(e.target.value)}
        onBlur={() => {
          if ((supersededBy.trim() || undefined) !== row.superseded_by)
            save(row.status);
        }}
        className="px-2 py-1 rounded-lg border border-warm-200 bg-white text-xs text-ink font-mono"
      />
    </div>
  );
}

/** Editor panel: one status row per task (tasks without a row are mainline). */
export default function TaskStatusManager() {
  const taskRows = useQuery(api.statuses.listTaskStatuses);
  const envs = useQuery(api.policies.environmentsDetailed);
  const setTaskStatus = useMutation(api.statuses.setTaskStatus);

  if (taskRows === undefined || envs === undefined) return null;

  const known = new Set(taskRows.map((r) => r.task));
  const rows: TaskRow[] = [
    ...taskRows,
    ...envs
      .filter((e) => !known.has(e.environment))
      .map((e) => ({ task: e.environment, status: "mainline" as EntityStatus })),
  ].sort((a, b) => a.task.localeCompare(b.task));

  return (
    <div className="bg-white rounded-2xl border border-warm-200 shadow-sm mb-8 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-warm-100 bg-warm-50 grid grid-cols-[1fr_110px_1fr_180px] gap-3">
        {["Task", "Status", "Reason", "Superseded by"].map((h) => (
          <span
            key={h}
            className="text-[11px] uppercase tracking-widest text-ink-muted font-medium"
          >
            {h}
          </span>
        ))}
      </div>
      {rows.map((row) => (
        <ManagerRow
          key={`${row.task}:${row.updated_at ?? 0}`}
          row={row}
          onSave={(fields) => setTaskStatus({ task: row.task, ...fields })}
        />
      ))}
    </div>
  );
}
