import { STATUS_VALUES, type EntityStatus } from "../../convex/statusShared";

const BADGE_STYLES: Record<string, string> = {
  retired: "bg-warm-100 text-ink-muted",
  ablation: "bg-purple-100 text-purple-700",
  testing: "bg-gold-light text-gold",
};

/** Lifecycle status pill; renders nothing for mainline. */
export function StatusBadge({
  status,
  reason,
}: {
  status: string;
  reason?: string | null;
}) {
  if (status === "mainline") return null;
  return (
    <span
      title={reason ?? undefined}
      className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-medium ${BADGE_STYLES[status] ?? "bg-warm-100 text-ink-muted"}`}
    >
      {status}
    </span>
  );
}

/** Editor control for a per-entity status override ("inherit" clears it). */
export function StatusSelect({
  value,
  onChange,
}: {
  value: EntityStatus | "inherit";
  onChange: (v: EntityStatus | "inherit") => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as EntityStatus | "inherit")}
      onClick={(e) => e.stopPropagation()}
      className="px-2 py-1 rounded-lg border border-warm-200 bg-white text-xs text-ink font-body cursor-pointer hover:border-warm-300 transition-colors"
    >
      <option value="inherit">inherit (task default)</option>
      {STATUS_VALUES.map((s) => (
        <option key={s} value={s}>
          {s}
        </option>
      ))}
    </select>
  );
}
