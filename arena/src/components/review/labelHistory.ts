import type { LabelEvent } from "../../lib/hf-api";

// Label-history (provenance ledger) formatting helpers — taxonomy-blind:
// unknown label kinds render as truncated JSON, so new kinds flow through
// without schema changes; known kinds get a friendly one-liner. (Split from
// LabelHistoryPanel.tsx so that file only exports a component — react-refresh.)

const SOURCE_TOOL_SHORT: Record<string, string> = {
  "web-review": "web",
  "cv2-editor": "cv2",
  "collection-results": "collection",
};

export function sourceLabel(event: LabelEvent): string {
  const tool = SOURCE_TOOL_SHORT[event.source.tool] ?? event.source.tool;
  const agent = event.source.agent ? ` ${event.source.agent}` : "";
  return `${event.source.kind}·${tool}${agent}`;
}

export function describeLabelPayload(
  kind: string,
  payload: Record<string, unknown>
): string {
  if (payload.action === "skip") return "skip (kept labels)";
  if (payload.action === "unlabel") return "unlabeled (decision retracted)";
  if (kind === "outcome" && typeof payload.new_outcome === "string") {
    let text = `${payload.new_outcome} @ ${payload.outcome_frame}`;
    if (payload.soft_truncate) text += " ·soft-trunc";
    const marks = payload.subtask_frames;
    if (Array.isArray(marks) && marks.length > 0) text += ` ·${marks.length} marks`;
    return text;
  }
  if (kind === "stage") {
    // Stage events carry a normalized `summary` beside the (task-keyed,
    // taxonomy-versioned) full row, so this stays taxonomy-blind.
    const summary = payload.summary as Record<string, unknown> | undefined;
    if (summary && summary.stage != null) {
      let text = `S${summary.stage}`;
      if (typeof summary.failure_mode === "string" && summary.failure_mode !== "none") {
        text += ` ${summary.failure_mode}`;
      }
      if (typeof summary.final_state === "string") text += ` → ${summary.final_state}`;
      return text;
    }
  }
  return `${kind}: ${JSON.stringify(payload).slice(0, 60)}`;
}

export function eventSeekFrame(event: LabelEvent): number | null {
  const frame = event.payload.outcome_frame;
  return typeof frame === "number" ? frame : null;
}
