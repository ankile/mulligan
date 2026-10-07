import type { LabelEvent } from "../../lib/hf-api";
import { clamp } from "./format";
import { describeLabelPayload, eventSeekFrame, sourceLabel } from "./labelHistory";

// Per-episode provenance chain with click-to-seek on events that carry a
// frame.
// `onSeek` must pause playback first (the parents pass jumpToFrame).

export function LabelHistoryPanel({
  chain,
  rawLength,
  onSeek,
}: {
  chain: LabelEvent[];
  rawLength: number;
  onSeek: (frame: number) => void;
}) {
  if (chain.length === 0) return null;
  return (
    <div className="mt-3 rounded-lg border border-warm-200 bg-warm-50 px-3 py-2">
      <div className="text-[10px] font-mono uppercase tracking-wide text-ink-muted mb-1">
        Label history · {chain.length} event{chain.length === 1 ? "" : "s"}
      </div>
      <div className="flex flex-col gap-0.5">
        {chain.map((event, i) => {
          const seek = eventSeekFrame(event);
          const current = i === chain.length - 1;
          return (
            <button
              key={i}
              disabled={seek === null}
              // onSeek (jumpToFrame, not raw setFrame): it pauses first — a
              // seek during playback would be silently reverted by the next
              // stopAndSnap.
              onClick={() => seek !== null && onSeek(clamp(seek, 0, rawLength - 1))}
              title={
                seek === null
                  ? undefined
                  : seek > rawLength - 1
                    ? `Seek to frame ${rawLength - 1} (event frame ${seek} is beyond this timeline)`
                    : `Seek to frame ${seek}`
              }
              className={`flex items-baseline gap-2 text-left text-[11px] font-mono rounded px-1 -mx-1 ${
                seek !== null ? "cursor-pointer hover:bg-warm-100" : "cursor-default"
              } ${current ? "text-ink" : "text-ink-muted"}`}
            >
              <span className="shrink-0 text-ink-muted/70">
                {event.ts.slice(0, 16).replace("T", " ")}
              </span>
              <span className="shrink-0">{sourceLabel(event)}</span>
              <span className={current ? "font-medium" : ""}>
                {describeLabelPayload(event.label_kind, event.payload)}
                {current ? "  ← current" : ""}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
