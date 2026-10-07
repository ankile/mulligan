import { useRef, useState } from "react";
import type { ReactNode } from "react";
import { clamp } from "./format";

// ---------------------------------------------------------------------------
// Timeline strip shared by the outcome and stage reviews. The bar,
// pointer-capture scrub, is_valid hatch, playhead, and frame footer are
// review-type-agnostic; domain markers (outcome triangle,
// soft-trunc hatch, subtask dots, event-time ticks) are injected via
// `renderOverlays`, which receives the same `pct` positioning helper the
// inline markers used.
// ---------------------------------------------------------------------------

export function Timeline({
  rawLength,
  frame,
  lastValidFrame,
  onScrub,
  renderOverlays,
}: {
  rawLength: number;
  frame: number;
  /** Drives the is_valid==0 padding hatch; null = unknown/none. */
  lastValidFrame: number | null;
  onScrub: (frame: number) => void;
  renderOverlays?: (pct: (value: number) => string) => ReactNode;
}) {
  const barRef = useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const pct = (value: number) => `${(value / rawLength) * 100}%`;

  const frameFromClientX = (clientX: number): number => {
    const bar = barRef.current;
    if (!bar) return frame;
    const rect = bar.getBoundingClientRect();
    if (rect.width <= 0) return frame;
    const ratio = (clientX - rect.left) / rect.width;
    return clamp(Math.floor(ratio * rawLength), 0, rawLength - 1);
  };

  const invalidStart =
    lastValidFrame !== null && lastValidFrame < rawLength - 1
      ? lastValidFrame + 1
      : null;

  return (
    <div className="mt-4">
      <div
        ref={barRef}
        className="relative h-12 rounded-lg bg-warm-100 border border-warm-200 cursor-pointer select-none touch-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          setDragging(true);
          onScrub(frameFromClientX(e.clientX));
        }}
        onPointerMove={(e) => {
          if (dragging) onScrub(frameFromClientX(e.clientX));
        }}
        onPointerUp={(e) => {
          e.currentTarget.releasePointerCapture(e.pointerId);
          setDragging(false);
        }}
        onPointerCancel={() => setDragging(false)}
      >
        {/* Invalid padding beyond the last valid frame */}
        {invalidStart !== null && (
          <div
            className="absolute top-0 bottom-0 bg-warm-300/60"
            style={{
              left: pct(invalidStart),
              right: 0,
              backgroundImage:
                "repeating-linear-gradient(45deg, rgba(138,127,114,0.35) 0 5px, transparent 5px 10px)",
            }}
            title={`is_valid==0 padding from frame ${invalidStart}`}
          />
        )}

        {renderOverlays?.(pct)}

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-ink"
          style={{ left: pct(frame), transform: "translateX(-50%)" }}
        />
      </div>
      <div className="flex justify-between text-[10px] font-mono text-ink-muted mt-1">
        <span>0</span>
        <span>{rawLength - 1}</span>
      </div>
    </div>
  );
}
