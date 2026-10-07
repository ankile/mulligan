import { useCallback, useEffect, useId, useRef, useState } from "react";

/** Commit valid edits immediately so focused input cannot bypass navigation
 * guards. Invalid partial text stays visible and blocks saves/navigation. */
function TimeInput({
  value,
  flagged,
  disabled,
  onCommit,
  preservePrecision = false,
  onPendingInputChange,
  comfortable = false,
}: {
  value: number | null;
  flagged: boolean;
  disabled: boolean;
  onCommit: (t: number | null) => void;
  preservePrecision?: boolean;
  onPendingInputChange?: (id: string, pending: boolean) => void;
  comfortable?: boolean;
}) {
  const inputId = useId();
  const edited = useRef(false);
  useEffect(() => () => onPendingInputChange?.(inputId, false), [inputId, onPendingInputChange]);
  const [text, setText] = useState(value === null ? "" : String(value));
  useEffect(() => {
    /* eslint-disable-next-line react-hooks/set-state-in-effect -- sync the
       local draft text to an externally-committed value (controlled bridge). */
    setText(value === null ? "" : String(value));
  }, [value]);
  return (
    <input
      type="text"
      inputMode="decimal"
      disabled={disabled}
      value={text}
      aria-invalid={text.trim() !== "" && !Number.isFinite(Number(text))}
      title={text.trim() !== "" && !Number.isFinite(Number(text)) ? "Enter a finite timestamp or clear this field before saving or leaving" : undefined}
      onChange={(event) => {
        const nextText = event.target.value;
        edited.current = true;
        setText(nextText);
        const parsed = nextText.trim() === "" ? null : Number(nextText);
        const invalid = parsed !== null && !Number.isFinite(parsed);
        onPendingInputChange?.(inputId, invalid);
        if (!invalid) onCommit(parsed);
      }}
      onBlur={() => {
        if (!edited.current) return;
        edited.current = false;
        const s = text.trim();
        if (s === "") {
          onCommit(null);
          return;
        }
        const parsed = Number(s);
        if (Number.isFinite(parsed)) onCommit(preservePrecision ? parsed : Math.round(parsed * 100) / 100);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        e.stopPropagation();
      }}
      placeholder="—"
      className={`${comfortable ? "w-20 px-2 py-1.5 text-sm" : "w-16 px-1 py-0.5 text-[11px]"} rounded border font-mono text-right ${
        flagged ? "border-coral ring-1 ring-coral/40" : "border-warm-200"
      }`}
    />
  );
}

/** Time input + frame equivalent + mark/seek/clear — shared by bool-paired
 *  and standalone time rows so EVERY event time gets the same affordances. */
export function TimeControls({
  t,
  fps,
  frame,
  flagged,
  disabled,
  markDisabled,
  markTitle,
  onCommit,
  onMark,
  onSeek,
  canClear,
  onClear,
  clearTitle,
  preservePrecision = false,
  onPendingInputChange,
  comfortable = false,
}: {
  t: number | null;
  fps: number;
  frame: number;
  flagged: boolean;
  disabled: boolean;
  markDisabled: boolean;
  markTitle: string;
  onCommit: (t: number | null) => void;
  onMark: () => boolean | void;
  onSeek: (t: number) => void;
  canClear: boolean;
  onClear: () => void;
  clearTitle: string;
  preservePrecision?: boolean;
  onPendingInputChange?: (id: string, pending: boolean) => void;
  comfortable?: boolean;
}) {
  const [resetGeneration, setResetGeneration] = useState(0);
  const [inputPending, setInputPending] = useState(false);
  const pendingChange = useCallback((id: string, pending: boolean) => {
    setInputPending(pending);
    onPendingInputChange?.(id, pending);
  }, [onPendingInputChange]);
  return (
    <>
      <TimeInput key={resetGeneration} value={t} flagged={flagged} disabled={disabled} onCommit={onCommit} preservePrecision={preservePrecision} onPendingInputChange={pendingChange} comfortable={comfortable} />
      <span className="text-[10px] font-mono text-ink-muted w-10">
        {t !== null ? `f${Math.round(t * fps)}` : ""}
      </span>
      <button
        disabled={disabled || markDisabled}
        onClick={() => { if (onMark() !== false) setResetGeneration((value) => value + 1); }}
        className={`${comfortable ? "px-2 py-2 text-xs" : "px-1.5 py-0.5 text-[10px]"} rounded font-mono ${
          markDisabled
            ? "bg-warm-100 text-ink-muted/40 cursor-not-allowed"
            : "bg-teal/10 text-teal hover:bg-teal/20 cursor-pointer"
        }`}
        title={
          markDisabled
            ? "frame drift detected — re-seek until the banner clears"
            : `${markTitle} (frame ${frame})`
        }
      >
        ◉ mark
      </button>
      <button
        disabled={t === null}
        onClick={() => t !== null && onSeek(t)}
        className={`${comfortable ? "px-2 py-2 text-xs" : "px-1.5 py-0.5 text-[10px]"} rounded font-mono ${
          t !== null
            ? "bg-warm-100 text-ink-muted hover:bg-warm-200 cursor-pointer"
            : "text-ink-muted/30"
        }`}
        title={t !== null ? `seek to ${t}s` : undefined}
      >
        →
      </button>
      <button
        disabled={disabled || (!canClear && !inputPending)}
        onClick={() => { onClear(); setResetGeneration((value) => value + 1); }}
        className={`${comfortable ? "px-2 py-2 text-xs" : "px-1.5 py-0.5 text-[10px]"} rounded font-mono text-ink-muted hover:text-coral cursor-pointer`}
        title={clearTitle}
      >
        ×
      </button>
    </>
  );
}
