import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "../../lib/arenaClient";
import { api } from "../../release/api";

export function EpisodeNotes({ repoId, episodeIndex }: { repoId: string; episodeIndex: number }) {
  const args = { dataset_repo: repoId, episode_index: BigInt(episodeIndex) };
  const saved = useQuery(api.reviews.episodeNotes, args);
  const suggestions = useQuery(api.reviews.noteSuggestions, args);
  const [suggestionOrder, setSuggestionOrder] = useState<"recent" | "mostUsed">("recent");
  const textarea = useRef<HTMLTextAreaElement>(null);
  const saveNotes = useMutation(api.reviews.saveNotes);
  // Keep drafts and failed saves when navigating between episodes.
  const key = JSON.stringify([repoId, episodeIndex]);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const notes = drafts[key] ?? saved?.notes ?? "";
  const dirty = notes !== (saved?.notes ?? "");

  function updateDraft(value: string) {
    setDrafts((prev) => {
      const next = { ...prev, [key]: value };
      if (value === (saved?.notes ?? "") && !saving[key]) delete next[key];
      return next;
    });
  }

  useEffect(() => {
    if (Object.keys(drafts).length === 0 && !Object.values(saving).some(Boolean)) return;
    const warnUnsaved = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warnUnsaved);
    return () => window.removeEventListener("beforeunload", warnUnsaved);
  }, [drafts, saving]);

  async function save() {
    if (!dirty || saving[key] || saved === undefined) return;
    setSaving((prev) => ({ ...prev, [key]: true }));
    setErrors((prev) => ({ ...prev, [key]: "" }));
    try {
      await saveNotes({ ...args, notes });
      setDrafts((prev) => {
        if (prev[key] !== notes) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      });
    } catch (error) {
      setErrors((prev) => ({ ...prev, [key]: String(error) }));
    } finally {
      setSaving((prev) => ({ ...prev, [key]: false }));
    }
  }

  return (
    <div className="mt-4 rounded-lg border border-warm-200 p-3">
      <div className="flex items-center justify-between gap-3 mb-2">
        <label htmlFor="episode-notes" className="text-xs font-medium text-ink">
          Notes <span className="font-normal text-ink-muted">(optional)</span>
        </label>
        <button
          type="button"
          disabled={saved === undefined || !dirty || saving[key]}
          onClick={() => void save()}
          className="text-xs text-teal cursor-pointer disabled:text-ink-muted disabled:cursor-default"
        >
          {saved === undefined ? "Loading…" : saving[key] ? "Saving…" : dirty ? "Save notes" : "Saved"}
        </button>
      </div>
      <textarea
        ref={textarea}
        id="episode-notes"
        rows={2}
        value={notes}
        disabled={saved === undefined}
        placeholder="Leave a note to revisit later…"
        aria-describedby="episode-notes-help"
        onChange={(event) => updateDraft(event.target.value)}
        onBlur={() => void save()}
        className="w-full resize-y rounded-lg border border-warm-200 bg-white px-3 py-2 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:border-teal"
      />
      {suggestions && suggestions.recent.length > 0 && (
        <div className="mt-2" aria-label="Prior notes for this task">
          <div className="mb-1.5 flex flex-wrap items-center gap-2 text-[11px] text-ink-muted">
            <span>Reuse a note for this task</span>
            {(["recent", "mostUsed"] as const).map((order) => (
              <button
                key={order}
                type="button"
                aria-pressed={suggestionOrder === order}
                onClick={() => setSuggestionOrder(order)}
                className={`rounded px-1.5 py-0.5 cursor-pointer ${suggestionOrder === order ? "bg-teal-light text-teal" : "hover:text-ink"}`}
              >
                {order === "recent" ? "Recent" : "Most used"}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {suggestions[suggestionOrder].map((suggestion) => (
              <button
                key={suggestion.notes}
                type="button"
                disabled={saved === undefined || saving[key]}
                title={`${suggestion.notes}\nUsed in ${suggestion.count} ${suggestion.count === 1 ? "episode" : "episodes"}`}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  updateDraft(notes.trim() ? `${notes.trimEnd()}\n${suggestion.notes}` : suggestion.notes);
                  textarea.current!.focus();
                }}
                className="max-w-full truncate rounded-full border border-warm-200 bg-white px-2.5 py-1 text-xs text-ink-muted hover:border-teal hover:text-teal cursor-pointer disabled:opacity-50 disabled:cursor-default"
              >
                {suggestion.notes}
              </button>
            ))}
          </div>
        </div>
      )}
      <p id="episode-notes-help" className="mt-1 text-[11px] text-ink-muted">
        Saved automatically when you leave this field. Notes stay in Policy Arena.
      </p>
      {Object.entries(errors).filter(([, error]) => error).map(([failedKey, error]) => (
        <p key={failedKey} role="alert" className="mt-2 text-xs text-coral">
          Could not save notes for episode {JSON.parse(failedKey)[1]}: {error}. Return to that episode to retry.
        </p>
      ))}
    </div>
  );
}
