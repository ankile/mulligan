import type { TrajectoryEventLink } from "../../convex/trajectoryEventLinks";
import { useCallback, useEffect, useState } from "react";
import type { StageLabelRow } from "../../convex/stageConsistency";
import type { PredictionAttribution, ReviewSeed } from "./stagePredictionReview";

export interface StageDraft extends ReviewSeed {
  key: string;
  dirty: boolean;
}

/**
 * Source-keyed local drafts protect URL changes as well as UI navigation.
 * A source change immediately hides old marks; seeding occurs only once for
 * each context, so reactive publications/reviews cannot alter a working form.
 * This hook never writes a human review. Explicit navigation owns autosave.
 */
export function useStageReviewDraft(key: string | null, seed: ReviewSeed | null) {
  const [drafts, setDrafts] = useState<Map<string, StageDraft>>(() => new Map());
  const draft = key === null ? null : drafts.get(key) ?? null;

  useEffect(() => {
    // Keep unsaved outgoing edits if the URL changed outside our navigation.
    // Clean forms are re-seeded on return from the latest saved human review.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDrafts((previous) => {
      const next = new Map([...previous].filter(([storedKey, value]) =>
        storedKey === key || value.dirty));
      if (key !== null && seed !== null && !next.has(key)) {
        next.set(key, { ...seed, key, dirty: false });
      }
      return next.size === previous.size && [...next].every(([k, value]) => previous.get(k) === value)
        ? previous : next;
    });
  }, [key, seed]);

  const edit = useCallback((patch: StageLabelRow) => {
    if (key === null) return;
    setDrafts((previous) => {
      const current = previous.get(key);
      if (!current) return previous;
      const label = { ...current.label };
      for (const [field, value] of Object.entries(patch)) {
        if (value === undefined) delete label[field];
        else label[field] = value;
      }
      return new Map(previous).set(key, { ...current, label, dirty: true });
    });
  }, [key]);

  const replaceLabel = useCallback((label: StageLabelRow, attribution: PredictionAttribution, eventLinks?: TrajectoryEventLink[]) => {
    if (key === null) return;
    setDrafts((previous) => {
      const current = previous.get(key);
      if (!current) return previous;
      return new Map(previous).set(key, { ...current, label: { ...label }, attribution: { ...attribution },
        eventLinks: eventLinks?.map((link) => ({ ...link })), inheritedSuccess: false, fromOwnReview: false, dirty: true });
    });
  }, [key]);

  const editHumanNotes = useCallback((humanNotes: string) => {
    if (key === null) return;
    setDrafts((previous) => {
      const current = previous.get(key);
      if (!current) return previous;
      return new Map(previous).set(key, { ...current, humanNotes, dirty: true });
    });
  }, [key]);

  const editEventLinks = useCallback((eventLinks: TrajectoryEventLink[]) => {
    if (key === null) return;
    setDrafts((previous) => {
      const current = previous.get(key);
      if (!current) return previous;
      return new Map(previous).set(key, { ...current, eventLinks: eventLinks.map((link) => ({ ...link })), dirty: true });
    });
  }, [key]);

  const markSaved = useCallback((savedKey: string) => {
    setDrafts((previous) => {
      const current = previous.get(savedKey);
      if (!current) return previous;
      return new Map(previous).set(savedKey, { ...current, dirty: false });
    });
  }, []);

  const unsavedCount = [...drafts.values()].filter((value) => value.dirty).length;
  useEffect(() => {
    if (unsavedCount === 0) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [unsavedCount]);

  return { draft, edit, editHumanNotes, editEventLinks, replaceLabel, markSaved, unsavedCount };
}
