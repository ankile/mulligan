/**
 * Pure placement rule for the review 'g' key (subtask marks).
 *
 * - On an already-marked frame: remove that mark.
 * - Below the required count: add a mark at this frame.
 * - At the required count: MOVE the nearest existing mark to this frame
 *   (ties resolve to the earlier mark). This replaces the old scrub-back /
 *   toggle-off / scrub-forward / toggle-on dance for re-placing a mark.
 *
 * `required` must be > 0 — the caller gates on that before invoking.
 * The returned array is always sorted ascending.
 */
export function placeSubtaskMark(
  marks: readonly number[],
  markAt: number,
  required: number,
): number[] {
  if (required <= 0) {
    throw new Error(`placeSubtaskMark requires required > 0, got ${required}`);
  }
  if (marks.includes(markAt)) {
    return marks.filter((m) => m !== markAt);
  }
  let next: number[];
  if (marks.length < required) {
    next = [...marks, markAt];
  } else {
    let nearest = marks[0];
    for (const m of marks) {
      if (Math.abs(m - markAt) < Math.abs(nearest - markAt)) nearest = m;
    }
    next = [...marks.filter((m) => m !== nearest), markAt];
  }
  return next.sort((a, b) => a - b);
}
