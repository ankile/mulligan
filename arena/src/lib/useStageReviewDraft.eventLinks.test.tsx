import { afterAll, afterEach, expect, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { useStageReviewDraft } from "./useStageReviewDraft";
import type { ReviewSeed } from "./stagePredictionReview";
import type { TrajectoryEventLink } from "../../convex/trajectoryEventLinks";

GlobalRegistrator.register({ url: "http://localhost/" });
const { act, cleanup, renderHook } = await import("@testing-library/react");
afterEach(() => cleanup());
afterAll(() => GlobalRegistrator.unregister());
const link: TrajectoryEventLink = { action_id: "arrival", stage_id: "arrived", attempt_index: 1, relation: "distinct" };
const seed: ReviewSeed = { label: { max_stage: 1 }, attribution: {}, fromOwnReview: true, inheritedSuccess: false };

test("metadata-only edits guard navigation and preserve outgoing draft across source changes", async () => {
  const { result, rerender } = renderHook(({ source }) => useStageReviewDraft(source, seed), { initialProps: { source: "A" } });
  await act(async () => {});
  act(() => result.current.editEventLinks([link]));
  expect(result.current.draft?.dirty).toBe(true);
  expect(result.current.unsavedCount).toBe(1);
  rerender({ source: "B" });
  await act(async () => {});
  expect(result.current.draft?.eventLinks).toBeUndefined();
  expect(result.current.unsavedCount).toBe(1);
  rerender({ source: "A" });
  await act(async () => {});
  expect(result.current.draft?.eventLinks).toEqual([link]);
});

test("batched time and association edits survive together; copying a label replaces old links", async () => {
  const { result } = renderHook(() => useStageReviewDraft("A", seed));
  await act(async () => {});
  act(() => {
    result.current.edit({ max_stage: 2 });
    result.current.editEventLinks([link]);
  });
  expect(result.current.draft?.label.max_stage).toBe(2);
  expect(result.current.draft?.eventLinks).toEqual([link]);
  act(() => result.current.replaceLabel({ max_stage: 3 }, {}, [{ ...link, relation: "shared" }]));
  expect(result.current.draft?.eventLinks?.[0].relation).toBe("shared");
  act(() => result.current.replaceLabel({ max_stage: 4 }, {}));
  expect(result.current.draft?.eventLinks).toBeUndefined();
});
