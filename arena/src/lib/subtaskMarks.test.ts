import { describe, expect, test } from "bun:test";

import { placeSubtaskMark } from "./subtaskMarks";

describe("placeSubtaskMark", () => {
  test("adds a mark below the required count", () => {
    expect(placeSubtaskMark([], 40, 1)).toEqual([40]);
    expect(placeSubtaskMark([40], 10, 2)).toEqual([10, 40]);
  });

  test("removes the mark on an already-marked frame", () => {
    expect(placeSubtaskMark([40], 40, 1)).toEqual([]);
    expect(placeSubtaskMark([10, 40], 10, 2)).toEqual([40]);
  });

  test("moves the single mark when the count is full (required = 1)", () => {
    expect(placeSubtaskMark([40], 55, 1)).toEqual([55]);
    expect(placeSubtaskMark([40], 12, 1)).toEqual([12]);
  });

  test("moves the nearest mark when the count is full (required > 1)", () => {
    expect(placeSubtaskMark([10, 40], 45, 2)).toEqual([10, 45]);
    expect(placeSubtaskMark([10, 40], 12, 2)).toEqual([12, 40]);
    // The moved mark may cross another one; output stays sorted.
    expect(placeSubtaskMark([10, 40], 100, 2)).toEqual([10, 100]);
    expect(placeSubtaskMark([10, 40], 0, 2)).toEqual([0, 40]);
  });

  test("distance ties resolve to the earlier mark", () => {
    expect(placeSubtaskMark([10, 20], 15, 2)).toEqual([15, 20]);
  });

  test("does not mutate the input", () => {
    const marks = [40];
    placeSubtaskMark(marks, 55, 1);
    expect(marks).toEqual([40]);
  });

  test("rejects a zero requirement", () => {
    expect(() => placeSubtaskMark([], 5, 0)).toThrow();
  });
});
