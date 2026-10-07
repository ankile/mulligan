import { describe, expect, test } from "bun:test";

import { episodeScore, meanScore, outcomeLabel, outcomeTone } from "./outcomeScore";

describe("graded outcome display", () => {
  test("score is marks plus success", () => {
    expect(episodeScore(true, 1)).toBe(2);
    expect(episodeScore(false, 1)).toBe(1);
    expect(episodeScore(false, 0)).toBe(0);
    expect(episodeScore(true, null)).toBe(1);
  });

  test("tone: pass, partial only for a failed episode with marks", () => {
    expect(outcomeTone(true, 1)).toBe("pass");
    expect(outcomeTone(true, 0)).toBe("pass");
    expect(outcomeTone(false, 1)).toBe("partial");
    expect(outcomeTone(false, 0)).toBe("fail");
    expect(outcomeTone(false, null)).toBe("fail");
  });

  test("binary tasks keep PASS / FAIL", () => {
    expect(outcomeLabel(true, 0, 0)).toBe("PASS");
    expect(outcomeLabel(false, 0, 0)).toBe("FAIL");
    expect(outcomeLabel(false, null, 0)).toBe("FAIL");
  });

  test("graded tasks show the score out of max", () => {
    expect(outcomeLabel(true, 1, 1)).toBe("PASS 2/2");
    expect(outcomeLabel(false, 1, 1)).toBe("PARTIAL 1/2");
    expect(outcomeLabel(false, 0, 1)).toBe("FAIL 0/2");
    // Submitted before mark counts existed: tone word only, no made-up score.
    expect(outcomeLabel(false, null, 1)).toBe("FAIL");
    expect(outcomeLabel(true, null, 1)).toBe("PASS");
  });

  test("mean score over rounds, null for binary tasks", () => {
    const rounds = [
      { success: true, marks: 1 },
      { success: false, marks: 1 },
      { success: false, marks: 0 },
    ];
    expect(meanScore(rounds, 1)).toBeCloseTo(1.0);
    expect(meanScore(rounds, 0)).toBeNull();
    expect(meanScore([], 1)).toBeNull();
  });
});
