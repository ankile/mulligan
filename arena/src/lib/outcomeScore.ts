// Graded outcome display for tasks with mid-episode sub-goal marks
// (RealTaskSpec.num_subtask_marks, exported to taskSpecs). Score = marks
// reached + success, so a routing_d2 episode scores 0, 1, or 2 "clips".
// Binary tasks (maxMarks = 0) keep the plain PASS / FAIL surface.

export type OutcomeTone = "pass" | "partial" | "fail";

export function episodeScore(success: boolean, marks: number | null): number {
  return (marks ?? 0) + (success ? 1 : 0);
}

/** Partial = failed but reached at least one sub-goal (marks known and > 0). */
export function outcomeTone(success: boolean, marks: number | null): OutcomeTone {
  if (success) return "pass";
  if (marks !== null && marks > 0) return "partial";
  return "fail";
}

/**
 * Pill text. Binary task: PASS / FAIL. Graded task: the score out of its max
 * (marks + 1), e.g. "PASS 2/2", "PARTIAL 1/2", "FAIL 0/2"; when the round was
 * submitted without a mark count the score is unknown, so only the tone word.
 */
export function outcomeLabel(
  success: boolean,
  marks: number | null,
  maxMarks: number
): string {
  const word = { pass: "PASS", partial: "PARTIAL", fail: "FAIL" }[outcomeTone(success, marks)];
  if (maxMarks <= 0 || marks === null) return word;
  return `${word} ${episodeScore(success, marks)}/${maxMarks + 1}`;
}

export const TONE_PILL: Record<OutcomeTone, string> = {
  pass: "bg-teal-light text-teal",
  partial: "bg-gold-light text-gold",
  fail: "bg-coral-light text-coral",
};

/** Overlay badge on a video tile (dark background). */
export const TONE_BADGE: Record<OutcomeTone, string> = {
  pass: "bg-emerald-500/80 text-white",
  partial: "bg-amber-500/80 text-white",
  fail: "bg-red-500/80 text-white",
};

/** Mean graded score over rounds, e.g. "1.30" — null when the task is binary. */
export function meanScore(
  rounds: Array<{ success: boolean; marks: number | null }>,
  maxMarks: number
): number | null {
  if (maxMarks <= 0 || rounds.length === 0) return null;
  const total = rounds.reduce((acc, r) => acc + episodeScore(r.success, r.marks), 0);
  return total / rounds.length;
}
