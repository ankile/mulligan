/**
 * Append-only label-provenance ledger (.label_history.jsonl).
 *
 * Port of mulligan/real/lifecycle/label_history.py build_outcome_events +
 * append_events for the headless apply path. Events are one sorted-keys JSON
 * object per line, appended in the same HF commit as the label change.
 */

import { dumpsSorted } from "./pyjson";
import type { Json } from "./pyjson";
import type { ChangedRecord, Overlay, ProgressRecord } from "./progress";

export const HISTORY_FILENAME = ".label_history.jsonl";
export const EVENT_VERSION = 1;
const SOURCE_KINDS = ["auto", "human", "vlm", "heuristic", "migration"];
export const SKIP_PAYLOAD = { action: "skip" } as const;

export interface LabelSource {
  kind: string;
  agent: string;
  tool: string;
}

export interface LabelEvent {
  v: number;
  label_kind: string;
  episode_index: number;
  payload: Json;
  source: LabelSource;
  evidence: Json;
  ts: string;
  prev?: Json;
}

/** datetime.now(timezone.utc).isoformat(timespec="seconds") */
export function utcNowIso(now: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${now.getUTCFullYear()}-${pad(now.getUTCMonth() + 1)}-${pad(now.getUTCDate())}` +
    `T${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())}+00:00`
  );
}

function makeEvent(args: {
  episodeIndex: number;
  payload: Json;
  source: LabelSource;
  evidence: Json;
  ts: string;
  prev: Json | null;
}): LabelEvent {
  if (!SOURCE_KINDS.includes(args.source.kind)) {
    throw new Error(`unknown source kind ${JSON.stringify(args.source.kind)}`);
  }
  if (!args.source.tool) throw new Error("source.tool is required");
  const event: LabelEvent = {
    v: EVENT_VERSION,
    label_kind: "outcome",
    episode_index: args.episodeIndex,
    payload: args.payload,
    source: args.source,
    evidence: args.evidence,
    ts: args.ts,
  };
  if (args.prev !== null) event.prev = args.prev;
  return event;
}

function recordJson(record: ChangedRecord): Json {
  // Plain-object clone so payload/prev comparisons and serialization see the
  // exact progress-record shape (subtask_frames key presence preserved).
  return JSON.parse(JSON.stringify(record)) as Json;
}

function jsonEqual(a: Json | null, b: Json | null): boolean {
  return dumpsSorted(a as Json) === dumpsSorted(b as Json);
}

function preState(preProgress: ProgressRecord, epIdx: number): Json | null {
  const record = preProgress.changed_episodes[String(epIdx)];
  if (record !== undefined) return recordJson(record);
  if (preProgress.skipped_episodes.some((s) => Number(s) === epIdx)) {
    return { ...SKIP_PAYLOAD };
  }
  return null;
}

/**
 * build_outcome_events: one event per overlay decision that CHANGES the
 * pre-apply progress state. No-op entries emit nothing — re-stamping
 * untouched episodes would rewrite the when/under-which-job provenance facts.
 */
export function buildOutcomeEvents(args: {
  preProgress: ProgressRecord;
  overlay: Overlay;
  sourceByEpisode: Map<number, LabelSource>;
  evidence: Json;
  ts: string;
}): LabelEvent[] {
  const events: LabelEvent[] = [];
  for (const [epStr, record] of Object.entries(args.overlay.changed_episodes)) {
    const epIdx = parseInt(epStr, 10);
    const prev = preState(args.preProgress, epIdx);
    if (jsonEqual(prev, recordJson(record))) continue;
    const source = args.sourceByEpisode.get(epIdx);
    if (!source) throw new Error(`No provenance source for episode ${epIdx}`);
    events.push(
      makeEvent({
        episodeIndex: epIdx,
        payload: recordJson(record),
        source,
        evidence: args.evidence,
        ts: args.ts,
        prev,
      })
    );
  }
  for (const epIdxRaw of args.overlay.skipped_episodes) {
    const epIdx = Number(epIdxRaw);
    const prev = preState(args.preProgress, epIdx);
    if (jsonEqual(prev, { ...SKIP_PAYLOAD })) continue;
    const source = args.sourceByEpisode.get(epIdx);
    if (!source) throw new Error(`No provenance source for episode ${epIdx}`);
    events.push(
      makeEvent({
        episodeIndex: epIdx,
        payload: { ...SKIP_PAYLOAD },
        source,
        evidence: args.evidence,
        ts: args.ts,
        prev,
      })
    );
  }
  return events.sort((a, b) => a.episode_index - b.episode_index);
}

/** append_events onto the existing ledger text (created on first use). */
export function appendEvents(existingText: string | null, events: LabelEvent[]): string {
  const lines = events.map((e) => dumpsSorted(e as unknown as Json) + "\n").join("");
  if (existingText === null) return lines;
  return existingText + lines;
}
