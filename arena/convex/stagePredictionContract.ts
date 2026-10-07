import { v } from "convex/values";

/** Shared, versioned wire contract. Numbers hash as binary64, not decimal JSON. */
export const CONTENT_PROTOCOL = "arena-prediction-content/v1";
export const pipelineValidator = v.object({
  name: v.string(), version: v.string(), git_commit: v.string(),
});
export const predictionFields = {
  episode_index: v.int64(),
  label: v.record(v.string(), v.any()),
  episode_duration_s: v.float64(),
  evidence: v.any(),
  canonical_response: v.optional(v.any()),
  source_revision: v.optional(v.string()),
  review_reason: v.optional(v.string()),
  violation_codes: v.optional(v.array(v.string())),
  confidence: v.optional(v.string()),
  vote_summary: v.optional(v.any()),
};
export const predictionValidator = v.object(predictionFields);

function validString(value: string): void {
  // TextEncoder replaces unpaired UTF-16 surrogates; that would hash different
  // strings identically, so reject them before encoding.
  for (let i = 0; i < value.length; i++) {
    const n = value.charCodeAt(i);
    if (n >= 0xd800 && n <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new Error("unpaired Unicode surrogate");
    } else if (n >= 0xdc00 && n <= 0xdfff) throw new Error("unpaired Unicode surrogate");
  }
}

const encoder = new TextEncoder();
function keyOrder(a: string, b: string): number {
  const x = encoder.encode(a), y = encoder.encode(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    if (x[i] !== y[i]) return x[i] - y[i];
  }
  return x.length - y.length;
}

export function canonicalEncoding(value: unknown, depth = 0): string {
  if (depth > 64) throw new Error("prediction JSON exceeds maximum nesting depth");
  if (value === null) return "z";
  if (typeof value === "boolean") return value ? "t" : "f";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("non-finite prediction number");
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value === 0 ? 0 : value, false);
    return "n" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }
  if (typeof value === "string") {
    validString(value);
    return `s${encoder.encode(value).length}:${value}`;
  }
  if (Array.isArray(value)) {
    return `a${value.length}:` + value.map((item) => canonicalEncoding(item, depth + 1)).join("");
  }
  if (typeof value === "object" && value !== null && Object.getPrototypeOf(value) === Object.prototype) {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record);
    keys.forEach(validString);
    keys.sort(keyOrder);
    return `o${keys.length}:` + keys.map((key) =>
      canonicalEncoding(key, depth + 1) + canonicalEncoding(record[key], depth + 1)
    ).join("");
  }
  throw new Error("prediction content must contain only strict JSON values");
}

export async function canonicalDigest(value: unknown): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(canonicalEncoding(value)));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export function hash(value: string, field: string): void {
  if (!/^[0-9a-f]{64}$/.test(value)) throw new Error(`${field} must be a lowercase SHA-256`);
}

export function predictionContent(row: Record<string, unknown>): Record<string, unknown> {
  const content: Record<string, unknown> = {};
  for (const key of Object.keys(predictionFields)) {
    if (row[key] !== undefined) content[key] = row[key];
  }
  if (typeof content.episode_index !== "bigint" || content.episode_index < 0n) {
    throw new Error("episode_index must be a nonnegative int64");
  }
  content.episode_index = content.episode_index.toString();
  return content;
}

export async function predictionDigest(row: Record<string, unknown>): Promise<string> {
  return canonicalDigest(predictionContent(row));
}

export async function manifestDigest(
  rows: { episode_index: bigint; content_sha256: string }[],
): Promise<string> {
  const sorted = [...rows].sort((a, b) => a.episode_index < b.episode_index ? -1 : a.episode_index > b.episode_index ? 1 : 0);
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].episode_index < 0n) throw new Error("negative manifest episode");
    hash(sorted[i].content_sha256, "content_sha256");
    if (i && sorted[i - 1].episode_index === sorted[i].episode_index) throw new Error("duplicate manifest episode");
  }
  return canonicalDigest(sorted.map((r) => [r.episode_index.toString(), r.content_sha256]));
}
