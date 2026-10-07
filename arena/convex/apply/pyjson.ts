/**
 * Python-compatible JSON serialization for apply-pipeline artifacts.
 *
 * mulligan/tools/outcome_review.py writes its sidecars with specific json.dumps
 * settings; matching them keeps the files uniform whichever tool wrote them
 * (values are what matter — key order and spacing are matched where cheap so
 * diffs stay readable):
 *  - progress record:      json.dump(indent=2)
 *  - results.json:         json.dumps(indent=2, sort_keys=True) + "\n"
 *  - label history lines:  json.dumps(sort_keys=True), one per line
 */

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

function serialize(value: Json, indent: number | null, sortKeys: boolean, depth: number): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`Cannot serialize non-finite number ${value}`);
    if (Number.isInteger(value) && Object.is(value, -0) === false) return String(value);
    // Match Python repr for floats closely enough: JS String() already emits
    // shortest round-trip representation, same as Python repr for doubles.
    return String(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  const nl = indent !== null ? "\n" + " ".repeat(indent * (depth + 1)) : "";
  const nlEnd = indent !== null ? "\n" + " ".repeat(indent * depth) : "";
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    const items = value.map((v) => serialize(v, indent, sortKeys, depth + 1));
    return indent !== null
      ? `[${nl}${items.join("," + nl)}${nlEnd}]`
      : `[${items.join(", ")}]`;
  }
  const keys = Object.keys(value);
  if (sortKeys) keys.sort();
  if (keys.length === 0) return "{}";
  const items = keys.map(
    (k) => `${JSON.stringify(k)}: ${serialize(value[k], indent, sortKeys, depth + 1)}`
  );
  return indent !== null
    ? `{${nl}${items.join("," + nl)}${nlEnd}}`
    : `{${items.join(", ")}}`;
}

/** json.dumps(value, indent=2) */
export function dumpsIndent2(value: Json): string {
  return serialize(value, 2, false, 0);
}

/** json.dumps(value, indent=2, sort_keys=True) */
export function dumpsIndent2Sorted(value: Json): string {
  return serialize(value, 2, true, 0);
}

/** json.dumps(value, sort_keys=True) — compact with Python default separators. */
export function dumpsSorted(value: Json): string {
  return serialize(value, null, true, 0);
}

/** json.dumps(value, indent=4) — used by meta/stats.json. */
export function dumpsIndent4(value: Json): string {
  return serialize(value, 4, false, 0);
}

export type { Json };
