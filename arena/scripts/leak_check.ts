/**
 * Scan a built site for references that must not ship.
 *
 *   bun scripts/leak_check.ts dist-release     read-only Mulligan snapshot (default mode)
 *   bun scripts/leak_check.ts --app dist       self-deploy build
 *
 * Both modes reject secrets: a PEM private key or a Hugging Face access token. The snapshot mode
 * also rejects any Convex URL, sign-in text and a live Convex client, because the snapshot has no
 * backend. The self-deploy mode allows those: its bundle holds the user's own VITE_CONVEX_URL and
 * the sign-in flow.
 *
 * Scans every built page and asset (not the frozen data files, which are checked by hash).
 */
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";

export const SECRET_PATTERNS: [string, RegExp][] = [
  ["private key", /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
  ["Hugging Face token", /\bhf_[A-Za-z0-9]{30,}\b/],
];

export const LEAK_PATTERNS: [string, RegExp][] = [
  ["Convex deployment URL", /convex\.(cloud|site)/i],
  ["sign-in text", /sign[ -]?in\b|signin|log[ -]?in\b/i],
  ["live Convex client", /VITE_CONVEX_URL|ConvexReactClient|ConvexAuthProvider/],
];

const SCANNED = /\.(js|css|html|txt|json|map)$|^_headers$/;
const SECRET_LABELS = new Set(SECRET_PATTERNS.map(([label]) => label));

function files(dir: string, top = true): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((d) => {
    const path = join(dir, d.name);
    if (d.isDirectory()) return top && d.name === "data" ? [] : files(path, false);
    return SCANNED.test(d.name) ? [path] : [];
  });
}

export function leakCheck(dir: string, mode: "release" | "app" = "release"): string[] {
  const found: string[] = [];
  const scanned = files(dir);
  if (!scanned.some((f) => f.endsWith(".js"))) throw new Error(`No built JavaScript under ${dir}`);
  const patterns = mode === "app" ? SECRET_PATTERNS : [...SECRET_PATTERNS, ...LEAK_PATTERNS];
  for (const file of scanned) {
    const text = readFileSync(file, "utf8");
    for (const [label, pattern] of patterns) {
      const m = pattern.exec(text);
      if (!m) continue;
      // A secret is reported by label only, so the log does not repeat it.
      const context = SECRET_LABELS.has(label)
        ? ""
        : ` (${JSON.stringify(text.slice(Math.max(0, m.index - 40), m.index + 40))})`;
      found.push(`${relative(dir, file)}: ${label}${context}`);
    }
  }
  return found;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const mode = args[0] === "--app" ? "app" : "release";
  const rest = mode === "app" ? args.slice(1) : args;
  if (rest.length !== 1 || rest[0].startsWith("--")) throw new Error("Usage: bun scripts/leak_check.ts [--app] OUTPUT_DIR");
  const found = leakCheck(rest[0], mode);
  if (found.length) {
    for (const f of found) console.error(f);
    process.exit(1);
  }
  console.log(`Leak check passed for ${rest[0]}${mode === "app" ? " (self-deploy build)" : ""}`);
}
