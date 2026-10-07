/**
 * Run the TS apply pipeline (convex/apply/) on a LOCAL dataset snapshot —
 * the offline half of the parity check against the Python outcome editor
 * (mulligan/tools/outcome_review.py) and a manual escape hatch.
 *
 * Usage:
 *   bun scripts/apply_local.ts <snapshot_dir> <config.json> <out_dir>
 *
 * config.json: {
 *   overlay: {changed_episodes: {...}, skipped_episodes: [...]},
 *   task_specs: [{task_name, num_subtask_marks}, ...],
 *   pre_apply_sha: "...",
 *   evidence: {...},                    // label-history evidence (sans pre_sha)
 *   source_by_episode: {"<idx>": {kind, agent, tool}},
 *   ts?: "1970-01-01T00:00:00+00:00"    // fixed timestamp for reproducibility
 * }
 *
 * Writes each changed file under <out_dir>/ (repo-relative paths) plus
 * <out_dir>/_summary.json.
 */

import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { join, dirname, relative } from "node:path";
import { headlessApply } from "../convex/apply/pipeline";
import type { LabelSource } from "../convex/apply/labelHistory";

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

const [snapshotDir, configPath, outDir] = process.argv.slice(2);
if (!snapshotDir || !configPath || !outDir) {
  console.error("usage: bun scripts/apply_local.ts <snapshot_dir> <config.json> <out_dir>");
  process.exit(1);
}

const config = JSON.parse(readFileSync(configPath, "utf8"));
const paths = walk(snapshotDir)
  .map((p) => relative(snapshotDir, p))
  .filter((p) => !p.startsWith(".cache/"));

const sourceByEpisode = new Map<number, LabelSource>(
  Object.entries(config.source_by_episode as Record<string, LabelSource>).map(([k, v]) => [
    parseInt(k, 10),
    v,
  ])
);

const result = await headlessApply({
  store: {
    paths,
    fetch: async (p: string) => new Uint8Array(readFileSync(join(snapshotDir, p))),
  },
  overlay: config.overlay,
  provenance: { sourceByEpisode, evidence: config.evidence },
  preApplySha: config.pre_apply_sha,
  taskSpecs: config.task_specs,
  now: config.ts ? new Date(config.ts) : undefined,
});

for (const [path, content] of result.changedFiles) {
  const dest = join(outDir, path);
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest, typeof content === "string" ? content : Buffer.from(content));
}
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, "_summary.json"), JSON.stringify(result.summary, null, 2));
console.log(
  `applied: ${result.summary.applied_episodes.length} episode(s), ` +
    `${result.changedFiles.size} changed file(s) -> ${outDir}`
);
