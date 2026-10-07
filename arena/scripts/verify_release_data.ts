import { dirname, join } from "node:path";
import { readFileSync } from "node:fs";
import { createReleaseAdapter } from "../src/release/adapter";
import { validateRelease } from "../src/release/types";
import { computeArenaStats } from "../src/lib/arenaRatings";
import type { SessionOutcome } from "../src/lib/arenaRatings";
const [releasePath, uiPath] = process.argv.slice(2);
if (!releasePath || !uiPath)
  throw new Error(
    "Usage: bun scripts/verify_release_data.ts RELEASE_JSON UI_JSON",
  );
const data = validateRelease(JSON.parse(readFileSync(releasePath, "utf8")));
const ui = JSON.parse(readFileSync(uiPath, "utf8"));
const sim = JSON.parse(readFileSync(join(dirname(releasePath), "sim-statistics.json"), "utf8"));
const adapter = createReleaseAdapter(data, ui, sim);
const sessions = adapter.resolve("ratings:sessionOutcomes") as SessionOutcome[];
const stats = computeArenaStats(sessions, true);
for (const task of data.tasks)
  for (const p of task.policies) {
    if (!stats.ratings.has(p.id) || !stats.wdl.has(p.id) || !stats.success.get(p.id)?.avgSuccessSteps)
      throw new Error(`Missing computed fields: ${p.id}`);
    if (task.domain === "real") {
      const s = stats.success.get(p.id);
      if (!s || s.successes !== p.successes || s.rollouts !== p.episodes)
        throw new Error(`Result mismatch: ${p.id}`);
    } else if (!stats.success.has(p.id) || !stats.ratings.has(p.id) || !stats.wdl.has(p.id) || !stats.success.get(p.id)!.avgSuccessSteps || !p.seeds || p.seeds.length !== 5)
      throw new Error(`Invalid simulation evidence: ${p.id}`);
  }
const repos = new Set(data.datasets.map((d) => d.id));
for (const c of ui.coverage)
  for (const r of c.repos) {
    if (!repos.has(r.repo)) throw new Error(`Out of scope coverage: ${r.repo}`);
  }
for (const bad of ["example/pilot", "mulligan/not-in-release"]) {
  let failed = false;
  try {
    adapter.resolve("datasets:getByRepo", { repo_id: bad });
  } catch {
    failed = true;
  }
  if (!failed) throw new Error("Unknown dataset did not fail closed");
}
console.log(
  `Verified ${data.tasks.flatMap((t) => t.policies).length} policy points; ${sessions.length} selected sessions; ${repos.size} repositories; verified fixed-grid simulation comparisons.`,
);
