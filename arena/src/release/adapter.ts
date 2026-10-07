import { pairOutcomesFromRounds } from "../../convex/bradleyTerry";
import type { SessionOutcome } from "../lib/arenaRatings";
import type { Release } from "./types";

export type UISnapshot = {
  selectionSha256: string;
  datasets: { repo: string; created: number; sourceType: string }[];
  blocks: { id: string; created: number; mode: string }[];
  coverage: {
    task: string;
    taxonomy_version: string;
    generated_at: number;
    repos: {
      repo: string;
      num_episodes: number | null;
      n_prefill: number;
      n_flagged: number;
      n_committed: number;
      n_uncertain: number;
      n_draft: number;
      n_vlm_only: number;
      n_human_only: number;
    }[];
  }[];
};
export type SimStatistics = {
  schemaVersion: number;
  selectionSha256: string;
  sessions: SessionOutcome[];
  evidence: {policyId: string; seeds: {seed: number; url: string; sha256: string; gridManifestHash: string; episodes: number; successes: number; successSteps: number}[]}[];
};
export function createReleaseAdapter(data: Release, ui: UISnapshot, sim?: SimStatistics) {
  const simulation = data.tasks.filter(t => t.domain === "sim").flatMap(t => t.policies);
  if (simulation.length && (!sim || sim.schemaVersion !== 1 || sim.selectionSha256 !== data.selectionSha256))
    throw new Error("Missing or mismatched simulation statistics");
  if (sim) {
    const rows = sim.sessions.flatMap(s => s.perPolicy);
    if (rows.length !== simulation.length || new Set(rows.map(r => r.policy_id)).size !== rows.length)
      throw new Error("Incomplete simulation statistics");
    for (const p of simulation) {
      const row = rows.find(r => r.policy_id === p.id);
      const evidence = sim.evidence.find(r => r.policyId === p.id);
      if (!row || !evidence || evidence.seeds.length !== 5 || !p.seeds ||
          !evidence.seeds.every(s => p.seeds!.some(e => e.seed === s.seed && e.dataUrl === s.url)) ||
          row.rollouts <= 0 || row.successFramesCount !== row.successes ||
          new Set(evidence.seeds.map(s => s.seed)).size !== 5 ||
          row.rollouts !== evidence.seeds.reduce((n, s) => n + s.episodes, 0) ||
          row.successes !== evidence.seeds.reduce((n, s) => n + s.successes, 0) ||
          row.successFramesSum !== evidence.seeds.reduce((n, s) => n + s.successSteps, 0) ||
          Math.abs(row.successes / row.rollouts - p.rate) > 0.00000051)
        throw new Error(`Invalid simulation statistics: ${p.id}`);
    }
    for (const s of sim.sessions) {
      const ids = s.perPolicy.map(p => p.policy_id);
      const pairIds = s.pairs.map(p => [p.a, p.b].sort().join("/"));
      if (!s.rating_group || new Set(pairIds).size !== pairIds.length ||
          s.pairs.length !== ids.length * (ids.length - 1) / 2 ||
          s.perPolicy.some(p => !sim.evidence.find(e => e.policyId === p.policy_id)?.seeds.every(e =>
            s.rating_group === `${s.task}/${e.gridManifestHash}`)) ||
          s.pairs.some(p => !ids.includes(p.a) || !ids.includes(p.b) || p.a === p.b ||
            [p.winsA, p.winsB, p.draws].some(n => !Number.isSafeInteger(n) || n < 0) ||
            p.winsA + p.winsB + p.draws !== s.perPolicy[0].rollouts ||
            p.winsA - p.winsB !== s.perPolicy.find(r => r.policy_id === p.a)!.successes - s.perPolicy.find(r => r.policy_id === p.b)!.successes))
        throw new Error("Invalid grid comparisons");
    }
  }
  if (ui.selectionSha256 !== data.selectionSha256)
    throw new Error("UI snapshot selection mismatch");
  const policies = data.tasks.flatMap((t) =>
    t.policies.map((p) => ({
      _id: p.id,
      _creationTime: 0,
      name: `${t.title} · ${p.method.replace(/\s*\+\s*/g, " + ")} · ${p.round}`,
      model_id: p.modelId,
      environment: t.id,
      round: Number(p.round.slice(1)),
      method: p.method.replace(/\s*\+\s*/g, " + "),
      tags: ["paper"],
      effective_status: "mainline",
      status: "mainline",
    })),
  );
  const datasets = data.datasets.map((d) => {
    const meta = ui.datasets.find((r) => r.repo === d.id);
    const task = data.tasks.find((t) => t.datasetTask === d.task);
    if (!meta || !task) throw new Error(`Missing dataset identity: ${d.id}`);
    return {
      _id: d.id,
      _creationTime: meta.created,
      repo_id: d.id,
      name: d.id.slice(9),
      task: task.id,
      environment: task.id,
      source_type: meta.sourceType,
      dataset_role: (
        {
          "raw-collection": "aggregate_parent",
          "training-view": "training_view",
          "validation-view": "validation_view",
          evaluation: "eval_session",
          "policy-rollouts": "rollout",
        } as Record<string, string>
      )[d.role],
      trainable: d.role === "training-view" || d.role === "policy-rollouts",
      num_episodes: d.episodes,
      num_frames: d.frames,
      fps: d.fps,
      total_duration_seconds: d.frames / d.fps,
      parent_repo_id: d.parent ?? undefined,
      derived_repo_ids: data.datasets
        .filter((c) => c.parent === d.id)
        .map((c) => c.id),
      effective_status: "mainline",
    };
  });
  const sessions = data.tasks
    .flatMap((t) =>
      t.blocks.map((b) => {
        const meta = ui.blocks.find((s) => s.id === b.id);
        if (!meta) throw new Error(`Missing session date: ${b.id}`);
        const selected = policies.filter((p) =>
          b.starts.some((s) => s.results.some((r) => r.policyId === p._id)),
        );
        const first = b.startRange?.[0];
        return {
          _id: b.id,
          _creationTime: meta.created,
          label: b.label,
          round_dataset: b.roundDataset,
          dataset_repo: b.dataset,
          num_rounds: b.starts.length,
          policy_ids: selected.map((p) => p._id),
          policyNames: selected.map((p) => p.name),
          policies: selected,
          session_mode: meta.mode,
          task: t.id,
          effective_status: "mainline",
          max_subtask_marks: t.metric === "task_progress" ? 1 : 0,
          derivedDatasetRepos: data.datasets
            .filter((d) => d.parent === b.dataset && (!d.parentSession || d.parentSession === b.session))
            .map((d) => d.id),
          rounds: b.starts.map((s) => ({
            index: s.index,
            // Start IDs align joined sessions on the same initial state; rows are
            // numbered by position in the session's Sobol block, as in the paper.
            label:
              first !== undefined && s.manifestIndex !== undefined
                ? `Round ${first + s.manifestIndex}`
                : undefined,
            results: s.results.map((r) => ({
              policy_id: r.policyId,
              policyName: policies.find((p) => p._id === r.policyId)!.name,
              success: r.success,
              episode_index: r.episode,
              num_subtask_marks: r.marks,
              num_frames: r.steps,
              visit_id: r.visit,
            })),
          })),
        };
      }),
    )
    .sort((a, b) => b._creationTime - a._creationTime);
  const outcomes: SessionOutcome[] = sessions.map((s) => ({
    session_id: s._id,
    creation_time: s._creationTime,
    session_mode: s.session_mode,
    task: s.task,
    effective_status: "mainline",
    pairs: pairOutcomesFromRounds(
      s.rounds.map((r) =>
        r.results.map((p) => ({ id: p.policy_id, success: p.success })),
      ),
    ),
    perPolicy: s.policy_ids.map((id) => {
      const rows = s.rounds
        .flatMap((r) => r.results)
        .filter((r) => r.policy_id === id);
      const successful = rows.filter((r) => r.success);
      return {
        policy_id: id,
        rollouts: rows.length,
        successes: successful.length,
        successFramesSum: successful.reduce((n, r) => n + r.num_frames, 0),
        successFramesCount: successful.length,
      };
    }),
  }));
  outcomes.push(...(sim?.sessions ?? []));
  const results = sessions.flatMap((s) =>
    s.rounds.flatMap((r) =>
      r.results.map((p) => ({
        ...p,
        session_id: s._id,
        dataset_repo: s.dataset_repo,
        round_index: r.index,
        session_creation_time: s._creationTime,
      })),
    ),
  );
  const cache = new Map<string, unknown>();
  const required = <T>(v: T | undefined, label: string): T => {
    if (v === undefined) throw new Error(`Not in release: ${label}`);
    return v;
  };
  function resolve(name: string, args: Record<string, unknown> = {}): unknown {
    const key = JSON.stringify([name, args]);
    if (cache.has(key)) return cache.get(key);
    let value: unknown;
    switch (name) {
      case "users:viewer":
        value = null;
        break;
      case "operators:list":
        value = [];
        break; // Editing is unavailable; no operator identities are exported.
      case "policies:environmentsDetailed":
        value = data.tasks.map((t) => ({
          environment: t.id,
          status: "mainline",
        }));
        break;
      case "statuses:listTaskStatuses":
        value = data.tasks.map((t) => ({ task: t.id, status: "mainline" }));
        break;
      case "policies:leaderboard":
      case "policies:listNames":
        value = policies.filter(
          (p) => !args.environment || p.environment === args.environment,
        );
        break;
      case "policies:get":
        value = required(
          policies.find((p) => p._id === args.id),
          String(args.id),
        );
        break;
      case "ratings:sessionOutcomes":
        value = outcomes;
        break;
      case "datasets:list":
        value = datasets.filter(
          (d) =>
            (!args.source_type || d.source_type === args.source_type) &&
            (!args.task || d.task === args.task) &&
            (!args.dataset_role || d.dataset_role === args.dataset_role) &&
            (args.trainable === undefined || args.trainable === d.trainable),
        );
        break;
      case "datasets:getByRepo":
        value = required(
          datasets.find((d) => d.repo_id === args.repo_id),
          String(args.repo_id),
        );
        break;
      case "evalSessions:list":
        value = sessions;
        break;
      case "evalSessions:getDetail":
        value = required(
          sessions.find((s) => s._id === args.id),
          String(args.id),
        );
        break;
      case "evalSessions:getByPolicy":
        value = sessions.filter((s) =>
          s.policy_ids.includes(String(args.policy_id)),
        );
        break;
      case "roundResults:getRecentByPolicy":
      case "roundResults:getFailuresByPolicy":
        value = results
          .filter(
            (r) =>
              r.policy_id === args.policy_id &&
              (name.endsWith("getRecentByPolicy") || !r.success),
          )
          .slice(0, 20);
        break;
      case "roundResults:getSuccessRateHistory":
        value = sessions
          .filter((s) => s.policy_ids.includes(String(args.policy_id)))
          .map((s) => {
            const rows = s.rounds
              .flatMap((r) => r.results)
              .filter((r) => r.policy_id === args.policy_id);
            const successes = rows.filter((r) => r.success).length;
            return {
              successRate: successes / rows.length,
              successes,
              total: rows.length,
              datasetRepo: s.dataset_repo,
              sessionCreationTime: s._creationTime,
            };
          })
          .reverse();
        break;
      case "pairings:listRounds":
        value = sessions.flatMap((s) =>
          s.rounds
            .filter(
              (r) =>
                r.results.some((p) => p.policy_id === args.policyIdA) &&
                r.results.length > 1 &&
                (!args.policyIdB ||
                  r.results.some((p) => p.policy_id === args.policyIdB)),
            )
            .map((r) => ({
              sessionId: s._id,
              sessionCreationTime: s._creationTime,
              datasetRepo: s.dataset_repo,
              sessionMode: s.session_mode,
              sessionLabel: s.label,
              roundIndex: r.index,
              label: r.label,
              results: r.results
                .map((p) => ({
                  policyId: p.policy_id,
                  policyName: p.policyName,
                  success: p.success,
                  episodeIndex: p.episode_index,
                }))
                .sort(
                  (a, b) =>
                    Number(b.policyId === args.policyIdA) -
                      Number(a.policyId === args.policyIdA) ||
                    Number(b.policyId === args.policyIdB) -
                      Number(a.policyId === args.policyIdB),
                ),
            })),
        );
        break;
      case "stageTaskSpecs:forTask":
        value = [];
        break; // Stage editing is not part of the read-only release.
      case "stageCoverage:tasks":
        value = ui.coverage.map((c) => ({
          task: c.task,
          taxonomy_version: c.taxonomy_version,
          status: "mainline",
        }));
        break;
      case "stageCoverage:forTask":
        value = required(
          ui.coverage.find((c) => c.task === args.task),
          String(args.task),
        );
        break;
      default:
        throw new Error(
          `Query is not available in the public release: ${name}`,
        );
    }
    cache.set(key, value);
    return value;
  }
  return { resolve };
}
