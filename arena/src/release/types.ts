export type Policy = {
  id: string; round: string; method: string; arm: string; modelId: string;
  rate: number; lo: number; hi: number; successes: number | null; episodes: number | null;
  provenance: string; blocks: string[];
  seeds?: { seed: number; rate: number; sourcePath: string; artifact?: string; dataUrl?: string }[];
};
export type Result = {
  policyId: string; episode: number; success: boolean; outcome: string;
  steps: number; frames: number; score: number; marks: number | null;
  videos: Record<string, {path: string; from: number; to: number; file: number}>;
  /** Route Cable launch that recorded this rollout. */
  visit?: string;
};
export type Block = {
  id: string; round: string; dataset: string; source: string; revision: string;
  fps: number; cameras: string[]; reviewedEpisodes: number | null; sourcePath: string;
  /** Session of the round dataset (fixed release block ID) and its readable label. */
  session?: string; roundDataset?: string; label?: string; date?: string; seed?: number;
  /** First/last Sobol start of the session, 1-based, as in the paper's session table. */
  startRange?: [number, number];
  /** index: the round dataset's start ID (joined on exact initial state); manifestIndex: position in the session's Sobol block. */
  starts: {index: number; manifestIndex?: number; results: Result[]}[];
};
export type Task = {
  id: string; title: string; domain: 'real' | 'sim'; datasetTask: string;
  evaluationDataset?: {id: string; revision: string};
  metric: 'full_success' | 'task_progress'; policies: Policy[]; blocks: Block[];
};
export type Dataset = {
  id: string; task: string; role: string; variant: string | null; episodes: number;
  frames: number; fps: number; cameras: string[]; parent: string | null;
  /** Session of the parent round dataset this view was split from. */
  parentSession?: string;
  source: string; revision: string; tier: string;
};
export type Release = {
  schemaVersion: number; version: string; state: string; tasks: Task[];
  datasets: Dataset[]; selectionSha256: string; sourceSha256: Record<string, string>;
};

export function validateRelease(data: Release): Release {
  if (data.schemaVersion !== 1 || !['draft', 'public_verified'].includes(data.state)) throw new Error('Unsupported release');
  const expected = ['marker_d2', 'square_d2', 'routing_d2', 'square_narrow', 'square_broad'];
  if (data.tasks.length !== 5 || expected.some(id => !data.tasks.some(t => t.id === id))) throw new Error('Incomplete task selection');
  const datasets = new Set(data.datasets.map(d => d.id));
  for (const d of data.datasets) {
    if (!d.id.startsWith('mulligan/') || !/^[a-f0-9]{40}$/.test(d.revision)) throw new Error('Unpinned dataset');
    if (d.parent !== null && !datasets.has(d.parent)) throw new Error('Missing dataset ancestor');
  }
  for (const t of data.tasks) {
    const policies = new Set(t.policies.map(p => p.id));
    if (policies.size !== t.policies.length) throw new Error('Duplicate policy');
    for (const b of t.blocks) {
      if (!datasets.has(b.dataset) || !/^[a-f0-9]{40}$/.test(b.revision)) throw new Error('Unpinned eval block');
      for (const start of b.starts) for (const result of start.results) {
        if (!policies.has(result.policyId)) throw new Error('Unselected policy in eval block');
      }
    }
  }
  return data;
}

/** Unknown IDs cannot resolve to a live/internal Arena entity. */
export function selectedTask(data: Release, id: string | null): Task {
  const task = data.tasks.find(t => t.id === (id ?? data.tasks[0].id));
  if (!task) throw new Error('This task is not part of the release.');
  return task;
}
