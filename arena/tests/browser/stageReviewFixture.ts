import { getFunctionName, type FunctionArgs } from "convex/server";
import { api } from "../../convex/_generated/api";
import type { EpisodeFrameSignals } from "../../src/lib/hf-api";
import type { StageReviewDataSource } from "../../src/lib/stageReviewDataSource";
import trajectoryFixtures from "../fixtures/trajectory-review-fixtures.json";
import fixtureDoc from "../fixtures/stage-consistency-fixtures.json";

/** Synthetic I/O only. Shared by DOM regression tests and the local visual fixture. */
type SaveArgs = FunctionArgs<typeof api.stageReviews.save>;
const spec = fixtureDoc["routing_d2@s10_v1"].spec;
const pipeline = { name: "test-pipeline", version: "v1", git_commit: "abc" };
const makeRun = (id: string) => ({
  _id: id, run_key: id, pipeline, published_at: 1_700_000_000_000,
  expected_count: 2, task: "routing_d2", dataset_repo: "org/repo", taxonomy_version: "s10_v1",
});
const prediction = (run: string, episode: number, stage: number) => ({
  _id: `${run}-prediction-${episode}`, run_id: run, episode_index: BigInt(episode),
  label: {
    max_stage: stage, failure_mode: "other", final_state: "rope_in_gripper", notes: `model ${run}`,
    rope_grasped: true, rope_grasped_time_s: 1, first_clip_seated: stage >= 6,
    ...(stage >= 6 ? { first_clip_seated_time_s: 2 } : {}),
    regrasped: false, second_clip_seated: false, rope_released: false,
  },
  content_sha256: run.repeat(64).slice(0, 64), validation_codes: [] as string[], episode_duration_s: run === "A" ? 12 : 20,
  pipeline, evidence: { artifact_sha256: "artifact" }, pushed_at: 1_700_000_000_000,
  canonical_response: { max_stage: { stage_index: stage }, immutable_marker: `raw-${run}` },
});

export function createStageReviewFixture(
  initialReviews: Array<Record<string, unknown>> = [],
  onChange: () => void = () => {},
) {
  const state = {
    viewerUsername: "annotator", viewerUserId: "user-1",
    active: "A", runs: [makeRun("A"), makeRun("B")],
    reviews: initialReviews,
    outcome: "success",
    ready: true,
    saves: [] as SaveArgs[],
    loadMore: [] as number[],
    save: (async () => {}) as (args: SaveArgs) => Promise<void>,
    queries: [] as string[],
    exits: 0,
    armFetches: 0,
    signalFetches: 0,
    missingPredictionEpisodes: new Set<number>(),
    fetchSignals: (async (): Promise<EpisodeFrameSignals> => { throw new Error("Fixture requires an explicit frame-signal response"); }),
    predictionOverrides: {} as Record<string, unknown>,
    specRows: [{ taxonomy_version: "s10_v1", live: true, spec: spec as Record<string, unknown> }],
    otherSchemas: [] as Array<{ taxonomy_version: string; run_id: string; expected_count: number; published_at: number }>,
  };
  const queryValues = () => ({
    "users:viewer": { userId: state.viewerUserId, username: state.viewerUsername, isEditor: true },
    "stageTaskSpecs:forTask": state.specRows,
    "taskSpecs:forTask": null,
    "reviews:latestForRepo": { episodes: [0, 1].map((ep) => ({ episode_index: BigInt(ep), status: "confirmed", new_outcome: state.outcome })) },
    "stageReviews:latestForRepo": { episodes: state.reviews, num_confirmed: 0, num_corrected: 0 },
    "stagePredictions:otherSchemasForEpisode": state.otherSchemas,
    "stagePredictions:listForRepo": { runs: state.runs, active_run_id: state.active },
  });
  const save = async (args: SaveArgs) => {
    state.saves.push(args);
    await state.save(args);
    state.reviews = state.reviews.filter((row) => row.episode_index !== args.episode_index);
    state.reviews.push({ ...args, _id: `human-review-${state.saves.length}`,
      reviewer_user_id: state.viewerUserId, reviewer: state.viewerUsername, saved_at: 1_700_000_100_000 });
    onChange();
    return "saved-review";
  };
  const dataSource: StageReviewDataSource = {
    useQuery: ((query: Parameters<StageReviewDataSource["useQuery"]>[0], args?: unknown) => {
      if (args === "skip") return undefined;
      const name = getFunctionName(query);
      state.queries.push(name);
      const values = queryValues();
      if (!(name in values)) throw new Error(`Unexpected query ${name}`);
      return values[name as keyof typeof values];
    }) as StageReviewDataSource["useQuery"],
    useMutation: (() => save) as unknown as StageReviewDataSource["useMutation"],
    usePaginatedQuery: ((_query: unknown, args: { run_id: string } | "skip") => ({
      results: args === "skip" ? [] : [prediction(args.run_id, 0, args.run_id === "A" ? 3 : 7), prediction(args.run_id, 1, 2)]
        .filter((row) => !state.missingPredictionEpisodes.has(Number(row.episode_index)))
        .map((row) => ({ ...row, ...state.predictionOverrides })),
      status: state.ready ? "Exhausted" : "LoadingMore",
      isLoading: !state.ready,
      loadMore: (count: number) => state.loadMore.push(count),
    })) as unknown as StageReviewDataSource["usePaginatedQuery"],
    fetchReviewEpisodes: async () => [0, 1].map((episodeIndex) => ({ episodeIndex, rawLength: 450, dataPath: "data.parquet", perCamera: {} })),
    fetchAppliedProgress: async () => null,
    fetchLabelHistory: async () => [],
    fetchLedgerArms: async () => { state.armFetches++; return new Map([[0, "ours"], [1, "baseline"]]); },
    fetchEpisodeFrameSignals: async () => { state.signalFetches++; return state.fetchSignals(); },
  };
  const props = { repoId: "org/repo", task: "routing_d2", onExit: () => { state.exits++; }, onOpenOutcomeReview: () => { state.exits++; }, dataSource };
  return { state, props };
}

/** Configure the same offline I/O path with exported generic contract data. */
export function configureTrajectoryFixture(fixture: ReturnType<typeof createStageReviewFixture>, sourceName = "routing_d2_v1", caseName = "historical_high_stage_final_failure") {
  const task = trajectoryFixtures.synthetic.tasks.find((task) => task.source_name === sourceName);
  if (!task) throw new Error(`Unknown trajectory fixture ${sourceName}`);
  const selected = task.cases.find((item) => item.name === caseName)
    ?? trajectoryFixtures.real_campaign_cases.find((item) => item.source_name === sourceName && item.name === caseName);
  if (!selected?.review_label) throw new Error(`Unknown trajectory case ${caseName}`);
  const { state, props } = fixture;
  props.task = task.spec.task;
  state.specRows = [{ taxonomy_version: task.spec.taxonomy_version, live: true, spec: task.spec }];
  state.runs = state.runs.map((run) => ({ ...run, task: task.spec.task, taxonomy_version: task.spec.taxonomy_version }));
  state.predictionOverrides = { label: structuredClone(selected.review_label), canonical_response: selected.canonical,
    episode_duration_s: selected.duration_s, source_revision: "f".repeat(40) };
  return { task, selected };
}
