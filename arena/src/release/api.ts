/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-empty-object-type -- captured backend types, kept verbatim */
/**
 * Function references for the shared screens, typed from the Convex backend.
 *
 * At runtime `api` is Convex's `anyApi` proxy, the same object the backend's generated `api`
 * module exports: `api.policies.leaderboard` is a reference whose name is the string key
 * "policies:leaderboard". In the read-only release build `client.ts` hands that key to the
 * frozen-export adapter (`adapter.ts`), which serves the release queries and throws for everything
 * else. In the self-deploy build (vite.config.ts) the same references go to the live deployment.
 * No backend module is imported, so the release build never type-checks convex/.
 *
 * The argument and result types below are captured from convex/ by scripts/capture_api.ts
 * (`FunctionArgs` / `FunctionReturnType` of each function the shared screens reference). Do not
 * edit them by hand: run `bun scripts/capture_api.ts --write` after changing those functions.
 */
import { anyApi } from "convex/server";
import type { DefaultFunctionArgs, FunctionReference } from "convex/server";
import type { GenericId } from "convex/values";
import type { PairOutcome } from "../../convex/bradleyTerry";

export type Id<TableName extends string> = GenericId<TableName>;
export type Doc<TableName extends keyof Documents> = Documents[TableName];
type Query<Args extends DefaultFunctionArgs, Result> = FunctionReference<"query", "public", Args, Result>;
type Mutation<Args extends DefaultFunctionArgs, Result> = FunctionReference<"mutation", "public", Args, Result>;

export const api = anyApi as unknown as ReleaseApi;

export type ReleaseApi = {
  applyJobs: {
    cancel: Mutation<{
      serviceToken?: string | undefined;
      id: Id<"applyJobs">;
    }, string & {
      __tableName: "applyJobs";
    }>;
    enqueue: Mutation<{
      dry_run?: boolean | undefined;
      serviceToken?: string | undefined;
      dataset_repo: string;
    }, string & {
      __tableName: "applyJobs";
    }>;
    forRepo: Query<{
      limit?: number | undefined;
      dataset_repo: string;
    }, {
      _id: Id<"applyJobs">;
      _creationTime: number;
      worker_id?: string | undefined;
      started_at?: number | undefined;
      finished_at?: number | undefined;
      hf_commit_sha?: string | undefined;
      pre_apply_sha?: string | undefined;
      error?: string | undefined;
      log_tail?: string | undefined;
      num_confirmed?: bigint | undefined;
      num_skipped?: bigint | undefined;
      dry_run?: boolean | undefined;
      status: string;
      dataset_repo: string;
      requested_by: string;
      requested_at: number;
    }[]>;
  };
  datasets: {
    getByRepo: Query<{
      repo_id: string;
    }, {
      effective_status: "mainline" | "retired" | "ablation" | "testing";
      _id: Id<"datasets">;
      _creationTime: number;
      notes?: string | undefined;
      status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
      model_id?: string | undefined;
      model_url?: string | undefined;
      status_reason?: string | undefined;
      dataset_role?: string | undefined;
      trainable?: boolean | undefined;
      num_episodes?: bigint | undefined;
      total_duration_seconds?: number | undefined;
      num_success?: bigint | undefined;
      num_failure?: bigint | undefined;
      num_human_frames?: bigint | undefined;
      num_policy_frames?: bigint | undefined;
      num_autonomous_success?: bigint | undefined;
      stats_status?: "error" | "pending" | "ready" | undefined;
      stats_hf_sha?: string | undefined;
      stats_computed_at?: number | undefined;
      stats_algorithm_version?: string | undefined;
      stats_error?: string | undefined;
      stats_refresh_requested_at?: number | undefined;
      parent_repo_id?: string | undefined;
      derived_repo_ids?: string[] | undefined;
      mutually_exclusive_with?: string[] | undefined;
      view_family_id?: string | undefined;
      view_id?: string | undefined;
      producer_model_ids?: string[] | undefined;
      target_model_id?: string | undefined;
      target_arm_key?: string | undefined;
      name: string;
      task: string;
      environment: string;
      repo_id: string;
      source_type: string;
    } | null>;
    list: Query<{
      task?: string | undefined;
      source_type?: string | undefined;
      dataset_role?: string | undefined;
      trainable?: boolean | undefined;
    }, {
      effective_status: "mainline" | "retired" | "ablation" | "testing";
      _id: Id<"datasets">;
      _creationTime: number;
      notes?: string | undefined;
      status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
      model_id?: string | undefined;
      model_url?: string | undefined;
      status_reason?: string | undefined;
      dataset_role?: string | undefined;
      trainable?: boolean | undefined;
      num_episodes?: bigint | undefined;
      total_duration_seconds?: number | undefined;
      num_success?: bigint | undefined;
      num_failure?: bigint | undefined;
      num_human_frames?: bigint | undefined;
      num_policy_frames?: bigint | undefined;
      num_autonomous_success?: bigint | undefined;
      stats_status?: "error" | "pending" | "ready" | undefined;
      stats_hf_sha?: string | undefined;
      stats_computed_at?: number | undefined;
      stats_algorithm_version?: string | undefined;
      stats_error?: string | undefined;
      stats_refresh_requested_at?: number | undefined;
      parent_repo_id?: string | undefined;
      derived_repo_ids?: string[] | undefined;
      mutually_exclusive_with?: string[] | undefined;
      view_family_id?: string | undefined;
      view_id?: string | undefined;
      producer_model_ids?: string[] | undefined;
      target_model_id?: string | undefined;
      target_arm_key?: string | undefined;
      name: string;
      task: string;
      environment: string;
      repo_id: string;
      source_type: string;
    }[]>;
    setStatus: Mutation<{
      status_reason?: string | undefined;
      serviceToken?: string | undefined;
      status: "mainline" | "retired" | "ablation" | "testing" | "inherit";
      repo_id: string;
    }, string & {
      __tableName: "datasets";
    }>;
  };
  evalSessions: {
    getByPolicy: Query<{
      policy_id: Id<"policies">;
    }, {
      _id: Id<"evalSessions">;
      _creationTime: number;
      notes?: string | undefined;
      status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
      status_reason?: string | undefined;
      submission_id?: string | undefined;
      submission_fingerprint?: string | undefined;
      session_mode?: string | undefined;
      operator?: string | undefined;
      dataset_repo: string;
      num_rounds: bigint;
      policy_ids: Id<"policies">[];
    }[]>;
    getDetail: Query<{
      id: Id<"evalSessions">;
    }, {
      policies: {
        _id: Id<"policies">;
        _creationTime: number;
        status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
        model_url?: string | undefined;
        training_url?: string | undefined;
        round?: number | undefined;
        method?: string | undefined;
        tags?: string[] | undefined;
        status_reason?: string | undefined;
        name: string;
        model_id: string;
        environment: string;
      }[];
      max_subtask_marks: number;
      rounds: {
        index: number;
        results: {
          policy_id: string;
          policyName: string;
          success: boolean;
          episode_index: number;
          num_subtask_marks: number | null;
          num_frames: number | null;
        }[];
      }[];
      _id: Id<"evalSessions">;
      _creationTime: number;
      notes?: string | undefined;
      status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
      status_reason?: string | undefined;
      submission_id?: string | undefined;
      submission_fingerprint?: string | undefined;
      session_mode?: string | undefined;
      operator?: string | undefined;
      dataset_repo: string;
      num_rounds: bigint;
      policy_ids: Id<"policies">[];
    } | null>;
    list: Query<{}, {
      policyNames: string[];
      task: string | null;
      effective_status: "mainline" | "retired" | "ablation" | "testing";
      derivedDatasetRepos: string[];
      _id: Id<"evalSessions">;
      _creationTime: number;
      notes?: string | undefined;
      status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
      status_reason?: string | undefined;
      submission_id?: string | undefined;
      submission_fingerprint?: string | undefined;
      session_mode?: string | undefined;
      operator?: string | undefined;
      dataset_repo: string;
      num_rounds: bigint;
      policy_ids: Id<"policies">[];
    }[]>;
    setOperator: Mutation<{
      serviceToken?: string | undefined;
      id: Id<"evalSessions">;
      operator: string;
    }, string & {
      __tableName: "evalSessions";
    }>;
    setStatus: Mutation<{
      status_reason?: string | undefined;
      serviceToken?: string | undefined;
      id: Id<"evalSessions">;
      status: "mainline" | "retired" | "ablation" | "testing" | "inherit";
    }, string & {
      __tableName: "evalSessions";
    }>;
  };
  operators: {
    list: Query<{}, {
      _id: Id<"operators">;
      _creationTime: number;
      hf_username: string;
      added_at: number;
      added_by: string;
    }[]>;
  };
  pairings: {
    listRounds: Query<{
      policyIdB?: Id<"policies"> | undefined;
      policyIdA: Id<"policies">;
    }, {
      sessionId: string;
      sessionCreationTime: number;
      datasetRepo: string;
      sessionMode: string;
      roundIndex: number;
      results: Array<{
        policyId: string;
        policyName: string;
        success: boolean;
        episodeIndex: number;
      }>;
    }[]>;
  };
  policies: {
    environmentsDetailed: Query<{}, {
      environment: string;
      status: "mainline" | "retired" | "ablation" | "testing";
    }[]>;
    get: Query<{
      id: Id<"policies">;
    }, {
      effective_status: "mainline" | "retired" | "ablation" | "testing";
      _id: Id<"policies">;
      _creationTime: number;
      status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
      model_url?: string | undefined;
      training_url?: string | undefined;
      round?: number | undefined;
      method?: string | undefined;
      tags?: string[] | undefined;
      status_reason?: string | undefined;
      name: string;
      model_id: string;
      environment: string;
    } | null>;
    leaderboard: Query<{
      environment?: string | undefined;
    }, {
      effective_status: "mainline" | "retired" | "ablation" | "testing";
      _id: Id<"policies">;
      _creationTime: number;
      status?: "mainline" | "retired" | "ablation" | "testing" | undefined;
      model_url?: string | undefined;
      training_url?: string | undefined;
      round?: number | undefined;
      method?: string | undefined;
      tags?: string[] | undefined;
      status_reason?: string | undefined;
      name: string;
      model_id: string;
      environment: string;
    }[]>;
    listNames: Query<Record<string, never>, {
      _id: Id<"policies">;
      name: string;
      environment: string;
      effective_status: "mainline" | "retired" | "ablation" | "testing";
    }[]>;
    setStatus: Mutation<{
      status_reason?: string | undefined;
      serviceToken?: string | undefined;
      status: "mainline" | "retired" | "ablation" | "testing" | "inherit";
      model_id: string;
    }, string & {
      __tableName: "policies";
    }>;
    setTags: Mutation<{
      serviceToken?: string | undefined;
      model_id: string;
      round: number | null;
      method: string | null;
      tags: string[];
    }, string & {
      __tableName: "policies";
    }>;
    tagOptions: Query<{}, {
      rounds: number[];
      methods: string[];
      tags: string[];
    }>;
  };
  ratings: {
    sessionOutcomes: Query<{}, {
      session_id: Id<"evalSessions">;
      creation_time: number;
      session_mode: string;
      task: string | null;
      effective_status: "mainline" | "retired" | "ablation" | "testing";
      pairs: PairOutcome[];
      perPolicy: {
        rollouts: number;
        successes: number;
        successFramesSum: number;
        successFramesCount: number;
        policy_id: string;
      }[];
    }[]>;
  };
  reviews: {
    episodeNotes: Query<{
      episode_index: bigint;
      dataset_repo: string;
    }, {
      _id: Id<"episodeNotes">;
      _creationTime: number;
      episode_index: bigint;
      notes: string;
      updated_at: number;
      updated_by: string;
      dataset_repo: string;
    } | null>;
    latestForRepo: Query<{
      dataset_repo: string;
    }, {
      episodes: {
        _id: Id<"outcomeReviews">;
        _creationTime: number;
        new_outcome?: string | undefined;
        outcome_frame?: bigint | undefined;
        soft_truncate?: boolean | undefined;
        subtask_frames?: bigint[] | undefined;
        reviewer_user_id?: Id<"users"> | undefined;
        episode_index: bigint;
        status: string;
        dataset_repo: string;
        reviewer: string;
        saved_at: number;
      }[];
      num_confirmed: number;
      num_skipped: number;
    }>;
    noteSuggestions: Query<{
      episode_index: bigint;
      dataset_repo: string;
    }, {
      recent: {
        notes: string;
        count: number;
        lastUsed: number;
      }[];
      mostUsed: {
        notes: string;
        count: number;
        lastUsed: number;
      }[];
    }>;
    save: Mutation<{
      new_outcome?: string | undefined;
      outcome_frame?: bigint | undefined;
      soft_truncate?: boolean | undefined;
      subtask_frames?: bigint[] | undefined;
      serviceToken?: string | undefined;
      episode_index: bigint;
      status: string;
      dataset_repo: string;
    }, string & {
      __tableName: "outcomeReviews";
    }>;
    saveNotes: Mutation<{
      episode_index: bigint;
      notes: string;
      dataset_repo: string;
    }, null>;
  };
  roundResults: {
    getFailuresByPolicy: Query<{
      policy_id: Id<"policies">;
    }, {
      session_id: Id<"evalSessions">;
      dataset_repo: string;
      round_index: number;
      episode_index: number;
      success: boolean;
      num_frames: number | null;
      session_creation_time: number;
    }[]>;
    getRecentByPolicy: Query<{
      policy_id: Id<"policies">;
    }, {
      session_id: Id<"evalSessions">;
      dataset_repo: string;
      round_index: number;
      episode_index: number;
      success: boolean;
      num_frames: number | null;
      session_creation_time: number;
    }[]>;
    getSuccessRateHistory: Query<{
      policy_id: Id<"policies">;
    }, {
      successRate: number;
      successes: number;
      total: number;
      datasetRepo: string;
      sessionCreationTime: number;
    }[]>;
  };
  stageCoverage: {
    forTask: Query<{
      includeAll?: boolean | undefined;
      task: string;
    }, {
      task: string;
      taxonomy_version: string;
      stage_field: string;
      n_stages: number;
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
      pipelines: {
        name: string;
        version: string;
        model: string;
        n: number;
      }[];
      reviewers: {
        reviewer: string;
        committed: number;
        uncertain: number;
        draft: number;
      }[];
      stage_hist_current: number[];
      stage_hist_committed: number[];
      n_unknown_stage: number;
      generated_at: number;
    } | null>;
    tasks: Query<{}, {
      task: string;
      taxonomy_version: string;
      status: "mainline" | "retired" | "ablation" | "testing";
    }[]>;
  };
  stagePredictions: {
    forRun: Query<{
      run_id: Id<"stagePredictionRuns">;
      paginationOpts: {
        id?: number;
        endCursor?: string | null;
        maximumRowsRead?: number;
        maximumBytesRead?: number;
        numItems: number;
        cursor: string | null;
      };
    }, {
      page: {
        task: string;
        dataset_repo: string;
        taxonomy_version: string;
        pipeline: {
          name: string;
          version: string;
          git_commit: string;
        };
        source: string;
        pushed_at: number;
        _id: Id<"stagePredictions">;
        _creationTime: number;
        canonical_response?: any;
        source_revision?: string | undefined;
        review_reason?: string | undefined;
        violation_codes?: string[] | undefined;
        confidence?: string | undefined;
        vote_summary?: any;
        episode_index: bigint;
        label: Record<string, any>;
        episode_duration_s: number;
        evidence: any;
        content_sha256: string;
        validation_codes: string[];
        run_id: Id<"stagePredictionRuns">;
      }[];
      isDone: boolean;
      continueCursor: string;
      splitCursor?: string | null;
      pageStatus?: "SplitRecommended" | "SplitRequired" | null;
    }>;
    listForRepo: Query<{
      taxonomy_version: string;
      dataset_repo: string;
    }, {
      runs: {
        _id: Id<"stagePredictionRuns">;
        _creationTime: number;
        taxonomy_version: string;
        task: string;
        dataset_repo: string;
        pipeline: {
          name: string;
          version: string;
          git_commit: string;
        };
        run_key: string;
        expected_count: number;
        published_at: number;
        run_id: Id<"stagePredictionRuns">;
      }[];
      active_run_id: Id<"stagePredictionRuns"> | null;
    }>;
    otherSchemasForEpisode: Query<{
      episode_index: bigint;
      taxonomy_version: string;
      task: string;
      dataset_repo: string;
    }, {
      taxonomy_version: string;
      run_id: Id<"stagePredictionRuns">;
      expected_count: number;
      published_at: number;
    }[]>;
  };
  stageReviews: {
    latestForRepo: Query<{
      taxonomy_version?: string | undefined;
      dataset_repo: string;
    }, {
      episodes: {
        _id: Id<"stageReviews">;
        _creationTime: number;
        label?: Record<string, any> | undefined;
        episode_duration_s?: number | undefined;
        notes?: string | undefined;
        reviewer_user_id?: Id<"users"> | undefined;
        taxonomy_hash?: string | undefined;
        prediction_id?: Id<"stagePredictions"> | undefined;
        review_coverage?: {
          protocol: "structured-v1";
          reviewed_fields: string[];
          excluded_fields: string[];
        } | undefined;
        event_links?: {
          attempt_index: number;
          action_id: string;
          stage_id: string;
          relation: "shared" | "distinct";
        }[] | undefined;
        prefill_pushed_at?: number | undefined;
        prediction_sha256?: string | undefined;
        prediction_run_id?: Id<"stagePredictionRuns"> | undefined;
        copied_from_review_id?: Id<"stageReviews"> | undefined;
        blind?: boolean | undefined;
        episode_index: bigint;
        taxonomy_version: string;
        status: string;
        task: string;
        dataset_repo: string;
        reviewer: string;
        saved_at: number;
      }[];
      num_confirmed: number;
      num_corrected: number;
      num_uncertain: number;
      num_draft: number;
    }>;
    save: Mutation<{
      label?: Record<string, any> | undefined;
      episode_duration_s?: number | undefined;
      notes?: string | undefined;
      prediction_id?: Id<"stagePredictions"> | undefined;
      event_links?: {
        attempt_index: number;
        action_id: string;
        stage_id: string;
        relation: "shared" | "distinct";
      }[] | undefined;
      prefill_pushed_at?: number | undefined;
      prediction_sha256?: string | undefined;
      copied_from_review_id?: Id<"stageReviews"> | undefined;
      blind?: boolean | undefined;
      serviceToken?: string | undefined;
      review_protocol?: "structured-v1" | undefined;
      reviewer_override?: string | undefined;
      saved_at_override?: number | undefined;
      episode_index: bigint;
      taxonomy_version: string;
      status: string;
      task: string;
      dataset_repo: string;
    }, string & {
      __tableName: "stageReviews";
    }>;
  };
  stageTaskSpecs: {
    forTask: Query<{
      task: string;
    }, {
      _id: Id<"stageTaskSpecs">;
      _creationTime: number;
      taxonomy_version: string;
      task: string;
      exported_at: number;
      source: string;
      taxonomy_hash: string;
      live: boolean;
      spec: any;
    }[]>;
  };
  statuses: {
    listTaskStatuses: Query<{}, {
      _id: Id<"taskStatuses">;
      _creationTime: number;
      reason?: string | undefined;
      superseded_by?: string | undefined;
      status: "mainline" | "retired" | "ablation" | "testing";
      task: string;
      updated_at: number;
      updated_by: string;
    }[]>;
    setTaskStatus: Mutation<{
      reason?: string | undefined;
      superseded_by?: string | undefined;
      serviceToken?: string | undefined;
      status: "mainline" | "retired" | "ablation" | "testing";
      task: string;
    }, string & {
      __tableName: "taskStatuses";
    }>;
  };
  taskSpecs: {
    forTask: Query<{
      task: string;
    }, {
      _id: Id<"taskSpecs">;
      _creationTime: number;
      task: string;
      num_subtask_marks: bigint;
      task_name: string;
      stored_frame_hw: bigint[];
      camera_keys_by_role: Record<string, string>;
      crop_boxes: Record<string, bigint[]>;
      review_camera_roles: string[];
      exported_at: number;
      source: string;
    } | null>;
  };
  users: {
    viewer: Query<{}, {
      userId: Id<"users">;
      name: string;
      username: string | null;
      image: string | null;
      isEditor: boolean;
    } | null>;
  };
};
type Documents = {
};
