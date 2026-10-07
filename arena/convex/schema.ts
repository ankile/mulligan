import { authTables } from "@convex-dev/auth/server";
import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";
import { statusValidator } from "./statusShared";
import { CONTENT_PROTOCOL, pipelineValidator, predictionFields } from "./stagePredictionContract";
import { reviewCoverageValidator } from "./stageReviewCoverage";
import { eventLinksValidator } from "./trajectoryEventLinks";

export default defineSchema({
  ...authTables,

  // Hand-managed task/line lifecycle status (see statusShared.ts). Lives in
  // its own table because taskSpecs/stageTaskSpecs are db.replace()d wholesale
  // by the Python exporters and would destroy hand-set fields. Per-entity
  // `status` overrides on policies/evalSessions/datasets take precedence;
  // tasks without a row are mainline.
  taskStatuses: defineTable({
    task: v.string(), // matches policies.environment / datasets.task
    status: statusValidator,
    reason: v.optional(v.string()),
    superseded_by: v.optional(v.string()), // e.g. pilot_task_v0 -> pilot_task
    updated_at: v.float64(),
    updated_by: v.string(),
  }).index("by_task", ["task"]),

  // Overrides authTables.users to add the Hugging Face username — the
  // DISPLAY/audit string; authorization keys on the OIDC sub via
  // authAccounts.providerAccountId + ARENA_EDITOR_SUBS (see access.ts).
  users: defineTable({
    name: v.optional(v.string()),
    image: v.optional(v.string()),
    email: v.optional(v.string()),
    emailVerificationTime: v.optional(v.float64()),
    phone: v.optional(v.string()),
    phoneVerificationTime: v.optional(v.float64()),
    isAnonymous: v.optional(v.boolean()),
    username: v.optional(v.string()),
  })
    .index("email", ["email"])
    .index("phone", ["phone"]),

  policies: defineTable({
    name: v.string(),
    model_id: v.string(),
    model_url: v.optional(v.string()),
    training_url: v.optional(v.string()),
    environment: v.string(),
    round: v.optional(v.number()),
    method: v.optional(v.string()),
    tags: v.optional(v.array(v.string())),
    status: v.optional(statusValidator), // override; absent = inherit from task
    status_reason: v.optional(v.string()),
  })
    .index("by_model_id", ["model_id"])
    .index("by_environment", ["environment"]),

  evalSessions: defineTable({
    dataset_repo: v.string(),
    num_rounds: v.int64(),
    policy_ids: v.array(v.id("policies")),
    submission_id: v.optional(v.string()),
    submission_fingerprint: v.optional(v.string()),
    notes: v.optional(v.string()),
    session_mode: v.optional(v.string()), // "manual" | "pool-sample" | "calibrate" | "rollout"
    status: v.optional(statusValidator), // override; absent = inherit from task
    status_reason: v.optional(v.string()),
    // Who physically ran the eval — an HF USERNAME from the `operators`
    // registry (validated at write). Stored as the username (not a users id)
    // so it can be recorded before the person ever signs in; once they log in
    // via HF OAuth, users.username makes the join to their account.
    operator: v.optional(v.string()),
  }).index("by_submission_id", ["submission_id"]),

  // Registry of known eval operators (HF usernames). Sessions validate their
  // `operator` against this table; the UI select is populated from it.
  operators: defineTable({
    hf_username: v.string(),
    added_at: v.float64(),
    added_by: v.string(),
  }).index("by_username", ["hf_username"]),

  roundResults: defineTable({
    session_id: v.id("evalSessions"),
    round_index: v.int64(),
    policy_id: v.id("policies"),
    success: v.boolean(),
    episode_index: v.int64(),
    num_frames: v.optional(v.int64()),
    // Mid-episode sub-goal marks reached (0..taskSpecs.num_subtask_marks); the
    // graded score is marks + success. Live 'g' presses at submit time, then
    // the review record after an outcome-review apply. Absent = never recorded.
    num_subtask_marks: v.optional(v.int64()),
  })
    .index("by_session", ["session_id"])
    .index("by_policy", ["policy_id"]),

  // Append-only human outcome reviews (web port of mulligan/tools/outcome_review.py).
  // Latest row per (dataset_repo, episode_index) wins; "cleared" undoes.
  outcomeReviews: defineTable({
    dataset_repo: v.string(),
    episode_index: v.int64(),
    status: v.string(), // "confirmed" | "skipped" | "cleared"
    new_outcome: v.optional(v.string()), // "success" | "failure" | "timeout"
    outcome_frame: v.optional(v.int64()),
    soft_truncate: v.optional(v.boolean()),
    subtask_frames: v.optional(v.array(v.int64())),
    reviewer: v.string(),
    reviewer_user_id: v.optional(v.id("users")),
    saved_at: v.float64(),
  })
    .index("by_repo", ["dataset_repo"])
    .index("by_repo_episode", ["dataset_repo", "episode_index"]),

  // Episode annotations are independent of outcome decisions and HF apply jobs.
  episodeNotes: defineTable({
    dataset_repo: v.string(),
    episode_index: v.int64(),
    notes: v.string(),
    updated_by: v.string(),
    updated_at: v.float64(),
  }).index("by_repo_episode", ["dataset_repo", "episode_index"]),

  // Jobs bridging web reviews to HF via the apply worker (applyWorker.ts).
  applyJobs: defineTable({
    dataset_repo: v.string(),
    status: v.string(), // pending | applying | applied | failed | cancelled
    requested_by: v.string(),
    requested_at: v.float64(),
    worker_id: v.optional(v.string()),
    started_at: v.optional(v.float64()),
    finished_at: v.optional(v.float64()),
    hf_commit_sha: v.optional(v.string()),
    pre_apply_sha: v.optional(v.string()),
    error: v.optional(v.string()),
    log_tail: v.optional(v.string()),
    num_confirmed: v.optional(v.int64()),
    num_skipped: v.optional(v.int64()),
    dry_run: v.optional(v.boolean()),
  })
    .index("by_repo", ["dataset_repo"])
    .index("by_status", ["status"]),

  // Task-spec data exported from the Python task registry (RealTaskSpec) by
  // mulligan/tools/export_arena_task_specs.py, never edited by hand. The UI
  // reads crop boxes and subtask-mark counts from here instead of mirroring
  // Python constants in TS.
  taskSpecs: defineTable({
    task: v.string(), // the task name datasets carry — matches datasets.task
    task_name: v.string(), // LeRobot/collection task name, e.g. "square_d2"
    num_subtask_marks: v.int64(),
    // Crop reference space [H, W] (stored-frame pixels; station frames are 480x640).
    stored_frame_hw: v.array(v.int64()),
    // Station role -> camera key of serial-named datasets (e.g. side_1 ->
    // "<serial>_left"). Role-named datasets match the role itself.
    camera_keys_by_role: v.record(v.string(), v.string()),
    // Station role -> effective display crop [x0, y0, x1, y1] in stored-frame
    // pixels, half-open (defaults merged with RealTaskSpec.camera_crop_overrides).
    crop_boxes: v.record(v.string(), v.array(v.int64())),
    // Default review camera roles in display order (RealTaskSpec.
    // consumed_camera_roles, e.g. marker_d2 -> ["side_1", "wrist_left"]);
    // empty = show every stream.
    review_camera_roles: v.array(v.string()),
    exported_at: v.float64(),
    source: v.string(), // name of the exporter that wrote the row
  }).index("by_task", ["task"]),

  // Stage-label task specs: the full StageLabelTaskSpec vocabulary, exported
  // per (task, taxonomy_version) so a live taxonomy and candidate ones (e.g. a
  // proposed stage split) coexist. `spec` is the JSON document written by
  // mulligan/tools/export_arena_task_specs.py (serialize_stage_spec), written
  // only through the machine API; storing it opaque keeps this schema stable
  // across taxonomy changes, and all numbers arrive as plain float64 (no
  // BigInt handling in the UI).
  stageTaskSpecs: defineTable({
    task: v.string(),
    taxonomy_version: v.string(),
    taxonomy_hash: v.string(),
    live: v.boolean(), // exactly one live version per task (enforced in upsert)
    spec: v.any(),
    exported_at: v.float64(),
    source: v.string(), // name of the exporter that wrote the row
  })
    .index("by_task", ["task"])
    .index("by_task_version", ["task", "taxonomy_version"]),

  stagePredictionRuns: defineTable({
    run_key: v.string(),
    dataset_repo: v.string(),
    task: v.string(),
    taxonomy_version: v.string(),
    taxonomy_hash: v.string(),
    spec_content_sha256: v.string(),
    content_protocol: v.literal(CONTENT_PROTOCOL),
    pipeline: pipelineValidator,
    expected_count: v.float64(),
    manifest_sha256: v.string(),
    identity_sha256: v.string(),
    source: v.string(),
    provenance: v.any(),
    status: v.union(v.literal("uploading"), v.literal("published")),
    received_count: v.float64(),
    created_at: v.float64(),
    published_at: v.optional(v.float64()),
  })
    .index("by_key", ["run_key"])
    .index("by_repo_taxonomy", ["dataset_repo", "taxonomy_version"])
    .index("by_task", ["task"]),

  stagePredictions: defineTable({
    ...predictionFields,
    validation_codes: v.array(v.string()),
    run_id: v.id("stagePredictionRuns"),
    content_sha256: v.string(),
  })
    .index("by_run_episode", ["run_id", "episode_index"]),

  // Published descriptors omit potentially large immutable provenance so the
  // selector/history can inspect many runs without reading their payloads.
  stagePredictionCatalog: defineTable({
    run_id: v.id("stagePredictionRuns"),
    run_key: v.string(),
    dataset_repo: v.string(),
    task: v.string(),
    taxonomy_version: v.string(),
    pipeline: pipelineValidator,
    expected_count: v.float64(),
    published_at: v.float64(),
  }).index("by_repo_taxonomy", ["dataset_repo", "taxonomy_version"]),

  // Compact immutable index: publication and coverage never read raw model
  // payloads. History size cannot inflate the active coverage read budget.
  stagePredictionMembers: defineTable({
    run_id: v.id("stagePredictionRuns"),
    prediction_id: v.id("stagePredictions"),
    episode_index: v.int64(),
    content_sha256: v.string(),
    stage: v.optional(v.float64()),
    flagged: v.boolean(),
  }).index("by_run_episode", ["run_id", "episode_index"]),

  stagePredictionSelections: defineTable({
    dataset_repo: v.string(),
    task: v.string(),
    taxonomy_version: v.string(),
    run_id: v.id("stagePredictionRuns"),
    generation: v.float64(),
  })
    .index("by_repo_taxonomy", ["dataset_repo", "taxonomy_version"])
    .index("by_task", ["task"]),

  stagePredictionSelectionHistory: defineTable({
    dataset_repo: v.string(),
    taxonomy_version: v.string(),
    previous_run_id: v.union(v.id("stagePredictionRuns"), v.null()),
    run_id: v.id("stagePredictionRuns"),
    generation: v.float64(),
    selected_at: v.float64(),
  }).index("by_repo_taxonomy", ["dataset_repo", "taxonomy_version"]),

  // Append-only human stage reviews (multi-reviewer). Latest row per
  // (dataset_repo, episode_index, taxonomy_version, reviewer) wins; "cleared"
  // folds out. Gold consolidation pulls committed rows (confirmed |
  // corrected) and re-validates them with the Python validator before they
  // can become consolidated_gold.csv rows.
  stageReviews: defineTable({
    task: v.string(),
    dataset_repo: v.string(),
    episode_index: v.int64(),
    taxonomy_version: v.string(),
    status: v.string(), // confirmed | corrected | uncertain | draft | cleared
    label: v.optional(v.record(v.string(), v.any())),
    notes: v.optional(v.string()),
    review_coverage: v.optional(reviewCoverageValidator),
    event_links: v.optional(eventLinksValidator),
    prefill_pushed_at: v.optional(v.float64()), // which prefill generation was shown
    prediction_id: v.optional(v.id("stagePredictions")),
    prediction_sha256: v.optional(v.string()),
    prediction_run_id: v.optional(v.id("stagePredictionRuns")),
    copied_from_review_id: v.optional(v.id("stageReviews")),
    taxonomy_hash: v.optional(v.string()),
    blind: v.optional(v.boolean()), // reviewed with policy/arm identity hidden
    // Bounds context persists even for unresolved pre-cutover source generations.
    episode_duration_s: v.optional(v.float64()),
    reviewer: v.string(),
    reviewer_user_id: v.optional(v.id("users")),
    saved_at: v.float64(),
  })
    .index("by_repo", ["dataset_repo"])
    .index("by_repo_episode", ["dataset_repo", "episode_index"])
    .index("by_task", ["task"]),

  datasets: defineTable({
    repo_id: v.string(),
    name: v.string(),
    task: v.string(),
    source_type: v.string(), // "teleop" | "rollout" | "dagger" | "eval"
    dataset_role: v.optional(v.string()), // "aggregate_parent" | "training_view" | "eval_session" | "rollout"
    trainable: v.optional(v.boolean()),
    status: v.optional(statusValidator), // override; absent = inherit from task
    status_reason: v.optional(v.string()),
    environment: v.string(),
    num_episodes: v.optional(v.int64()),
    total_duration_seconds: v.optional(v.float64()),
    num_success: v.optional(v.int64()),
    num_failure: v.optional(v.int64()),
    num_human_frames: v.optional(v.int64()),
    num_policy_frames: v.optional(v.int64()),
    num_autonomous_success: v.optional(v.int64()),
    stats_status: v.optional(
      v.union(v.literal("pending"), v.literal("ready"), v.literal("error")),
    ),
    stats_hf_sha: v.optional(v.string()),
    stats_computed_at: v.optional(v.float64()),
    stats_algorithm_version: v.optional(v.string()),
    stats_error: v.optional(v.string()),
    stats_refresh_requested_at: v.optional(v.float64()),
    model_id: v.optional(v.string()), // programmatic policy lookup key (URI-prefixed)
    model_url: v.optional(v.string()), // human-facing link (W&B artifact/run, HF Hub, etc.)
    parent_repo_id: v.optional(v.string()),
    derived_repo_ids: v.optional(v.array(v.string())),
    mutually_exclusive_with: v.optional(v.array(v.string())),
    view_family_id: v.optional(v.string()),
    view_id: v.optional(v.string()),
    producer_model_ids: v.optional(v.array(v.string())),
    target_model_id: v.optional(v.string()),
    target_arm_key: v.optional(v.string()),
    notes: v.optional(v.string()),
  })
    .index("by_repo", ["repo_id"])
    .index("by_task", ["task"])
    .index("by_dataset_role", ["dataset_role"])
    .index("by_source_type", ["source_type"]),
});
