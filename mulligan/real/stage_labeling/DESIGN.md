# VLM Stage-Labeling Pipeline

Auto-label the progress **stage** (S0..SN) of every episode in a real-robot rollout
dataset by prompting a video VLM (Gemini; the paper's labels used `gemini-3.5-flash`), then
calibrate the prompt against human-corrected labels. The paper's substage panels use the
**frozen** labels that ship with the paper data; re-running the labeler will not
reproduce them byte for byte (the model drifts, and reviewed corrections are human
decisions). This package is what a new campaign uses to label its own episodes.

## Package layout

```
mulligan/real/stage_specs/      static task specs (no Gemini code; the paper imports these)
  tasks.py               StageLabelTaskSpec + registry (get_label_task_spec), taxonomy pins
  ladder.py              StageLevel / StageLadder
  prompts.py             PromptNode / PromptLibrary (base + append-only ruling DAG)
  schema.py              SchemaField + build_schema_fields (pure)
  sensor_constraints.py  declarative proprioceptive physics caps
  marker_d2.py, square_d2.py, routing_d2.py   the three paper tasks
  data/                  marker_prompts.json, marker_enums.json (marker prompt chain and enums),
                         marker_d2_heldout_calibrations.json (held-out eval label overlays)

mulligan/real/stage_labeling/   the labeling machinery
  events.py / prepare_events.py  per-episode gripper events CSV (CLI: prepare_events)
  assets.py              media builders (clips, stills, crops, montages) + 1 Hz gripper trace
  labeler.py             Gemini engine: schema-constrained generation, retry, checkpoint/resume
  genai_schema.py        responseSchema materialization (lazy google.genai import)
  label.py               backbone labeler CLI
  consensus.py           self-consistency fold + adjudication flags (never overwrites)
  cascade.py             focused refinement nodes (marker_d2 H2 / T)
  cascade_pipeline/      marker_d2 and square_d2 cascade runners (+ shared IO)
  apply_cascade.py       cascade CLI over a backbone run
  eval_battery.py / stage_eval_battery.py   stage-share + rung-conversion plots (CLI)
```

`stage_specs` is a sibling package, not part of `stage_labeling`, so the paper and the
lifecycle tools can read ladders and enums without pulling in `google.genai`, ffmpeg or
the labeler (`tests/real/test_stage_specs_golden.py` checks the import boundary).

## The pipeline

1. **Events CSV** (`prepare_events`): gripper close/release from
   `observation.state.gripper_position` (rising 0.2 threshold = jaw close; reverse crossing
   or a sustained final decline = reopen), clipped to the policy phase (`num_steps` from
   the canonical, outcome-edited `results.json`; teleop collections use the full
   recording). A recorded real episode keeps running through the operator's physical
   reset, and a re-open in that tail is not a policy release, so the policy boundary is
   required.
2. **Backbone labels** (`label`): N self-consistency samples per episode. The labeler is
   *blind*: it sees only the videos and the proprioceptive trace, never human priors, so
   reviewed labels remain a held-out accuracy estimate. Physics caps (jaws never closed =>
   cap S1; never reopened => no release) are applied per sample.
3. **Consensus + adjudication** (`consensus`): majority vote over samples; disagreements
   with proprioception and outcome priors are flagged into a review queue, never
   overwritten.
4. **Cascade** (`apply_cascade`): focused boundary nodes re-decide only contested
   sub-judgments (marker_d2 held-at-holder S3/S4; square_d2 grasp subtype, transport
   boundary, endpoint action, peg arrival), each behind a vote-fraction override gate.
5. **Evaluation** (`stage_eval_battery`): stage shares, rung conversion and paired
   McNemar tests per arm.

The human correction UI and the scoring tools used to calibrate the prompts are not part
of the release; reviewed labels ship as data.

### Gemini route

`GEMINI_API_KEY` in the environment selects the Gemini Developer API (the default);
`MULLIGAN_GEMINI_ROUTE=vertex` with `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`
selects Vertex AI with application-default credentials. Nothing in the code carries a
key or a project.

### Few-shot episodes are repository-local

Focused-node few-shot episode indices are identities in one dataset. The marker_d2 H2/T
bank belongs to the Marker R0 held-out eval (`mulligan/real-marker-d2-r00-eval`);
applying that cascade to another dataset requires `--exemplar-dataset-repo-id` (the R0
repo) and a separate `--exemplar-build-dir`, and the runner fails before labeling if
either is missing. The square_d2 anchors are rendered from the Nut R1 eval
(`mulligan/real-square-d2-r01-eval`) into `outputs/real/stage_labeling_anchors/`.

The marker_d2 spec also carries human-reviewed calibrations for exactly two eval
datasets (Marker R0 and R1 held-out evals). They are gated by dataset id and never apply
elsewhere.

## Per-task data contract

- **Dataset + cameras.** An HF LeRobot dataset (`dataset_repo_id`) with two role-named
  cameras (`side_camera_key`, `wrist_camera_key`; routing labels its second side view as
  `SIDE-2` and adds `wrist_left` as an extra stream). Clips are cut from each camera's
  chunk MP4 at the episode's `from_timestamp` for `episode_length / fps` seconds.
- **Events CSV.** Mandatory columns: `episode_index`, `episode_length`,
  `gripper_hold_time_s`, `gripper_release_time_s`, `gripper_reopened_at_end`; optional
  summarizer context: `policy_short`, `original_outcome` and the task's state columns.
- **Scoring contract.** A stage ladder with a full-success rung, a failure-mode enum
  (must include `none`), a final-state enum, the response-schema event fields, the prompt
  library, and the physics caps. Each vocabulary is fingerprinted in
  `stage_specs/taxonomy_pins.json`; editing a ladder or enum without bumping
  `taxonomy_version` fails at import.

## The three abstractions

1. **Prompts as a DAG of rulings.** Each variant is a base (`parent=None`) or a delta
   appended to a parent; `assemble(variant)` concatenates the chain byte for byte. Each
   delta is a ruling for one reviewed error pattern; marker_d2 and square_d2 prepend a task context block to their base chain and append
   their own rulings.
2. **Schema derived from the spec.** `build_schema_fields(spec)` assembles
   `episode_index` + the ladder-bounded stage + the task event fields + the two enums +
   the shared `confidence` / `needs_human_review` / `notes` trailer, so the schema cannot
   drift from the ladder.
3. **Physics caps as declarative rules.** An ordered `tuple[SensorConstraintRule]` folded
   over each parsed label; the fold is task-agnostic, the rules are task data.

Prompts, schema, fingerprints and a sensor-cap battery of the three paper tasks are
pinned by golden outputs in `tests/real/test_stage_specs_golden.py`.

## Onboarding a new task

1. Register a lifecycle `RealTaskSpec` (`mulligan/real/lifecycle/tasks.py`).
2. Write a `StageLabelTaskSpec` module in `mulligan/real/stage_specs/` (ladder, enums,
   event fields, a base prompt, physics caps built from the shared factories in
   `sensor_constraints.py`), import it in the package `__init__`, and add its taxonomy
   fingerprint to `taxonomy_pins.json`.
3. Generate the events CSV with `prepare_events`, run `label` on ~16 episodes, review
   them by hand, and grow the task's ruling list from the residual errors.

## Known limitations

- **Impossible-final-state retention.** The never-closed invalidation only resets
  `*_held` / `*_released` final states; an `*_in_gripper_*` state survives a never-closed
  gripper. The released marker labels were produced with this rule.
- **Factory preconditions are conventions.** The shared caps assume final-state
  vocabularies use `_held` / `_released` suffixes and that the success rung sits one
  above the seated-held rung. They are validated at registration (field references and
  emitted final states must exist), but a task with a different ladder shape should
  supply its own caps.
