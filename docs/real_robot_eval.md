# Blind evaluation on the real robot

The paper compares policies with paired, blinded evaluations: every policy is rolled out
once from each start of a locked held-out manifest, in a per-start shuffled order under
anonymous labels (A, B, C, ...), so the operator who resets the scene and judges the
outcome cannot tell which policy is running. This page shows how to run such an eval
with `mulligan.real.eval.manifest_eval` and how its outputs feed the analysis. Station
setup is in [station.md](station.md).

## Manifests

A manifest is a JSON list of start states (object pose, placements) with a stable
`manifest_idx`. The paper's 53 locked manifests are in `data/real/manifests/`:

```
data/real/manifests/<task>/<rNN | final | lineage>/<name>.json
data/real/manifests/INDEX.csv     path, sha256, bytes, task, round, public_round, kind, phase
```

`kind` is `collection` (blind DAgger / R0 teleop starts) or `eval` (held-out starts).
Cable manifests are numbered by source collection round; `public_round` maps them to the
released `routing_d2` rounds (R0/R2/R4/R6/R8/R9 -> R0-R5), and the 15-arm lineage eval
under `routing_d2/lineage/` covers all six. The same bytes are stored in each released
dataset as `meta/initial_states_manifest.json`.

A new held-out manifest must not reuse Sobol points of earlier manifests.
`mulligan.real.eval.eval_manifest.build_fresh_eval_manifest` draws a fresh block,
checks it against the stream registry `data/real/sobol_stream_ranges.csv` (every Sobol
range the paper's manifests use) and appends its own claim. A test checks that the registry
covers every point of every shipped manifest.

Three evaluation sessions ran on 50-start subsets of the 100-start `final_dp_vs_iql` manifests
(Marker R2 `b03`, Nut R3 `b02` and `b03`). Those subsets are not in `data/real/manifests/`;
each is stored in its eval dataset at `meta/sessions/<session>/meta/initial_states_manifest.json`,
and `configs/real/<task>/rNN_eval.yaml` points there (`manifest_in_eval_dataset`, matching
`manifest_sha256`).

Preview the operator cards of any manifest without a robot:

```bash
uv run python -m mulligan.real.operator_ui.preview \
  --manifest data/real/manifests/square_d2/r05/square_d2_r5_eval_heldout_independent_sobol.json \
  --idx 0 --idx 1 --out-dir /tmp/cards
```

## Running an eval

```bash
uv run --project robot --frozen python -m mulligan.real.eval.manifest_eval \
  --environment square_d2 \
  --initial-states-manifest data/real/manifests/square_d2/r05/square_d2_r5_eval_heldout_independent_sobol.json \
  --fixed-policy baseline=hf://mulligan/real-square-d2-r05-baseline-dp \
  --fixed-policy mulligan=hf://mulligan/real-square-d2-r05-mulligan-dp \
  --dataset-name square-d2-r5-heldout --hf-repo-id <user>/square-d2-r5-heldout \
  --random-seed 2026070901 --max-steps 300
```

- `--environment` is the task name stored in the dataset (`marker_d2`, `square_d2`,
  `routing_d2`); it must match the manifest's `task`.
- Each `--fixed-policy NAME=MODEL_ID` is one arm. `MODEL_ID` names its source: `hf://<repo>`
  for a released checkpoint (`release/models.json` lists them) or a local checkpoint directory.
  The same checkpoint may appear under two names with different settings
  (`--fixed-policy-num-action-samples NAME=N` for Vision-IDQL arms;
  `--fixed-policy-dp-override NAME=DP_MODEL_ID` makes an IDQL critic
  re-rank a different DP actor).
- Before the first rollout, every round plan (the anonymous slot order of every start,
  seeded by `--random-seed` + round) is written to `results.json`. All scheduled policies
  are loaded once and kept resident.
- The operator sets up the target shown on the card, presses a key, and marks the
  outcome: `1` success, `9` failure, `0` timeout (`--no-auto-timeout` disables the
  automatic timeout at `--max-steps`). Tasks with sub-goals (Route Cable: first clip
  seated) take a live sub-goal mark with `g` / numpad `3`. `r` re-homes the robot, `q`
  ends the session.
- Episodes are saved to a LeRobot dataset (`--dataset-path`/`--dataset-name`) with
  `policy_id`, `round_id` and the start state per frame, and pushed to `--hf-repo-id` at
  the end unless `--no-push`. Console results stay blinded until every round of the
  manifest is complete.

`--inference-backend remote` moves policy inference to another GPU machine
([station.md](station.md#remote-inference-optional)).

### Resuming and phased evals

Re-running the same command resumes: completed rounds are skipped and a partially run
round continues with its pre-baked labels. A resume that finds an incomplete round before
the last recorded one stops unless `--rerun-incomplete-rounds` is given (for example after
removing a bad rollout record by hand). If the resume guard refuses a dataset whose episodes and
`results.json` disagree (a gap after a crash, a junk tail, or rounds to redo),
`python -m mulligan.tools.repair_eval_dataset --dataset-path ./data/<name>` repairs it
(`--keep-episodes N` truncates the tail, `--drop-rounds` / `--drop-episodes` remove specific ones);
then resume with `--rerun-incomplete-rounds`.

A phased eval collects some arms first and the rest later on the same starts:

1. Phase 1: pass every final arm, retire the late ones with `--drop-fixed-policy NAME`
   (repeatable), optionally cap it with `--stop-after-round N`.
2. Phase 2: same command without the drops, plus `--rerun-incomplete-rounds`. The earlier
   rounds get only their missing slots, in the pre-baked order and labels; later rounds
   run every arm. `--progress-total-all-arms` shows progress against the full design.

Every rollout record carries the `visit_id` of the invocation that collected it, and every
graceful shutdown appends a `phase_stops` entry, so the data says which records of a
start were collected in which physical visit. `tests/real/test_manifest_eval_phased_simulation.py`
replays the three phases of the Cable 15-arm lineage eval against the real scheduling and
results code.

## results.json

| Key | Content |
|---|---|
| `args` | the command-line arguments (tokens redacted) |
| `summary` | per policy: `policy_id`, `model_id`, `name`, rounds, successes, success rate |
| `rollouts` | per rollout: `round`, `policy_id`, `model_id`, `anonymous_label`, `outcome`, `num_steps`, `episode_index`, `manifest_idx`, start pose, `subtask_frames`, `visit_id` |
| `round_plans` | per round: `manifest_idx`, start pose, and `policy_order` (slot, label, policy) |
| `phase_stops` | one entry per graceful shutdown (visit id, retired arms, cap, record keys) |

The file is rewritten atomically after every durable episode; a record is added only
after its episode's parquet footer is written, so `results.json` never lists an episode
that is not on disk.

## After the eval

1. **Outcome review.** `python -m mulligan.tools.outcome_review --repo-id <eval repo>
   --filter all --push` opens the cv2 editor to correct outcomes and mark outcome frames
   (and sub-goal frames); decisions go to `.outcome_edit_progress.json` and
   `.label_history.jsonl`, and the reconciled `results.json` is pushed. `--apply-overlay`
   applies decisions captured elsewhere without a display.
2. **Held-out ingest.** `mulligan.real.lifecycle.heldout_eval` reads `results.json` + the
   outcome record, checks them against the frame labels, pairs arms by `manifest_idx`, and
   writes per-arm Wilson intervals, paired deltas with bootstrap intervals, exact McNemar tests
   (sign-flip permutation for graded scores) and the result figure:

   ```bash
   python -m mulligan.real.lifecycle.heldout_eval <eval repo> --out outputs/real/heldout/<name>
   ```

   Your own eval repo is read at Hub `main` (`--revision` to pin a commit). A released
   `mulligan/*-eval` dataset is read at its pin in `release/revisions.json`; its `results.json`
   records the source repo id of the session, which the release manifests map back to it (the
   source id is accepted as input too). The round datasets that merge several sessions
   (`mulligan/real-marker-d2-r02-eval`, `real-square-d2-r03-eval` and the two `*-r05-screen`)
   keep each session's files under `meta/sessions/<bNN>/`: pass `--session bNN`. The CLI takes
   the arms of the session from `release/round-datasets.json` (or every arm of your
   `results.json`), the start manifest from the dataset's `meta/initial_states_manifest.json`
   and applies `.outcome_edit_progress.json` when present (`--no-outcome-record` skips it; Nut
   R5's `results.json` already carries its record); `--arm NAME[=PREFIX]` (first =
   baseline), `--pair A:B` and `--bootstrap-seed` override the defaults. Naming the arms with
   the paper's column prefixes and seed gives the paper's paired-round table, e.g. Nut R2
   (byte-identical to `paired_round_outcomes.csv` in the paper evidence; 18/50 vs 27/50,
   McNemar p = 0.078):

   ```bash
   python -m mulligan.real.lifecycle.heldout_eval mulligan/real-square-d2-r02-eval --out outputs/real/heldout/square_d2_r02 \
       --arm baseline_r02_dp=baseline --arm mulligan_no_cf_r02_dp=mulligan_no_cf \
       --arm mulligan_r02_dp=mulligan_with_cf --bootstrap-seed 20260629
   ```

   For sub-goal scoring or custom labels, build a `HeldoutEvalConfig` and call `run`.
   `python -m mulligan.real.lifecycle.routing_d2_lineage` rebuilds the six Cable rounds from
   `mulligan/real-routing-d2-r00-r05-eval` with sub-goal scores (human review where an episode
   was reviewed, else the eval-time label; `mulligan.real.lifecycle.pinned_eval_snapshot`); its
   `heldout_config` is a complete configuration (arms, labels, manifest and file pins).
3. **Per-policy views.** `python -m mulligan.real.eval.split_policies` writes one LeRobot
   dataset per policy (for training on a policy's own rollouts), with lineage metadata.
4. **Stage labels** (optional, Gemini, `stage-labeling` extra):
   `python -m mulligan.real.stage_labeling.prepare_events` writes the per-episode gripper events,
   `python -m mulligan.real.stage_labeling.label` runs the labeler and
   `python -m mulligan.real.stage_labeling.apply_cascade` the task refinements; see
   `mulligan/real/stage_labeling/DESIGN.md`.
5. **Stage eval battery.** `python -m mulligan.real.stage_labeling.stage_eval_battery --task <task>
   --labels-csv <run>/labels_joined.csv --paired-rounds-csv <ingest data dir>/paired_round_outcomes.csv
   --plot-dir DIR --csv-dir DIR --prefix NAME` writes per-arm stage shares and the
   `*_rung_conversions.csv` tables that the per-stage results of the paper (App. B.7) read.
