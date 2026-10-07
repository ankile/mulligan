# Data and checkpoints

The paper's recorded data and trained checkpoints are public on the Hugging Face organization
[`mulligan`](https://huggingface.co/mulligan): 223 dataset repositories (221 LeRobot v3.0 trajectory repos and two
simulation evaluation bundles) and 180 model repositories with 596 checkpoints. Checkpoints that dataset metadata
names but that are not released are in the `not_released` list of `release/models.json`. This page explains how the repos are named,
which revision to read, and how to check the paper's numbers against them. The Hub groups the repos in one collection
per task: [Insert Marker](https://huggingface.co/collections/mulligan/real-marker-d2-6ab2fb65f74c1c6cd807e8c4),
[Thread Nut](https://huggingface.co/collections/mulligan/real-square-d2-6ab2fb7db2157ae28cbbbbce),
[Route Cable](https://huggingface.co/collections/mulligan/real-routing-d2-6ab2fb7d730388cfe7aaa3f5),
[Square-Narrow](https://huggingface.co/collections/mulligan/simulation-square-narrow-6ab2fb7d7046e56db7bd2c72),
[Square-Broad](https://huggingface.co/collections/mulligan/simulation-square-broad-6ab2fb7def3ad8e8f0b05681).

## Pins: always read the release revision

`release/revisions.json` is the single lookup `repo -> {revision, tag}`. Every loader, config, manifest and test in
this repository reads repos at that revision, never at `main`; `tests/release/test_revision_consistency.py` fails if
any file pins a `mulligan/*` repo at another revision. The start manifests under `data/real/manifests/` keep the
commits they cite (their sha256 is the start identity).

- Every repo (223 datasets, 180 models and the paper evidence `mulligan/paper-evidence`) has one commit, tagged
  `release-1`.
- `v3.0` (LeRobot's codebase-version tag, which `LeRobotDataset` reads when no revision is given) points at the
  `release-1` commit on the 223 dataset repos.

```sh
python -m mulligan.release.download datasets --task real-marker-d2 --role evaluation --round 5 --dry-run
python -m mulligan.release.download datasets --task sim-square-narrow --method mulligan --meta-only
python -m mulligan.release.download models --task real-square-d2 --kind idql-critic --round 5
python -m mulligan.release.download repo mulligan/real-routing-d2-r00-r05-eval --include "meta/*"
```

In Python, `mulligan.release.download.pinned_revision(repo)` returns the pin, and `download(repo, include=...)` calls
`huggingface_hub.snapshot_download(repo, revision=pin)`. Reads are anonymous unless `HF_TOKEN` is set.

## Manifests

| File | Contents |
|---|---|
| `release/revisions.json` | the revision and tag of every public repo |
| `release/datasets.json`, `.csv` | one row per dataset repo: task, role, tier, variant, collection round, model rounds, parent, episodes, frames, fps, cameras; round datasets also carry their lock kind, round, sessions and counts |
| `release/models.json`, `.csv` | one row per model repo with its checkpoints (per-seed subfolders for simulation), file hashes, training and validation datasets, deployments, `retrain`, and for critics `dp_artifact_resolved`; `checkpoint` is each checkpoint's release id |
| `release/round-datasets.json` | the round-dataset lock; the public `meta/round_dataset.json` of all 15 round datasets cites its sha256 |
| `release/training-views.json` | the training-data audit of the 76 real checkpoints: per checkpoint, the selectors over public repos that give its training and validation episodes, and its `retrain` class with the `rule` behind it |
| `release/paper-results.json` | the 110 frozen result points (46 real, 64 simulation) the verifier checks |

`arena/data/` holds the Policy Arena and catalog files in the schema the site reads, including the
episode-level ancestry of the derived views (`episode-lineage-audit.json`).

## Names

All IDs are lowercase kebab case, `{task}-...`, with two-digit indexes. Task tokens: `real-marker-d2` (Insert
Marker), `real-square-d2` (Thread Nut), `real-routing-d2` (Route Cable), `sim-square-narrow`, `sim-square-broad`.
The `_d2` suffix of a real task names the generation of its physical setup.

| Kind | Pattern | Example |
|---|---|---|
| Teleop collection (original recording) | `{task}-c00-teleop-mixed` | `mulligan/real-marker-d2-c00-teleop-mixed` |
| DAgger collection (original recording) | `{task}-cNN-dagger-mixed` | `mulligan/real-routing-d2-c09-dagger-mixed` |
| Training view of a collection | `{task}-cNN-{teleop,dagger}-{variant}` | `mulligan/real-square-d2-c03-dagger-mulligan` |
| Held-out teleop view | `{task}-c00-teleop-validation` | `mulligan/real-marker-d2-c00-teleop-validation` |
| Round evaluation | `{task}-rNN-eval` | `mulligan/real-marker-d2-r05-eval` |
| Screening round evaluation | `{task}-rNN-screen` | `mulligan/real-square-d2-r05-screen` |
| Cable evaluation (all rounds, one session) | `real-routing-d2-r00-r05-eval` | |
| Single-policy view of an evaluation block | `{eval}-bNN-{method}-policy-rollouts` | `mulligan/real-square-d2-r03-eval-b01-mulligan-dp-policy-rollouts` |
| Simulation autonomous rollouts | `{task}-cNN-{arm}-policy-rollouts` | `mulligan/sim-square-narrow-c02-auto-iql-n32-policy-rollouts` |
| Simulation evaluation bundle | `{task}-r00-r03-eval` | `mulligan/sim-square-broad-r00-r03-eval` |
| Robot actor / critic | `{task}-rNN-{arm}-dp`, `{task}-rNN-mulligan-idql-critic[-screen-bNN]` | `mulligan/real-marker-d2-r05-mulligan-idql-critic` |
| Collector critic | `real-marker-d2-c05-collector-idql-critic` | |
| Route Cable velocity-action lineage | `real-routing-d2-velocity-rNN-*` | `mulligan/real-routing-d2-velocity-r05-mulligan-idql-critic` |
| Simulation agent (5 seed folders) | `{task}-rNN-{arm}-{idql,divl}` | `mulligan/sim-square-broad-r00-baseline-divl` |

`cNN` is a collection increment, `rNN` an evaluated model round, `bNN` a recorded evaluation session (block), never a
split. DAgger variants: `baseline` (HG-DAgger), `mulligan` (HG-DAgger+Mulligan, with counterfactuals), `mulligan-no-cf`
(Mulligan sampling without counterfactuals; the CF ablation), and in simulation `sobol-with-cf` / `sobol-no-cf`. R0
treatment demonstrations use `sobol`. Robot policy tokens: `baseline-dp`, `sobol-dp`, `mulligan-dp`, `mulligan-idql`,
`mulligan-no-cf-dp`.

## Roles and tiers

`role` in `datasets.json` is one of `raw-collection` (an original recording), `training-view` (a subset of a
collection that a model trained on), `validation-view`, `evaluation` (a round evaluation or its source block),
`policy-rollouts` (one policy's episodes of an evaluation block, or simulation autonomous rollouts) and
`evaluation-bundle`. `tier` is `mainline` (109 repos: the results the paper and the Arena show), `supporting` (90:
required ancestry, earlier blocks, velocity-action Cable rounds) or `ablation` (24 named same-campaign ablation views).

Derived views share episodes with their parents (`parent`, and `arena/data/episode-lineage-audit.json` per
episode), so do not concatenate a view with its parent.

**Holdout caveat.** An evaluation recording can later become training data for another model (for example the Marker
R5 critic trains on an earlier DP-versus-IQL evaluation). Whether a repo is held out depends on the consuming model:
check its `training_datasets` and `validation_datasets` in `release/models.json`, not the repo name.

## Round datasets and the 2,550 episodes

The real evaluations are grouped into one dataset per task-round (`{task}-rNN-eval`), plus the two
`*-r05-screen` sets and the single Cable evaluation. Each carries `meta/round_dataset.json` (kind, round,
sessions, counts, `lock_sha256`) and `meta/episode_provenance.parquet` (per episode: `session_id`, `policy`,
`policy_key`, `method`, `role`, `success`, source repo and index). `role` is `counted` (the episode enters the paper's
success rates), `no-cf-ablation` (the counterfactual ablation arm) or excluded (not in the repo). The paper's 2,550
evaluation episodes are the `counted` episodes of the 13 datasets whose `kind` is not `screen` (Marker 850, Nut 950,
Cable 750); adding `no-cf-ablation` gives 2,850. The two screening R5 datasets also mark their episodes `counted`
(their own pooling), so a role-only sum over all 15 gives 2,950 / 3,250.

Success rates pool every `counted` episode of one checkpoint (actor, critic, N) in the round's dataset, across
sessions; `policy_key` names the arm (`hg_dagger`, `hg_dagger_mulligan`, `hil_idql_mulligan`).

The `lock` and `plan` fields of `meta/round_dataset.json` name the files the round datasets were assembled from.
The lock ships here as `release/round-datasets.json`, with the same sha256 as the `lock_sha256` field.

## Route Cable: two round axes

The public Cable rounds R0-R5 are the source training rounds R0, R2, R4, R6, R8 and R9. Each public round after R0
merges two collection increments:

| Public round | Collection increments | Source round |
|---|---|---|
| R0 | c00 | R0 |
| R1 | c01 + c02 | R2 |
| R2 | c03 + c04 | R4 |
| R3 | c05 + c06 | R6 |
| R4 | c07 + c08 | R8 |
| R5 | c09 | R9 |

The source increments are kept as separate repos and never relabelled. **Collectors:** increments c01-c08 (public
R1-R4) were collected by the velocity-action DP lineage (the `real-routing-d2-velocity-rNN-*` models and evaluations),
not by the UMI-relative DPs that the headline plots; c09 (R5) used UMI-relative collectors. So for Cable R1-R4 the
data were not collected by the policy plotted for the previous round, and the Cable collector-success numbers for
those rounds come from the velocity-action collector evaluations. The final evaluation is one 15-policy, 750-episode
session in `mulligan/real-routing-d2-r00-r05-eval`; `policy_key` carries the public round (`hg_dagger_r3`, ...).
Its headline metric is clip progress (seats per two clips); per-episode seats come from the latest human label in
`.label_history.jsonl`. All 750 episodes are human-reviewed (final labels): an episode that seats only
the right-most (second) clip scores no first-clip seat, so the 31 episodes noted "Second clip only" carry no clip mark
(label-history tool `rule:second-clip-only`). The release reflects the final review of the original recording;
`results_eval_time.json` keeps the eval-time labels.

## Models

`release/models.json` has 58 real DP actors, 18 real IDQL critics, and 104 simulation repos with five seed folders
each (290 IDQL agents, 230 DIVL agents). For real checkpoints:

- `retrain` is `exact` when the checkpoint's training and validation episodes are reproducible from public repos with
  the selectors in `release/training-views.json`, else `approximate`: 75 exact, 1 approximate
  (`real-routing-d2-velocity-r05-mulligan-idql-critic`). The five Marker critics that trained on session b03
  of the R2 evaluation are exact: they read `mulligan/real-marker-d2-r02-eval-b03`, the session published whole
  (125 episodes), including the 25 episodes of a policy that is not released, which the round dataset
  `mulligan/real-marker-d2-r02-eval` does not hold (their source in `release/training-views.json` carries
  `round_dataset_session`).
- `dp_artifact_resolved` maps each critic's `dp_artifact` (the `hf://` id in its `metadata.json`) to the released DP
  repo and revision it loads. 17 of 18 resolve; the exception is the approximate critic above
  ([real_robot.md](real_robot.md#approximate-checkpoints)).

## Simulation evaluation bundles

`mulligan/sim-square-narrow-r00-r03-eval` (8,000 Sobol starts per evaluation) and
`mulligan/sim-square-broad-r00-r03-eval` (30,000) hold per-state outcomes (`success`, `length`, `point_idx`) of every
evaluated seed, the exact grids (`grids/<manifest hash>.json`), the evaluation index `meta/evaluations.json`, and
`meta/mainline-evaluations.json` / `meta/mainline-headline.csv` for the 160 seed results per task that the paper plots.
They contain no videos and are not LeRobot trajectories. The simulation headline is the mean over five training
seeds of the per-seed rate as stored (percent, four decimals), with a Student-t 95% interval over seeds.

## Checking the paper's numbers

```sh
python -m mulligan.release.verify_results [--paper-evidence <mirror>] [--no-sim-recount] --out report.json
```

reads only the pinned public repos and recomputes, from per-episode records, the 46 real and 64 simulation result
points, the 2,550 / 2,850 episode totals, and the 988 / 1,009 Mulligan collection successes (credited fresh
attempts in the `*-dagger-mixed` `meta/protocol_quota_ledger.jsonl`; lowest round Cable R4, 100/111 = 90.1%). It
first checks integer counts, then each stored rate and interval under the rule the paper used, and prints the
number of checks per group and a one-line summary of the recomputed values. A failed check prints a `FAIL` line
and the exit status is non-zero. It also checks the frozen paper CSVs of `mulligan/paper-evidence`, read at its
pin or from a local mirror (`--paper-evidence` or `$MULLIGAN_PAPER_EVIDENCE`). A cold full recount downloads about
0.5 GB (mostly the simulation per-state Parquet) and takes a few minutes; `--no-sim-recount` uses the bundles'
index instead.

## Licenses

The code, the datasets, the models and the paper evidence are MIT.
The simulation data were generated with robosuite and MimicGen environments; see `THIRD_PARTY_NOTICES.md` for their
terms.

## What the public `meta/` files record

Models and datasets are named by their release ids (`mulligan/<repo>`, `mulligan/<repo>/<seed>`, or
`unreleased/...` for runs and recordings that are not published). A file path names a released file
by its path in this repository (`data/real/manifests/...`, `data/sim/start_manifests/...`); paths to files
that are not released are left out.

`.label_history.jsonl` and `.outcome_edit_progress.json` are data: the
Cable clip-progress numbers are recomputed from them.
