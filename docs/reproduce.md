# Reproducing each experiment

This page goes through the paper's experiments one by one: the command that rebuilds each figure or
table, the command that recomputes its numbers, how to retrain or re-run it, and what that costs.
The tiers are the ones in the [README](../README.md#what-you-can-reproduce):

1. **Rebuild** the figure or table from frozen inputs (no GPU, no accounts).
2. **Recompute** its numbers from the public Hugging Face release.
3. **Retrain / re-evaluate** from released datasets and checkpoints (GPU).
4. **Re-run the protocol**: a new collection or robot evaluation. The results will differ from the paper's.

IDs (F1-F9, N, A1-A34) are the row IDs used throughout the release documentation. Figure names are
entries of the figure registry (`python -m paper.figures --list`); table files are the outputs of
`python -m paper.appendix.build` in `paper/build/tables/`. Commands run from the repository root after
`uv sync --frozen` (prefix them with `uv run` or activate `.venv`).

## Setup for tier 1

```bash
python -m paper.figures                                  # all 42 figure files -> paper/build/figs
python -m paper.appendix.build --check                   # 16 generated + 15 authored tables; checks everything
python -m paper.figures --check                          # figure bytes vs paper/figures_manifest.json (32 entries)
```

Most tier-1 rows read the frozen appendix inputs from the `mulligan/paper-evidence` dataset at its pin
(downloaded on first use; `MULLIGAN_PAPER_EVIDENCE` points at a local mirror instead). Offline and without a
mirror, the builders stop with an error that names the variable; `python -m paper.figures --keep-going` builds
what it can (Fig. 2, Fig. 3, the Fig. 4 reset card, the App. C initial-state photos and the evaluation-protocol
schematic), reports the rest and
exits non-zero, and `python -m paper.figures --list` shows which entries need the evidence. These rows
also rebuild without it: N (`paper.stats.count_eval_episodes`), A3, A4 (`paper.stats.sim_welch_table`), A5
(`paper.appendix.paper_side.divl_comparison`), A18, A29 and A30/A33, and the numbers printed by
`paper.stats.final_round` and `paper.stats.sim_contrasts`. Every other tier-1 command below needs it. Run `paper.figures` (or `paper.appendix.build` without
`--check`) before the appendix `--check`: the check compares figure files it expects to exist.
Details, platform notes and the pixel tolerances of `--check` are in [paper_figures.md](paper_figures.md).

To rebuild one figure: `python -m paper.figures --only <name> [<name> ...]`. The numbers quoted in
the text are printed by `python -m paper.stats.<module>`.

## Setup for tier 2

```bash
python -m mulligan.release.verify_results --out report.json                 # public HF only
python -m mulligan.release.verify_results --paper-evidence "$MULLIGAN_PAPER_EVIDENCE" --out report.json
```

The verifier reads the pinned public repos anonymously and recomputes, from per-episode records, the
110 result points of `release/paper-results.json` (46 real, 64 simulation), the 2,550 / 2,850 evaluation
episode totals and the 988 / 1,009 collection successes. With the evidence mirror it also checks the
frozen headline and collection CSVs the figures are drawn from. A cold run downloads about 0.5 GB and
takes a few minutes; it prints a line per stage and a one-line summary at the end. `--no-sim-recount` uses
the bundle indexes instead of the per-state Parquet. See [data.md](data.md#checking-the-papers-numbers) for what each check compares.

## Setup for tiers 3 and 4

- Simulation: `configs/sim/recipes.json` holds the 58 training recipes and the RLPD and HiL-SERL
  baseline recipes (each with the `paper` results it feeds) and the R0-R3 round specs;
  `scripts/sim/{train_cell,train_divl_heads,eval_cell,collect_round,split_round}.sh` run them
  ([sim.md](sim.md), [configs/sim/README.md](../configs/sim/README.md)). List recipe ids with
  `python -m mulligan.sim.recipes list`.
- Real robot: `configs/real/<task>/` holds one config per released checkpoint
  (`rNN_<arm>_dp.yaml`, `rNN_critic.yaml`; its `paper` block names the figure, round, arm and evaluation
  sessions) and per round evaluation (`rNN_eval.yaml`); train with
  `python -m mulligan.real.train.launch <config> --output-dir DIR` ([real_robot.md](real_robot.md),
  [configs/real/README.md](../configs/real/README.md)).
  Collection and evaluation need a robot station ([station.md](station.md),
  [real_robot_eval.md](real_robot_eval.md)).
- Baselines: the RLPD and HiL-SERL recipes train with `scripts/sim/train_cell.sh`; the HiL-SERL operator
  session uses `scripts/hilserl/*.sh` ([baselines.md](baselines.md)).
- Budgets: [compute.md](compute.md). Sim training needs an Ampere or newer GPU (bf16 autocast).

## Main text

### F1: teaser (Fig. 1, `fig:overview`)

- Rebuild: `python -m paper.figures --only overview_teaser` (needs Chrome and Ghostscript;
  the teaser's bytes change with every Chrome build, so `--check` only tests that it exists).
- Recompute: none. The 80-state bucket counts are digitized from a raster
  (`paper/appendix/productivity/README.md`).
- Retrain: none.

### F2: state-based RLPD vs HiL-IDQL+Mulligan vs HiL-SERL (Fig. 2, `fig:intro-rlpd`)

- Rebuild: `python -m paper.figures --only sim_state_rlpd_vs_mulligan_compact`, from the frozen learning
  curves in `paper/data/online_rl/`.
- Recompute: none; the curves are the paper runs' eval logs.
- Retrain RLPD (the paper's 84% Square-Narrow and 0% Square-Broad curves, and the released-demo curves),
  seeds 1-5 each:

  ```bash
  scripts/sim/train_cell.sh square-narrow-rlpd    # also square-broad-rlpd, square-narrow-rlpd-robomimic-ph, square-broad-rlpd-mimicgen-core
  ```

  The paper's RLPD runs used robosuite 1.4.1; this release's simulator is robosuite 1.5.2, so a re-run
  can differ slightly.
- Re-run HiL-SERL (94% and 2%): a no-human run, a fork at 150k (Square-Narrow; Square-Broad at 200000),
  then the learner, eval watcher and actor on the fork with an operator at a SpaceMouse. The no-human
  half is autonomous (tier 3); the operator half is tier 4.

  ```bash
  scripts/sim/train_cell.sh square-narrow-hilserl
  scripts/hilserl/fork.sh outputs/sim/train/square-narrow-hilserl/hilserl_agent/seed-1 outputs/hilserl/narrow_fork150k 150000
  scripts/hilserl/learner.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
  scripts/hilserl/eval_watcher.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
  scripts/hilserl/actor.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k <learner host>
  ```

  [baselines.md](baselines.md) gives a few-minute smoke configuration for both.
- The HiL-IDQL+Mulligan reference curve comes from the sim headline (F5, simulation).
- Compute: RLPD per seed on one GPU: 2.8-4.5 h for the 300k-step Square-Narrow recipes, 8.1-13.9 h for
  the 1M-step Square-Broad recipes; about 180 GPU-h for all six recipes x 5 seeds (including A18).
  HiL-SERL: one GPU for the learner, CPU for the eval watcher, plus operator time.

### F3: task sequences (Fig. 3, `fig:task-sequences`)

- Rebuild: `python -m paper.figures --only task_sequences` (downloads three released episodes' videos
  from HF; network).
- Numbers: `python -m paper.stats.r0_demo_durations` prints the median round-0 demonstration lengths
  (12.7 / 10.1 / 18.6 s real, 8.3 / 8.6 s simulation; rounded half up) from the public
  `mulligan/*-c00-teleop-mixed` metadata (network).
- Each row is the median-length success of the final-round HiL-IDQL+Mulligan arm on that task's held-out
  block (Nut: `mulligan/real-square-d2-r05-eval` episode 136).

### F4: reset card and reset ranges (Fig. 4)

- Rebuild: `python -m paper.figures --only real_world_square_d2_reset_card reset_ranges`. The
  overlays draw on photo composites pinned in the paper evidence; the reset ranges are
  `mulligan/real/lifecycle/tasks.py`.
- Recompute / retrain: not applicable.

### F5 (real): headline over rounds, Marker, Nut, Cable (Fig. 5 top, `fig:real-world-headline`)

- Rebuild: `python -m paper.figures --only headline_real_sim`.
- Numbers: `python -m paper.stats.final_round` (+34 / +10 / +16 pp at R5); the round-pooled Cable
  test (p = 0.04 in the main text) is the Cable row of `tab:appendix-campaign-mcnemar` (A1, A2, A6).
- Recompute: `verify_results` recounts all 46 real points (success counts, then Wilson intervals) from
  `meta/episode_provenance.parquet` of the round datasets; Cable seats come from `.label_history.jsonl`.
- Retrain: 58 DP actors and 18 IDQL critics, one config each:

  ```bash
  python -m mulligan.real.train.launch configs/real/marker_d2/r05_mulligan_dp.yaml --output-dir out/marker_r05_dp
  python -m mulligan.real.train.launch configs/real/marker_d2/r05_critic.yaml --output-dir out/marker_r05_critic
  ```

  75 checkpoints are `exact` (the public repos hold exactly their training episodes) and one is
  `approximate` (`real-routing-d2-velocity-r05-mulligan-idql-critic`, [real_robot.md](real_robot.md#approximate-checkpoints)).
- Re-run (tier 4): blind paired evaluation with `mulligan.real.eval.manifest_eval`, using the arms of
  `configs/real/<task>/rNN_eval.yaml` and the locked held-out manifest its `manifest:` field names. For
  Marker and Nut that is `data/real/manifests/<task>/rNN/`. Cable's manifests are under
  `data/real/manifests/routing_d2/rNN/` with the source round numbers (public R0-R5 are source
  R0/R2/R4/R6/R8/R9), and all six public rounds were evaluated on the 15-arm lineage manifest in
  `data/real/manifests/routing_d2/lineage/`. `data/real/manifests/INDEX.csv` lists
  every manifest with its task, round, public round and kind ([real_robot_eval.md](real_robot_eval.md)).
- Compute (logged runs behind the released curves, App. G.7, one GPU each): DP actors median
  3.2 h (Marker), 4.3 h (Nut), 3.1 h (Cable), range 1.7-7.9 h; critics median 5.3 h, 6.6 h and 3.7 h,
  range 1.2-10.3 h (`paper/data/sim/training_compute/summary.json`). Each robot evaluation is 50 starts
  per policy with a human resetting the scene.

### F5 (simulation): DAgger rounds and failure log (Fig. 5 bottom, `fig:sim-headline-dagger-rounds`)

- Rebuild: `python -m paper.figures --only headline_real_sim` (reads `paper/data/sim/partial_headline.csv`).
- Recompute: `verify_results` recounts the 64 simulation points (5 seeds each) from the per-state
  Parquet of the public eval bundles `mulligan/sim-square-{narrow,broad}-r00-r03-eval`.
- Retrain and re-evaluate: every cell is a recipe (`python -m mulligan.sim.recipes list`). For one cell
  and seed:

  ```bash
  scripts/sim/train_cell.sh square-narrow-r01-mulligan --seed 1                       # IDQL agent
  scripts/sim/train_divl_heads.sh square-narrow-r01-mulligan --seed 1 --parent local   # DIVL head
  scripts/sim/eval_cell.sh square-narrow-r01-mulligan --seed 1 --checkpoint local      # 8,000-start grid
  ```

  `eval_cell.sh` without `--checkpoint` evaluates the released checkpoint. Its default stage and N are
  what the headline reports (DIVL head at N=32 for the human-in-the-loop arms, the IDQL agent for the
  autonomous baselines); for the N=1 points of the human-in-the-loop arms pass
  `--stage idql_agent --num-action-samples 1`. Evaluations are unseeded by default; `--eval-seed S
  --env-seed S` seeds them ([sim.md](sim.md#grid-evaluation)).
- Re-run the rounds (tier 4): `scripts/sim/collect_round.sh dagger --task T --round N` with a SpaceMouse
  operator, or `--operator replay` to replay the recorded human segments headless (this reproduces the
  protocol, not the data); then `split_round.sh` and `python -m mulligan.sampling.sim_design` for the next
  round's starts ([sim.md](sim.md#one-round-step-by-step)).
- Compute: per seed, about 1.9 GPU-h (Square-Narrow) or 3.0 GPU-h (Square-Broad) for the agent and
  0.65 / 0.97 GPU-h for the DIVL head; about 900 GPU-h for all 58 recipes x 5 seeds. A full 8,000-start
  Square-Narrow evaluation takes a few GPU-hours; the 30,000-start Square-Broad grid runs as four shards
  ([compute.md](compute.md)).

### F6: throughput (Fig. 6, `fig:real-world-unit-throughput`)

- Rebuild: `python -m paper.figures --only real_world_success_throughput_headline`.
- Recompute: partial. The success counts are the F5 points that `verify_results` recounts; the episode
  durations are read from the frozen `real_results` evidence. The released evaluation datasets carry
  every episode's length, but no release command re-derives throughput from them.
- Retrain / re-run: as F5 (real).

### F7: component ablations (Fig. 7, `fig:component-ablations`)

- Rebuild: `python -m paper.figures --only ablations_sim_real`.
- Numbers: `python -m paper.stats.sim_contrasts` (actor data -3.36 pp, Welch [-6.53, -0.18]; sampler
  cells 85.73 / 92.69 / 93.52; R0 Sobol +8.23 / +3.53 pp).
- Panels 1-2 (simulation state sampling, actor data):
  - Recompute: partial. The per-seed success counts of these cells are in the eval bundles and
    `verify_results` checks them against `release/models.json`; it does not re-derive the plotted CSV.
  - Retrain: recipes `square-narrow-r01-{baseline,mulligan-no-cf,mulligan}` (state sampling) and
    `square-narrow-r01-baseline-straddled-auto-success` (actor data), the four plotted bars
    (`python -m mulligan.sim.recipes list --paper "Fig. 7"`), with the F5 simulation commands. The R0 Sobol
    numbers quoted with the figure come from `square-{narrow,broad}-r00-{baseline,sobol}`;
    `square-narrow-r01-sobol-{no-cf,with-cf}` feed A8 and the printed `sim_contrasts` numbers, not the figure.
- Panels 3-4 (real counterfactual ablation, Marker R1-R2, Nut R1-R4):
  - Recompute: none (the no-CF arm is not among the verifier's points).
  - Retrain: `configs/real/{marker_d2,square_d2}/rNN_mulligan_no_cf_dp.yaml`.
  - Re-run: robot evaluation as F5 (real).

### F8: collection success (Fig. 8, `fig:dagger-collection-success`)

- Rebuild: `python -m paper.figures --only real_world_dagger_collection_success_square`.
- Numbers: `python -m paper.stats.print_collection_totals` (988/1,009 = 97.9%, lowest round Cable R4
  100/111 = 90.1%; reads the paper evidence).
- Recompute: `verify_results` rederives 988 / 1,009 from `meta/protocol_quota_ledger.jsonl` of the 19
  public real `*-dagger-mixed` datasets.
- Re-run: `python -m mulligan.real.collect.blind_dagger` ([real_robot.md](real_robot.md#collect-and-evaluate)).
  Cable R1-R4 were collected by the velocity-action DPs, not by the plotted policies
  ([data.md](data.md#route-cable-two-round-axes)).

### F9: intervention burden (Fig. 9, `fig:intervention-burden`)

- Rebuild: `python -m paper.figures --only collection_burden_real_sim`.
- Recompute: none; the burden counts come from the frozen `component_panels` evidence.
- Re-run: the real collectors and the sim DAgger collector record interventions per episode.

### N: 2,550 held-out evaluation episodes (abstract, introduction)

- Rebuild: `python -m paper.stats.count_eval_episodes` (offline, from `release/round-datasets.json`).
- Recompute: `verify_results` sums the `counted` episodes of the 13 non-screen round datasets from
  their public `meta/round_dataset.json` and `meta/episode_provenance.parquet` (2,850 with the no-CF
  ablation arms). [data.md](data.md#round-datasets-and-the-2550-episodes) explains why the two screening
  R5 datasets are left out.

### Alg. 1 and Alg. 2: the improvement loop and start selection (Sec. 3)

- Simulation start selection (Alg. 2) rebuilds the locked R1-R3 collection starts of both tasks bit for
  bit, offline, in seconds:

  ```bash
  python -m mulligan.sampling.sim_design --config configs/sim/square_narrow/r01_sampler.yaml --out out/narrow_r01 --check
  ```

  (six configs: `configs/sim/{square_narrow,square_broad}/r0{1,2,3}_sampler.yaml`).
- Real start selection: `python -m mulligan.sampling.real_design --config configs/real/<task>/rNN_sampler.yaml
  --inputs-root ROOT --out FILE` for Marker and Nut. A round's inputs include the previous round's
  evaluation outcome table and stage labels, which are not released, so a round cannot be rebuilt end to
  end from the release; `tests/real/test_real_design.py` rebuilds the locked Marker R1 manifest from trimmed
  copies of its recorded inputs. The locked manifests themselves are in `data/real/manifests/`.
- Best-of-N and the critic: `mulligan.agents.idql` (simulation) and `mulligan.real.policy.vision_idql`
  (real); `tests/real/test_bon_equivalence.py` checks the real path against recorded reference outputs.
- The loop itself is the tier-4 pipeline of F5: collect, split, design the next starts, train, evaluate.

## Appendix

### A1, A2, A6: real results in detail, statistics tables, speed and throughput

- Rebuild figures: `python -m paper.figures --only real_world_headline_mulligan_vs_baseline_detailed
  real_world_success_speed real_world_success_throughput`.
- Rebuild tables: `python -m paper.appendix.build` writes `real_results_final_round.tex`,
  `real_results_campaign.tex` and `real_results_routing_progress.tex` (`tab:appendix-final-round`,
  `tab:appendix-campaign-mcnemar`, `tab:routing-d2-progress`) from the frozen `real_results` evidence.
- Recompute: the per-point counts and intervals of A1 are the 46 real points of `verify_results`; the
  paired tests and the speed/throughput numbers are recomputed from the frozen evidence only.
- Retrain / re-run: as F5 (real).

### A3: Cable snapshot and endpoint tests

- Rebuild: `python -m paper.stats.routing_d2_verify` (checks the 750 outcomes and source hashes of the
  frozen Cable snapshot in `paper/data/real/routing/` and recomputes the R5 endpoint tests, +16 pp,
  p = 0.077).
- Recompute: the 15 Cable points are among the `verify_results` points (clip seats from the latest human
  label in `.label_history.jsonl`).

### A4, A5, A8: simulation significance, DIVL frozen-actor comparison, sampler ablation

- Rebuild: `python -m paper.stats.sim_welch_table` (`sim_significance_table.tex`),
  `python -m paper.appendix.paper_side.divl_comparison` (`divl_frozen_actor_table.tex`) and
  `python -m paper.appendix.build` (`sampler_ablation_table.tex`).
- Recompute: partial, as F7 panels 1-2 (per-seed counts in the eval bundles).
- Retrain: the DIVL heads of the 46 recipes that have one
  (`python -m mulligan.sim.recipes list --with-divl`), with `train_divl_heads.sh` and `eval_cell.sh`.

### A7: simulation speed and throughput

- Rebuild: `python -m paper.figures --only sim_success_speed sim_success_throughput`.
- Tier 1 only (archived per-seed summary).

### A9: Marker and Nut counterfactual ablation

- Rebuild: `python -m paper.figures --only real_world_cf_ablation`.
- Retrain: `configs/real/{marker_d2,square_d2}/rNN_mulligan_no_cf_dp.yaml`. Re-run: robot evaluation.

### A10, A15: substage success and stage labels

- Rebuild: `python -m paper.figures --only real_world_marker_d2_substage real_world_square_d2_substage`.
- The stage labels are frozen Gemini outputs. The labeler ships (`mulligan/real/stage_labeling/`,
  `stage-labeling` extra, `GEMINI_API_KEY`; see `DESIGN.md` there), but a new labeling run gives
  different labels.

### A11, A12, A13: collection success, burden, data ledger and composition

- Rebuild: `python -m paper.figures --only real_world_dagger_collection_success
  real_world_dagger_collection_success_detailed real_world_burden_progression real_world_data_composition`
  and `python -m paper.appendix.data_ledger.prepare` (`data_ledger.tex`, `data_composition.tex`).
- Recompute: the Mulligan collection totals as F8. The Cable increments table
  (`tab:cable-collection-increments`) is authored and restored byte for byte.
- Re-run: the collectors, as F8.

### A14, A26, A27, A28: authored tables and illustrations

- Rebuild: `python -m paper.appendix.reference_data.prepare --check` compares the authored table bodies
  (task summary, initial-state ranges, scoring, stage ladders, CF counts, hyperparameters, prior work)
  with the manuscript, and
  `--restore-assets` restores the funnel, operator-card and state-evolution illustrations.
- These are authored, so rebuilding them only proves that the archived bytes are unchanged. The
  hyperparameter tables (A27) describe the configs in `configs/sim/recipes.json`, `configs/real/` and
  `mulligan/real/lifecycle/tasks.py`; see "Where the configs and the manuscript differ" below.

### A16: RECAP

- Rebuild: `python -m paper.appendix.build` (`reference_recap.tex`, the locked held-out grid readout of
  each seed's final checkpoint, from the per-seed records).
- Retrain: not possible from this release. The RECAP learner and its recipe are not released; its training
  datasets are those of the matched HiL-IDQL cells.

### A17: HiL-SERL sessions

- Rebuild: `python -m paper.figures --only sim_hilserl_sessions` and
  `python -m paper.appendix.hilserl.prepare` (`hilserl_sessions.tex`).
- Re-run: as F2 (HiL-SERL). `hilserl_sessions.tex` lists every session the paper ran.

### A18: RLPD ablations (early kill, UTD 40)

- Rebuild: `python -m paper.stats.rlpd_ablations` recomputes the Welch summary
  (`paper/data/online_rl/rlpd_square_narrow_ablations_summary.json`) from the frozen per-seed eval curves
  (`rlpd_square_narrow_ablations_eval.csv`), requires it to equal the frozen copy, and prints the quoted
  numbers: 5 seeds per arm, the median seed's first nonzero eval at 80-90k, and the whole-curve mean eval
  difference to the reference (early kill -0.011 [-0.087, +0.066], UTD 40 -0.012 [-0.118, +0.094], so
  within +/-0.12). The ablation curves are not plotted in the paper.
- Retrain: `scripts/sim/train_cell.sh square-narrow-rlpd-early-kill` and
  `scripts/sim/train_cell.sh square-narrow-rlpd-utd40`, as for F2 (2.9-4.6 h per seed).

### A19, A20: real value learning and FQE

- Rebuild: `python -m paper.figures --only real_world_value_training real_world_value_fqe`
  (the `value_learning` package re-extracts its tables from the pinned producer outputs byte for byte).
- Retrain: the critics as F5 (real). FQE results are archived (tier 1).
- A20 (AUROC 0.943 to 0.897, Nut FQE 0.487-0.503, UTD 1.87 / 1.12 / 0.89) is quoted from an archived study.

### A21, A22, A23: critic objective, DIVL regimes, REDQ

- Rebuild: `python -m paper.figures --only value_learning_bar_chart` and `python -m paper.appendix.build`
  (`critic_objective_table.tex`, `divl_regime_table.tex`, `divl_sampler_table.tex`, `redq_table.tex`),
  from archived aggregates.
- Archived aggregates; not retrainable from this release.

### A24: bucket success rates

- Rebuild: `python -m paper.figures --only square_narrow_r2_bucket_success_sorted`. The values are digitized
  from an archived raster; tier 1 only.

### A25, A34: Sobol coverage and the eval-grid term in the sampler

- Rebuild: `python -m paper.figures --only square_narrow_init_distribution_sobol_vs_uniform`. It reads the two
  Square-Narrow R0 start files from the `simulation` evidence; the same bytes (same sha256) are in git
  under `data/sim/start_manifests/square_narrow/r00/init_states/`.
- Recompute: `sim_design --check` rebuilds the three rounds that used the eval-grid term (Square-Narrow
  R3, Square-Broad R2-R3) along with the other three, bit for bit.

### A29: training compute

- Rebuild: `python -m paper.appendix.paper_side.training_compute` (`training_compute.tex`) from the frozen
  run extract in `paper/data/sim/training_compute/`.

### A30, A33: best-of-N cost and the choice of N

- Rebuild: `python -m paper.appendix.paper_side.reranking_check` verifies the ten reranked real settings,
  the frozen latency summary (12.2% maximum) and the offline N sweep in `paper/data/real/reranking/`.
- The simulation N = 1 / 32 / 128 numbers are quoted from an archived study
  (`paper/data/real/reranking/README.md`).

### A32: critic training data (human-only vs all data)

- Rebuild: `python -m paper.appendix.critic_data.prepare` (`critic_data_table.tex`).
- Retrain: DIVL heads with human-only critic sampling behind the released R3 Mulligan agents, five
  seeds per task. The last flag overrides the recipe's `straddled` sampling:

  ```bash
  scripts/sim/train_divl_heads.sh square-narrow-r03-mulligan --seed 1 --dataset.critic_sampling_mode=human_only
  scripts/sim/train_divl_heads.sh square-broad-r03-mulligan --seed 1 --dataset.critic_sampling_mode=human_only
  ```

  The table reads the in-training evaluation on 400 fixed Sobol starts at N = 1 and N = 32, which the
  recipes already run at the final step. The paired controls are the recipes' own DIVL heads.

## Where the configs and the manuscript differ

The configs and recipes follow what the runs did. Where the frozen manuscript describes a run
differently, the configs are authoritative.

- Simulation start design: `tab:appendix-hp-sampler` gives the noise of Square-Narrow's local
  perturbations (nut `x` redrawn uniformly, N(0, 0.025) on `y`, N(0, 0.05) on yaw); the code also clips
  `y` to its range and wraps yaw.
- The task appendix calls Square-Narrow MimicGen `Square_D0`; the code instantiates robosuite
  `NutAssemblySquare` ([sim.md](sim.md#tasks-and-environments)).

## Known issues

These behaviors are kept so the recipes compute what the paper computed. Each item says what happens,
what it affects and how to avoid it.

- **Stage-label booleans without their timestamps.** The outcome endpoint priors set event booleans
  without their `*_time_s` (`mulligan/real/stage_specs/marker_d2.py` `apply_marker_d2_endpoint_prior`,
  `square_d2.py` `apply_square_d2_endpoint_prior`); the consensus fold majority-votes each boolean but takes
  the median time of every sample that gave one (`mulligan/real/stage_labeling/consensus.py`
  `_consensus_episode`); the jaws-never-closed caps clear the booleans but not their times
  (`mulligan/real/stage_specs/sensor_constraints.py` `cap_grasp_when_jaws_never_closed`, `routing_d2.py`
  `_cap_never_closed`). Effect: some label rows have a boolean and a time that disagree; the stage column,
  which every reported stage number reads, is unaffected. Fix: clear the time whenever the boolean is False,
  give the prior's forced S7 no booleans it cannot time, and regenerate the labels.
- **Nut S1-subtype few-shot anchors come from the labeled dataset.** The Nut cascade's S1-subtype node cuts
  its seven calibration crops from whatever dataset it labels, unlike the other anchor banks, which refuse a
  non-source dataset (`mulligan/real/stage_labeling/cascade_pipeline/square_d2.py`
  `_ensure_s1_subtype_anchor_crops`). Effect: on another dataset the node calibrates on unrelated wrist
  crops; it only picks the S1 failure mode (`pregrasp_misalignment` or `missed_grasp_after_alignment`), never
  the stage. Fix: pin the anchors to their source repo as `_ensure_transport_boundary_anchor_videos` does,
  and relabel.
- **Sim collectors: the object-motion recording trigger never fires.** `check_objects_moving`
  (`mulligan/sim/collect/utils.py`) looks for `env.objects`, but the Square envs keep the nut in `nuts`, so it
  always returns False. Effect: a human-segment frame is recorded only while there is operator input or the
  gripper moves; frames where only the nut moves are stepped but not recorded. Fix: read the active nut from
  `nuts`; this changes which frames a new collection records.
- **Released sim checkpoints record `step: 150001`** in `metadata.json` (and `release/models.json`) for the
  weights after 150,000 updates; `mulligan.training.train` records the last completed step.
