# Simulation configs

## What is here

| File | Contents |
|---|---|
| `recipes.json` | The 58 IDQL training recipes (one per cell of the paper's simulation experiments, covering all 104 `mulligan/sim-square-{narrow,broad}-rNN-*` model repos), the six RLPD and two HiL-SERL baseline recipes, the three locked evaluation grids, and the R0-R3 collection, split and rollout steps of each round |
| `<task>/rNN_sampler.yaml` | The start design (Alg. 2) of round N: rebuilds the locked collection starts of that round bit for bit |
| `<task>/rNN_grid_cell_sr.csv` | Input of a sampler that uses the evaluation-grid term: per-cell success of the previous round's Mulligan agent on the grid, five seeds |

`tests/sim/test_recipes.py` checks `recipes.json` (schema, pins, and that every command line parses). The
sampler configs are checked by `tests/sim/test_select_initial_states.py`.

## Naming

- `<task>` is `square_narrow` (Square-Narrow, robosuite `NutAssemblySquare`) or
  `square_broad` (Square-Broad, MimicGen `Square_D1`).
- `rNN` is the round, two digits. A sampler `rNN` designs the starts that collected
  `mulligan/sim-<task>-cNN-dagger-mixed`.
- A recipe id is the model repo stem: recipe `square-narrow-r01-mulligan` trains
  `mulligan/sim-square-narrow-r01-mulligan-idql` (IDQL agent) and
  `mulligan/sim-square-narrow-r01-mulligan-divl` (DIVL head). The method token is the arm:
  `baseline` (HG-DAgger data), `mulligan` (HG-DAgger+Mulligan data), `sobol` (R0 Sobol demos),
  `mulligan-no-cf`, `sobol-no-cf`, `sobol-with-cf` (sampler ablations), `auto-*` (autonomous baselines).
- The baseline recipes have no released checkpoints; their ids are `<task>-rlpd[-<variant>]` and
  `<task>-hilserl`.
- Seeds are `seeds: [1, 2, 3, 4, 5]` (HiL-SERL: `[1]`); seed N is released as `seed-N/` in the model repo.

## Run one

```sh
python -m mulligan.sim.recipes list --task square_narrow       # recipe ids
python -m mulligan.sim.recipes list --paper "Fig. 7"            # the recipes one paper result reads
python -m mulligan.sim.recipes show square-narrow-r01-mulligan  # one recipe as JSON
scripts/sim/train_cell.sh square-narrow-r01-mulligan --seed 1                      # IDQL agent
scripts/sim/train_cell.sh square-narrow-rlpd --seed 1                              # RLPD baseline
scripts/sim/train_divl_heads.sh square-narrow-r01-mulligan --seed 1 --parent local  # DIVL head
scripts/sim/eval_cell.sh square-narrow-r01-mulligan --seed 1 --checkpoint local     # locked grid
python -m mulligan.sampling.sim_design --config configs/sim/square_narrow/r01_sampler.yaml \
    --out outputs/sim/design/narrow_r01 --check                                    # start design
```

Every script prints its usage with `--help`. Pipeline, collection and replay: [docs/sim.md](../../docs/sim.md);
the baselines: [docs/baselines.md](../../docs/baselines.md).
Budgets: [docs/compute.md](../../docs/compute.md).

## Paper mapping

Each recipe's `paper` field lists the figures and appendix sections (manuscript numbering) that read its
checkpoints. `python -m mulligan.sim.recipes list --paper "<result>"` lists them.

| Recipes | Paper |
|---|---|
| `*-r00-baseline`, `*-r0{1,2,3}-baseline` (Square-Narrow R3: `square-narrow-r03-baseline-with-c03-rollouts`) | Fig. 5 HG-DAgger (IDQL agent, N=1) and HiL-IDQL (DIVL head, N=32); App. B.2 |
| `*-r00-sobol`, `*-r0{1,2,3}-mulligan` | Fig. 5 HG-DAgger+Mulligan (N=1) and HiL-IDQL+Mulligan (N=32); the HiL-IDQL+Mulligan curve of Fig. 2; App. B.2; R3: App. J.2 |
| `*-r0{1,2,3}-auto-{plain-il-n1,filtered-bc-n1,iql-n32,iql-success-bc-n32}` | Fig. 5 autonomous series (IDQL agent); their R0 point is the `*-r00-baseline` agent at N=32 |
| `square-narrow-r01-{baseline,mulligan,mulligan-no-cf,baseline-straddled-auto-success}` | Fig. 7 panels 1-2; App. B.5 |
| `square-narrow-r01-sobol-{no-cf,with-cf}` | App. B.5 and the numbers `python -m paper.stats.sim_contrasts` prints |
| every human-in-the-loop recipe (`family: hil`, 34) | App. G.7 (training compute) and App. J.4 (DIVL vs scalar critic) |
| `*-rlpd`, `square-narrow-rlpd-robomimic-ph`, `square-broad-rlpd-mimicgen-core`, `*-hilserl` | Fig. 2 baselines |
| `square-narrow-rlpd-{early-kill,utd40}`, `*-hilserl` | App. I.4 |

The remaining human-in-the-loop recipes (`*-first100`, `*-mulligan-no-cf` other than Square-Narrow R1,
Square-Broad `sobol-*` and `baseline-straddled-auto-success`, `square-narrow-r03-baseline`,
`square-narrow-r03-mulligan-no-c03-rollouts`)
appear only in App. G.7 and J.4. The DIVL heads of the autonomous `auto-iql-*` recipes are released but no
paper result reads them. App. B.4 (simulation speed and throughput) is an archived per-seed summary of
the headline series and does not rebuild from these recipes. Per-result commands are in
[docs/reproduce.md](../../docs/reproduce.md).

## Recipe fields

| Field | Meaning |
|---|---|
| `id`, `task`, `round`, `method` | Cell identity (see Naming). |
| `family` | `hil` (human-in-the-loop arms, 34 cells), `autonomous_baseline` (24 cells), `rlpd` (6) or `hilserl` (2). |
| `seeds` | Training seeds. |
| `eval` | The locked grid (key into `grids`), the `stage` the paper's headline (`partial_headline.csv`) reports, and the deployment best-of-N (32; 1 for the `auto-*-n1` baselines). RLPD and HiL-SERL evaluate during training: `log` names the file in the run directory. |
| `paper` | Figures and appendix sections that read the recipe's checkpoints (see Paper mapping). |
| `datasets` | Training datasets in order (the first is the human-demo dataset): `repo`, `revision` (the `release-1` pin), `role`, `episodes`. Empty for the RLPD recipes on third-party demos (`--offline_data`). |
| `idql_agent` | The scalar-IQL IDQL agent (critic + diffusion actor, `--policy.type=idql`). |
| `divl_head` | 46 recipes: the frozen-actor DIVL head (`--policy.type=idql_divl --training.update_components=critic_value_only`), trained from the seed-matched agent. |
| `rlpd_agent`, `hilserl_agent` | The baseline's stage: `trainer` (the module `train_cell.sh` runs) and its `args`. |
| `autonomous_collection` | Autonomous baselines: the rollout collection that produced this round's autonomous data (predecessor checkpoint, starts, seed, N). |
| `checkpoint`, `parent` | Release ids of the checkpoints a stage produced or started from. |

Each IDQL stage (`idql_agent`, `divl_head`) has

| Field | Meaning |
|---|---|
| `args` | `mulligan.training.train` draccus flags shared by all seeds, as `--key=value` (without the W&B, output-path and dataset flags). Training steps: 150,000 (Square-Narrow) and 250,000 (Square-Broad), also in `tasks`. |
| `seed_args` | Per-seed extra flags (optional). Five R0 agents train some seeds in plain FP32 with an eager actor (`--training.amp_dtype=none --training.enable_tf32=False --training.compile_actor=False`; Square-Narrow `baseline` and `sobol`: seeds 1-3 and 5; Square-Broad `baseline`, `baseline-first100` and `sobol-first100`: seed 1), as their released checkpoints did. |
| `model` | Released repo and revision. |
| `training_data` | `exact` if every training dataset's episode and frame counts at the paper run equal the released revision, else `differs` with a `training_data_note`. Three R2 Square-Narrow autonomous agents (`auto-iql-n32`, `auto-filtered-bc-n1`, `auto-plain-il-n1`) trained on an earlier version of their R2 rollout dataset than the one released; their retrains are approximate. |
| `checkpoints` | Per seed: `path` in the model repo and `parent` (DIVL heads: the `hf://` agent it was trained from). |

`scripts/sim/train_cell.sh` builds the command line as

```
python -m mulligan.training.train \
  --dataset.repo_ids=<datasets, comma-separated> --dataset.revisions=<{repo: revision}> \
  <args> <seed_args[seed]> --training.seed=<seed> \
  [--pretrained_artifact=hf://mulligan/sim-...-idql@<revision>/seed-<seed>]   # DIVL heads
  --system.checkpoint_dir=<output>
```

## Evaluation grids

Every simulation success rate in the paper is an evaluation on a fixed grid of initial states: one
episode per grid point. `grids` in `recipes.json` lists the three locked grids. They are not stored in
git (31 MB); they regenerate byte for byte in a few seconds, and public copies are in the evaluation
bundles. `scripts/sim/eval_cell.sh` generates the grid of a recipe on first use at
`outputs/sim/grids/<grid id>.json` and checks its `manifest_hash`.

| Grid id | Task | Points | Cells | Used by | `manifest_hash` | File sha256 |
|---|---|---|---|---|---|---|
| `square_narrow_sobol8k` | Square-Narrow | 8,000 | 80 | R1-R3 | `45a9963f5592f495477c44beabe8d3c3e3173e920c2b2ba009c1e69532e53b17` | `068b2deb11e553cdedfb0f789081b6fd7c4112aa7e22ecc3b655f90492168476` |
| `square_narrow_equal_tile` | Square-Narrow | 8,000 | 80 | R0 | `41166760da933ccca99e5ac01694fbcb1ab2042312abcdded7ac7a67a724dbea` | `d8ee57bff9e88fbe4ba2807d8d9ae672b2f62ccd6af3c9c02baeccf9f7117624` |
| `square_broad_sobol30k` | Square-Broad | 30,000 | 720 | R0-R3 | `e0b5056648639f54189386a92384212d4fb09949bda23b622c10f8427aaa085c` | `304210db9e7b3d175970f846753e2cd26ab59686fef3693c6f24f29c3f55a5e5` |

The valid-Sobol grids take scrambled Sobol points over the task's placement box, reject Square-Broad
points where the nut and peg overlap (`min_clearance` 0.13263 m), and bin the accepted points into cells
for the per-cell success maps. The equal-tile grid (100 scrambled Sobol points in each of 80
equal-width cells) is the grid of the two Square-Narrow R0 cells and a disjointness reference in the
Square-Narrow R1 start design. Regenerate a grid by hand (`generate` in `grids` holds the arguments):

```bash
python -m mulligan.sim.eval.grid_eval make-valid-sobol-manifest \
    --task square_narrow --num-points 8000 --seed 2026052402 \
    --output outputs/sim/grids/square_narrow_sobol8k.json
python -m mulligan.sim.eval.grid_eval make-valid-sobol-manifest \
    --task square_broad --num-points 30000 --seed 2026052499 --candidate-power 16 \
    --output outputs/sim/grids/square_broad_sobol30k.json
python -m mulligan.sim.eval.grid_eval make-equal-tile-manifest \
    --points-per-cell 100 --seed 20260524 \
    --output outputs/sim/grids/square_narrow_equal_tile.json
python -m mulligan.sim.eval.grid_eval validate-manifest \
    --point-manifest outputs/sim/grids/square_narrow_sobol8k.json   # prints the manifest_hash
```

The outputs match the sha256 values above. `manifest_hash` is the sha256 of the canonical JSON (sorted
keys, no whitespace) without the `manifest_hash` field. `tests/sim/test_eval_grids.py` regenerates all
three grids and checks both hashes (about 5 s on CPU).

The public simulation evaluation bundles hold the grids under `grids/<manifest_hash>.json`, and their
`evaluations.csv` names the grid of every evaluation (`gridManifestHash`, `gridFileSha256`):

| Bundle (HF dataset) | Revision | Grids |
|---|---|---|
| `mulligan/sim-square-narrow-r00-r03-eval` | `8826cfbdae8bbc7e03014753e85f4ea8e90cc056` | `45a9963f....json`, `41166760....json` |
| `mulligan/sim-square-broad-r00-r03-eval` | `79300a38305873d2869abab1538c3ed039bd08f0` | `e0b50566....json` |

The bundle copies at these revisions (the `release-1` pins) have the sha256 values in the table above.

## Rounds

`rounds` has one entry per task and round with

- `dagger` (R1-R3): the blinded DAgger collection. `policies.mulligan` / `policies.baseline` are the
  previous round's seed-1 agents with the routing labels of the manifest, `starts` and `manifest` are
  the locked files in `data/sim/start_manifests/`, and `protocol_*` the no-CF / with-CF quota. `dataset`
  is the released mixed collection.
- `splits`: how the mixed collection was split into the per-arm training datasets (`protocol_quota`
  for R1-R3, `blind` for R0), with the released `input` revision and the `targets` (arm -> released
  dataset).
- `rollouts`: the policy-rollout collections of the round (checkpoint, starts, episodes, released
  dataset).

Run them with `scripts/sim/collect_round.sh` and `scripts/sim/split_round.sh`.

## Start-design configs

`<task>/rNN_sampler.yaml` (R1-R3, both tasks) lists the arms of the round and, for the Mulligan arm,
the `select_initial_states` inputs: the previous round's diagnostic rollouts, earlier starts, the
perturbation and candidate seeds, the distance, the hardness terms, and the evaluation-grid term where
App. F.5 uses it (Square-Narrow R3, Square-Broad R2-R3). `manifest` builds the blinded list, `guardrail`
is the coverage check and `locked` names the locked files and their sha256 that `--check` compares.
The header of each file names its source builder and the paper sections.
