# Mulligan

Code, configs and paper builders for **Mulligan: Performance-Guided Data Collection for Efficient
On-Robot Learning**. Project page: [mulligan.page](https://mulligan.page/?ref=github). Paper: <https://arxiv.org/abs/2610.05882>.

Mulligan improves a robot policy over rounds of supervised deployment. Each round, an operator places
the objects and takes over when the policy fails. Failures tend to concentrate in a small part of the
initial-state space, so Mulligan starts each round's episodes at the previous round's observed failures
and at states not yet tried, instead of sampling them uniformly. It also trains a value function on all
collected data, failures included, and uses it to rerank sampled action chunks (HiL-IDQL). On three
real tasks (Insert Marker, Thread Nut, Route Cable), scored on 2,550 blinded held-out episodes, and two
simulated Square tasks (Square-Narrow, Square-Broad), this beats uniform initial-state sampling at the
same collection budget.

This repository holds the learners, the simulation and real-robot pipelines, the RLPD and HiL-SERL
baselines, the locked start manifests and evaluation grids, the configs of every released checkpoint, and
the code that rebuilds every figure and table of the paper. The data and checkpoints are on Hugging Face
under [`mulligan`](https://huggingface.co/mulligan) (223 dataset and 180 model repositories). The Policy
Arena at [arena.mulligan.page](https://arena.mulligan.page/?ref=github) browses them.

**Contents:** [What you can reproduce](#what-you-can-reproduce) · [Layout](#repository-layout) ·
[Setup](#setup) · [Data and models](#data-and-models-on-hugging-face) ·
[Run the experiments](#run-the-experiments) · [Tests](#tests) ·
[License](#license) · [Citation](#citation) · [Appendix](#appendix-reproducibility-by-experiment)

## What you can reproduce

The paper's human-in-the-loop data cannot be collected again as it was, but everything downstream of
the recorded data can be recomputed. The release supports four tiers:

1. **Rebuild** a figure or table from frozen inputs, with one command, without a GPU or any account.
2. **Recompute** its numbers from the public Hugging Face datasets.
3. **Retrain / re-evaluate** from released datasets and checkpoints with pinned configs.
4. **Re-run the protocol**: collect new data or run new robot evaluations with the released collection,
   routing, start-selection and blind-evaluation code. A new campaign gives new results.

The [appendix](#appendix-reproducibility-by-experiment) lists the tiers each paper figure and table
supports; [docs/reproduce.md](docs/reproduce.md) has the commands and compute.

### What cannot be reproduced, and why

- **Human-in-the-loop collection.** Every real and simulated DAgger round, and the HiL-SERL operator
  sessions, had a live SpaceMouse operator. A new operator makes different interventions, so the data and
  results differ. The released recordings are the paper's data. The simulation replay operator re-drives
  the recorded human segments headless; it reproduces the protocol, not the data.
- **Real-robot evaluations.** They depend on the physical scene, the operator who resets it and judges
  outcomes, and human-reconciled labels. The collectors and the blind evaluator are runnable protocol code
  for a new campaign ([docs/real_robot_eval.md](docs/real_robot_eval.md)).
- **Gemini stage labels** (A10, A15). The labeling model changes over time; the release ships the frozen
  labels and the labeler.
- **Archived-only results.** A7, A20-A24 rebuild from archived aggregates only (tier 1).
- **The paper's RLPD curves** were run with robosuite 1.4.1; this release's simulator is robosuite
  1.5.2, so an RLPD re-run can differ slightly ([docs/baselines.md](docs/baselines.md)).

## Repository layout

```
mulligan/      Python package: agents and networks, training, simulation (sim/), real robot (real/),
               start-state selection (sampling/), baselines (baselines/), release manifests and verifier (release/)
paper/         figure and table builders (python -m paper.figures, paper/appendix/), frozen paper data, text statistics
configs/       every run setting: sim recipes (incl. the RLPD and HiL-SERL baselines) and start designs, real trainer/eval/sampler configs (configs/README.md)
scripts/       shell entry points: sim rounds and training, HiL-SERL operator sessions, robot environment
data/          locked start manifests: real (data/real/manifests) and simulation (data/sim/start_manifests)
release/       manifests of the public datasets and models with pinned revisions, the paper result points
robot/         separate uv project for the Franka/ZED robot workstation
arena/         Policy Arena: self-deploy web app (Convex backend, Python client) and the read-only snapshot
docs/          documentation (start at docs/README.md)
tests/         offline, network, gpu and hardware tests
```

## Setup

### Main environment

One [uv](https://docs.astral.sh/uv/) project (Python 3.12) covers everything except the robot
workstation: simulation, real-robot training and evaluation code, both baselines and the paper builders.
Supported platforms: Linux x86_64 and macOS arm64 (the only two in `uv.lock`).

```bash
# Debian/Ubuntu system packages (OpenCV, video decoding, headless MuJoCo)
sudo apt-get install -y libgl1 libglib2.0-0 ffmpeg libegl1
export MUJOCO_GL=egl                              # headless rendering; omit on a desktop or macOS

git clone <repository-url> mulligan && cd mulligan
uv sync --frozen                                  # sim and real training/eval, both baselines, paper figures
uv sync --frozen --extra teleop                   # plus SpaceMouse teleop
```

Each `uv sync` installs exactly the listed extras and removes the others, so name all you need in one
command. Extras: `teleop` (SpaceMouse and keyboard teleop in simulation),
`stage-labeling` (Gemini stage labeler, needs `GEMINI_API_KEY`), or `--all-extras`. Run the commands
below with `uv run ...` or after `source .venv/bin/activate`; the commands in this README omit `uv run`.

- **GPU.** On Linux, torch comes from the PyTorch CUDA 12.8 index, so GPU work needs a driver that
  supports CUDA 12.8; the wheels also run on CPU. The simulation recipes train with bf16 autocast and need
  an Ampere or newer GPU ([docs/compute.md](docs/compute.md#gpu-generation)). jax preallocates 75% of
  GPU memory; set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when torch shares the GPU (the baseline trainers
  set it).
- **macOS.** torch runs on CPU/MPS and jax on CPU. Interactive MuJoCo viewers (sim teleop, sim DAgger,
  the HiL-SERL actor) need `mjpython` instead of `python`: `uv run mjpython -m mulligan.sim.collect.teleop ...`.
  FFmpeg: `brew install ffmpeg`.
- **Paper teaser.** Chrome or Chromium (`MULLIGAN_CHROME`, or `google-chrome` / `chromium` on `PATH`) and
  Ghostscript (`gs`). Without them, build the figures with `--exclude teaser`.

Lock details, pinned versions and the expected import warnings: [docs/install.md](docs/install.md).

### Robot workstation (`robot/`)

The Franka/ZED station runs a second uv project, `robot/`, because the ZED SDK's `pyzed` 4.2 needs numpy
1.x. It installs this repository (editable, with the `teleop` extra) plus the pinned DROID fork, `gym`,
`zerorpc` and `pyzed`. It needs Linux x86_64, about 9 GB of disk, ZED SDK 4.2 under `/usr/local/zed`,
the DROID/Polymetis stack on the NUC ([docs/hardware/droid_fork.md](docs/hardware/droid_fork.md#deploying-to-the-nuc))
and a station file at `~/.config/droid/station.env`
([configs/real/station.env.example](configs/real/station.env.example)).

```bash
bash scripts/sync_robot_env.sh                    # build or repair robot/.venv and check the imports
uv run --project robot --frozen python -m mulligan.real.eval.manifest_eval --help
```

Always pass `--project robot` for robot commands. Station setup (cameras, DROID fork, SpaceMouse,
operator display): [docs/station.md](docs/station.md).

### Policy Arena (`arena/`, optional)

`arena/` is the web app the project used for its real-robot evaluations (pairwise sessions,
Bradley-Terry leaderboards, outcome and stage labeling, dataset statistics, Hugging Face sign-in).
You can deploy your own, with your own Convex backend and a Vercel or Cloudflare Pages frontend
([docs/arena.md](docs/arena.md)); the same tree also builds the read-only Mulligan snapshot. Both
need [bun](https://bun.sh) 1.3 and Node 20.19+, 22.13+ or 24:

```bash
cd arena
bun install --frozen-lockfile
bun test                                          # backend, UI and release-adapter tests
bun run package:release data/release.json         # the snapshot: type-check, build, stage in dist-release/
```

Commands, layout and checks: [arena/README.md](arena/README.md).

### Check the install (a few minutes)

```bash
# imports and the patched Square environments (prints "ok" after some expected robosuite warnings)
python -c "from mulligan.sim.envs import register_square_environments; register_square_environments(); import lerobot, torchcodec, torch, cv2; print('ok')"
python -c "import torch; print('cuda', torch.cuda.is_available())"
# start design: rebuilds the locked Square-Narrow R1 starts bit for bit (CPU, seconds)
python -m mulligan.sampling.sim_design --config configs/sim/square_narrow/r01_sampler.yaml --out outputs/check/narrow_r01 --check
# end to end on CPU (a few minutes): data, 200 training steps,
# a replayed DAgger round, a 10-start grid evaluation and a plot
pytest tests/sim/test_smoke.py -m network
```

## Data and models on Hugging Face

All data and checkpoints are public under the [`mulligan`](https://huggingface.co/mulligan)
organization; reads are anonymous (set `HF_TOKEN` only for your own private repos). Everything in this
repository reads a repo at its release pin, never at `main`.

### Names

IDs are lowercase kebab case, `{task}-...`, with two-digit indexes. Task tokens: `real-marker-d2`
(Insert Marker), `real-square-d2` (Thread Nut), `real-routing-d2` (Route Cable), `sim-square-narrow`,
`sim-square-broad`. `cNN` is a collection increment, `rNN` a model round, `bNN` a recorded evaluation
session.

| Kind | Pattern | Example |
|---|---|---|
| Raw collection | `{task}-c00-teleop-mixed`, `{task}-cNN-dagger-mixed` | `mulligan/real-marker-d2-c03-dagger-mixed` |
| Training view of a collection | `{task}-cNN-{teleop,dagger}-{arm}` | `mulligan/sim-square-narrow-c01-dagger-mulligan` |
| Round evaluation | `{task}-rNN-eval` (Cable: `real-routing-d2-r00-r05-eval`) | `mulligan/real-square-d2-r05-eval` |
| Policy rollouts | `{eval}-bNN-{arm}-policy-rollouts`, sim `{task}-cNN-{arm}-policy-rollouts` | `mulligan/sim-square-narrow-c02-auto-iql-n32-policy-rollouts` |
| Simulation evaluation bundle | `{task}-r00-r03-eval` | `mulligan/sim-square-broad-r00-r03-eval` |
| Real DP actor / IDQL critic | `{task}-rNN-{arm}-dp`, `{task}-rNN-mulligan-idql-critic` | `mulligan/real-marker-d2-r05-mulligan-idql-critic` |
| Simulation agent (folders `seed-1` ... `seed-5`) | `{task}-rNN-{arm}-{idql,divl}` | `mulligan/sim-square-broad-r01-mulligan-divl` |

Arms: `baseline` (HG-DAgger), `mulligan` (HG-DAgger+Mulligan), `sobol` (R0 Sobol demonstrations),
`mulligan-no-cf` (no-counterfactual ablation). The full table, roles, tiers and the Route Cable round axes
are in [docs/data.md](docs/data.md#names).

### Pins and the `hf://` grammar

`release/revisions.json` maps every repo to its release revision (tag `release-1`, the single
commit of every dataset, model and paper-evidence repo). Model and checkpoint IDs are written

```
hf://<org>/<name>[@<revision>][/<subfolder>]      e.g. hf://mulligan/sim-square-narrow-r01-mulligan-divl/seed-1
```

(`hf://<org>/<name>[/<subfolder>][@<revision>]` is accepted too). `<revision>` is a commit sha, a tag or
`refs/pr/<N>`. Without `@<revision>`, a `mulligan/*` repo resolves to its pin in `release/revisions.json`;
`tests/release/test_revision_consistency.py` fails if any file pins a `mulligan/*` repo elsewhere. The
manifests `release/datasets.json` and `release/models.json` list every repo with its training data,
evaluations and provenance ([docs/data.md](docs/data.md#manifests)).

### Download

Trainers, evaluators and the verifier download what they need at the pinned revisions on first use, so
downloading ahead is optional (useful on a cluster without network on the compute nodes):

```bash
python -m mulligan.release.download repo mulligan/real-marker-d2-r05-eval --include "meta/*"   # one repo
python -m mulligan.release.download models --task real-marker-d2 --round 5 --dry-run          # list, no download
python -m mulligan.release.download models --task real-marker-d2 --round 5                    # Marker R5 checkpoints
python -m mulligan.release.download datasets --task real-marker-d2 --round 5 --meta-only      # its datasets, metadata only
python -m mulligan.release.download datasets --task sim-square-narrow --round 1               # Square-Narrow R1 data
```

Filters (`--task`, `--round`, `--method`, `--role` for datasets, `--kind` for models) combine with AND.
`python -m mulligan.sim.recipes show <recipe>` and the `dataset_revisions` of a real config list the exact
training data of one checkpoint.

**Cache.** Checkpoints and `download` go to the huggingface_hub cache (`HF_HUB_CACHE`, default
`~/.cache/huggingface/hub`; `--cache-dir` or `--local-dir DIR` to put them elsewhere). LeRobot datasets that
the trainers read go to `HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`). Outputs of the
pipeline scripts go to `outputs/` (`MULLIGAN_OUTPUT_DIR`).

### Load a released checkpoint

Simulation agent (IDQL or DIVL head; Best-of-N with `num_action_samples` candidates):

```python
from mulligan.release.hub import resolve_checkpoint
from mulligan.utils.load_pretrained import load_policy_from_checkpoint

path = resolve_checkpoint("hf://mulligan/sim-square-narrow-r01-mulligan-divl/seed-1")  # pinned revision
policy, preprocessor, postprocessor = load_policy_from_checkpoint(checkpoint_path=path, device="cuda")
policy.eval()
```

`python -m mulligan.sim.eval.grid_eval eval --artifact-path <hf:// id> ...` runs it on an evaluation grid
([docs/sim.md](docs/sim.md#grid-evaluation)).

Real DP actor, or an IDQL critic with the DP actor it reranks (Best-of-N):

```python
from mulligan.real.policy.loader import load_policy_by_model_id

dp = load_policy_by_model_id("hf://mulligan/real-marker-d2-r05-mulligan-dp", policy_id=0, device="cuda")
bon = load_policy_by_model_id("hf://mulligan/real-marker-d2-r05-mulligan-idql-critic", policy_id=1, device="cuda")
bon.policy.num_action_samples = 32   # N as deployed in R5 (configs/real/marker_d2/r05_eval.yaml)
```

A critic repo holds the critic only; its `metadata.json` names the DP actor it reranks and the loader
fetches that DP too. Each returned `PolicyEntry` holds the policy and its pre/post-processing
([docs/real_robot.md](docs/real_robot.md#deploy-load-a-policy)).

### Skip a step with its released output

Every pipeline step below names the released artifact it produced. The training configs already read the
released datasets, and `eval_cell.sh` evaluates the released checkpoints by default, so you can start
anywhere:

| To skip | Use |
|---|---|
| Recording R0 demonstrations or a DAgger round | the `*-c00-teleop-mixed` / `*-cNN-dagger-mixed` datasets (the split training views are `*-cNN-*-{arm}`) |
| Splitting a collection | the per-arm training views, e.g. `mulligan/sim-square-narrow-c01-dagger-mulligan` |
| Training | the checkpoint repos (`hf://mulligan/sim-...-{idql,divl}/seed-N`, `hf://mulligan/real-...-{dp,idql-critic}`) |
| Simulation grid evaluation | the evaluation bundles `mulligan/sim-square-{narrow,broad}-r00-r03-eval` (per-start outcomes of every seed) |
| Start design | the locked starts in git: `data/sim/start_manifests/`, `data/real/manifests/` |
| A real evaluation and its review | the round datasets `mulligan/real-*-rNN-eval` and `mulligan/real-routing-d2-r00-r05-eval` (reviewed outcomes, `meta/episode_provenance.parquet`) |
| Recomputing a paper number | `release/paper-results.json` and the frozen paper evidence |

## Run the experiments

Each track below is a numbered pipeline. Every step gives the command, its inputs, its outputs and the
released artifact that replaces it. All commands run from the repository root. Compute budgets are in
[docs/compute.md](docs/compute.md); per-figure commands are in [docs/reproduce.md](docs/reproduce.md).

### Simulation (Square-Narrow, Square-Broad)

Configs: `configs/sim/recipes.json` (58 training recipes, the RLPD and HiL-SERL baselines, the locked
grids, the round specs) and
`configs/sim/<task>/rNN_sampler.yaml` ([configs/sim/README.md](configs/sim/README.md)). A recipe id is the
model repo stem, e.g. `square-narrow-r01-mulligan`; `python -m mulligan.sim.recipes list --paper "Fig. 5"`
lists the recipes behind one paper result. `<task>` is `square_narrow` or `square_broad`; `N` is the round.

1. **R0 data.** Record the R0 demonstrations with a SpaceMouse (`uv sync --frozen --extra teleop`; human,
   about 200/400 episodes), then split them by arm:

   ```bash
   python -m mulligan.sim.collect.teleop --r0-preset square_narrow          # mjpython on macOS
   scripts/sim/split_round.sh --task square_narrow --round 0
   ```

   In: `data/sim/start_manifests/<task>/r00/`. Out: `outputs/sim/data/sim-<task>-c00-teleop-mixed`, then
   `outputs/sim/splits/`. Released: `mulligan/sim-<task>-c00-teleop-{mixed,baseline,sobol}`. To split the
   released recording instead, fetch it first:
   `python -m mulligan.release.download repo mulligan/sim-square-narrow-c00-teleop-mixed --local-dir outputs/sim/data`.
2. **Train the round's agents** (one GPU; per seed about 1.9 GPU-h Square-Narrow / 3.0 GPU-h Square-Broad
   for the IDQL agent, 0.65 / 0.97 GPU-h for the DIVL head):

   ```bash
   scripts/sim/train_cell.sh square-narrow-r00-sobol --seed 1                         # IDQL agent
   scripts/sim/train_divl_heads.sh square-narrow-r00-sobol --seed 1 --parent local    # DIVL head on it
   ```

   In: the recipe's datasets at their pins. Out: `outputs/sim/train/<recipe>/{idql_agent,divl_head}/seed-1/`.
   Released: `mulligan/sim-<task>-rNN-<arm>-{idql,divl}`, folder `seed-N`. Without `--seed`, all five seeds.
3. **Evaluate on the locked grid** (8,000 starts Square-Narrow, a few GPU-hours; 30,000 starts
   Square-Broad, run as 4 shards):

   ```bash
   scripts/sim/eval_cell.sh square-narrow-r00-sobol --seed 1 --checkpoint local       # your checkpoint
   scripts/sim/eval_cell.sh square-narrow-r01-mulligan --seed 1 --shard 1 --n-shards 160   # released, 50 starts
   ```

   The grid is generated and hash-checked on first use. Out:
   `outputs/sim/eval/<recipe>/<stage>/seed-N/n<N>/results.json`. Released: the bundles
   `mulligan/sim-<task>-r00-r03-eval`.
4. **Policy rollouts for the next start design** (one GPU, within 4 h per 100 episodes):
   `scripts/sim/collect_round.sh rollouts --task square_narrow --round 1`. In: the previous round's seed-1
   agents. Out: `outputs/sim/data/sim-<task>-cNN-*-policy-rollouts`. Released: the same names under
   `mulligan/`.
5. **Start design (Alg. 2)** (CPU, seconds):

   ```bash
   python -m mulligan.sampling.sim_design --config configs/sim/square_narrow/r01_sampler.yaml --out outputs/sim/design/narrow_r01 --check
   ```

   In: the rollout audits and earlier starts in `data/sim/start_manifests/`. Out: the round's start lists;
   `--check` compares them with the locked files bit for bit. Released: `data/sim/start_manifests/<task>/rNN/`.
6. **Blinded DAgger round** with a SpaceMouse operator, or replaying the recorded human segments headless:

   ```bash
   scripts/sim/collect_round.sh dagger --task square_narrow --round 1                       # human
   scripts/sim/collect_round.sh dagger --task square_narrow --round 1 --operator replay     # headless
   scripts/sim/split_round.sh --task square_narrow --round 1
   ```

   In: the locked starts and the previous round's agents (N=32). Out: `outputs/sim/data/sim-<task>-cNN-dagger-mixed`,
   its ledger in `outputs/sim/ledgers/` (the split needs it; the released mixed collections do not carry
   one), and the per-arm splits. Released: `mulligan/sim-<task>-cNN-dagger-{mixed,baseline,mulligan,mulligan-no-cf,...}`.
   The autonomous baselines collect with `scripts/sim/collect_round.sh autonomous <recipe>`.
7. **Next round.** For N = 0, 1, 2: train and evaluate round N (steps 2-3, recipes `square-narrow-r0N-*`,
   `square-broad-r0N-*`), then collect round N+1 (steps 4-6). Finish with steps 2-3 for R3.
8. **Aggregate.** The paper reports the mean over five training seeds of each seed's success rate, with a
   Student-t 95% interval over seeds:

   ```python
   import json
   from mulligan.release.verify_results import sim_rule
   seeds = []
   for s in range(1, 6):
       r = json.load(open(f"outputs/sim/eval/square-narrow-r01-mulligan/divl_head/seed-{s}/n32/results.json"))
       seeds.append((round(r["overall_success_rate"] * r["num_points"]), r["num_points"]))
   print(sim_rule(seeds))   # (mean, low, high)
   ```

All 58 recipes x 5 seeds are about 900 GPU-h. Evaluations are unseeded by default;
`--eval-seed S --env-seed S` makes a re-run repeatable ([docs/sim.md](docs/sim.md#grid-evaluation)).
Pipeline details, the replay operator and R0 flags: [docs/sim.md](docs/sim.md).

### Real robot (Insert Marker, Thread Nut, Route Cable)

Configs: `configs/real/<task>/` with `<task>` in `marker_d2`, `square_d2`, `routing_d2`: one trainer
config per released checkpoint (`rNN_<arm>_dp.yaml`, `rNN_critic.yaml`), one `rNN_eval.yaml` per evaluated
round and one `rNN_sampler.yaml` per Marker/Nut collection round ([configs/real/README.md](configs/real/README.md)).
Each trainer config's `paper` block says which figure, round, arm and evaluation sessions it belongs to.
In repo names below, `<task>` is written `marker-d2`, `square-d2` or `routing-d2`. The examples use
Insert Marker. Collection and evaluation need the robot environment and a station; training needs only
the main environment and one GPU.

1. **R0 demonstrations** (robot, human): teleoperate from the locked R0 start list, then split by arm.

   ```bash
   uv run --project robot --frozen python -m mulligan.real.collect.teleop --task-name marker_d2 --save-data \
       --dataset-name marker-r0 \
       --initial-states-manifest data/real/manifests/marker_d2/r00/marker_d2_r0_three_arm_uniform100_sobol100_evalsobol50_scrambled.json
   python -m mulligan.real.data.split --source-repo <you>/marker-r0 --source-root ./data/marker-r0 \
       --ledger ./data/marker-r0/meta/teleop_manifest_ledger.jsonl --split-key manifest_source \
       --expected-success-per-arm baseline_uniform=100,mulligan_sobol=100,eval_heldout=50 \
       --target baseline_uniform=<you>/marker-r0-baseline --target mulligan_sobol=<you>/marker-r0-sobol \
       --target eval_heldout=<you>/marker-r0-validation --output-root outputs/real/splits
   ```

   Out: `./data/marker-r0` and one dataset per start source (the held-out 50 become the validation
   view). Released: `mulligan/real-<task>-c00-teleop-{mixed,baseline,sobol,validation}`.
2. **Train the round's DP actors and IDQL critic** (one GPU each; DP median 3.1-4.3 h, critic 3.7-6.6 h):

   ```bash
   python -m mulligan.real.train.launch configs/real/marker_d2/r05_mulligan_dp.yaml --output-dir outputs/real/marker_r05_dp
   python -m mulligan.real.train.launch configs/real/marker_d2/r05_critic.yaml --output-dir outputs/real/marker_r05_critic
   ```

   In: the pinned datasets and episode selectors of the config. `--print-argv` shows the trainer command;
   flags after `--` override it. Released: `mulligan/real-<task>-rNN-<arm>-dp`,
   `mulligan/real-<task>-rNN-mulligan-idql-critic`.
3. **Blind held-out evaluation** (robot, human resetting the scene; 50 starts per policy):

   ```bash
   uv run --project robot --frozen python -m mulligan.real.eval.manifest_eval --environment marker_d2 \
       --initial-states-manifest data/real/manifests/marker_d2/r05/marker_d2_r5_eval_heldout_independent_sobol.json \
       --fixed-policy baseline=hf://mulligan/real-marker-d2-r05-baseline-dp \
       --fixed-policy mulligan=hf://mulligan/real-marker-d2-r05-mulligan-dp \
       --fixed-policy iql=hf://mulligan/real-marker-d2-r05-mulligan-idql-critic --fixed-policy-num-action-samples iql=32 \
       --random-seed 2026070704 --dataset-name marker-r5-heldout --hf-repo-id <you>/marker-r5-heldout
   ```

   The arms, manifest and seed of every paper session are in `configs/real/<task>/rNN_eval.yaml`.
   Released: `mulligan/real-<task>-rNN-eval`.
4. **Outcome review**: `python -m mulligan.tools.outcome_review --repo-id <you>/marker-r5-heldout --filter all --push`
   (cv2 editor; corrects outcomes and marks outcome and sub-goal frames). Released datasets are already
   reviewed.
5. **Held-out ingest** (CPU): pairs the arms by start and writes Wilson intervals, paired deltas, exact
   McNemar tests and a figure. It reads a released evaluation dataset at its pin, or your own eval repo at
   Hub `main`:

   ```bash
   python -m mulligan.real.lifecycle.heldout_eval mulligan/real-square-d2-r02-eval --out outputs/real/heldout/square_d2_r02
   python -m mulligan.real.lifecycle.heldout_eval <you>/marker-r5-heldout --out outputs/real/heldout/marker_r5
   ```

   `HeldoutEvalConfig` + `run` is the Python API (sub-goal scoring, custom labels);
   `python -m mulligan.real.lifecycle.routing_d2_lineage` rebuilds the six Cable rounds with sub-goal
   scores ([docs/real_robot_eval.md](docs/real_robot_eval.md#after-the-eval)). The released round
   datasets also hold the paired outcomes directly (`meta/episode_provenance.parquet`).
6. **Stage labels** (optional; Gemini, `--extra stage-labeling`): `python -m mulligan.real.stage_labeling.prepare_events`,
   then `...stage_labeling.label` and `...stage_labeling.apply_cascade`, then
   `python -m mulligan.real.stage_labeling.stage_eval_battery` for per-arm stage shares and rung
   conversions (the inputs of the per-stage results, App. B.7;
   [mulligan/real/stage_labeling/DESIGN.md](mulligan/real/stage_labeling/DESIGN.md)). The joined labels
   (`labels_joined.csv`) are also an input of the next start design. The paper's labels are frozen in the
   paper evidence.
7. **Start design (Alg. 2)** for the next round (Marker and Nut), here the Marker R1 design from the R0
   evaluation:

   ```bash
   python -m mulligan.sampling.real_design --config configs/real/marker_d2/r01_sampler.yaml \
       --inputs-root . --input eval_outcomes=<paired_round_outcomes.csv> --input stage_labels=<labels.csv> \
       --out outputs/real/marker_r01_manifest.json
   ```

   Run it from the repository root: the earlier manifests the design reads ship in `data/real/manifests/`
   (the config's `inputs` name them relative to `--inputs-root .`). The evaluation outcome table and the stage
   labels are yours; repeat `--input` to replace a list input such as `support_manifests`. The locked
   manifests in `data/real/manifests/` are the released output.
8. **Blinded DAgger round** (robot, operator with a SpaceMouse), then split by arm:

   ```bash
   uv run --project robot --frozen python -m mulligan.real.collect.blind_dagger --task-name marker_d2 \
       --arm baseline_uniform=hf://mulligan/real-marker-d2-r01-baseline-dp \
       --arm mulligan_sobol=hf://mulligan/real-marker-d2-r01-mulligan-dp \
       --initial-states-manifest data/real/manifests/marker_d2/r02/marker_d2_r2_promote25_fill25_blind_dagger.json \
       --protocol-quota-targets no_cf=50,with_cf=50 \
       --protocol-quota-arms 'no_cf=baseline_uniform,mulligan_sobol;with_cf=mulligan_sobol' \
       --protocol-quota-selection-mode soft_weighted \
       --protocol-quota-ledger ./data/marker-r2/meta/protocol_quota_ledger.jsonl \
       --dataset-name marker-r2
   python -m mulligan.data.split_protocol_quota --source-repo <you>/marker-r2 --source-root ./data/marker-r2 \
       --manifest data/real/manifests/marker_d2/r02/marker_d2_r2_promote25_fill25_blind_dagger.json \
       --ledger ./data/marker-r2/meta/protocol_quota_ledger.jsonl --expected-per-protocol no_cf=50,with_cf=50 \
       --target no_cf.baseline_uniform=<you>/marker-r2-baseline --target no_cf.mulligan_sobol=<you>/marker-r2-mulligan-no-cf \
       --target with_cf.mulligan_sobol=<you>/marker-r2-mulligan --output-root outputs/real/splits
   ```

   The protocol split gives the paper's per-arm training views (HG-DAgger: no-CF baseline episodes;
   HG-DAgger+Mulligan: with-CF; the no-CF ablation). `python -m mulligan.real.data.split` instead splits a
   collection by arm only (`--split-key arm_key`). Released:
   `mulligan/real-<task>-cNN-dagger-{mixed,baseline,mulligan,mulligan-no-cf}`.
9. **Next round**: train on the new views (step 2) and repeat steps 3-8 for R1-R5.
10. **Final evaluations**: the paper's evaluation of each round is the `rNN_eval.yaml` session list; Route
    Cable evaluated all six rounds in one 15-arm session on the lineage manifest
    (`data/real/manifests/routing_d2/lineage/`).

Collector options, loading and the Best-of-N path: [docs/real_robot.md](docs/real_robot.md); evaluation
details and outputs: [docs/real_robot_eval.md](docs/real_robot_eval.md); station: [docs/station.md](docs/station.md).

### Baselines (RLPD, HiL-SERL)

Both are recipes in `configs/sim/recipes.json` (`python -m mulligan.sim.recipes list --family rlpd`,
`--family hilserl`) and train with `train_cell.sh` like the IDQL recipes ([docs/baselines.md](docs/baselines.md)).

1. **RLPD**, seeds 1-5 (one GPU; per seed 2.8-4.5 h for 300k Square-Narrow steps, 8.1-13.9 h for 1M
   Square-Broad steps):

   ```bash
   scripts/sim/train_cell.sh square-narrow-rlpd --seed 1   # also square-broad-rlpd, square-narrow-rlpd-robomimic-ph, square-broad-rlpd-mimicgen-core
   ```

   In: the recipe's offline demos, fetched and hash-checked on first use. Out:
   `outputs/sim/train/<recipe>/rlpd_agent/seed-N/eval.jsonl` (the learning curve). No checkpoints are
   released; the paper's curves are in `paper/data/online_rl/`.
2. **HiL-SERL**: a no-human run, a fork once its policy starts succeeding (150k Square-Narrow, 200k
   Square-Broad), then the operator session (learner and eval watcher on the GPU machine, actor at the
   SpaceMouse):

   ```bash
   scripts/sim/train_cell.sh square-narrow-hilserl
   scripts/hilserl/fork.sh outputs/sim/train/square-narrow-hilserl/hilserl_agent/seed-1 outputs/hilserl/narrow_fork150k 150000
   scripts/hilserl/learner.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
   scripts/hilserl/eval_watcher.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
   scripts/hilserl/actor.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k <learner host>
   ```

### Paper figures and tables

CPU only; no accounts. The frozen paper evidence is read from the `mulligan/paper-evidence` dataset at its pin:

```bash
python -m paper.figures                                  # 42 figure files -> paper/build/figs
python -m paper.appendix.build --check                   # appendix tables vs the manuscript; run after paper.figures
python -m paper.figures --check                          # figure bytes vs paper/figures_manifest.json
python -m paper.figures --only headline_real_sim         # one figure; --list shows all entries
```

Without the evidence, `python -m paper.figures --keep-going` builds what it can and lists the rest with
the reason; add `--exclude teaser` without Chrome. The numbers quoted in the text print with
`python -m paper.stats.<module>` (for example `paper.stats.final_round`, `paper.stats.sim_contrasts`).

**Recompute the paper's numbers from the public data** (CPU; about 0.5 GB, a few minutes cold):

```bash
python -m mulligan.release.verify_results --out report.json
```

It recomputes the 110 headline result points, the 2,550 evaluation episodes and the 988 / 1,009
collection successes from per-episode records and exits non-zero on any mismatch
([docs/data.md](docs/data.md#checking-the-papers-numbers)). Figures and inputs: [docs/paper_figures.md](docs/paper_figures.md).

## Tests

```bash
pytest -m "not network and not gpu and not hardware" -n 8   # offline suite, CPU (what CI runs)
pytest -m network                                           # also reads public Hugging Face repos, anonymously
pytest -m gpu                                               # needs CUDA
```

Markers: `network` (Hugging Face or other downloads, including the Arena build's `bun install`), `gpu`
(needs CUDA), `hardware` (robot or SpaceMouse) and `slow`. Tests that read the paper evidence read it from
`MULLIGAN_PAPER_EVIDENCE` (a local mirror) when set and are `network` tests otherwise; `MULLIGAN_HF_CACHE=DIR` shares one HF cache
across network runs; `MULLIGAN_BITWISE_PARITY=1` makes the real-robot equivalence tests compare bitwise
(only on the machine class the references were recorded on). Details: [docs/README.md](docs/README.md#tests).

## License

The code is released under the MIT License ([LICENSE](LICENSE)). The Hugging Face datasets and models
are MIT at their release revisions ([docs/data.md](docs/data.md#licenses)).

MimicGen, a dependency used for the Square-Broad environment, is under the NVIDIA Source Code License,
which permits non-commercial use only (research or evaluation). `uv sync` installs it; commercial users
must remove it or obtain a license from NVIDIA. Vendored code (EXPO and RLPD, MIT) and other third-party
terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Upstream DROID, which the
robot project installs from a fork, states no license; the release ships none of its code.

## Citation

```bibtex
@misc{mulligan2026,
  title         = {Mulligan: Performance-Guided Data Collection for Efficient On-Robot Learning},
  author        = {Lars Ankile and Perry Dong and Rohan Bhowmik and Aneesh Muppidi and David D. Yuan and
                   Shuran Song and Chelsea Finn},
  year          = {2026},
  eprint        = {2610.05882},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.05882}
}
```

## Appendix: reproducibility by experiment

The columns are the four tiers of [What you can reproduce](#what-you-can-reproduce). "yes" means the tier works as described; "partial" is explained in the notes. Rows use the release IDs
of [docs/reproduce.md](docs/reproduce.md) (F = main-text figure, A = appendix item), which has the
commands and compute of each row; the paper section is in parentheses.

Tier 1 needs the frozen paper evidence for most rows ("yes (E)"): the builders read it from the
`mulligan/paper-evidence` dataset at its pin, or from a local copy at `MULLIGAN_PAPER_EVIDENCE`. Without
either (offline), these rebuild: Fig. 2, Fig. 3 (reads released videos from
Hugging Face), the Fig. 4 reset card, the 2,550 episode count, the A3 Cable
snapshot check, the A4 significance and A5 DIVL tables, the A18 RLPD ablation numbers, the A29 compute table and
the A30/A33 best-of-N checks, plus the Fig. 5 and Fig. 7 numbers quoted in the text
(`paper.stats.final_round`, `paper.stats.sim_contrasts`).

| Paper experiment | 1 Rebuild | 2 Recompute from HF | 3 Retrain / re-evaluate | 4 Re-run protocol | Notes |
|---|---|---|---|---|---|
| F1 teaser (Fig. 1) | yes (E) | - | - | - | Bucket counts digitized from a raster |
| F2 RLPD vs HiL-IDQL+Mulligan vs HiL-SERL (Fig. 2) | yes | - | RLPD: yes | HiL-SERL: yes | RLPD re-runs use robosuite 1.5.2 (see [above](#what-cannot-be-reproduced-and-why)) |
| F3 task sequences (Fig. 3) | yes (network) | - | - | - | Frames from released episodes |
| F4 reset card and ranges (Fig. 4) | card: yes; ranges: yes (E) | - | - | - | |
| F5 real headline (Fig. 5 top) | yes (E) | yes | yes | yes | 75 of 76 real checkpoints retrain from exactly their public training episodes; one is approximate |
| F5 simulation headline (Fig. 5 bottom) | yes (E) | yes | yes | yes | 58 recipes, 5 seeds each; DAgger rounds replay headless or run with a new operator |
| F6 throughput (Fig. 6) | yes (E) | partial | as F5 | as F5 | Success counts recomputed; durations from frozen evidence |
| F7 sim ablations, panels 1-2 (Fig. 7) | yes (E) | partial | yes | - | Per-seed counts in the public eval bundles |
| F7 real counterfactual ablation, panels 3-4 (Fig. 7) | yes (E) | - | yes | yes | |
| F8 collection success (Fig. 8) | yes (E) | yes | - | yes | 988 / 1,009 from the public collection ledgers |
| F9 intervention burden (Fig. 9) | yes (E) | - | - | yes | |
| N: 2,550 evaluation episodes | yes | yes | - | - | |
| Alg. 1 and Alg. 2 (Sec. 3, App. F) | - | - | sim starts: yes | yes | Sim start selection rebuilds the locked R1-R3 starts bit for bit; real locked manifests need recorded inputs not all in the release |
| A1, A2, A6 real detail, statistics, speed (App. B.1-B.3, D) | yes (E) | partial | as F5 | as F5 | Counts recomputed; paired tests from frozen evidence |
| A3 Cable snapshot and endpoint tests (App. B.2) | yes | partial | - | - | Cable points recomputed from the public label history |
| A4, A5, A8 sim significance, DIVL, sampler (App. B.2, J.4, B.5) | A4, A5: yes; A8: yes (E) | partial | yes | - | |
| A7 sim speed and throughput (App. B.4) | yes (E) | - | - | - | Archived aggregate |
| A9 Marker and Nut counterfactual ablation (App. B.6) | yes (E) | - | yes | yes | |
| A10, A15 substage, stage labels (App. B.7, C.3) | yes (E) | - | - | labeler only | Gemini labels are frozen; a new labeling run differs |
| A11-A13 collection, burden, data ledger (App. B.8-B.10, E.2) | yes (E) | partial | - | yes | |
| A14, A26-A28 authored tables and illustrations (App. C, E, F.4, H) | yes (E) | - | - | - | Authored |
| A16 RECAP (App. I.3) | yes (E) | - | - | - | The RECAP learner and recipe are not released |
| A17 HiL-SERL sessions (App. I.4) | yes (E) | - | no-human run | yes | Operator sessions give new results |
| A18 RLPD ablations (App. I.4) | yes | - | yes | - | Quoted numbers recomputed from the frozen per-seed eval curves |
| A19 real value learning, FQE (App. J.1) | yes (E) | - | critics: yes | - | FQE primitives only |
| A21-A23 critic objective, DIVL regimes, REDQ (App. J.3-J.5) | yes (E) | - | - | - | Archived aggregates |
| A24 bucket success rates (App. F.1) | yes (E) | - | - | - | Digitized from an archived raster |
| A25, A34 Sobol coverage, eval-grid term (App. F.2, F.5) | yes (E) | - | yes | - | The start design with the eval-grid term rebuilds the locked starts bit for bit |
| A29 training compute (App. G.7) | yes | - | - | - | Frozen run extract |
| A30, A33 best-of-N cost, choice of N (App. G.4, J.6) | yes | - | - | - | Frozen summaries, hash-checked |
| A32 critic training data (App. J.2) | yes (E) | - | yes | - | Human-only DIVL heads behind the released R3 agents |
