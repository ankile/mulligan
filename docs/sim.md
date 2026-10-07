# Simulation pipeline

This page covers the simulated Square tasks: the environments, collection (R0 teleop, DAgger with
counterfactual replay), splitting, training, start-state design and grid evaluation. Commands assume
`uv sync --frozen` (plus `--extra teleop` for the SpaceMouse) and run from the repo root.

## Tasks and environments

| Public name | robosuite env | Datasets record | Notes |
|---|---|---|---|
| `square_narrow` (Square-Narrow) | `NutAssemblySquare` | task `NutAssemblySquare_Panda` | robosuite 1.5.2 task |
| `square_broad` (Square-Broad) | `Square_D1` | task `Square_D1_Panda` | MimicGen task, nut and peg randomized |

`mulligan.sim.envs.create_robosuite_env` is the only env factory. It accepts the public names or the
robosuite IDs, uses a Panda with robosuite's default composite controller at 20 Hz, sparse reward,
`ignore_done=True` and soft resets (`hard_reset=False`); callers enforce the 400-step horizon. The
task keys `square_narrow` / `square_broad` are what manifests and data record; the robosuite env IDs
are the ones in the table above.

Three patches are installed explicitly by `register_square_environments()` (called by the factory and
by every vec-env worker), see `mulligan/sim/_patches.py`:

- MimicGen was written for robosuite 1.4; a shim aliases the removed `SingleArmEnv` to `ManipulationEnv`.
- robosuite's relative object sensors (`nut_to_eef_pos`) returned zeros on the first observation after
  a reset.
- MimicGen places the Square_D1 peg only when the model is built, so with soft resets every env instance
  kept one peg position. The patch re-samples the peg on every reset, before the nut is placed.

MimicGen is under the NVIDIA Source Code License (non-commercial); it is a dependency, not vendored.

### Seeding

Resets are unseeded by default (robot init noise; the policy's diffusion sampling and BoN candidates are
seeded separately, see [Grid evaluation](#grid-evaluation)). For reproducible starts pass `seed`:

```python
from mulligan.sim.envs import create_robosuite_env
from mulligan.sim.vec_env import AsyncVectorEnv

env = create_robosuite_env("square_broad", use_render_wrapper=False, seed=0)
vec = AsyncVectorEnv([make_env] * 8, seed=0)   # env i is seeded with 0 + i
```

A seeded env reseeds its own `rng`, the MimicGen placement generators and a private generator for the
Square_D1 peg, so seeded sync and async vec envs produce the same starts. Grid evaluation pins the
object placement of every start explicitly; seeding only affects the remaining randomness.

### Rendering

Headless Linux: `MUJOCO_GL=egl` (the async vec env then uses `spawn` workers). macOS: the viewer needs
`mjpython` instead of `python` (MuJoCo creates its window on the main thread); offscreen camera renders
use a CGL context (`mulligan/sim/render_cgl.py`), which lets the viewer and camera observations run
together. `configure_viewer_shadows(env)` turns on viewer shadows and hides helper geoms without
changing recorded camera images.

## One round, step by step

The scripts in `scripts/sim/` read `configs/sim/recipes.json` (the paper's 58 training recipes, the RLPD
and HiL-SERL baseline recipes ([baselines.md](baselines.md)) and the R1-R3 round specs, with every dataset
and checkpoint pinned to a `mulligan/*` HF revision; schema in `configs/sim/README.md`). They call plain `python` (set `PYTHON`, or run them with `uv run`) and write
under `outputs/` (`MULLIGAN_OUTPUT_DIR`). Compute budgets are in `docs/compute.md`.

| Step | Script | Module |
|---|---|---|
| R0 teleop (human) | none (see below) | `mulligan.sim.collect.teleop` |
| DAgger collection, R1-R3 | `collect_round.sh dagger --task T --round N` | `mulligan.sim.collect.dagger` |
| Policy rollouts for the next start design | `collect_round.sh rollouts ...` | `mulligan.sim.collect.rollouts` |
| Split into per-arm datasets | `split_round.sh --task T --round N` | `mulligan.data.split_protocol_quota`, `mulligan.data.split_blind` |
| Next round's starts | `python -m mulligan.sampling.sim_design --config configs/sim/<task>/rNN_sampler.yaml --out DIR [--check]` | `mulligan.sampling` |
| Train an agent / DIVL head | `train_cell.sh <recipe>`, `train_divl_heads.sh <recipe>` | `mulligan.training.train` |
| Grid evaluation | `eval_cell.sh <recipe> --seed N` | `mulligan.sim.eval.grid_eval` |

Human-in-the-loop collection cannot be replayed exactly; a new operator produces a new campaign, with
different numbers. The released recordings are the paper's data.

### R0 teleoperation

```bash
mjpython -m mulligan.sim.collect.teleop --r0-preset square_narrow   # or square_broad
```

(`python` instead of `mjpython` on Linux.) `--r0-preset` sets the flags of the paper's R0 session:
the blinded mix of uniform and Sobol starts in `data/sim/start_manifests/<task>/r00/blind_inputs/`
in file order, cameras `agentview,robot0_eye_in_hand`, auto-save on success, 200 (Square-Narrow) or
400 (Square-Broad) episodes, SpaceMouse sensitivity 1.2/1.2 on Square-Narrow and the defaults
(position 1.0, rotation 1.5) on Square-Broad. Flags given explicitly override it. The Square-Narrow
session spelled out:

```bash
mjpython -m mulligan.sim.collect.teleop --env NutAssemblySquare --robot Panda \
    --cameras "agentview,robot0_eye_in_hand" --save-data \
    --dataset-path outputs/sim/data --dataset-name sim-square-narrow-c00-teleop-mixed \
    --pos-sensitivity 1.2 --rot-sensitivity 1.2 --auto-save-on-success --target-episodes 200 \
    --sampler list --no-sampler-shuffle \
    --initial-states-file data/sim/start_manifests/square_narrow/r00/blind_inputs/square_narrow_r0_mixed.json
```

The recording lands in `outputs/sim/data/sim-<task>-c00-teleop-mixed`, where `split_round.sh --round 0`
reads it and splits it by its manifest. The SpaceMouse drives the end
effector; the left button toggles the gripper. Pushing to the Hub is optional
(`--push-to-hub --hub-namespace NS`). The SpaceMouse needs the `teleop` extra (`uv sync --extra
teleop`); see `docs/hardware/spacemouse.md` for device setup.

### DAgger with counterfactual replay

`collect_round.sh dagger` runs the blinded collector with the round's start list and routing manifest
(`data/sim/start_manifests/<task>/rNN/`), the previous round's baseline and Mulligan agents (N=32
best-of-N) and the adaptive no-CF / with-CF protocol quota. Keys: `h` take over / hand back, `1` success,
`0` recoverable failure, `9` terminal failure, `d` discard; between episodes `n` next start, `c`
counterfactual replay of the same start (human first; granted only while the with-CF quota allows it),
`q` quit. The ledger that the splitter needs is written to `outputs/sim/ledgers/`.

All human input goes through one interface, `mulligan.sim.collect.utils.Operator`; `--operator
module:factory` plugs in another implementation. `mulligan.sim.collect.replay_operator` replays the
recorded human segments of a released collection, so a round runs headless without a person:

```bash
scripts/sim/collect_round.sh dagger --task square_narrow --round 1 --operator replay \
    --replay-max-episodes 2
```

It matches each placed start to a recorded episode by pose, lets the live policy drive for the recorded
number of steps, then replays the recorded human actions open loop. The policy samples its own actions, so
the replayed episodes differ from the recording; it reproduces the protocol, not the data. A counterfactual
recording is replayed once; when the quota redraws the start of a failed fresh episode, that start's
policy-first recording is replayed again. The collector refuses the replay operator without
`--auto-save-on-success` (the recipes always pass it). `--replay-dataset` replays another recording:
`REPO@REVISION`, a released `mulligan/*` dataset (read at its `release/revisions.json` pin) or a local
dataset directory such as `outputs/sim/data/sim-square-narrow-c01-dagger-mixed`.

### Start-state design (Alg. 2)

`mulligan.sampling.select_initial_states` implements the sim start selection: failed starts of the
previous round's rollouts are promoted verbatim, Square-Narrow R1-R3 add 20 local perturbations (one around
the start of each of the 20 longest diagnostic episodes), and the rest is filled by hardness-weighted farthest-point sampling
`d(c, .) (1 + beta h(c))` with periodic yaw, disjoint from earlier rounds and the eval grid, under a p95
coverage guardrail. The eval-grid per-cell term is an explicit input that is off by default; the round
configs turn it on exactly where App. F.5 says (Square-Narrow R3, Square-Broad R2-R3). Each
`configs/sim/<task>/rNN_sampler.yaml` rebuilds the locked collection-start manifest of that round
bit for bit (`--check`).

### Grid evaluation

The paper evaluates every checkpoint on locked Sobol grids: 8,000 valid starts on Square-Narrow and
30,000 on Square-Broad (plus an equal-tile 8,000-start grid for the two Square-Narrow R0 cells).
The "Evaluation grids" section of `configs/sim/README.md` gives the regeneration commands and hashes; the grids are also in the
public eval bundles. `scripts/sim/eval_cell.sh` generates the grid of a recipe on first use at
`outputs/sim/grids/<grid id>.json` (`square_narrow_sobol8k`, `square_broad_sobol30k` or
`square_narrow_equal_tile`; `python -m mulligan.sim.recipes grid <recipe>` prints a recipe's grid id and
hash). To evaluate any checkpoint directly, generate the grid first (a few seconds; skip it if
`eval_cell.sh` already wrote the file), then evaluate:

```bash
python -m mulligan.sim.eval.grid_eval make-valid-sobol-manifest \
    --task square_narrow --num-points 8000 --seed 2026052402 \
    --output outputs/sim/grids/square_narrow_sobol8k.json
MUJOCO_GL=egl python -m mulligan.sim.eval.grid_eval eval \
    --artifact-path hf://mulligan/sim-square-narrow-r01-mulligan-divl@05e15eb4f0324587337fc7a8e3374398c6044443/seed-1 \
    --point-manifest outputs/sim/grids/square_narrow_sobol8k.json \
    --output-dir results/narrow-r01-divl/seed-1 --num-action-samples 32 --num-envs 10
```

`--shard-idx k --n-shards n` evaluates a contiguous slice (shard 1 of 10 is the first 800 starts).
`grid_eval merge <dir> [<dir> ...]` combines the shards of each output directory (the grid and shard
count come from the shard files); `--auto-find <root>` merges every unmerged directory below a root,
`--status` reports which shards are missing or unfinished, and `--verify <root>` checks that every merged
`results.json` covers its whole grid. Checkpoints are local directories
or `hf://<org>/<repo>@<revision>/<subdir>` references (without `@<revision>` the pin comes from
`release/revisions.json`).

Evaluations are unseeded by default. Two flags seed the two random streams (the start itself comes
from the grid); `eval_cell.sh` forwards both:

- `--eval-seed S` seeds the policy's sampling (diffusion noise and the best-of-N candidates): before
  each batch of `--num-envs` starts it seeds Python, NumPy and torch (CPU and CUDA) from a sha256 of `S`
  and the batch's point indices.
- `--env-seed S` seeds the robot's reset noise: before each batch, every start's env is reseeded from `S`
  and the start's point index, so a resumed run repeats the uninterrupted one.

With both, rerunning the same command (same shard, `--num-envs` and machine, from an empty output
directory) repeats every per-start outcome. `eval_cell.sh` writes seeded results to
`.../n<N>/eval-seed-S-env-seed-S/` (or the one seed given), apart from the unseeded ones, and `merge`
refuses shards whose seeds differ.

### Tests

`pytest tests/sim` covers all of this on CPU; `tests/sim/test_smoke.py` (marked `network`, `slow`) runs
the pipeline end to end with tiny settings: download, 200 training steps, a 2-start replayed DAgger round,
a 10-start grid evaluation and one plot.
