# Baselines: RLPD and HiL-SERL

Two online-RL baselines on the simulated Square tasks, both on the same JAX SAC/RLPD agent
(`mulligan.baselines.rlpd.agent`). The RLPD code is based on the EXPO source code
(<https://github.com/pd-perry/EXPO>, MIT), which builds on the RLPD source code
(<https://github.com/ikostrikov/rlpd>, MIT); see the file headers and `THIRD_PARTY_NOTICES.md`.

Both are recipes in `configs/sim/recipes.json`, next to the IDQL recipes, and train with
`scripts/sim/train_cell.sh` like them. Commands assume `uv sync --frozen` and run from the repo root.

| Recipe | Baseline | Offline data | Steps | Paper |
|---|---|---|---|---|
| `square-narrow-rlpd` | RLPD | 100 Square-Narrow R0 teleop demos | 300k | Fig. 2, "RLPD" (0.84 mean final success, 5 seeds) |
| `square-broad-rlpd` | RLPD | 200 Square-Broad R0 teleop demos | 1M | Fig. 2, "RLPD" (0.004) |
| `square-narrow-rlpd-robomimic-ph` | RLPD | robomimic Square PH, 200 demos | 300k | Fig. 2, "RLPD, released demos" (0.98) |
| `square-broad-rlpd-mimicgen-core` | RLPD | MimicGen core `Square_D1`, 1,000 demos | 1M | Fig. 2, "RLPD, released demos" (0.44) |
| `square-narrow-rlpd-early-kill` | RLPD, early kill | as `square-narrow-rlpd` | 300k | App. I.4 (0.78) |
| `square-narrow-rlpd-utd40` | RLPD, UTD 40 | as `square-narrow-rlpd` | 300k | App. I.4 (0.81) |
| `square-narrow-hilserl` | HiL-SERL | as `square-narrow-rlpd` | 300k | Fig. 2, "HiL-SERL" (0.94); App. I.4 |
| `square-broad-hilserl` | HiL-SERL | as `square-broad-rlpd` | 1M | Fig. 2, "HiL-SERL" (0.02); App. I.4 |

The paper's RLPD curves were run with robosuite 1.4.1; this release's simulator is robosuite 1.5.2, so
a re-run can differ slightly. No baseline checkpoints are released; the paper's curves are in
`paper/data/online_rl/`.

## RLPD

```bash
python -m mulligan.sim.recipes list --family rlpd
scripts/sim/train_cell.sh square-narrow-rlpd --seed 1     # one seed, one GPU
scripts/sim/train_cell.sh square-narrow-rlpd              # seeds 1-5 in turn
```

A run writes `outputs/sim/train/<recipe>/rlpd_agent/seed-N/`: `flags.json`, `train.jsonl`, `eval.jsonl`
(the learning curve) and the resume state `resume/state.pkl`. The state is saved at every evaluation and
on SIGTERM (exit code 75); rerunning the same command continues from it. Flags after the recipe id
override the recipe's, e.g. `--wandb=True` (W&B project `mulligan-rlpd`, your default entity) or
`--checkpoint_model=True` (keeps `checkpoints/agent_<step>.msgpack`). A few-minute check of the whole
pipeline:

```bash
scripts/sim/train_cell.sh square-narrow-rlpd --seed 1 --output-dir outputs/smoke \
    --max_steps=1500 --start_training=500 --eval_interval=1000 --eval_episodes=2
```

### Data

`--offline_data` names the demos (`mulligan.baselines.rlpd.datasets`). Each source is fetched on first
use and checked against a pinned content sha256 of the arrays the agent trains on:

| `--offline_data` | Source | Transitions |
|---|---|---|
| `teleop` | the task's R0 teleop demos, `mulligan/sim-<task>-c00-teleop-baseline` at its pin | 16,233 / 35,486 |
| `robomimic_ph` (Square-Narrow) | robomimic Square PH `low_dim_v141.hdf5` (53 MB, cached in `~/.cache/mulligan/rlpd`) | 30,154 |
| `mimicgen_core` (Square-Broad) | Hugging Face `amandlek/mimicgen_datasets` `core/square_d1.hdf5` (1.7 GB, Hub cache) | 152,400 |

A path to any robomimic low_dim `.hdf5` with the same observation keys also works. Observations (23-D
Square-Narrow, 26-D Square-Broad) are `robot0_eef_pos (3), robot0_eef_quat (4), robot0_gripper_qpos (2),
object`, where `object` is `nut_pos, nut_quat, nut_to_eef_pos, nut_to_eef_quat[, peg_pos]`. The env builds
the same vector by key from the robosuite observation (`tests/baselines/rlpd/test_env.py`).

### Loop

`python -m mulligan.baselines.rlpd.train` runs `start_training` (5,000) uniform-random steps, then one
`SACLearner.update` of `utd_ratio` (20) minibatches of 256 per env step, each batch half offline and half
online. Reward is 1 at success and 0 otherwise; an episode ends at the first success or at 400 steps,
and every stored transition that ends an episode (success, timeout or early kill) has mask 0. Every 10k
steps (step 0 included) 50 evaluation episodes run with deterministic actions, episode `i` reset with
seed `10000 + i`, on a persistent pool of CPU workers. Agent: hidden dims (256, 256, 256), 10 critics
with 2 sampled for the target, critic LayerNorm, no entropy term in the critic backup.

`--early_kill` (Square-Narrow only) ends a training episode as a failure when the nut never rose 2 cm
above its reset height by step 300 (150 before 100k training steps), or when, after that, it stays
below 5 mm above its reset height, more than 3 cm from the peg, for 100 consecutive steps. The reference
height is the nut's spawn height (z = 0.89 m), before it drops onto the table (about 0.83 m).

Before training, `check_controller_semantics` compares the live controller with the offline data's
robomimic `env_args` (OSC pose, delta input in [-1, 1], output_max [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
kp 150, damping 1, 20 Hz, no interpolation, robot base aligned with the world) and fails on any
difference.

## HiL-SERL

HiL-SERL splits RLPD into a learner (GPU), an actor that steps the env with the learner's latest policy
and lets an operator take over with a SpaceMouse, and an eval watcher (CPU) that evaluates every
checkpoint. A session is a directory (`mulligan/baselines/hilserl/session.py` documents the layout);
every role can be killed and restarted on the same session.

The paper's sessions follow two steps: a **no-human run** (the actor never intervenes), then a **fork
after takeoff**: once the no-human policy starts succeeding, fork the session at that checkpoint and an
operator continues the fork. The paper forked Square-Narrow at 150k (eval 0.36; the fork reached 0.94 at
300k) and Square-Broad at 200k (0.02 at 500k).

The no-human run trains like any recipe; the learner, eval watcher and a headless actor run on one
machine (`mulligan.baselines.hilserl.nohuman`):

```bash
scripts/sim/train_cell.sh square-narrow-hilserl     # -> outputs/sim/train/square-narrow-hilserl/hilserl_agent/seed-1/
```

The actor runs 13 episodes of the untrained policy (about 5.2k steps; updates start at 5,000), waits
until the learner is past its JIT compiles, then runs the policy alone, unpaced, to `max_steps`.
Evaluations (50 episodes, seeds 10000-10049, per 10k-step checkpoint) go to `eval/ledger.jsonl`. A
few-minute check: append `--max_steps=6000 --start_training=500 --eval_interval=3000 --eval_episodes=2
--eval_workers=2` and give it its own `--output-dir` (a session refuses a restart with a different pinned
config).

Fork and operator session, each role with the recipe's config:

```bash
S=outputs/sim/train/square-narrow-hilserl/hilserl_agent/seed-1
scripts/hilserl/fork.sh $S outputs/hilserl/narrow_fork150k 150000
# on the GPU machine
scripts/hilserl/learner.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
scripts/hilserl/eval_watcher.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
# on the operator's machine (a copy of the forked session; localhost through an ssh tunnel)
scripts/hilserl/actor.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k <learner host>
```

| Role | Script | Where |
|---|---|---|
| learner | `scripts/hilserl/learner.sh <recipe> <session>` | GPU machine; serves on the recipe's `port` and `port+1`, writes `learner/endpoint.json` |
| eval watcher | `scripts/hilserl/eval_watcher.sh <recipe> <session>` | CPU, next to the learner |
| actor | `scripts/hilserl/actor.sh <recipe> <session> <learner_ip>` | the operator's machine (SpaceMouse, MuJoCo viewer; `PYTHON="uv run mjpython"` on macOS) |
| fork | `scripts/hilserl/fork.sh <src> <dst> <step>` | anywhere the source session is staged |

Every script prints its usage with `--help`; flags after the arguments override the recipe's
(`--max_steps 500000`, `--port 5688`, ...). Without a SpaceMouse, `actor.sh ... --no-spacemouse
--no-render --unpaced` runs the actor headless. The learner does exactly one update call per env step
beyond 5,000 and the actor waits at an episode boundary when the learner owes more than 400 calls. The
demos are the RLPD recipe's `teleop` data.

### Operating the actor

The policy drives until the operator deflects the SpaceMouse puck or toggles the gripper (left button);
control stays with the operator while deflecting plus a 0.5 s latch. The SpaceMouse gains match the teleop
collector that recorded the demos (`actor.sh`).

| Key | Effect |
|---|---|
| `h` | hold the takeover on/off |
| `r` | redo: restore the sim to the start of the current or last intervention bout and drop its steps |
| `x` | end the episode now as a failure |
| `p` | pause / resume |
| `t` | toggle real-time 20 Hz pacing vs as fast as possible |
| `q` | finish the current episode, flush, exit |

On macOS the key listener needs Input Monitoring permission for the terminal, and keys count only while the
MuJoCo viewer has focus.

- With `demo_gate: success` (the default) intervened transitions always enter the online buffer, and they
  enter the demo half of each batch only if the episode succeeds. `none` always adds them (HiL-SERL).
- The idle filter skips steps inside an intervention bout where the operator holds still (no puck input,
  gripper settled, nut at rest): the actor steps the sim but does not record them or advance the episode
  clock. `--no-idle-filter` turns it off.
- Timeouts get mask 0 (`truncation_bootstrap: false`), as in RLPD.
- The actor refuses a learner whose code hash (`hilserl_sha`: this package minus `tools/`, plus the agent)
  or pinned config differs. The actor's episode log is the source of truth; unacknowledged episodes are
  re-pushed on reconnect.

W&B is off by default; `--wandb_mode online` (project `mulligan-hilserl`, your default entity unless
`--wandb_entity` is set) logs `train/*`, `eval/*`, `hil/*` and `buffer/*`.

## Tests

- `tests/baselines/rlpd/test_agent_reference.py` (network): the agent reproduces its recorded reference
  updates bitwise on CPU (fixture `fixtures/rlpd_sac_parity.npz` of `mulligan/paper-evidence`, sha256 in
  `tests/baselines/rlpd/fixtures.json`).
- `tests/baselines/rlpd/`: env semantics and a recorded reset/step trace, the import closure, the offline
  sources (network), a train/resume smoke.
- `tests/baselines/hilserl/`: session, ledger, governor, transport, kill/resume of the split system, the
  recipe configs.
- `tests/baselines/test_launch_scripts.py` and `tests/sim/test_recipes.py`: the scripts and recipes.
