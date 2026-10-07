# Documentation

Start with the [repository README](../README.md): what can be reproduced, setup, the Hugging Face data
and models, and the step-by-step pipelines. The pages below go into depth.

## Using the release

| Page | What it covers |
|---|---|
| [install.md](install.md) | uv install, extras, system packages, macOS notes, the robot workstation project |
| [reproduce.md](reproduce.md) | Per-experiment guide: rebuild, recompute, retrain or re-run each figure and table, with compute |
| [paper_figures.md](paper_figures.md) | Rebuilding and checking every figure and table; the paper-evidence inputs; platform differences |
| [data.md](data.md) | The Hugging Face datasets and models: names, pinned revisions, manifests, and `verify_results` |
| [compute.md](compute.md) | GPU, CPU and time budgets for the simulation experiments; running on a cluster |
| [sim.md](sim.md) | Simulation pipeline: environments, R0 teleop, DAgger with counterfactual replay, start design, grid evaluation |
| [real_robot.md](real_robot.md) | Real robot: retraining actors and critics from the released configs, loading released checkpoints, collection |
| [real_robot_eval.md](real_robot_eval.md) | Blind paired evaluation on the robot with `manifest_eval`, its outputs and the analysis after it |
| [station.md](station.md) | Setting up a robot station: ZED SDK, robot project, station identity file, camera roles, operator display |
| [baselines.md](baselines.md) | RLPD and HiL-SERL: recipes, offline data, the training loop, HiL-SERL sessions and the operator |
| [arena.md](arena.md) | Policy Arena: deploying your own (Convex, Vercel / Cloudflare Pages, robot client); the Mulligan snapshot's contents |

Experiment tracking with Weights & Biases is optional and off by default in every trainer and collector.
Turn it on with the trainer's W&B flag (`--use-wandb`, `wandb.enabled=true`, `--wandb=True` or
`wandb_mode: online`); runs then go to your own W&B account, under your default entity unless you set one,
and the project you choose. Nothing in reproducing the paper reads from W&B: data, checkpoints and
configs all come from Hugging Face and this repository.

## Hardware

| Page | What it covers |
|---|---|
| [hardware/spacemouse.md](hardware/spacemouse.md) | SpaceMouse driver, device permissions on Linux and macOS, a quick check |
| [hardware/droid_fork.md](hardware/droid_fork.md) | What the pinned DROID fork changes, stock vs patched Polymetis, NUC deployment |
| [hardware/objects.md](hardware/objects.md) | Task objects and table layout for Insert Marker, Thread Nut and Route Cable: placement, randomization, success, printing the marker holder, nut, peg and cable clips |
| [configs/real/station.env.example](../configs/real/station.env.example) | Template for the per-machine station identity file |

Other places with documentation: [`configs/README.md`](../configs/README.md) (where each kind of config
lives), [`configs/sim/README.md`](../configs/sim/README.md) (recipes incl. the baselines, grids, start
designs, paper mapping), [`configs/real/README.md`](../configs/real/README.md) (real trainer, eval and
start-design configs),
[`paper/appendix/README.md`](../paper/appendix/README.md) (appendix packages),
[`arena/README.md`](../arena/README.md) (Arena build) and [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Tests

```bash
pytest -m "not network and not gpu and not hardware"   # offline suite (CPU, what CI runs)
pytest -m network                                        # reads public Hugging Face repos anonymously
```

| Marker | Meaning |
|---|---|
| `network` | Downloads from Hugging Face (datasets, checkpoints, eval bundles) or other remote data, e.g. the Arena build's `bun install` from the npm registry. Each test has a wall-clock limit; `MULLIGAN_HF_CACHE=DIR` shares one HF cache across runs (default: a fresh temp dir per session) |
| `gpu` | Needs a CUDA GPU; not run in CI |
| `hardware` | Needs the robot or a SpaceMouse |
| `slow` | Long tests, e.g. the simulation pipeline smoke test (`tests/sim/test_smoke.py`, also `network`) |

| Variable | Effect |
|---|---|
| `MULLIGAN_PAPER_EVIDENCE=DIR` | Local mirror of the paper evidence (default: `mulligan/paper-evidence` at its pin, downloaded on first use) |
| `MULLIGAN_BITWISE_PARITY=1` | The real-robot equivalence tests (`tests/real/test_bon_equivalence.py`) compare bitwise instead of within their tolerances. Only meaningful on the machine class the references were recorded on (torch 2.11.0, x86 with AVX2 kernels); elsewhere the last float bits move |
| `MULLIGAN_CHROME` | Chrome or Chromium binary for the teaser build and the Arena browser check |

The Arena tests also need [bun](https://bun.sh) on `PATH`; without it they skip.
