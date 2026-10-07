# Compute for the simulation experiments

All sim jobs need one GPU (Ampere or newer for training, see below), up to 12 CPU cores and at most
64 GB of host memory. Training is GPU-bound; grid evaluation and collection are mostly MuJoCo
simulation on CPU workers, with the policy on the GPU.

## GPU generation

The sim recipes train with bf16 autocast (`--training.amp_dtype=bfloat16`, plus TF32 and a
compiled actor): 53 of the 58 IDQL agent recipes and all 46 DIVL heads set it for every seed,
and the other five recipes set it for some seeds (`seed_args` in `configs/sim/recipes.json`).
Needs compute capability >= 8.0 (Ampere or newer); the trainer stops otherwise. On older GPUs pass
`--training.compile_actor=False` or `--training.amp_dtype=none` (numerics differ).

## Budgets

Training times per seed on one GPU (App. G.7; they include setup and the in-run 400-episode
evaluation):

| Job | Steps | Median GPU-h | Range | Fast GPU (H100) | Slower GPU (A40 / RTX 4500 Ada) |
|---|---|---|---|---|---|
| IDQL agent, Square-Narrow | 150k | 1.9 | 1.3-5.2 | 1.45 | 2.8-3.0 |
| IDQL agent, Square-Broad | 250k | 3.0 | 1.7-6.5 | 2.2 | 4.5-4.8 |
| DIVL head, Square-Narrow | 150k | 0.65 | 0.5-1.2 | 0.7 | 0.6-1.0 |
| DIVL head, Square-Broad | 250k | 0.97 | 0.8-1.9 | 1.0 | 0.9-1.5 |

A DIVL head is cheaper than its agent because the diffusion actor is frozen (only the Q
ensemble and the distributional value head are trained). Each job needs up to 12 CPU cores and
64 GB of host memory.

Whole-release totals at the median times: the 58 recipes x 5 seeds are about 720 GPU-h of
IDQL agents (28 Square-Narrow and 30 Square-Broad recipes) and 190 GPU-h of DIVL heads (46
recipes), about 900 GPU-h in all.

| Job | Resources | Time |
|---|---|---|
| Grid eval, Square-Narrow (8,000 starts) | 1 GPU, 12 CPUs (10 env workers), 24 GB | about 1 s per start; 2-8 h per checkpoint |
| Grid eval, Square-Broad (30,000 starts) | same, 4 shards | up to 14 h per shard |
| Autonomous collection (100 episodes) | 1 GPU, 10 env workers | up to 4 h |
| DAgger round (human, SpaceMouse) | workstation with a display; GPU, MPS or CPU for the policy | interactive; each arm collects until 100 no-CF (and 100 with-CF) successes |

A 10-worker evaluation runs one episode per start (at most 400 steps) with best-of-N action
selection (N=32; N=1 for the `auto-*-n1` baselines).

## Baselines

- One RLPD seed of `square-narrow-rlpd` (300k env steps) takes 1.5-4.5 h on one GPU; a 1M-step
  Square-Broad seed 8-14 h.
- A HiL-SERL no-human Square-Narrow run to 200k env steps takes under an hour on one GPU.
- The RLPD and HiL-SERL smoke runs ([baselines.md](baselines.md)) take a few minutes on one GPU; the
  simulation pipeline smoke test (`tests/sim/test_smoke.py`) takes a few minutes on CPU.

## One GPU

Run the scripts sequentially; each takes `--seed` (default: all five seeds in a row):

```sh
uv run scripts/sim/train_cell.sh square-narrow-r01-baseline --seed 1
uv run scripts/sim/train_divl_heads.sh square-narrow-r01-baseline --seed 1   # released parent
uv run scripts/sim/train_divl_heads.sh square-narrow-r01-baseline --seed 1 --parent local
uv run scripts/sim/eval_cell.sh square-narrow-r01-baseline --seed 1 --checkpoint local
```

Outputs go to `outputs/sim/` (set `MULLIGAN_OUTPUT_DIR` or `--output-dir` to change it).
The Square-Broad grid is sharded for clusters; on one machine either run it unsharded
(the default) or run the shards one after another and merge:

```sh
for k in 1 2 3 4; do
  uv run scripts/sim/eval_cell.sh square-broad-r01-mulligan --seed 1 --shard "$k" --n-shards 4
done
uv run scripts/sim/eval_cell.sh square-broad-r01-mulligan --seed 1 --merge
```

For a quick end-to-end check on a small GPU, shorten training with draccus overrides and
evaluate on a small grid (these numbers are not comparable to the paper):

```sh
uv run scripts/sim/train_cell.sh square-narrow-r01-baseline --seed 1 \
  --training.training_steps=2000 --eval.freq=2000 --eval.episodes=10
uv run python -m mulligan.sim.eval.grid_eval make-valid-sobol-manifest \
  --task square_narrow --num-points 50 --allow-empty-cells --output outputs/sim/grids/smoke50.json
uv run scripts/sim/eval_cell.sh square-narrow-r01-baseline --seed 1 --stage idql_agent \
  --checkpoint local --grid-file outputs/sim/grids/smoke50.json
```

## A cluster

Every training seed, every grid-eval shard and every collection job is independent, so
they map onto array jobs. `scripts/examples/slurm_train.sbatch` trains one seed per array
task (the array index is the seed):

```sh
sbatch --array=1-5 scripts/examples/slurm_train.sbatch square-broad-r02-mulligan
sbatch --array=1-5 --export=ALL,STAGE=divl_head \
    scripts/examples/slurm_train.sbatch square-broad-r02-mulligan
```

Fill in `--partition` and `--account` for your cluster and activate the project
environment on the compute node. The same pattern works for `eval_cell.sh` with
`--shard "$SLURM_ARRAY_TASK_ID" --n-shards 4` followed by one `--merge` job.
Models, datasets and grids are fetched from the Hugging Face Hub on first use; set
`HF_HOME` to node-local scratch if the shared filesystem is slow.
