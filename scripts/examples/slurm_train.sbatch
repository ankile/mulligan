#!/usr/bin/env bash
# Example SLURM wrapper: train one seed of a sim recipe per array task.
#
#   sbatch --array=1-5 scripts/examples/slurm_train.sbatch square-narrow-r01-baseline
#   sbatch --array=1-5 --export=ALL,STAGE=divl_head \
#       scripts/examples/slurm_train.sbatch square-narrow-r01-baseline
#   sbatch --array=1-5 scripts/examples/slurm_train.sbatch square-narrow-rlpd
#
# The array index is the seed. By default it runs scripts/sim/train_cell.sh (the recipe's
# IDQL, RLPD or HiL-SERL agent); STAGE=divl_head runs scripts/sim/train_divl_heads.sh
# (parent: the released agent). Budgets per seed (docs/compute.md): Square-Narrow agent
# ~2 GPU-h, Square-Broad agent ~3 GPU-h, DIVL head ~1 GPU-h, RLPD 3-14 h (an RLPD or
# HiL-SERL run stopped by the time limit exits 75 with its state saved; resubmit to resume).
# Replace the placeholders below with your cluster's values.
#
#SBATCH --job-name=mulligan-sim-train
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

RECIPE="${1:?usage: sbatch --array=1-5 scripts/examples/slurm_train.sbatch <recipe_id> [FLAG ...]}"
shift
SEED="${SLURM_ARRAY_TASK_ID:?submit as an array job; the array index is the seed}"
STAGE="${STAGE:-train}"

# Run from the repository root; activate the project environment first if it is not the
# default python on the compute node (e.g. `source .venv/bin/activate`).
cd "${SLURM_SUBMIT_DIR}"

case "${STAGE}" in
  train) exec scripts/sim/train_cell.sh "${RECIPE}" --seed "${SEED}" "$@" ;;
  divl_head) exec scripts/sim/train_divl_heads.sh "${RECIPE}" --seed "${SEED}" "$@" ;;
  *) echo "error: STAGE must be train (default) or divl_head, got ${STAGE}" >&2; exit 2 ;;
esac
