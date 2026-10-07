#!/usr/bin/env bash
# Train one sim recipe: the IDQL agent (scalar IQL critic + diffusion actor), an RLPD agent or a
# HiL-SERL no-human session.
#
# Usage: scripts/sim/train_cell.sh <recipe_id> [--seed N] [--output-dir DIR] [FLAG ...]
#
# Runs the recipe's trainer (`python -m mulligan.sim.recipes trainer <recipe_id>`) with its
# arguments and the seed, once per seed (all recipe seeds unless --seed):
#   IDQL      mulligan.training.train, the datasets at their pinned revisions,
#             --training.seed=N; FLAGs are draccus overrides, e.g. --training.training_steps=200
#   RLPD      mulligan.baselines.rlpd.train --seed=N; FLAGs override, e.g. --max_steps=1500
#   HiL-SERL  mulligan.baselines.hilserl.nohuman --seed=N (learner + eval watcher + headless
#             actor on this machine); FLAGs override, e.g. --max_steps=6000
# Runs go to DIR/sim/train/<recipe_id>/<stage>/seed-N/ (IDQL: <run>/checkpoints/final_model;
# RLPD: eval.jsonl, resume/; HiL-SERL: the session directory). Rerunning an RLPD or HiL-SERL
# command resumes it; exit code 75 = stopped by SIGTERM with its state saved.
#
# Run inside the project environment (`uv run scripts/sim/train_cell.sh ...` or an
# activated venv). Environment: PYTHON (default: python), MULLIGAN_OUTPUT_DIR (default: outputs).
# Recipe ids: python -m mulligan.sim.recipes list
set -euo pipefail

read -r -a PY <<< "${PYTHON:-python}"
OUTPUT_DIR="${MULLIGAN_OUTPUT_DIR:-outputs}"

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  exit "${1:-2}"
}

# Run a command that prints a NUL-separated argv and read it into ARGV.
read_argv() {
  local tmp item
  tmp="$(mktemp)"
  if ! "$@" >"${tmp}"; then
    rm -f "${tmp}"
    echo "error: $* failed" >&2
    exit 1
  fi
  ARGV=()
  while IFS= read -r -d '' item; do ARGV+=("${item}"); done <"${tmp}"
  rm -f "${tmp}"
}

[[ $# -ge 1 ]] || usage
case "$1" in -h | --help) usage 0 ;; esac
RECIPE="$1"
shift
SEEDS=()
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEEDS=("$2"); shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h | --help) usage 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
read -r STAGE MODULE < <("${PY[@]}" -m mulligan.sim.recipes trainer "${RECIPE}")
[[ -n "${MODULE:-}" ]] || { echo "error: no trainer for ${RECIPE}" >&2; exit 1; }
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  while IFS= read -r seed; do SEEDS+=("${seed}"); done < <("${PY[@]}" -m mulligan.sim.recipes seeds "${RECIPE}")
  [[ ${#SEEDS[@]} -gt 0 ]] || { echo "error: no seeds for ${RECIPE}" >&2; exit 1; }
fi

for seed in "${SEEDS[@]}"; do
  ckpt_dir="${OUTPUT_DIR}/sim/train/${RECIPE}/${STAGE}/seed-${seed}"
  read_argv "${PY[@]}" -m mulligan.sim.recipes train-argv "${RECIPE}" \
    --stage "${STAGE}" --seed "${seed}" --checkpoint-dir "${ckpt_dir}"
  echo "== ${RECIPE} ${STAGE} seed ${seed} -> ${ckpt_dir}"
  PYTHONHASHSEED="${seed}" "${PY[@]}" -m "${MODULE}" "${ARGV[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
done
