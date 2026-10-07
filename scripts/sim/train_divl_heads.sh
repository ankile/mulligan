#!/usr/bin/env bash
# Retrain the frozen-actor DIVL head of one sim recipe from its IDQL parent.
#
# Usage: scripts/sim/train_divl_heads.sh <recipe_id> [--seed N] [--parent released|local|REF]
#                                        [--output-dir DIR] [DRACCUS_FLAG ...]
#
# Runs `python -m mulligan.training.train --policy.type=idql_divl
# --training.update_components=critic_value_only` with the recipe's arguments: the parent's
# actor and normalizer are loaded and frozen, only the Q ensemble and the distributional
# value head are trained. The parent (seed-matched) is
#   released  the released agent, hf://mulligan/sim-...-idql@<revision>/seed-N (default)
#   local     the agent trained by scripts/sim/train_cell.sh under DIR
#   REF       any local checkpoint directory or hf:// reference (needs --seed)
# All recipe seeds unless --seed. Other arguments are appended as draccus overrides.
# Checkpoints go to DIR/sim/train/<recipe_id>/divl_head/seed-N/<run>/checkpoints/final_model.
#
# Run inside the project environment. Environment: PYTHON (default: python),
# MULLIGAN_OUTPUT_DIR (default: outputs). Recipes with a DIVL head:
#   python -m mulligan.sim.recipes list --with-divl
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${MULLIGAN_OUTPUT_DIR:-outputs}"

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  exit "${1:-2}"
}

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

# The single final checkpoint written by train_cell.sh for one seed.
local_parent() {
  local seed="$1" matches=()
  shopt -s nullglob
  matches=("${OUTPUT_DIR}/sim/train/${RECIPE}/idql_agent/seed-${seed}"/*/checkpoints/final_model)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "error: expected one local final_model for ${RECIPE} seed ${seed}, found ${#matches[@]}" >&2
    exit 1
  fi
  echo "${matches[0]}"
}

[[ $# -ge 1 ]] || usage
case "$1" in -h | --help) usage 0 ;; esac
RECIPE="$1"
shift
SEEDS=()
PARENT="released"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEEDS=("$2"); shift 2 ;;
    --parent) PARENT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h | --help) usage 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  if [[ "${PARENT}" != released && "${PARENT}" != local ]]; then
    echo "error: an explicit --parent needs --seed" >&2
    exit 2
  fi
  while IFS= read -r seed; do SEEDS+=("${seed}"); done < <("${PYTHON}" -m mulligan.sim.recipes seeds "${RECIPE}")
  [[ ${#SEEDS[@]} -gt 0 ]] || { echo "error: no seeds for ${RECIPE}" >&2; exit 1; }
fi

for seed in "${SEEDS[@]}"; do
  ckpt_dir="${OUTPUT_DIR}/sim/train/${RECIPE}/divl_head/seed-${seed}"
  parent_args=()
  case "${PARENT}" in
    released) ;;
    local)
      parent="$(local_parent "${seed}")"
      parent_args=(--parent "${parent}")
      ;;
    *) parent_args=(--parent "${PARENT}") ;;
  esac
  read_argv "${PYTHON}" -m mulligan.sim.recipes train-argv "${RECIPE}" \
    --stage divl_head --seed "${seed}" --checkpoint-dir "${ckpt_dir}" ${parent_args[@]+"${parent_args[@]}"}
  echo "== ${RECIPE} divl_head seed ${seed} -> ${ckpt_dir}"
  PYTHONHASHSEED="${seed}" "${PYTHON}" -m mulligan.training.train "${ARGV[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
done
