#!/usr/bin/env bash
# Evaluate one seed of a sim recipe on its locked evaluation grid.
#
# Usage: scripts/sim/eval_cell.sh <recipe_id> --seed N [--stage idql_agent|divl_head]
#            [--checkpoint released|local|REF] [--num-action-samples N]
#            [--shard K --n-shards M | --merge] [--grid-file FILE] [--output-dir DIR]
#            [--eval-seed S] [--env-seed S]
#
# The grid (8,000 starts on Square-Narrow, 30,000 on Square-Broad; the equal-tile 8,000-start
# grid for Square-Narrow R0) is regenerated with `mulligan.sim.eval.grid_eval` on first use
# and its manifest_hash checked against configs/sim/recipes.json. Then runs
# `python -m mulligan.sim.eval.grid_eval eval` on the checkpoint:
#   released  the released checkpoint of --stage (default)
#   local     the final checkpoint written by train_cell.sh / train_divl_heads.sh under DIR
#   REF       any local checkpoint directory or hf:// reference
# --stage defaults to the stage the paper's headline reports (DIVL head for the human-in-the-
# loop arms, IDQL agent for the autonomous baselines).
# --num-action-samples defaults to the recipe's deployment N (32; 1 for the N=1 baselines).
# With --shard K --n-shards M only shard K of M runs; --merge then combines the M shards.
# Results go to DIR/sim/eval/<recipe_id>/<stage>/seed-N/n<N>/results.json. A --grid-file
# (e.g. a small smoke grid) is used as given, without the hash check; its results go to
# .../n<N>/grid-<file stem>-<manifest_hash[:8]>/ so they never mix with the locked grid's, and
# --merge then needs --n-shards.
# Environment resets and policy sampling are unseeded by default. --eval-seed S seeds the policy's
# action sampling per batch of starts and --env-seed S the reset noise (per start, from S and the
# point index); pass both for a repeatable run on the same machine. Seeded results go to .../n<N>[/grid-...]/eval-seed-S,
# .../env-seed-S or .../eval-seed-S-env-seed-S, apart from the unseeded ones.
#
# Run inside the project environment. Environment: PYTHON (default: python),
# MULLIGAN_OUTPUT_DIR (default: outputs).
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

manifest_hash() {
  "${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["manifest_hash"])' "$1"
}

[[ $# -ge 1 ]] || usage
case "$1" in -h | --help) usage 0 ;; esac
RECIPE="$1"
shift
SEED=""
STAGE=""
CHECKPOINT="released"
NUM_SAMPLES=""
SHARD=""
N_SHARDS=""
MERGE=0
GRID_FILE=""
EVAL_SEED=""
ENV_SEED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --num-action-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --shard) SHARD="$2"; shift 2 ;;
    --n-shards) N_SHARDS="$2"; shift 2 ;;
    --merge) MERGE=1; shift ;;
    --grid-file) GRID_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --env-seed) ENV_SEED="$2"; shift 2 ;;
    -h | --help) usage 0 ;;
    *) echo "error: unknown argument $1" >&2; usage ;;
  esac
done
[[ -n "${SEED}" ]] || { echo "error: --seed is required" >&2; usage; }
for value in "${EVAL_SEED}" "${ENV_SEED}"; do
  [[ -z "${value}" || "${value}" =~ ^[0-9]+$ ]] || { echo "error: seeds must be non-negative integers, got ${value}" >&2; exit 2; }
done
[[ -n "${STAGE}" ]] || STAGE="$("${PYTHON}" -m mulligan.sim.recipes default-stage "${RECIPE}")"

read -r GRID_ID GRID_HASH GRID_SHARDS DEPLOY_N < <("${PYTHON}" -m mulligan.sim.recipes grid "${RECIPE}")
[[ -n "${GRID_HASH:-}" ]] || { echo "error: no grid for ${RECIPE}" >&2; exit 1; }
GRID_SUBDIR=""
GRID_LABEL="${GRID_ID}"
if [[ -n "${GRID_FILE}" ]]; then
  [[ -f "${GRID_FILE}" ]] || { echo "error: --grid-file ${GRID_FILE} not found" >&2; exit 1; }
  got="$(manifest_hash "${GRID_FILE}")"
  stem="$(basename "${GRID_FILE}" .json)"
  GRID_SUBDIR="/grid-${stem}-${got:0:8}"
  GRID_LABEL="grid file ${GRID_FILE} (manifest_hash ${got:0:8}..., not the locked ${GRID_ID})"
else
  GRID_FILE="${OUTPUT_DIR}/sim/grids/${GRID_ID}.json"
  if [[ ! -f "${GRID_FILE}" ]]; then
    mkdir -p "$(dirname "${GRID_FILE}")"
    read_argv "${PYTHON}" -m mulligan.sim.recipes grid-generate-argv "${GRID_ID}" --output "${GRID_FILE}"
    echo "== generating ${GRID_ID} -> ${GRID_FILE}"
    "${PYTHON}" -m mulligan.sim.eval.grid_eval "${ARGV[@]}"
  fi
  got="$(manifest_hash "${GRID_FILE}")"
  if [[ "${got}" != "${GRID_HASH}" ]]; then
    echo "error: ${GRID_FILE} has manifest_hash ${got}, recipes.json locks ${GRID_HASH}" >&2
    exit 1
  fi
fi

case "${CHECKPOINT}" in
  released)
    CHECKPOINT="$("${PYTHON}" -m mulligan.sim.recipes checkpoint "${RECIPE}" --stage "${STAGE}" --seed "${SEED}")"
    ;;
  local)
    shopt -s nullglob
    matches=("${OUTPUT_DIR}/sim/train/${RECIPE}/${STAGE}/seed-${SEED}"/*/checkpoints/final_model)
    shopt -u nullglob
    if [[ ${#matches[@]} -ne 1 ]]; then
      echo "error: expected one local final_model for ${RECIPE} ${STAGE} seed ${SEED}, found ${#matches[@]}" >&2
      exit 1
    fi
    CHECKPOINT="${matches[0]}"
    ;;
esac

[[ -n "${NUM_SAMPLES}" ]] || NUM_SAMPLES="${DEPLOY_N}"
SEED_SUBDIR=""
seed_args=()
if [[ -n "${EVAL_SEED}" ]]; then
  SEED_SUBDIR="eval-seed-${EVAL_SEED}"
  seed_args+=(--eval-seed "${EVAL_SEED}")
fi
if [[ -n "${ENV_SEED}" ]]; then
  SEED_SUBDIR="${SEED_SUBDIR:+${SEED_SUBDIR}-}env-seed-${ENV_SEED}"
  seed_args+=(--env-seed "${ENV_SEED}")
fi
OUT="${OUTPUT_DIR}/sim/eval/${RECIPE}/${STAGE}/seed-${SEED}/n${NUM_SAMPLES}${GRID_SUBDIR}${SEED_SUBDIR:+/${SEED_SUBDIR}}"

if [[ ${MERGE} -eq 1 ]]; then
  if [[ -z "${N_SHARDS}" ]]; then
    [[ -z "${GRID_SUBDIR}" ]] || { echo "error: --merge with --grid-file needs --n-shards" >&2; exit 2; }
    N_SHARDS="${GRID_SHARDS}"
  fi
  echo "== merging ${N_SHARDS} shards of ${RECIPE} ${STAGE} seed ${SEED} N=${NUM_SAMPLES} on ${GRID_LABEL}${SEED_SUBDIR:+ (${SEED_SUBDIR})}"
  "${PYTHON}" -m mulligan.sim.eval.grid_eval merge \
    --point-manifest "${GRID_FILE}" --output-dir "${OUT}" --n-shards "${N_SHARDS}"
  exit 0
fi

shard_args=()
if [[ -n "${SHARD}" || -n "${N_SHARDS}" ]]; then
  [[ -n "${SHARD}" && -n "${N_SHARDS}" ]] || { echo "error: --shard and --n-shards go together" >&2; exit 2; }
  shard_args=(--shard "${SHARD}" --n-shards "${N_SHARDS}")
fi
read_argv "${PYTHON}" -m mulligan.sim.recipes eval-argv "${RECIPE}" \
  --checkpoint "${CHECKPOINT}" --grid-file "${GRID_FILE}" --output-dir "${OUT}" \
  --num-action-samples "${NUM_SAMPLES}" ${shard_args[@]+"${shard_args[@]}"}
echo "== ${RECIPE} ${STAGE} seed ${SEED} N=${NUM_SAMPLES} on ${GRID_LABEL}${SEED_SUBDIR:+ (${SEED_SUBDIR})} -> ${OUT}"
"${PYTHON}" -m mulligan.sim.eval.grid_eval "${ARGV[@]}" ${seed_args[@]+"${seed_args[@]}"}
