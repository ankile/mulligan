#!/usr/bin/env bash
# Split one sim round's mixed collection into the per-arm training datasets.
#
# Usage: scripts/sim/split_round.sh --task T --round N [--index I] [--source-root DIR]
#            [--ledger FILE] [--output-dir DIR] [--push] [-- SPLITTER_FLAG ...]
#
# Rounds 1-3 split the blinded DAgger collection by the protocol-quota ledger
# (`python -m mulligan.data.split_protocol_quota`) into no-CF and with-CF datasets per arm.
# Round 0 splits the blinded teleop collection by its manifest
# (`python -m mulligan.data.split_blind`); Square-Broad R0 has two splits (index 0: all 200
# episodes per arm, index 1: the first 100). All splits of the round run unless --index.
# Defaults match collect_round.sh: the collection in DIR/sim/data/<name>, the ledger in
# DIR/sim/ledgers/<task>_r<N>_ledger.jsonl. Outputs go to DIR/sim/splits/<task>_r<N>_<index>/;
# --push also pushes them to the HF Hub. Flags after `--` are appended to the splitter.
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

TASK=""
ROUND=""
INDEX=""
SOURCE_ROOT=""
LEDGER=""
PUSH=()
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --round) ROUND="$2"; shift 2 ;;
    --index) INDEX="$2"; shift 2 ;;
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --ledger) LEDGER="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --push) PUSH=(--push); shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h | --help) usage 0 ;;
    *) echo "error: unknown argument $1" >&2; usage ;;
  esac
done
[[ -n "${TASK}" && -n "${ROUND}" ]] || { echo "error: --task and --round are required" >&2; usage; }
[[ -n "${LEDGER}" ]] || LEDGER="${OUTPUT_DIR}/sim/ledgers/${TASK}_r${ROUND}_ledger.jsonl"

if [[ -n "${INDEX}" ]]; then
  splits=("${INDEX}")
else
  info="$("${PYTHON}" -m mulligan.sim.recipes round-info --task "${TASK}" --round "${ROUND}")"
  n="${info#*splits=}"
  n="${n%% *}"
  [[ "${n}" =~ ^[0-9]+$ && "${n}" -gt 0 ]] || { echo "error: ${TASK} R${ROUND} has no splits" >&2; exit 1; }
  splits=()
  for ((i = 0; i < n; i++)); do splits+=("${i}"); done
fi

for i in "${splits[@]}"; do
  source_args=()
  [[ -z "${SOURCE_ROOT}" ]] || source_args=(--source-root "${SOURCE_ROOT}")
  read_argv "${PYTHON}" -m mulligan.sim.recipes split-argv --task "${TASK}" --round "${ROUND}" \
    --index "${i}" --dataset-root "${OUTPUT_DIR}/sim/data" --ledger "${LEDGER}" \
    --output-root "${OUTPUT_DIR}/sim/splits/${TASK}_r${ROUND}_${i}" \
    ${source_args[@]+"${source_args[@]}"} ${PUSH[@]+"${PUSH[@]}"}
  module="${ARGV[0]}"
  echo "== split ${TASK} R${ROUND} #${i} (${module})"
  "${PYTHON}" -m "${module}" "${ARGV[@]:1}" ${EXTRA[@]+"${EXTRA[@]}"}
done
