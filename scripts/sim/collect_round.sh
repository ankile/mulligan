#!/usr/bin/env bash
# Collect the data of one sim round (Square-Narrow / Square-Broad, rounds 1-3).
#
# Usage:
#   scripts/sim/collect_round.sh dagger --task T --round N [--operator human|replay|MOD:FN]
#       [--replay-dataset REPO@REV|mulligan/NAME|DIR] [--replay-max-episodes K]
#       [--output-dir DIR]
#       [--push --hub-namespace NS]
#       [-- COLLECTOR_FLAG ...]
#   scripts/sim/collect_round.sh rollouts --task T --round N [--index I] [--output-dir DIR]
#       [--push] [-- COLLECTOR_FLAG ...]
#   scripts/sim/collect_round.sh autonomous <recipe_id> [--predecessor REF] [--output-dir DIR]
#       [--push] [-- COLLECTOR_FLAG ...]
#
# dagger      Blinded DAgger collection (`python -m mulligan.sim.collect.dagger`): the round's
#             start list and routing manifest (data/sim/start_manifests), the previous round's
#             baseline and Mulligan agents (seed 1, from configs/sim/recipes.json), N=32
#             best-of-N, the adaptive no-CF / with-CF protocol quota. --operator human drives
#             interventions with a SpaceMouse ('h' take over, 'c' counterfactual replay);
#             --operator replay replays the recorded human segments of the released collection
#             headlessly, for CI; --replay-dataset replays another recording instead: REPO@REV,
#             a released mulligan/* dataset (read at its pin in release/revisions.json) or a
#             local dataset directory. Any other module:factory operator is passed through.
#             The ledger for split_round.sh is written to
#             DIR/sim/ledgers/<task>_r<N>_ledger.jsonl.
# rollouts    Policy-rollout collection (`python -m mulligan.sim.collect.rollouts`) that feeds
#             the next round's start design and training data. Runs every rollout job of the
#             round, or only job I (0-based, see `python -m mulligan.sim.recipes round-info`).
# autonomous  The autonomous-baseline collection that feeds <recipe_id>: 100 episodes of the
#             previous round's seed-1 agent of the same lineage on the round's baseline starts.
#
# Datasets are written under DIR/sim/data/<name>; --push also pushes them to the HF Hub
# (DAgger: to --hub-namespace; rollouts: to your namespace).
# Flags after `--` are appended to the collector's command line (e.g. --num-envs 2).
# Run inside the project environment. Environment: PYTHON (default: python; mjpython on
# macOS for the human DAgger viewer), MULLIGAN_OUTPUT_DIR (default: outputs).
set -euo pipefail

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

[[ $# -ge 1 ]] || usage
MODE="$1"
shift
case "${MODE}" in
  dagger | rollouts) ;;
  autonomous)
    [[ $# -ge 1 ]] || usage
    RECIPE="$1"
    shift
    ;;
  -h | --help) usage 0 ;;
  *) echo "error: unknown mode ${MODE}" >&2; usage ;;
esac

TASK=""
ROUND=""
OPERATOR="human"
REPLAY_DATASET=""
REPLAY_MAX_EPISODES=""
INDEX=""
PREDECESSOR=""
PUSH=()
HUB_NAMESPACE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --round) ROUND="$2"; shift 2 ;;
    --operator) OPERATOR="$2"; shift 2 ;;
    --replay-dataset) REPLAY_DATASET="$2"; shift 2 ;;
    --replay-max-episodes) REPLAY_MAX_EPISODES="$2"; shift 2 ;;
    --index) INDEX="$2"; shift 2 ;;
    --predecessor) PREDECESSOR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --push) PUSH=(--push); shift ;;
    --hub-namespace) HUB_NAMESPACE="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h | --help) usage 0 ;;
    *) echo "error: unknown argument $1" >&2; usage ;;
  esac
done

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON=python
  if [[ "${MODE}" == dagger && "${OPERATOR}" == human && "$(uname)" == Darwin ]]; then
    PYTHON=mjpython
  fi
fi
DATA_ROOT="${OUTPUT_DIR}/sim/data"
mkdir -p "${DATA_ROOT}" "${OUTPUT_DIR}/sim/ledgers" "${OUTPUT_DIR}/sim/audits"

case "${MODE}" in
  dagger)
    [[ -n "${TASK}" && -n "${ROUND}" ]] || { echo "error: --task and --round are required" >&2; usage; }
    dagger_args=()
    [[ -z "${REPLAY_DATASET}" ]] || dagger_args=(--replay-dataset "${REPLAY_DATASET}")
    [[ -z "${REPLAY_MAX_EPISODES}" ]] || dagger_args+=(--replay-max-episodes "${REPLAY_MAX_EPISODES}")
    [[ -z "${HUB_NAMESPACE}" ]] || dagger_args+=(--hub-namespace "${HUB_NAMESPACE}")
    read_argv "${PYTHON}" -m mulligan.sim.recipes dagger-argv --task "${TASK}" --round "${ROUND}" \
      --dataset-root "${DATA_ROOT}" --ledger "${OUTPUT_DIR}/sim/ledgers/${TASK}_r${ROUND}_ledger.jsonl" \
      --operator "${OPERATOR}" ${dagger_args[@]+"${dagger_args[@]}"} ${PUSH[@]+"${PUSH[@]}"}
    echo "== DAgger ${TASK} R${ROUND} (operator: ${OPERATOR})"
    "${PYTHON}" -m mulligan.sim.collect.dagger "${ARGV[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  rollouts)
    [[ -n "${TASK}" && -n "${ROUND}" ]] || { echo "error: --task and --round are required" >&2; usage; }
    if [[ -n "${INDEX}" ]]; then
      jobs=("${INDEX}")
    else
      info="$("${PYTHON}" -m mulligan.sim.recipes round-info --task "${TASK}" --round "${ROUND}")"
      n="${info##*rollouts=}"
      [[ "${n}" =~ ^[0-9]+$ && "${n}" -gt 0 ]] || { echo "error: ${TASK} R${ROUND} has no rollout jobs" >&2; exit 1; }
      jobs=()
      for ((i = 0; i < n; i++)); do jobs+=("${i}"); done
    fi
    for i in "${jobs[@]}"; do
      read_argv "${PYTHON}" -m mulligan.sim.recipes rollouts-argv --task "${TASK}" --round "${ROUND}" \
        --index "${i}" --dataset-root "${DATA_ROOT}" \
        --audit-output "${OUTPUT_DIR}/sim/audits/${TASK}_r${ROUND}_rollouts_${i}.json" ${PUSH[@]+"${PUSH[@]}"}
      echo "== rollouts ${TASK} R${ROUND} job ${i}"
      "${PYTHON}" -m mulligan.sim.collect.rollouts "${ARGV[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
    done
    ;;
  autonomous)
    pred_args=()
    [[ -z "${PREDECESSOR}" ]] || pred_args=(--predecessor "${PREDECESSOR}")
    read_argv "${PYTHON}" -m mulligan.sim.recipes autonomous-argv "${RECIPE}" \
      --dataset-root "${DATA_ROOT}" --audit-output "${OUTPUT_DIR}/sim/audits/${RECIPE}_collection.json" \
      ${pred_args[@]+"${pred_args[@]}"} ${PUSH[@]+"${PUSH[@]}"}
    echo "== autonomous collection for ${RECIPE}"
    "${PYTHON}" -m mulligan.sim.collect.rollouts "${ARGV[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
esac
