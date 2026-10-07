#!/usr/bin/env bash
# HiL-SERL learner (GPU) for one session.
#
# Usage: scripts/hilserl/learner.sh <recipe> <session_dir> [flags]
#   scripts/hilserl/learner.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
#
# Runs with the recipe's flags (seed 1; later flags override, e.g. --seed 2). Serves on the
# recipe's port (REP) and port+1 (PUB) and writes <session>/learner/endpoint.json once ready.
# Checkpoints for the eval watcher land in <session>/learner/checkpoints/ every eval_interval
# env steps. SIGTERM saves learner/state.pkl and exits 75; rerun the same command to continue.
# Run scripts/hilserl/eval_watcher.sh next to it.
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    exit 0
fi
# PYTHON may hold several words (e.g. "uv run mjpython"): split it into an array.
read -r -a PY <<< "${PYTHON:-python}"

# The recipe's HiL-SERL flags (NUL-separated from mulligan.sim.recipes) into CONFIG.
read_config() {
  local tmp item
  tmp="$(mktemp)"
  if ! "${PY[@]}" -m mulligan.sim.recipes train-argv "$1" --stage hilserl_agent --seed 1 >"${tmp}"; then
    rm -f "${tmp}"
    echo "error: $1 is not a HiL-SERL recipe (python -m mulligan.sim.recipes list --family hilserl)" >&2
    exit 1
  fi
  CONFIG=()
  while IFS= read -r -d '' item; do CONFIG+=("${item}"); done <"${tmp}"
  rm -f "${tmp}"
}
if [[ $# -lt 2 ]]; then
    echo "usage: $0 <recipe> <session_dir> [flags]" >&2
    exit 2
fi
RECIPE="$1"
SESSION="$2"
shift 2
read_config "${RECIPE}"
mkdir -p "${SESSION}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# a previous run's endpoint would point actors at a dead learner
rm -f "${SESSION}/learner/endpoint.json"
exec "${PY[@]}" -u -m mulligan.baselines.hilserl --learner --session "${SESSION}" "${CONFIG[@]}" "$@"
