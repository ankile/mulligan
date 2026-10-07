#!/usr/bin/env bash
# HiL-SERL eval watcher (CPU): evaluates every learner checkpoint of a session
# (50 episodes, seeds 10000..10049, deterministic actions) and exits once the learner is
# done and every checkpoint is evaluated. Stateless: rerun it any time.
#
# Usage: scripts/hilserl/eval_watcher.sh <recipe> <session_dir> [flags, e.g. --once]
#   scripts/hilserl/eval_watcher.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k
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
export JAX_PLATFORMS=cpu
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "${PY[@]}" -u -m mulligan.baselines.hilserl --eval-watcher --session "${SESSION}" --poll_s 20 \
    "${CONFIG[@]}" "$@"
