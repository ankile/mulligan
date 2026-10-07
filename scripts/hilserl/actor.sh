#!/usr/bin/env bash
# HiL-SERL actor: steps the env with the learner's latest policy; with a SpaceMouse the
# operator takes over by deflecting the puck or toggling the gripper (keys h r x p t q,
# docs/baselines.md).
#
# Usage: scripts/hilserl/actor.sh <recipe> <session_dir> <learner_ip> [flags]
#   operator (MuJoCo viewer + SpaceMouse; on macOS set PYTHON="uv run mjpython"):
#     scripts/hilserl/actor.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k localhost
#   no human (headless, as fast as the learner allows):
#     scripts/hilserl/actor.sh square-narrow-hilserl outputs/hilserl/narrow_fork150k localhost \
#         --no-spacemouse --no-render --unpaced
#
# The actor keeps the authoritative episode log in <session>/actor/ (its own copy of the
# session directory when it runs on another machine) and re-pushes anything the learner
# has not acknowledged. Through an ssh tunnel use `ssh -L <port>:<node>:<port> -L
# <port+1>:<node>:<port+1>` and learner_ip=localhost. It refuses a learner whose code
# (hilserl_sha) or pinned config differs.
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
if [[ $# -lt 3 ]]; then
    echo "usage: $0 <recipe> <session_dir> <learner_ip> [flags]" >&2
    exit 2
fi
RECIPE="$1"
SESSION="$2"
IP="$3"
shift 3
read_config "${RECIPE}"
export JAX_PLATFORMS=cpu
# SpaceMouse gains of the teleop collector that recorded the demos (base step 0.005 m,
# pos 1.0, rot 1.5) expressed in the takeover mapping (base 0.0055): 0.005/0.0055 and 1.5x that.
exec "${PY[@]}" -u -m mulligan.baselines.hilserl --actor --session "${SESSION}" --ip "${IP}" \
    --pos_sensitivity 0.909090909090909 --rot_sensitivity 1.363636363636364 "${CONFIG[@]}" "$@"
