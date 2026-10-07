#!/usr/bin/env bash
# Fork a session at one of its eval checkpoints into a new session ("fork after takeoff": fork
# the no-human session once its policy starts succeeding, then an operator continues the fork).
#
# Usage: scripts/hilserl/fork.sh <src_session> <dst_session> <step>
#   scripts/hilserl/fork.sh outputs/sim/train/square-narrow-hilserl/hilserl_agent/seed-1 \
#       outputs/hilserl/narrow_fork150k 150000
#
# The source must hold actor/{ledger.jsonl,episodes/}, learner/state.pkl,
# learner/checkpoints/step_<step>/ and eval/ledger.jsonl (stage them in one directory
# when the actor and the learner ran on different machines). The new session gets the
# checkpoint's agent, the online rows up to its env step, the demos plus the checkpoint's
# added demo rows, its counters and the matching episode log; start learner.sh,
# eval_watcher.sh and actor.sh on it with the same recipe. Refuses to overwrite <dst_session>.
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    exit 0
fi
if [[ $# -ne 3 ]]; then
    echo "usage: $0 <src_session> <dst_session> <step>" >&2
    exit 2
fi
read -r -a PY <<< "${PYTHON:-python}"
exec "${PY[@]}" -m mulligan.baselines.hilserl.tools.fork_session --src "$1" --dst "$2" --step "$3"
