"""HiL-SERL no-human run on one machine: the learner (GPU), the eval watcher (CPU) and a
headless actor over localhost that never intervenes.

    python -m mulligan.baselines.hilserl.nohuman --session S [HilSerlConfig flags]

``scripts/sim/train_cell.sh <hilserl recipe>`` runs it with the recipe's flags. The actor first
runs 13 episodes of the untrained policy (about 5,200 steps; updates start at 5,000), waits until
the learner is past both JIT compiles (a heartbeat 10 update calls past its start), then runs the
policy alone, unpaced, to ``max_steps``. The learner and the watcher log to ``<session>/logs/``.
Rerun the same command to resume (the learner restarts from ``learner/state.pkl``, the actor from
its episode log). Exit 75: the learner was stopped by SIGTERM with its state saved.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

WARMUP_EPISODES = 13
JIT_CALLS = 10
POLL_S = 5.0
_RESUMED = re.compile(r"RESUMED from .* calls=(\d+) ")
_HEARTBEAT = re.compile(r"heartbeat env_steps=\d+ calls=(\d+)")


def _role(role: str, session: Path, flags: list[str], *extra: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "mulligan.baselines.hilserl",
        role,
        "--session",
        str(session),
        *extra,
        *flags,
    ]


def _env(**overrides: str) -> dict[str, str]:
    env = dict(os.environ, **overrides)
    env.setdefault("MUJOCO_GL", "egl")
    return env


def _calls(log: Path, pattern: re.Pattern) -> list[int]:
    return [int(m.group(1)) for m in pattern.finditer(log.read_text())] if log.exists() else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", required=True, type=Path)
    args, flags = parser.parse_known_args(argv)
    session = args.session
    logs = session / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    learner_log = logs / f"learner-{time.strftime('%Y%m%dT%H%M%S')}.log"
    # a previous run's endpoint would point the actor at a dead learner
    (session / "learner" / "endpoint.json").unlink(missing_ok=True)

    with open(logs / "eval_watcher.log", "a") as out:
        watcher = subprocess.Popen(
            _role("--eval-watcher", session, flags, "--poll_s", "20"),
            stdout=out,
            stderr=subprocess.STDOUT,
            env=_env(JAX_PLATFORMS="cpu"),
        )
    with open(learner_log, "w") as out:
        learner = subprocess.Popen(
            _role("--learner", session, flags),
            stdout=out,
            stderr=subprocess.STDOUT,
            env=_env(XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        )
    children = [learner, watcher]

    def stop(signum, _frame):
        print(f"signal {signum}: stopping the learner (it saves its state)", flush=True)
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def finish(actor_exit: int) -> int:
        if actor_exit != 0:
            print(f"actor exited {actor_exit}; stopping the learner", flush=True)
            if learner.poll() is None:
                learner.send_signal(signal.SIGTERM)
        learner_exit = learner.wait()
        print(f"actor exited {actor_exit}, learner exited {learner_exit}", flush=True)
        print("".join(learner_log.read_text().splitlines(keepends=True)[-5:]), end="", flush=True)
        if learner_exit == 0:
            watcher.wait()
        else:
            watcher.terminate()
        return 1 if actor_exit != 0 and learner_exit == 0 else learner_exit

    endpoint = session / "learner" / "endpoint.json"
    while not endpoint.exists():
        if learner.poll() is not None:
            return finish(1)
        time.sleep(POLL_S)

    actor = _role(
        "--actor",
        session,
        flags,
        "--ip",
        "127.0.0.1",
        "--no-spacemouse",
        "--no-render",
        "--unpaced",
    )
    actor_env = _env(JAX_PLATFORMS="cpu")
    children.append(
        subprocess.Popen(actor + ["--max_episodes", str(WARMUP_EPISODES)], env=actor_env)
    )
    actor_exit = children[-1].wait()
    if actor_exit != 0:
        return finish(actor_exit)

    # On a resume the learner's call counter starts at the resumed value and it JITs again.
    base = (_calls(learner_log, _RESUMED) or [0])[-1]
    while not any(c >= base + JIT_CALLS for c in _calls(learner_log, _HEARTBEAT)):
        if learner.poll() is not None:
            return finish(1)
        time.sleep(POLL_S)
    print(
        f"learner past both JIT compiles (resumed at calls={base}); starting the autonomous actor",
        flush=True,
    )
    children.append(subprocess.Popen(actor, env=actor_env))
    return finish(children[-1].wait())


if __name__ == "__main__":
    sys.exit(main())
