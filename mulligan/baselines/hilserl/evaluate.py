"""Native eval (as RLPD's: N episodes, deterministic ``mode()`` actions, fixed
per-episode seeds ``seed_base + i``) on CPU worker processes,
and the checkpoint watcher that runs it out-of-process so the learner never
stalls.

Watcher: polls ``learner/checkpoints/step_*`` and evaluates each new one in
ascending order, writing ``eval/native_NNNNNNN.json`` and one line in
``eval/ledger.jsonl`` (which the learner tails into W&B).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from mulligan.baselines.hilserl.config import HilSerlConfig
from mulligan.baselines.hilserl.session import (
    Session,
    append_jsonl,
    read_jsonl,
    write_json_atomic,
)

_WORKER_ENV = None
_WORKER_POLICY = None


def _worker_init(cfg_dict: dict, params: dict):
    global _WORKER_ENV, _WORKER_POLICY
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from mulligan.baselines.hilserl.env import HilSerlEnv
    from mulligan.baselines.hilserl.policy import ActorPolicy

    cfg = HilSerlConfig(**cfg_dict)
    _WORKER_ENV = HilSerlEnv(
        cfg.task, horizon=cfg.horizon, truncation_bootstrap=cfg.truncation_bootstrap
    )
    _WORKER_POLICY = ActorPolicy(cfg, rng_seed=0, params=params)


def _run_episode(seed: int) -> dict:
    env, policy = _WORKER_ENV, _WORKER_POLICY
    obs = env.reset(seed=seed)
    ret, length, success = 0.0, 0, False
    while True:
        res = env.step(policy.act_deterministic(obs))
        ret += res.reward
        length += 1
        obs = res.obs
        if res.success:
            success = True
        if res.done:
            break
    return {
        "seed": int(seed),
        "success": bool(success),
        "length": int(length),
        "return": float(ret),
    }


def summarize(episodes: list[dict]) -> dict:
    succ = [e["success"] for e in episodes]
    succ_len = [e["length"] for e in episodes if e["success"]]
    return {
        "return": float(np.mean([e["return"] for e in episodes])),
        "length": float(np.mean([e["length"] for e in episodes])),
        "success_rate": float(np.mean(succ)),
        "success_length_mean": float(np.mean(succ_len)) if succ_len else 0.0,
        "n": len(episodes),
    }


def evaluate_native(
    params: dict,
    cfg: HilSerlConfig,
    num_episodes: Optional[int] = None,
    workers: Optional[int] = None,
) -> tuple[dict, list[dict]]:
    n = num_episodes or cfg.eval_episodes
    seeds = [cfg.eval_seed_base + i for i in range(n)]
    workers = max(1, min(workers or cfg.eval_workers, n))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(cfg.to_dict(), params)) as pool:
        episodes = pool.map(_run_episode, seeds, chunksize=1)
    return summarize(episodes), episodes


def load_checkpoint_params(ckpt_dir: Path) -> tuple[dict, int]:
    with open(ckpt_dir / "actor_params.pkl", "rb") as f:
        payload = pickle.load(f)
    return payload["params"], int(payload["param_version"])


def final_checkpoint_dir(session: Session, cfg: HilSerlConfig) -> Path:
    """The learner's last milestone checkpoint (the last multiple of eval_interval)."""
    final_step = cfg.max_steps // cfg.eval_interval * cfg.eval_interval
    return session.checkpoints_dir / f"step_{final_step:07d}"


def pending_checkpoints(session: Session) -> list[tuple[int, Path]]:
    done = {int(r["env_step"]) for r in read_jsonl(session.eval_ledger)}
    out = []
    for d in sorted(session.checkpoints_dir.glob("step_*")):
        if not d.is_dir() or d.name.startswith("."):
            continue
        step = int(d.name.split("_")[1])
        if step not in done:
            out.append((step, d))
    return out


def watch(
    session: Session,
    cfg: HilSerlConfig,
    poll_s: float = 15.0,
    once: bool = False,
    workers: Optional[int] = None,
) -> None:
    session.eval_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[eval-watcher] watching {session.checkpoints_dir} every {poll_s}s (episodes={cfg.eval_episodes}, workers={workers or cfg.eval_workers})",
        flush=True,
    )
    while True:
        pending = pending_checkpoints(session)
        for step, ckpt in pending:
            params, version = load_checkpoint_params(ckpt)
            t0 = time.time()
            metrics, episodes = evaluate_native(params, cfg, workers=workers)
            dt = time.time() - t0
            out = session.eval_dir / f"native_{step:07d}.json"
            write_json_atomic(
                out,
                {
                    "env_step": step,
                    "param_version": version,
                    "metrics": metrics,
                    "episodes": episodes,
                    "checkpoint": str(ckpt),
                    "wall_s": dt,
                },
            )
            append_jsonl(
                session.eval_ledger,
                {
                    "env_step": step,
                    "metrics": metrics,
                    "param_version": version,
                    "mode": "watcher",
                    "time": time.time(),
                    "wall_s": dt,
                },
            )
            print(f"[eval-watcher] step {step}: {json.dumps(metrics)} ({dt:.0f}s)", flush=True)
        if once:
            return
        # The learner checkpoints at 0, eval_interval, ... <= max_steps. Once the last of
        # those exists and is evaluated no checkpoint can follow (checking the directory
        # avoids unpickling the learner's whole resume state on every poll).
        if final_checkpoint_dir(session, cfg).is_dir() and not pending_checkpoints(session):
            print(
                "[eval-watcher] learner finished and every checkpoint is evaluated; exiting",
                flush=True,
            )
            return
        time.sleep(poll_s)
