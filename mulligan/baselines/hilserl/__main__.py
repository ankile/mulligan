"""HiL-SERL on Square-Narrow (``--task square_narrow``, default) or Square-Broad
(``--task square_broad``). Roles:

    uv run python  -m mulligan.baselines.hilserl --learner      --session S            (GPU machine)
    uv run mjpython -m mulligan.baselines.hilserl --actor       --session S --ip HOST  (Mac; `python` on Linux)
    uv run python  -m mulligan.baselines.hilserl --eval-watcher --session S            (CPU job next to the learner)
    uv run python  -m mulligan.baselines.hilserl --probe        --session S --ip HOST  (reachability + code check)

Config flags are HilSerlConfig fields (the HiL-SERL recipes of configs/sim/recipes.json hold
each run's; scripts/hilserl/*.sh pass them). Sessions live under --session (a directory; its
basename is the session id). Every role can be killed and restarted with the same --session.
See docs/baselines.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

from mulligan.baselines.hilserl.config import HilSerlConfig
from mulligan.baselines.hilserl.session import Session


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mulligan.baselines.hilserl",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    role = p.add_mutually_exclusive_group(required=True)
    role.add_argument("--learner", action="store_true")
    role.add_argument("--actor", action="store_true")
    role.add_argument("--eval-watcher", action="store_true")
    role.add_argument("--probe", action="store_true")
    p.add_argument(
        "--session", required=True, type=Path, help="session directory (created if missing)"
    )
    p.add_argument(
        "--ip",
        default=None,
        help="learner host for the actor/probe (localhost through an ssh tunnel); default = learner/endpoint.json",
    )
    # config overrides (HilSerlConfig fields)
    for f in dataclasses.fields(HilSerlConfig):
        if f.name in ("hidden_dims",):
            p.add_argument(
                "--hidden_dims", type=str, default=None, help="comma-separated, default 256,256,256"
            )
            continue
        if f.type in ("bool", bool):
            p.add_argument(
                f"--{f.name}",
                type=lambda s: s.lower() in ("1", "true", "yes"),
                default=None,
                metavar="BOOL",
            )
        elif f.type in ("int", int):
            p.add_argument(f"--{f.name}", type=int, default=None)
        elif f.type in ("float", float):
            p.add_argument(f"--{f.name}", type=float, default=None)
        elif f.type in ("Optional[float]",):
            p.add_argument(f"--{f.name}", type=float, default=None)
        else:
            p.add_argument(f"--{f.name}", type=str, default=None)
    # actor-only
    p.add_argument(
        "--no-spacemouse",
        action="store_true",
        help="autonomous actor (no takeover, no keyboard); the RLPD-in-split ablation",
    )
    p.add_argument(
        "--no-keyboard",
        action="store_true",
        help="skip the pynput listener (headless Linux / permission issues)",
    )
    p.add_argument("--no-render", action="store_true", help="no MuJoCo viewer")
    p.add_argument(
        "--unpaced", action="store_true", help="step as fast as possible (unattended runs)"
    )
    p.add_argument("--pos_sensitivity", type=float, default=1.0)
    p.add_argument("--rot_sensitivity", type=float, default=1.0)
    p.add_argument("--deadzone", type=float, default=None)
    p.add_argument("--latch_s", type=float, default=None)
    p.add_argument("--no-idle-filter", action="store_true")
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="stop after this many episodes in this run (testing)",
    )
    # watcher-only
    p.add_argument("--poll_s", type=float, default=15.0)
    p.add_argument("--once", action="store_true")
    return p


def config_from_args(args) -> HilSerlConfig:
    kw = {}
    for f in dataclasses.fields(HilSerlConfig):
        v = getattr(args, f.name, None)
        if v is None:
            continue
        if f.name == "hidden_dims":
            v = tuple(int(x) for x in v.split(","))
        kw[f.name] = v
    return HilSerlConfig(**kw)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    from mulligan.baselines.hilserl.session import hilserl_sha, repo_sha

    session = Session(args.session)
    if args.probe:
        return probe(session, cfg, args.ip)
    role = "learner" if args.learner else "eval-watcher" if args.eval_watcher else "actor"
    session.ensure_meta(role, cfg.pinned())
    print(
        f"[mulligan.baselines.hilserl] role={role} session={session.root} hilserl_sha={hilserl_sha()} "
        f"repo_sha={repo_sha()[:10]} config_hash={cfg.pinned_hash()}",
        flush=True,
    )
    if args.learner:
        return run_learner(cfg, session)
    if args.eval_watcher:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from mulligan.baselines.hilserl.evaluate import watch

        watch(session, cfg, poll_s=args.poll_s, once=args.once)
        return 0
    return run_actor(cfg, session, args)


def run_learner(cfg: HilSerlConfig, session: Session) -> int:
    from mulligan.baselines.hilserl.learner import Learner, init_wandb
    from mulligan.baselines.hilserl.transport import LearnerServer

    wandb_run = init_wandb(cfg, session, "learner")
    learner = Learner(cfg, session, wandb_run)
    learner.load_resume()
    server = LearnerServer(cfg.port, learner.request_callback)
    try:
        learner.run_split(server)
    finally:
        server.stop()
    return 0


def run_actor(cfg: HilSerlConfig, session: Session, args) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from mulligan.baselines.hilserl.actor import Actor
    from mulligan.baselines.hilserl.env import HilSerlEnv
    from mulligan.baselines.hilserl.policy import ActorPolicy
    from mulligan.baselines.hilserl.takeover import (
        KeyQueue,
        ScriptedKeys,
        SpacemouseDevice,
        SpacemouseTakeover,
    )
    from mulligan.baselines.hilserl.transport import ActorClient

    ip = args.ip or session.read_endpoint()["ip"]
    print(f"[actor] learner at {ip}:{cfg.port} (broadcast {cfg.port + 1})", flush=True)
    if session.env_steps_logged() >= cfg.max_steps:
        return flush_complete_session(cfg, session, ip)
    client = ActorClient(ip, cfg.port)
    env = HilSerlEnv(
        cfg.task,
        render=not args.no_render,
        horizon=cfg.horizon,
        truncation_bootstrap=cfg.truncation_bootstrap,
    )
    policy = ActorPolicy(cfg, rng_seed=cfg.seed * 1000 + session.next_episode_id())
    takeover = None
    keys = ScriptedKeys()
    if not args.no_spacemouse:
        kw = {}
        if args.deadzone is not None:
            kw["deadzone"] = args.deadzone
        if args.latch_s is not None:
            kw["latch_s"] = args.latch_s
        takeover = SpacemouseTakeover(
            SpacemouseDevice(args.pos_sensitivity, args.rot_sensitivity),
            args.pos_sensitivity,
            args.rot_sensitivity,
            idle_filter=not args.no_idle_filter,
            **kw,
        )
        if not args.no_keyboard:
            keys = KeyQueue()
        print(
            "[actor] keys: h hold takeover | r redo bout | x fail-terminate | p pause | t pacing | q stop after episode"
            + (
                " (click the MuJoCo viewer window first; keys typed elsewhere are ignored)"
                if sys.platform == "darwin"
                else ""
            ),
            flush=True,
        )
    actor = Actor(
        cfg,
        session,
        env,
        policy,
        client,
        takeover=takeover,
        keys=keys,
        paced=not args.unpaced,
        max_episodes=args.max_episodes,
    )
    try:
        actor.run()
    finally:
        keys.close()
        if takeover is not None:
            takeover.device.close()
        env.close()
    return 0


def flush_complete_session(cfg: HilSerlConfig, session: Session, ip: str) -> int:
    """The ledger already holds ``max_steps`` env steps: collect nothing, but re-send
    every episode a (possibly preempted and resumed) learner has not ingested. A
    finished learner stops serving, so connect once instead of waiting for it."""
    from mulligan.baselines.hilserl.actor import Actor
    from mulligan.baselines.hilserl.transport import ActorClient

    logged = session.env_steps_logged()
    print(f"[actor] {session.root} already holds {logged} >= {cfg.max_steps} env steps", flush=True)
    try:
        client = ActorClient(ip, cfg.port, wait_for_server=False)
    except Exception as exc:  # agentlace raises a bare Exception when nobody answers
        print(
            f"[actor] no learner answers at {ip}:{cfg.port} ({exc}). A finished learner has "
            "ingested every episode; a learner that is still resuming has not: rerun the "
            "actor once it serves.",
            flush=True,
        )
        return 1
    actor = Actor(cfg, session, env=None, policy=None, client=client)
    try:
        complete = actor.flush()
    finally:
        client.stop()
    print(
        f"[actor] learner {'holds every' if complete else 'is missing some'} logged episode",
        flush=True,
    )
    return 0 if complete else 1


def probe(session: Session, cfg: HilSerlConfig, ip: str | None) -> int:
    from mulligan.baselines.hilserl.session import hilserl_sha, repo_sha
    from mulligan.baselines.hilserl.transport import ActorClient

    ep = session.read_endpoint()
    ip = ip or ep["ip"]
    print(json.dumps(ep, indent=2))
    t0 = time.time()
    client = ActorClient(ip, cfg.port, timeout_ms=3000, wait_for_server=False)
    reply = client.request("status", {})
    dt = time.time() - t0
    client.stop()
    if reply is None:
        print(f"PROBE FAILED: no reply from {ip}:{cfg.port} within 3 s")
        return 1
    print(
        f"PROBE OK in {dt * 1000:.0f} ms: counters={reply['counters']} owed={reply['owed_calls']} param_version={reply['param_version']}"
    )
    sha_ok = ep["hilserl_sha"] == hilserl_sha()
    cfg_ok = ep["config_hash"] == cfg.pinned_hash()
    print(
        f"hilserl content match: {sha_ok} ({ep['hilserl_sha']} vs {hilserl_sha()}); "
        f"config hash match: {cfg_ok}; learner commit {ep['repo_sha'][:10]}, "
        f"this checkout {repo_sha()[:10]}"
    )
    return 0 if (sha_ok and cfg_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
