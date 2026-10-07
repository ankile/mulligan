# Based on the EXPO source code (https://github.com/pd-perry/EXPO) and the RLPD source code
# (https://github.com/ikostrikov/rlpd), on which EXPO builds.
# Vendored from EXPO's train_robo.py. Both projects are MIT-licensed; their notices follow.
#
# ---- EXPO ----
# MIT License
#
# Copyright (c) 2025 pd-perry
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---- RLPD (https://github.com/ikostrikov/rlpd, LICENCE) ----
# MIT License
#
# Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""RLPD on Square-Narrow / Square-Broad with offline demos.

    python -m mulligan.baselines.rlpd.train --task=square_narrow --offline_data=teleop \\
        --seed=1 --output_dir=outputs/rlpd/square_narrow_s1

The RLPD recipes of ``configs/sim/recipes.json`` hold the flags of every run, and
``scripts/sim/train_cell.sh <recipe>`` runs them. Defaults are the Square-Narrow recipe's.

``start_training`` uniform-random steps, then one ``SACLearner.update`` of ``utd_ratio``
minibatches per env step on a batch that interleaves ``offline_ratio`` offline and
``1 - offline_ratio`` online samples; evaluation every ``eval_interval`` steps (step 0
included); every stored transition that ends an episode (success, timeout or early kill) has
mask 0. ``<output_dir>`` gets ``flags.json``, ``train.jsonl``, ``eval.jsonl``, the resume state
``resume/state.pkl`` (written at every evaluation and on SIGTERM; rerunning the same command
continues from it) and, with ``--checkpoint_model``, ``checkpoints/agent_<step>.msgpack``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tqdm

# XLA must not grab the whole GPU before the eval workers and MuJoCo start.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Exit code of a run that saved its resume state after SIGTERM (preemption /
# scancel) and expects to be requeued or relaunched.
PREEMPTED_EXIT_CODE = 75
_RESUME_STATE = "state.pkl"
_stop_requested = False


def build_parser() -> argparse.ArgumentParser:
    from mulligan.baselines.rlpd.configs import add_agent_flags, parse_bool
    from mulligan.baselines.rlpd.datasets import OFFLINE_DATA

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a = p.add_argument
    a("--task", default="square_narrow", choices=["square_narrow", "square_broad"])
    a(
        "--offline_data",
        default="teleop",
        help=f"One of {', '.join(OFFLINE_DATA)} (mulligan.baselines.rlpd.datasets) or a "
        "robomimic low_dim .hdf5 file.",
    )
    a("--output_dir", default="", help="Run directory (default outputs/rlpd/<task>_s<seed>).")
    a("--seed", type=int, default=1)
    a("--max_steps", type=int, default=300_000)
    a("--start_training", type=int, default=5000)
    a("--utd_ratio", type=int, default=20)
    a("--batch_size", type=int, default=256)
    a("--offline_ratio", type=float, default=0.5)
    a("--eval_interval", type=int, default=10_000)
    a("--eval_episodes", type=int, default=50)
    a(
        "--eval_workers",
        type=int,
        default=0,
        help="0 = min(usable CPUs // 2, eval_episodes).",
    )
    a("--log_interval", type=int, default=1000)
    a(
        "--early_kill",
        type=parse_bool,
        default=False,
        help="End doomed training episodes early as failures (square_narrow only).",
    )
    a("--tqdm", type=parse_bool, default=True)
    a(
        "--checkpoint_model",
        type=parse_bool,
        default=False,
        help="Keep the agent of every --checkpoint_interval-th step as "
        "<output_dir>/checkpoints/agent_<step>.msgpack (flax serialization.to_bytes).",
    )
    a(
        "--checkpoint_interval",
        type=int,
        default=0,
        help="Steps between kept agent checkpoints; a multiple of --eval_interval. 0 = every evaluation.",
    )
    a("--wandb", type=parse_bool, default=False, help="Log to Weights & Biases (optional).")
    a("--project_name", default="mulligan-rlpd", help="W&B project.")
    a("--wandb_entity", default=None, help="W&B entity (default: your W&B default entity).")
    add_agent_flags(p)
    return p


def combine(one_dict, other_dict):
    combined = {}
    for k, v in one_dict.items():
        tmp = np.empty((v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype)
        tmp[0::2] = v
        tmp[1::2] = other_dict[k]
        combined[k] = tmp
    return combined


def _request_stop(signum, _frame):
    global _stop_requested
    _stop_requested = True
    print(f"SIGNAL {signum}: will save resume state and exit {PREEMPTED_EXIT_CODE}", flush=True)


def save_resume_state(resume_dir: str | Path, agent, replay_buffer, next_step: int) -> None:
    """Atomically write the state a restarted process needs to continue at
    ``next_step`` (the first online step it has NOT executed yet). The replay
    buffer is stored as its filled rows + counters + sampler RNG state."""
    from flax import serialization

    os.makedirs(resume_dir, exist_ok=True)
    n = len(replay_buffer)
    payload = {
        "next_step": int(next_step),
        "agent": serialization.to_bytes(agent),
        "buffer": {
            "data": {k: v[:n] for k, v in replay_buffer.dataset_dict.items()},
            "size": replay_buffer._size,
            "insert_index": replay_buffer._insert_index,
            "rng_state": replay_buffer.np_random.bit_generator.state,
        },
    }
    fd, tmp_path = tempfile.mkstemp(dir=resume_dir, prefix=".state-", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, os.path.join(resume_dir, _RESUME_STATE))
    print(f"RESUME_STATE saved next_step={next_step} -> {resume_dir}", flush=True)


def load_resume_state(resume_dir: str | Path, agent, replay_buffer):
    """Restore into ``agent`` / ``replay_buffer``; return (agent, next_step) or
    None when no state exists."""
    from flax import serialization

    path = os.path.join(resume_dir, _RESUME_STATE)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    agent = serialization.from_bytes(agent, payload["agent"])
    buf = payload["buffer"]
    # A completed run can fill its ring. When extending the horizon, unwrap
    # those rows before appending so new transitions cannot overwrite the prefix.
    capacity = len(next(iter(replay_buffer.dataset_dict.values())))
    expanding_full_ring = buf["insert_index"] < buf["size"] < capacity
    for k, v in buf["data"].items():
        if expanding_full_ring:
            v = np.concatenate((v[buf["insert_index"] :], v[: buf["insert_index"]]))
        replay_buffer.dataset_dict[k][: len(v)] = v
    replay_buffer._size = buf["size"]
    replay_buffer._insert_index = buf["size"] if expanding_full_ring else buf["insert_index"]
    replay_buffer.np_random.bit_generator.state = buf["rng_state"]
    return agent, int(payload["next_step"])


def save_agent_checkpoint(chkpt_dir: str | Path, agent, step: int) -> None:
    """Atomically keep the agent evaluated at ``step``; load with
    ``serialization.from_bytes(<freshly created agent>, bytes)``."""
    from flax import serialization

    fd, tmp_path = tempfile.mkstemp(dir=chkpt_dir, prefix=".agent-", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(serialization.to_bytes(agent))
    path = os.path.join(chkpt_dir, f"agent_{step:07d}.msgpack")
    os.replace(tmp_path, path)
    print(f"CHECKPOINT saved step={step} -> {path}", flush=True)


class _Logger:
    """JSONL logs under the run directory, mirrored to W&B when enabled."""

    def __init__(self, log_dir: Path, wandb_run):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.files = {
            "train": open(log_dir / "train.jsonl", "a"),
            "eval": open(log_dir / "eval.jsonl", "a"),
        }
        self.wandb_run = wandb_run

    def log(self, kind: str, step: int, values: dict) -> None:
        row = {"step": int(step), "time": time.time(), **{k: float(v) for k, v in values.items()}}
        self.files[kind].write(json.dumps(row) + "\n")
        self.files[kind].flush()
        if self.wandb_run is not None:
            self.wandb_run.log({f"{kind}/{k}": float(v) for k, v in values.items()}, step=step)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from mulligan.baselines.hilserl.obs import HORIZON
    from mulligan.baselines.rlpd.agent import SACLearner
    from mulligan.baselines.rlpd.configs import agent_kwargs as get_agent_kwargs
    from mulligan.baselines.rlpd.data import RoboReplayBuffer
    from mulligan.baselines.rlpd.datasets import load_offline
    from mulligan.baselines.rlpd.env import RLPDEnv
    from mulligan.baselines.rlpd.evaluation import Evaluator

    assert 0.0 <= args.offline_ratio <= 1.0
    agent_kwargs = get_agent_kwargs(args)

    output_dir = Path(args.output_dir or Path("outputs/rlpd") / f"{args.task}_s{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = output_dir / "resume"
    run_name = os.environ.get("WANDB_NAME") or f"rlpd_{output_dir.name}"
    (output_dir / "flags.json").write_text(
        json.dumps({**vars(args), "agent_kwargs": agent_kwargs}, indent=1, default=str)
    )
    wandb_run = None
    if args.wandb:
        import wandb

        # WANDB_RUN_ID / WANDB_RESUME let a resumed process continue the same run.
        wandb_run = wandb.init(project=args.project_name, entity=args.wandb_entity, name=run_name)
        wandb_run.config.update({**vars(args), "agent_kwargs": agent_kwargs}, allow_val_change=True)
    logger = _Logger(output_dir, wandb_run)

    if args.checkpoint_model:
        chkpt_dir = output_dir / "checkpoints"
        chkpt_dir.mkdir(exist_ok=True)
    checkpoint_interval = args.checkpoint_interval or args.eval_interval
    assert checkpoint_interval % args.eval_interval == 0, (
        "--checkpoint_interval must be a multiple of --eval_interval",
        checkpoint_interval,
        args.eval_interval,
    )

    ds, env_meta = load_offline(args.task, args.offline_data)
    example_observation = ds.dataset_dict["observations"][0]
    example_action = ds.dataset_dict["actions"][0]
    env = RLPDEnv(env_meta, horizon=HORIZON, early_kill=args.early_kill)
    if env.observation_space.shape != example_observation.shape:
        raise ValueError(
            f"env observation {env.observation_space.shape} != offline data {example_observation.shape}"
        )
    evaluator = Evaluator(
        env_meta,
        HORIZON,
        agent_kwargs,
        example_observation.shape[0],
        example_action.shape[0],
        args.eval_episodes,
        args.eval_workers,
    )

    agent = SACLearner.create(args.seed, example_observation, example_action, **agent_kwargs)
    replay_buffer = RoboReplayBuffer(example_observation, example_action, args.max_steps + 1)
    replay_buffer.seed(args.seed)

    start_step = 0
    restored = load_resume_state(resume_dir, agent, replay_buffer)
    if restored is not None:
        agent, start_step = restored
        assert len(replay_buffer) <= start_step, (len(replay_buffer), start_step)
        print(
            f"RESUMED from {resume_dir} at step {start_step} "
            f"(replay buffer size {len(replay_buffer)})",
            flush=True,
        )
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    n_online = int(args.batch_size * args.utd_ratio * (1 - args.offline_ratio))
    n_offline = int(args.batch_size * args.utd_ratio * args.offline_ratio)
    observation, _ = env.reset()
    for i in tqdm.tqdm(range(start_step, args.max_steps + 1), smoothing=0.1, disable=not args.tqdm):
        if _stop_requested:
            # Step i has not run yet: everything up to i-1 is in the buffer
            # and logged, so the restarted process resumes at i.
            save_resume_state(resume_dir, agent, replay_buffer, i)
            print(f"PREEMPTED at step {i}; exiting {PREEMPTED_EXIT_CODE}", flush=True)
            sys.stdout.flush()
            os._exit(PREEMPTED_EXIT_CODE)
        if i < args.start_training:
            # Seeded per step, so --seed fixes the warm-up actions and a resumed
            # run draws the same ones.
            action = np.random.default_rng([args.seed, i]).uniform(-1, 1, size=example_action.shape)
        else:
            action, agent = agent.sample_actions(observation)
        if args.early_kill:
            env.env_step = i
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=0.0 if done else 1.0,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, _ = env.reset()
            logger.log("train", i, info["episode"])

        if i >= args.start_training:
            online_batch = replay_buffer.sample(n_online)
            offline_batch = ds.sample(n_offline)
            batch = combine(offline_batch, online_batch)
            agent, update_info = agent.update(batch, args.utd_ratio)
            if i % args.log_interval == 0:
                logger.log("train", i, update_info)

        if i % args.eval_interval == 0:
            t0 = time.time()
            eval_info, _ = evaluator(agent)
            eval_info["wall_s"] = time.time() - t0
            logger.log("eval", i, eval_info)
            print(f"EVAL step={i} {json.dumps(eval_info)}", flush=True)

            if i < args.max_steps:
                save_resume_state(resume_dir, agent, replay_buffer, i + 1)

            if args.checkpoint_model and i % checkpoint_interval == 0:
                save_agent_checkpoint(chkpt_dir, agent, i)

    if start_step <= args.max_steps:
        save_resume_state(resume_dir, agent, replay_buffer, args.max_steps + 1)

    evaluator.close()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
