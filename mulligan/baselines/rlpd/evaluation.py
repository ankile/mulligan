# Based on the EXPO source code (https://github.com/pd-perry/EXPO) and the RLPD source code
# (https://github.com/ikostrikov/rlpd), on which EXPO builds.
# Vendored from EXPO's expo/evaluation.py (evaluate_robo, QueueTrajSampler). Both projects are
# MIT-licensed; their notices follow.
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

"""RLPD evaluation: ``num_episodes`` episodes with deterministic (``mode()``) actions, episode
``i`` reset with seed ``10000 + i``, success = any step reported success. Each worker of a
persistent ``spawn`` pool builds its own env once and runs the actor on the JAX CPU backend; the
actor parameters travel with each episode. Metrics: ``return``, ``length``, ``success_rate``,
``success_length_mean``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Optional

import numpy as np

EVAL_SEED_BASE = 10000

_WORKER: dict = {}


def _worker_init(env_meta: dict, horizon: int, agent_kwargs: dict, obs_dim: int, act_dim: int):
    from mulligan.baselines.rlpd.agent import SACLearner
    from mulligan.baselines.rlpd.env import RLPDEnv

    _WORKER["env"] = RLPDEnv(env_meta, horizon=horizon)
    agent = SACLearner.create(
        0, np.zeros(obs_dim, np.float32), np.zeros(act_dim, np.float32), **agent_kwargs
    )
    _WORKER["agent"] = agent


def _run_episode(args) -> dict:
    actor_params, seed = args
    env, agent = _WORKER["env"], _WORKER["agent"]
    agent = agent.replace(actor=agent.actor.replace(params=actor_params))
    obs, _ = env.reset(seed=seed)
    ret, length, success = 0.0, 0, False
    while True:
        action = np.asarray(agent.eval_actions(obs))
        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"non-finite eval action at seed {seed}: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        length += 1
        success = success or bool(info["success"])
        if terminated or truncated:
            break
    return {"seed": int(seed), "success": success, "length": length, "return": float(ret)}


def summarize(episodes: list[dict]) -> dict:
    succ_len = [e["length"] for e in episodes if e["success"]]
    return {
        "return": float(np.mean([e["return"] for e in episodes])),
        "length": float(np.mean([e["length"] for e in episodes])),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "success_length_mean": float(np.mean(succ_len)) if succ_len else 0.0,
    }


class Evaluator:
    """Persistent pool of eval workers (one env + CPU actor each)."""

    def __init__(
        self,
        env_meta: dict,
        horizon: int,
        agent_kwargs: dict,
        obs_dim: int,
        act_dim: int,
        num_episodes: int,
        workers: Optional[int] = None,
    ):
        self.num_episodes = num_episodes
        if not workers:
            # min(cpu_count // 2, num_episodes), counting the CPUs this process may use
            # (e.g. a SLURM allocation), not the node's
            workers = min(len(os.sched_getaffinity(0)) // 2, num_episodes)
        workers = max(1, min(workers, num_episodes))
        ctx = mp.get_context("spawn")
        # Spawned children read the environment at start-up: keep them off the GPU.
        saved = os.environ.get("JAX_PLATFORMS")
        os.environ["JAX_PLATFORMS"] = "cpu"
        try:
            self.pool = ctx.Pool(
                workers,
                initializer=_worker_init,
                initargs=(env_meta, horizon, agent_kwargs, obs_dim, act_dim),
            )
        finally:
            if saved is None:
                os.environ.pop("JAX_PLATFORMS")
            else:
                os.environ["JAX_PLATFORMS"] = saved
        self.workers = workers

    def __call__(self, agent) -> tuple[dict, list[dict]]:
        import jax

        params = jax.device_get(agent.actor.params)
        seeds = [EVAL_SEED_BASE + i for i in range(self.num_episodes)]
        episodes = self.pool.map(_run_episode, [(params, s) for s in seeds], chunksize=1)
        return summarize(episodes), episodes

    def close(self):
        self.pool.close()
        self.pool.join()
