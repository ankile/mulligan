#!/usr/bin/env python3
"""Vectorized robosuite environments: in-process (sync) and one process per env (async).

Both classes expose the same reset variants. Each variant is one function below,
shared by the sync path and the async worker loop, so the two cannot drift.

Resets are unseeded by default.
Pass ``seed`` to seed env ``i`` with ``seed + i`` at construction (see
``mulligan.sim.envs.seed_env``); later resets then follow a reproducible stream
per env.
"""

import multiprocessing as mp
import os
import platform
import sys
import time
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mulligan.sim.envs import resolve_env_name, unwrap_env
from mulligan.sim.placement import nut_qpos


def _fresh_observations(env) -> Dict[str, np.ndarray]:
    """Forward the sim and recompute observations after a state edit.

    The cache must be cleared on the base env: a wrapper attribute would shadow
    it, and relative sensors (``nut_to_eef_pos``) would read stale values.
    """
    env.sim.forward()
    unwrap_env(env)._obs_cache = {}
    return env._get_observations(force_update=True)


def _sim_state(env) -> Tuple[np.ndarray, np.ndarray]:
    sim_state = env.sim.get_state()
    return sim_state.qpos.copy(), sim_state.qvel.copy()


def _reset_to_eval_initial_state(env, state_spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Reset one environment and apply a fixed evaluation object placement."""
    base_env = unwrap_env(env)
    reset_seed = state_spec.get("reset_seed")
    if reset_seed is not None:
        # In place: robosuite's placement samplers hold a reference to env.rng.
        base_env.rng.bit_generator.state = np.random.default_rng(
            int(reset_seed)
        ).bit_generator.state
    env.reset()

    env_name = resolve_env_name(state_spec["env_name"])
    nut_x, nut_y, nut_yaw = state_spec["nut"]
    nut = base_env.nuts[0]
    joint_id = base_env.sim.model.joint_name2id(nut.joints[0])
    start_idx = base_env.sim.model.jnt_qposadr[joint_id]
    base_env.sim.data.qpos[start_idx : start_idx + 7] = nut_qpos(nut_x, nut_y, nut_yaw)

    if env_name == "Square_D1":
        peg_x, peg_y = state_spec["peg"]
        base_env.sim.model.body_pos[base_env.peg1_body_id][0] = peg_x
        base_env.sim.model.body_pos[base_env.peg1_body_id][1] = peg_y
    elif env_name != "NutAssemblySquare":
        raise ValueError(f"Fixed eval initial states are not implemented for env_name={env_name!r}")

    base_env.sim.forward()
    base_env._obs_cache = {}
    return base_env._get_observations(force_update=True)


def _reset_with_sim_state(env, _data=None):
    obs = env.reset()
    return (obs, *_sim_state(env))


def _reset_to_object_state(env, data) -> Dict[str, np.ndarray]:
    """Normal reset (robot and objects random), then override the object qpos only.

    The robot keeps its reset pose, so its controller needs no reset.
    """
    object_qpos, object_start_idx = data
    env.reset()
    env.sim.data.qpos[object_start_idx : object_start_idx + len(object_qpos)] = object_qpos
    return _fresh_observations(env)


def _reset_to_object_state_with_sim_state(env, data):
    obs = _reset_to_object_state(env, data)
    return (obs, *_sim_state(env))


def _reset_with_placements(env, placements):
    """Reset, then apply placement operations; returns (obs, qpos, qvel).

    Each placement is one of:
        ("qpos", start_idx, values)    write ``sim.data.qpos[start_idx:...]``
        ("body_pos", body_name, xy)    move a static body (the Square_D1 peg)
        ("joint", joint_name, values)  write the qpos of a named joint
    """
    env.reset()
    for placement in placements:
        kind = placement[0]
        if kind == "qpos":
            _, start_idx, values = placement
            env.sim.data.qpos[start_idx : start_idx + len(values)] = values
        elif kind == "body_pos":
            _, body_name, xy = placement
            body_id = env.sim.model.body_name2id(body_name)
            env.sim.model.body_pos[body_id][0] = xy[0]
            env.sim.model.body_pos[body_id][1] = xy[1]
        elif kind == "joint":
            _, joint_name, values = placement
            joint_id = env.sim.model.joint_name2id(joint_name)
            start_idx = env.sim.model.jnt_qposadr[joint_id]
            env.sim.data.qpos[start_idx : start_idx + len(values)] = values
        else:
            raise ValueError(f"Unknown placement kind {kind!r} in {placement!r}")
    obs = _fresh_observations(env)
    return (obs, *_sim_state(env))


def _step(env, action):
    return env.step(action)


def _reset(env, _data=None):
    return env.reset()


def _render(env, _data=None):
    return env.render()


def _seed(env, seed):
    from mulligan.sim.envs import seed_env

    seed_env(env, int(seed))


_COMMANDS: Dict[str, Callable] = {
    "step": _step,
    "reset": _reset,
    "render": _render,
    "seed": _seed,
    "reset_to_eval_initial_state": _reset_to_eval_initial_state,
    "reset_with_sim_state": _reset_with_sim_state,
    "reset_to_object_state_with_sim_state": _reset_to_object_state_with_sim_state,
    "reset_with_placements": _reset_with_placements,
}


def _make_env(env_fn: Callable, seed: Optional[int]):
    env = env_fn()
    if seed is not None:
        from mulligan.sim.envs import seed_env

        seed_env(env, seed)
    return env


def worker_process(
    remote: Connection,
    parent_remote: Connection,
    env_fn: Callable,
    seed: Optional[int] = None,
) -> None:
    """Run one environment and serve commands from the parent over ``remote``."""
    import traceback

    if platform.system() == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"

    parent_remote.close()
    if seed is None:
        # A forked worker inherits the parent's global NumPy state (once the parent
        # has used or seeded it), and unseeded Square_D1 draws its peg from it:
        # without this every worker would draw the same pegs.
        np.random.seed()

    # 'spawn' workers do not inherit the parent's patched robosuite / MimicGen modules.
    try:
        from mulligan.sim.envs import register_square_environments

        register_square_environments()
        env = _make_env(env_fn, seed)
    except Exception:
        remote.send(("error", f"Worker failed to create environment:\n{traceback.format_exc()}"))
        remote.close()
        return

    remote.send(("ready", None))

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                # The finally block owns teardown; closing here too ran robosuite /
                # MuJoCo cleanup twice and could leave a worker alive at shutdown.
                remote.close()
                break
            if cmd not in _COMMANDS:
                raise NotImplementedError(f"Unknown command: {cmd}")
            remote.send(_COMMANDS[cmd](env, data))
    except KeyboardInterrupt:
        print("Worker process interrupted", file=sys.stderr)
    except Exception:
        error_msg = f"Worker crashed during operation:\n{traceback.format_exc()}"
        try:
            remote.send(("error", error_msg))
        except (BrokenPipeError, EOFError, OSError):
            print(f"Worker error (couldn't send to parent): {error_msg}", file=sys.stderr)
    finally:
        env.close()


def _unzip3(results) -> Tuple[list, list, list]:
    a, b, c = zip(*results)
    return list(a), list(b), list(c)


class _VectorEnvBase:
    """Reset/step API shared by both vector envs; subclasses implement ``_call``."""

    num_envs: int
    closed: bool

    def _call(self, cmd: str, per_env_data: List[Any]) -> List[Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def _check_len(self, items: List[Any], what: str) -> None:
        if len(items) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} {what}, got {len(items)}")

    def seed_envs(self, seeds: List[int]) -> None:
        """Reseed env ``i`` with ``seeds[i]`` (see ``mulligan.sim.envs.seed_env``)."""
        self._check_len(seeds, "seeds")
        self._call("seed", list(seeds))

    def reset(self) -> List[Dict[str, np.ndarray]]:
        """Reset all environments; returns one observation dict per env."""
        return self._call("reset", [None] * self.num_envs)

    def reset_to_eval_initial_states(
        self, state_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, np.ndarray]]:
        """Reset each environment to one fixed evaluation initial state."""
        self._check_len(state_specs, "state specs")
        return self._call("reset_to_eval_initial_state", state_specs)

    def reset_with_sim_state(
        self,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[np.ndarray]]:
        """Reset all environments; returns (observations, qpos_list, qvel_list)."""
        return _unzip3(self._call("reset_with_sim_state", [None] * self.num_envs))

    def reset_to_object_states_with_sim_state(
        self,
        object_qpos_list: List[np.ndarray],
        object_start_idx: int,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[np.ndarray]]:
        """Reset env ``i`` to ``object_qpos_list[i]``; returns (observations, qpos_list, qvel_list)."""
        self._check_len(object_qpos_list, "object qpos")
        data = [(object_qpos, object_start_idx) for object_qpos in object_qpos_list]
        return _unzip3(self._call("reset_to_object_state_with_sim_state", data))

    def reset_with_placements(
        self,
        placements_per_env: List[List[tuple]],
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[np.ndarray]]:
        """Reset env ``i`` and apply ``placements_per_env[i]`` (see ``_reset_with_placements``).

        Returns (observations, qpos_list, qvel_list).
        """
        self._check_len(placements_per_env, "placement lists")
        return _unzip3(self._call("reset_with_placements", placements_per_env))

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List[Dict[str, np.ndarray]], List[float], List[bool], List[Dict]]:
        """Step every env; returns (observations, rewards, dones, infos)."""
        self._check_len(actions, "actions")
        observations, rewards, dones, infos = zip(*self._call("step", list(actions)))
        return list(observations), list(rewards), list(dones), list(infos)

    def render(self) -> np.ndarray:
        """Frames of all envs stacked as (num_envs, H, W, 3) uint8."""
        return np.stack(self._call("render", [None] * self.num_envs), axis=0)

    def __len__(self) -> int:
        return self.num_envs

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close()


class SyncVectorEnv(_VectorEnvBase):
    """Run the environments sequentially in this process (easier to debug).

    Usage:
        vec_env = SyncVectorEnv([make_env for _ in range(4)])
        obs = vec_env.reset()
        obs, rewards, dones, infos = vec_env.step(actions)
        vec_env.close()
    """

    def __init__(self, env_fns: List[Callable], seed: Optional[int] = None):
        self.num_envs = len(env_fns)
        self.envs = [
            _make_env(env_fn, None if seed is None else seed + i)
            for i, env_fn in enumerate(env_fns)
        ]
        self.closed = False

    def _call(self, cmd: str, per_env_data: List[Any]) -> List[Any]:
        fn = _COMMANDS[cmd]
        return [fn(env, data) for env, data in zip(self.envs, per_env_data)]

    def close(self) -> None:
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True


class AsyncVectorEnv(_VectorEnvBase):
    """Run each environment in its own worker process.

    Usage:
        vec_env = AsyncVectorEnv([make_env for _ in range(4)])
        obs = vec_env.reset()
        obs, rewards, dones, infos = vec_env.step(actions)
        vec_env.close()
    """

    def __init__(
        self,
        env_fns: List[Callable],
        start_method: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            env_fns: one factory per environment.
            start_method: multiprocessing start method. Default: 'spawn' on macOS
                (MuJoCo) and with ``MUJOCO_GL=egl`` (EGL state does not survive
                fork on some NVIDIA drivers, see dm_control issue 95), else 'fork'.
            seed: seed env ``i`` with ``seed + i``; ``None`` keeps resets unseeded.
        """
        self.num_envs = len(env_fns)
        self.closed = False

        if start_method is None:
            if platform.system() == "Darwin" or os.environ.get("MUJOCO_GL") == "egl":
                start_method = "spawn"
            else:
                start_method = "fork"

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])

        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(
            zip(self.work_remotes, self.remotes, env_fns)
        ):
            env_seed = None if seed is None else seed + i
            process = ctx.Process(
                target=worker_process, args=(work_remote, remote, env_fn, env_seed), daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        errors = []
        for i, remote in enumerate(self.remotes):
            try:
                status, data = remote.recv()
                if status == "error":
                    errors.append(f"Worker {i} failed:\n{data}")
                elif status != "ready":
                    errors.append(f"Worker {i} sent unexpected status: {status}")
            except EOFError:
                errors.append(f"Worker {i} died during initialization (no message received)")

        if errors:
            self.close()
            raise RuntimeError(
                f"Failed to initialize vectorized environment. "
                f"{len(errors)}/{self.num_envs} workers failed:\n\n" + "\n\n".join(errors)
            )

    def _check_workers_alive(self) -> None:
        dead_workers = [i for i, proc in enumerate(self.processes) if not proc.is_alive()]
        if dead_workers:
            raise RuntimeError(
                f"Worker processes {dead_workers} have died. "
                f"This usually indicates an error during environment operation. "
                f"Check for errors above or run with sync_envs=True for better error messages."
            )

    def _call(self, cmd: str, per_env_data: List[Any]) -> List[Any]:
        self._check_workers_alive()
        for remote, data in zip(self.remotes, per_env_data):
            remote.send((cmd, data))
        results = [remote.recv() for remote in self.remotes]
        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
                raise RuntimeError(f"Worker {i} failed on {cmd!r}:\n{result[1]}")
        return results

    def close(self) -> None:
        """Close all environments and worker processes."""
        if self.closed:
            return

        close_send_errors = []
        for worker_idx, (remote, process) in enumerate(zip(self.remotes, self.processes)):
            if not process.is_alive():
                continue
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError) as exc:
                close_send_errors.append((worker_idx, type(exc).__name__, str(exc)))

        # Workers close concurrently, so use one shared deadline instead of
        # waiting five seconds independently for every environment. Merely
        # calling terminate() is insufficient: multiprocessing's atexit hook
        # can later wait forever on an unreaped child and keep the GPU job
        # alive after terminal success has already been emitted.
        self._join_until_dead(self.processes, timeout_s=5.0)
        remaining = [process for process in self.processes if process.is_alive()]
        for process in remaining:
            process.terminate()
        self._join_until_dead(remaining, timeout_s=2.0)

        remaining = [process for process in remaining if process.is_alive()]
        for process in remaining:
            process.kill()
        self._join_until_dead(remaining, timeout_s=2.0)
        survivors = [process for process in remaining if process.is_alive()]

        for remote in self.remotes:
            remote.close()

        self.closed = True
        if close_send_errors:
            print(
                "WARNING: AsyncVectorEnv close command failed for workers "
                f"{close_send_errors}; forced process cleanup was applied",
                file=sys.stderr,
            )
        if survivors:
            survivor_pids = [process.pid for process in survivors]
            raise RuntimeError(
                "AsyncVectorEnv failed to stop worker processes after close, "
                f"terminate, and kill: pids={survivor_pids}"
            )

    @staticmethod
    def _join_until_dead(processes: List[mp.Process], *, timeout_s: float) -> None:
        """Reap concurrently-exiting children against one wall-clock deadline."""
        deadline = time.monotonic() + timeout_s
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
