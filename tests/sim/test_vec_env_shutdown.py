"""Regression tests for bounded AsyncVectorEnv worker teardown."""

from __future__ import annotations

import pytest

from mulligan.sim.vec_env import AsyncVectorEnv


class _Remote:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    def send(self, payload) -> None:
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


class _Process:
    _next_pid = 1000

    def __init__(self, *, survives_terminate: bool = False, survives_kill: bool = False) -> None:
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.alive = True
        self.survives_terminate = survives_terminate
        self.survives_kill = survives_kill
        self.terminated = False
        self.killed = False
        self.join_calls = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None) -> None:
        self.join_calls.append(timeout)
        if self.terminated and not self.survives_terminate:
            self.alive = False
        if self.killed and not self.survives_kill:
            self.alive = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _vec_env_with(processes: list[_Process]) -> tuple[AsyncVectorEnv, list[_Remote]]:
    vec_env = AsyncVectorEnv.__new__(AsyncVectorEnv)
    remotes = [_Remote() for _ in processes]
    vec_env.remotes = tuple(remotes)
    vec_env.processes = processes
    vec_env.closed = False
    return vec_env, remotes


def test_close_terminates_and_reaps_workers_that_ignore_close() -> None:
    processes = [_Process(), _Process()]
    vec_env, remotes = _vec_env_with(processes)

    vec_env.close()

    assert all(remote.sent == [("close", None)] for remote in remotes)
    assert all(remote.closed for remote in remotes)
    assert all(process.terminated for process in processes)
    assert all(not process.is_alive() for process in processes)
    assert vec_env.closed is True


def test_close_escalates_from_terminate_to_kill() -> None:
    process = _Process(survives_terminate=True)
    vec_env, _remotes = _vec_env_with([process])

    vec_env.close()

    assert process.terminated is True
    assert process.killed is True
    assert process.is_alive() is False


def test_close_fails_loudly_if_worker_survives_kill() -> None:
    process = _Process(survives_terminate=True, survives_kill=True)
    vec_env, remotes = _vec_env_with([process])

    with pytest.raises(RuntimeError, match=str(process.pid)):
        vec_env.close()

    assert remotes[0].closed is True
    assert vec_env.closed is True
