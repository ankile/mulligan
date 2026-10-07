"""Exercise the actual training loop and persistence with a cheap deterministic env."""

import json
import os
import pickle
import signal
import subprocess
import sys

import numpy as np
import pytest

from tests.baselines.rlpd.test_train import REPO, _synthetic

PROBE = r"""
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mulligan.baselines.rlpd import agent, env, evaluation, train

class Env:
    observation_space = SimpleNamespace(shape=(23,))
    def __init__(self, *args, **kwargs):
        self.n = 0
    def reset(self):
        return np.zeros(23, np.float32), {}
    def step(self, action):
        self.n += 1
        if self.n == 3 and sys.argv[3] == "interrupt":
            # Parent sends SIGTERM after seeing the marker. sigtimedwait avoids
            # a race between receiving that signal and entering a blocking wait.
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            Path(sys.argv[1], "blocked").touch()
            info = signal.sigtimedwait({signal.SIGTERM}, 30)
            assert info is not None
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
            signal.raise_signal(signal.SIGTERM)
        return np.full(23, self.n, np.float32), float(self.n), False, False, {}
    def close(self):
        pass

class Evaluator:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, agent):
        return {"return": 0, "success_rate": 0}, None
    def close(self):
        pass

env.RLPDEnv = Env
evaluation.Evaluator = Evaluator
agent.SACLearner = SimpleNamespace(create=lambda *a, **k: {"weight": np.array([7.])})
root = Path(sys.argv[1])
raise SystemExit(train.main([
    f"--offline_data={root / 'data' / 'toy.hdf5'}",
    "--start_training=1000", f"--max_steps={sys.argv[2]}", "--eval_interval=2",
    "--tqdm=False", f"--output_dir={root / 'run'}",
]))
"""


def command(root, steps, mode="normal"):
    return [sys.executable, "-c", PROBE, str(root), str(steps), mode]


def run(root, steps):
    result = subprocess.run(
        command(root, steps),
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def state(root):
    return pickle.loads((root / "run/resume/state.pkl").read_bytes())


def eval_steps(root):
    return [json.loads(line)["step"] for line in (root / "run/eval.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("steps", [3, 4])
def test_completion_and_extension_do_not_repeat_steps(tmp_path, steps):
    pytest.importorskip("jax")
    _synthetic(tmp_path / "data")
    run(tmp_path, steps)
    saved = state(tmp_path)
    assert saved["next_step"] == steps + 1
    assert saved["buffer"]["size"] == steps + 1
    run(tmp_path, 6)
    final = state(tmp_path)
    assert final["next_step"] == 7 and final["buffer"]["size"] == 7
    for key, rows in saved["buffer"]["data"].items():
        np.testing.assert_array_equal(final["buffer"]["data"][key][: len(rows)], rows)
    assert final["agent"] == saved["agent"]
    assert eval_steps(tmp_path) == [0, 2, 4, 6]


def test_warmup_actions_follow_the_seed(tmp_path):
    pytest.importorskip("jax")
    for name in ("a", "b"):
        _synthetic(tmp_path / name / "data")
        run(tmp_path / name, 4)
    a, b = (state(tmp_path / name)["buffer"]["data"]["actions"] for name in ("a", "b"))
    assert len(a) == 5 and len(np.unique(a[:, 0])) == 5
    np.testing.assert_array_equal(a, b)


def test_sigterm_saves_and_resumes_in_a_new_process(tmp_path):
    pytest.importorskip("jax")
    import time

    _synthetic(tmp_path / "data")
    with (tmp_path / "output").open("w+") as output:
        proc = subprocess.Popen(
            command(tmp_path, 6, "interrupt"),
            cwd=REPO,
            env=dict(os.environ, JAX_PLATFORMS="cpu"),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 90
            while (
                not (tmp_path / "blocked").exists()
                and proc.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert (tmp_path / "blocked").exists()
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=60) == 75
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        output.seek(0)
        assert "PREEMPTED at step 3" in output.read()
    saved = state(tmp_path)
    assert saved["next_step"] == 3 and saved["buffer"]["size"] == 3
    assert "at step 3" in run(tmp_path, 6)
    final = state(tmp_path)
    for key, rows in saved["buffer"]["data"].items():
        np.testing.assert_array_equal(final["buffer"]["data"][key][:3], rows)
    assert final["next_step"] == 7 and eval_steps(tmp_path) == [0, 2, 4, 6]


def test_extend_previously_wrapped_replay_preserves_chronological_rows(tmp_path):
    pytest.importorskip("jax")
    from mulligan.baselines.rlpd.data import RoboReplayBuffer
    from mulligan.baselines.rlpd.train import load_resume_state, save_resume_state

    def transition(i):
        return dict(
            observations=np.array([i]),
            next_observations=np.array([i + 1]),
            actions=np.array([i]),
            rewards=i,
            masks=1,
            dones=False,
        )

    small = RoboReplayBuffer(np.zeros(1), np.zeros(1), 3)
    for i in range(5):
        small.insert(transition(i))
    save_resume_state(str(tmp_path), {"weight": np.array([7.0])}, small, 5)
    large = RoboReplayBuffer(np.zeros(1), np.zeros(1), 7)
    _, next_step = load_resume_state(str(tmp_path), {"weight": np.array([0.0])}, large)
    assert next_step == 5 and len(large) == 3
    large.insert(transition(5))
    np.testing.assert_array_equal(large.dataset_dict["rewards"][:4], [2, 3, 4, 5])
