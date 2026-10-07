"""Critic auto-resume continues from the saved step with the scheduled Q/V learning rate."""

from __future__ import annotations

import os
import sys

import pytest

from mulligan.real.train import critic
from mulligan.training.resume import AutoResumeManager


@pytest.fixture
def harness_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


class _Preempted(Exception):
    pass


@pytest.mark.slow
def test_auto_resume_continues_the_critic_lr_schedule(tmp_path, harness_environ, monkeypatch):
    from tests.real import trainer_harness_iql as harness

    class _Recorder(harness._Recorder):
        def __init__(self, fail_at_forward=None):
            super().__init__()
            self.fail_at_forward = fail_at_forward
            self.forwards = 0

        def on_forward(self, losses):
            self.forwards += 1
            if self.forwards == self.fail_at_forward:
                raise _Preempted
            super().on_forward(losses)

    managers: list[AutoResumeManager] = []
    real_acquire = AutoResumeManager.acquire_lock

    def acquire(self):
        managers.append(self)
        real_acquire(self)

    monkeypatch.setattr(AutoResumeManager, "acquire_lock", acquire)

    data_root = tmp_path / "data"
    with harness._patched_env():
        _, encoder_path = harness.ensure_fixture(data_root, seed=0)
        argv = [
            a
            for a in harness.build_argv(
                data_root=data_root,
                encoder_path=encoder_path,
                out_dir=tmp_path / "out",
                steps=4,
                seed=0,
                extra_argv=["--resume-checkpoint-freq", "2"],
            )
            if a != "--no-auto-resume"
        ]
        monkeypatch.setattr(sys, "argv", argv)
        # First launch: preempted during step 3, after the step-2 resume state was saved.
        with harness._patched_run(_Recorder(fail_at_forward=3)), pytest.raises(_Preempted):
            critic.main()
        for manager in managers:
            manager._lock_file.close()
        resumed = _Recorder()
        with harness._patched_run(resumed):
            critic.main()
    steps = resumed.finish()
    assert len(steps) == 2  # steps 3 and 4 only
    # The harness runs --lr 3e-4 with a 3-step warmup_cosine schedule (min frac 0.1).
    expected = [
        critic.critic_lr_at_step(
            step,
            base_lr=3e-4,
            schedule="warmup_cosine",
            warmup_steps=3,
            total_steps=4,
            min_frac=0.1,
        )
        for step in (3, 4)
    ]
    for group in ("q", "v"):
        lrs = [float.fromhex(step["lrs"][group]) for step in steps]
        assert lrs == pytest.approx(expected)
    assert all(set(step["lrs"]) == {"q", "v"} for step in steps)
