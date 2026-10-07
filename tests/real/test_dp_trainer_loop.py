"""DP trainer loop: validation schedule and the first-validation RNG scope.

The trainer tests run the real ``train()`` for two steps on the tiny local LeRobot fixture of
the trainer harness (CPU, no network).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from mulligan.real.train import policy


@pytest.fixture
def harness_environ():
    """The trainer harness sets CPU-only/threading env vars at import; keep them out of later tests."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _run_trainer(monkeypatch, argv):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["mulligan.real.train.policy", *argv])
    policy.train(policy.parse_args())


@pytest.mark.slow
def test_validation_runs_on_eval_steps_that_are_not_log_steps(
    tmp_path, harness_environ, monkeypatch
):
    from tests.real import trainer_harness_dp as harness
    from tests.real.tiny_real_dataset import build_tiny_real_dataset

    data_root = tmp_path / "data"
    harness.ensure_fixture(data_root)
    held_out = "tiny/real-tiny-eval"
    build_tiny_real_dataset(
        data_root,
        repo_id=held_out,
        n_episodes=2,
        ep_len=harness.EP_LEN,
        cameras=harness.CAMERAS,
        seed=2,
    )
    argv = harness.dp_argv(
        data_root=data_root,
        output_dir=tmp_path / "out",
        steps=2,
        seed=0,
        extra_argv=[
            "--eval-repo-ids",
            held_out,
            "--eval-freq",
            "2",
            "--log-freq",
            "3",
            "--num-eval-batches",
            "1",
        ],
    )
    val_calls: list[int] = []
    logged: list[tuple[int, set]] = []
    real_val = policy.compute_val_loss

    def val_spy(*args, **kwargs):
        val_calls.append(1)
        return real_val(*args, **kwargs)

    def log_spy(self, metrics, *, step):
        logged.append((step, set(metrics)))

    monkeypatch.setattr(policy, "compute_val_loss", val_spy)
    monkeypatch.setattr(policy.RunLogger, "log", log_spy)
    _run_trainer(monkeypatch, argv)
    assert len(val_calls) == 1
    assert [step for step, _ in logged] == [2]
    assert "val/loss" in logged[0][1] and "train/loss" not in logged[0][1]


def test_first_val_image_grid_restores_the_cuda_generators(tmp_path, monkeypatch):
    """torch.seed() reseeds every CUDA generator; the grid helper must put them back."""
    calls: list = []
    saved = [torch.tensor([7], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: saved)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: calls.append(states))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(("seed", seed)))
    batch = {"observation.images.cam": torch.rand(4, 3, 8, 8)}
    cpu_before = torch.random.get_rng_state()
    grid, _ = policy.save_first_val_image_grid(
        [batch, batch],
        ["observation.images.cam"],
        tmp_path,
        7,
        uint8_native=False,
    )
    assert grid is not None and grid[1].is_file()
    assert calls[-1] is saved
    assert torch.equal(torch.random.get_rng_state(), cpu_before)


@pytest.mark.gpu
def test_first_val_image_grid_leaves_the_cuda_stream_unchanged(tmp_path):
    torch.cuda.manual_seed_all(0)
    expected = torch.randn(4, device="cuda")
    torch.cuda.manual_seed_all(0)
    batch = {"observation.images.cam": torch.rand(4, 3, 8, 8)}
    policy.save_first_val_image_grid(
        [batch, batch],
        ["observation.images.cam"],
        tmp_path,
        1,
        uint8_native=False,
    )
    torch.testing.assert_close(torch.randn(4, device="cuda"), expected)
