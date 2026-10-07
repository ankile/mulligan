"""Local resume state used by mulligan.training.train (mulligan.training.resume)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mulligan.configs.training import TrainingConfig
from mulligan.configs.wandb import WandBConfig
from mulligan.training.resume import AutoResumeManager, should_save_resume_checkpoint


def _make_manager(tmp_path: Path, enabled: bool = True) -> AutoResumeManager:
    return AutoResumeManager(tmp_path, run_name="run-1", enabled=enabled)


def _state_payload() -> dict:
    return {
        "policy_state_dict": {"w": torch.zeros(4)},
        "optimizer_state_dicts": {"opt": {"step": 0}},
    }


def test_resume_manager_round_trip(tmp_path):
    manager = _make_manager(tmp_path)
    manager.acquire_lock()
    run_dir = tmp_path / "run"
    manager.initialize_fresh_run(run_dir)

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    manager.save_training_state(
        policy_state_dict=model.state_dict(),
        optimizer_state_dicts={"policy": optimizer.state_dict()},
        step=5_000,
        best_success_rate=91.5,
    )

    meta = manager.load_metadata()
    state = manager.load_training_state()
    assert meta is not None
    assert meta.run_dir == str(run_dir)
    assert meta.last_checkpoint_step == 5_000
    assert not meta.completed
    assert state is not None
    assert state["last_checkpoint_step"] == 5_000
    assert state["best_success_rate"] == 91.5
    assert "policy" in state["optimizer_state_dicts"]

    manager.mark_completed()
    assert manager.load_metadata().completed


def test_resume_identity_is_the_run_name(tmp_path):
    a = AutoResumeManager(tmp_path, run_name="seed-1", enabled=True)
    b = AutoResumeManager(tmp_path, run_name="seed-2", enabled=True)
    assert a.resume_dir != b.resume_dir
    assert a.resume_dir == AutoResumeManager(tmp_path, run_name="seed-1", enabled=True).resume_dir


def test_resume_manager_rejects_concurrent_lock(tmp_path):
    manager_a = _make_manager(tmp_path)
    manager_b = _make_manager(tmp_path)
    manager_a.acquire_lock()
    with pytest.raises(RuntimeError, match="Resume lock already held"):
        manager_b.acquire_lock()


def test_resume_manager_disabled_is_noop(tmp_path):
    manager = _make_manager(tmp_path, enabled=False)
    manager.acquire_lock()
    manager.initialize_fresh_run(tmp_path / "run")
    manager.save_training_state(
        policy_state_dict={}, optimizer_state_dicts={}, step=1, best_success_rate=0.0
    )
    manager.mark_completed()
    assert manager.load_metadata() is None
    assert manager.load_training_state() is None
    assert not (tmp_path / "_resume").exists()


def test_save_after_disk_meta_disappears(tmp_path: Path) -> None:
    """Metadata vanishing mid-run (network filesystem hiccup) must not crash a save."""
    mgr = _make_manager(tmp_path)
    mgr.acquire_lock()
    mgr.initialize_fresh_run(tmp_path / "run-x")
    mgr.save_training_state(**_state_payload(), step=100, best_success_rate=0.5)
    mgr.meta_path.unlink()
    mgr.save_training_state(**_state_payload(), step=200, best_success_rate=0.6)
    payload = json.loads(mgr.meta_path.read_text())
    assert payload["last_checkpoint_step"] == 200
    assert payload["run_name"] == "run-1"


def test_save_without_init_raises(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.acquire_lock()
    with pytest.raises(RuntimeError, match="initializing resume metadata"):
        mgr.save_training_state(**_state_payload(), step=100, best_success_rate=0.5)


def test_load_metadata_caches_for_later_saves(tmp_path: Path) -> None:
    mgr_a = _make_manager(tmp_path)
    mgr_a.acquire_lock()
    mgr_a.initialize_fresh_run(tmp_path / "run-x")
    mgr_a._lock_file.close()

    mgr_b = _make_manager(tmp_path)
    mgr_b.acquire_lock()
    assert mgr_b.load_metadata().run_name == "run-1"
    mgr_b.meta_path.unlink()
    mgr_b.save_training_state(**_state_payload(), step=300, best_success_rate=0.7)
    assert json.loads(mgr_b.meta_path.read_text())["last_checkpoint_step"] == 300


def test_mark_completed_uses_cached_meta(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.acquire_lock()
    mgr.initialize_fresh_run(tmp_path / "run-x")
    mgr.meta_path.unlink()
    mgr.mark_completed()
    assert json.loads(mgr.meta_path.read_text())["completed"] is True


def test_mark_completed_without_meta_raises(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.acquire_lock()
    with pytest.raises(RuntimeError, match="initializing resume metadata"):
        mgr.mark_completed()


def test_resume_checkpoint_boundaries():
    assert should_save_resume_checkpoint(step=1_000, resume_checkpoint_freq=1_000)
    assert not should_save_resume_checkpoint(step=250, resume_checkpoint_freq=1_000)
    assert not should_save_resume_checkpoint(step=1_000, resume_checkpoint_freq=0)
    assert not should_save_resume_checkpoint(step=0, resume_checkpoint_freq=1_000)


def test_config_validation():
    with pytest.raises(ValueError, match="log_freq must be >= 1"):
        WandBConfig(log_freq=0)
    with pytest.raises(ValueError, match="resume_checkpoint_freq must be >= 0"):
        TrainingConfig(resume_checkpoint_freq=-1)
    with pytest.raises(ValueError, match="amp_dtype must be"):
        TrainingConfig(amp_dtype="fp8")
    # float16 needs a grad scaler, which the trainer does not use
    with pytest.raises(ValueError, match="amp_dtype must be"):
        TrainingConfig(amp_dtype="float16")
    assert TrainingConfig(compile_mode="max-autotune-no-cudagraphs").compile_mode == (
        "max-autotune-no-cudagraphs"
    )
    assert WandBConfig().project == "mulligan"
