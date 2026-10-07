"""``python -m mulligan.training.train`` resume: bitwise continuation and config pinning (CPU)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mulligan.training.train import _check_resume_config
from tests.unit.sim_dataset import build_sim_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _argv(dataset: Path, ckpt: Path, run_name: str, steps: int, *extra: str) -> list[str]:
    return [
        f"--dataset.repo_ids={dataset}",
        "--env.name=NutAssemblySquare",
        "--env.robot=Panda",
        "--policy.type=idql",
        "--policy.hidden_dims=[32, 32]",
        "--policy.down_dims=[16, 32]",
        "--policy.num_action_samples=2",
        "--policy.chunk_size=4",
        "--policy.n_action_steps=2",
        "--training.batch_size=8",
        f"--training.training_steps={steps}",
        "--training.seed=0",
        "--training.resume_checkpoint_freq=10",
        "--training.amp_dtype=none",
        "--training.enable_tf32=False",
        "--training.compile_actor=False",
        "--wandb.enabled=False",
        f"--wandb.run_name={run_name}",
        "--system.device=cpu",
        f"--system.checkpoint_dir={ckpt}",
        *extra,
    ]


def _train(dataset: Path, ckpt: Path, run_name: str, steps: int, *extra: str):
    argv = _argv(dataset, ckpt, run_name, steps, *extra)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", "mulligan.training.train", *argv],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _ok(proc) -> str:
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return proc.stdout


def _reopen(ckpt: Path) -> None:
    """Clear the completed flag, as if the job had been killed right after its last
    resume checkpoint."""
    (meta_path,) = ckpt.glob("_resume/*/resume_meta.json")
    meta = json.loads(meta_path.read_text())
    meta["completed"] = False
    meta_path.write_text(json.dumps(meta))


def _tensors(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tensors(v, f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _tensors(v, f"{prefix}/{i}")


def _final_weights(ckpt: Path) -> dict:
    (path,) = ckpt.glob("*/checkpoints/final_model/policy.pt")
    return dict(_tensors(torch.load(path, map_location="cpu", weights_only=False)))


@pytest.mark.slow
def test_resume_is_a_bitwise_continuation(tmp_path):
    dataset = build_sim_dataset(tmp_path / "ds")

    _ok(_train(dataset, tmp_path / "straight", "straight", 30))

    _ok(_train(dataset, tmp_path / "resumed", "resumed", 10))
    _reopen(tmp_path / "resumed")
    out = _ok(_train(dataset, tmp_path / "resumed", "resumed", 30))
    assert "Restored local training state from step 10" in out

    a, b = _final_weights(tmp_path / "straight"), _final_weights(tmp_path / "resumed")
    assert len(a) > 20 and a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key

    # anything but training_steps must match the saved config
    _reopen(tmp_path / "resumed")
    proc = _train(dataset, tmp_path / "resumed", "resumed", 40, "--policy.expectile=0.8")
    assert proc.returncode != 0
    assert "different config" in proc.stdout + proc.stderr
    assert "policy.expectile" in proc.stdout + proc.stderr


def test_check_resume_config_allows_only_training_steps():
    saved = {"training": {"training_steps": 10, "batch_size": 8}, "seed": 1}
    _check_resume_config(saved, {"training": {"training_steps": 99, "batch_size": 8}, "seed": 1})
    with pytest.raises(RuntimeError, match="training.batch_size"):
        _check_resume_config(
            saved, {"training": {"training_steps": 10, "batch_size": 4}, "seed": 1}
        )
    with pytest.raises(RuntimeError, match="seed"):
        _check_resume_config(saved, {"training": {"training_steps": 10, "batch_size": 8}})


@pytest.mark.slow
def test_single_eval_num_action_samples_is_honored(tmp_path):
    """``--eval.eval_num_action_samples=[1]`` evaluates at N=1, not at policy.num_action_samples."""
    dataset = build_sim_dataset(tmp_path / "ds")
    eval_flags = [
        "--eval.freq=5",
        "--eval.episodes=1",
        "--eval.max_steps=4",
        "--eval.num_envs=1",
        "--eval.sync_envs=True",
        "--eval.eval_num_action_samples=[1]",
    ]
    out = _ok(_train(dataset, tmp_path / "ck", "eval-n1", 5, *eval_flags))
    assert "Evaluating policy (n=1)..." in out


_WANDB_PROBE = r"""
import sys
import wandb
import mulligan.training.train as train

_init = wandb.init


def init(**kw):
    run = _init(**{**kw, "mode": "disabled"})
    # wandb.init rebinds wandb.log to the run; record the steps instead
    wandb.log = lambda payload, step=None, **kw: print("WANDB_LOG_STEP", step, flush=True)
    return run


wandb.init = init
train.upload_checkpoint_to_wandb = lambda **kw: print("ARTIFACT", kw["artifact_name"], kw["metadata"]["step"], flush=True)
sys.argv = ["train", *sys.argv[1:]]
train.main()
"""


@pytest.mark.slow
def test_wandb_log_freq_and_final_checkpoint_step(tmp_path):
    """W&B metrics follow wandb.log_freq (not only multiples of 1000), and the final
    checkpoint records the last completed step."""
    dataset = build_sim_dataset(tmp_path / "ds")
    argv = _argv(dataset, tmp_path / "ck", "logfreq", 4, "--wandb.log_freq=2")
    argv = [a for a in argv if a != "--wandb.enabled=False"] + ["--wandb.enabled=True"]
    proc = subprocess.run(
        [sys.executable, "-c", _WANDB_PROBE, *argv],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = _ok(proc)
    assert [line for line in out.splitlines() if line.startswith("WANDB_LOG_STEP")] == [
        "WANDB_LOG_STEP 2",
        "WANDB_LOG_STEP 4",
    ]
    (artifact,) = [line for line in out.splitlines() if line.startswith("ARTIFACT")]
    assert artifact.endswith("-final-step-4 4"), artifact
    (meta,) = (tmp_path / "ck").glob("*/checkpoints/final_model/metadata.json")
    assert json.loads(meta.read_text())["step"] == 4


def test_eval_num_action_samples_validation():
    from mulligan.configs.eval import EvalConfig

    assert EvalConfig(eval_num_action_samples=[1]).eval_num_action_samples == [1]
    for bad in ([], [0], [4, -1]):
        with pytest.raises(ValueError, match="eval_num_action_samples"):
            EvalConfig(eval_num_action_samples=bad)
