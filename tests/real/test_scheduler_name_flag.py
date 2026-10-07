"""Unit tests for ``--scheduler-name`` on the real DP trainer.

The flag exists for warm-start continuation runs (routing_d2 R8 LR-schedule
counterfactual): resuming a finished 100k cosine run under a *fresh* cosine
would re-raise the LR back to its 1e-4 peak, so the continuation asks for a flat
schedule instead. These tests pin the two properties that matter:

* the CLI accepts the three supported names and defaults to ``None`` (= keep the
  policy preset's ``cosine``), and
* ``scheduler_name="constant"`` actually produces a FLAT LR across the whole
  horizon, while the default ``cosine`` decays to ~0 -- i.e. the config field
  reaches diffusers' ``get_scheduler`` and changes the schedule.

CPU-only: a one-parameter dummy optimizer, no policy, no dataset, no GPU.
"""

from __future__ import annotations

import sys

import pytest
import torch
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

from mulligan.real.train.policy import parse_args

TOTAL_STEPS = 10_000  # >> the 500-step preset warmup, so the cosine tail is visible
BASE_LR = 1e-5


def _build_scheduler(scheduler_name: str):
    """Build the LR scheduler exactly the way ``dp_build_optimizer`` does."""
    cfg = DiffusionConfig(scheduler_name=scheduler_name, device="cpu")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    return optimizer, cfg.get_scheduler_preset().build(optimizer, TOTAL_STEPS)


def _lr_trace(scheduler_name: str) -> list[float]:
    optimizer, scheduler = _build_scheduler(scheduler_name)
    trace = []
    for _ in range(TOTAL_STEPS):
        trace.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return trace


def test_constant_scheduler_lr_is_flat():
    trace = _lr_trace("constant")
    assert len(trace) == TOTAL_STEPS
    for step, lr in enumerate(trace):
        assert lr == pytest.approx(BASE_LR, rel=1e-9), f"step {step} lr={lr} != {BASE_LR}"


def test_cosine_scheduler_lr_decays_to_zero():
    """Control: the preset default is NOT flat, so the flat trace above is the flag's doing."""
    trace = _lr_trace("cosine")
    assert trace[-1] < 1e-3 * BASE_LR
    assert trace[8 * TOTAL_STEPS // 10] < 0.2 * BASE_LR
    # Half-way is near half the peak (the 500-step warmup shifts progress slightly).
    assert trace[TOTAL_STEPS // 2] == pytest.approx(0.5 * BASE_LR, rel=0.1)


def test_constant_with_warmup_ramps_then_holds():
    trace = _lr_trace("constant_with_warmup")
    warmup = DiffusionConfig(device="cpu").scheduler_warmup_steps
    assert trace[0] < BASE_LR
    for lr in trace[warmup:]:
        assert lr == pytest.approx(BASE_LR, rel=1e-9)


@pytest.mark.parametrize("name", ["cosine", "constant", "constant_with_warmup"])
def test_cli_accepts_scheduler_name(monkeypatch, name):
    monkeypatch.setattr(
        sys, "argv", ["mulligan.real.train.policy", "--repo-ids", "x/y", "--scheduler-name", name]
    )
    assert parse_args().scheduler_name == name


def test_cli_scheduler_name_defaults_to_none(monkeypatch):
    """Default None means the policy preset's own 'cosine' is left untouched."""
    monkeypatch.setattr(sys, "argv", ["mulligan.real.train.policy", "--repo-ids", "x/y"])
    assert parse_args().scheduler_name is None


def test_cli_rejects_unknown_scheduler_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["mulligan.real.train.policy", "--repo-ids", "x/y", "--scheduler-name", "linear"],
    )
    with pytest.raises(SystemExit):
        parse_args()
