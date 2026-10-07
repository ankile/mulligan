"""Tests for training precision helper behavior."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from mulligan.training.precision import (
    autocast_context,
    configure_torch_precision,
    require_compiled_bf16_support,
)


def test_autocast_context_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="amp_dtype must be"):
        autocast_context("cuda", "fp8")


def test_autocast_context_rejects_float16_without_grad_scaler():
    with pytest.raises(ValueError, match="amp_dtype must be"):
        autocast_context("cuda", "float16")


def test_autocast_context_returns_nullcontext_for_cpu_amp_request():
    context = autocast_context("cpu", "bfloat16")

    assert isinstance(context, nullcontext)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_autocast_context_uses_cuda_bfloat16_autocast():
    with autocast_context("cuda", "bfloat16"):
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") is torch.bfloat16


def test_configure_torch_precision_no_cuda_is_false(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert configure_torch_precision(True) is False
    assert configure_torch_precision(False) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_configure_torch_precision_toggles_cuda_tf32():
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    try:
        assert configure_torch_precision(True) is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"

        assert configure_torch_precision(False) is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
        torch.set_float32_matmul_precision(original_precision)


def _fake_gpu(monkeypatch, name, major, minor):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "hip", None)
    props = SimpleNamespace(name=name, major=major, minor=minor)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)


def test_compiled_bf16_fails_loudly_on_turing(monkeypatch):
    _fake_gpu(monkeypatch, "NVIDIA GeForce RTX 2070 SUPER", 7, 5)
    with pytest.raises(RuntimeError, match=r"RTX 2070 SUPER.*7\.5.*compile_actor=False"):
        require_compiled_bf16_support("cuda")
    with pytest.raises(RuntimeError, match="cuda:1"):
        require_compiled_bf16_support("cuda:1")


def test_compiled_bf16_passes_on_ampere_and_cpu(monkeypatch):
    _fake_gpu(monkeypatch, "NVIDIA A40", 8, 6)
    require_compiled_bf16_support("cuda")
    _fake_gpu(monkeypatch, "NVIDIA GeForce RTX 2070 SUPER", 7, 5)
    require_compiled_bf16_support("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    require_compiled_bf16_support("cuda")
