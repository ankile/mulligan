from __future__ import annotations

import pytest
import torch

from mulligan.configs.system import SystemConfig


def test_cpu_config_is_valid() -> None:
    assert SystemConfig(device="cpu").device == "cpu"


def test_invalid_device_rejected() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        SystemConfig(device="tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a host without CUDA")
def test_cuda_request_without_cuda_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        SystemConfig(device="cuda")
