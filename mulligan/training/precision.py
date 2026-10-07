"""Training precision helpers shared by training and profiling entry points."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

AMP_DTYPE_CHOICES = ("none", "bfloat16")


def configure_torch_precision(enable_tf32: bool) -> bool:
    """Apply process-wide CUDA TF32 switches.

    Returns True when TF32 was requested and CUDA was available.
    """
    if not torch.cuda.is_available():
        return False

    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
    return enable_tf32


def require_compiled_bf16_support(device: str | torch.device) -> None:
    """Fail at startup when the compiled actor would run under bf16 on a pre-Ampere GPU.

    GPUs below compute capability 8.0 have no native bf16. A Square-Narrow agent recipe
    (bf16 autocast plus the compiled actor) logged no training step in 45 minutes on an
    RTX 2070 SUPER, while the DIVL head recipe (bf16 autocast, actor frozen, nothing
    compiled) trained normally on the same GPU model. CPU devices, a missing CUDA runtime
    and ROCm are left to the caller.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip:
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    if props.major >= 8:
        return
    raise RuntimeError(
        f"cuda:{index} is {props.name} (compute capability {props.major}.{props.minor}), which "
        "has no native bf16; with bf16 autocast and the compiled actor, training did not reach "
        "its first step on such a GPU. Rerun with --training.compile_actor=False (keeps bf16 "
        "autocast; eager kernels) or --training.amp_dtype=none (FP32; there is no float16 "
        "path), or use an Ampere or newer GPU. See docs/compute.md."
    )


def autocast_context(device: str | torch.device, amp_dtype: str) -> ContextManager[None]:
    """Return the autocast context for the measured policy-update hot path."""
    if amp_dtype == "none":
        return nullcontext()
    if amp_dtype not in AMP_DTYPE_CHOICES:
        raise ValueError(f"amp_dtype must be one of {AMP_DTYPE_CHOICES}, got {amp_dtype!r}")

    device_type = torch.device(device).type
    if device_type != "cuda":
        return nullcontext()

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
