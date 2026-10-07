"""Shared deterministic image preprocessing for real-robot policies.

This module owns the crop+resize operation that must match between:

- training dataset proxies, which receive LeRobot CHW tensors, and
- live real-world eval, which receives HWC RGB numpy frames.

Train-time augmentation is intentionally outside this module. Callers should run
augmentation after these helpers if they need it.
"""

from __future__ import annotations

import operator

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as v2F


def coerce_int_coord(value, *, cam_key: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} for {cam_key!r} has non-integer coords")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as e:
            raise ValueError(f"{context} for {cam_key!r} has non-integer coords") from e
    try:
        return operator.index(value)
    except TypeError as e:
        raise ValueError(f"{context} for {cam_key!r} has non-integer coords") from e


def coerce_crop_box(
    cam_key: str,
    raw_box,
    *,
    context: str,
) -> tuple[int, int, int, int]:
    """Validate and normalize one native-pixel crop box (replacement or additive ROI)."""
    if not cam_key:
        raise ValueError(f"{context} has an empty camera key")
    try:
        parts = list(raw_box)
    except TypeError as e:
        raise ValueError(f"{context} for {cam_key!r} must be x0,y0,x1,y1") from e
    if len(parts) != 4:
        raise ValueError(
            f"{context} for {cam_key!r} must have exactly 4 ints x0,y0,x1,y1; got {raw_box!r}"
        )
    x0, y0, x1, y1 = (coerce_int_coord(p, cam_key=cam_key, context=context) for p in parts)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"{context} for {cam_key!r} is empty/inverted: "
            f"(x0={x0}, y0={y0}, x1={x1}, y1={y1}); require x1>x0 and y1>y0"
        )
    if x0 < 0 or y0 < 0:
        raise ValueError(
            f"{context} for {cam_key!r} has negative origin "
            f"(x0={x0}, y0={y0}); native pixels must be >= 0"
        )
    return (x0, y0, x1, y1)


def _validate_hw(name: str, hw: tuple[int, int]) -> tuple[int, int]:
    try:
        h, w = hw
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a (height, width) tuple, got {hw!r}") from e
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must be positive, got {hw!r}")
    return int(h), int(w)


def _validate_hwc_rgb_uint8(img: np.ndarray, *, context: str) -> None:
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{context} must be a numpy array, got {type(img).__name__}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{context} must have shape (H, W, 3), got {img.shape}")
    if img.dtype != np.uint8:
        raise TypeError(f"{context} must be uint8, got {img.dtype}")


def quantize_float01_to_uint8(imgs: torch.Tensor) -> torch.Tensor:
    """Quantize float32 [0,1] images to uint8 via ``clamp(0,1)*255`` truncation.

    SINGLE source of truth for the quantization formula: the replay buffer's
    ``VisionReplayBuffer.pack_images_uint8`` and the DP worker-side post-aug
    quantize (``QuantizeUint8PostTransform``) both route through here, so the
    bytes are identical wherever the formula is applied.
    """
    if imgs.dtype != torch.float32:
        raise TypeError(f"quantize_float01_to_uint8 expects float32 [0,1], got {imgs.dtype}")
    return (imgs.clamp(0.0, 1.0) * 255).to(torch.uint8)


def uint8_to_float01(imgs: torch.Tensor) -> torch.Tensor:
    """Re-float packed uint8 images to float32 [0,1] (inverse of the quantize).

    Fails loudly on non-uint8 input: on the uint8-quantized throughput tier a
    float tensor here means the worker-side quantize did not run, i.e. the
    pipeline is not in the state the tier claims.
    """
    if imgs.dtype != torch.uint8:
        raise TypeError(f"uint8_to_float01 expects uint8 input, got {imgs.dtype}")
    # div by 255, NOT mul by (1/255): lerobot's video decode converts uint8
    # frames with `/ 255`, and the reciprocal differs by 1 ULP for some values
    # -> 1-LSB pixel flips after resize+quantize. Same formula = bitwise-equal
    # floats between the uint8 and float decode paths.
    return imgs.to(dtype=torch.float32).div_(255.0)


class QuantizeUint8PostTransform:
    """Picklable worker-side post-transform: run ``inner`` (may be ``None``),
    then quantize the float [0,1] result to uint8.

    Used by the DP ``--uint8-native-images`` quantized-aug tier: the FULL
    existing worker transform stack (crop+resize+aug) runs unchanged, and this
    wrapper quantizes the final float 224^2 frame so worker->main IPC ships
    uint8 (4x smaller). The trainer re-floats on GPU with
    :func:`uint8_to_float01`; round-trip error is bounded by 1/255.
    Module-level class so spawn-context DataLoader workers can pickle it.
    """

    def __init__(self, inner=None):
        if inner is not None and not callable(inner):
            raise TypeError(f"inner transform must be callable or None, got {type(inner).__name__}")
        self.inner = inner

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.inner is not None:
            img = self.inner(img)
        return quantize_float01_to_uint8(img)


def _chw_tensor_to_float01(img: torch.Tensor) -> torch.Tensor:
    if img.dtype == torch.uint8:
        # div, not mul-by-reciprocal — see uint8_to_float01.
        return img.to(dtype=torch.float32).div_(255.0)
    if img.is_floating_point():
        min_v = float(img.min())
        max_v = float(img.max())
        if min_v < -1e-6 or max_v > 1.0 + 1e-6:
            raise ValueError(
                "floating image tensor must be in [0, 1] before policy preprocessing, "
                f"got min={min_v:.6g}, max={max_v:.6g}"
            )
        return img.to(dtype=torch.float32)
    raise TypeError(f"image tensor must be uint8 or floating [0,1], got {img.dtype}")


def _crop_chw_tensor(
    img: torch.Tensor,
    box: tuple[int, int, int, int],
    *,
    context: str = "image crop",
) -> torch.Tensor:
    x0, y0, x1, y1 = box
    h, w = img.shape[-2:]
    if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0:
        raise ValueError(
            f"{context} box (x0={x0}, y0={y0}, x1={x1}, y1={y1}) is invalid; "
            "require nonnegative origin and x1>x0, y1>y0"
        )
    if x1 > w or y1 > h:
        raise ValueError(
            f"{context} box (x0={x0}, y0={y0}, x1={x1}, y1={y1}) exceeds frame "
            f"of size HxW={h}x{w}; box must lie inside the frame"
        )
    return img[..., y0:y1, x0:x1]


def preprocess_hwc_rgb_uint8_for_policy(
    img: np.ndarray,
    *,
    target_hw: tuple[int, int],
    crop_box: tuple[int, int, int, int] | None = None,
    crop_reference_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply the deterministic real-policy image path to one HWC RGB uint8 frame.

    If ``crop_reference_hw`` is supplied and the input frame is not already that
    size, the frame is first resized to that reference with ``INTER_AREA``. This
    is the live-eval path: native ZED frames are downscaled to the
    stored 640x480 frame before applying role-keyed crop boxes or the final
    policy resize.

    The final policy resize uses torchvision's tensor resize path with bilinear
    interpolation and antialiasing, matching the training transform. The result
    is returned as CHW float32 in ``[0, 1]``.
    """
    _validate_hwc_rgb_uint8(img, context="policy image")
    target_h, target_w = _validate_hw("target_hw", target_hw)

    if crop_reference_hw is not None:
        ref_h, ref_w = _validate_hw("crop_reference_hw", crop_reference_hw)
        if img.shape[:2] != (ref_h, ref_w):
            import cv2

            img = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    out = preprocess_chw_tensor_for_policy(
        tensor,
        target_hw=(target_h, target_w),
        crop_box=crop_box,
    )
    return np.ascontiguousarray(out.numpy())


def preprocess_chw_tensor_for_policy(
    img: torch.Tensor,
    *,
    target_hw: tuple[int, int],
    crop_box: tuple[int, int, int, int] | None = None,
    crop_reference_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Torch/LeRobot adapter for :func:`preprocess_hwc_rgb_uint8_for_policy`.

    Supports ``(C,H,W)`` and batched ``(...,C,H,W)`` tensors. The returned tensor
    is float32 in ``[0,1]`` with the same leading dimensions.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img must be a torch.Tensor, got {type(img).__name__}")
    if img.ndim < 3 or img.shape[-3] != 3:
        raise ValueError(f"expected tensor shape (...,3,H,W), got {tuple(img.shape)}")

    target_h, target_w = _validate_hw("target_hw", target_hw)
    out = _chw_tensor_to_float01(img)

    if crop_reference_hw is not None:
        ref_h, ref_w = _validate_hw("crop_reference_hw", crop_reference_hw)
        if out.shape[-2:] != (ref_h, ref_w):
            raise ValueError(
                "tensor policy preprocessing with crop_reference_hw expects an already "
                f"reference-sized tensor, got HxW={tuple(out.shape[-2:])}, "
                f"expected {(ref_h, ref_w)}"
            )

    if crop_box is not None:
        out = _crop_chw_tensor(out, crop_box, context="policy image crop")

    if out.shape[-2:] != (target_h, target_w):
        out = v2F.resize(
            out,
            size=(target_h, target_w),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if out.shape[-2:] != (target_h, target_w):
            raise RuntimeError(
                f"torchvision resize returned shape {tuple(out.shape[-2:])}, "
                f"expected {(target_h, target_w)}"
            )
    return out.contiguous()


class GpuImagePreprocessor:
    """Batched crop+resize+float image preprocessing, runnable on GPU.

    This is the main-process (typically CUDA) counterpart of the per-worker
    :class:`mulligan.real.policy.side_crop._PerCameraCropSubset` transform. With the
    ``--uint8-native-images`` throughput path, DataLoader workers return RAW
    uint8 native frames and the (expensive) crop+antialias-resize+float machinery
    is moved out of the workers and run here, batched, on the accelerator.

    It applies the exact same :func:`preprocess_chw_tensor_for_policy` as the
    per-worker proxy, so results are bit-identical on the SAME device and bounded
    to <=1 uint8 LSB across CPU->GPU (torchvision resize rounds differently per
    backend). Per-camera semantics mirror ``_PerCameraCropSubset.__getitem__``:

    - a camera key present in ``crop_feature_map`` gets crop-box + resize;
    - a camera key absent from the map gets resize-only (the no-crop-key branch);
    - ``crop_reference_hw`` (when set) validates the native input size, matching
      the per-worker proxy's ``crop_reference_hw`` contract (it is a no-op when
      the native frame already equals the reference, which is the training case).

    Input tensors may carry leading batch/temporal dims ``(..., 3, H, W)``; the
    underlying helper preserves them, so ``(B, T, 3, H, W)`` refresh/DP batches
    work unchanged. Camera keys the preprocessor is called with need not exhaust
    ``crop_feature_map`` and vice-versa; only the keys passed in ``images_by_cam``
    are processed.
    """

    def __init__(
        self,
        crop_feature_map: dict[str, tuple[int, int, int, int]],
        target_hw: tuple[int, int],
        *,
        crop_reference_hw: tuple[int, int] | None = None,
    ):
        self.crop_feature_map = dict(crop_feature_map)
        self.target_hw = _validate_hw("target_hw", target_hw)
        self.crop_reference_hw = crop_reference_hw

    def __call__(self, images_by_cam: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for cam, img in images_by_cam.items():
            out[cam] = preprocess_chw_tensor_for_policy(
                img,
                target_hw=self.target_hw,
                crop_box=self.crop_feature_map.get(cam),
                crop_reference_hw=self.crop_reference_hw,
            )
        return out


class PolicyImagePreprocessTransform:
    """Picklable transform wrapper for LeRobot dataset workers.

    ``torchvision`` transform objects are applied inside DataLoader workers. A
    named class avoids non-picklable lambdas/closures while routing resize-only
    training paths through the exact same helper as live eval.
    """

    def __init__(
        self,
        target_hw: tuple[int, int],
        *,
        crop_box: tuple[int, int, int, int] | None = None,
        crop_reference_hw: tuple[int, int] | None = None,
    ):
        self.target_hw = _validate_hw("target_hw", target_hw)
        self.crop_box = crop_box
        self.crop_reference_hw = crop_reference_hw

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return preprocess_chw_tensor_for_policy(
            img,
            target_hw=self.target_hw,
            crop_box=self.crop_box,
            crop_reference_hw=self.crop_reference_hw,
        )
