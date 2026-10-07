"""Tests for the ``--uint8-native-images`` throughput path (both real trainers).

The feature moves the per-worker crop+antialias-resize+float machinery OUT of the
DataLoader workers (which now return RAW uint8 native frames) and onto the GPU in
the main process, via the shared :class:`GpuImagePreprocessor` (same
``preprocess_chw_tensor_for_policy`` the per-worker proxy uses). These tests pin:

  1. CPU parity: GpuImagePreprocessor output == the ``_PerCameraCropSubset`` proxy
     output, bitwise (same function on the same device), incl. the resize-only
     (no-crop-key) branch.
  2. GPU parity: CPU-path vs GPU-path pixels after ``pack_images_uint8`` differ by
     <=1 uint8 LSB with <1% of pixels mismatching (bounds device resize rounding).
  3. Raw-mode flip flips the inner reader's ``return_uint8`` and short-circuits the
     per-camera transform loop.
  4. Pickling isolation: flipping raw mode on a pickled copy never mutates the
     original proxy (the worker-isolation invariant the trainers rely on).
  5. Loud failures on a missing ``return_uint8`` reader, a non-proxy sub-dataset,
     and an invalid image shape.
"""

import pickle

import numpy as np
import pytest
import torch

from mulligan.real.policy.image_preprocess import (
    GpuImagePreprocessor,
    PolicyImagePreprocessTransform,
    QuantizeUint8PostTransform,
    quantize_float01_to_uint8,
    uint8_to_float01,
)
from mulligan.real.policy.side_crop import (
    _PerCameraCropSubset,
    build_crop_feature_map,
    set_raw_uint8_mode_on_subdatasets,
)
from mulligan.data.vision_replay_buffer import VisionReplayBuffer

CAM = "20000002_left"
FKEY = f"observation.images.{CAM}"
WRIST_FKEY = "observation.images.10000001_left"
BOX = (360, 120, 540, 330)  # x0, y0, x1, y1 on a 480x640 native frame
NATIVE_H, NATIVE_W = 480, 640
TARGET_HW = (224, 224)


def _native_uint8_chw(seed: int = 0, *, temporal: bool = False) -> torch.Tensor:
    """Deterministic native uint8 CHW (or (T,3,H,W)) frame."""
    rng = np.random.default_rng(seed)
    hwc = rng.integers(0, 256, size=(NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    chw = torch.from_numpy(hwc.transpose(2, 0, 1).copy())
    if temporal:
        chw2 = torch.from_numpy(
            np.random.default_rng(seed + 1)
            .integers(0, 256, size=(NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
            .transpose(2, 0, 1)
            .copy()
        )
        return torch.stack([chw, chw2], dim=0)  # (2, 3, H, W)
    return chw


# ---- Picklable mock LeRobot sub-dataset (module-level so it survives pickling) ----


class _FakeReader:
    def __init__(self):
        self._return_uint8 = False


class _FakeMeta:
    def __init__(self, camera_keys):
        self.camera_keys = camera_keys


class _FakeInner:
    """Minimal stand-in for a LeRobotDataset sub-dataset with a reader."""

    def __init__(self, frames_by_cam, *, with_reader: bool = True):
        self._frames = frames_by_cam
        self.reader = _FakeReader() if with_reader else None
        self._return_uint8 = False
        self.cleared = False
        self.meta = _FakeMeta(list(frames_by_cam.keys()))

    def clear_image_transforms(self):
        self.cleared = True

    def __getitem__(self, idx):
        assert idx == 0
        return {cam: f.clone() for cam, f in self._frames.items()}

    def __len__(self):
        return 1


def _make_proxy(
    frames_by_cam,
    crop_feature_map,
    *,
    crop_resize_hw=TARGET_HW,
    base_transform=None,
    with_reader=True,
):
    inner = _FakeInner(frames_by_cam, with_reader=with_reader)
    return _PerCameraCropSubset(
        inner,
        crop_feature_map,
        base_transform=base_transform,
        crop_resize_hw=crop_resize_hw,
        crop_post_transform=None,
        crop_reference_hw=None,
    )


# ---------------------------------------------------------------------------
# 1. CPU parity: GpuImagePreprocessor == _PerCameraCropSubset proxy, bitwise
# ---------------------------------------------------------------------------


def test_gpu_preprocessor_cpu_parity_matches_proxy_cropped():
    native = _native_uint8_chw(seed=1)
    crop_map = build_crop_feature_map({CAM: BOX})
    proxy = _make_proxy({FKEY: native}, crop_map)

    proxy_out = proxy[0][FKEY]
    gpu = GpuImagePreprocessor(crop_map, TARGET_HW)
    helper_out = gpu({FKEY: native.clone()})[FKEY]

    assert proxy_out.shape == (3, *TARGET_HW)
    assert helper_out.dtype == torch.float32
    torch.testing.assert_close(helper_out, proxy_out, rtol=0, atol=0)
    assert helper_out.numpy().tobytes() == proxy_out.numpy().tobytes()


def test_gpu_preprocessor_cpu_parity_resize_only_branch():
    """A camera key ABSENT from the crop map gets resize-only on both paths."""
    native = _native_uint8_chw(seed=2)
    crop_map = build_crop_feature_map({CAM: BOX})  # WRIST not in the map
    # In production the proxy's non-cropped path runs base_transform = the shared
    # resize (PolicyImagePreprocessTransform), never None.
    proxy = _make_proxy(
        {WRIST_FKEY: native},
        crop_map,
        base_transform=PolicyImagePreprocessTransform(TARGET_HW),
    )

    proxy_out = proxy[0][WRIST_FKEY]
    gpu = GpuImagePreprocessor(crop_map, TARGET_HW)
    helper_out = gpu({WRIST_FKEY: native.clone()})[WRIST_FKEY]

    torch.testing.assert_close(helper_out, proxy_out, rtol=0, atol=0)
    assert helper_out.numpy().tobytes() == proxy_out.numpy().tobytes()


def test_gpu_preprocessor_batched_temporal_shape():
    native = _native_uint8_chw(seed=3, temporal=True)  # (2, 3, H, W)
    batched = torch.stack([native, native], dim=0)  # (B=2, T=2, 3, H, W)
    crop_map = build_crop_feature_map({CAM: BOX})
    gpu = GpuImagePreprocessor(crop_map, TARGET_HW)
    out = gpu({FKEY: batched})[FKEY]
    assert out.shape == (2, 2, 3, *TARGET_HW)
    # Each (T, 3, H, W) slice matches the single-item path.
    single = GpuImagePreprocessor(crop_map, TARGET_HW)({FKEY: native})[FKEY]
    torch.testing.assert_close(out[0], single, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. GPU parity (skip if no CUDA): device rounding bounded to <=1 uint8 LSB
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_vs_cpu_pack_within_one_lsb():
    native = _native_uint8_chw(seed=4)
    crop_map = build_crop_feature_map({CAM: BOX})

    cpu_gpu = GpuImagePreprocessor(crop_map, TARGET_HW)
    cpu_packed = VisionReplayBuffer.pack_images_uint8(cpu_gpu({FKEY: native.clone()})[FKEY])

    dev = torch.device("cuda")
    device_gpu = GpuImagePreprocessor(crop_map, TARGET_HW)
    device_packed = VisionReplayBuffer.pack_images_uint8(
        device_gpu({FKEY: native.clone().to(dev)})[FKEY]
    ).cpu()

    diff = (cpu_packed.to(torch.int16) - device_packed.to(torch.int16)).abs()
    assert int(diff.max()) <= 1, f"max abs diff {int(diff.max())} exceeds 1 uint8 LSB"
    mismatch_frac = float((diff > 0).float().mean())
    assert mismatch_frac < 0.01, f"mismatch fraction {mismatch_frac:.4f} exceeds 1%"


# ---------------------------------------------------------------------------
# 3. Raw-mode flip
# ---------------------------------------------------------------------------


def test_raw_mode_flips_reader_and_skips_transforms():
    native = _native_uint8_chw(seed=5)
    crop_map = build_crop_feature_map({CAM: BOX})
    proxy = _make_proxy({FKEY: native}, crop_map)

    # Normal mode: reader.return_uint8 off, item is crop+resized float32.
    assert proxy._inner.reader._return_uint8 is False
    assert proxy[0][FKEY].shape == (3, *TARGET_HW)

    proxy.set_raw_uint8_mode(True)
    assert proxy._inner.reader._return_uint8 is True
    assert proxy._inner._return_uint8 is True
    raw_item = proxy[0][FKEY]
    # Raw mode returns the inner item unchanged (native shape, untouched dtype).
    assert raw_item.shape == native.shape
    assert raw_item.dtype == torch.uint8
    torch.testing.assert_close(raw_item, native, rtol=0, atol=0)

    proxy.set_raw_uint8_mode(False)
    assert proxy._inner.reader._return_uint8 is False
    assert proxy[0][FKEY].shape == (3, *TARGET_HW)


def test_set_raw_uint8_mode_on_subdatasets_helper():
    crop_map = build_crop_feature_map({CAM: BOX})
    proxies = [_make_proxy({FKEY: _native_uint8_chw(seed=i)}, crop_map) for i in range(3)]
    set_raw_uint8_mode_on_subdatasets(proxies, True)
    assert all(p._raw_uint8_mode for p in proxies)
    assert all(p._inner.reader._return_uint8 for p in proxies)
    set_raw_uint8_mode_on_subdatasets(proxies, False)
    assert not any(p._raw_uint8_mode for p in proxies)


# ---------------------------------------------------------------------------
# 4. Pickling isolation (the worker-isolation invariant)
# ---------------------------------------------------------------------------


def test_raw_mode_flip_on_pickled_copy_does_not_touch_original():
    native = _native_uint8_chw(seed=6)
    crop_map = build_crop_feature_map({CAM: BOX})
    original = _make_proxy({FKEY: native}, crop_map)

    # A DataLoader worker receives a PICKLED copy; flipping raw mode on that copy
    # must not affect the main-process (original) proxy or its inner reader.
    worker_copy = pickle.loads(pickle.dumps(original))
    worker_copy.set_raw_uint8_mode(True)

    assert worker_copy._raw_uint8_mode is True
    assert worker_copy._inner.reader._return_uint8 is True
    # Original untouched.
    assert original._raw_uint8_mode is False
    assert original._inner.reader._return_uint8 is False
    assert original[0][FKEY].shape == (3, *TARGET_HW)


# ---------------------------------------------------------------------------
# 5. Loud failures
# ---------------------------------------------------------------------------


def test_set_raw_uint8_mode_fails_without_reader():
    native = _native_uint8_chw(seed=7)
    crop_map = build_crop_feature_map({CAM: BOX})
    proxy = _make_proxy({FKEY: native}, crop_map, with_reader=False)
    with pytest.raises(AttributeError, match="return_uint8"):
        proxy.set_raw_uint8_mode(True)


def test_set_raw_uint8_mode_on_subdatasets_rejects_non_proxy():
    with pytest.raises(TypeError, match="_PerCameraCropSubset"):
        set_raw_uint8_mode_on_subdatasets([object()], True)


def test_gpu_preprocessor_rejects_bad_image_shape():
    gpu = GpuImagePreprocessor({}, TARGET_HW)
    with pytest.raises(ValueError, match=r"\(\.\.\.,3,H,W\)"):
        gpu({FKEY: torch.zeros(4, NATIVE_H, NATIVE_W, dtype=torch.uint8)})
    with pytest.raises(ValueError, match=r"\(\.\.\.,3,H,W\)"):
        gpu({FKEY: torch.zeros(NATIVE_H, NATIVE_W, dtype=torch.uint8)})


# ---------------------------------------------------------------------------
# 6. Quantized-aug tier (DP): single-source quantize formula + round-trip bound
# ---------------------------------------------------------------------------


def test_quantize_formula_byte_identical_to_pack_images_uint8():
    """The worker-side quantize and the buffer pack MUST be one formula."""
    torch.manual_seed(0)
    # Include out-of-range values to exercise the clamp identically.
    x = torch.rand(4, 3, 224, 224) * 1.2 - 0.1
    q_helper = quantize_float01_to_uint8(x.clone())
    q_buffer = VisionReplayBuffer.pack_images_uint8(x.clone())
    assert q_helper.dtype == torch.uint8
    torch.testing.assert_close(q_helper, q_buffer, rtol=0, atol=0)
    assert q_helper.numpy().tobytes() == q_buffer.numpy().tobytes()


def test_worker_quantize_gpu_refloat_round_trip_within_one_255th():
    """Fixed float 224^2 batch -> worker quantize + GPU-side refloat: every pixel
    within 1/255 of the original (the quantized-aug tier numerics bound)."""
    torch.manual_seed(1)
    x = torch.rand(8, 3, 224, 224)  # post-aug float [0,1] policy-res frames
    quantized = QuantizeUint8PostTransform()(x)
    assert quantized.dtype == torch.uint8
    refloated = uint8_to_float01(quantized)
    diff = (refloated - x).abs()
    assert float(diff.max()) <= 1.0 / 255.0, (
        f"round-trip max abs diff {float(diff.max()):.8f} exceeds 1/255"
    )


def test_quantize_post_transform_runs_inner_first():
    """With an inner transform, the wrapper quantizes the INNER's output (i.e.
    the post-aug frame), identical to quantizing that output directly."""
    native = _native_uint8_chw(seed=8)
    inner = PolicyImagePreprocessTransform(TARGET_HW)
    wrapped = QuantizeUint8PostTransform(inner)
    out = wrapped(native.clone())
    expected = quantize_float01_to_uint8(inner(native.clone()))
    assert out.dtype == torch.uint8
    assert out.shape == (3, *TARGET_HW)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_quantize_post_transform_none_inner_is_quantize_only():
    x = torch.rand(3, 32, 32)
    out = QuantizeUint8PostTransform(None)(x)
    torch.testing.assert_close(out, quantize_float01_to_uint8(x), rtol=0, atol=0)


def test_quantize_post_transform_is_picklable():
    wrapped = QuantizeUint8PostTransform(PolicyImagePreprocessTransform(TARGET_HW))
    clone = pickle.loads(pickle.dumps(wrapped))
    native = _native_uint8_chw(seed=9)
    torch.testing.assert_close(clone(native.clone()), wrapped(native.clone()), rtol=0, atol=0)


def test_quantize_and_refloat_fail_loudly_on_wrong_dtype():
    with pytest.raises(TypeError, match="float32"):
        quantize_float01_to_uint8(torch.zeros(3, 4, 4, dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8"):
        uint8_to_float01(torch.zeros(3, 4, 4, dtype=torch.float32))
    with pytest.raises(TypeError, match="callable"):
        QuantizeUint8PostTransform(inner=42)
