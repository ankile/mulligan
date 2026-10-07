"""End-to-end regression guards for the real-policy image normalization pipeline.

A normalization bug (below) would corrupt every real DP/IQL checkpoint. These tests
drive the SAME LeRobot objects the train and eval paths use (``get_feature_stats`` -> dataset stats
-> ``NormalizerProcessorStep``) and
assert the resulting normalizer is sane, against the uint8 fix that lerobot >=0.5.2
ships natively.

Bug 1 — uint8 image-stats overflow (fixed natively in lerobot >=0.5.2):
    LeRobot ``RunningQuantileStats.update`` computes ``np.mean(batch**2)`` on a
    uint8 image batch; 255**2 wraps mod 256, collapsing the per-channel image std
    to a corrupted-tiny value (~0 / ~0.016 in [0,1]) while the mean stays sane.
    Under VISUAL=MEAN_STD that bakes a ~10-40x high-gain, shift-fragile image
    normalizer into the checkpoint: normalized features blow up and a tiny
    lighting shift is amplified by >10x.

These run on tiny synthetic batches (no dataset / GPU) and exercise the real
``NormalizerProcessorStep`` that ``make_pre_post_processors`` installs in both the
train preprocessor and the eval-reconstructed preprocessor.
"""

from __future__ import annotations


import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.processor.normalize_processor import NormalizerProcessorStep


CAM_KEY = "observation.images.10000001_left"

# Canonical ImageNet stats the fix installs (must match lerobot IMAGENET_STATS).
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# --------------------------------------------------------------------------- #
# Pipeline primitives — the real train-stats path and the real eval normalizer.
# --------------------------------------------------------------------------- #
def _realistic_uint8_batch(seed: int = 0, n: int = 64, hw: int = 16) -> np.ndarray:
    """A natural-image-like uint8 batch (N,3,H,W): per-channel mean ~115, std ~58
    in [0,255] -> a physical [0,1] std ~0.22 (well away from both the corrupted
    ~0/0.016 and the saturated extremes)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(115.0, 58.0, size=(n, 3, hw, hw))
    return np.clip(img, 0, 255).astype(np.uint8)


def _image_stats_from_uint8(batch_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The real LeRobot per-channel image-stats computation (the train pipeline).

    Mirrors ``compute_episode_stats``' image branch exactly: ``get_feature_stats``
    with ``axis=(0,2,3)`` (which drives ``RunningQuantileStats.update`` — the
    patched code) then the ``/255`` normalization to [0,1]. Returns (mean, std)
    each shaped (3,1,1).
    """
    s = get_feature_stats(batch_u8, axis=(0, 2, 3), keepdims=True)
    mean = (s["mean"] / 255.0).reshape(3, 1, 1).astype(np.float32)
    std = (s["std"] / 255.0).reshape(3, 1, 1).astype(np.float32)
    return mean, std


def _mean_std_normalizer(stats: dict, keys: list[str]) -> NormalizerProcessorStep:
    """The exact normalizer step ``make_pre_post_processors`` installs for a
    diffusion policy (VISUAL=MEAN_STD), restricted to the given image keys."""
    feats = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)) for k in keys}
    sub = {k: {sk: np.asarray(sv, dtype=np.float32) for sk, sv in stats[k].items()} for k in keys}
    return NormalizerProcessorStep(
        features=feats,
        norm_map={FeatureType.VISUAL: NormalizationMode.MEAN_STD},
        stats=sub,
    )


def _normalize_image(step: NormalizerProcessorStep, key: str, img01: np.ndarray) -> torch.Tensor:
    """Push one [0,1] CHW image through the normalizer (the eval/test image path)."""
    out = step._normalize_observation(
        {key: torch.as_tensor(img01, dtype=torch.float32)}, inverse=False
    )
    return out[key]


def _eval_frame(seed: int = 7) -> np.ndarray:
    """A realistic [0,1] eval frame from the same distribution as the train batch."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(115.0, 58.0, size=(3, 16, 16)) / 255.0, 0, 1).astype(np.float32)


# =========================================================================== #
# Bug 1 — uint8 overflow: train-side stats
# =========================================================================== #
def test_train_image_std_is_physical_with_fix():
    batch = _realistic_uint8_batch()
    _, std = _image_stats_from_uint8(batch)  # native uint8-safe stats (lerobot >=0.5.2)
    flat = std.reshape(-1)
    assert np.all(flat > 0.1) and np.all(flat < 0.5), flat
    true_std = batch.astype(np.float64).std(axis=(0, 2, 3)) / 255.0
    np.testing.assert_allclose(flat, true_std, rtol=0.05)


# =========================================================================== #
# Bug 1 — uint8 overflow: eval-side normalizer magnitude (end-to-end)
# =========================================================================== #
def _norm_magnitude() -> tuple[float, float]:
    batch = _realistic_uint8_batch()
    mean, std = _image_stats_from_uint8(batch)
    step = _mean_std_normalizer({CAM_KEY: {"mean": mean, "std": std}}, [CAM_KEY])
    z = _normalize_image(step, CAM_KEY, _eval_frame())
    return float(z.abs().mean()), float(z.abs().max())


def test_eval_normalizer_magnitude_bounded_with_fix():
    mean_abs, max_abs = _norm_magnitude()
    # z = (x - mean)/std with std ~0.22 and x spread ~0.22 -> O(1) features.
    assert mean_abs < 5.0 and max_abs < 20.0, (mean_abs, max_abs)


# =========================================================================== #
# Bug 1 — uint8 overflow: brightness distribution-shift amplification
# (the manual pre-robot sanity check, made into a guard)
# =========================================================================== #
def _brightness_delta(dbright: float = 0.06) -> float:
    batch = _realistic_uint8_batch()
    mean, std = _image_stats_from_uint8(batch)
    step = _mean_std_normalizer({CAM_KEY: {"mean": mean, "std": std}}, [CAM_KEY])
    img = _eval_frame(seed=11)
    z0 = _normalize_image(step, CAM_KEY, img)
    z1 = _normalize_image(step, CAM_KEY, np.clip(img + dbright, 0, 1).astype(np.float32))
    return float((z1 - z0).abs().mean())


def test_brightness_shift_amplification_bounded_with_fix():
    # dz = dbright/std ~ 0.06/0.22 ~ 0.27; comfortably < 1.
    assert _brightness_delta() < 1.0
