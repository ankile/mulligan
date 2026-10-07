"""Pin the ImageNet visual-normalization override in the DP trainer.

When --visual-normalization resolves to 'imagenet' (the default for
ImageNet-pretrained encoders), every observation.images.* feature's MEAN_STD
stats must be replaced by the canonical ImageNet mean/std, while non-image
features (action/state) are untouched and the input dict is never mutated. This
sidesteps the corrupted dataset image std (uint8 overflow in LeRobot compute_stats).
"""

from __future__ import annotations

import numpy as np
import pytest

from mulligan.real.policy.visual_norm import apply_imagenet_visual_stats

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _corrupt_image_stats():
    # The pathological tiny std the uint8 overflow produces, plus a non-image key.
    return {
        "observation.images.10000001_left": {
            "mean": np.array([0.266, 0.262, 0.237], dtype=np.float32).reshape(3, 1, 1),
            "std": np.array([0.016, 0.016, 0.020], dtype=np.float32).reshape(3, 1, 1),
        },
        "observation.images.side_2": {
            "mean": np.array([0.266, 0.262, 0.237], dtype=np.float32).reshape(3, 1, 1),
            "std": np.array([0.016, 0.016, 0.020], dtype=np.float32).reshape(3, 1, 1),
        },
        "action": {
            "min": np.full((7,), -1.0, dtype=np.float32),
            "max": np.full((7,), 1.0, dtype=np.float32),
        },
    }


def test_overrides_all_image_features():
    stats = _corrupt_image_stats()
    out, image_keys = apply_imagenet_visual_stats(stats)

    assert set(image_keys) == {
        "observation.images.10000001_left",
        "observation.images.side_2",
    }
    for key in image_keys:
        np.testing.assert_allclose(out[key]["mean"], IMAGENET_MEAN, rtol=0, atol=1e-6)
        np.testing.assert_allclose(out[key]["std"], IMAGENET_STD, rtol=0, atol=1e-6)
    # no stream keeps the corrupted std
    assert not np.allclose(out["observation.images.side_2"]["std"], 0.016)


def test_non_image_features_untouched():
    stats = _corrupt_image_stats()
    out, _ = apply_imagenet_visual_stats(stats)
    np.testing.assert_array_equal(out["action"]["min"], stats["action"]["min"])
    np.testing.assert_array_equal(out["action"]["max"], stats["action"]["max"])
    assert "mean" not in out["action"]  # action stays MIN_MAX, no mean/std injected


def test_input_not_mutated():
    stats = _corrupt_image_stats()
    apply_imagenet_visual_stats(stats)
    # original corrupted std preserved on the input object
    np.testing.assert_allclose(
        stats["observation.images.10000001_left"]["std"],
        np.array([0.016, 0.016, 0.020], dtype=np.float32).reshape(3, 1, 1),
    )


def test_raises_when_no_image_features():
    with pytest.raises(ValueError, match="no observation.images"):
        apply_imagenet_visual_stats({"action": {"min": np.zeros(7), "max": np.ones(7)}})
