"""Anti-regression guard for the uint8 image-std normalization bug.

``assert_image_std_physical`` is the last line of defense against re-baking the
corrupted ~0.016 image std (uint8 ``compute_stats`` overflow) into a deployed
checkpoint. These tests prove it accepts healthy/ImageNet stats and RAISES on the
corrupted-std signature, on non-finite stats, and on a silent no-op (no image
features present).
"""

from __future__ import annotations

import numpy as np
import pytest

from mulligan.real.policy.visual_norm import (
    PHYSICAL_IMAGE_STD_HI,
    PHYSICAL_IMAGE_STD_LO,
    apply_imagenet_visual_stats,
    assert_image_std_physical,
)


def _img_stats(std):
    return {
        "observation.images.side_left": {
            "mean": np.array([[[0.5]], [[0.5]], [[0.5]]], dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
        },
        "observation.state": {  # non-image key must be ignored
            "mean": np.zeros(7, dtype=np.float32),
            "std": np.full(7, 1e-4, dtype=np.float32),
        },
    }


def test_accepts_healthy_per_channel_std():
    checked = assert_image_std_physical(_img_stats([[[0.21]], [[0.22]], [[0.23]]]))
    assert checked == ["observation.images.side_left"]


def test_corrupted_uint8_std_raises():
    # The ~0.016 collapse the bug produces.
    with pytest.raises(ValueError, match="non-physical image std"):
        assert_image_std_physical(_img_stats([[[0.016]], [[0.016]], [[0.016]]]))


def test_any_corrupted_channel_raises():
    # One bad channel is enough to fail (min over channels < LO).
    with pytest.raises(ValueError, match="non-physical image std"):
        assert_image_std_physical(_img_stats([[[0.21]], [[0.005]], [[0.22]]]))


def test_non_finite_std_raises():
    with pytest.raises(ValueError, match="non-finite"):
        assert_image_std_physical(_img_stats([[[np.nan]], [[0.22]], [[0.23]]]))


def test_no_image_features_raises_by_default():
    with pytest.raises(ValueError, match="no observation.images"):
        assert_image_std_physical({"observation.state": {"std": np.ones(7)}})


def test_imagenet_override_passes_the_guard():
    # In --visual-normalization imagenet mode the std is swapped to ImageNet, which
    # must pass; this pins that the override and the guard agree.
    corrupted = _img_stats([[[0.016]], [[0.016]], [[0.016]]])
    overridden, _ = apply_imagenet_visual_stats(corrupted)
    checked = assert_image_std_physical(overridden)
    assert "observation.images.side_left" in checked


def test_physical_bounds_are_sane():
    # ImageNet std (~0.224) must sit comfortably inside the accepted band.
    assert PHYSICAL_IMAGE_STD_LO < 0.224 < PHYSICAL_IMAGE_STD_HI
    assert PHYSICAL_IMAGE_STD_LO < 0.02 + 1e-9  # the corrupted ~0.016 is below LO
