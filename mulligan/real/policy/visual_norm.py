"""Visual (image) normalization helpers for real-world policies.

Kept separate from :mod:`mulligan.real.train.policy` so the pure stats logic is
importable (and unit-testable) without that module's heavy top-level imports
(torch, wandb, ...).
"""

from __future__ import annotations

import copy

import numpy as np


def apply_imagenet_visual_stats(dataset_stats: dict) -> tuple[dict, list[str]]:
    """Override every ``observation.images.*`` feature's MEAN_STD stats with the
    canonical ImageNet mean/std. Returns ``(new_stats, image_keys)``; the input is
    deep-copied, never mutated.

    ImageNet-pretrained ResNet encoders expect ImageNet input statistics, not
    per-dataset stats. Using ImageNet stats also sidesteps the corrupted dataset
    image std (uint8 overflow in LeRobot ``compute_stats`` -> tiny std -> high-gain,
    shift-fragile normalization): every image stream gets the SAME canonical stats.
    VISUAL stays MEAN_STD; only the mean/std are swapped. Mirrors LeRobot's own
    ``use_imagenet_stats`` path (``datasets/factory.py``).
    """
    from lerobot.utils.constants import IMAGENET_STATS  # (c,1,1) lists; no wandb dep

    stats = copy.deepcopy(dataset_stats)
    imagenet = {k: np.asarray(v, dtype=np.float32) for k, v in IMAGENET_STATS.items()}
    image_keys = [k for k in stats if k.startswith("observation.images.")]
    if not image_keys:
        raise ValueError(
            "visual-normalization=imagenet but no observation.images.* features have "
            f"dataset stats to override; available keys: {sorted(stats)}"
        )
    for key in image_keys:
        for stat_type, value in imagenet.items():  # mean, std
            stats[key][stat_type] = value.copy()
    return stats, image_keys


# Physical bounds for a healthy per-channel image std in the normalized [0,1]
# domain. The corrupted uint8-overflow std collapses to ~0.016 (well below LO);
# a healthy natural-image std sits ~0.2-0.3 and ImageNet std is ~0.22-0.23. HI
# guards the opposite degeneracy (near-constant or mis-scaled stats).
PHYSICAL_IMAGE_STD_LO = 0.02
PHYSICAL_IMAGE_STD_HI = 0.6


def assert_image_std_physical(
    dataset_stats: dict,
    *,
    lo: float = PHYSICAL_IMAGE_STD_LO,
    hi: float = PHYSICAL_IMAGE_STD_HI,
    require_image_keys: bool = True,
) -> list[str]:
    """Fail loudly if any ``observation.images.*`` std is non-physical.

    This is the last-line guard against re-baking the uint8 ``compute_stats``
    overflow bug (corrupted std ~0.016 -> high-gain, shift-fragile normalizer)
    into a deployed checkpoint. Run it on the FINAL stats
    handed to the normalizer (i.e. AFTER any ImageNet override), so it both catches a corrupted
    per-dataset std (``--visual-normalization dataset``) and confirms the ImageNet
    override actually took (``--visual-normalization imagenet``).

    Returns the list of checked image keys. Raises ``ValueError`` naming the
    offending key and its min/max std if any channel falls outside ``(lo, hi)`` or
    is non-finite. With ``require_image_keys`` (default), also raises when there
    are no ``observation.images.*`` features to check (a silent no-op would defeat
    the guard).
    """
    image_keys = [k for k in dataset_stats if k.startswith("observation.images.")]
    if require_image_keys and not image_keys:
        raise ValueError(
            "assert_image_std_physical: no observation.images.* features in stats; "
            f"available keys: {sorted(dataset_stats)}"
        )
    for key in image_keys:
        feature_stats = dataset_stats[key]
        if "std" not in feature_stats:
            raise ValueError(f"{key}: stats missing 'std' (have {sorted(feature_stats)})")
        std = np.asarray(feature_stats["std"], dtype=np.float64)
        if not np.all(np.isfinite(std)):
            raise ValueError(f"{key}: image std has non-finite values: {std.ravel().tolist()}")
        smin = float(std.min())
        smax = float(std.max())
        if smin <= lo or smax >= hi:
            raise ValueError(
                f"{key}: non-physical image std (min={smin:.5f}, max={smax:.5f}); expected "
                f"every channel in ({lo}, {hi}). A std near ~0.016 is the uint8 "
                "compute_stats overflow bug -- recompute stats with "
                "apply_compute_stats_uint8_patch applied, or use "
                "--visual-normalization imagenet."
            )
    return image_keys
