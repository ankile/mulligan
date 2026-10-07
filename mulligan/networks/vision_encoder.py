#!/usr/bin/env python3
"""
Vision encoder utilities for extracting frozen encoders from trained policies.

Extracts and freezes the RGB encoder from a trained DiffusionPolicy checkpoint,
together with the image normalization it was trained behind.

The dominant approach in real-world robot RL (SERL, RL-100, ConRFT) is to freeze
the BC policy's vision encoder and train only the value function heads on top of
cached visual features. This module supports that workflow.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load frozen encoder from a trained DiffusionPolicy checkpoint
# ---------------------------------------------------------------------------


def _extract_image_normalization(
    preprocessor, image_keys: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the per-channel image (mean, std) the DP preprocessor applies.

    The extracted ``rgb_encoder`` was trained BEHIND this normalization, so every
    consumer that feeds it float [0,1] images must apply the same affine transform
    first. Probes the normalizer step directly (T(0), T(1), T(0.5)) so the result
    exactly reproduces whatever affine transform the step applies — including the
    identity for visual_normalization=identity policies — independent of how the
    stats are stored internally.

    Returns ((3,1,1) mean, (3,1,1) std) on CPU. Raises if the normalizer step
    cannot be found or its image transform is not affine.
    """
    from lerobot.configs.types import FeatureType

    step = None
    steps = getattr(preprocessor, "steps", None)
    if steps is not None:
        for s in steps:
            if hasattr(s, "_apply_transform") and hasattr(s, "_tensor_stats"):
                step = s
                break
    elif hasattr(preprocessor, "_apply_transform") and hasattr(preprocessor, "_tensor_stats"):
        step = preprocessor
    if step is None:
        raise ValueError(
            "Could not locate the normalizer step in the DP preprocessor "
            f"({type(preprocessor).__name__}); cannot mirror the image normalization "
            "the frozen encoder was trained behind."
        )

    per_key: list[tuple[torch.Tensor, torch.Tensor]] = []
    for key in image_keys:
        zeros = torch.zeros(1, 3, 2, 2)
        ones = torch.ones(1, 3, 2, 2)
        half = torch.full((1, 3, 2, 2), 0.5)
        t0 = step._apply_transform(zeros, key, FeatureType.VISUAL, inverse=False)
        t1 = step._apply_transform(ones, key, FeatureType.VISUAL, inverse=False)
        th = step._apply_transform(half, key, FeatureType.VISUAL, inverse=False)
        tz = t0[0, :, 0, 0]
        to = t1[0, :, 0, 0]
        std = 1.0 / (to - tz)
        mean = -tz * std
        # Affinity self-check: the recovered (mean, std) must reproduce T(0.5).
        want = (0.5 - mean) / std
        if not torch.allclose(th[0, :, 0, 0], want, atol=1e-5):
            raise ValueError(
                f"DP image normalization for {key!r} is not a per-channel affine "
                "transform; cannot mirror it with (x - mean) / std."
            )
        per_key.append((mean.reshape(3, 1, 1), std.reshape(3, 1, 1)))

    mean0, std0 = per_key[0]
    for key, (m, s) in zip(image_keys[1:], per_key[1:], strict=True):
        if not (torch.allclose(m, mean0, atol=1e-6) and torch.allclose(s, std0, atol=1e-6)):
            raise ValueError(
                "Per-camera image normalization stats differ across image keys "
                f"({image_keys[0]!r} vs {key!r}); the shared-tensor IQL encode path "
                "assumes one transform for all cameras."
            )
    return mean0.detach().cpu(), std0.detach().cpu()


def load_frozen_encoder_from_dp(
    artifact_or_path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Extract and freeze the vision encoder from a trained DiffusionPolicy.

    The encoder is the `policy.diffusion.rgb_encoder` module, which is a
    ResNet-18 + SpatialSoftmax(32kp) + Linear(64->64) + ReLU pipeline
    producing a 64D feature vector per camera view.

    Args:
        artifact_or_path: A local checkpoint directory, an ``hf://`` URI or a W&B
            artifact (see :func:`mulligan.release.hub.resolve_checkpoint`).
        device: Device to load the encoder to.

    Returns:
        (encoder, metadata) where:
        - encoder: Frozen nn.Module ready for inference
        - metadata: Dict with encoder configuration info
    """
    from mulligan.utils.load_pretrained import load_policy

    logger.info(f"Loading DiffusionPolicy from {artifact_or_path}")
    policy, preprocessor, _ = load_policy(artifact_or_path, device=device, strict=False)

    # Extract the RGB encoder from the DiffusionPolicy
    # Path: DiffusionPolicy -> DiffusionModel -> rgb_encoder
    if not hasattr(policy, "diffusion"):
        raise ValueError(
            f"Expected a DiffusionPolicy with a .diffusion attribute, "
            f"got {type(policy).__name__}. Make sure the artifact contains "
            f"a DiffusionPolicy checkpoint."
        )

    diffusion_model = policy.diffusion
    if not hasattr(diffusion_model, "rgb_encoder"):
        raise ValueError(
            "DiffusionModel does not have an rgb_encoder attribute. "
            "This policy may be state-only (no image features)."
        )

    encoder = diffusion_model.rgb_encoder

    # Extract camera key ordering from the policy config
    # This determines which encoder index corresponds to which camera
    camera_key_order = list(policy.config.image_features.keys())

    # Handle both single encoder and ModuleList (per-camera encoders)
    if isinstance(encoder, nn.ModuleList):
        logger.info(f"Found {len(encoder)} per-camera encoders (ModuleList)")
        logger.info(f"Camera key order: {camera_key_order}")
        feature_dim = encoder[0].feature_dim
        separate_encoders = True
        # Convert to ModuleDict keyed by raw camera name (no dots — nn.ModuleDict
        # forbids "." in keys, so strip the "observation.images." prefix)
        encoder = nn.ModuleDict(
            {
                key.removeprefix("observation.images."): enc
                for key, enc in zip(camera_key_order, encoder, strict=True)
            }
        )
    else:
        logger.info("Found shared encoder (single DiffusionRgbEncoder)")
        feature_dim = encoder.feature_dim
        separate_encoders = False

    # Freeze all parameters
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Move to target device
    encoder = encoder.to(device)

    # Mirror the image normalization the encoder was trained behind. The DP's
    # preprocessor normalizes images (ImageNet mean/std; older policies used dataset
    # stats) BEFORE rgb_encoder — feeding raw [0,1] frames to the
    # extracted encoder silently shifts its input distribution.
    image_norm_mean, image_norm_std = _extract_image_normalization(preprocessor, camera_key_order)
    logger.info(
        "Image normalization mirrored from DP preprocessor: "
        f"mean={image_norm_mean.flatten().tolist()}, std={image_norm_std.flatten().tolist()}"
    )

    metadata = {
        "encoder_type": "dp_frozen",
        "feature_dim": feature_dim,
        "separate_encoders": separate_encoders,
        "camera_key_order": camera_key_order,
        "image_norm_mean": image_norm_mean,
        "image_norm_std": image_norm_std,
        "camera_crop_boxes": getattr(policy.config, "camera_crop_boxes", {}) or {},
        "dual_side_crop_boxes": getattr(policy.config, "dual_side_crop_boxes", {}) or {},
        "action_target": getattr(policy.config, "action_target", None) or "cartesian_velocity",
        "cartesian_action_frame": getattr(policy.config, "cartesian_action_frame", None) or "base",
        "source": str(artifact_or_path),
    }

    logger.info(
        f"Frozen encoder loaded: feature_dim={feature_dim}, "
        f"params={sum(p.numel() for p in encoder.parameters()):,}"
    )

    return encoder, metadata
