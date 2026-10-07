"""The lerobot runtime patches kept for the release (mulligan.utils.lerobot_patches)."""

from __future__ import annotations

import pytest
import torch

from mulligan.utils.lerobot_patches import apply_all_patches

apply_all_patches()

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel  # noqa: E402

from mulligan.utils.load_pretrained import _load_lerobot_config_with_compat  # noqa: E402


def _image_config(**overrides) -> DiffusionConfig:
    kwargs = dict(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            "observation.images.side_1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.wrist_left": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 64, 64)
            ),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        horizon=8,
        n_action_steps=6,
        n_obs_steps=1,
        down_dims=(32, 64),
        crop_shape=None,
        pretrained_backbone_weights=None,
        use_separate_rgb_encoder_per_camera=True,
        do_mask_loss_for_padding=True,
        device="cpu",
    )
    kwargs.update(overrides)
    return DiffusionConfig(**kwargs)


def test_contract_fields_are_dataclass_fields():
    fields = DiffusionConfig.__dataclass_fields__
    for name in (
        "camera_crop_boxes",
        "action_target",
        "cartesian_action_frame",
        "action_mode",
        "vision_pooling",
        "stereo_pairs",
    ):
        assert name in fields, name


def test_released_config_fields_parse(tmp_path):
    """Fields found in the released real DP config.json files parse and round-trip."""
    data = {
        "type": "diffusion",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [4]},
            "observation.images.side_1": {"type": "VISUAL", "shape": [3, 224, 224]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "horizon": 12,
        "n_action_steps": 6,
        "down_dims": [512, 1024],
        "camera_crop_boxes": {"side_1": [138, 0, 580, 447]},
        "action_target": "cartesian_velocity",
        "cartesian_action_frame": "base",
        "action_mode": "relative",
        "vision_pooling": "mean",
        "attn_num_queries": 8,
        "stereo_pairs": "",
        "proprio_mode": "current",
        "use_peft": False,
    }
    cfg = _load_lerobot_config_with_compat(tmp_path, data)
    assert cfg.camera_crop_boxes == {"side_1": [138, 0, 580, 447]}
    assert cfg.action_mode == "relative"
    assert cfg.vision_pooling == "mean"


def test_resnet_image_model_builds_and_rejects_other_frontends():
    model = DiffusionModel(_image_config())
    assert len(model.rgb_encoder) == 2
    with pytest.raises(NotImplementedError, match="vision frontend"):
        DiffusionModel(_image_config(vision_pooling="attention"))
    with pytest.raises(NotImplementedError, match="vision frontend"):
        DiffusionModel(_image_config(stereo_pairs="side_1:side_2"))
    with pytest.raises(ValueError, match="ResNet"):
        _image_config(vision_backbone="dinov2-small")


class _ZeroUnet(torch.nn.Module):
    def forward(self, noisy, timesteps, global_cond):
        return torch.zeros_like(noisy)


def test_masked_loss_is_mean_over_real_actions():
    """lerobot PR #3442 semantics: padded steps are excluded from numerator and denominator."""
    torch.manual_seed(0)
    # "sample" prediction with a zero U-Net makes the loss mean(action^2) over real steps.
    model = DiffusionModel(_image_config(prediction_type="sample"))
    model.unet = _ZeroUnet()
    b = 3
    batch = {
        "observation.state": torch.randn(b, 1, 4),
        "observation.images": torch.rand(b, 1, 2, 3, 64, 64),
        "action": torch.randn(b, 8, 2),
        "action_is_pad": torch.zeros(b, 8, dtype=torch.bool),
    }
    batch["action_is_pad"][:, 6:] = True
    loss = model.compute_loss(batch)
    torch.testing.assert_close(loss, batch["action"][:, :6].square().mean())
    per_element = (-batch["action"]).square()
    valid = (~batch["action_is_pad"]).unsqueeze(-1).to(per_element.dtype).expand_as(per_element)
    assert torch.equal(loss, (per_element * valid).sum() / valid.sum().clamp_min(1e-8))


def test_unmasked_loss_is_mean_over_all_steps():
    torch.manual_seed(0)
    model = DiffusionModel(_image_config(prediction_type="sample", do_mask_loss_for_padding=False))
    model.unet = _ZeroUnet()
    b = 3
    batch = {
        "observation.state": torch.randn(b, 1, 4),
        "observation.images": torch.rand(b, 1, 2, 3, 64, 64),
        "action": torch.randn(b, 8, 2),
        "action_is_pad": torch.zeros(b, 8, dtype=torch.bool),
    }
    batch["action_is_pad"][:, 6:] = True
    loss = model.compute_loss(batch)
    per_element = (-batch["action"]).square()
    # Bit-identical to the pre-cleanup formula (sum over an all-ones mask).
    assert torch.equal(loss, per_element.sum() / torch.ones_like(per_element).sum())
