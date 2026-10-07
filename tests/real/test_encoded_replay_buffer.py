"""Contracts for the frozen-encoder multi-view replay cache."""

import pytest
import torch

from mulligan.data.vision_replay_buffer import VisionReplayBuffer
from mulligan.real.train.critic import (
    augment_iql_images,
    iql_augmentation_spec,
    load_encoded_eval_cache,
    module_state_sha256,
    prepare_cached_iql_images,
    replay_pixel_roundtrip,
    save_encoded_eval_cache,
)


def test_cached_augmented_pixels_match_online_uint8_replay_roundtrip() -> None:
    images = torch.tensor([0.0, 0.1, 0.499, 0.501, 0.9, 1.0], dtype=torch.float32)

    actual = replay_pixel_roundtrip(images)

    expected = (images.clamp(0, 1) * 255).to(torch.uint8).float() / 255
    assert torch.equal(actual, expected)


def test_clean_cached_images_always_use_replay_pixel_contract() -> None:
    images = torch.tensor([0.0, 0.1, 0.499, 0.501, 0.9, 1.0], dtype=torch.float32)

    actual = prepare_cached_iql_images(images)

    assert torch.equal(actual, replay_pixel_roundtrip(images))


def test_clean_cache_and_online_buffer_produce_identical_frozen_features() -> None:
    torch.manual_seed(3)
    images = torch.rand(4, 3, 8, 8)
    online_pixels = VisionReplayBuffer.pack_images_uint8(images).float().div_(255.0)
    cached_pixels = prepare_cached_iql_images(images)
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 11))
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    torch.testing.assert_close(cached_pixels, online_pixels, rtol=0, atol=0)
    torch.testing.assert_close(
        encoder(cached_pixels),
        encoder(online_pixels),
        rtol=0,
        atol=0,
    )


def test_shared_iql_augmentation_is_seed_reproducible() -> None:
    images = replay_pixel_roundtrip(torch.rand(4, 3, 16, 16))
    blur_kernel = (
        torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        .div_(16.0)
        .view(1, 1, 3, 3)
        .expand(3, 1, 3, 3)
    )
    torch.manual_seed(17)
    first = augment_iql_images(images, blur_kernel=blur_kernel, shift_frac=0.05)
    torch.manual_seed(17)
    second = augment_iql_images(images, blur_kernel=blur_kernel, shift_frac=0.05)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()
    assert first.min() >= 0
    assert first.max() <= 1


def test_augmentation_spec_records_every_nonrandom_transform_parameter() -> None:
    spec = iql_augmentation_spec(shift_frac=0.075)

    assert spec == {
        "version": 1,
        "brightness_scale": 0.1,
        "contrast_scale": 0.2,
        "saturation_scale": 0.5,
        "sharpness_range": [0.0, 2.0],
        "rotation_degrees": 5.0,
        "translation_fraction": 0.075,
        "blur_kernel": [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        "blur_kernel_divisor": 16.0,
        "affine_align_corners": False,
        "affine_padding_mode": "zeros",
    }


def test_module_state_hash_is_deterministic_and_weight_sensitive() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    ).to(dtype=torch.bfloat16)
    first = module_state_sha256(module)
    second = module_state_sha256(module)

    assert first == second
    assert len(first) == 64
    with torch.no_grad():
        module[0].weight[0, 0] += 1
    assert module_state_sha256(module) != first


def test_encoded_eval_cache_roundtrip_and_strict_metadata(tmp_path) -> None:
    path = tmp_path / "eval-cache.pt"
    states = torch.randn(5, 11)
    holdout_data = {
        "actions": torch.randn(5, 42),
        "success": torch.tensor([1, 0, 1, 0, 1]),
        "source": torch.zeros(5, dtype=torch.long),
        "done": torch.tensor([False, False, False, False, True]),
        "is_valid": torch.ones(5, dtype=torch.bool),
        "frame_indices": torch.arange(5),
        "episode_indices": torch.arange(5),
        "dataset_indices": torch.zeros(5, dtype=torch.long),
        "original_frame_indices": torch.arange(5),
    }
    metadata = {
        "encoder": "artifact:v0",
        "holdout_hash": "abc",
        "n_holdout": 5,
        "state_dim": 11,
        "action_flat_dim": 42,
    }

    save_encoded_eval_cache(
        path,
        states=states,
        holdout_data=holdout_data,
        metadata=metadata,
    )
    loaded = load_encoded_eval_cache(path, metadata)

    torch.testing.assert_close(loaded["states"], states)
    torch.testing.assert_close(loaded["holdout_data"]["actions"], holdout_data["actions"])
    assert path.with_suffix(".pt.json").is_file()
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_encoded_eval_cache(path, {**metadata, "holdout_hash": "different"})
