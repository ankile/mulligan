"""Tests for real Vision-IQL replay-buffer sampling."""

from __future__ import annotations

import torch

from mulligan.data.vision_replay_buffer import VisionReplayBuffer


def _make_buffer(capacity: int = 8, reward_horizon_size: int | None = None) -> VisionReplayBuffer:
    return VisionReplayBuffer(
        capacity=capacity,
        camera_keys=["observation.images.cam_left"],
        n_image_timestamps=2,
        img_h=4,
        img_w=4,
        action_chunk_size=2,
        action_dim=3,
        state_dim=5,
        n_state_timestamps=2,
        reward_horizon_size=reward_horizon_size,
    )


def _batch(success: list[int]) -> dict[str, torch.Tensor]:
    n = len(success)
    return {
        "observation.images.cam_left": torch.zeros(n, 2, 3, 4, 4, dtype=torch.float32),
        "action": torch.zeros(n, 2, 3),
        "reward": torch.zeros(n, 2),
        "done": torch.zeros(n, 2),
        "observation.state": torch.zeros(n, 2, 5),
        "success": torch.tensor(success, dtype=torch.long),
        "source": torch.ones(n, dtype=torch.long),
        "episode_index": torch.arange(n, dtype=torch.long),
    }


def test_reward_horizon_can_exceed_ranked_action_chunk() -> None:
    buffer = _make_buffer(reward_horizon_size=6)
    batch = _batch([1, 0, 1])
    batch["reward"] = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    batch["done"] = torch.zeros(3, 6)
    buffer.refresh(batch)

    sampled = buffer._build_batch_from_indices(  # noqa: SLF001
        torch.arange(3),
        image_format="float32",
    )
    assert sampled["action"].shape == (3, 2, 3)
    assert sampled["reward"].shape == (3, 6)
    assert sampled["done"].shape == (3, 6)
