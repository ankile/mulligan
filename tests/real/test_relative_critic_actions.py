"""Pins for the UMI-relative critic action path (--action-mode relative IQL).

Three contracts:
1. ``command_pose_chunk_to_relative_torch`` (the trainer's batch-time transform)
   matches the convention-pinned numpy ``relativize_pose`` path bit-to-tolerance.
2. Train↔deploy representation parity: the physical relative chunk the trainer
   feeds the critic equals the deploy-side per-timestep MIN_MAX un-normalization
   of the DP's normalized chunk (what vision_idql_policy scores), and the shared
   ``decode_normalized_relative_chunk`` composes that same chunk back to the
   absolute command poses.
3. The Vision-IDQL DP/critic action-mode pairing contract fails loud on a
   mismatch in either direction.
"""

import numpy as np
import pytest
import torch

from mulligan.real.policy.relative_pose import (
    POSE_DIM,
    command_pose_chunk_to_relative_torch,
    decode_normalized_relative_chunk,
    relativize_pose,
)
from mulligan.real.policy.rotation6d import euler_to_r6


def _random_pose7(rng: np.random.Generator, *shape: int) -> np.ndarray:
    """Random [xyz (m), rpy (rad), grip (0/1)] rows."""
    xyz = rng.uniform(-0.6, 0.6, size=(*shape, 3))
    rpy = rng.uniform(-2.8, 2.8, size=(*shape, 3))
    grip = rng.integers(0, 2, size=(*shape, 1)).astype(np.float64)
    return np.concatenate([xyz, rpy, grip], axis=-1)


def _pose7_to_pose10(pose7: np.ndarray) -> np.ndarray:
    r6 = np.asarray(euler_to_r6(pose7[..., 3:6]), dtype=np.float64)
    return np.concatenate([pose7[..., 0:3], r6, pose7[..., 6:7]], axis=-1)


def test_command_pose_chunk_matches_numpy_relativize():
    rng = np.random.default_rng(0)
    B, H = 5, 6
    chunk7 = _random_pose7(rng, B, H)  # (B, H, 7) command poses
    state = np.concatenate(
        [_random_pose7(rng, B)[:, :6], rng.uniform(0, 1, size=(B, 1))], axis=-1
    )  # (B, 7) proprio [cart_pos(6), grip]

    got = command_pose_chunk_to_relative_torch(
        torch.as_tensor(chunk7, dtype=torch.float32),
        torch.as_tensor(state, dtype=torch.float32),
    ).numpy()

    abs10 = _pose7_to_pose10(chunk7)  # (B, H, 10)
    base10 = _pose7_to_pose10(
        np.concatenate([state[:, :6], np.zeros((B, 1))], axis=-1)
    )  # gripper slot unused by the anchor
    want = relativize_pose(abs10, base10[:, None, :])  # (B, H, 10)

    assert got.shape == (B, H, POSE_DIM)
    np.testing.assert_allclose(got, want, atol=2e-5)


def test_anchor_equal_command_relativizes_to_identity():
    rng = np.random.default_rng(1)
    B = 4
    pose7 = _random_pose7(rng, B)  # anchor == command
    chunk = torch.as_tensor(pose7[:, None, :], dtype=torch.float32)  # (B, 1, 7)
    state = torch.as_tensor(
        np.concatenate([pose7[:, :6], rng.uniform(0, 1, size=(B, 1))], axis=-1),
        dtype=torch.float32,
    )
    rel = command_pose_chunk_to_relative_torch(chunk, state).numpy()[:, 0]
    np.testing.assert_allclose(rel[:, 0:3], 0.0, atol=1e-5)  # trans → 0
    identity_r6 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    np.testing.assert_allclose(rel[:, 3:9], np.tile(identity_r6, (B, 1)), atol=1e-5)
    np.testing.assert_allclose(rel[:, 9], pose7[:, 6], atol=1e-6)  # grip passthrough


def test_train_deploy_representation_parity_and_decode_roundtrip():
    """The critic's training-time physical relative chunk == the deploy-side
    un-normalization of the DP's normalized chunk, and the shared decode
    recovers the absolute command poses from it."""
    rng = np.random.default_rng(2)
    H = 6
    chunk7 = _random_pose7(rng, 1, H)[0]  # (H, 7) command poses
    anchor7 = _random_pose7(rng, 1)[0]  # (7,) proprio

    phys_rel = command_pose_chunk_to_relative_torch(
        torch.as_tensor(chunk7[None], dtype=torch.float32),
        torch.as_tensor(anchor7[None], dtype=torch.float32),
    ).numpy()[0]  # (H, 10) — what the critic trains on

    # Synthetic per-timestep MIN_MAX bounds enclosing the chunk (the DP's (T,10)
    # action stats), plus margin so normalization stays strictly inside (-1, 1).
    a_min = phys_rel - rng.uniform(0.05, 0.2, size=phys_rel.shape)
    a_max = phys_rel + rng.uniform(0.05, 0.2, size=phys_rel.shape)
    chunk_norm = (phys_rel - a_min) / (a_max - a_min) * 2.0 - 1.0  # what the DP emits

    # Deploy scoring path (vision_idql_policy): (norm+1)/2*(mx-mn)+mn.
    deploy_phys = (chunk_norm + 1.0) / 2.0 * (a_max - a_min) + a_min
    np.testing.assert_allclose(deploy_phys, phys_rel, atol=1e-6)

    # Deploy execution path: decode back to absolute 7D euler poses == command chunk.
    poses7 = decode_normalized_relative_chunk(chunk_norm, anchor7[:6], a_min, a_max)
    np.testing.assert_allclose(poses7[:, 0:3], chunk7[:, 0:3], atol=2e-5)  # xyz
    np.testing.assert_allclose(poses7[:, 6], chunk7[:, 6], atol=1e-5)  # grip
    # Euler triples are not unique per rotation; compare the ROTATIONS (via the
    # convention-pinned euler->r6 map) rather than the angle components.
    np.testing.assert_allclose(
        np.asarray(euler_to_r6(poses7[:, 3:6]), dtype=np.float64),
        np.asarray(euler_to_r6(chunk7[:, 3:6]), dtype=np.float64),
        atol=2e-4,
    )


def test_trainer_relativize_hook_state_layouts():
    """The trainer's _relativize_action_chunk accepts both the flat-cache
    (B, 2, D) [current, successor] state stack and a bare (B, D) state, anchoring
    on the current step either way."""
    from mulligan.real.train.critic import _relativize_action_chunk

    torch.manual_seed(0)
    B, H = 3, 6
    chunk = torch.randn(B, H, 7)
    state2 = torch.randn(B, 2, 7)
    out2 = _relativize_action_chunk(chunk, state2)
    out1 = _relativize_action_chunk(chunk, state2[:, 0, :])
    assert out2.shape == (B, H, POSE_DIM)
    assert torch.allclose(out2, out1)
    with pytest.raises(ValueError, match="observation.state"):
        _relativize_action_chunk(chunk, torch.randn(B, 2, 2, 7))


def test_command_pose_chunk_validation_errors():
    ok_chunk = torch.zeros(2, 3, 7)
    ok_state = torch.zeros(2, 7)
    with pytest.raises(ValueError, match=r"\(B, H, 7\)"):
        command_pose_chunk_to_relative_torch(torch.zeros(2, 3, 10), ok_state)
    with pytest.raises(ValueError, match="anchor_state"):
        command_pose_chunk_to_relative_torch(ok_chunk, torch.zeros(2, 3, 7))
    with pytest.raises(ValueError, match="batch mismatch"):
        command_pose_chunk_to_relative_torch(ok_chunk, torch.zeros(3, 7))


class _DPConfigStub:
    action_target = "cartesian_velocity"
    cartesian_action_frame = "base"
    dual_side_crop_boxes: dict = {}

    def __init__(self, action_mode: str | None):
        if action_mode is not None:
            self.action_mode = action_mode


@pytest.mark.parametrize(
    ("dp_mode", "critic_mode", "ok"),
    [
        (None, "absolute", True),  # velocity DP + absolute critic
        ("relative", "relative", True),  # UMI-relative pairing
        ("relative", "absolute", False),  # velocity critic cannot rank rel chunks
        (None, "relative", False),  # relative critic cannot rank velocity chunks
    ],
)
def test_vision_idql_action_mode_pairing(dp_mode, critic_mode, ok):
    from mulligan.real.policy.vision_idql import (
        assert_vision_idql_supported_dp_policy_contract,
    )

    cfg = _DPConfigStub(dp_mode)
    if ok:
        assert_vision_idql_supported_dp_policy_contract(cfg, critic_action_mode=critic_mode)
    else:
        with pytest.raises(NotImplementedError, match="action-representation mismatch"):
            assert_vision_idql_supported_dp_policy_contract(cfg, critic_action_mode=critic_mode)
