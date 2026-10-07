from types import SimpleNamespace

import pytest

from mulligan.real.policy.vision_idql import (
    assert_vision_idql_supported_dp_policy_contract,
)


def test_vision_idql_accepts_velocity_base_uncropped_dp_contract() -> None:
    assert_vision_idql_supported_dp_policy_contract(
        SimpleNamespace(
            action_target="cartesian_velocity",
            cartesian_action_frame="base",
            camera_crop_boxes={},
            dual_side_crop_boxes={},
        )
    )


def test_vision_idql_accepts_per_camera_crop_dp_contract() -> None:
    # Per-camera replacement crops (camera_crop_boxes) ARE supported: both the
    # IQL training data path and the rollout wrapper reconstruct the DP actor's
    # box and apply the same crop-then-resize, so a cropped single-stream DP
    # actor is a valid frozen encoder. Only dual_side_crop_boxes remains
    # unimplemented.
    assert_vision_idql_supported_dp_policy_contract(
        SimpleNamespace(
            action_target="cartesian_velocity",
            cartesian_action_frame="base",
            camera_crop_boxes={"10000001_left": (126, 0, 640, 357)},
            dual_side_crop_boxes={},
        )
    )


def test_vision_idql_rejects_dual_side_crop_dp_contract() -> None:
    with pytest.raises(NotImplementedError, match="dual_side_crop_boxes"):
        assert_vision_idql_supported_dp_policy_contract(
            SimpleNamespace(
                action_target="cartesian_velocity",
                cartesian_action_frame="base",
                camera_crop_boxes={},
                dual_side_crop_boxes={"10000001_left": (126, 0, 640, 357)},
            )
        )


def test_vision_idql_rejects_non_base_dp_action_contract() -> None:
    with pytest.raises(NotImplementedError, match="cartesian_action_frame"):
        assert_vision_idql_supported_dp_policy_contract(
            SimpleNamespace(
                action_target="cartesian_velocity",
                cartesian_action_frame="eef",
                camera_crop_boxes={},
                dual_side_crop_boxes={},
            )
        )
