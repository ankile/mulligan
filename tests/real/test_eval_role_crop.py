"""Tests for ROLE camera resolution + crop coordinate-space in eval.

Policies are trained on ROLE-named camera features (``observation.images.side_1``,
``observation.images.wrist_left``) with role-keyed ``camera_crop_boxes`` defined in the
640x480 STORED space. At eval the LIVE robot camera dict is keyed by raw ZED serials
(``20000002_left`` == side_1 in the placeholder station config) at the native 1280x720
resolution.

These tests pin the behaviors of ``LeRobotRealWorldPolicy``:

- eval-load role resolution: a live serial frame is read by serial, but the
  output feature key + crop-lookup key become the ROLE.
- crop coordinate-space: the native 1280x720 frame is downscaled to 640x480
  (INTER_AREA) BEFORE the crop, so the role's box (in stored space) selects the
  intended region.
- a policy whose image features are not station roles is refused.
"""

import numpy as np
import pytest
import cv2
import torch
from torchvision.transforms import v2

from mulligan.real.robot.cameras import (
    STATION_CAMERA_DEFAULT_CROPS,
    STATION_CAMERA_KEYS_BY_ROLE,
)
from mulligan.real.policy.loader import LeRobotRealWorldPolicy, build_real_world_policy


SIDE_1_SERIAL = STATION_CAMERA_KEYS_BY_ROLE["side_1"]  # "20000002_left"
WRIST_LEFT_SERIAL = STATION_CAMERA_KEYS_BY_ROLE["wrist_left"]  # "10000001_left"

POLICY_H = 64
POLICY_W = 64
NATIVE_H = 720
NATIVE_W = 1280


class _IdentityProcessor:
    """Stand-in for the LeRobot pre/post processor: returns its argument unchanged."""

    def __call__(self, x):
        return x


class _StubPolicyConfig:
    # build_processed_obs only consults policy.config via _use_joint_positions, which
    # checks input_features for the joint-position key. An empty mapping => no joints.
    input_features = {}


class _StubPolicy:
    config = _StubPolicyConfig()


def _make_native_gradient_frame() -> np.ndarray:
    """A native 1280x720 HWC-RGB-uint8 frame with a deterministic 2D gradient.

    A gradient makes a wrong crop-coordinate-space (native vs stored) produce a
    DIFFERENT output, so the cross-resolution test is not satisfiable by chance.
    """
    ys = np.linspace(0, 255, NATIVE_H, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, NATIVE_W, dtype=np.float32)[None, :]
    r = ys + 0.0 * xs
    g = 0.0 * ys + xs
    b = 0.5 * (ys + xs)
    frame = np.stack([r, g, b], axis=-1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _role_crops_subset() -> dict[str, tuple[int, int, int, int]]:
    return {
        "side_1": STATION_CAMERA_DEFAULT_CROPS["side_1"],
        "wrist_left": STATION_CAMERA_DEFAULT_CROPS["wrist_left"],
    }


def _training_resize_hwc_to_chw_float(img: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
    # div(255) matches lerobot's video-decode float conversion (the training
    # data path); mul(1/255) would be 1 ULP off.
    tensor /= 255.0
    out = v2.Resize((POLICY_H, POLICY_W), antialias=True)(tensor).contiguous()
    return out.numpy()


def _make_role_policy() -> LeRobotRealWorldPolicy:
    return LeRobotRealWorldPolicy(
        policy=_StubPolicy(),
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        camera_keys=[SIDE_1_SERIAL, WRIST_LEFT_SERIAL],
        camera_height=POLICY_H,
        camera_width=POLICY_W,
        camera_crops=_role_crops_subset(),
    )


def test_build_processed_obs_emits_role_named_features():
    """(a) live serial frames -> ROLE-named CHW policy-resolution features."""
    policy = _make_role_policy()
    state = np.zeros(7, dtype=np.float32)
    native = _make_native_gradient_frame()
    native_rgb_images = {
        SIDE_1_SERIAL: native.copy(),
        WRIST_LEFT_SERIAL: native.copy(),
    }

    obs = policy.build_processed_obs(state, native_rgb_images)

    # The identity preprocessor returns the dict unchanged.
    assert "observation.images.side_1" in obs
    assert "observation.images.wrist_left" in obs
    # The serial-named keys must NOT appear — only the role names.
    assert f"observation.images.{SIDE_1_SERIAL}" not in obs
    assert f"observation.images.{WRIST_LEFT_SERIAL}" not in obs

    for role in ("side_1", "wrist_left"):
        tensor = obs[f"observation.images.{role}"]
        assert tuple(tensor.shape) == (3, POLICY_H, POLICY_W)
        assert tensor.dtype.is_floating_point


def test_crop_happens_in_stored_640x480_space():
    """(b) crops happen in 640x480 stored space, not native 1280x720."""
    policy = _make_role_policy()
    native = _make_native_gradient_frame()

    got = policy._crop_and_resize_native_rgb("side_1", native.copy())

    # Manual reference: downscale native -> 640x480 (INTER_AREA), crop with the
    # role's stored-space box, then use the same torchvision final resize the
    # training transform applies.
    ref_h, ref_w = 480, 640
    stored = cv2.resize(native.copy(), (ref_w, ref_h), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = STATION_CAMERA_DEFAULT_CROPS["side_1"]
    cropped = stored[y0:y1, x0:x1]
    expected = _training_resize_hwc_to_chw_float(cropped)

    np.testing.assert_array_equal(got, expected)

    # Sanity: a NATIVE-space crop would differ on a gradient frame, so this test
    # actually distinguishes the two coordinate spaces.
    native_cropped = native[y0:y1, x0:x1]
    native_expected = _training_resize_hwc_to_chw_float(native_cropped)
    assert not np.array_equal(got, native_expected)


# --- The shared build_real_world_policy helper and the role-feature guard ---------------


class _FeatShape:
    def __init__(self, shape):
        self.shape = shape


class _BuildStubConfig:
    """Minimal policy.config for build_real_world_policy: image_features, optional crop
    boxes, and an explicit action contract."""

    def __init__(self, image_feature_keys, camera_crop_boxes=None):
        self.image_features = {k: _FeatShape((3, POLICY_H, POLICY_W)) for k in image_feature_keys}
        self.input_features = {}
        self.camera_crop_boxes = camera_crop_boxes or {}
        self.dual_side_crop_boxes = {}
        self.action_target = "cartesian_velocity"
        self.cartesian_action_frame = "base"


class _BuildStubPolicy:
    def __init__(self, config):
        self.config = config

    def eval(self):
        return self


def test_build_real_world_policy_role_named_sets_crops_and_resolution():
    cfg = _BuildStubConfig(
        ["observation.images.side_1", "observation.images.wrist_left"],
        camera_crop_boxes={
            "side_1": list(STATION_CAMERA_DEFAULT_CROPS["side_1"]),
            "wrist_left": list(STATION_CAMERA_DEFAULT_CROPS["wrist_left"]),
        },
    )
    wrapped = build_real_world_policy(
        _BuildStubPolicy(cfg),
        _IdentityProcessor(),
        _IdentityProcessor(),
        camera_keys=[],
        default_camera_height=240,
        default_camera_width=320,
    )
    # resolution auto-detected from config (not the 240x320 fallback)
    assert (wrapped._camera_height, wrapped._camera_width) == (POLICY_H, POLICY_W)
    # the role-keyed crop flowed through
    assert "side_1" in wrapped._camera_crops


def test_build_real_world_policy_role_named_without_crop_contract_fails_loud():
    cfg = _BuildStubConfig(
        ["observation.images.side_1", "observation.images.wrist_left"],
        camera_crop_boxes=None,
    )
    with pytest.raises(ValueError, match="camera_crop_boxes is missing"):
        build_real_world_policy(
            _BuildStubPolicy(cfg),
            _IdentityProcessor(),
            _IdentityProcessor(),
            camera_keys=[],
            default_camera_height=240,
            default_camera_width=320,
        )


def test_build_real_world_policy_refuses_dual_side_crop_boxes():
    cfg = _BuildStubConfig(
        ["observation.images.side_1", "observation.images.wrist_left"],
        camera_crop_boxes={
            "side_1": list(STATION_CAMERA_DEFAULT_CROPS["side_1"]),
            "wrist_left": list(STATION_CAMERA_DEFAULT_CROPS["wrist_left"]),
        },
    )
    cfg.dual_side_crop_boxes = {"20000002_left": [0, 0, 100, 100]}
    with pytest.raises(NotImplementedError, match="dual-stream"):
        build_real_world_policy(
            _BuildStubPolicy(cfg),
            _IdentityProcessor(),
            _IdentityProcessor(),
            camera_keys=[],
            default_camera_height=240,
            default_camera_width=320,
        )


def test_build_real_world_policy_refuses_non_role_image_features():
    cfg = _BuildStubConfig(
        [f"observation.images.{SIDE_1_SERIAL}", f"observation.images.{WRIST_LEFT_SERIAL}"]
    )
    with pytest.raises(ValueError, match="not station camera roles"):
        build_real_world_policy(
            _BuildStubPolicy(cfg),
            _IdentityProcessor(),
            _IdentityProcessor(),
            camera_keys=[],
            default_camera_height=240,
            default_camera_width=320,
        )


def test_constructor_refuses_non_role_image_features():
    cfg = _BuildStubConfig(["observation.images.side_1", "observation.images.overhead"])
    with pytest.raises(ValueError, match=r"\['overhead'\] are not station camera roles"):
        LeRobotRealWorldPolicy(
            policy=_BuildStubPolicy(cfg),
            preprocessor=_IdentityProcessor(),
            postprocessor=_IdentityProcessor(),
            camera_keys=[],
            camera_height=POLICY_H,
            camera_width=POLICY_W,
        )
