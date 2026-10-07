"""Pin the per-camera replacement-crop train/eval consistency invariant.

The correctness point of the replacement crop (mulligan/real/policy/side_crop.py) is
that the training data path and the eval inference path produce the *same*
cropped-then-resized tensor for a cropped camera. The train path slices CHW
tensors with the same native box the eval path slices from HWC numpy frames.
These tests pin:

  1. policy config serialization round-trips the box exactly.
  2. The crop is applied to the NATIVE frame and then resized to the policy
     resolution (so the resulting tensor shape is the policy shape, NOT a
     smaller native-crop shape) — i.e. the crop is *pre-resize*, matching the
     unchanged ``input_feature.shape``.
  3. Train-side crop+resize and eval-side crop+resize produce identical float
     tensors.
"""

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

# Applies the Mulligan policy I/O contract fields (camera_crop_boxes / action_target /
# ...) onto lerobot's PreTrainedConfig, the same import-side-effect production
# eval uses. The config save/load round-trip below depends on these being real
# dataclass fields (carried as a runtime patch, not a lerobot fork).
import mulligan.real.policy.lerobot_patches  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

from mulligan.real.policy.side_crop import (
    build_crop_feature_map,
    crop_box_for_chw_tensor,
    install_per_camera_crop,
    merge_default_crops,
    normalize_crop_map,
    parse_side_crop,
)
from mulligan.real.policy.image_preprocess import (
    preprocess_chw_tensor_for_policy,
    preprocess_hwc_rgb_uint8_for_policy,
)

CAM = "20000002_left"
BOX = (360, 120, 540, 330)  # x0, y0, x1, y1 -> 180w x 210h on a 480x640 frame
WRIST_CAM = "10000001_left"
BOUNDARY_WRIST_BOX = (126, 0, 640, 357)
NATIVE_H, NATIVE_W = 480, 640


def test_parse_side_crop_roundtrip():
    crop_map = parse_side_crop([f"{CAM}=360,120,540,330"])
    assert crop_map == {CAM: BOX}


def test_parse_boundary_touching_wrist_replacement_crop_roundtrip():
    crop_map = parse_side_crop([f"{WRIST_CAM}=126,0,640,357"])
    assert crop_map == {WRIST_CAM: BOUNDARY_WRIST_BOX}


def test_parse_side_crop_rejects_inverted_box():
    with pytest.raises(ValueError):
        parse_side_crop([f"{CAM}=540,120,360,330"])  # x1 < x0
    with pytest.raises(ValueError):
        parse_side_crop([f"{CAM}=360,120,540"])  # only 3 coords
    with pytest.raises(ValueError):
        parse_side_crop([f"{CAM}-360,120,540,330"])  # missing '='
    with pytest.raises(ValueError):
        parse_side_crop([f"{CAM}=360,120,540,330", f"{CAM}=361,120,540,330"])


def test_normalize_crop_map_rejects_non_integer_config_coords():
    with pytest.raises(ValueError, match="non-integer"):
        normalize_crop_map({WRIST_CAM: [126.5, 0, 640, 357]}, context="test camera_crop_boxes")
    with pytest.raises(ValueError, match="non-integer"):
        normalize_crop_map({WRIST_CAM: [True, 0, 640, 357]}, context="test camera_crop_boxes")


def test_boundary_touching_wrist_crop_roundtrips_in_policy_config(tmp_path):
    cfg = DiffusionConfig(
        camera_crop_boxes={WRIST_CAM: BOUNDARY_WRIST_BOX},
        action_target="cartesian_velocity",
        cartesian_action_frame="base",
    )
    cfg._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.camera_crop_boxes == {WRIST_CAM: BOUNDARY_WRIST_BOX}
    assert loaded.dual_side_crop_boxes == {}
    assert loaded.action_target == "cartesian_velocity"
    assert loaded.cartesian_action_frame == "base"


def _make_frame():
    """Deterministic native frame: pixel value encodes (y, x) so we can verify
    exactly which native pixels survive the crop on both paths."""
    yy, xx = np.meshgrid(
        np.arange(NATIVE_H, dtype=np.float32),
        np.arange(NATIVE_W, dtype=np.float32),
        indexing="ij",
    )
    # 3-channel HWC: ch0 = y, ch1 = x, ch2 = y+x (all uint8-safe via modulo)
    frame_hwc = np.stack([yy % 256, xx % 256, (yy + xx) % 256], axis=-1).astype(np.uint8)
    return frame_hwc


def test_train_eval_select_identical_native_roi():
    """The train CHW slice and the eval HWC slice must pick the SAME native
    pixels for the crop box."""
    frame_hwc = _make_frame()
    x0, y0, x1, y1 = BOX

    # Eval path: slice raw HWC numpy frame (the loader's predict does img[y0:y1, x0:x1]).
    eval_crop_hwc = frame_hwc[y0:y1, x0:x1]

    # Train path: lerobot serves CHW tensors; crop_box_for_chw_tensor slices [..., y0:y1, x0:x1].
    frame_chw = torch.from_numpy(frame_hwc.transpose(2, 0, 1).copy())
    train_crop_chw = crop_box_for_chw_tensor(frame_chw, BOX)
    train_crop_hwc = train_crop_chw.numpy().transpose(1, 2, 0)

    assert eval_crop_hwc.shape == (y1 - y0, x1 - x0, 3)
    assert train_crop_hwc.shape == (y1 - y0, x1 - x0, 3)
    # Byte-identical native ROI selection.
    np.testing.assert_array_equal(train_crop_hwc, eval_crop_hwc)


def test_boundary_touching_wrist_crop_selects_identical_native_pixels():
    frame_hwc = _make_frame()
    x0, y0, x1, y1 = BOUNDARY_WRIST_BOX

    eval_crop_hwc = frame_hwc[y0:y1, x0:x1]
    frame_chw = torch.from_numpy(frame_hwc.transpose(2, 0, 1).copy())
    train_crop_hwc = (
        crop_box_for_chw_tensor(frame_chw, BOUNDARY_WRIST_BOX).numpy().transpose(1, 2, 0)
    )

    assert eval_crop_hwc.shape == (y1 - y0, x1 - x0, 3)
    assert eval_crop_hwc.shape == (357, 514, 3)
    assert x1 == NATIVE_W
    assert y0 == 0
    np.testing.assert_array_equal(train_crop_hwc, eval_crop_hwc)


def test_crop_then_resize_yields_policy_shape():
    """Crop is PRE-resize: native ROI (210x180) resizes to the policy 224x224,
    so the feature shape is unchanged and the marker barrel is magnified."""
    frame_hwc = _make_frame()
    x0, y0, x1, y1 = BOX
    pol_h, pol_w = 224, 224

    eval_resized = preprocess_hwc_rgb_uint8_for_policy(
        frame_hwc, target_hw=(pol_h, pol_w), crop_box=BOX
    )
    assert eval_resized.shape == (3, pol_h, pol_w)

    # The native ROI is 180px wide; resized to 224px wide => >1x magnification of
    # the holder region (vs the 0.35x squash a full 640->224 resize applies).
    native_roi_w = x1 - x0
    assert pol_w / native_roi_w > 1.0


def test_train_eval_crop_resize_outputs_identical_tensor():
    frame_hwc = _make_frame()
    frame_chw = torch.from_numpy(frame_hwc.transpose(2, 0, 1).copy())
    target_hw = (224, 224)

    train = preprocess_chw_tensor_for_policy(frame_chw, target_hw=target_hw, crop_box=BOX)
    eval_ = torch.from_numpy(
        preprocess_hwc_rgb_uint8_for_policy(frame_hwc, target_hw=target_hw, crop_box=BOX)
    )
    # Reference formula is div(255), matching lerobot's video-decode float
    # conversion — training floats were ALWAYS div-based (decoded floats pass
    # through _chw_tensor_to_float01 untouched); mul(1/255) would be a
    # 1-ULP train/deploy mismatch.
    old_training_path = v2.Resize(target_hw, antialias=True)(
        crop_box_for_chw_tensor(frame_chw.float().div(255.0), BOX)
    ).contiguous()

    assert train.shape == (3, *target_hw)
    assert train.dtype == torch.float32
    torch.testing.assert_close(train, eval_, rtol=0, atol=0)
    torch.testing.assert_close(train, old_training_path, rtol=0, atol=0)
    assert train.numpy().tobytes() == eval_.numpy().tobytes()
    assert train.numpy().tobytes() == old_training_path.numpy().tobytes()


def test_train_eval_role_reference_crop_resize_outputs_identical_tensor():
    native_h, native_w = 720, 1280
    yy, xx = np.meshgrid(
        np.arange(native_h, dtype=np.float32),
        np.arange(native_w, dtype=np.float32),
        indexing="ij",
    )
    native_hwc = np.stack([yy % 256, xx % 256, (yy + xx) % 256], axis=-1).astype(np.uint8)

    import cv2

    stored_hwc = cv2.resize(native_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    stored_chw = torch.from_numpy(stored_hwc.transpose(2, 0, 1).copy())
    target_hw = (64, 64)

    train = preprocess_chw_tensor_for_policy(
        stored_chw,
        target_hw=target_hw,
        crop_box=BOX,
        crop_reference_hw=(480, 640),
    )
    eval_ = torch.from_numpy(
        preprocess_hwc_rgb_uint8_for_policy(
            native_hwc,
            target_hw=target_hw,
            crop_box=BOX,
            crop_reference_hw=(480, 640),
        )
    )

    torch.testing.assert_close(train, eval_, rtol=0, atol=0)
    assert train.numpy().tobytes() == eval_.numpy().tobytes()


def test_train_eval_role_reference_resize_outputs_identical_tensor_without_crop():
    native_h, native_w = 720, 1280
    yy, xx = np.meshgrid(
        np.arange(native_h, dtype=np.float32),
        np.arange(native_w, dtype=np.float32),
        indexing="ij",
    )
    native_hwc = np.stack([yy % 256, xx % 256, (2 * yy + xx) % 256], axis=-1).astype(np.uint8)

    import cv2

    stored_hwc = cv2.resize(native_hwc, (640, 480), interpolation=cv2.INTER_AREA)
    stored_chw = torch.from_numpy(stored_hwc.transpose(2, 0, 1).copy())
    target_hw = (64, 64)

    train = preprocess_chw_tensor_for_policy(
        stored_chw,
        target_hw=target_hw,
        crop_reference_hw=(480, 640),
    )
    eval_ = torch.from_numpy(
        preprocess_hwc_rgb_uint8_for_policy(
            native_hwc,
            target_hw=target_hw,
            crop_reference_hw=(480, 640),
        )
    )

    torch.testing.assert_close(train, eval_, rtol=0, atol=0)
    assert train.numpy().tobytes() == eval_.numpy().tobytes()


def test_installed_per_camera_crop_proxy_uses_shared_resize_path():
    fkey = f"observation.images.{CAM}"
    frame_hwc = _make_frame()
    frame_chw = torch.from_numpy(frame_hwc.transpose(2, 0, 1).copy())
    target_hw = (64, 64)

    class _Meta:
        camera_keys = [fkey]

    class _Inner:
        meta = _Meta()

        def __init__(self):
            self.cleared = False

        def clear_image_transforms(self):
            self.cleared = True

        def __getitem__(self, idx):
            assert idx == 0
            return {fkey: frame_chw.clone()}

        def __len__(self):
            return 1

    class _Dataset:
        def __init__(self):
            self._datasets = [_Inner()]

    def _forbidden_base_transform(_img):
        raise AssertionError("cropped shared path must not call base_transform")

    dataset = _Dataset()
    install_per_camera_crop(
        dataset,
        {fkey: BOX},
        _forbidden_base_transform,
        crop_resize_hw=target_hw,
    )

    got = dataset._datasets[0][0][fkey]
    expected = preprocess_chw_tensor_for_policy(frame_chw, target_hw=target_hw, crop_box=BOX)
    assert dataset._datasets[0].cleared
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert got.numpy().tobytes() == expected.numpy().tobytes()


def test_crop_box_outside_frame_fails_loudly():
    frame_chw = torch.zeros(3, NATIVE_H, NATIVE_W)
    with pytest.raises(ValueError, match="side-crop box"):
        crop_box_for_chw_tensor(frame_chw, (0, 0, NATIVE_W + 10, 10))


def test_direct_crop_helpers_reject_inverted_boxes():
    frame_chw = torch.zeros(3, NATIVE_H, NATIVE_W)
    with pytest.raises(ValueError, match="invalid"):
        crop_box_for_chw_tensor(frame_chw, (10, 0, 10, 20))
    with pytest.raises(ValueError, match="invalid"):
        crop_box_for_chw_tensor(frame_chw, (10, 0, 20, 0))


def test_merge_default_crops_applies_only_to_selected_without_explicit():
    defaults = {"side_1": (1, 2, 3, 4), "wrist_left": (5, 6, 7, 8)}
    # side_1 has an explicit box (wins); wrist_left gets the default; side_2 is not
    # selected so its default (if any) is ignored.
    explicit = {"side_1": (10, 20, 30, 40)}
    merged = merge_default_crops(explicit, {"side_1", "wrist_left"}, defaults)
    assert merged == {"side_1": (10, 20, 30, 40), "wrist_left": (5, 6, 7, 8)}


def test_merge_default_crops_empty_defaults_is_noop():
    explicit = {"side_1": (1, 2, 3, 4)}
    assert merge_default_crops(explicit, {"side_1", "wrist_left"}, {}) == explicit


def test_merge_default_crops_skips_unselected_cameras():
    defaults = {"side_2": (1, 2, 3, 4)}  # not consumed by the DP
    assert merge_default_crops({}, {"side_1", "wrist_left"}, defaults) == {}


def test_merge_default_crops_warns_loud_on_camera_name_mismatch(capsys):
    # STATION_CAMERA_DEFAULT_CROPS is role-keyed (side_1, ...); a selection whose names are
    # not station roles binds 0 defaults and, with no explicit box, the policy would
    # silently train UNCROPPED. merge_default_crops must WARN loudly (returning {} -- no
    # boxes applied) so the mismatch cannot pass in silence.
    from mulligan.real.robot.cameras import STATION_CAMERA_DEFAULT_CROPS

    merged = merge_default_crops({}, {"cam_a_left", "cam_b_left"}, STATION_CAMERA_DEFAULT_CROPS)
    assert merged == {}
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "0 applied" in out
    # The warning names BOTH the considered default keys and the selected keys.
    assert "side_1" in out
    assert "cam_a_left" in out


def test_merge_default_crops_binds_role_keyed_defaults_to_role_selection(capsys):
    # The intended marker_d2 path: role-named selection {side_1, wrist_left} binds the
    # role-keyed default boxes for those two roles (and only those), with no warning.
    from mulligan.real.robot.cameras import STATION_CAMERA_DEFAULT_CROPS

    merged = merge_default_crops({}, {"side_1", "wrist_left"}, STATION_CAMERA_DEFAULT_CROPS)
    assert merged == {
        "side_1": STATION_CAMERA_DEFAULT_CROPS["side_1"],
        "wrist_left": STATION_CAMERA_DEFAULT_CROPS["wrist_left"],
    }
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_build_crop_feature_map_keys_on_feature_path():
    """build_crop_feature_map maps a bare camera name onto its
    ``observation.images.<camera>`` feature key."""
    assert build_crop_feature_map({"cam9": (1, 2, 3, 4)}) == {
        "observation.images.cam9": (1, 2, 3, 4)
    }
