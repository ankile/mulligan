"""Tests for mulligan.data.frame_cache (decode-once shared-RAM frame cache).

Builds a synthetic on-disk repo and pins:

- cache-fed frames are BYTE-IDENTICAL to quantize(preprocess(native frames))
  in float and uint8 decode modes,
  including multi-timestamp (delta_timestamps) queries;
- the build refuses missing crop boxes and stock readers (loud, not silent);
- episode-subset datasets (served-row remapping) cache correctly;
- the memory gate falls back with False instead of building an oversized cache;
- the crop proxy's precropped mode skips crop/resize but still runs the post
  transform.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.fast_lerobot_reader import enable_fast_reader
from mulligan.data.frame_cache import (
    attach_frame_cache,
    build_and_attach_frame_caches,
    build_frame_cache_for_dataset,
    preprocess_uint8_batch,
)
from mulligan.real.policy.image_preprocess import (
    preprocess_chw_tensor_for_policy,
    quantize_float01_to_uint8,
)

FPS = 15
CAMS = ["observation.images.side_1", "observation.images.wrist_left"]
CROP_BOX = (4, 2, 52, 62)  # x0, y0, x1, y1 inside the 64x64 native frame
TARGET_HW = (96, 96)
CROP_MAP_FULL = {CAMS[0]: CROP_BOX, CAMS[1]: (0, 0, 64, 64)}

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (4,), "names": ["a", "b", "c", "d"]},
    "action": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    **{
        cam: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]}
        for cam in CAMS
    },
}

DELTA_TIMESTAMPS = {cam: [0, 5 / FPS] for cam in CAMS}


@pytest.fixture(scope="module")
def cache_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("frame_cache_repo_root")
    repo_id = "test/frame_cache_main"
    ds = LeRobotDataset.create(repo_id, fps=FPS, features=FEATURES, root=root / repo_id)
    rng = np.random.RandomState(1)
    for _ep in range(2):
        for _t in range(15):
            frame = {
                "observation.state": rng.randn(4).astype(np.float32),
                "action": rng.randn(3).astype(np.float32),
                "task": "t",
            }
            for cam in CAMS:
                frame[cam] = rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    return root, repo_id


def _open(root, repo_id, **kw):
    return LeRobotDataset(
        repo_id,
        root=root / repo_id,
        delta_timestamps=DELTA_TIMESTAMPS,
        video_backend="torchcodec",
        **kw,
    )


def _build_and_attach(ds, crop_map=CROP_MAP_FULL):
    enable_fast_reader(ds)
    caches = build_frame_cache_for_dataset(
        ds,
        crop_feature_map=crop_map,
        crop_reference_hw=None,
        target_hw=TARGET_HW,
        threads=2,
    )
    attach_frame_cache(ds, caches)
    return caches


def _expected(ref_frames: torch.Tensor, cam: str) -> torch.Tensor:
    """quantize(preprocess(native)), the representation the cache serves."""
    return quantize_float01_to_uint8(
        preprocess_chw_tensor_for_policy(
            ref_frames, target_hw=TARGET_HW, crop_box=CROP_MAP_FULL[cam]
        )
    )


def test_preprocess_uint8_batch_is_quantized_policy_preprocess():
    g = torch.Generator().manual_seed(0)
    native = torch.randint(0, 256, (3, 3, 48, 64), dtype=torch.uint8, generator=g)
    box = (4, 2, 60, 44)
    got = preprocess_uint8_batch(native, crop_box=box, crop_reference_hw=None, target_hw=(24, 32))
    want = quantize_float01_to_uint8(
        preprocess_chw_tensor_for_policy(native, target_hw=(24, 32), crop_box=box)
    )
    assert got.dtype == torch.uint8
    assert torch.equal(got, want)


def test_cached_items_byte_identical_to_pipeline(cache_repo):
    root, repo_id = cache_repo
    ref = _open(root, repo_id)
    cached = _open(root, repo_id)
    _build_and_attach(cached)

    for idx in (0, 7, 16, 29):
        ref_item = ref[idx]
        cached_item = cached[idx]
        for cam in CAMS:
            expected = _expected(ref_item[cam], cam)  # (2,3,96,96) uint8
            got = cached_item[cam]
            assert got.dtype == torch.float32 and got.shape == expected.shape
            torch.testing.assert_close(got, expected.to(torch.float32) / 255.0, rtol=0, atol=0)


def test_cached_items_uint8_mode(cache_repo):
    root, repo_id = cache_repo
    ref = _open(root, repo_id)
    cached = _open(root, repo_id)
    _build_and_attach(cached)
    cached.reader._return_uint8 = True

    item = cached[3]
    ref_item = ref[3]
    for cam in CAMS:
        assert item[cam].dtype == torch.uint8
        assert torch.equal(item[cam], _expected(ref_item[cam], cam))


def test_episode_subset_served_rows(cache_repo):
    root, repo_id = cache_repo
    ref = _open(root, repo_id)
    subset = _open(root, repo_id, episodes=[1])
    _build_and_attach(subset)
    assert len(subset) == 15

    ref_item = ref[15 + 4]  # episode 1, frame 4 in the full dataset
    item = subset[4]
    for cam in CAMS:
        torch.testing.assert_close(
            item[cam], _expected(ref_item[cam], cam).to(torch.float32) / 255.0, rtol=0, atol=0
        )


def test_requires_box_for_every_camera(cache_repo):
    root, repo_id = cache_repo
    ds = _open(root, repo_id)
    with pytest.raises(ValueError, match="crop box for every video key"):
        _build_and_attach(ds, crop_map={CAMS[0]: CROP_BOX})


def test_requires_fast_reader(cache_repo):
    root, repo_id = cache_repo
    ds = _open(root, repo_id)
    _ = ds[0]  # force reader construction
    with pytest.raises(TypeError, match="FastDatasetReader"):
        build_frame_cache_for_dataset(
            ds,
            crop_feature_map=CROP_MAP_FULL,
            crop_reference_hw=None,
            target_hw=TARGET_HW,
        )


def test_size_estimate_requires_fast_reader(cache_repo):
    """The size estimate raises the TypeError guard on a stock reader, not an
    AttributeError."""
    from mulligan.data.frame_cache import frame_cache_size_bytes

    root, repo_id = cache_repo
    ds = _open(root, repo_id)
    _ = ds[0]
    with pytest.raises(TypeError, match="FastDatasetReader"):
        frame_cache_size_bytes([ds], TARGET_HW)


def test_single_row_uint8_frame_does_not_alias_cache(cache_repo):
    """The single-timestamp path returns a copy, so an in-place op on the item
    does not overwrite the cached frame."""
    root, repo_id = cache_repo
    ds = LeRobotDataset(repo_id, root=root / repo_id, video_backend="torchcodec")
    caches = _build_and_attach(ds)
    ds.reader._return_uint8 = True
    before = caches[CAMS[0]][3].clone()
    ds[3][CAMS[0]].zero_()
    assert torch.equal(caches[CAMS[0]][3], before)
    assert torch.equal(ds[3][CAMS[0]], before)


def test_build_and_attach_full_path_with_serve_check(cache_repo):
    # Exercises the trainer entrypoint end-to-end: budget resolution, build,
    # attach, and the SERVE-path self-check (boundary + random rows through
    # reader._query_frame_cache against stock decode).
    root, repo_id = cache_repo
    ds = _open(root, repo_id)
    enable_fast_reader(ds)
    ok = build_and_attach_frame_caches(
        [ds],
        crop_feature_map=CROP_MAP_FULL,
        crop_reference_hw=None,
        target_hw=TARGET_HW,
    )
    assert ok is True
    assert ds.reader._frame_cache is not None
    # Serve-path flag restored after the uint8-mode self-check.
    assert ds.reader._return_uint8 is False


def test_non_contiguous_episode_runs_refused():
    from types import SimpleNamespace

    from mulligan.data.frame_cache import _served_episode_ranges

    fake = SimpleNamespace(
        _tensor_columns={"episode_index": torch.tensor([0, 0, 1, 1, 0], dtype=torch.long)}
    )
    with pytest.raises(ValueError, match="multiple non-contiguous served-row runs"):
        _served_episode_ranges(fake)


def test_mem_gate_falls_back(cache_repo, monkeypatch, capsys):
    root, repo_id = cache_repo
    ds = _open(root, repo_id)
    enable_fast_reader(ds)
    monkeypatch.setenv("MULLIGAN_REAL_FRAME_CACHE_MAX_GB", "0.000001")
    ok = build_and_attach_frame_caches(
        [ds],
        crop_feature_map=CROP_MAP_FULL,
        crop_reference_hw=None,
        target_hw=TARGET_HW,
    )
    assert ok is False
    assert ds.reader._frame_cache is None
    assert "falling back to per-sample video decode" in capsys.readouterr().out


def test_mem_gate_charges_reserved_bytes(cache_repo, monkeypatch):
    # A budget that admits ONE cache must refuse the second call once the first
    # cache's bytes are charged as reserved (train+eval sum stays under limit).
    root, repo_id = cache_repo
    ds = _open(root, repo_id)
    enable_fast_reader(ds)
    from mulligan.data.frame_cache import frame_cache_size_bytes

    need = frame_cache_size_bytes([ds], TARGET_HW)
    monkeypatch.setenv("MULLIGAN_REAL_FRAME_CACHE_MAX_GB", str(need * 1.5 / 1e9))
    kw = dict(crop_feature_map=CROP_MAP_FULL, crop_reference_hw=None, target_hw=TARGET_HW)
    assert build_and_attach_frame_caches([ds], **kw) is True
    ds2 = _open(root, repo_id)
    enable_fast_reader(ds2)
    assert build_and_attach_frame_caches([ds2], reserved_bytes=need, **kw) is False
    assert ds2.reader._frame_cache is None


def test_resume_frame_cache_state_bookkeeping(monkeypatch):
    from mulligan.real.train.policy import check_resume_frame_cache_state

    # Older checkpoint (no key) = uncached so far: mixed only if now active.
    assert check_resume_frame_cache_state({}, True) is True
    assert check_resume_frame_cache_state({}, False) is False
    # Matching state stays pure; flips mark mixed in both directions.
    assert check_resume_frame_cache_state({"decoded_frame_cache": True}, True) is False
    assert check_resume_frame_cache_state({"decoded_frame_cache": True}, False) is True
    assert check_resume_frame_cache_state({"decoded_frame_cache": False}, True) is True
    # Mixed marker is sticky even when the current state matches again.
    assert (
        check_resume_frame_cache_state(
            {"decoded_frame_cache": True, "decoded_frame_cache_mixed": True}, True
        )
        is True
    )


def test_crop_proxy_precropped_mode(cache_repo):
    from mulligan.real.policy.side_crop import _PerCameraCropSubset

    root, repo_id = cache_repo
    cached = _open(root, repo_id)
    _build_and_attach(cached)

    marks = []

    def post(img):
        marks.append(img.shape)
        return img

    proxy = _PerCameraCropSubset(
        cached,
        CROP_MAP_FULL,
        None,
        crop_resize_hw=TARGET_HW,
        crop_post_transform=post,
    )
    proxy.set_frames_precropped(True)
    item = proxy[0]
    # Post transform ran once per camera on ALREADY-policy-res frames (no re-crop).
    assert len(marks) == len(CAMS)
    for cam in CAMS:
        assert item[cam].shape[-2:] == TARGET_HW
