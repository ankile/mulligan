"""Byte-exact parity tests for mulligan.data.fast_lerobot_reader.

Builds a real on-disk LeRobotDataset (videos included) and pins that
FastDatasetReader.get_item returns EXACTLY the stock DatasetReader output —
same keys, dtypes, shapes, and bitwise-equal values — across:

- every index (covers episode-boundary padding),
- delta_timestamps on and off,
- episodes filtering,
- uint8 decode mode,
- pickled round-trip (spawn-worker simulation),
- MultiLeRobotDataset composition (trainer integration path).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.multi_dataset import MultiLeRobotDataset

from mulligan.data.fast_lerobot_reader import (
    FastDatasetReader,
    enable_fast_reader,
    enable_fast_reader_on_subdatasets,
)

FPS = 15
K = 5
CAMS = ["observation.images.side_1", "observation.images.wrist_left"]
EPISODE_LENGTHS = [20, 17, 23]
TASKS = ["task_A", "task_B", "task_A"]

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (4,), "names": ["a", "b", "c", "d"]},
    "action": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    "reward": {"dtype": "float32", "shape": (1,), "names": None},
    "done": {"dtype": "bool", "shape": (1,), "names": None},
    **{
        cam: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]}
        for cam in CAMS
    },
}

DELTA_TIMESTAMPS = {
    "observation.state": [0, K / FPS],
    "action": [i / FPS for i in range(K)],
    "reward": [i / FPS for i in range(K)],
    "done": [i / FPS for i in range(K)],
    **{cam: [0, K / FPS] for cam in CAMS},
}


def _write_repo(root, repo_id: str, episode_lengths, tasks) -> None:
    ds = LeRobotDataset.create(repo_id, fps=FPS, features=FEATURES, root=root / repo_id)
    rng = np.random.RandomState(0)
    for ep, (n, task) in enumerate(zip(episode_lengths, tasks, strict=True)):
        for t in range(n):
            frame = {
                "observation.state": rng.randn(4).astype(np.float32),
                "action": rng.randn(3).astype(np.float32),
                "reward": np.array([float(ep * 100 + t)], dtype=np.float32),
                "done": np.array([t == n - 1]),
                "task": task,
            }
            for cam in CAMS:
                frame[cam] = rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()


@pytest.fixture(scope="module")
def repo_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fast_reader_repos")
    _write_repo(root, "test/parity_main", EPISODE_LENGTHS, TASKS)
    _write_repo(root, "test/parity_second", [12], ["task_C"])
    return root


def _open(root, repo_id: str, *, delta=True, episodes=None) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id,
        root=root / repo_id,
        episodes=episodes,
        delta_timestamps=DELTA_TIMESTAMPS if delta else None,
        video_backend="torchcodec",
    )


def _assert_item_equal(stock: dict, fast: dict, idx: int) -> None:
    assert set(stock.keys()) == set(fast.keys()), (
        f"idx {idx}: key mismatch {set(stock) ^ set(fast)}"
    )
    for key in stock:
        s, f = stock[key], fast[key]
        if isinstance(s, torch.Tensor):
            assert isinstance(f, torch.Tensor), f"idx {idx} key {key}: {type(f)}"
            assert s.dtype == f.dtype, f"idx {idx} key {key}: dtype {s.dtype} vs {f.dtype}"
            assert s.shape == f.shape, f"idx {idx} key {key}: shape {s.shape} vs {f.shape}"
            assert torch.equal(s, f), f"idx {idx} key {key}: values differ"
        else:
            assert type(s) is type(f), f"idx {idx} key {key}: type {type(s)} vs {type(f)}"
            assert s == f, f"idx {idx} key {key}: {s!r} vs {f!r}"


def _assert_full_parity(stock_ds, fast_ds) -> None:
    assert len(stock_ds) == len(fast_ds)
    for idx in range(len(stock_ds)):
        _assert_item_equal(stock_ds[idx], fast_ds[idx], idx)


def test_full_parity_with_delta_timestamps(repo_root):
    stock = _open(repo_root, "test/parity_main")
    fast = _open(repo_root, "test/parity_main")
    enable_fast_reader(fast)
    assert isinstance(fast.reader, FastDatasetReader)
    _assert_full_parity(stock, fast)


def test_full_parity_without_delta_timestamps(repo_root):
    stock = _open(repo_root, "test/parity_main", delta=False)
    fast = _open(repo_root, "test/parity_main", delta=False)
    enable_fast_reader(fast)
    _assert_full_parity(stock, fast)


def test_full_parity_with_episode_filter(repo_root):
    stock = _open(repo_root, "test/parity_main", episodes=[0, 2])
    fast = _open(repo_root, "test/parity_main", episodes=[0, 2])
    enable_fast_reader(fast)
    assert len(fast) == EPISODE_LENGTHS[0] + EPISODE_LENGTHS[2]
    _assert_full_parity(stock, fast)


def test_parity_in_uint8_decode_mode(repo_root):
    stock = _open(repo_root, "test/parity_main")
    fast = _open(repo_root, "test/parity_main")
    enable_fast_reader(fast)
    stock._ensure_reader()._return_uint8 = True
    fast.reader._return_uint8 = True
    for idx in [0, 7, len(stock) - 1]:
        _assert_item_equal(stock[idx], fast[idx], idx)


def test_pickle_round_trip(repo_root):
    fast = _open(repo_root, "test/parity_main")
    enable_fast_reader(fast)
    item_before = fast[5]
    clone = pickle.loads(pickle.dumps(fast))
    assert isinstance(clone.reader, FastDatasetReader)
    assert clone.reader._video_pool is None
    _assert_item_equal(item_before, clone[5], 5)


def test_multidataset_integration(repo_root):
    kwargs = dict(
        repo_ids=["test/parity_main", "test/parity_second"],
        root=repo_root,
        delta_timestamps=DELTA_TIMESTAMPS,
        video_backend="torchcodec",
    )
    stock = MultiLeRobotDataset(**kwargs)
    fast = MultiLeRobotDataset(**kwargs)
    cached_bytes = enable_fast_reader_on_subdatasets(list(fast._datasets))
    assert cached_bytes > 0
    assert len(stock) == len(fast) == sum(EPISODE_LENGTHS) + 12
    for idx in range(0, len(stock), 3):
        _assert_item_equal(stock[idx], fast[idx], idx)
    # cross-repo boundary indices
    for idx in [sum(EPISODE_LENGTHS) - 1, sum(EPISODE_LENGTHS)]:
        _assert_item_equal(stock[idx], fast[idx], idx)


def test_enable_is_idempotent(repo_root):
    fast = _open(repo_root, "test/parity_main")
    r1 = enable_fast_reader(fast)
    r2 = enable_fast_reader(fast)
    assert r1 is r2


def test_indices_mapping_raises_loudly(repo_root):
    fast = _open(repo_root, "test/parity_main")
    reader = fast._ensure_reader()
    if reader.hf_dataset is None:
        reader.load_and_activate()
    # Non-contiguous selection forces a real _indices mapping (a contiguous
    # range is served as a table slice without one).
    reader.hf_dataset = reader.hf_dataset.select([5, 3, 1])
    with pytest.raises(RuntimeError, match="_indices"):
        enable_fast_reader(fast)


def test_table_replacement_rebuilds_cache(repo_root):
    """Regression: mulligan code replaces reader.hf_dataset wholesale
    (null-fill, DP action remaps in mulligan/data/transforms.py). The cache must
    follow the assignment, never serve stale pre-remap values."""
    fast = _open(repo_root, "test/parity_main", delta=False)
    enable_fast_reader(fast)
    before = fast[0]["action"].clone()

    from lerobot.datasets.io_utils import hf_transform_to_torch

    hf = fast.reader.hf_dataset
    doubled = hf.map(
        lambda batch: {"action": [[2 * v for v in row] for row in batch["action"]]},
        batched=True,
    )
    doubled.set_transform(hf_transform_to_torch)
    fast.reader.hf_dataset = doubled

    after = fast[0]["action"]
    assert torch.equal(after, before * 2), "cache served stale values after table replacement"


def test_returned_tensors_do_not_alias_cache(repo_root):
    """Stock torch.tensor(x) returns fresh tensors; mutating a returned item must
    not corrupt subsequent reads."""
    fast = _open(repo_root, "test/parity_main", delta=False)
    enable_fast_reader(fast)
    first = fast[1]["action"].clone()
    fast[1]["action"].zero_()
    assert torch.equal(fast[1]["action"], first)


def test_video_pool_is_rebuilt_for_new_pid(repo_root):
    fast = _open(repo_root, "test/parity_main")
    reader = enable_fast_reader(fast)
    fast[0]
    pool = reader._video_pool
    assert pool is not None
    # Simulate a fork: stale PID must force a fresh pool.
    reader._video_pool_pid = -1
    fresh = reader._get_video_pool(2)
    assert fresh is not pool


DEPTH_KEY = "observation.images.depth"


@pytest.fixture(scope="module")
def depth_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("fast_reader_depth")
    repo_id = "test/depth"
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        CAMS[0]: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
        DEPTH_KEY: {
            "dtype": "video",
            "shape": (32, 32, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
    }
    ds = LeRobotDataset.create(repo_id, fps=10, features=features, root=root / repo_id)
    rng = np.random.RandomState(0)
    for _ in range(6):
        ds.add_frame(
            {
                "observation.state": rng.randn(2).astype(np.float32),
                CAMS[0]: rng.randint(0, 255, (32, 32, 3), dtype=np.uint8),
                DEPTH_KEY: rng.randint(300, 2000, (32, 32, 1)).astype(np.uint16),
                "task": "t",
            }
        )
    ds.save_episode()
    ds.finalize()
    return root / repo_id, repo_id


def _constant_transform(x):
    return x * 0 + 7.0


@pytest.mark.parametrize("unit", ["mm", "m"])
def test_depth_keys_match_stock_reader(depth_repo, unit):
    """The fast reader decodes depth videos like the stock reader: dequantization
    and the depth-key skip in the image transforms both apply."""
    root, repo_id = depth_repo
    kw = dict(
        root=root,
        video_backend="torchcodec",
        image_transforms=_constant_transform,
        depth_output_unit=unit,
    )
    stock = LeRobotDataset(repo_id, **kw)
    fast = LeRobotDataset(repo_id, **kw)
    enable_fast_reader(fast)
    assert fast.meta.depth_keys == [DEPTH_KEY]
    for idx in range(len(stock)):
        _assert_item_equal(stock[idx], fast[idx], idx)
    item = fast[2]
    assert torch.all(item[CAMS[0]] == 7.0)
    assert item[DEPTH_KEY].mean() > (100 if unit == "mm" else 0.1)


def test_rgb_preprocess_paths_refuse_depth_keys(depth_repo):
    from mulligan.data.frame_cache import frame_cache_size_bytes

    root, repo_id = depth_repo
    fast = LeRobotDataset(repo_id, root=root, video_backend="torchcodec")
    enable_fast_reader(fast)
    with pytest.raises(ValueError, match="depth video keys"):
        frame_cache_size_bytes([fast], (16, 16))
