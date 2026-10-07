from pathlib import Path

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mulligan.data.transforms import (
    compute_multidataset_valid_boundaries,
    fill_null_float_columns_with_nan,
    fill_null_floats_in_lerobot_subdatasets,
    remap_action_to_position_r6_in_subdatasets,
)
from mulligan.data.recording import consolidate_episodes_parquet


def _episode_meta_frame(episode_indices: list[int], chunk_idx: int, file_idx: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_index": episode_indices,
            "length": [10] * len(episode_indices),
            "meta/episodes/chunk_index": [chunk_idx] * len(episode_indices),
            "meta/episodes/file_index": [file_idx] * len(episode_indices),
        }
    )


def test_consolidate_episodes_parquet_rewrites_self_pointers(tmp_path: Path) -> None:
    episodes_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    _episode_meta_frame([0, 1], 0, 0).to_parquet(episodes_dir / "file-000.parquet", index=False)
    _episode_meta_frame([2, 3], 0, 1).to_parquet(episodes_dir / "file-001.parquet", index=False)

    consolidate_episodes_parquet(tmp_path)

    remaining = sorted((tmp_path / "meta" / "episodes").glob("*/*.parquet"))
    assert remaining == [episodes_dir / "file-000.parquet"]
    merged = pd.read_parquet(remaining[0])
    assert sorted(merged["episode_index"]) == [0, 1, 2, 3]
    # Every row now lives in chunk-000/file-000; stale self-pointers would
    # break per-episode metadata readers (lerobot _load_episode_with_stats).
    assert (merged["meta/episodes/chunk_index"] == 0).all()
    assert (merged["meta/episodes/file_index"] == 0).all()


# ---------------------------------------------------------------------------
# Canonical action remap engine (_apply_action_remap)
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, n: int) -> None:
        self.info = SimpleNamespace(
            features={"action": {"dtype": "float32", "shape": (7,), "names": ["vel"] * 7}}
        )
        self.stats = {
            "action": {
                "min": np.full(7, -1.0, np.float32),
                "max": np.full(7, 1.0, np.float32),
                "mean": np.zeros(7, np.float32),
                "std": np.ones(7, np.float32),
                "count": np.array([n]),
            }
        }


class _FakeReader:
    """Minimal stand-in for LeRobotDataset.reader: holds a settable hf_dataset
    (mirrors dataset_reader.py:82). The remap helpers write through
    ``sub.reader.hf_dataset`` now that LeRobotDataset.hf_dataset is a read-only
    property delegating to the reader (lerobot 0.5.2 DatasetReader split)."""

    def __init__(self, hf_dataset) -> None:
        self.hf_dataset = hf_dataset


class _FakeSub:
    def __init__(self, hf_dataset, n: int) -> None:
        self.reader = _FakeReader(hf_dataset)
        self.meta = _FakeMeta(n)
        self.repo_id = "synthetic/test"

    @property
    def hf_dataset(self):
        return self.reader.hf_dataset


class _FakeMulti:
    def __init__(self, subs) -> None:
        self._datasets = subs
        self.stats: dict = {}


def _make_sub(n: int):
    from datasets import Dataset
    from lerobot.datasets.io_utils import hf_transform_to_torch

    rng = np.random.default_rng(0)
    vel = rng.uniform(-1, 1, (n, 7)).astype(np.float32)
    cart_pos = rng.uniform(0.1, 0.7, (n, 6)).astype(np.float32)
    grip_pos = rng.integers(0, 2, (n, 1)).astype(np.float32)
    hf = Dataset.from_dict(
        {
            "action": [r.tolist() for r in vel],
            "action.cartesian_position": [r.tolist() for r in cart_pos],
            "action.gripper_position": [r.tolist() for r in grip_pos],
            "episode_index": [0] * n,
        }
    )
    hf.set_transform(hf_transform_to_torch)
    return _FakeSub(hf, n), cart_pos, grip_pos


# ---------------------------------------------------------------------------
# fill_null_float_columns_with_nan (marker_d2 R0 null-telemetry crash)
# ---------------------------------------------------------------------------


def _dataset_with_columns(columns: dict):
    """Build a torch-formatted HF Dataset from a {name: pyarrow Array} mapping."""
    import pyarrow as pa
    from datasets import Dataset
    from lerobot.datasets.io_utils import hf_transform_to_torch

    table = pa.table(columns)
    hf = Dataset(table)
    hf.set_transform(hf_transform_to_torch)
    return hf


def _fixed_list_f32(rows, size):
    import pyarrow as pa

    return pa.array(rows, type=pa.list_(pa.float32(), size))


def test_fill_null_float_columns_substitutes_nan_in_fixed_size_list() -> None:
    import pyarrow as pa
    import torch

    # A telemetry-like fixed_size_list<float32>[3]: row 0 fully recorded, row 1 fully
    # unrecorded (all child elements null), row 2 partially null. This is exactly the
    # marker_d2 R0 shape that makes torch.tensor(None) raise in hf_transform_to_torch.
    hf = _dataset_with_columns(
        {
            "telemetry.x": _fixed_list_f32(
                [[1.0, 2.0, 3.0], [None, None, None], [4.0, None, 6.0]], 3
            ),
            "observation.state": _fixed_list_f32([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], 2),
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    # Before the fill, reading the all-null row raises.
    with pytest.raises(RuntimeError):
        _ = hf[1]

    new_hf, filled = fill_null_float_columns_with_nan(hf)
    assert filled == {"telemetry.x": 4}, filled  # 3 (row1) + 1 (row2) null child floats

    # Row 1 is now an all-NaN vector; row 2 keeps its real entries with NaN in the gap.
    row1 = new_hf[1]["telemetry.x"]
    assert row1.dtype == torch.float32 and bool(torch.isnan(row1).all())
    row2 = new_hf[2]["telemetry.x"]
    assert row2[0].item() == 4.0 and bool(torch.isnan(row2[1])) and row2[2].item() == 6.0
    # The clean observation.state column is untouched (no NaN introduced).
    assert not bool(torch.isnan(new_hf[1]["observation.state"]).any())


def test_fill_null_float_columns_noop_on_clean_dataset() -> None:
    import pyarrow as pa

    hf = _dataset_with_columns(
        {
            "telemetry.x": _fixed_list_f32([[1.0, 2.0], [3.0, 4.0]], 2),
            "episode_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    new_hf, filled = fill_null_float_columns_with_nan(hf)
    assert filled == {}
    assert new_hf is hf  # no rebuild when nothing is null


def test_fill_null_float_columns_raises_on_protected_column() -> None:
    import pyarrow as pa

    # A null in a policy-critical column (action / observation.*) is corrupt supervision,
    # not an unrecorded aux signal — it must fail loudly, never be silently NaN-filled.
    hf = _dataset_with_columns(
        {
            "action": _fixed_list_f32([[0.0, 1.0], [None, None]], 2),
            "episode_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    with pytest.raises(ValueError, match="policy-critical column 'action'"):
        fill_null_float_columns_with_nan(hf)


def test_fill_null_floats_in_lerobot_subdatasets_mutates_and_aggregates() -> None:
    import pyarrow as pa
    import torch

    hf = _dataset_with_columns(
        {
            "telemetry.x": _fixed_list_f32([[1.0, 2.0], [None, None]], 2),
            "episode_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    sub = _FakeSub(hf, 2)
    total = fill_null_floats_in_lerobot_subdatasets([sub])
    assert total == {"telemetry.x": 2}
    # The sub-dataset's reader.hf_dataset was swapped in place and now reads without crashing.
    assert bool(torch.isnan(sub.reader.hf_dataset[1]["telemetry.x"]).all())


def test_compute_multidataset_valid_boundaries_requires_len() -> None:
    class NoLenSubDataset:
        hf_dataset = {"episode_index": [0, 0]}

    with pytest.raises(TypeError, match="sub_dataset must define __len__"):
        compute_multidataset_valid_boundaries([NoLenSubDataset()])


def test_remap_reads_features_without_the_deprecated_dict_access() -> None:
    """The remap must not read ``info["features"]``, which lerobot's DatasetInfo
    marks deprecated and warns on for every sub-dataset."""
    import warnings

    from lerobot.datasets.utils import DatasetInfo

    sub, _, _ = _make_sub(8)
    sub.meta.info = DatasetInfo(
        codebase_version="v3.0",
        fps=15,
        features={"action": {"dtype": "float32", "shape": (7,), "names": ["vel"] * 7}},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        remap_action_to_position_r6_in_subdatasets(_FakeMulti([sub]))
    assert sub.meta.info.features["action"]["names"][0] == "x"


# ---------------------------------------------------------------------------
# Filtered-episode padding
#
# lerobot's native fix (PR #2612) handles a dataset re-opened with the leading
# episode filtered out and delta_timestamps that look ahead a few action steps:
# the per-key padding mask (`action_is_pad`) must line up with the absolute
# frame index of the retained episodes, or valid mid-episode actions of a
# retained episode are wrongly flagged as padding (mask all-True).
# `_get_query_indices` keys the mask off the absolute `index`
# column and the absolute `dataset_from_index`/`dataset_to_index` episode
# bounds (dataset_reader.py:_get_query_indices), so mid-episode look-ahead
# windows stay (mostly) unpadded.
#
# This builds a real minimal LeRobotDataset (state-only, no video — fast) so it
# exercises the actual native code path rather than a stub.
# ---------------------------------------------------------------------------


def _build_state_only_lerobot_dataset(root: Path, *, fps: int, ep_len: int, n_episodes: int):
    """Create + finalize a tiny video-free LeRobotDataset on disk.

    Frame ``t`` of episode ``ep`` encodes action[:, 0] == ep * 100 + t so the
    absolute-index alignment of a retained episode is checkable after filtering.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["da", "db"]},
    }
    ds = LeRobotDataset.create(
        repo_id="local/filtered-pad-regression",
        fps=fps,
        features=features,
        root=root,
        use_videos=False,
    )
    for ep in range(n_episodes):
        for t in range(ep_len):
            v = float(ep * 100 + t)
            ds.add_frame(
                {
                    "observation.state": np.array([v, v + 0.5], np.float32),
                    "action": np.array([v, -v], np.float32),
                    "task": "regression",
                }
            )
        ds.save_episode()
    ds.finalize()


def test_filtered_leading_episode_does_not_pad_midepisode_actions(tmp_path: Path) -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps, ep_len, n_episodes, look = 10, 7, 3, 3
    root = tmp_path / "ds"
    _build_state_only_lerobot_dataset(root, fps=fps, ep_len=ep_len, n_episodes=n_episodes)

    # Re-open dropping the LEADING episode (0) and looking ahead `look` action steps.
    delta_timestamps = {"action": [i / fps for i in range(look + 1)]}
    filtered = LeRobotDataset(
        repo_id="local/filtered-pad-regression",
        root=root,
        episodes=[1, 2],
        delta_timestamps=delta_timestamps,
    )
    assert filtered.num_episodes == 2
    assert filtered.num_frames == 2 * ep_len

    # Iterate the FIRST retained episode (now relative indices 0..ep_len-1).
    first_retained = filtered[0]["episode_index"].item()
    assert first_retained == 1  # leading episode 0 really was dropped

    midepisode_all_false = 0
    for rel in range(ep_len):
        item = filtered[rel]
        assert item["episode_index"].item() == 1
        is_pad = item["action_is_pad"]
        action = item["action"]

        assert is_pad.shape == (look + 1,)
        # Absolute-index alignment survived the filter: episode 1's frame `rel`
        # encodes ep*100 + rel, and the first delta (0) is always the current frame.
        assert action[0, 0].item() == float(100 + rel)
        # The current-frame action is real data, so its pad flag is never set —
        # a wrong mask would show up as this (and the whole window) being True.
        assert not bool(is_pad[0])
        assert bool(torch.isfinite(action).all())

        # Mid-episode frames have a full look-ahead window inside the episode, so
        # none of their action steps are padding. Tail frames legitimately pad the overflowing steps.
        is_midepisode = rel + look < ep_len
        if is_midepisode:
            assert not bool(is_pad.any()), (
                f"mid-episode frame rel={rel} of retained episode 1 has padding "
                f"{is_pad.tolist()} despite a fully in-bounds look-ahead window"
            )
            midepisode_all_false += 1

    # Sanity: at least one mid-episode frame yielded an all-False pad mask.
    assert midepisode_all_false >= 1
