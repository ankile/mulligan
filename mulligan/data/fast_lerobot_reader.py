# FastDatasetReader.get_item is adapted from LeRobot (https://github.com/huggingface/lerobot,
# commit 0530dd9b), DatasetReader.get_item in src/lerobot/datasets/dataset_reader.py.
# Modified by the Mulligan authors: rows, delta-index queries and timestamps are served from
# a column cache built once, and video decode uses a persistent per-process thread pool.
# The rest of the file is by the Mulligan authors (MIT).
#
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Column-cache fast path for LeRobot ``DatasetReader.get_item``.

On the real-robot training data, profiling attributes
~20 ms/sample of pure Python overhead to HF ``datasets`` machinery in the
``__getitem__`` hot path — dominated by ``Features`` deepcopy churn inside
``hf_dataset[key][indices]`` column queries (~530k ``copy.deepcopy`` calls per
100 samples), plus per-row transform dispatch and a ``ThreadPoolExecutor``
constructed per call in ``_query_videos``.

``FastDatasetReader`` removes HF ``datasets`` from the hot path entirely:

- All parquet-backed (non-video) columns are cached ONCE at enable time as
  contiguous torch tensors (numeric) or plain Python lists (strings/dicts),
  replicating ``hf_transform_to_torch`` dtype semantics exactly
  (floats -> float32, ints -> int64, bools -> bool, strings stay str).
- ``get_item`` mirrors the upstream implementation statement-for-statement
  (lerobot ``dataset_reader.py``) but serves row access, delta-key queries,
  and timestamp queries from the cache via pure tensor indexing.
- Video decode goes through a per-process persistent thread pool instead of a
  new executor per item (pool is dropped on pickle and rebuilt per PID, so
  fork/spawn DataLoader workers each get their own).

Values are byte-identical to the stock reader (pinned by
``tests/unit/test_fast_lerobot_reader.py``); only the Python overhead changes.
Activation is via ``enable_fast_reader(dataset)`` which swaps the loaded
reader's ``__class__`` — instances stay picklable for spawn workers because the
subclass is importable from this module.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from lerobot.datasets.dataset_reader import DatasetReader

# Escape hatch: MULLIGAN_FAST_READER=0 keeps the stock reader (see trainers).
FAST_READER_ENV = "MULLIGAN_FAST_READER"


def fast_reader_enabled_by_env() -> bool:
    return os.environ.get(FAST_READER_ENV, "1") != "0"


def _column_to_cache_value(name: str, chunked: pa.ChunkedArray) -> torch.Tensor | list:
    """Convert one arrow column to its cache representation.

    Must replicate ``hf_transform_to_torch`` per-row semantics exactly:
    ``torch.tensor(x)`` on Python scalars/lists infers float32 for floats,
    int64 for ints, bool for bools; strings and dicts pass through unchanged.
    """
    from lerobot.datasets.language import LANGUAGE_COLUMNS

    col = chunked.combine_chunks()
    typ = col.type

    if name in LANGUAGE_COLUMNS:
        # Stock hf_transform_to_torch skips these entirely -> raw Python values.
        return col.to_pylist()
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        return col.to_pylist()
    if pa.types.is_struct(typ) or pa.types.is_null(typ):
        return col.to_pylist()

    if col.null_count > 0:
        # The stock reader would crash on torch.tensor(None); caching would
        # instead silently NaN/garbage-fill. Fail loudly — DP-style datamixes
        # must run fill_null_floats_in_lerobot_subdatasets BEFORE enabling.
        raise RuntimeError(
            f"fast reader: column '{name}' has {col.null_count} null values; "
            "null-fill the dataset before enabling the fast reader"
        )

    if pa.types.is_fixed_size_list(typ) or pa.types.is_list(typ) or pa.types.is_large_list(typ):
        if pa.types.is_fixed_size_list(typ):
            width = typ.list_size
            # .flatten(), NOT .values: .values ignores slice offsets on a
            # sliced FixedSizeListArray and would silently misalign rows.
            values = col.flatten()
        else:
            lengths = np.asarray(col.value_lengths())
            widths = np.unique(lengths)
            if len(widths) != 1:
                # Ragged column: keep exact per-row torch.tensor semantics.
                return [torch.tensor(row) for row in col.to_pylist()]
            width = int(widths[0])
            values = col.flatten()
        if width == 0:
            # torch.tensor([]) infers float32 — match stock inference for empty lists.
            return torch.zeros((len(col), 0), dtype=torch.float32)
        if values.null_count > 0:
            raise RuntimeError(
                f"fast reader: column '{name}' has {values.null_count} null list elements; "
                "null-fill the dataset before enabling the fast reader"
            )
        if not (
            pa.types.is_floating(values.type)
            or pa.types.is_integer(values.type)
            or pa.types.is_boolean(values.type)
        ):
            raise TypeError(
                f"fast reader cannot cache column '{name}' with nested value type {values.type}; "
                "extend _column_to_cache_value or exclude the column"
            )
        arr = values.to_numpy(zero_copy_only=False).reshape(len(col), width)
        return _normalize_dtype(torch.from_numpy(_owned_contiguous(arr)))

    if pa.types.is_floating(typ) or pa.types.is_integer(typ) or pa.types.is_boolean(typ):
        arr = col.to_numpy(zero_copy_only=False)
        return _normalize_dtype(torch.from_numpy(_owned_contiguous(arr)))

    raise TypeError(
        f"fast reader cannot cache column '{name}' of arrow type {typ}; "
        "extend _column_to_cache_value or exclude the column"
    )


def _owned_contiguous(arr: np.ndarray) -> np.ndarray:
    """RAM-owned, writable, contiguous copy — never a view onto arrow's
    (possibly memory-mapped, read-only) buffers."""
    out = np.ascontiguousarray(arr)
    if out is arr or not out.flags.writeable:
        out = out.copy()
    return out


def _normalize_dtype(t: torch.Tensor) -> torch.Tensor:
    """Match torch.tensor() inference on Python values: float->f32, int->i64."""
    if t.is_floating_point():
        return t.float()
    if t.dtype == torch.bool:
        return t
    return t.long()


class FastDatasetReader(DatasetReader):
    """DatasetReader with RAM-tensor column cache in the get_item hot path.

    Instances are produced by ``enable_fast_reader`` via ``__class__`` swap on
    an already-loaded stock reader; do not construct directly.
    """

    _tensor_columns: dict[str, torch.Tensor]
    _list_columns: dict[str, list]
    _task_names: list[str]
    _video_pool: ThreadPoolExecutor | None
    _video_pool_pid: int | None
    # Decode-once shared-RAM frame cache (mulligan.data.frame_cache): served-row ->
    # policy-res uint8 frames per video key, plus per-episode served-row ranges.
    # Class-level defaults so pre-attach instances resolve.
    _frame_cache = None
    _frame_cache_ranges = None

    # ── auto-rebuild on table replacement ────────────────────────────────
    # mulligan code replaces reader.hf_dataset wholesale (null-fill, action remaps
    # in mulligan/data/transforms.py). A stale column cache after such an
    # assignment would silently train on pre-remap values, so hf_dataset is a
    # data descriptor here: the setter rebuilds the cache on every assignment.
    # (The property shadows the pre-swap instance-dict entry, which
    # init_column_cache migrates to _hf_dataset_storage.)
    @property
    def hf_dataset(self):
        return self._hf_dataset_storage

    @hf_dataset.setter
    def hf_dataset(self, value) -> None:
        self.__dict__["_hf_dataset_storage"] = value
        if value is not None and "_tensor_columns" in self.__dict__:
            self._rebuild_column_cache()
            if self._frame_cache is not None:
                # Column rewrites (action remaps, null-fill, state widening) keep the
                # row layout and are safe; a replacement that reorders/filters rows
                # would make the frame cache serve WRONG frames — refuse loudly.
                from mulligan.data.frame_cache import _served_episode_ranges

                new_ranges = {e: (f, t) for e, f, t in _served_episode_ranges(self)}
                if new_ranges != self._frame_cache_ranges:
                    raise RuntimeError(
                        "hf_dataset was replaced with a different row layout while a "
                        "decoded frame cache is attached; the cache would serve wrong "
                        "frames. Rebuild/attach the cache AFTER all table rewrites."
                    )

    def init_column_cache(self) -> None:
        if "_hf_dataset_storage" not in self.__dict__:
            # Migrate the stock instance-dict attribute behind the property.
            self.__dict__["_hf_dataset_storage"] = self.__dict__.pop("hf_dataset", None)
        if self.hf_dataset is None:
            raise RuntimeError(
                "fast reader requires a loaded hf_dataset; call load_and_activate first"
            )
        self._rebuild_column_cache()

    def _rebuild_column_cache(self) -> None:
        if getattr(self.hf_dataset, "_indices", None) is not None:
            # .data would be the UNFILTERED table while __getitem__ maps through
            # the indices mapping -> silent row misalignment. Episode filtering in
            # this lerobot version uses parquet predicate pushdown (no _indices),
            # so this should never trigger; fail loudly if it ever does.
            raise RuntimeError(
                "fast reader: hf_dataset has an _indices mapping; the arrow table is not "
                "row-aligned with __getitem__ and caching would corrupt samples"
            )
        table = self.hf_dataset.data
        self._tensor_columns = {}
        self._list_columns = {}
        for name in table.column_names:
            value = _column_to_cache_value(name, table.column(name))
            if isinstance(value, torch.Tensor):
                self._tensor_columns[name] = value
            else:
                self._list_columns[name] = value
        n = len(self.hf_dataset)
        for name, t in self._tensor_columns.items():
            if len(t) != n:
                raise RuntimeError(
                    f"fast reader: column '{name}' has {len(t)} rows, dataset has {n}"
                )
        # .iloc[task_idx].name == index label at position task_idx
        self._task_names = list(self._meta.tasks.index)
        self._video_pool = None
        self._video_pool_pid = None

    # ── pickling: drop the thread pool (rebuilt per PID) ─────────────────
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_video_pool"] = None
        state["_video_pool_pid"] = None
        return state

    def _get_video_pool(self, n_workers: int) -> ThreadPoolExecutor:
        # Rebuild after fork: an inherited pool's worker threads do not survive
        # fork and submits would hang forever.
        pid = os.getpid()
        if self._video_pool is None or self._video_pool_pid != pid:
            self._video_pool = ThreadPoolExecutor(max_workers=n_workers)
            self._video_pool_pid = pid
        return self._video_pool

    # ── hot-path overrides (mirror stock implementations, cache-backed) ──

    def _row_from_cache(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for name, t in self._tensor_columns.items():
            # .clone(): stock torch.tensor(x) returns a FRESH tensor per access;
            # returning a view would let caller-side mutation corrupt the cache.
            item[name] = t[idx].clone()
        for name, values in self._list_columns.items():
            item[name] = values[idx]
        return item

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            result[key] = self._tensor_columns[key][
                torch.tensor(relative_indices, dtype=torch.long)
            ]
        return result

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        ts_col = self._tensor_columns["timestamp"]
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [
                        self._absolute_to_relative_idx[idx] for idx in query_indices[key]
                    ]
                else:
                    relative_indices = query_indices[key]
                query_timestamps[key] = ts_col[
                    torch.tensor(relative_indices, dtype=torch.long)
                ].tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        from lerobot.datasets import dataset_reader as _dr

        if self._frame_cache is not None:
            return self._query_frame_cache(query_timestamps, ep_idx)

        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = _dr.decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
                is_depth=vid_key in self._meta.depth_keys,
            )
            if vid_key in self._meta.depth_keys:
                depth_encoder = self._depth_encoder_configs[vid_key]
                frames = _dr.dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        pool = self._get_video_pool(len(items))
        futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
        return dict(f.result() for f in futures)

    def _query_frame_cache(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        """Serve frames from the decode-once shared-RAM cache (mulligan.data.frame_cache).

        Cached frames are POLICY-RES (crop+resize already applied) uint8; the
        float branch mirrors uint8_to_float01 (in-place div by 255, same op the
        GPU refloat uses). Row resolution mirrors lerobot's torchcodec index
        math (round(ts * fps) within the episode) and is verified against the
        row's own timestamp — a mismatch raises rather than serving a wrong
        frame.
        """
        row_from, row_to = self._frame_cache_ranges[ep_idx]
        ts_col = self._tensor_columns["timestamp"]
        fps = self._meta.fps
        out: dict[str, torch.Tensor] = {}
        for vid_key, query_ts in query_timestamps.items():
            cache = self._frame_cache[vid_key]
            rows = []
            for ts in query_ts:
                row = row_from + int(round(ts * fps))
                if (
                    not (row_from <= row < row_to)
                    or abs(float(ts_col[row]) - ts) > self._tolerance_s
                ):
                    raise ValueError(
                        f"frame-cache row resolution failed: ts={ts} -> row {row} "
                        f"(episode {ep_idx} rows [{row_from}, {row_to})); refusing to "
                        "serve a possibly-wrong cached frame"
                    )
                rows.append(row)
            # .clone() on the single-row path: basic indexing returns a view into
            # the shared cache, and caller-side in-place ops would corrupt it for
            # every worker (advanced indexing below already copies).
            frames = (
                cache[rows[0]].clone()
                if len(rows) == 1
                else cache[torch.tensor(rows, dtype=torch.long)]
            )
            if not self._return_uint8:
                frames = frames.to(torch.float32).div_(255.0)
            out[vid_key] = frames
        return out

    def get_item(self, idx) -> dict:
        from lerobot.datasets import dataset_reader as _dr

        item = self._row_from_cache(idx)
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            for cam in self._meta.camera_keys:
                if cam in self._meta.depth_keys:
                    continue
                item[cam] = self._image_transforms(item[cam])

        for key, stored_unit in self._image_depth_units.items():
            if key in item and stored_unit is not None and stored_unit != self._depth_output_unit:
                item[key] = (
                    item[key] * _dr.MM_PER_METRE
                    if stored_unit == _dr.DEPTH_METER_UNIT
                    else item[key] / _dr.MM_PER_METRE
                )

        task_idx = item["task_index"].item()
        item["task"] = self._task_names[task_idx]

        return item


def enable_fast_reader(dataset) -> FastDatasetReader:
    """Swap a LeRobotDataset's reader to the column-cache fast path.

    Forces reader creation + activation (mirroring ``__getitem__``'s lazy
    load), then swaps ``__class__`` and builds the cache. Idempotent.
    """
    reader = dataset._ensure_reader()
    if reader.hf_dataset is None:
        reader.load_and_activate()
    if isinstance(reader, FastDatasetReader):
        return reader
    reader.__class__ = FastDatasetReader
    reader.init_column_cache()
    return reader


def enable_fast_reader_on_subdatasets(sub_datasets) -> int:
    """Enable the fast reader on every sub-dataset; returns total cached bytes."""
    total_bytes = 0
    for sub in sub_datasets:
        reader = enable_fast_reader(sub)
        total_bytes += sum(t.numel() * t.element_size() for t in reader._tensor_columns.values())
    return total_bytes
