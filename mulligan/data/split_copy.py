"""Per-frame LeRobot episode copy shared by the dataset splitters.

:mod:`mulligan.data.split_blind`, :mod:`mulligan.data.split_protocol_quota` and
:mod:`mulligan.real.data.split` choose which source episodes go to which target; the
helpers here create the targets and copy episodes frame by frame (``add_frame`` /
``save_episode``). The file-level copy lives in :mod:`mulligan.tools.lerobot_fast_split`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

VISUAL_DTYPES = {"image", "video"}
# Written by LeRobot itself on add_frame/save_episode; never copied from the source.
AUTO_FRAME_KEYS = {"episode_index", "frame_index", "index", "timestamp", "task_index"}
NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": bool,
}


def without_visual_features(features: dict) -> dict:
    return {
        key: value for key, value in features.items() if value.get("dtype") not in VISUAL_DTYPES
    }


def adapt_value(val, feature_meta):
    """Convert a source frame value to what ``add_frame`` expects for ``feature_meta``."""
    if hasattr(val, "numpy"):
        val = val.numpy()
    val = np.asarray(val)
    dtype = feature_meta.get("dtype")
    if dtype in VISUAL_DTYPES and val.ndim == 3 and val.shape[0] in (1, 3):
        val = np.transpose(val, (1, 2, 0))
        if val.dtype != np.uint8:
            val = (
                (val * 255.0).clip(0, 255).astype(np.uint8)
                if val.max() <= 1.0
                else val.astype(np.uint8)
            )
        return val
    if val.ndim == 0:
        val = np.array([val.item()])
    target_dtype = NUMPY_DTYPE_MAP.get(dtype)
    if target_dtype is not None and val.dtype != target_dtype:
        val = val.astype(target_dtype)
    return val


def task_from_raw_item(source: LeRobotDataset, item: dict) -> str:
    task_idx = int(np.asarray(item["task_index"]).item())
    return source.meta.tasks.iloc[task_idx].name


# Reads full rows on purpose: --drop-visual-features copies every non-visual column of each kept
# frame, and the global-index lookup guards against stale rows from resumed collections.
def build_frame_lookup_by_index(source: LeRobotDataset) -> dict[int, dict]:
    """Return canonical non-visual frame rows keyed by global frame index.

    Some resumed collections can leave stale rows in an earlier parquet file at a
    file boundary. The episode metadata is expressed in global frame indices, so
    copying by Hugging Face row position can silently read the stale rows. When
    duplicate global indices exist, later parquet rows overwrite earlier rows;
    this matches the resumed-file layout where the new file is the canonical one.
    """
    lookup: dict[int, dict] = {}
    duplicates = 0
    for item in source.hf_dataset:
        idx = int(np.asarray(item["index"]).item())
        if idx in lookup:
            duplicates += 1
        lookup[idx] = item

    missing = sorted(set(range(source.num_frames)) - set(lookup))
    if missing:
        raise RuntimeError(
            f"Source dataset is missing {len(missing)} global frame indices; "
            f"first missing indices: {missing[:10]}"
        )
    if duplicates:
        print(
            f"WARNING: source dataset has {duplicates} duplicate global frame rows; "
            "using the last row for each duplicate index"
        )
    return lookup


def frame_at(
    source: LeRobotDataset,
    frame_idx: int,
    *,
    frame_lookup: dict[int, dict] | None,
) -> dict:
    """Non-visual row ``frame_idx``: from ``frame_lookup`` if given, else by row position."""
    return frame_lookup[frame_idx] if frame_lookup is not None else source.hf_dataset[frame_idx]


def last_frame_success(
    source: LeRobotDataset,
    ep_idx: int,
    *,
    frame_lookup: dict[int, dict] | None,
) -> bool:
    end = int(source.meta.episodes[ep_idx]["dataset_to_index"])
    return int(frame_at(source, end - 1, frame_lookup=frame_lookup)["success"]) == 1


def target_features(template: LeRobotDataset, *, drop_visual_features: bool) -> dict:
    return without_visual_features(template.features) if drop_visual_features else template.features


def create_target_dataset(
    template: LeRobotDataset,
    repo_id: str,
    root: Path,
    *,
    drop_visual_features: bool,
    features: dict | None = None,
    default_robot_type: str = "panda",
) -> LeRobotDataset:
    """New empty dataset with ``template``'s fps and robot type.

    ``features`` defaults to :func:`target_features`; ``default_robot_type`` is used only
    when the template metadata has no ``robot_type`` entry.
    """
    if features is None:
        features = target_features(template, drop_visual_features=drop_visual_features)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=template.fps,
        root=str(root),
        robot_type=template.meta.info.get("robot_type", default_robot_type),
        features=features,
        use_videos=not drop_visual_features,
    )


def copy_episode(
    source: LeRobotDataset,
    ep_idx: int,
    target: LeRobotDataset,
    *,
    drop_visual_features: bool,
    frame_lookup: dict[int, dict] | None,
    zero_missing_intervention: bool = False,
) -> None:
    """Append source episode ``ep_idx`` to ``target`` and save it.

    Visual copies read decoded frames (``source[frame_idx]``); non-visual copies read raw
    rows through :func:`frame_at`. With ``zero_missing_intervention``, a target
    ``intervention`` feature that the source frame lacks is written as 0.
    """
    ep_meta = source.meta.episodes[ep_idx]
    start = int(ep_meta["dataset_from_index"])
    end = int(ep_meta["dataset_to_index"])
    for frame_idx in range(start, end):
        item = (
            frame_at(source, frame_idx, frame_lookup=frame_lookup)
            if drop_visual_features
            else source[frame_idx]
        )
        frame = {}
        for key, fmeta in target.features.items():
            if key in AUTO_FRAME_KEYS:
                continue
            if zero_missing_intervention and key == "intervention" and key not in item:
                frame[key] = np.array([0], dtype=np.int64)
            else:
                frame[key] = adapt_value(item[key], fmeta)
        frame["task"] = task_from_raw_item(source, item) if drop_visual_features else item["task"]
        target.add_frame(frame)
    target.save_episode()
