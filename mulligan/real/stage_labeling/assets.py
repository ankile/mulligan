"""Per-episode media + dataset layer for the VLM labeler (genai-free).

Builds everything the labeler feeds to the VLM: per-episode video clips (side +
wrist), final-frame stills, zoomed crops, grasp/release-moment crops, the combo
side-by-side video, the 1 Hz gripper-aperture series, and the proprioceptive
sensor trace. All of it is driven by a :class:`StageLabelTaskSpec` — there are NO
hardcoded camera serials / FPS / gripper threshold here (mixing them up would
silently run one task with another's dataset, which is why the spec is the sole
source — see ``DESIGN.md`` Known limitations).

ffmpeg invocations are built by pure ``_*_cmd`` functions (returning the argv)
and run by thin wrappers, so the exact commands are unit-testable without
invoking ffmpeg. This module imports neither ``google.genai`` (the part-wrapping
lives in :mod:`mulligan.real.stage_labeling.labeler`) nor any experiment script.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.lib

if TYPE_CHECKING:
    from mulligan.real.stage_specs.tasks import StageLabelTaskSpec

# Events-CSV columns the labeler needs from every task (the mandatory contract;
# see DESIGN.md "Per-task data contract"). Standard across tasks, so module-level.
HOLD_TIME_COL = "gripper_hold_time_s"
RELEASE_TIME_COL = "gripper_release_time_s"
REOPENED_COL = "gripper_reopened_at_end"
EPISODE_LENGTH_COL = "episode_length"


# --------------------------------------------------------------------------- #
# Dataset resolution (spec-driven).
# --------------------------------------------------------------------------- #


def repo_file(spec: StageLabelTaskSpec, path: str) -> Path:
    """Download a file from the task's HF dataset at the mutable ``main`` branch.

    ``revision="main"`` (not the ``v3.0`` codebase tag) so in-place dataset edits
    are seen; the codebase tag would pin a stale snapshot.
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(spec.dataset_repo_id, path, repo_type="dataset", revision="main"))


def _dataset_file_sort_key(path: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"data/chunk-(\d+)/file-(\d+)\.parquet", path)
    if match is None:
        return (10**9, 10**9, path)
    return (int(match.group(1)), int(match.group(2)), path)


def frame_parquet_rel_paths(spec: StageLabelTaskSpec) -> list[str]:
    """Actual frame parquet files in the dataset, ordered by chunk/file index.

    Real eval repos can have non-contiguous shard indices when a policy was
    dropped or an interrupted session was resumed. Enumerating the repository
    contents avoids treating the first missing file index as end-of-data.
    """
    from huggingface_hub import list_repo_files

    rel_paths = [
        path
        for path in list_repo_files(spec.dataset_repo_id, repo_type="dataset", revision="main")
        if path.startswith("data/") and path.endswith(".parquet")
    ]
    if not rel_paths:
        raise FileNotFoundError(f"{spec.dataset_repo_id}: no data parquet files found")
    return sorted(rel_paths, key=_dataset_file_sort_key)


def video_rel_path(camera_key: str, file_index: int) -> str:
    return f"videos/{camera_key}/chunk-000/file-{file_index:03d}.mp4"


# --------------------------------------------------------------------------- #
# ffmpeg command builders (pure) + thin runners.
# --------------------------------------------------------------------------- #


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def clip_cmd(source: Path, start_s: float, duration_s: float, out: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.6f}", "-i", str(source), "-t", f"{duration_s:.6f}",
        "-an", "-vf", "scale=640:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]  # fmt: skip


def still_cmd(video: Path, out: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-sseof", "-0.3", "-i", str(video),
        "-frames:v", "1", "-update", "1", str(out),
    ]  # fmt: skip


def crop_cmd(still: Path, out: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(still),
        "-vf", "crop=iw*0.6:ih*0.6,scale=iw*3:ih*3:flags=lanczos", str(out),
    ]  # fmt: skip


def moment_crop_cmd(video: Path, t: float, out: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{t:.2f}", "-i", str(video),
        "-frames:v", "1", "-update", "1",
        # bottom-aligned: the wrist camera has the jaws at the bottom edge.
        "-vf", "crop=iw*0.7:ih*0.7:(iw-ow)/2:ih-oh,scale=iw*3:ih*3:flags=lanczos", str(out),
    ]  # fmt: skip


def montage_crop_cmd(video: Path, times: list[float], out: Path) -> list[str]:
    """Horizontal strip of zoomed wrist stills at ``times`` (a short window around
    the grasp/approach moment), hstacked left-to-right in time order.

    Same bottom-aligned center crop + 3x upscale as :func:`moment_crop_cmd` per
    frame, so the jaws-vs-object region is as sharp as the single-frame crop while
    spanning the approach so the model sees the descent and the close attempt, not
    one frozen instant. One ``-ss/-i`` input per time (seek is cheap; a handful of
    frames), filtered into one row."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for t in times:
        cmd += ["-ss", f"{t:.2f}", "-i", str(video)]
    crop = "crop=iw*0.7:ih*0.7:(iw-ow)/2:ih-oh,scale=iw*3:ih*3:flags=lanczos,setsar=1"
    labels = "".join(f"[{i}:v]{crop}[v{i}];" for i in range(len(times)))
    stack = "".join(f"[v{i}]" for i in range(len(times)))
    cmd += [
        "-filter_complex", f"{labels}{stack}hstack=inputs={len(times)}[v]",
        "-map", "[v]", "-frames:v", "1", "-update", "1", str(out),
    ]  # fmt: skip
    return cmd


def wrist_span_montage_cmd(video: Path, times: list[float], out: Path) -> list[str]:
    """Horizontal strip of FULL wrist frames at ``times`` (the standard wrist view,
    NO zoom / center-crop), hstacked left-to-right in time order.

    Replaces the zoomed bottom-center crop, which (a) zoomed 3x into the centre and
    could crop the nut OUT of frame when it sat off-centre, and (b) was anchored at a
    single EE-z-min moment that was often the wrong instant. Full frames spanning the
    clip keep the whole gripper-vs-nut relationship visible and always include the
    decisive moment. Each frame scaled to a common height; one ``-ss/-i`` per time."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for t in times:
        cmd += ["-ss", f"{t:.2f}", "-i", str(video)]
    scale = "scale=-1:480,setsar=1"
    labels = "".join(f"[{i}:v]{scale}[v{i}];" for i in range(len(times)))
    stack = "".join(f"[v{i}]" for i in range(len(times)))
    cmd += [
        "-filter_complex", f"{labels}{stack}hstack=inputs={len(times)}[v]",
        "-map", "[v]", "-frames:v", "1", "-update", "1", str(out),
    ]  # fmt: skip
    return cmd


def combo_cmd(
    side: Path,
    wrist: Path,
    out: Path,
    side_label: str = "SIDE",
    wrist_label: str = "WRIST",
) -> list[str]:
    # Camera-role drawtext tokens; defaults reproduce the SIDE/WRIST combo byte-for-byte.
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(side), "-i", str(wrist),
        "-filter_complex",
        (
            "[0:v]scale=640:-2,setsar=1,"
            f"drawtext=text={side_label}:x=10:y=10:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.55[v0];"
            "[1:v]scale=640:-2,setsar=1,"
            f"drawtext=text={wrist_label}:x=10:y=10:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.55[v1];"
            "[v0][v1]hstack=inputs=2[v]"
        ),
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]  # fmt: skip


def clip_video(source: Path, start_s: float, duration_s: float, out: Path) -> Path:
    if not out.exists():
        _run(clip_cmd(source, start_s, duration_s, out))
    return out


def episode_video_clip(
    spec: StageLabelTaskSpec, episode_row: pd.Series, camera_key: str, duration_s: float, out: Path
) -> Path:
    file_index = int(episode_row[f"videos/{camera_key}/file_index"])
    start_s = float(episode_row[f"videos/{camera_key}/from_timestamp"])
    source = repo_file(spec, video_rel_path(camera_key, file_index))
    return clip_video(source, start_s, duration_s, out)


def final_frame_still(video: Path, out: Path) -> Path:
    if not out.exists():
        _run(still_cmd(video, out))
    return out


def final_frame_crop(still: Path, out: Path) -> Path:
    if not out.exists():
        _run(crop_cmd(still, out))
    return out


def grasp_moment_crop(
    video: Path, jaw_close_time_s: float, duration_s: float, out: Path, offset_s: float = 1.5
) -> Path:
    """Zoomed wrist still ``offset_s`` after the proprioceptive jaw close.

    ``jaw_close_time_s`` comes from proprioception (not the model); the time is
    clamped into the clip. Two offsets are used per episode (+0.5s transient
    window, +1.5s settled grasp) plus a release-moment crop at the reopen.
    """
    if not out.exists():
        t = min(jaw_close_time_s + offset_s, max(duration_s - 0.5, 0.0))
        _run(moment_crop_cmd(video, t, out))
    return out


def grasp_window_crop(video: Path, duration_s: float, out: Path, frames: int = 6) -> Path:
    """Full-frame wrist montage spanning the WHOLE episode (the standard wrist view
    at ``frames`` evenly-spaced moments).

    Spanning the clip with full frames keeps the decisive grasp moment in view and never
    crops away the gripper-vs-nut relationship."""
    if not out.exists():
        hi = max(duration_s - 0.2, 0.0)
        if frames <= 1 or hi <= 0.0:
            times = [hi / 2.0]
        else:
            times = [hi * (0.08 + 0.92 * i / (frames - 1)) for i in range(frames)]
        _run(wrist_span_montage_cmd(video, times, out))
    return out


def local_wrist_window_crop(
    video: Path,
    center_t: float,
    duration_s: float,
    out: Path,
    *,
    span_s: float = 0.8,
    frames: int = 5,
) -> Path:
    """Zoomed wrist strip centered on an operator/model-selected timestamp."""
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    if span_s < 0:
        raise ValueError(f"span_s must be nonnegative, got {span_s}")
    if not out.exists():
        max_t = max(duration_s - 0.1, 0.0)
        lo = max(0.0, center_t - span_s / 2.0)
        hi = min(max_t, center_t + span_s / 2.0)
        if frames == 1 or hi <= lo:
            times = [min(max(center_t, 0.0), max_t)]
        else:
            times = [lo + (hi - lo) * i / (frames - 1) for i in range(frames)]
        _run(montage_crop_cmd(video, times, out))
    return out


def full_frame_window_montage(
    video: Path,
    center_t: float,
    duration_s: float,
    out: Path,
    *,
    span_s: float = 1.2,
    frames: int = 5,
) -> Path:
    """Full-frame strip around a local event, preserving the whole camera view."""
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    if span_s < 0:
        raise ValueError(f"span_s must be nonnegative, got {span_s}")
    if not out.exists():
        max_t = max(duration_s - 0.1, 0.0)
        lo = max(0.0, center_t - span_s / 2.0)
        hi = min(max_t, center_t + span_s / 2.0)
        if frames == 1 or hi <= lo:
            times = [min(max(center_t, 0.0), max_t)]
        else:
            times = [lo + (hi - lo) * i / (frames - 1) for i in range(frames)]
        _run(wrist_span_montage_cmd(video, times, out))
    return out


def combo_video(
    side: Path,
    wrist: Path,
    out: Path,
    side_label: str = "SIDE",
    wrist_label: str = "WRIST",
) -> Path:
    if not out.exists():
        if not side.exists():
            raise FileNotFoundError(side)
        if not wrist.exists():
            raise FileNotFoundError(wrist)
        _run(combo_cmd(side, wrist, out, side_label, wrist_label))
    return out


# --------------------------------------------------------------------------- #
# Proprioception (pure where possible).
# --------------------------------------------------------------------------- #


def last_close_time_s(series_1hz: list[float], threshold: float) -> float | None:
    """Time of the LAST jaw close (rising ``threshold``-crossing) in the 1 Hz
    aperture series; matches the events-pipeline crossing rule. Drop-and-regrasp
    episodes have several closes — the last one is where the held grasp is."""
    last = None
    for i in range(1, len(series_1hz)):
        if float(series_1hz[i - 1]) < threshold <= float(series_1hz[i]):
            last = float(i)
    return last


def sensor_trace(spec: StageLabelTaskSpec, event_row: pd.Series) -> dict[str, Any]:
    """Proprioceptive trace from the events row: jaw close/reopen times, the
    reopened-before-end flag, and episode duration (``length / spec.fps`` —
    NOT a hardcoded 15.0)."""
    hold = event_row[HOLD_TIME_COL]
    release = event_row[RELEASE_TIME_COL]
    return {
        "jaw_close_time_s": None if pd.isna(hold) else round(float(hold), 2),
        "jaw_reopen_time_s": None if pd.isna(release) else round(float(release), 2),
        "jaws_reopened_before_episode_end": bool(event_row[REOPENED_COL]),
        "episode_duration_s": round(float(event_row[EPISODE_LENGTH_COL]) / spec.fps, 2),
    }


def read_frame_parquet(path: Path, value_columns: list[str]) -> pd.DataFrame:
    """Read the standard frame columns (+ ``value_columns``) from one frame parquet.

    Requests ``is_valid`` and retries without it for pre-outcome-editor datasets
    that lack the column (pyarrow raises ArrowInvalid on a missing column)."""
    columns = ["episode_index", "frame_index", *value_columns, "is_valid"]
    try:
        return pd.read_parquet(path, columns=columns)
    except pyarrow.lib.ArrowInvalid as exc:
        if "is_valid" not in str(exc):
            raise
        return pd.read_parquet(path, columns=columns[:-1])


def valid_frame_prefix(frames: pd.DataFrame, *, episode_index: int) -> pd.DataFrame:
    """Return the contiguous ``is_valid=True`` prefix for one episode.

    Outcome-editor soft truncation marks post-freeze timeout tails with
    ``is_valid=0``. Stage events should follow the same training/data contract as
    downstream dataset loaders: ignore invalid rows, but fail if the mask is not a
    valid prefix.
    """
    frames = frames.sort_values("frame_index").reset_index(drop=True)
    if "is_valid" not in frames.columns:
        return frames
    valid_mask = frames["is_valid"].astype(bool)
    seen_invalid = False
    for is_valid in valid_mask.tolist():
        if not is_valid:
            seen_invalid = True
        elif seen_invalid:
            raise RuntimeError(
                f"episode {episode_index}: is_valid is not a valid prefix followed by padding"
            )
    valid = frames[valid_mask].copy()
    if valid.empty:
        raise RuntimeError(f"episode {episode_index}: no valid frames")
    return valid


def gripper_series_1hz(spec: StageLabelTaskSpec, build_dir: Path) -> dict[int, list[float]]:
    """1 Hz aperture series per episode (0=open, ~0.8=object in jaws, ~0.99=closed
    on air), cached to JSON. Reads the dataset's frame parquets via the spec's
    gripper column and bins at ``int(spec.fps)``."""
    cache = build_dir / "gripper_series_1hz.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    col = spec.gripper_state_column
    frames = [
        read_frame_parquet(repo_file(spec, rel_path), [col])
        for rel_path in frame_parquet_rel_paths(spec)
    ]
    table = pd.concat(frames, ignore_index=True)
    step = int(spec.fps)
    series: dict[int, list[float]] = {}
    for episode, group in table.groupby("episode_index"):
        valid = valid_frame_prefix(group, episode_index=int(episode))
        g = valid.sort_values("frame_index")[col].to_numpy()
        series[int(episode)] = [
            round(float(g[i : i + step].mean()), 2) for i in range(0, len(g), step)
        ]
    build_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(series))
    return series


def valid_frame_counts(spec: StageLabelTaskSpec, build_dir: Path) -> dict[int, int]:
    """Per-episode count of the contiguous ``is_valid=True`` frame prefix."""
    cache = build_dir / "valid_frame_counts.json"
    if cache.exists():
        return {int(k): int(v) for k, v in json.loads(cache.read_text()).items()}
    frames = [
        read_frame_parquet(repo_file(spec, rel_path), [])
        for rel_path in frame_parquet_rel_paths(spec)
    ]
    table = pd.concat(frames, ignore_index=True)
    counts: dict[int, int] = {}
    for episode, group in table.groupby("episode_index"):
        valid = valid_frame_prefix(group, episode_index=int(episode))
        counts[int(episode)] = int(len(valid))
    build_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(counts))
    return counts


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def _build_manifest(spec: StageLabelTaskSpec) -> dict[str, Any]:
    """The dataset/camera identity an assets dir was built against.

    Only fields the cached ASSETS (videos/stills/crops/1 Hz series) depend on —
    deliberately NOT events_csv: the events CSV feeds sensor_trace, not the
    assets, and its absolute path is worktree-dependent (an over-broad manifest
    wrongly rejected a valid cache reused from another worktree)."""
    return {
        "task": spec.name,
        "dataset_repo_id": spec.dataset_repo_id,
        "fps": spec.fps,
        "side_camera_key": spec.side_camera_key,
        "wrist_camera_key": spec.wrist_camera_key,
        "gripper_state_column": spec.gripper_state_column,
        "frame_parquet_enumeration_version": 2,
        "valid_prefix_cache_version": 2,
        "events_length_guard_version": 1,
    }


def task_build_dir(spec: StageLabelTaskSpec, build_dir: Path) -> Path:
    """Per-task subdir of ``build_dir``, so two tasks never collide in a shared
    build dir (cached ``episode_000_side.mp4`` / gripper JSON are filename-keyed
    by existence, so a shared dir would silently feed one task another's assets)."""
    return build_dir / spec.name


def _verify_build_manifest(spec: StageLabelTaskSpec, task_dir: Path) -> None:
    """Fail loud if ``task_dir`` was built against a different dataset/camera/FPS
    config (e.g. the dataset repo was re-pointed) — stale cached assets would
    otherwise be reused silently."""
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / "build_manifest.json"
    current = _build_manifest(spec)
    if path.exists():
        prev = json.loads(path.read_text())
        if prev != current:
            raise RuntimeError(
                f"{task_dir} was built for a different config; clear it or use a fresh "
                f"build_dir.\n  cached: {prev}\n  current: {current}"
            )
    else:
        path.write_text(json.dumps(current, indent=2))


def _asset_lengths_path(task_dir: Path) -> Path:
    return task_dir / "asset_episode_lengths.json"


def _load_asset_lengths(task_dir: Path) -> dict[int, int]:
    path = _asset_lengths_path(task_dir)
    if not path.exists():
        return {}
    return {int(k): int(v) for k, v in json.loads(path.read_text()).items()}


def _write_asset_lengths(task_dir: Path, lengths: dict[int, int]) -> None:
    path = _asset_lengths_path(task_dir)
    payload = {str(k): int(v) for k, v in sorted(lengths.items())}
    path.write_text(json.dumps(payload, indent=2))


def _clear_episode_assets(assets_dir: Path, episode: int) -> None:
    for path in assets_dir.glob(f"episode_{episode:03d}_*"):
        if path.is_file():
            path.unlink()


def _episode_gripper_series(
    series_1hz: list[float], episode_length: int, fps: float
) -> list[float]:
    bins = max(1, int(math.ceil(episode_length / fps)))
    return series_1hz[:bins]


def build_items(
    spec: StageLabelTaskSpec,
    build_dir: Path,
    episodes: set[int] | None = None,
    *,
    allow_invalid_tail_for_reference_media: bool = False,
) -> list[dict[str, Any]]:
    """Build per-episode clips/stills/crops + blind metadata for ``spec``.

    Returns one item dict per episode (sorted), each carrying the asset paths,
    the proprioceptive ``sensor_trace``, and the 1 Hz gripper series. Camera
    keys, FPS, and the gripper threshold all come from ``spec``; assets are
    cached under a per-task subdir guarded by a config manifest.
    """
    if spec.events_csv is None:
        raise ValueError(
            f"{spec.name}: no events CSV; build one with mulligan.real.stage_labeling.events and "
            "pass it (dataclasses.replace(spec, events_csv=...) or the pipeline's --events-csv)"
        )
    events = pd.read_csv(spec.events_csv)
    episode_meta = pd.read_parquet(
        repo_file(spec, "meta/episodes/chunk-000/file-000.parquet")
    ).set_index("episode_index")
    task_dir = task_build_dir(spec, build_dir)
    _verify_build_manifest(spec, task_dir)
    assets_dir = task_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_lengths = _load_asset_lengths(task_dir)
    asset_lengths_changed = False
    gripper_series = gripper_series_1hz(spec, task_dir)
    valid_counts = valid_frame_counts(spec, task_dir)
    side_key, wrist_key = spec.side_camera_key, spec.wrist_camera_key

    items: list[dict[str, Any]] = []
    for row in events.sort_values("episode_index").itertuples(index=False):
        row_series = pd.Series(row._asdict())
        episode = int(row_series["episode_index"])
        if episodes is not None and episode not in episodes:
            continue
        meta_row = episode_meta.loc[episode]
        episode_length = int(row_series[EPISODE_LENGTH_COL])
        if episode not in valid_counts:
            raise KeyError(f"valid frame counts missing episode {episode}")
        valid_count = valid_counts[episode]
        if allow_invalid_tail_for_reference_media:
            physical_length = int(meta_row["length"])
            if episode_length > physical_length:
                raise RuntimeError(
                    f"{spec.events_csv}: episode {episode} episode_length={episode_length} "
                    f"exceeds physical dataset length {physical_length}"
                )
        if episode_length > valid_count and not allow_invalid_tail_for_reference_media:
            raise RuntimeError(
                f"{spec.events_csv}: episode {episode} episode_length={episode_length} "
                f"but dataset is_valid prefix has {valid_count} frames. Regenerate the events CSV "
                "from the outcome-edited dataset before building labeler assets."
            )
        if asset_lengths.get(episode) != episode_length:
            _clear_episode_assets(assets_dir, episode)
            asset_lengths[episode] = episode_length
            asset_lengths_changed = True
        duration_s = episode_length / spec.fps

        side_name = f"episode_{episode:03d}_side.mp4"
        wrist_name = f"episode_{episode:03d}_wrist.mp4"
        side_path = assets_dir / side_name
        wrist_path = assets_dir / wrist_name
        episode_video_clip(spec, meta_row, side_key, duration_s, side_path)
        episode_video_clip(spec, meta_row, wrist_key, duration_s, wrist_path)

        side_final = final_frame_still(
            side_path, assets_dir / f"episode_{episode:03d}_side_final.png"
        )
        wrist_final = final_frame_still(
            wrist_path, assets_dir / f"episode_{episode:03d}_wrist_final.png"
        )
        episode_gripper_series = _episode_gripper_series(
            gripper_series[episode], episode_length, spec.fps
        )
        item: dict[str, Any] = {
            "episode_index": episode,
            "duration_s": duration_s,
            "side_video": side_name,
            "wrist_video": wrist_name,
            "side_path": str(side_path),
            "wrist_path": str(wrist_path),
            "side_final_frame": str(side_final),
            "wrist_final_frame": str(wrist_final),
            "side_final_crop": str(
                final_frame_crop(
                    side_final, assets_dir / f"episode_{episode:03d}_side_final_crop.png"
                )
            ),
            "wrist_final_crop": str(
                final_frame_crop(
                    wrist_final, assets_dir / f"episode_{episode:03d}_wrist_final_crop.png"
                )
            ),
            "sensor_trace": sensor_trace(spec, row_series),
            "gripper_series_1hz": episode_gripper_series,
        }

        # Optional extra camera streams (e.g. routing's wrist_left as a 3rd view).
        # Each is a video + final frame with its role label; empty by default so
        # marker/square (side+wrist only) build exactly as before.
        if spec.extra_camera_keys:
            extra_paths, extra_finals = [], []
            for k, extra_key in enumerate(spec.extra_camera_keys):
                extra_path = assets_dir / f"episode_{episode:03d}_extra{k}.mp4"
                episode_video_clip(spec, meta_row, extra_key, duration_s, extra_path)
                extra_finals.append(
                    str(
                        final_frame_still(
                            extra_path, assets_dir / f"episode_{episode:03d}_extra{k}_final.png"
                        )
                    )
                )
                extra_paths.append(str(extra_path))
            item["extra_paths"] = extra_paths
            item["extra_final_frames"] = extra_finals
            item["extra_labels"] = list(spec.extra_camera_labels)

        # Grasp/release crops keyed off the LAST proprioceptive close, taken from
        # the wrist view (the jaws are off-frame in a side-view center crop).
        last_close = last_close_time_s(episode_gripper_series, spec.gripper_close_threshold)
        if last_close is None:
            last_close = item["sensor_trace"]["jaw_close_time_s"]
        if last_close is not None:
            item["grasp_moment_crop"] = str(
                grasp_moment_crop(
                    wrist_path,
                    float(last_close),
                    duration_s,
                    assets_dir / f"episode_{episode:03d}_grasp_crop.png",
                )
            )
            item["grasp_early_crop"] = str(
                grasp_moment_crop(
                    wrist_path,
                    float(last_close),
                    duration_s,
                    assets_dir / f"episode_{episode:03d}_grasp_crop_early.png",
                    offset_s=0.5,
                )
            )
        else:
            item["grasp_moment_crop"] = None
            item["grasp_early_crop"] = None

        # Grasp-window montage over the whole episode: the grasp evidence for the
        # failed-grasp S0/S1 episodes, where jaw_close_time_s is NaN and the final-frame
        # crops show the nut on the table either way.
        item["grasp_window_crop"] = str(
            grasp_window_crop(
                wrist_path,
                duration_s,
                assets_dir / f"episode_{episode:03d}_grasp_window_crop.png",
            )
        )

        reopen = item["sensor_trace"]["jaw_reopen_time_s"]
        item["release_moment_crop"] = (
            str(
                grasp_moment_crop(
                    wrist_path,
                    max(float(reopen) - 0.3, 0.0),
                    duration_s,
                    assets_dir / f"episode_{episode:03d}_release_crop.png",
                    offset_s=0.0,
                )
            )
            if reopen is not None
            else None
        )
        # NB: the combo (hstack) asset is only consumed in input_mode="combo"; the
        # active tasks (marker/square/routing) label in "dual" mode, so its SIDE/WRIST
        # drawtext never reaches a Gemini call. combo_cmd/combo_video ARE parameterized
        # (default SIDE/WRIST, overridable) for a future combo-mode task, but the
        # per-task labels are intentionally NOT threaded here so the build_items
        # combo_video contract (and its stubs) stays stable.
        item["combo_path"] = str(
            combo_video(side_path, wrist_path, assets_dir / f"episode_{episode:03d}_combo.mp4")
        )
        items.append(item)
    if asset_lengths_changed:
        _write_asset_lengths(task_dir, asset_lengths)
    return items
