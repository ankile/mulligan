"""
Outcome review for LeRobot real-robot datasets: a cv2 editor and a headless apply.

Interactive (cv2): scrub through episodes, change their outcome label
(success / failure / timeout), mark the exact frame where the outcome occurs,
and set reward/done from that frame onward. By default, all existing valid
frames remain valid. Soft truncation is optional and sets is_valid=0 after
the marked outcome frame. Decisions are recorded in the dataset's
``.outcome_edit_progress.json`` (and one event per decision in
``.label_history.jsonl``); ``--push`` uploads the edited frames, the refreshed
episode stats, the progress record and the reconciled ``results.json``.

Headless (``--apply-overlay decisions.json``): merge review decisions captured
elsewhere (same shape as the progress record:
``{"changed_episodes": {"<idx>": {"new_outcome", "outcome_frame", "soft_truncate",
"subtask_frames"?}}, "skipped_episodes": [...]}``) and run the same apply path
without a display.

Subtask reward marks (sub-goal budget N, resolved from the task config
``RealTaskSpec.num_subtask_marks``; overridable via ``--subtask-marks``): in
addition to the terminal outcome, the operator marks mid-episode sub-goal frames
(e.g. the frame where "rope seats in the first clip" completes). Each subtask
mark writes ``reward=1.0`` at exactly that frame while leaving ``done=0`` and
``is_valid=1`` there, producing a single-frame reward spike that downstream
Monte-Carlo / k-step reward consumers pick up. With ``N=0`` the subtask machinery
is fully inert and the editor writes only the terminal outcome.
When ``N>0`` the required mark count is OUTCOME-AWARE (see
:func:`subtask_mark_count_error`): a ``success`` reached every sub-goal so it must
carry exactly N marks (the final sub-goal is the terminal success), while a
``failure``/``timeout`` reached some prefix so it carries 0..N (e.g. routing:
0 = never seated a clip, 1 = seated the first clip then fumbled). Every subtask
frame must be strictly before a success/failure outcome frame. A timeout has no
terminal transition, so its final valid frame may also carry a subtask mark
(e.g. the sub-goal completed immediately before the episode ended). This keeps
the mark valid once the edit re-validates the episode; soft truncation only
invalidates frames after the outcome frame. Subtask frames are stored
per-episode in the progress dotfile under ``subtask_frames`` (deduplicated +
sorted) and round-trip on resume.
When ``N>0``, episodes that already carry a terminal ``done==1`` plateau in the
data (e.g. collection-time success labels) enter with the outcome mark PREFILLED
at that existing onset, so adding only the subtask mark(s) and confirming
(g -> c) rewrites the terminal labels to identical values — the existing
end-of-episode plateau is not disturbed unless you re-mark it with m.
An episode counts as REVIEWED once confirmed in subtask mode, which stamps a
``subtask_frames`` key (possibly empty for a 0-mark failure). Episodes recorded as
changed by an N=0 session lack that key and RE-ENTER the work queue to be
reviewed (terminal outcome/frame resumes from the existing record); re-applying
the progress file tolerates such records without the key —
terminal edits applied, no spike — while a reviewed record with an
outcome-illegal mark count still fails loud.

Usage:
    # Review failure episodes
    python -m mulligan.tools.outcome_review \
        --repo-id <user>/my-dataset --filter failure --push

    # Start each episode with soft truncation enabled
    python -m mulligan.tools.outcome_review --repo-id <user>/my-dataset --soft-truncate --push

    # Review only one arm/policy in a blinded aggregate dataset
    python -m mulligan.tools.outcome_review \
        --repo-id <user>/my-blind-dagger-dataset --arm-key mulligan_sobol --filter all --push

    # Apply decisions without a display
    python -m mulligan.tools.outcome_review \
        --repo-id <user>/my-dataset --apply-overlay decisions.json --push

Controls:
    Left/Right arrow  Step one frame back/forward
    [ / ]             Jump 10 frames back/forward
    Space             Play/pause
    s                 Change pending outcome -> SUCCESS
    f                 Change pending outcome -> FAILURE (terminal)
    t                 Change pending outcome -> TIMEOUT (non-terminal)
    m or Enter        Mark current frame as outcome frame
    g                 Toggle a subtask reward mark at the current frame
                      (only active with --subtask-marks N > 0; appends while
                      under N marks, removes the mark if the current frame is
                      already marked)
    x                 Toggle soft truncation for this episode
    u                 Unmark outcome frame, clear pending outcome, and clear all
                      subtask marks for this episode
    c                 Confirm -> apply changes, save progress, next episode
    n                 Skip episode (no changes), next episode
    p or b            Go back to previous episode
    q                 Quit and save progress
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.real.robot.cameras import (
    STATION_CAMERA_DEFAULT_CROPS,
    STATION_CAMERA_KEYS_BY_ROLE,
)
from mulligan.real.collect.hf_utils import DEFAULT_DATASET_LICENSE, add_license_arg
from mulligan.real.lifecycle import find_task_spec_by_task_name
from mulligan.real.eval.outcome_results import (
    _detect_frame_outcome,
    canonicalize_results_file,
    subtask_frames_for_validation,
    subtask_mark_count_error,
)
from mulligan.tools.lerobot_hub import (
    advance_lerobot_version_tag,
    refresh_lerobot_dataset_from_main,
)

PROGRESS_FILENAME = ".outcome_edit_progress.json"
WINDOW_NAME = "Outcome Editor"

# Outcome types and their reward/done semantics
OUTCOME_TYPES = {
    "success": {"success": 1, "reward": 1.0, "done": 1},
    "failure": {"success": 0, "reward": 0.0, "done": 1},
    "timeout": {"success": 0, "reward": 0.0, "done": 0},
}

# Frame-level columns rewritten by outcome edits. Their per-episode `stats/...`
# columns in meta/episodes parquet and the global meta/stats.json are computed at
# collection time and are NOT touched by LeRobotDataset.push_to_hub(), so they must
# be refreshed here after every edit or metadata-based consumers read stale outcomes.
EDIT_AFFECTED_FEATURES = ("success", "reward", "done", "is_valid")

# Colors (BGR) for each outcome type
OUTCOME_COLORS = {
    "success": (0, 200, 0),  # green
    "failure": (0, 0, 200),  # red
    "timeout": (0, 200, 200),  # yellow
}

# Subtask reward marks use a distinct magenta so a single-frame sub-goal spike
# never reads as the translucent full-frame outcome band above.
SUBTASK_MARK_COLOR = (255, 0, 255)  # magenta (BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outcome review for LeRobot real-robot datasets")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace Hub dataset ID")
    parser.add_argument(
        "--apply-overlay",
        type=Path,
        default=None,
        metavar="DECISIONS_JSON",
        help=(
            "Headless: merge these review decisions (progress-record shape) into the dataset's "
            ".outcome_edit_progress.json and apply them without opening the editor."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "HF revision to load for mutable real datasets (default: main). "
            "LeRobot's library default is the codebase version tag, which can lag behind label edits."
        ),
    )
    parser.add_argument(
        "--no-cache-refresh",
        dest="force_cache_sync",
        action="store_false",
        help="Use the existing local LeRobot cache instead of refreshing files from the selected revision.",
    )
    parser.set_defaults(force_cache_sync=True)
    parser.add_argument(
        "--filter",
        choices=["success", "failure", "timeout", "all"],
        default="all",
        help="Which episodes to review (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=_parse_episode_indices,
        default=None,
        help=(
            "Restrict to these comma-separated episode indices — a spot-check of "
            "specific episodes (e.g. reconciliation-flagged ones), combined with "
            "--filter. Fails loud if an index is absent. RE-OPENS the listed episodes "
            "even if a prior session already confirmed them (resuming from their "
            "existing record), so it's the way to redo a finished episode. "
            "Example: --episodes 60,65,67"
        ),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated video keys to show side-by-side (default: auto-detect)",
    )
    parser.add_argument(
        "--push", action="store_true", help="Push modified dataset back to Hub after editing"
    )
    add_license_arg(parser)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Display scale factor (default: 1.0)"
    )
    parser.add_argument(
        "--soft-truncate",
        action="store_true",
        help="Default to setting is_valid=0 on frames after the marked outcome frame",
    )
    parser.add_argument(
        "--subtask-marks",
        type=int,
        default=None,
        help=(
            "Override the number of mid-episode subtask reward marks required per changed "
            "episode. By default (unset) this is resolved from the dataset's task config "
            "(RealTaskSpec.num_subtask_marks) so the operator never has to remember it — "
            "e.g. routing_d2 resolves to 1, marker/square to 0. Pass this only to override "
            "the task default (e.g. an unregistered/ad-hoc dataset). 0 disables the subtask "
            "machinery and reproduces the single-terminal-mark editor exactly. When > 0, "
            "press 'g' to toggle a subtask mark; the required count is outcome-aware — a "
            "success needs exactly N marks, a failure/timeout 0..N. Marks must precede "
            "a success/failure outcome; a timeout mark may equal the final valid-frame "
            "boundary. Each mark writes reward=1.0 there (done=0, is_valid=1)."
        ),
    )
    parser.add_argument(
        "--arm-key",
        action="append",
        default=[],
        help=(
            "Only review episodes whose ledger arm_key, arm, or manifest_source matches this value. "
            "Can be repeated, e.g. --arm-key mulligan_sobol --arm-key mulligan."
        ),
    )
    parser.add_argument(
        "--policy-id",
        action="append",
        default=[],
        help="Only review episodes whose parquet/ledger policy_id matches this value. Can be repeated.",
    )
    parser.add_argument(
        "--arm-id",
        action="append",
        default=[],
        help="Only review episodes whose parquet/ledger arm_id matches this value. Can be repeated.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Only review episodes whose ledger model_id matches this value. Can be repeated.",
    )
    parser.add_argument(
        "--episode-filter",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=(
            "Extra exact episode filter against ledger fields or parquet columns. "
            "Can be repeated; all filters must match."
        ),
    )
    parser.add_argument(
        "--ledger",
        action="append",
        type=Path,
        default=[],
        help=(
            "JSONL ledger used for arm/model/episode filters. Default auto-loads known ledgers "
            "from dataset meta when present."
        ),
    )
    args = parser.parse_args()
    if args.subtask_marks is not None and args.subtask_marks < 0:
        parser.error(f"--subtask-marks must be >= 0, got {args.subtask_marks}")
    return args


@dataclass(frozen=True)
class EpisodeFieldFilter:
    fields: tuple[str, ...]
    values: frozenset[str]
    label: str


@dataclass(frozen=True)
class EpisodeSelector:
    filters: tuple[EpisodeFieldFilter, ...]
    ledger_rows_by_episode: dict[int, dict[str, Any]]

    @property
    def enabled(self) -> bool:
        return bool(self.filters)

    @property
    def description(self) -> str:
        if not self.filters:
            return "none"
        return ", ".join(f.label for f in self.filters)

    def matches(self, ep_idx: int, ep_data: pd.DataFrame) -> bool:
        for field_filter in self.filters:
            if not _episode_matches_filter(
                ep_idx,
                ep_data,
                self.ledger_rows_by_episode,
                field_filter,
            ):
                return False
        return True


def auto_detect_cameras(video_keys: list[str]) -> list[str]:
    """Pick cameras to display: prefer '_left' suffix, else first two."""
    left_cams = [k for k in video_keys if k.endswith("_left")]
    if len(left_cams) >= 2:
        return left_cams[:2]
    if left_cams:
        return left_cams[:1] + [k for k in video_keys if k not in left_cams][:1]
    return video_keys[:2]


def load_lerobot_dataset(
    repo_id: str,
    *,
    download_videos: bool,
    args: argparse.Namespace,
) -> LeRobotDataset:
    if args.force_cache_sync and args.revision == "main":
        return refresh_lerobot_dataset_from_main(
            repo_id,
            download_videos=download_videos,
            revision=args.revision,
        )
    return LeRobotDataset(
        repo_id=repo_id,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=download_videos,
    )


def _info_value(dataset: LeRobotDataset, key: str) -> Any:
    """Read ``dataset.meta.info`` across LeRobot dict/dataclass variants."""
    info = dataset.meta.info
    if isinstance(info, dict):
        return info[key]
    return getattr(info, key)


def dataset_total_episodes(dataset: LeRobotDataset) -> int:
    return int(_info_value(dataset, "total_episodes"))


def dataset_fps(dataset: LeRobotDataset) -> int:
    return int(_info_value(dataset, "fps"))


def active_data_parquet_files(dataset: LeRobotDataset) -> list[Path]:
    """Return current-revision data files referenced by LeRobot episode metadata."""
    total_episodes = dataset_total_episodes(dataset)
    rel_paths = {dataset.meta.get_data_file_path(ep_idx) for ep_idx in range(total_episodes)}
    paths = sorted(dataset.root / rel_path for rel_path in rel_paths)
    missing = [path for path in paths if not path.exists()]
    assert not missing, "Current dataset metadata references missing parquet files:\n" + "\n".join(
        str(path) for path in missing[:20]
    )
    return paths


def load_parquet_data(
    dataset_root: Path,
    parquet_files: list[Path] | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load all parquet files into one DataFrame and return sorted source paths."""
    parquet_files = parquet_files or sorted((dataset_root / "data").glob("**/*.parquet"))
    assert len(parquet_files) > 0, f"No parquet files found in {dataset_root / 'data'}"

    dfs = []
    for file_path in parquet_files:
        table = pq.read_table(file_path)
        dfs.append(table.to_pandas())

    return pd.concat(dfs, ignore_index=True), parquet_files


def write_parquet_back(data_df: pd.DataFrame, parquet_files: list[Path]) -> None:
    """Write a modified DataFrame back to the original parquet file layout."""
    offset = 0
    for file_path in parquet_files:
        original_table = pq.read_table(file_path)
        n_rows = len(original_table)
        fixed_slice = data_df.iloc[offset : offset + n_rows]
        fixed_table = pa.Table.from_pandas(
            fixed_slice,
            schema=original_table.schema,
            preserve_index=False,
        )
        pq.write_table(fixed_table, file_path)
        offset += n_rows


def _episode_meta_parquet_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    assert files, f"No episode metadata parquet files under {dataset_root / 'meta' / 'episodes'}"
    return files


def refresh_episode_stats(
    dataset_root: Path,
    data_df: pd.DataFrame,
    episode_indices: set[int],
) -> list[Path]:
    """Recompute stats metadata for the outcome-edited columns from frame data.

    Patches the per-episode `stats/<feature>/<stat>` columns in meta/episodes
    parquet and the matching entries in meta/stats.json, for the features in
    EDIT_AFFECTED_FEATURES, for the edited `episode_indices`.

    Returns the list of modified files.
    """
    from lerobot.datasets.compute_stats import get_feature_stats

    features = [feat for feat in EDIT_AFFECTED_FEATURES if feat in data_df.columns]
    frames_by_episode = dict(tuple(data_df.groupby("episode_index")))
    modified: list[Path] = []
    for meta_path in _episode_meta_parquet_files(dataset_root):
        original_table = pq.read_table(meta_path)
        meta_df = original_table.to_pandas()
        episode_order = [int(ep) for ep in meta_df["episode_index"]]
        file_changed = False
        for feat in features:
            stat_cols = [c for c in meta_df.columns if c.startswith(f"stats/{feat}/")]
            if not stat_cols:
                continue
            # Rebuild columns from python lists: pandas .at unwraps length-1 arrays
            # to 0-d on assignment, which breaks the arrow fixed-size-list round-trip.
            col_values = {col: list(meta_df[col]) for col in stat_cols}
            changed_cols: set[str] = set()
            for row_pos, ep_idx in enumerate(episode_order):
                if ep_idx not in episode_indices:
                    continue
                assert ep_idx in frames_by_episode, (
                    f"Episode {ep_idx} listed in {meta_path} has no frames in the data parquet"
                )
                values = np.asarray(frames_by_episode[ep_idx][feat])
                stats = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)
                for col in stat_cols:
                    stat_key = col.rsplit("/", 1)[1]
                    old_cell = np.asarray(col_values[col][row_pos])
                    # stats[stat_key] missing would mean lerobot stats-format drift; KeyError is correct.
                    new_cell = np.asarray(stats[stat_key], dtype=old_cell.dtype).reshape(
                        old_cell.shape
                    )
                    if not np.array_equal(old_cell, new_cell):
                        col_values[col][row_pos] = new_cell
                        changed_cols.add(col)
            for col in changed_cols:
                meta_df[col] = pd.Series(col_values[col], index=meta_df.index, dtype=object)
                file_changed = True
        if file_changed:
            fixed_table = pa.Table.from_pandas(
                meta_df, schema=original_table.schema, preserve_index=False
            )
            pq.write_table(fixed_table, meta_path)
            modified.append(meta_path)

    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.exists():
        global_stats = json.loads(stats_path.read_text())
        global_changed = False
        for feat in features:
            if feat not in global_stats:
                continue
            values = data_df[feat].to_numpy()
            stats = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)
            for stat_key, old_val in global_stats[feat].items():
                old_arr = np.asarray(old_val, dtype=np.float64)
                new_arr = np.asarray(stats[stat_key], dtype=np.float64).reshape(old_arr.shape)
                if not np.allclose(old_arr, new_arr, atol=1e-9):
                    global_stats[feat][stat_key] = new_arr.tolist()
                    global_changed = True
        if global_changed:
            stats_path.write_text(json.dumps(global_stats, indent=4))
            modified.append(stats_path)

    return modified


def _root_results_payload(dataset_root: Path) -> dict | None:
    """Root ``results.json`` (eval datasets), the provenance of LIVE subtask marks."""
    path = dataset_root / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def repair_results_json_file(repo_id: str, dataset_root: Path, *, push: bool) -> list[Path]:
    results_path, backup_path, reconciliation = canonicalize_results_file(dataset_root)
    if results_path is None:
        print("No root results.json found; nothing to repair.")
        return []
    if backup_path is None:
        print("results.json already matches outcome edits and frame labels.")
        return []

    changed = [results_path, backup_path]
    print("Repaired results.json from outcome edits:")
    if reconciliation is not None:
        print(
            f"  reviewed={reconciliation['episodes_reviewed']} "
            f"class_changes={reconciliation['outcome_class_changes']} "
            f"success_flips={reconciliation['success_flips']}"
        )
    print(f"  backup: {backup_path.relative_to(dataset_root)}")
    if push:
        push_results_json_files(repo_id, dataset_root, changed)
    return changed


def push_results_json_files(repo_id: str, dataset_root: Path, paths: list[Path]) -> None:
    if not paths:
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    operations = []
    for path in paths:
        rel_path = path.relative_to(dataset_root)
        operations.append(CommitOperationAdd(path_in_repo=str(rel_path), path_or_fileobj=str(path)))
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Sync results.json with outcome-edited labels",
    )
    for path in paths:
        rel_path = path.relative_to(dataset_root)
        print(f"  pushed {rel_path}")
    advance_lerobot_version_tag(repo_id)


def append_session_label_events(
    repo_id: str,
    dataset_root: Path,
    pre_session_progress: dict,
    progress: dict,
    pre_session_sha: str | None,
) -> int:
    """Append label-history events for what THIS cv2 session decided.

    Diffed against the session-start progress snapshot, so re-opened but
    unchanged episodes emit nothing. Returns the number of events appended.
    """
    from huggingface_hub import HfApi

    from mulligan.real.lifecycle.label_history import (
        append_events,
        diff_progress_events,
        utc_now_iso,
    )

    events = diff_progress_events(
        pre_session_progress,
        progress,
        source={"kind": "human", "agent": HfApi().whoami()["name"], "tool": "cv2-editor"},
        evidence={"pre_sha": pre_session_sha},
        ts=utc_now_iso(),
    )
    if events:
        append_events(dataset_root, events)
        print(f"  Appended {len(events)} label-history event(s).")
    return len(events)


def push_label_history_file(repo_id: str, dataset_root: Path) -> Path | None:
    """Upload the label-history ledger when a push path skips the folder commit."""
    from mulligan.real.lifecycle.label_history import HISTORY_FILENAME

    path = dataset_root / HISTORY_FILENAME
    if not path.exists():
        return None

    from huggingface_hub import HfApi

    HfApi().upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(path),
        path_in_repo=HISTORY_FILENAME,
        commit_message="Append label-history events from cv2 outcome-editor session",
    )
    print(f"  pushed {HISTORY_FILENAME}")
    return path


def push_progress_file(repo_id: str, dataset_root: Path) -> Path | None:
    """Explicitly upload the editor progress dotfile.

    ``LeRobotDataset.push_to_hub()`` uploads the dataset folder, but hidden
    bookkeeping files are not a stable part of that contract. Outcome-edit
    completion depends on this dotfile, so push it as its own required artifact.
    """
    path = dataset_root / PROGRESS_FILENAME
    if not path.exists():
        return None

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(path),
        path_in_repo=PROGRESS_FILENAME,
        commit_message="Upload outcome editor progress record",
    )
    print(f"  pushed {PROGRESS_FILENAME}")
    advance_lerobot_version_tag(repo_id)
    return path


def extract_episode_frames(dataset: LeRobotDataset, ep_idx: int, camera: str) -> list[np.ndarray]:
    """Extract all frames for an episode+camera into BGR numpy arrays.

    Decodes with PyAV (dav1d) instead of ``cv2.VideoCapture``. LeRobot v3.0
    writes AV1-encoded videos, and OpenCV's bundled FFmpeg selects a
    hardware-only AV1 path that fails on machines without AV1 hardware decode
    ("Your platform doesn't support hardware accelerated AV1 decoding"). PyAV
    ships its own FFmpeg with the dav1d software decoder, so it decodes the same
    files identically on Linux and macOS. Frames are returned as BGR uint8 to
    match the ``cv2.imshow`` display path.
    """
    # Import PyAV lazily, AFTER cv2 is already loaded. PyAV bundles its own
    # libxcb in av.libs/; if it loads before cv2's Qt HighGUI initializes, the
    # Qt xcb platform plugin binds the incompatible bundled libxcb and
    # cv2.imshow deadlocks (no window, frozen CLI). cv2 is the first third-party
    # import in this module, so a lazy import here keeps the safe ordering even
    # though isort would otherwise hoist a top-level `import av` above `cv2`.
    import av

    ep = dataset.meta.episodes[ep_idx]
    video_rel_path = dataset.meta.get_video_file_path(ep_idx, camera)
    video_path = dataset.root / video_rel_path

    # v3.0 concatenates many episodes per video file; this episode is the
    # contiguous frame window [from_timestamp, to_timestamp). A half-frame
    # tolerance keeps float-rounded packet timestamps inside the window while
    # excluding the neighbouring episode's boundary frame.
    from_ts = ep[f"videos/{camera}/from_timestamp"]
    to_ts = ep[f"videos/{camera}/to_timestamp"]
    length = ep["length"]
    tol = 0.5 / dataset.meta.fps

    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # Seek to the keyframe at or before the episode start, then drop the
        # pre-roll frames the decoder emits before from_ts.
        container.seek(max(int(from_ts / stream.time_base), 0), stream=stream, backward=True)
        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t < from_ts - tol:
                continue
            if t >= to_ts - tol:
                break
            frames.append(frame.to_ndarray(format="bgr24"))

    assert len(frames) == length, (
        f"Decoded {len(frames)} frames for episode {ep_idx} camera {camera}, "
        f"expected episode length {length} ({video_path})"
    )
    return frames


def detect_episode_outcome(ep_data: pd.DataFrame) -> str:
    """Determine the current outcome of an episode from its last valid frame.

    Reads the last frame with is_valid==1 and classifies:
      - reward==1.0 and done==1 -> "success"
      - reward==0.0 and done==1 -> "failure"
      - reward==0.0 and done==0 -> "timeout"
      - reward==1.0 and done==0 -> "timeout" (a subtask spike the editor wrote on a
        timeout's final valid frame; see :func:`_validate_subtask_frame`)
    """
    valid_frames = ep_data[ep_data["is_valid"] == 1]
    assert len(valid_frames) > 0, "Episode has no valid frames"
    last_valid = valid_frames.iloc[-1]

    reward = float(last_valid["reward"])
    done = int(last_valid["done"])

    if reward == 1.0 and done == 1:
        return "success"
    elif reward == 0.0 and done == 1:
        return "failure"
    elif reward in (0.0, 1.0) and done == 0:
        return "timeout"
    else:
        raise ValueError(f"Unexpected reward={reward}, done={done} on last valid frame")


def detect_existing_outcome_frame(ep_data: pd.DataFrame) -> int | None:
    """Frame index where the episode's existing terminal outcome begins, if any.

    Returns the first frame of the ``done==1`` plateau already present in the
    data (e.g. the collection-time success onset), or None when the episode has
    no terminal frame (timeout). Used to prefill the outcome mark in
    subtask-annotation mode (``--subtask-marks N``) so an operator can add a
    mid-episode reward mark and confirm without moving the existing
    end-of-episode labels: re-applying the same onset rewrites the terminal
    reward/done plateau to identical values.
    """
    done_frames = ep_data[ep_data["done"] == 1]
    if done_frames.empty:
        return None
    return int(done_frames["frame_index"].min())


def last_valid_frame_index(ep_data: pd.DataFrame) -> int | None:
    """Last ``is_valid==1`` frame — the natural outcome frame for a timeout.

    A timeout has no ``done==1`` plateau, so :func:`detect_existing_outcome_frame`
    returns None for it. Its effective end is the last valid frame (downstream
    readers ignore ``is_valid==0`` padding). Used as the subtask-mode prefill
    fallback so ``g -> c`` works on timeouts without a manual re-mark.
    """
    valid = ep_data[ep_data["is_valid"] == 1] if "is_valid" in ep_data.columns else ep_data
    if valid.empty:
        return None
    return int(valid["frame_index"].max())


def dataset_task_names(dataset: LeRobotDataset) -> set[str]:
    tasks = getattr(dataset.meta, "tasks", None)
    if tasks is None or len(tasks) == 0:
        return set()
    if hasattr(tasks, "columns"):
        if "name" in tasks.columns:
            return {str(v) for v in tasks["name"].tolist()}
        return {str(idx) for idx in tasks.index.tolist()}
    return {str(t) for t in tasks}


def resolve_camera_crop_boxes(
    dataset_tasks: set[str],
) -> dict[str, tuple[int, int, int, int]]:
    """Resolve the station crop map for the task represented by a dataset.

    Every station role starts with its station-wide default. Registered real tasks may
    replace individual role crops through ``RealTaskSpec.camera_crop_overrides``. A
    mixed-task dataset is supported only when all registered tasks resolve to the same
    effective map; otherwise there is no correct dataset-wide display crop and choosing
    one would make outcome review misleading.
    """
    effective_maps: dict[tuple[tuple[str, tuple[int, int, int, int]], ...], list[str]] = {}
    for task_name in sorted(dataset_tasks):
        spec = find_task_spec_by_task_name(task_name)
        if spec is None:
            continue
        effective = {**STATION_CAMERA_DEFAULT_CROPS, **spec.camera_crop_overrides}
        effective_maps.setdefault(tuple(sorted(effective.items())), []).append(task_name)

    if len(effective_maps) > 1:
        task_groups = [tasks for tasks in effective_maps.values()]
        raise ValueError(
            "dataset mixes registered tasks with incompatible camera crop maps: "
            f"{task_groups}; edit one task at a time"
        )
    if effective_maps:
        return dict(next(iter(effective_maps)))
    return dict(STATION_CAMERA_DEFAULT_CROPS)


def camera_role_for_video_key(camera: str) -> str | None:
    """Return the station role a LeRobot video key names, or None for other cameras."""
    bare_key = camera.rsplit(".", 1)[-1]
    return bare_key if bare_key in STATION_CAMERA_KEYS_BY_ROLE else None


def crop_frame_for_outcome_review(
    frame: np.ndarray,
    camera: str,
    crop_boxes: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    """Apply a role's stored-frame crop, leaving non-station cameras unchanged."""
    role = camera_role_for_video_key(camera)
    if role is None or role not in crop_boxes:
        return frame

    x0, y0, x1, y1 = crop_boxes[role]
    height, width = frame.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"Invalid outcome-editor crop {crop_boxes[role]} for camera {camera!r} "
            f"(role={role!r}) on stored frame {width}x{height}"
        )
    return frame[y0:y1, x0:x1]


def resolve_subtask_marks(dataset_tasks: set[str], override: int | None) -> int:
    """Resolve the required per-episode subtask-mark count for a dataset.

    The count is a task constant living in ``RealTaskSpec.num_subtask_marks`` (single
    source of truth), resolved from the dataset's collection task name(s) so the
    operator never has to pass ``--subtask-marks`` by hand. ``override`` (the CLI flag,
    ``None`` when unset) wins when given — for an unregistered/ad-hoc dataset or to
    review terminal outcomes only.

    - ``override is not None`` -> return it verbatim.
    - No registered task among ``dataset_tasks`` -> 0 (terminal mark only).
    - Registered tasks that AGREE on a count -> that count.
    - Registered tasks that DISAGREE -> raise (a mixed-task dataset can't have one
      well-defined mark count; fail loud rather than silently pick one).
    """
    if override is not None:
        return override
    counts = {
        spec.num_subtask_marks
        for name in dataset_tasks
        if (spec := find_task_spec_by_task_name(name)) is not None
    }
    if not counts:
        return 0
    if len(counts) > 1:
        raise ValueError(
            "dataset mixes tasks with differing num_subtask_marks "
            f"{sorted(counts)} across tasks {sorted(dataset_tasks)}; pass --subtask-marks "
            "explicitly to disambiguate"
        )
    return counts.pop()


def _normalize_filter_value(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"true", "false", "none"}:
            return lowered
        return stripped
    if value is None:
        return "none"
    return str(value)


def _value_matches_filter(value: Any, accepted: frozenset[str]) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_value_matches_filter(item, accepted) for item in value)
    return _normalize_filter_value(value) in accepted


def _parse_episode_filter(spec: str) -> EpisodeFieldFilter:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--episode-filter must be FIELD=VALUE, got {spec!r}")
    field, value = spec.split("=", 1)
    field = field.strip()
    value = value.strip()
    if not field or not value:
        raise argparse.ArgumentTypeError(f"--episode-filter must be FIELD=VALUE, got {spec!r}")
    return EpisodeFieldFilter(
        fields=(field,),
        values=frozenset({_normalize_filter_value(value)}),
        label=f"{field}={value}",
    )


def _parse_episode_indices(spec: str) -> frozenset[int]:
    """Parse ``--episodes 60,65,67`` to a frozenset of episode indices."""
    try:
        idxs = frozenset(int(tok) for tok in spec.split(",") if tok.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--episodes must be comma-separated ints, got {spec!r}"
        ) from exc
    if not idxs:
        raise argparse.ArgumentTypeError("--episodes must list at least one index")
    return idxs


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: ledger row must be a JSON object")
        rows.append(row)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _default_ledger_paths(dataset_root: Path) -> list[Path]:
    meta_dir = dataset_root / "meta"
    names = (
        "blind_dagger_ledger.jsonl",
        "protocol_quota_ledger.jsonl",
        "teleop_manifest_ledger.jsonl",
    )
    return [meta_dir / name for name in names if (meta_dir / name).exists()]


def _selected_ledger_paths(args: argparse.Namespace, dataset_root: Path) -> list[Path]:
    return list(args.ledger) if args.ledger else _default_ledger_paths(dataset_root)


def _load_ledger_rows_by_episode(paths: list[Path]) -> dict[int, dict[str, Any]]:
    rows_by_episode: dict[int, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Ledger path does not exist: {path}")
        for row in _load_jsonl_rows(path):
            if "episode_index" not in row:
                raise ValueError(f"{path}: ledger row missing episode_index: {row}")
            ep_idx = int(row["episode_index"])
            if ep_idx in rows_by_episode:
                raise ValueError(
                    f"Duplicate episode_index {ep_idx} across loaded ledgers; "
                    "pass one explicit --ledger to disambiguate"
                )
            rows_by_episode[ep_idx] = row
    return rows_by_episode


def build_episode_selector(args: argparse.Namespace, dataset_root: Path) -> EpisodeSelector:
    filters: list[EpisodeFieldFilter] = []

    def add_filter(fields: tuple[str, ...], values: list[str], label_name: str) -> None:
        if values:
            normalized = frozenset(_normalize_filter_value(value) for value in values)
            labels = ",".join(sorted(normalized))
            filters.append(
                EpisodeFieldFilter(fields=fields, values=normalized, label=f"{label_name}={labels}")
            )

    add_filter(("arm_key", "arm", "manifest_source", "manifest_sources"), args.arm_key, "arm-key")
    add_filter(("policy_id",), args.policy_id, "policy-id")
    add_filter(("arm_id",), args.arm_id, "arm-id")
    add_filter(("model_id",), args.model_id, "model-id")
    filters.extend(_parse_episode_filter(spec) for spec in args.episode_filter)

    if not filters:
        return EpisodeSelector(filters=(), ledger_rows_by_episode={})

    ledger_paths = _selected_ledger_paths(args, dataset_root)
    ledger_rows_by_episode = _load_ledger_rows_by_episode(ledger_paths) if ledger_paths else {}
    return EpisodeSelector(filters=tuple(filters), ledger_rows_by_episode=ledger_rows_by_episode)


def _episode_matches_filter(
    ep_idx: int,
    ep_data: pd.DataFrame,
    ledger_rows_by_episode: dict[int, dict[str, Any]],
    field_filter: EpisodeFieldFilter,
) -> bool:
    ledger_row = ledger_rows_by_episode.get(ep_idx, {})
    saw_field = False

    for field in field_filter.fields:
        if field in ledger_row:
            saw_field = True
            if _value_matches_filter(ledger_row[field], field_filter.values):
                return True
        if field in ep_data.columns:
            saw_field = True
            values = ep_data[field].dropna().unique().tolist()
            if any(_value_matches_filter(value, field_filter.values) for value in values):
                return True

    if not saw_field:
        fields = ", ".join(field_filter.fields)
        raise ValueError(
            f"Episode filter {field_filter.label!r} could not find any of [{fields}] "
            "in the loaded ledger rows or parquet columns"
        )
    return False


def remaining_episodes_to_review(
    filtered_eps: list[int],
    changed_episodes: dict[str, Any],
    skipped: set,
    subtask_marks: int,
    *,
    force_redo: bool = False,
) -> list[int]:
    """Episodes from ``filtered_eps`` that still need review.

    Normally drops episodes the operator already skipped or fully processed.
    ``force_redo`` (set by an explicit ``--episodes`` spot-check) re-includes every
    listed episode regardless of prior completion so a finished episode can be redone
    — it resumes from its existing progress record.
    """
    skipped_set = {str(e) for e in skipped}
    return [
        ep
        for ep in filtered_eps
        if force_redo
        or (
            str(ep) not in skipped_set
            and not episode_fully_processed(changed_episodes.get(str(ep)), subtask_marks)
        )
    ]


def get_filtered_episodes(
    data_df: pd.DataFrame,
    outcome_filter: str,
    selector: EpisodeSelector | None = None,
    only_episodes: frozenset[int] | None = None,
) -> list[int]:
    """Return episode indices matching the outcome filter.

    ``only_episodes`` restricts to an explicit index set (the ``--episodes``
    spot-check); a requested index absent from the dataset fails loud rather than
    silently selecting nothing.
    """
    present = {int(e) for e in data_df["episode_index"].unique()}
    if only_episodes is not None:
        missing = only_episodes - present
        if missing:
            raise ValueError(f"--episodes not present in dataset: {sorted(missing)}")
    episodes = []
    for ep_idx in sorted(data_df["episode_index"].unique()):
        if only_episodes is not None and int(ep_idx) not in only_episodes:
            continue
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        if selector is not None and not selector.matches(int(ep_idx), ep_data):
            continue
        outcome = detect_episode_outcome(ep_data)
        if outcome_filter == "all" or outcome == outcome_filter:
            episodes.append(int(ep_idx))
    return episodes


def load_progress(dataset_root: Path) -> dict:
    """Load editing progress from disk."""
    path = dataset_root / PROGRESS_FILENAME
    if path.exists():
        with open(path) as f:
            progress = json.load(f)
    else:
        progress = {"changed_episodes": {}, "skipped_episodes": []}

    progress.setdefault("changed_episodes", {})
    progress.setdefault("skipped_episodes", [])

    return progress


def save_progress(dataset_root: Path, progress: dict) -> None:
    """Save editing progress to disk."""
    path = dataset_root / PROGRESS_FILENAME
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def _remove_skipped_episode(progress: dict, ep_idx: int) -> None:
    progress["skipped_episodes"] = [
        skipped for skipped in progress["skipped_episodes"] if int(skipped) != ep_idx
    ]


def mark_episode_changed(
    progress: dict,
    ep_idx: int,
    *,
    new_outcome: str,
    outcome_frame: int,
    soft_truncate: bool,
    subtask_frames: list[int] | tuple[int, ...] = (),
    subtask_reviewed: bool = False,
) -> None:
    _remove_skipped_episode(progress, ep_idx)
    record = {
        "new_outcome": new_outcome,
        "outcome_frame": outcome_frame,
        "soft_truncate": soft_truncate,
    }
    # Persist the subtask_frames key when the operator reviewed this episode in subtask
    # mode — even EMPTY, so a legitimate 0-mark failure review is distinguishable from an
    # unreviewed record (key present ⇔ reviewed; see episode_fully_processed).
    # N=0 sessions never set subtask_reviewed, so their records carry no key (absent ⇒ []). Deduplicated so a repeated hand-edited
    # frame cannot double-count toward the mark budget.
    if subtask_reviewed or subtask_frames:
        record["subtask_frames"] = sorted({int(f) for f in subtask_frames})
    progress["changed_episodes"][str(ep_idx)] = record


def mark_episode_skipped(progress: dict, ep_idx: int) -> None:
    progress["changed_episodes"].pop(str(ep_idx), None)
    _remove_skipped_episode(progress, ep_idx)
    progress["skipped_episodes"].append(ep_idx)


def normalize_outcome_frame(ep_data: pd.DataFrame, ep_idx: int, outcome_frame: int) -> int:
    """Return a valid frame index for outcome placement.

    LeRobot episodes often include one invalid terminal padding row. Outcome
    labels must land on the last valid frame, because downstream readers ignore
    rows with is_valid=0.
    """
    # Bounds check FIRST, unconditionally: without it a missing is_valid
    # column let an out-of-range frame through, and apply_outcome_edits then
    # wrote reward/done to NO frames while flipping success on all of them.
    frame_rows = ep_data[ep_data["frame_index"] == outcome_frame]
    if frame_rows.empty:
        raise ValueError(f"Episode {ep_idx}: frame {outcome_frame} not found")

    if "is_valid" not in ep_data.columns:
        return outcome_frame

    if bool(frame_rows["is_valid"].iloc[0]):
        return outcome_frame

    valid_frames = ep_data[ep_data["is_valid"] == 1]
    if valid_frames.empty:
        raise ValueError(f"Episode {ep_idx}: no valid frames available for outcome placement")

    last_valid_frame = int(valid_frames["frame_index"].max())
    last_frame = int(ep_data["frame_index"].max())
    if outcome_frame == last_frame and outcome_frame > last_valid_frame:
        print(
            f"  Episode {ep_idx}: frame {outcome_frame} is invalid terminal padding; "
            f"using last valid frame {last_valid_frame}."
        )
        return last_valid_frame

    # A non-terminal is_valid=0 frame is a previously soft-truncated tail
    # frame (a prior apply invalidated it). The apply resets is_valid=1 on
    # every frame < last_frame before re-applying truncation, so it is a
    # legal new mark — rejecting it made a later re-review impossible.
    print(
        f"  Episode {ep_idx}: frame {outcome_frame} was invalidated by a prior "
        f"soft-truncation; accepting it — the apply re-validates pre-terminal frames."
    )
    return outcome_frame


def episode_fully_processed(entry: dict | None, subtask_marks: int) -> bool:
    """Whether a changed-episode progress record satisfies this session's contract.

    With ``subtask_marks == 0`` any changed record counts as processed. In subtask mode a changed episode counts as processed once
    the operator has CONFIRMED it in a subtask session, which stamps a ``subtask_frames``
    key (possibly empty — a failure that reached no sub-goal is a legitimate 0-mark
    review). Records from an N=0 session lack the key entirely, so those
    episodes re-enter the work queue to be reviewed, resuming from the record's terminal
    outcome/frame. Key PRESENCE, not mark count, is the reviewed signal — the count is
    validated (outcome-aware, see :func:`subtask_mark_count_error`) at confirm time.
    """
    if entry is None:
        return False
    if subtask_marks <= 0:
        return True
    return "subtask_frames" in entry


def _validate_subtask_frame(
    ep_data: pd.DataFrame,
    ep_idx: int,
    frame: int,
    outcome_frame: int,
    new_outcome: str,
) -> None:
    """Raise (never clamp) if a subtask reward frame is not a legal spike site.

    Validity is judged against the POST-edit state: ``apply_outcome_edits``
    re-validates every frame before the terminal padding row (soft truncation
    only invalidates frames strictly AFTER the outcome frame), so any existing
    frame strictly before the (normalized) outcome frame is valid once the edit
    lands — including mid-episode frames a prior soft-truncate had invalidated.
    A timeout has no terminal ``done==1`` transition, so a subtask completed on
    the final valid frame may equal its normalized timeout boundary. This is the
    only equality case: a success/failure outcome owns its frame, and a subtask
    after any outcome is impossible. The terminal padding row itself remains
    invalid because normalization moves the outcome to the last valid frame.
    """
    frame_rows = ep_data[ep_data["frame_index"] == frame]
    if frame_rows.empty:
        raise ValueError(f"Episode {ep_idx}: subtask frame {frame} not found")
    if frame > outcome_frame or (frame == outcome_frame and new_outcome != "timeout"):
        raise ValueError(
            f"Episode {ep_idx}: subtask frame {frame} must be strictly before the "
            f"outcome frame {outcome_frame} (equality is allowed only for timeout)"
        )


def apply_outcome_edits(
    data_df: pd.DataFrame,
    progress: dict,
    *,
    subtask_marks: int = 0,
) -> bool:
    """Apply all outcome edits from progress to the DataFrame. Returns True if any applied.

    ``subtask_marks`` is the required per-episode subtask-mark count (the CLI
    ``--subtask-marks``). Each changed episode must carry exactly this many
    ``subtask_frames``; with the default 0, every changed episode must have none,
    the subtask branch is a no-op, and writes are byte-identical to the
    single-mark editor.
    """
    changed = progress.get("changed_episodes", {})
    if not changed:
        return False

    for ep_str, info in changed.items():
        ep_idx = int(ep_str)
        new_outcome = info["new_outcome"]
        outcome_frame = info["outcome_frame"]
        soft_truncate = info.get("soft_truncate", True)
        # Key PRESENCE marks a record the operator has reviewed in subtask mode; a
        # record from an N=0 session lacks it. sorted(set(...)): a hand-edited duplicate
        # like [2, 2] must not double-count toward the mark budget.
        subtask_reviewed = "subtask_frames" in info
        subtask_frames = sorted({int(f) for f in info.get("subtask_frames", [])})
        outcome_vals = OUTCOME_TYPES[new_outcome]

        ep_mask = data_df["episode_index"] == ep_idx
        ep_frame_indices = data_df.loc[ep_mask, "frame_index"]
        if ep_frame_indices.empty:
            raise ValueError(f"Episode {ep_idx} not found in dataset")
        ep_data = data_df.loc[ep_mask]
        outcome_frame = normalize_outcome_frame(ep_data, ep_idx, int(outcome_frame))
        last_frame = int(ep_frame_indices.max())

        if subtask_marks > 0:
            # The count contract binds only to records the operator has REVIEWED
            # (subtask_frames key present). A record with no key is not yet revisited
            # (an N=0 session, or an episode still ahead of the marking frontier): apply its terminal edits and write no spike. Reviewed records
            # are checked outcome-aware (success == N, failure/timeout 0..N).
            if subtask_reviewed:
                err = subtask_mark_count_error(new_outcome, len(subtask_frames), subtask_marks)
                if err:
                    raise ValueError(f"Episode {ep_idx}: {err} ({subtask_frames})")
        elif subtask_frames:
            raise ValueError(
                f"Episode {ep_idx}: progress record carries subtask marks {subtask_frames} "
                "but --subtask-marks is 0; rerun with the matching --subtask-marks"
            )

        # Set success on ALL frames (episode-level constant)
        data_df.loc[ep_mask, "success"] = outcome_vals["success"]

        # Frames before the outcome frame: reward=0.0, done=0
        before_mask = ep_mask & (data_df["frame_index"] < outcome_frame)
        data_df.loc[before_mask, "reward"] = 0.0
        data_df.loc[before_mask, "done"] = 0

        # Frames from the outcome frame onward get the selected outcome semantics.
        from_mask = ep_mask & (data_df["frame_index"] >= outcome_frame)
        data_df.loc[from_mask, "reward"] = outcome_vals["reward"]
        data_df.loc[from_mask, "done"] = outcome_vals["done"]

        if "is_valid" in data_df.columns:
            default_valid_mask = ep_mask & (data_df["frame_index"] < last_frame)
            terminal_pad_mask = ep_mask & (data_df["frame_index"] == last_frame)
            data_df.loc[default_valid_mask, "is_valid"] = 1
            data_df.loc[terminal_pad_mask, "is_valid"] = 0
            if soft_truncate:
                after_mask = ep_mask & (data_df["frame_index"] > outcome_frame)
                data_df.loc[after_mask, "is_valid"] = 0

        # Subtask reward spikes: reward=1.0 at each marked frame, leaving done and
        # is_valid untouched. This MUST run after the before-mask reward=0 write
        # above (subtask frames are before the outcome frame, except that a
        # timeout may carry a spike on its boundary), or the spike would be
        # clobbered back to zero.
        for frame in subtask_frames:
            _validate_subtask_frame(ep_data, ep_idx, frame, outcome_frame, new_outcome)
            data_df.loc[ep_mask & (data_df["frame_index"] == frame), "reward"] = 1.0

    return True


def _episode_outcomes_by_index(
    data_df: pd.DataFrame,
    *,
    subtask_frames_by_episode: dict[int, list[int]] | None = None,
) -> dict[int, str]:
    subtask_by_ep = subtask_frames_by_episode or {}
    outcomes: dict[int, str] = {}
    for ep_idx in sorted(data_df["episode_index"].unique()):
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        index = int(ep_idx)
        outcomes[index] = _detect_frame_outcome(
            ep_data,
            subtask_frames=tuple(subtask_by_ep.get(index, ())),
        ).outcome
    return outcomes


def _sync_ledger_rows_with_outcomes(
    rows: list[dict[str, Any]],
    outcomes_by_episode: dict[int, str],
    *,
    only_episodes: set[int] | None = None,
    path: Path,
) -> tuple[list[str], bool]:
    errors: list[str] = []
    changed = False
    seen: set[int] = set()
    for row in rows:
        if "episode_index" not in row:
            errors.append(f"{path}: ledger row missing episode_index: {row}")
            continue
        ep_idx = int(row["episode_index"])
        if ep_idx in seen:
            errors.append(f"{path}: duplicate episode_index {ep_idx}")
            continue
        seen.add(ep_idx)
        if only_episodes is not None and ep_idx not in only_episodes:
            continue
        if ep_idx not in outcomes_by_episode:
            errors.append(f"{path}: episode_index {ep_idx} not found in dataset parquet")
            continue

        outcome = outcomes_by_episode[ep_idx]
        success = outcome == "success"
        row_changed = False
        if "success" not in row or bool(row["success"]) != success:
            row["success"] = success
            row_changed = True
        if "outcome" not in row or str(row["outcome"]) != outcome:
            row["outcome"] = outcome
            row_changed = True

        changed = changed or row_changed
    return errors, changed


def repair_ledgers(
    *,
    data_df: pd.DataFrame,
    ledger_paths: list[Path],
    only_episodes: set[int] | None = None,
    subtask_frames_by_episode: dict[int, list[int]] | None = None,
) -> list[Path]:
    if not ledger_paths:
        return []
    outcomes_by_episode = _episode_outcomes_by_index(
        data_df,
        subtask_frames_by_episode=subtask_frames_by_episode,
    )
    updated_paths: list[Path] = []
    all_errors: list[str] = []
    for path in ledger_paths:
        if not path.exists():
            raise FileNotFoundError(f"Ledger path does not exist: {path}")
        rows = _load_jsonl_rows(path)
        errors, changed = _sync_ledger_rows_with_outcomes(
            rows,
            outcomes_by_episode,
            only_episodes=only_episodes,
            path=path,
        )
        all_errors.extend(errors)
        if changed:
            _write_jsonl_rows(path, rows)
            updated_paths.append(path)
    if all_errors:
        preview = "\n".join(all_errors[:20])
        extra = "" if len(all_errors) <= 20 else f"\n... {len(all_errors) - 20} more"
        raise ValueError(f"Ledger validation failed:\n{preview}{extra}")
    return updated_paths


def push_ledger_files(repo_id: str, dataset_root: Path, ledger_paths: list[Path]) -> None:
    if not ledger_paths:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    for path in ledger_paths:
        try:
            rel_path = path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"Cannot push ledger {path}: it is not under dataset root {dataset_root}"
            ) from exc
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(path),
            path_in_repo=str(rel_path),
            commit_message="Update outcome ledger labels",
        )
        print(f"  pushed ledger {rel_path}")
    advance_lerobot_version_tag(repo_id)


def build_display_frame(
    camera_frames: dict[str, list[np.ndarray]],
    frame_idx: int,
    total_frames: int,
    ep_idx: int,
    current_outcome: str,
    pending_outcome: str | None,
    marked_frame: int | None,
    soft_truncate: bool,
    progress_str: str,
    scale: float,
    camera_crop_boxes: dict[str, tuple[int, int, int, int]] | None = None,
    subtask_marks: int = 0,
    subtask_frames: list[int] | None = None,
) -> np.ndarray:
    """Build a composite display frame with outcome info and visual overlays.

    When ``subtask_marks == 0`` the subtask overlays/help are fully suppressed so
    the display shows only the terminal outcome. When ``> 0`` each frame
    that is a subtask mark gets a distinct magenta border + label (visually
    unlike the translucent outcome band), and a dedicated info line is drawn.
    """
    subtask_frames = subtask_frames or []
    subtask_active = subtask_marks > 0
    is_subtask_frame = subtask_active and frame_idx in subtask_frames
    panels = []
    for cam_name, frames in camera_frames.items():
        frame = frames[frame_idx].copy()
        if camera_crop_boxes is not None:
            frame = crop_frame_for_outcome_review(frame, cam_name, camera_crop_boxes)

        # Color overlay for frames at or after the marked outcome frame
        if marked_frame is not None and pending_outcome is not None and frame_idx >= marked_frame:
            color = OUTCOME_COLORS[pending_outcome]
            overlay = frame.copy()
            overlay[:, :] = color
            alpha = 0.4 if soft_truncate and frame_idx > marked_frame else 0.2
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Distinct single-frame subtask spike marker: solid magenta border + tag.
        if is_subtask_frame:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), SUBTASK_MARK_COLOR, 6)
            cv2.putText(
                frame,
                "SUBTASK REWARD",
                (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                SUBTASK_MARK_COLOR,
                2,
            )

        # Camera label
        short_name = cam_name.split(".")[-1] if "." in cam_name else cam_name
        cv2.putText(frame, short_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        panels.append(frame)

    # Crops intentionally zoom into the task workspace; restore them to the decoded
    # frame height so enabling crops does not shrink the editor window (and its status
    # text) to the ROI height. Aspect ratios remain intact.
    display_h = min(frames[frame_idx].shape[0] for frames in camera_frames.values())
    resized = []
    for panel in panels:
        if panel.shape[0] != display_h:
            ratio = display_h / panel.shape[0]
            new_w = int(panel.shape[1] * ratio)
            panel = cv2.resize(panel, (new_w, display_h))
        resized.append(panel)

    # Stack panels side-by-side.
    if len(resized) > 1:
        composite = np.concatenate(resized, axis=1)
    else:
        composite = resized[0]

    # Info bar at the bottom. Grows by one line only when subtask marks are
    # enabled, so the N=0 display keeps its exact original 70px / 3-line layout.
    bar_height = 90 if subtask_active else 70
    bar = np.zeros((bar_height, composite.shape[1], 3), dtype=np.uint8)

    # Line 1: episode, frame, current outcome
    outcome_color = OUTCOME_COLORS[current_outcome]
    # Convert BGR to something readable for text
    line1 = (
        f"Ep {ep_idx} | Frame {frame_idx}/{total_frames - 1} | Current: {current_outcome.upper()}"
    )
    if pending_outcome:
        line1 += f" -> {pending_outcome.upper()}"
    if marked_frame is not None:
        line1 += f" | Marked @ {marked_frame}"
    line1 += f" | truncate={'ON' if soft_truncate else 'OFF'}"
    if subtask_active:
        line1 += f" | Sub {len(subtask_frames)}/{subtask_marks} @ {sorted(subtask_frames)}"
    line1 += f" | {progress_str}"
    cv2.putText(bar, line1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, outcome_color, 1)

    # Line 2: pending action status
    if pending_outcome and marked_frame is not None:
        line2 = f"Ready to confirm: {pending_outcome.upper()} from frame {marked_frame} (press c)"
        cv2.putText(bar, line2, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    elif pending_outcome:
        line2 = f"Outcome set to {pending_outcome.upper()} - press m/Enter to mark the frame"
        cv2.putText(bar, line2, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    else:
        line2 = "Press s/f/t to set outcome, then m to mark frame"
        cv2.putText(bar, line2, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Line 3: controls
    line3 = (
        "L/R: step | [/]: jump10 | Space: play | s/f/t: outcome | m: mark | "
        "x: trunc | u: undo | c: confirm | n: skip | p/b: prev | q: quit"
    )
    cv2.putText(bar, line3, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1)

    # Line 4 (subtask marks only): dedicated subtask help/status.
    if subtask_active:
        line4 = (
            f"g: toggle subtask reward mark ({len(subtask_frames)}/{subtask_marks}) | "
            "u also clears subtask marks | need exactly N marks to confirm"
        )
        cv2.putText(bar, line4, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.32, SUBTASK_MARK_COLOR, 1)

    composite = np.concatenate([composite, bar], axis=0)

    if scale != 1.0:
        new_w = int(composite.shape[1] * scale)
        new_h = int(composite.shape[0] * scale)
        composite = cv2.resize(composite, (new_w, new_h))

    return composite


def apply_progress_and_push(
    repo_id: str,
    dataset: LeRobotDataset,
    *,
    data_files: list[str],
    ledger_paths: list[Path],
    progress: dict,
    subtask_marks: int,
    push: bool,
    license: str = DEFAULT_DATASET_LICENSE,
) -> dict:
    """Apply a complete progress record to local parquet and optionally push.

    The interactive editor's all-processed branch uses it too, so the
    headless worker path and the UI path share one apply implementation
    (parquet rewrite, stats refresh, ledger repair, results.json canonicalize,
    push, v3.0 tag advance).
    """
    updated_ledgers: list[Path] = []
    data_df_after: pd.DataFrame | None = None
    if progress["changed_episodes"]:
        print("Re-applying edits from progress file...")
        data_df_fresh, parquet_files = load_parquet_data(dataset.root, data_files)
        apply_outcome_edits(data_df_fresh, progress, subtask_marks=subtask_marks)
        write_parquet_back(data_df_fresh, parquet_files)
        changed_eps = {int(ep) for ep in progress["changed_episodes"]}
        refreshed_stats = refresh_episode_stats(dataset.root, data_df_fresh, changed_eps)
        if refreshed_stats:
            print(f"Refreshed {len(refreshed_stats)} stats metadata file(s).")
        updated_ledgers = repair_ledgers(
            data_df=data_df_fresh,
            ledger_paths=ledger_paths,
            only_episodes=changed_eps,
            subtask_frames_by_episode=subtask_frames_for_validation(
                progress, _root_results_payload(dataset.root)
            ),
        )
        if updated_ledgers:
            print(f"Updated {len(updated_ledgers)} ledger file(s).")
        print("Parquet files updated.")
        data_df_after = data_df_fresh
    pushed = False
    if push:
        repaired_results = repair_results_json_file(repo_id, dataset.root, push=False)
        print("Pushing dataset to Hub...")
        dataset.push_to_hub(license=license)
        push_progress_file(repo_id, dataset.root)
        if progress["changed_episodes"]:
            push_ledger_files(repo_id, dataset.root, updated_ledgers)
            if not updated_ledgers and not repaired_results:
                advance_lerobot_version_tag(repo_id)
        else:
            advance_lerobot_version_tag(repo_id)
        push_results_json_files(repo_id, dataset.root, repaired_results)
        print("Done!")
        pushed = True
    if data_df_after is None:
        data_df_after, _ = load_parquet_data(dataset.root, data_files)
    episode_success = _episode_success_and_frames(
        data_df_after,
        subtask_frames_by_episode=subtask_frames_for_validation(
            progress, _root_results_payload(dataset.root)
        ),
    )
    return {
        "applied_episodes": sorted(int(ep) for ep in progress["changed_episodes"]),
        "skipped_episodes": sorted(int(ep) for ep in progress["skipped_episodes"]),
        "updated_ledgers": [str(path) for path in updated_ledgers],
        "episode_success": episode_success,
        "pushed": pushed,
    }


def _episode_success_and_frames(
    data_df: pd.DataFrame,
    *,
    subtask_frames_by_episode: dict[int, list[int]] | None = None,
) -> list[dict]:
    """Per-episode success / valid-frame / sub-goal-mark summary of the applied parquet rows.

    ``subtask_frames_by_episode`` (review record over live marks, see
    ``subtask_frames_for_validation``) yields each episode's sub-goal mark
    count for the graded score.
    """
    marks_by_episode = subtask_frames_by_episode or {}
    episode_success = []
    for ep_idx in sorted(int(v) for v in data_df["episode_index"].unique()):
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        first_rows = ep_data[ep_data["frame_index"] == 0]
        if len(first_rows) != 1:
            raise ValueError(f"Episode {ep_idx}: expected one frame_index==0 row")
        success_values = {int(v) for v in ep_data["success"].unique()}
        if len(success_values) != 1:
            raise ValueError(
                f"Episode {ep_idx}: success must be episode-constant, saw {sorted(success_values)}"
            )
        if not success_values <= {0, 1}:
            raise ValueError(
                f"Episode {ep_idx}: success must be binary 0/1, saw {sorted(success_values)}"
            )
        num_frames = (
            int((ep_data["is_valid"] == 1).sum()) if "is_valid" in ep_data.columns else len(ep_data)
        )
        episode_success.append(
            {
                "episode_index": ep_idx,
                "success": bool(next(iter(success_values))),
                "num_frames": num_frames,
                "num_subtask_marks": len(marks_by_episode.get(ep_idx, [])),
            }
        )
    return episode_success


def normalize_overlay_records(
    overlay: dict,
    data_df: pd.DataFrame,
    *,
    subtask_marks: int,
) -> dict:
    """Normalize an externally captured overlay to interactive-confirm record semantics.

    The interactive editor normalizes ``outcome_frame`` (invalid terminal
    padding → last valid frame, via :func:`normalize_outcome_frame`) and stamps
    the ``subtask_frames`` key only when the task runs in subtask mode
    (``subtask_marks > 0``) BEFORE recording. An external review tool may record the
    operator's raw frame and send an empty ``subtask_frames`` on a 0-mark task, so the
    headless path must apply the same normalization or the two tools write
    divergent progress records for identical decisions. Mutates IN PLACE.
    """
    for ep_str, record in overlay.get("changed_episodes", {}).items():
        ep_idx = int(ep_str)
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        if ep_data.empty:
            raise ValueError(f"Overlay episode {ep_idx} not found in parquet data")
        record["outcome_frame"] = normalize_outcome_frame(
            ep_data, ep_idx, int(record["outcome_frame"])
        )
        if subtask_marks == 0 and "subtask_frames" in record:
            if record["subtask_frames"]:
                raise ValueError(
                    f"Overlay episode {ep_idx} carries subtask_frames "
                    f"{record['subtask_frames']} but the task requires 0 subtask marks"
                )
            # Interactive semantics: subtask_reviewed = subtask_marks > 0, so a
            # 0-mark task never stamps the key (see mark_episode_changed).
            del record["subtask_frames"]
    return overlay


def merge_overlay_into_progress(progress: dict, overlay: dict) -> dict:
    """Merge a review overlay onto an existing progress record IN PLACE.

    Uses the same ``mark_episode_changed``/``mark_episode_skipped`` primitives
    as the interactive editor so overlay entries produce byte-identical
    records (subtask_frames key present ⇔ the overlay entry carried one).
    """
    for ep_str, record in overlay.get("changed_episodes", {}).items():
        mark_episode_changed(
            progress,
            int(ep_str),
            new_outcome=str(record["new_outcome"]),
            outcome_frame=int(record["outcome_frame"]),
            soft_truncate=bool(record.get("soft_truncate", False)),
            subtask_frames=record.get("subtask_frames", ()),
            subtask_reviewed="subtask_frames" in record,
        )
    for ep_idx in overlay.get("skipped_episodes", []):
        mark_episode_skipped(progress, int(ep_idx))
    return progress


def headless_apply_and_push(
    repo_id: str, overlay: dict, *, push: bool = True, license: str = DEFAULT_DATASET_LICENSE
) -> dict:
    """Merge an externally-captured review overlay into the dataset's progress
    record and run the standard apply+push path without any UI.

    ``overlay`` uses the progress-record shape:
    ``{"changed_episodes": {"<idx>": {new_outcome, outcome_frame, soft_truncate,
    subtask_frames?}}, "skipped_episodes": [idx, ...]}``. Entries are merged
    onto the dataset's existing ``.outcome_edit_progress.json`` via the same
    ``mark_episode_changed``/``mark_episode_skipped`` primitives the UI uses,
    then applied with :func:`apply_progress_and_push`.
    """
    args = argparse.Namespace(
        revision="main",
        force_cache_sync=True,
        ledger=None,
        subtask_marks=None,
    )
    # Captured immediately before the main-refresh download so the recorded
    # pre-state matches the base this apply actually edits (minimal race
    # window; a worker-side capture would leave the whole retried download
    # between capture and base).
    from huggingface_hub import HfApi

    pre_apply_sha = HfApi().dataset_info(repo_id).sha
    dataset = load_lerobot_dataset(repo_id, download_videos=False, args=args)
    dataset_tasks = dataset_task_names(dataset)
    subtask_marks = resolve_subtask_marks(dataset_tasks, None)

    data_files = active_data_parquet_files(dataset)
    data_df, _ = load_parquet_data(dataset.root, data_files)
    normalize_overlay_records(overlay, data_df, subtask_marks=subtask_marks)

    progress = load_progress(dataset.root)
    merge_overlay_into_progress(progress, overlay)
    save_progress(dataset.root, progress)

    ledger_paths = _default_ledger_paths(dataset.root)
    summary = apply_progress_and_push(
        repo_id,
        dataset,
        data_files=data_files,
        ledger_paths=ledger_paths,
        progress=progress,
        subtask_marks=subtask_marks,
        push=push,
        license=license,
    )
    summary["subtask_marks"] = subtask_marks
    summary["pre_apply_sha"] = pre_apply_sha
    summary["overlay_changed"] = sorted(int(ep) for ep in overlay.get("changed_episodes", {}))
    summary["overlay_skipped"] = sorted(int(ep) for ep in overlay.get("skipped_episodes", []))
    return summary


def process_single_dataset(
    repo_id: str,
    outcome_filter: str,
    cameras_arg: str | None,
    push: bool,
    scale: float,
    default_soft_truncate: bool,
    selection_args: argparse.Namespace,
    *,
    license: str = DEFAULT_DATASET_LICENSE,
) -> str:
    """Process a single dataset through the outcome editor.

    Returns:
        "done" — all filtered episodes processed (or no work to do)
        "quit" — user pressed q during editing
    """
    print(f"\nLoading dataset: {repo_id}")
    dataset = load_lerobot_dataset(
        repo_id,
        # Determine the work queue from metadata/parquet/progress first. Completed
        # sessions must not fetch the full video payload.
        download_videos=False,
        args=selection_args,
    )
    print(f"  Root: {dataset.root}")
    print(f"  Total episodes: {dataset_total_episodes(dataset)}")
    print(f"  Video keys: {dataset.meta.video_keys}")
    dataset_tasks = dataset_task_names(dataset)
    camera_crop_boxes = resolve_camera_crop_boxes(dataset_tasks)
    subtask_marks = resolve_subtask_marks(dataset_tasks, selection_args.subtask_marks)
    marks_source = (
        "--subtask-marks override"
        if selection_args.subtask_marks is not None
        else f"task config ({sorted(dataset_tasks) or 'no registered task'})"
    )
    print(f"  Subtask marks required per episode: {subtask_marks} (from {marks_source})")
    selector = build_episode_selector(selection_args, dataset.root)
    if selector.enabled:
        print(f"  Episode filters: {selector.description}")
        if selector.ledger_rows_by_episode:
            print(f"  Loaded ledger rows: {len(selector.ledger_rows_by_episode)}")

    # Determine cameras
    if cameras_arg:
        cameras = [c.strip() for c in cameras_arg.split(",")]
        for cam in cameras:
            assert cam in dataset.meta.video_keys, (
                f"Camera '{cam}' not found. Available: {dataset.meta.video_keys}"
            )
    else:
        cameras = auto_detect_cameras(dataset.meta.video_keys)
    print(f"  Using cameras: {cameras}")
    displayed_crops = {
        camera: camera_crop_boxes[role]
        for camera in cameras
        if (role := camera_role_for_video_key(camera)) is not None
    }
    print(f"  Outcome review crops: {displayed_crops or 'none (non-station cameras)'}")

    # Load parquet data
    print("Loading parquet data...")
    data_files = active_data_parquet_files(dataset)
    data_df, parquet_files = load_parquet_data(dataset.root, data_files)
    print(f"  Loaded {len(data_df)} frames")
    ledger_paths = _selected_ledger_paths(selection_args, dataset.root)

    # Find episodes matching the filter
    filtered_eps = get_filtered_episodes(
        data_df, outcome_filter, selector, only_episodes=selection_args.episodes
    )
    total_filtered = len(filtered_eps)
    print(
        f"  Episodes matching filter '{outcome_filter}': {total_filtered} / "
        f"{dataset_total_episodes(dataset)}"
    )

    if not filtered_eps:
        print(f"No episodes matching filter '{outcome_filter}'. Nothing to edit.")
        return "done"

    # Load progress. Snapshot it (plus the repo sha) so the session's pushes
    # can append label-history events for exactly what THIS session decided —
    # without this the cv2 path silently made the provenance ledger stale.
    progress = load_progress(dataset.root)
    pre_session_progress = json.loads(json.dumps(progress))
    from huggingface_hub import HfApi as _HfApi

    pre_session_sha = _HfApi().dataset_info(repo_id).sha if push else None
    changed = set(progress["changed_episodes"].keys())
    skipped = set(str(e) for e in progress["skipped_episodes"])

    filtered_ep_strings = {str(ep) for ep in filtered_eps}
    changed_filtered = changed & filtered_ep_strings
    skipped_filtered = skipped & filtered_ep_strings
    # An explicit --episodes request is a deliberate spot-check: re-open exactly the
    # listed episodes even if a prior session already confirmed them (they resume from
    # their existing record). Without it, keep the normal skip-already-processed flow.
    force_redo = selection_args.episodes is not None
    remaining_eps = remaining_episodes_to_review(
        filtered_eps, progress["changed_episodes"], skipped, subtask_marks, force_redo=force_redo
    )

    print(
        f"  Already changed: {len(changed_filtered)}, skipped: {len(skipped_filtered)}, "
        f"remaining: {len(remaining_eps)}"
    )
    if subtask_marks > 0:
        unmarked_changed = sum(
            1
            for ep in changed_filtered
            if not episode_fully_processed(progress["changed_episodes"][ep], subtask_marks)
        )
        if unmarked_changed:
            print(
                f"  Subtask mode: {unmarked_changed} changed episode(s) not yet reviewed for "
                f"sub-goals re-enter the queue (a success needs {subtask_marks} mark(s); a "
                "failure/timeout 0..N; terminal outcome resumes from the existing record)."
            )

    if not remaining_eps:
        print("All filtered episodes have been processed!")
        apply_progress_and_push(
            repo_id,
            dataset,
            data_files=data_files,
            ledger_paths=ledger_paths,
            progress=progress,
            subtask_marks=subtask_marks,
            push=push,
            license=license,
        )
        return "done"

    any_edits = False
    progress_changed_this_session = False

    print("Downloading videos for interactive review...")
    dataset = load_lerobot_dataset(repo_id, download_videos=True, args=selection_args)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    work_queue = filtered_eps
    queue_pos = work_queue.index(remaining_eps[0])
    while queue_pos < len(work_queue):
        ep_idx = work_queue[queue_pos]
        skipped_now = set(str(e) for e in progress["skipped_episodes"])
        processed_count = sum(
            1
            for ep in filtered_eps
            if str(ep) in skipped_now
            or episode_fully_processed(progress["changed_episodes"].get(str(ep)), subtask_marks)
        )
        remaining_count = total_filtered - processed_count
        progress_str = (
            f"Progress: {processed_count}/{total_filtered} | View: {queue_pos + 1}/{total_filtered}"
        )

        # Load episode data to detect current outcome
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        current_outcome = detect_episode_outcome(ep_data)

        # Load frames for this episode
        print(
            f"\nLoading episode {ep_idx} "
            f"(view {queue_pos + 1}/{len(work_queue)}, {remaining_count} remaining) "
            f"[{current_outcome}]..."
        )
        camera_frames = {}
        for cam in cameras:
            camera_frames[cam] = extract_episode_frames(dataset, ep_idx, cam)

        total_frames = len(next(iter(camera_frames.values())))
        saved_edit = progress["changed_episodes"].get(str(ep_idx))
        if saved_edit is not None:
            current_frame = int(saved_edit["outcome_frame"])
            pending_outcome: str | None = str(saved_edit["new_outcome"])
            marked_frame: int | None = current_frame
            soft_truncate = bool(saved_edit.get("soft_truncate", default_soft_truncate))
            subtask_frames: list[int] = sorted(
                {int(f) for f in saved_edit.get("subtask_frames", [])}
            )
        else:
            current_frame = 0
            pending_outcome = current_outcome
            marked_frame = None
            soft_truncate = default_soft_truncate
            subtask_frames = []
            if subtask_marks > 0:
                # Subtask-annotation mode: prefill the outcome mark from the
                # episode's existing done==1 onset so adding a subtask mark and
                # confirming rewrites the terminal plateau to identical values
                # (g -> c, no manual re-mark). N=0 behavior is untouched.
                marked_frame = detect_existing_outcome_frame(ep_data)
                if marked_frame is None:
                    # Timeout: no done==1 plateau, so the natural outcome frame is the
                    # last valid frame. Confirming re-applies an identical timeout and
                    # only adds the subtask mark. Without this prefill, g -> c is blocked
                    # on timeouts (marked_frame None) and skipping past loses the mark.
                    marked_frame = last_valid_frame_index(ep_data)
                if marked_frame is not None:
                    print(
                        f"  Prefilled outcome mark from existing data: "
                        f"{current_outcome.upper()} @ frame {marked_frame} "
                        "(add subtask mark with g, then c; m to move the onset)"
                    )
        playing = False
        fps = dataset_fps(dataset)
        frame_delay = 1.0 / fps

        quit_requested = False
        go_previous = False
        skip_armed = False  # a first 'n' with unsaved subtask marks arms; a second confirms

        while True:
            display = build_display_frame(
                camera_frames,
                current_frame,
                total_frames,
                ep_idx,
                current_outcome,
                pending_outcome,
                marked_frame,
                soft_truncate,
                progress_str,
                scale,
                camera_crop_boxes=camera_crop_boxes,
                subtask_marks=subtask_marks,
                subtask_frames=subtask_frames,
            )
            cv2.imshow(WINDOW_NAME, display)

            wait_ms = max(1, int(frame_delay * 1000)) if playing else 0
            key = cv2.waitKeyEx(wait_ms)

            if key == -1:
                if playing:
                    current_frame = min(current_frame + 1, total_frames - 1)
                    if current_frame == total_frames - 1:
                        playing = False
                continue

            # Any real keypress cancels a prior armed skip (only a second consecutive
            # 'n' confirms discarding unsaved subtask marks).
            was_skip_armed = skip_armed
            skip_armed = False

            # Navigation
            if key in (63234, 65361, 2, 81):  # Left arrow
                current_frame = max(0, current_frame - 1)
                playing = False
            elif key in (63235, 65363, 3, 83):  # Right arrow
                current_frame = min(total_frames - 1, current_frame + 1)
                playing = False
            elif key == ord("["):
                current_frame = max(0, current_frame - 10)
                playing = False
            elif key == ord("]"):
                current_frame = min(total_frames - 1, current_frame + 10)
                playing = False
            elif key == ord(" "):
                playing = not playing

            # Outcome selection
            elif key == ord("s"):
                pending_outcome = "success"
                print("  Outcome set to SUCCESS")
            elif key == ord("f"):
                pending_outcome = "failure"
                print("  Outcome set to FAILURE")
            elif key == ord("t"):
                pending_outcome = "timeout"
                print("  Outcome set to TIMEOUT")
            elif key == ord("x"):
                soft_truncate = not soft_truncate
                print(f"  Soft truncation {'enabled' if soft_truncate else 'disabled'}")

            # Mark / unmark
            elif key == ord("m") or key == 13:  # m or Enter
                if pending_outcome is None:
                    print("  Set an outcome first (s/f/t), then mark the frame.")
                    continue
                marked_frame = current_frame
                print(f"  Marked frame {marked_frame} as {pending_outcome.upper()} frame")

            # Subtask reward mark toggle
            elif key == ord("g"):
                if subtask_marks <= 0:
                    print(
                        "  Subtask marks disabled for this task (RealTaskSpec.num_subtask_marks=0; "
                        "override with --subtask-marks N)."
                    )
                elif current_frame in subtask_frames:
                    subtask_frames.remove(current_frame)
                    print(f"  Removed subtask mark at frame {current_frame}")
                elif len(subtask_frames) >= subtask_marks:
                    print(
                        f"  Already have {subtask_marks} subtask mark(s); press g on a marked "
                        "frame to remove one first."
                    )
                else:
                    subtask_frames.append(current_frame)
                    subtask_frames.sort()
                    print(
                        f"  Added subtask mark at frame {current_frame} "
                        f"({len(subtask_frames)}/{subtask_marks})"
                    )

            elif key == ord("u"):
                marked_frame = None
                pending_outcome = current_outcome
                subtask_frames = []
                print(f"  Cleared mark; outcome reset to {current_outcome.upper()}")

            # Confirm
            elif key == ord("c"):
                if pending_outcome is None:
                    print("  Set an outcome (s/f/t) before confirming.")
                    continue
                if marked_frame is None:
                    # No explicit 'm' mark: fall back to the outcome's natural frame so an
                    # operator who only added a subtask mark can confirm directly. Success/
                    # failure -> the existing done==1 onset; timeout (no plateau) -> the last
                    # valid frame. This is the robust site for the timeout g->c fix: it works
                    # regardless of how marked_frame ended up None (fresh, skipped, undone).
                    marked_frame = detect_existing_outcome_frame(ep_data)
                    if marked_frame is None:
                        marked_frame = last_valid_frame_index(ep_data)
                    if marked_frame is None:
                        print("  No valid frame to place the outcome; mark one with m.")
                        continue
                    print(
                        f"  Using {pending_outcome.upper()} frame {marked_frame} (no manual mark)."
                    )

                try:
                    marked_frame = normalize_outcome_frame(ep_data, ep_idx, marked_frame)
                except ValueError as exc:
                    print(f"  {exc}")
                    continue

                # Subtask-mark gating (outcome-aware): a SUCCESS must carry exactly N marks
                # (it reached the mid-episode sub-goal), a FAILURE/timeout 0..N (it reached
                # some prefix, possibly none). Marks must precede a success/failure
                # outcome. A timeout mark may equal the normalized final-frame boundary.
                if subtask_marks > 0:
                    err = subtask_mark_count_error(
                        pending_outcome, len(subtask_frames), subtask_marks
                    )
                    if err:
                        hint = (
                            "press g to mark the mid-episode sub-goal frame"
                            if pending_outcome == "success"
                            else "press g on a marked frame to remove one"
                        )
                        print(f"  Cannot confirm: {err}; {hint}.")
                        continue
                    late = [
                        f
                        for f in subtask_frames
                        if f > marked_frame or (f == marked_frame and pending_outcome != "timeout")
                    ]
                    if late:
                        print(
                            f"  Subtask mark(s) {sorted(late)} must be before the outcome frame "
                            f"{marked_frame} (equality is allowed only for timeout); "
                            "move them earlier (g to toggle)."
                        )
                        continue

                # Record in progress. subtask_reviewed stamps the key even for a 0-mark
                # failure so this review is not mistaken for an unreviewed record.
                mark_episode_changed(
                    progress,
                    ep_idx,
                    new_outcome=pending_outcome,
                    outcome_frame=marked_frame,
                    soft_truncate=soft_truncate,
                    subtask_frames=subtask_frames,
                    subtask_reviewed=subtask_marks > 0,
                )
                save_progress(dataset.root, progress)
                any_edits = True
                progress_changed_this_session = True

                ep_frames = ep_data.shape[0]
                truncated = ep_frames - marked_frame - 1
                truncation_msg = (
                    f"{truncated} frames will be soft-truncated"
                    if soft_truncate
                    else "no frames will be soft-truncated"
                )
                print(
                    f"  Confirmed: ep {ep_idx} -> {pending_outcome.upper()} @ frame {marked_frame} "
                    f"({truncation_msg})"
                )
                break

            # Skip
            elif key == ord("n"):
                if subtask_frames and not was_skip_armed:
                    # Loud guard: skipping would silently drop the marks the operator
                    # already placed.
                    skip_armed = True
                    print(
                        f"  ⚠ ep {ep_idx} has {len(subtask_frames)} unsaved subtask mark(s) at "
                        f"{sorted(subtask_frames)}. Press c to SAVE them, or n again to skip and DISCARD."
                    )
                    continue
                mark_episode_skipped(progress, ep_idx)
                save_progress(dataset.root, progress)
                progress_changed_this_session = True
                print(f"  Skipped episode {ep_idx}")
                break

            # Previous episode
            elif key in (ord("p"), ord("b")):
                if queue_pos == 0:
                    print("  Already at the first filtered episode.")
                    continue
                go_previous = True
                break

            # Quit
            elif key == ord("q"):
                quit_requested = True
                break

        if quit_requested:
            break
        if go_previous:
            queue_pos -= 1
        else:
            queue_pos += 1

    cv2.destroyAllWindows()

    # Write modified parquet files if any edits exist
    if progress["changed_episodes"]:
        print("\nRe-applying all edits and writing parquet files...")
        data_df_fresh, parquet_files = load_parquet_data(dataset.root, data_files)
        apply_outcome_edits(data_df_fresh, progress, subtask_marks=subtask_marks)
        write_parquet_back(data_df_fresh, parquet_files)
        changed_eps = {int(ep) for ep in progress["changed_episodes"]}
        refreshed_stats = refresh_episode_stats(dataset.root, data_df_fresh, changed_eps)
        if refreshed_stats:
            print(f"Refreshed {len(refreshed_stats)} stats metadata file(s).")
        updated_ledgers = repair_ledgers(
            data_df=data_df_fresh,
            ledger_paths=ledger_paths,
            only_episodes=changed_eps,
            subtask_frames_by_episode=subtask_frames_for_validation(
                progress, _root_results_payload(dataset.root)
            ),
        )
        if updated_ledgers:
            print(f"Updated {len(updated_ledgers)} ledger file(s).")
        print("Done writing.")
    else:
        updated_ledgers = []

    if push and any_edits:
        append_session_label_events(
            repo_id, dataset.root, pre_session_progress, progress, pre_session_sha
        )
        repaired_results = repair_results_json_file(repo_id, dataset.root, push=False)
        print("Pushing dataset to Hub...")
        dataset.push_to_hub(license=license)
        push_progress_file(repo_id, dataset.root)
        push_ledger_files(repo_id, dataset.root, updated_ledgers)
        if not updated_ledgers:
            advance_lerobot_version_tag(repo_id)
        push_results_json_files(repo_id, dataset.root, repaired_results)
        print("Push complete!")
    elif push and progress_changed_this_session:
        print("No parquet edits were made; pushing outcome editor progress record...")
        appended = append_session_label_events(
            repo_id, dataset.root, pre_session_progress, progress, pre_session_sha
        )
        if appended:
            push_label_history_file(repo_id, dataset.root)
        push_progress_file(repo_id, dataset.root)
        print("Progress push complete!")
    elif push and not any_edits:
        print("No edits were made, skipping push.")

    print(
        f"\nSummary: changed {len(progress['changed_episodes'])} episodes, skipped {len(progress['skipped_episodes'])}"
    )
    if subtask_marks > 0:
        entries = list(progress["changed_episodes"].values())
        reviewed = sum(1 for e in entries if episode_fully_processed(e, subtask_marks))
        total_spikes = sum(len({int(f) for f in e.get("subtask_frames", [])}) for e in entries)
        print(
            f"Subtask review: {reviewed}/{len(entries)} changed episode(s) reviewed for "
            f"sub-goals (N={subtask_marks}: success=N marks, failure/timeout 0..N); "
            f"{total_spikes} reward spike(s) total."
        )

    if quit_requested:
        return "quit"
    return "done"


def main():
    args = parse_args()
    if args.apply_overlay is not None:
        overlay = json.loads(args.apply_overlay.read_text())
        summary = headless_apply_and_push(
            args.repo_id, overlay, push=args.push, license=args.license
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return
    process_single_dataset(
        args.repo_id,
        args.filter,
        args.cameras,
        args.push,
        args.scale,
        args.soft_truncate,
        args,
        license=args.license,
    )


if __name__ == "__main__":
    main()
