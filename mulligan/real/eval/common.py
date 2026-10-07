"""Shared helpers of the real-robot evaluation entry points.

Rollout records, the ``results.json`` file (round plan, per-rollout outcomes,
``visit_id`` provenance, phase stops), eval-dataset creation / append / checkpoint,
and the console results table. Used by :mod:`mulligan.real.eval.manifest_eval` and
:mod:`mulligan.real.eval.blind_eval_helpers`; policy loading lives in
:mod:`mulligan.real.policy.loader`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import mulligan.real.policy.lerobot_patches  # noqa: F401  (h264 video codec)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.constants import DataSource
from mulligan.real.collect.dataset_features import (
    build_real_lerobot_features,
    ensure_dataset_can_store_episode_telemetry,
    record_or_verify_camera_role_serials,
    save_episode_to_dataset as _save_episode_to_dataset,
)
from mulligan.real.collect.save_utils import wait_for_background_save
from mulligan.real.lifecycle.tasks import find_task_spec_by_task_name
from mulligan.real.operator_ui.gates import GateOutcome
from mulligan.real.operator_ui.session import OperatorUI
from mulligan.real.robot.cameras import serial_key_to_role

_PRECREATE_DATASET_SIDECAR_FILES = frozenset({"policy_loading.log", "results.json"})
# Sidecar DIRECTORIES that may exist before LeRobot dataset creation: per-chunk BoN
# diagnostics are written after each episode, but the dataset is created lazily after
# the FIRST episode — an IQL arm in round 1 slot A writes chunk_info/ before create().
_PRECREATE_DATASET_SIDECAR_DIRS = frozenset({"chunk_info"})


def _streaming_encoding_enabled() -> bool:
    return os.environ.get("MULLIGAN_STREAMING_ENCODING", "1").lower() not in {
        "0",
        "false",
        "no",
        "",
    }


def _ensure_model_id_prefix(raw_id: str) -> str:
    """Add ``wandb://`` prefix if no URI scheme is present (results files written before
    model ids carried a scheme). Local checkpoint directories are stored as absolute
    paths and stay as they are."""
    if not raw_id:
        return raw_id
    if "://" in raw_id or os.path.isabs(raw_id):
        return raw_id
    return f"wandb://{raw_id}"


@dataclass
class PolicyEntry:
    model_id: str  # Full URI: "wandb://..." or "hf://..."
    policy_id: int  # stable index (0-based) for dataset annotation
    policy: object  # RealWorldPolicy (LeRobot DP or Vision-IDQL wrapper)
    camera_height: int = 480  # per-policy image resolution (auto-detected)
    camera_width: int = 640
    results: list[bool] = field(default_factory=list)


@dataclass
class RolloutRecord:
    """One rollout's metadata for the results file."""

    round_num: int
    policy_id: int
    model_id: str  # Full URI: "wandb://..." or "hf://..."
    anonymous_label: str
    outcome: str
    num_steps: int
    episode_index: int  # index in the dataset (-1 if not saved)
    manifest_idx: int | None = None
    pen_x: float | None = None
    pen_y: float | None = None
    pen_yaw: float | None = None
    # Frames the operator marked live with the sub-goal key ('g' / numpad '3'); each
    # carries a reward=1.0 spike in the saved episode. Empty for tasks without marks.
    subtask_frames: tuple[int, ...] = ()
    # Invocation that collected this rollout (manifest_eval stamps one id per launch), so a
    # PHASED eval can tell from data which records of a start were collected in a later
    # physical visit. None for records without the field.
    visit_id: str | None = None


def num_subtask_marks_for_task(task_name: str) -> int:
    """Mid-episode sub-goal marks the task defines (``RealTaskSpec.num_subtask_marks``).

    0 for tasks without a registered real spec (sim manifests, other names), which
    keeps the live sub-goal key inert and the results table binary-only.
    """
    spec = find_task_spec_by_task_name(task_name)
    return 0 if spec is None else spec.num_subtask_marks


def episode_score(outcome: str, subtask_frames: Sequence[int]) -> int:
    """Graded per-episode score: sub-goal marks + 1 for terminal success.

    Max is ``num_subtask_marks + 1`` (routing_d2: 2 = first clip + both clips), matching
    the ingest's ``<arm>_score`` column in ``paired_round_outcomes.csv``.
    """
    return len(subtask_frames) + (1 if outcome == "success" else 0)


def rollout_outcome_line(
    label: str,
    outcome: str,
    num_steps: int,
    subtask_frames: Sequence[int],
    num_subtask_marks: int,
) -> str:
    """Per-rollout console line; appends the graded score for tasks with sub-goal marks."""
    line = f"Policy {label}: {outcome.upper()} ({num_steps} steps)"
    if num_subtask_marks > 0:
        line += (
            f" | score {episode_score(outcome, subtask_frames)}/{num_subtask_marks + 1}"
            f" (sub-goal marks at frames {list(subtask_frames)})"
        )
    return line


def wait_until_ready(
    ui: OperatorUI,
    *,
    prompt: str,
    on_reset: Callable[[], object] | None,
) -> None:
    """Eval-loop gate: any key continues, ``r`` re-homes the robot, ``q`` ends the session.

    Quitting raises KeyboardInterrupt, the evals' orderly-shutdown path (results are
    saved and the dataset finalized in the ``finally`` blocks).
    """
    if ui.gate(prompt=prompt, on_reset=on_reset, any_key_starts=True) is GateOutcome.QUIT:
        raise KeyboardInterrupt


def save_episode_to_dataset(
    dataset,
    episode_data,
    episode_success,
    camera_keys,
    policy_id,
    round_num,
    task="blind_eval",
    extra_frame_fields: dict | None = None,
    verbose: bool = True,
):
    """Write all frames for one episode into the dataset and call save_episode().

    Thin wrapper around the shared save_episode_to_dataset that adds
    eval-specific extra fields (policy_id, round_id). Cameras are written under their
    station role names, matching the schema ``create_dataset`` declares.
    """
    frame_fields = {
        "policy_id": np.array([policy_id], dtype=np.int64),
        "round_id": np.array([round_num], dtype=np.int64),
    }
    if extra_frame_fields is not None:
        frame_fields.update(extra_frame_fields)
    _save_episode_to_dataset(
        dataset=dataset,
        episode_data=episode_data,
        episode_success=episode_success,
        camera_keys=camera_keys,
        task_name=task,
        saved_episode_count=0,  # not used for return value here
        default_source=DataSource.AUTONOMOUS,
        extra_frame_fields=frame_fields,
        camera_name_fn=serial_key_to_role,
        verbose=verbose,
    )


def create_dataset(
    dataset_path: Path,
    hf_repo_id: str,
    episode_data: dict,
    cam_data_keys: list[str],
    freq: int,
    extra_features: dict | None = None,
) -> LeRobotDataset:
    """Create a new LeRobotDataset from the first episode's shape info.

    Cameras are stored under their station role names (``serial_key_to_role``), the
    same names the collection path uses, and the serial<->role provenance sidecar
    (``meta/camera_role_serials.json``) is recorded.
    """
    print(f"Creating new dataset at {dataset_path}")

    features = build_real_lerobot_features(
        episode_data,
        cam_data_keys,
        extra_features={
            "policy_id": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["policy_id"],
            },
            "round_id": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["round_id"],
            },
        },
        camera_name_fn=serial_key_to_role,
    )
    if extra_features is not None:
        features.update(extra_features)

    sidecar_backup_path = _prepare_new_dataset_root(dataset_path)
    try:
        streaming_encoding = _streaming_encoding_enabled()
        dataset = LeRobotDataset.create(
            repo_id=hf_repo_id,
            fps=freq,
            root=str(dataset_path),
            robot_type="franka",
            features=features,
            image_writer_threads=0 if streaming_encoding else 4,
            streaming_encoding=streaming_encoding,
        )
    except Exception:
        if sidecar_backup_path is not None:
            print(
                "Dataset creation failed after moving pre-existing sidecar file(s) "
                f"to backup: {sidecar_backup_path}"
            )
        raise
    _restore_precreate_sidecars(dataset_path, sidecar_backup_path)
    record_or_verify_camera_role_serials(dataset, cam_data_keys)
    print(f"Dataset created at {dataset_path}")
    return dataset


def reopen_dataset_for_append(
    hf_repo_id: str,
    dataset_path: Path,
    episode_data: dict,
    cam_data_keys: list[str],
) -> LeRobotDataset:
    """Re-open an existing on-disk eval dataset in WRITE mode to append more episodes.

    FF lerobot's DatasetReader/DatasetWriter split made the bare ``LeRobotDataset(...)``
    constructor read-only (``writer is None`` → ``add_frame``/``save_episode`` raise).
    ``resume()`` rebuilds the DatasetWriter for appending. Streaming follows the
    process-wide default unless ``MULLIGAN_STREAMING_ENCODING=0`` is set; the
    LeRobot patch blocks instead of dropping frames, so buffered eval saves keep
    the same row/video-frame contract as live collection. This path then runs the
    Franka-torque schema upgrade (:func:`ensure_dataset_can_store_episode_telemetry`),
    which itself returns a write-mode dataset.

    This is the ONE re-open path of the eval entry points, so the FF write-API contract (resume, not the bare read-only constructor) can never drift
    between the eval entrypoints.

    It then VERIFIES the live serial<->role cabling of ``cam_data_keys`` still matches the
    provenance recorded at creation, failing loud on a re-cabled / wrong-station camera
    before any role-named frame is appended.

    Before resuming, :func:`reconcile_resumed_dataset` ensures ``info.json``'s episode
    counter has not run ahead of the durably-footered parquet (which a mid-session
    crash such as a failed verified reset can cause). It heals the safe
    trailing-partial case and fails loud on a mid-dataset gap, so resume never
    silently appends past orphaned episodes.
    """
    from mulligan.data.recording import reconcile_resumed_dataset

    reconcile_result = reconcile_resumed_dataset(dataset_path)
    if reconcile_result.reset_empty:
        # The whole dataset was quarantined (no durable episodes survived a crash),
        # so there is nothing left to resume -- info.json is gone. Fail loud and tell
        # the operator to re-run, which takes the fresh-create path. We cannot create
        # here (no freq/feature schema in this helper), and proceeding to resume would
        # 404 against the Hub.
        raise RuntimeError(
            f"{dataset_path}: no durable episodes survived the previous crash; the dataset "
            "was reset. Re-run the eval to start a fresh collection (it will re-create the "
            "dataset from scratch)."
        )

    streaming_encoding = _streaming_encoding_enabled()
    dataset = LeRobotDataset.resume(
        repo_id=hf_repo_id,
        root=str(dataset_path),
        image_writer_threads=0 if streaming_encoding else 4,
        streaming_encoding=streaming_encoding,
    )
    dataset = ensure_dataset_can_store_episode_telemetry(
        dataset,
        episode_data,
        dataset_path=dataset_path,
        dataset_name=hf_repo_id,
    )
    record_or_verify_camera_role_serials(dataset, cam_data_keys)
    return dataset


def cleanup_stale_image_episode_dirs(dataset_path: Path) -> list[Path]:
    info_path = dataset_path / "meta" / "info.json"
    images_path = dataset_path / "images"
    if not info_path.exists() or not images_path.exists():
        return []
    info = json.loads(info_path.read_text())
    total_episodes = int(info["total_episodes"])
    removed: list[Path] = []
    for episode_dir in sorted(images_path.glob("*/episode-*")):
        if not episode_dir.is_dir():
            continue
        try:
            episode_index = int(episode_dir.name.removeprefix("episode-"))
        except ValueError as exc:
            raise ValueError(f"Unexpected image episode directory name: {episode_dir}") from exc
        if episode_index >= total_episodes:
            shutil.rmtree(episode_dir)
            removed.append(episode_dir)
    if removed:
        print(
            "Removed stale incomplete image episode directorie(s) beyond "
            f"dataset total_episodes={total_episodes}:"
        )
        for path in removed:
            print(f"  {path}")
    return removed


def _prepare_new_dataset_root(dataset_path: Path) -> Path | None:
    """Make ``dataset_path`` absent for LeRobotDataset.create().

    Real eval scripts write ``results.json`` before the first episode is saved so
    that a false start preserves the hidden round plan. LeRobotDataset.create()
    requires the dataset root not to exist, so we temporarily move that
    metadata-only directory aside and restore the sidecar file after creation.
    Any actual dataset artifact without ``meta/info.json`` is ambiguous and
    remains a hard error.
    """
    if not dataset_path.exists():
        return None
    if not dataset_path.is_dir():
        raise FileExistsError(f"Dataset path exists and is not a directory: {dataset_path}")
    if (dataset_path / "meta" / "info.json").exists():
        raise FileExistsError(
            f"Dataset metadata already exists at {dataset_path / 'meta' / 'info.json'}; "
            "load the existing dataset instead of creating a new one."
        )

    # save_results_file stages results.json through this file; a leftover one is a torn
    # write whose previous results.json (if any) is intact.
    stale_results_tmp = dataset_path / "results.json.tmp"
    if stale_results_tmp.is_file():
        stale_results_tmp.unlink()
        print(f"Removed stale partial results write: {stale_results_tmp}")

    entries = sorted(dataset_path.iterdir(), key=lambda path: path.name)
    disallowed = [
        entry
        for entry in entries
        if not (
            (entry.is_file() and entry.name in _PRECREATE_DATASET_SIDECAR_FILES)
            or (entry.is_dir() and entry.name in _PRECREATE_DATASET_SIDECAR_DIRS)
        )
    ]
    if disallowed:
        disallowed_names = ", ".join(str(path.relative_to(dataset_path)) for path in disallowed)
        raise FileExistsError(
            f"Refusing to create dataset at {dataset_path}: path exists without "
            f"meta/info.json and contains non-sidecar entries: {disallowed_names}"
        )

    if not entries:
        dataset_path.rmdir()
        return None

    backup_root = dataset_path.parent / ".dataset_create_sidecar_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_root / f"{dataset_path.name}_{timestamp}"
    suffix = 1
    while backup_path.exists():
        backup_path = backup_root / f"{dataset_path.name}_{timestamp}_{suffix}"
        suffix += 1

    shutil.move(str(dataset_path), str(backup_path))
    print(
        "Temporarily moved metadata-only dataset directory aside before "
        f"LeRobot creation: {backup_path}"
    )
    return backup_path


def _restore_precreate_sidecars(dataset_path: Path, backup_path: Path | None) -> None:
    if backup_path is None:
        return
    for entry in sorted(backup_path.iterdir(), key=lambda path: path.name):
        is_sidecar_file = entry.is_file() and entry.name in _PRECREATE_DATASET_SIDECAR_FILES
        is_sidecar_dir = entry.is_dir() and entry.name in _PRECREATE_DATASET_SIDECAR_DIRS
        if not (is_sidecar_file or is_sidecar_dir):
            raise RuntimeError(f"Unexpected dataset sidecar backup entry: {entry}")
        destination = dataset_path / entry.name
        if destination.exists():
            raise FileExistsError(
                f"Refusing to restore sidecar {entry.name}; destination already exists: "
                f"{destination}"
            )
        if is_sidecar_dir:
            shutil.copytree(entry, destination)
        else:
            shutil.copy2(entry, destination)
    print(f"Restored pre-existing dataset sidecar file(s) from: {backup_path}")


def checkpoint_dataset(
    dataset: LeRobotDataset,
    save_future: concurrent.futures.Future | None,
    hf_repo_id: str,
    dataset_path: Path,
    *,
    verbose: bool = True,
    consolidate: bool = True,
) -> tuple[LeRobotDataset, int]:
    """Finalize the current dataset and re-open it for continued writing.

    This writes parquet footers and metadata to disk so that the data
    collected so far is recoverable if the process crashes later. After this
    returns, ``info.json``'s ``total_episodes`` is in lockstep with the
    durably-footered parquet data on disk -- which is the invariant that keeps a
    later crash (e.g. a failed robot reset) from silently orphaning episodes
    whose data never got a footer. See :func:`reconcile_resumed_dataset`.

    ``consolidate`` controls the O(n) episode-metadata merge: pass ``False`` for
    the per-episode durability footer on the hot path (the merge re-reads every
    episode-meta parquet, so doing it every episode is O(n^2)); the merge is only
    needed periodically and at shutdown to collapse the per-session meta files
    into one. The cheap finalize+reopen is what makes the data durable, NOT the
    consolidate.

    Returns ``(new_dataset, saved_episode_count)``.
    """
    # Wait for any in-flight background save
    if save_future is not None:
        wait_for_background_save(
            save_future,
            description="Background save during checkpoint",
            timeout=120,
        )

    # Finalize current dataset (writes parquet footers + metadata)
    dataset.stop_image_writer()
    dataset.finalize()

    # Consolidate episodes parquet files to prevent schema mismatches
    # when the dataset was resumed across multiple sessions
    if consolidate:
        from mulligan.data.recording import consolidate_episodes_parquet

        consolidate_episodes_parquet(dataset_path)

    if verbose:
        print(f"Checkpoint: finalized dataset ({dataset.num_episodes} episodes)")

    # Re-open from disk in WRITE mode. FF lerobot makes the bare LeRobotDataset(...)
    # constructor read-only (no writer); resume() rebuilds the DatasetWriter so the
    # post-checkpoint add_frame/save_episode work. Streaming follows the same
    # MULLIGAN_STREAMING_ENCODING knob as dataset creation.
    streaming_encoding = _streaming_encoding_enabled()
    dataset = LeRobotDataset.resume(
        repo_id=hf_repo_id,
        root=str(dataset_path),
        image_writer_threads=0 if streaming_encoding else 4,
        streaming_encoding=streaming_encoding,
    )
    saved_episode_count = dataset.num_episodes
    if verbose:
        print(f"Checkpoint: re-opened dataset ({saved_episode_count} episodes)")

    return dataset, saved_episode_count


def run_with_timeout(fn, seconds: int, description: str):
    """Run *fn()* with a SIGALRM timeout (Linux only).

    Returns the result of *fn()* on success. On timeout, raises TimeoutError
    after printing a prominent error so failed uploads/registrations cannot be
    missed in noisy robot logs.
    Exceptions raised by *fn()* propagate so hardware/IO failures do not look
    like optional timeouts.
    """
    if sys.platform != "linux":
        raise NotImplementedError("run_with_timeout requires Linux SIGALRM support")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("run_with_timeout uses SIGALRM and must run on the main thread")

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return fn()
    except _Timeout:
        print(f"ERROR: {description} timed out after {seconds}s.")
        print("  The operation did not complete; retry or run it manually before trusting results.")
        raise TimeoutError(f"{description} timed out after {seconds}s") from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def hub_push_timeout_s(dataset_root: Path | str) -> int:
    """Timeout budget for a full-dataset Hub push, scaled to dataset size.

    A flat cap would abort large pushes mid-upload (2.8 GB at ~200 kB/s needs
    hours) and skip the finalize chain behind them. Budget the worst case at 100 kB/s sustained
    uplink with a 10-minute floor; the alarm still bounds a wedged push.
    """
    total_bytes = sum(f.stat().st_size for f in Path(dataset_root).rglob("*") if f.is_file())
    return max(600, total_bytes // 100_000)


def _short_policy_name(
    entry: PolicyEntry,
    *,
    policy_names: list[str] | None,
    policy_index: int,
) -> str:
    if policy_names is not None:
        if policy_index < len(policy_names):
            return policy_names[policy_index]
        if entry.policy_id < len(policy_names):
            return policy_names[entry.policy_id]
    model_tail = entry.model_id.rsplit("/", 1)[-1]
    return model_tail.split(":", 1)[0]


def _success_rate_string(results: list[bool]) -> str:
    total = len(results)
    if total == 0:
        return "0/0"
    successes = sum(results)
    return f"{successes}/{total} ({successes / total:.1%})"


def _score_string(records: list[RolloutRecord], num_subtask_marks: int) -> str:
    """``total/max (mean/max_per_ep per ep)`` graded score over a policy's rollouts."""
    max_score = num_subtask_marks + 1
    n = len(records)
    if n == 0:
        return "0/0"
    total = sum(episode_score(rec.outcome, rec.subtask_frames) for rec in records)
    return f"{total}/{n * max_score} ({total / n:.2f}/{max_score} per ep)"


def print_results(
    policies: list[PolicyEntry],
    policy_names: list[str] | None = None,
    *,
    rollout_records: list[RolloutRecord] | None = None,
    num_subtask_marks: int = 0,
):
    """Print a compact final results table with policy short names.

    With ``num_subtask_marks > 0`` (tasks with mid-episode sub-goals, e.g. routing_d2)
    a graded ``Score`` column is added: per episode ``marks + success`` out of
    ``num_subtask_marks + 1``, computed from ``rollout_records`` (which carry the live
    sub-goal marks). The success-rate column stays binary.
    """
    print()
    print("=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)

    graded = num_subtask_marks > 0
    if graded and rollout_records is None:
        raise ValueError("rollout_records are required for the graded score column")

    headers = ["Policy", "Success Rate"]
    if graded:
        headers.append(f"Score (max {num_subtask_marks + 1}/ep)")
    rows = []
    for i, entry in enumerate(policies):
        row = [
            _short_policy_name(entry, policy_names=policy_names, policy_index=i),
            _success_rate_string(entry.results),
        ]
        if graded:
            records = [rec for rec in rollout_records if rec.policy_id == entry.policy_id]
            if len(records) != len(entry.results):
                raise RuntimeError(
                    f"policy_id={entry.policy_id}: {len(records)} rollout records but "
                    f"{len(entry.results)} results; the graded score would not cover the "
                    "same episodes as the success rate"
                )
            row.append(_score_string(records, num_subtask_marks))
        rows.append(row)

    widths = [max(len(header), *(len(row[c]) for row in rows)) for c, header in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        first = f"{cells[0]:<{widths[0]}}"
        rest = [f"{cell:>{widths[c]}}" for c, cell in enumerate(cells) if c > 0]
        return "  ".join([first, *rest])

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    print("=" * 72)


def load_previous_results(
    results_path: Path,
) -> tuple[list[RolloutRecord], dict[str, tuple[int, str]]]:
    """Load rollout records and the policy map from a previous results file.

    Returns ``(previous_records, policy_map)`` where *policy_map* is
    ``{model_id: (policy_id, name)}``; ``([], {})`` if the file does not exist.
    Keys written by older tooling (e.g. ``arena_session_id``) are ignored.
    """
    if not results_path.exists():
        return [], {}

    with open(results_path) as f:
        data = json.load(f)

    records = []
    for r in data.get("rollouts", []):
        # A record carries "model_id" or the "artifact" alias without URI prefix. Either key
        # must be present and non-empty -- an empty model_id is a corrupt rollout
        # record (it would silently produce an unkeyed RolloutRecord), so raise
        # rather than defaulting to "".
        raw_id = r.get("model_id") or r.get("artifact")
        if not raw_id:
            raise ValueError(
                "Corrupt rollout record in "
                f"{results_path}: missing/empty 'model_id' (and 'artifact' "
                f"back-compat fallback). Record: {r}"
            )
        mid = _ensure_model_id_prefix(raw_id)
        if "round" in r:
            round_num = r["round"]
        elif "round_num" in r:
            round_num = r["round_num"]
        else:
            raise KeyError(f"rollout record is missing required round/round_num: {r}")
        records.append(
            RolloutRecord(
                round_num=round_num,
                policy_id=r["policy_id"],
                model_id=mid,
                anonymous_label=r.get("anonymous_label", "?"),
                outcome=r["outcome"],
                num_steps=r["num_steps"],
                episode_index=r["episode_index"],
                manifest_idx=r.get("manifest_idx"),
                pen_x=r.get("pen_x"),
                pen_y=r.get("pen_y"),
                pen_yaw=r.get("pen_yaw"),
                # Genuinely optional: results files written before live sub-goal
                # marks existed carry no key, and no key means no marks.
                subtask_frames=tuple(int(f) for f in r.get("subtask_frames", [])),
                visit_id=r.get("visit_id"),
            )
        )

    # Reconstruct policy map — check both "summary" (new format: list of
    # per-policy dicts) and "policies" (old format: separate list).
    # Keys are now full model_ids (with URI prefix).
    policy_map: dict[str, tuple[int, str]] = {}
    policy_sources = data.get("summary", [])
    if not isinstance(policy_sources, list) or (
        policy_sources and not isinstance(policy_sources[0], dict)
    ):
        # Old format: summary is an aggregate dict, policies is the list
        policy_sources = data.get("policies", [])
    for entry in policy_sources:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("model_id", entry.get("artifact", ""))
        if not raw_id:
            continue
        mid = _ensure_model_id_prefix(raw_id)
        pid = entry["policy_id"]
        name = entry.get("name", mid.rsplit("/", 1)[-1])
        policy_map[mid] = (pid, name)

    return records, policy_map


def save_results_file(
    output_path: Path,
    policies: list[PolicyEntry],
    rollout_records: list[RolloutRecord],
    args: argparse.Namespace,
    dataset_name: str | None,
    policy_names: list[str] | None = None,
    round_plans: list[dict] | None = None,
    phase_stops: list[dict] | None = None,
):
    """Write a JSON results file with per-rollout details and aggregated stats.

    ``round_plans`` (the pre-baked anonymous slot order of every round) and
    ``phase_stops`` (one entry per graceful shutdown of a phased eval) make the
    file the resume ledger of :mod:`mulligan.real.eval.manifest_eval`.
    """

    def jsonable_arg(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [jsonable_arg(item) for item in value]
        if isinstance(value, dict):
            return {key: jsonable_arg(item) for key, item in value.items()}
        return value

    # Per-policy aggregation
    per_policy = []
    for i, entry in enumerate(policies):
        n = len(entry.results)
        wins = sum(entry.results) if n > 0 else 0
        d = {
            "policy_id": entry.policy_id,
            "model_id": entry.model_id,
            # "artifact" mirrors model_id for readers of the results.json schema
            "artifact": entry.model_id,
            "num_rounds": n,
            "successes": wins,
            "failures": n - wins,
            "success_rate": wins / n if n > 0 else None,
        }
        if policy_names is not None and i < len(policy_names):
            d["name"] = policy_names[i]
        per_policy.append(d)

    # Per-rollout details
    rollouts = []
    for rec in rollout_records:
        rollout = {
            "round": rec.round_num,
            "policy_id": rec.policy_id,
            "model_id": rec.model_id,
            # "artifact" mirrors model_id for readers of the results.json schema
            "artifact": rec.model_id,
            "anonymous_label": rec.anonymous_label,
            "outcome": rec.outcome,
            "num_steps": rec.num_steps,
            "episode_index": rec.episode_index,
            "subtask_frames": list(rec.subtask_frames),
        }
        if rec.visit_id is not None:
            rollout["visit_id"] = rec.visit_id
        if rec.manifest_idx is not None:
            rollout.update(
                {
                    "manifest_idx": rec.manifest_idx,
                    "pen_x": rec.pen_x,
                    "pen_y": rec.pen_y,
                    "pen_yaw": rec.pen_yaw,
                }
            )
        rollouts.append(rollout)

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "args": {
            k: ("<redacted>" if k.endswith("_token") and v else jsonable_arg(v))
            for k, v in vars(args).items()
            if k != "func"
        },
        "summary": per_policy,
        "rollouts": rollouts,
    }
    if round_plans is not None:
        results["round_plans"] = round_plans
    if phase_stops is not None:
        results["phase_stops"] = list(phase_stops)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement: a hard kill mid-write must never leave a truncated results.json
    # (the resume ledger + per-record visit provenance live here). Write the full snapshot
    # to a sibling temp file, fsync, then rename over the previous valid file.
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    print(f"\nResults saved to: {output_path}")
