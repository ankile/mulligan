"""HuggingFace Hub helpers for LeRobot datasets."""

from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import time
from collections.abc import Callable
from typing import TypeVar

from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from lerobot.configs.video import VIDEO_CODECS_ALIASES
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from mulligan.real.collect.hf_utils import DEFAULT_DATASET_LICENSE


_T = TypeVar("_T")


_REPLACE_PREFIXES = (
    "data/",
    "meta/episodes/",
    "videos/",
)
_LOCAL_GENERATED_SHARD_DIRS = (
    "data",
    "meta/episodes",
)


def _hf_retry_attempts() -> int:
    return int(os.environ.get("MULLIGAN_HF_RETRY_ATTEMPTS", "5"))


def _hf_retry_base_sleep_s() -> float:
    return float(os.environ.get("MULLIGAN_HF_RETRY_BASE_SLEEP_S", "10"))


def _is_transient_hf_error(exc: BaseException) -> bool:
    text = repr(exc)
    transient_markers = (
        "SSLError",
        "SSLEOFError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "MaxRetryError",
        "ProtocolError",
        "RemoteDisconnected",
        "temporarily unavailable",
        "502",
        "503",
        "504",
    )
    return any(marker in text for marker in transient_markers)


def retry_transient_hf_operation(label: str, fn: Callable[[], _T]) -> _T:
    """Retry transient HuggingFace/network operations with bounded backoff."""
    attempts = max(1, _hf_retry_attempts())
    base_sleep_s = max(0.0, _hf_retry_base_sleep_s())
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient_hf_error(exc) or attempt == attempts:
                raise
            last_exc = exc
            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            print(
                f"Transient HF/network error during {label} "
                f"(attempt {attempt}/{attempts}); retrying in {sleep_s:.1f}s: {exc!r}",
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"unreachable retry exhaustion for {label}") from last_exc


def advance_lerobot_version_tag(
    repo_id: str,
    *,
    revision: str = "main",
    tag: str = CODEBASE_VERSION,
) -> str:
    """Move the LeRobot codebase-version tag to the selected dataset revision."""
    api = HfApi()
    target_sha = retry_transient_hf_operation(
        f"repo_info({repo_id}@{revision})",
        lambda: (
            api.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            ).sha
        ),
    )

    def _delete() -> None:
        try:
            api.delete_tag(repo_id=repo_id, repo_type="dataset", tag=tag)
        except RevisionNotFoundError:
            pass  # absent, or deleted by an attempt whose response was lost

    # Both calls retry transient errors: a failed create after a successful delete
    # would leave the dataset without its codebase-version tag. exist_ok covers a
    # create whose response was lost; the read-back below confirms the target.
    retry_transient_hf_operation(f"delete_tag({repo_id}:{tag})", _delete)
    retry_transient_hf_operation(
        f"create_tag({repo_id}:{tag})",
        lambda: api.create_tag(
            repo_id=repo_id,
            repo_type="dataset",
            tag=tag,
            revision=target_sha,
            exist_ok=True,
        ),
    )
    tag_sha = retry_transient_hf_operation(
        f"repo_info({repo_id}@{tag})",
        lambda: api.repo_info(repo_id=repo_id, repo_type="dataset", revision=tag).sha,
    )
    if tag_sha != target_sha:
        raise RuntimeError(
            f"{repo_id}: '{tag}' points at {tag_sha[:8]} after the move, expected "
            f"{target_sha[:8]} ({revision}); another writer moved it. Re-run "
            f"advance_lerobot_version_tag({repo_id!r})."
        )
    print(f"  moved {repo_id}:{tag} -> {target_sha[:8]} ({revision})")
    return target_sha


def assert_lerobot_version_tag_tracks(
    repo_id: str,
    *,
    branch: str | None = None,
    tag: str = CODEBASE_VERSION,
    warn_only: bool = False,
) -> None:
    """Fail loudly unless the codebase-version tag points at the pushed branch HEAD.

    Training and metadata reads resolve the blessed ``CODEBASE_VERSION`` (v3.0) tag,
    not raw ``main`` (see :func:`mulligan.real.train.hub_data.sync_datasets`).
    ``LeRobotDataset.push_to_hub`` defaults to ``tag_version=True``, which moves v3.0
    onto the just-pushed commit — but
    that guarantee rests on a vendored, fast-churning upstream default. If a LeRobot bump
    flips it (or a caller passes ``tag_version=False`` / a non-default branch), pushed data
    silently never reaches v3.0-pinned training. This converts that implicit upstream
    contract into an explicit, asserted local one.

    ``warn_only=True`` downgrades a mismatch to a loud WARNING instead of raising. Use it
    on the EVAL push path: the eval is already done and the dataset already pushed, so a
    momentary tag desync (or a push that overran ``run_with_timeout``) should not crash the
    end of a long human-in-the-loop session — it is a recoverable follow-up, not corrupted
    training input. Collection/training-data producers keep the hard raise (default), since
    stale training data would go unnoticed.
    """
    api = HfApi()
    target_rev = branch or "main"
    target_sha = retry_transient_hf_operation(
        f"repo_info({repo_id}@{target_rev})",
        lambda: api.repo_info(repo_id=repo_id, repo_type="dataset", revision=target_rev).sha,
    )
    tag_sha = retry_transient_hf_operation(
        f"repo_info({repo_id}@{tag})",
        lambda: api.repo_info(repo_id=repo_id, repo_type="dataset", revision=tag).sha,
    )
    if tag_sha != target_sha:
        msg = (
            f"{repo_id}: '{tag}' tag points at {tag_sha[:8]} but '{target_rev}' HEAD is "
            f"{target_sha[:8]} after push. v3.0-pinned training/eval reads would load STALE "
            f"data. push_to_hub(tag_version=True) should keep them in sync; a vendored "
            f"LeRobot default change or a stale tag is the likely cause. Repair with "
            f"advance_lerobot_version_tag({repo_id!r})."
        )
        if warn_only:
            print(f"WARNING: {msg}", flush=True)
            return
        raise RuntimeError(msg)


def push_lerobot_dataset_tagged_main(dataset, **push_kwargs) -> None:
    """Push a LeRobot dataset to main and assert the v3.0 tag tracks the pushed commit.

    Thin wrapper over ``dataset.push_to_hub`` that makes the "every push blesses v3.0"
    invariant explicit and asserts it (:func:`assert_lerobot_version_tag_tracks`). Use this
    for real-robot dataset producers (collection, eval rollouts) so a LeRobot default flip
    trips immediately instead of silently stranding v3.0-pinned training. Editors that push
    via raw ``HfApi.upload_file`` (e.g. ``mulligan.tools.outcome_review``) must keep calling
    :func:`advance_lerobot_version_tag` directly — ``upload_file`` does not tag.

    Before the push, remote generated shards absent from the local tree are deleted
    (main-branch pushes only): the producer's local dataset is the complete dataset, and
    ``push_to_hub`` alone never removes the ``meta/episodes`` shard an interrupted session
    left on the Hub.

    The dataset card's license defaults to ``DEFAULT_DATASET_LICENSE`` (mit), not LeRobot's
    apache-2.0.
    """
    if push_kwargs.get("tag_version") is False:
        raise ValueError(
            "push_lerobot_dataset_tagged_main requires the v3.0 tag to track the pushed "
            "commit; tag_version=False would orphan v3.0-pinned training/eval reads."
        )
    push_kwargs.setdefault("tag_version", True)
    push_kwargs.setdefault("license", DEFAULT_DATASET_LICENSE)
    branch = push_kwargs.get("branch")
    repo_id = dataset.repo_id
    # A producer's local tree IS the whole dataset, so any remote generated shard it does
    # not carry is stale (see delete_stale_remote_generated_shards for how resumed
    # collections leave duplicate meta/episodes shards behind).
    if branch is None:
        delete_stale_remote_generated_shards(repo_id, Path(dataset.root))
    dataset.push_to_hub(**push_kwargs)
    assert_lerobot_version_tag_tracks(repo_id, branch=branch)


def refresh_lerobot_dataset_from_main(
    repo_id: str,
    *,
    root: str | Path | None = None,
    download_videos: bool = True,
    revision: str = "main",
) -> LeRobotDataset:
    """Load a mutable HF dataset from current main after pruning stale local shards."""
    dataset_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    for rel_path in _LOCAL_GENERATED_SHARD_DIRS:
        path = dataset_root / rel_path
        if path.exists():
            shutil.rmtree(path)
    return retry_transient_hf_operation(
        f"refresh_lerobot_dataset_from_main({repo_id}@{revision})",
        lambda: LeRobotDataset(
            repo_id=repo_id,
            root=str(dataset_root),
            revision=revision,
            force_cache_sync=True,
            download_videos=download_videos,
        ),
    )


def _video_codecs_in_info(info: dict) -> list[str]:
    """Codec strings of every video feature in a LeRobot ``meta/info.json`` payload.

    Returns raw, un-aliased strings (``"av1"``/``"h264"``/``"libsvtav1"``) so a caller
    can report exactly what the file says. An empty list means the dataset carries no
    video features at all.
    """
    codecs = []
    for feature in info.get("features", {}).values():
        if feature.get("dtype") != "video":
            continue
        # A video feature without a recorded codec is a broken info.json, not an
        # optional field: name it here rather than comparing against None.
        codecs.append(feature["info"]["video.codec"])
    return codecs


def _remote_video_codecs(repo_id: str) -> list[str] | None:
    """Video codecs recorded in the Hub dataset's ``meta/info.json`` at ``main``.

    ``None`` means "nothing to compare against": the repo does not exist yet, or it
    exists without a ``meta/info.json``. The read is pinned to the resolved ``main``
    commit so a stale local HF cache cannot answer for the Hub.
    """
    api = HfApi()
    try:
        main_sha = retry_transient_hf_operation(
            f"repo_info({repo_id}@main)",
            lambda: api.repo_info(repo_id=repo_id, repo_type="dataset", revision="main").sha,
        )
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return None
    try:
        info_path = retry_transient_hf_operation(
            f"hf_hub_download({repo_id}@{main_sha[:8]} meta/info.json)",
            lambda: hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="meta/info.json",
                revision=main_sha,
            ),
        )
    except EntryNotFoundError:
        return None
    with open(info_path) as handle:
        return _video_codecs_in_info(json.load(handle))


def assert_remote_codec_matches(repo_id: str, root: Path, *, replace_remote_codec: bool) -> None:
    """Refuse to overwrite a Hub dataset whose videos were written with another codec.

    The splitter writes one codec (LeRobot's RGB default, AV1), but some datasets on
    the Hub were written with h264. Overwriting one would replace every remote video
    with different pixels under the same repo id and advance the ``v3.0`` tag, so every
    v3.0-pinned reader silently switches codec. That is a new dataset, not a re-push:
    it needs a new repo id, or the explicit ``--replace-remote-codec`` acknowledgement.

    No-ops when there is nothing to compare: a repo that does not exist yet, a remote
    without ``meta/info.json``, or either side carrying no video features.
    """
    local_info_path = root / "meta" / "info.json"
    if not local_info_path.exists():
        raise RuntimeError(f"{repo_id}: no meta/info.json at {root}; nothing to push")
    local_codecs = _video_codecs_in_info(json.loads(local_info_path.read_text()))
    if not local_codecs:
        return
    remote_codecs = _remote_video_codecs(repo_id)
    if not remote_codecs:
        return

    # "av1" (stream name) and "libsvtav1" (encoder name) are the same codec; LeRobot's
    # own alias table maps between them, so a same-codec re-push stays a no-op here.
    def canonical(codecs: list[str]) -> set[str]:
        return {VIDEO_CODECS_ALIASES.get(c, c) for c in codecs}

    if canonical(remote_codecs) == canonical(local_codecs):
        return
    remote_names = ", ".join(sorted(set(remote_codecs)))
    local_names = ", ".join(sorted(set(local_codecs)))
    message = (
        f"CODEC GUARD: {repo_id} on the Hub was written with video codec "
        f"'{remote_names}', but this local dataset is '{local_names}'. Pushing would "
        f"replace every remote video with different pixels under the same repo id and "
        f"advance the {CODEBASE_VERSION} tag. Push to a new repo id, or pass "
        f"--replace-remote-codec to re-encode the remote dataset on purpose."
    )
    if not replace_remote_codec:
        raise RuntimeError(message)
    print(f"  {message}\n  --replace-remote-codec passed: replacing anyway.", flush=True)


def delete_stale_remote_generated_shards(
    repo_id: str, root: Path, *, api: HfApi | None = None
) -> list[str]:
    """Delete remote generated shards (``data/``, ``meta/episodes/``, ``videos/``) absent locally.

    ``LeRobotDataset.push_to_hub()`` is an ``upload_folder``: it never removes remote files.
    Two producers leave stale generated shards behind otherwise: split datasets regenerated
    with fewer shards, and multi-session collections — LeRobot opens a NEW ``meta/episodes``
    shard on every resume and the collectors consolidate them into ``file-000`` only on a deliberate
    exit, so an interrupted session's shard is pushed and then orphaned by the next
    consolidation, and LeRobot's globbing loader reads the duplicate episodes-meta rows as
    ghost episodes. Regular metadata and sidecars are overwritten by the push itself and are left
    alone. Returns the deleted paths (empty when nothing was stale).
    """
    api = api or HfApi()
    root = Path(root)
    try:
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return []
    local_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    stale_files = sorted(
        path
        for path in remote_files
        if path.startswith(_REPLACE_PREFIXES) and path not in local_files
    )
    if stale_files:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=path) for path in stale_files],
            commit_message="Remove stale LeRobot generated shards before replacement push",
        )
        print(f"  deleted {len(stale_files)} stale remote shard file(s) from {repo_id}")
    return stale_files


def push_lerobot_dataset_replacing_remote(
    dataset, *, replace_remote_codec: bool = False, **push_kwargs
) -> None:
    """Push a LeRobot dataset and delete stale remote shards first.

    ``LeRobotDataset.push_to_hub()`` uploads the local folder but does not remove
    remote files that are absent locally. For split datasets that are regenerated
    with fewer parquet/video shards, those stale files make HF dataset row counts
    disagree with ``meta/info.json``. Delete only generated shard prefixes before
    pushing; regular metadata and sidecars are overwritten by the push itself.

    ``replace_remote_codec`` is the only override for :func:`assert_remote_codec_matches`,
    which runs first and refuses a codec-changing overwrite. The dataset card's license
    defaults to ``DEFAULT_DATASET_LICENSE`` (mit), not LeRobot's apache-2.0.
    """
    if push_kwargs.get("branch") not in (None, "main"):
        raise ValueError(
            "push_lerobot_dataset_replacing_remote replaces main (stale-shard deletion and "
            f"the {CODEBASE_VERSION} tag move act on main); got branch="
            f"{push_kwargs['branch']!r}. Use push_lerobot_dataset_tagged_main for branches."
        )
    push_kwargs.setdefault("license", DEFAULT_DATASET_LICENSE)
    api = HfApi()
    repo_id = dataset.repo_id
    root = Path(dataset.root)

    assert_remote_codec_matches(repo_id, root, replace_remote_codec=replace_remote_codec)
    delete_stale_remote_generated_shards(repo_id, root, api=api)
    dataset.push_to_hub(**push_kwargs)
    advance_lerobot_version_tag(repo_id)
