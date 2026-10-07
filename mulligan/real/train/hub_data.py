"""Hub I/O shared by the real trainers: dataset sync at pinned revisions, episode selection,
and resolution of the frozen DP encoder source for the critic."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import lerobot.datasets.multi_dataset as lerobot_multi_dataset
from lerobot.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from mulligan.real.train.dataset_selectors import (
    EpisodeSelector,
    parse_dataset_episodes,
    parse_dataset_revisions,
    resolve_episode_indices,
    validate_against_repo_ids,
)

# Unpinned repos resolve LeRobot's codebase-version tag, never raw ``main``.
DEFAULT_DATASET_REVISION = CODEBASE_VERSION
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HF_PREFIX = "hf://"
WANDB_SCHEME = "wandb://"
# [entity/]project/name:version (W&B artifact paths always carry a version or alias).
_WANDB_ARTIFACT_RE = re.compile(r"^(?:[^/:]+/)?[^/:]+/[^/:]+:[^/:]+$")


def dataset_dir(parent_root: Path | None, repo_id: str) -> Path:
    """Concrete LeRobot dataset directory for ``repo_id`` (``HF_LEROBOT_HOME`` by default)."""
    return (Path(parent_root) if parent_root is not None else HF_LEROBOT_HOME) / repo_id


def dataset_revision(revisions: Mapping[str, str], repo_id: str) -> str:
    return revisions.get(repo_id, DEFAULT_DATASET_REVISION)


def sync_datasets(
    repo_ids: Sequence[str], parent_root: Path | None, revisions: Mapping[str, str]
) -> Path:
    """Publish immutable snapshots under a per-revision lock and return their layout.

    Each run reads a layout of symlinks to complete snapshots. Different revisions
    never replace files beneath an active reader, and failed downloads stay private.
    Snapshots are shared across runs with overlapping dataset selections.
    """
    cache = (Path(parent_root) if parent_root is not None else HF_LEROBOT_HOME).resolve()
    cache = cache / ".mulligan-pinned"
    commits = {
        repo: resolve_dataset_commit(repo, dataset_revision(revisions, repo))
        for repo in sorted(set(repo_ids))
    }
    layout_id = hashlib.sha256(json.dumps(commits, sort_keys=True).encode()).hexdigest()
    layout = cache / "layouts" / layout_id
    for repo_id, commit in commits.items():
        snapshot = cache / "snapshots" / repo_id / commit
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with (snapshot.parent / f"{commit}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not snapshot.exists():
                print(f"Syncing dataset {repo_id}@{commit} -> {snapshot}")
                with tempfile.TemporaryDirectory(dir=snapshot.parent, prefix=".download-") as tmp:
                    staged = Path(tmp) / "dataset"
                    LeRobotDataset(
                        repo_id,
                        root=staged,
                        revision=commit,
                        force_cache_sync=True,
                        download_videos=True,
                    )
                    os.replace(staged, snapshot)
            link = layout / repo_id
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(snapshot, target_is_directory=True)
    return layout


def add_dataset_revision_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--dataset-revisions`` and ``--dataset-episodes`` (see :func:`parse_dataset_pins`)."""
    parser.add_argument(
        "--dataset-revisions",
        nargs="+",
        action="extend",
        default=None,
        metavar="REPO=REVISION",
        help=(
            "Pin the Hub revision (commit sha or tag) of training/eval repos. Unpinned repos "
            f"resolve the {DEFAULT_DATASET_REVISION!r} tag. Repeatable."
        ),
    )
    parser.add_argument(
        "--dataset-episodes",
        nargs="+",
        action="extend",
        default=None,
        metavar="REPO=SELECTOR",
        help=(
            "Read only these episodes of a repo: 'session_id:<id>[,<id>...]' (resolved via "
            "meta/episode_provenance.parquet) or 'episode_index:<i>[,<j>-<k>...]'. The repo "
            "must also be pinned with --dataset-revisions. Repeatable; one selector per repo."
        ),
    )


def parse_dataset_pins(
    revision_items: Sequence[str] | None,
    episode_items: Sequence[str] | None,
    repo_ids: Sequence[str],
) -> tuple[dict[str, str], dict[str, EpisodeSelector]]:
    """Parse and validate ``--dataset-revisions`` / ``--dataset-episodes`` for a run."""
    revisions = parse_dataset_revisions(revision_items)
    selectors = parse_dataset_episodes(episode_items)
    validate_against_repo_ids(repo_ids, revisions, selectors)
    return revisions, selectors


def prepare_datasets(
    repo_ids: Sequence[str],
    parent_root: Path | None,
    revisions: Mapping[str, str],
    *,
    sync: bool,
) -> Path | None:
    """Return the synced layout, or validate and return the original local root."""
    if sync:
        return sync_datasets(repo_ids, parent_root, revisions)
    for repo_id in sorted(set(repo_ids)):
        root = dataset_dir(parent_root, repo_id)
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"--no-dataset-sync reads local dataset copies, but {root} has no meta/info.json"
            )
        print(f"Dataset {repo_id}: using the local copy at {root} as is (--no-dataset-sync)")

    return parent_root


def dataset_commits(
    repo_ids: Sequence[str], revisions: Mapping[str, str], *, synced: bool
) -> dict[str, str | None]:
    """Hub commit each repo was read at; ``None`` for local copies read without a sync."""
    return {
        repo_id: resolve_dataset_commit(repo_id, dataset_revision(revisions, repo_id))
        if synced
        else None
        for repo_id in repo_ids
    }


@contextlib.contextmanager
def _pinned_subdatasets(revisions: Mapping[str, str]):
    original = lerobot_multi_dataset.LeRobotDataset

    def pinned(repo_id, *args, **kwargs):
        return original(repo_id, *args, revision=dataset_revision(revisions, repo_id), **kwargs)

    lerobot_multi_dataset.LeRobotDataset = pinned
    try:
        yield
    finally:
        lerobot_multi_dataset.LeRobotDataset = original


def load_multi_dataset(
    repo_ids: Sequence[str], parent_root: Path | None, revisions: Mapping[str, str], **kwargs
) -> MultiLeRobotDataset:
    """``MultiLeRobotDataset`` whose sub-datasets load at their pinned revisions (LeRobot's
    constructor takes no revision, so a missing local file would come from the default tag)."""
    with _pinned_subdatasets(revisions):
        return MultiLeRobotDataset(repo_ids=list(repo_ids), root=parent_root, **kwargs)


def resolve_selected_episodes(
    repo_ids: Sequence[str],
    parent_root: Path | None,
    revisions: Mapping[str, str],
    selectors: Mapping[str, EpisodeSelector],
) -> dict[str, list[int] | None]:
    """Episode indices each repo contributes (``None`` = every episode).

    Reads the local copy, so call :func:`prepare_datasets` first.
    """
    selected: dict[str, list[int] | None] = {}
    for repo_id in repo_ids:
        selector = selectors.get(repo_id)
        if selector is None:
            selected[repo_id] = None
            continue
        root = dataset_dir(parent_root, repo_id)
        indices = resolve_episode_indices(repo_id, selector, revisions[repo_id], local_root=root)
        total = LeRobotDatasetMetadata(
            repo_id, root=root, revision=revisions[repo_id]
        ).total_episodes
        out_of_range = [i for i in indices if not 0 <= i < total]
        if out_of_range:
            raise ValueError(
                f"--dataset-episodes {repo_id}: episode indices {out_of_range[:10]} are outside "
                f"[0, {total}) at revision {revisions[repo_id]}"
            )
        if not indices:
            raise ValueError(f"--dataset-episodes {repo_id} selected no episodes")
        print(f"Episode selector {repo_id}: {selector.to_cli()} -> {len(indices)} episode(s)")
        selected[repo_id] = indices
    return selected


def is_commit_sha(revision: str) -> bool:
    return bool(_SHA_RE.match(revision))


def resolve_dataset_commit(repo_id: str, revision: str) -> str:
    """Commit sha a dataset revision (tag, branch or sha) points to on the Hub."""
    if is_commit_sha(revision):
        return revision
    from huggingface_hub import HfApi

    return HfApi().repo_info(repo_id, repo_type="dataset", revision=revision).sha


def resolve_encoder_source(spec: str) -> tuple[Path, str]:
    """Resolve ``--encoder-artifact`` to a local DP checkpoint dir and its recorded source.

    Accepts ``hf://<repo>[@<revision>]`` (recorded as ``hf://<repo>@<commit>``), a local
    directory (recorded as its absolute path), or, when ``wandb`` is installed, a W&B
    artifact ``[wandb://]<entity>/<project>/<name>:<version>`` (recorded with the ``wandb://``
    prefix, the model-id form the deploy loader reads).
    """
    if spec.startswith(HF_PREFIX):
        from huggingface_hub import HfApi, snapshot_download

        from mulligan.release.hub import default_revision, parse_hf_uri

        ref = parse_hf_uri(spec)
        if ref.subfolder is not None:
            raise ValueError(f"--encoder-artifact {spec!r}: pass the DP repo root, not a subfolder")
        repo = ref.repo_id
        revision = ref.revision or default_revision(repo)
        commit = HfApi().model_info(repo, revision=revision).sha
        local = Path(snapshot_download(repo, revision=commit))
        return local, f"{HF_PREFIX}{repo}@{commit}"
    if spec.startswith(WANDB_SCHEME):
        return _wandb_encoder_source(spec.removeprefix(WANDB_SCHEME))
    path = Path(spec)
    if path.is_dir():
        return path.resolve(), str(path.resolve())
    if path.exists():
        raise ValueError(f"--encoder-artifact {spec!r} is a file; pass the DP checkpoint directory")
    if not _WANDB_ARTIFACT_RE.match(spec):
        raise FileNotFoundError(
            f"--encoder-artifact {spec!r}: no such local directory, and not a W&B artifact "
            "path ([entity/]project/name:version) or hf://<repo>[@<rev>]"
        )
    return _wandb_encoder_source(spec)


def _wandb_encoder_source(artifact: str) -> tuple[Path, str]:
    try:
        import wandb
    except ImportError as exc:
        raise ValueError(
            f"--encoder-artifact {artifact!r} is neither hf://<repo>[@<rev>] nor a local "
            "directory; W&B artifact paths need the optional wandb package"
        ) from exc
    local = Path(wandb.Api().artifact(artifact).download())
    return local, f"{WANDB_SCHEME}{artifact}"
