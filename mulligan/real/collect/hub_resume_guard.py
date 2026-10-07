"""Safety checks for resuming real-robot datasets from local disk vs Hub state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError


@dataclass(frozen=True)
class HubResumeCheck:
    repo_id: str
    local_total_episodes: int
    remote_total_episodes: int | None


_EMPTY_REPO_SENTINEL_FILES = frozenset({".gitattributes"})


def read_local_total_episodes(dataset_path: Path) -> int:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return 0
    info = json.loads(info_path.read_text())
    return int(info["total_episodes"])


def assert_local_episode_count_matches_ledger(
    *,
    dataset_path: str | Path,
    ledger_count: int,
    ledger_path: str | Path | None = None,
) -> int:
    """Fail if a manifest ledger and local dataset metadata disagree."""
    dataset_root = Path(dataset_path)
    info_path = dataset_root / "meta" / "info.json"
    ledger_label = str(ledger_path) if ledger_path is not None else "manifest ledger"

    if not info_path.exists():
        if ledger_count:
            raise RuntimeError(
                f"{ledger_label} has {ledger_count} row(s), but {info_path} does not exist; "
                "refusing to start collection from an ambiguous resume state."
            )
        return 0

    local_total = read_local_total_episodes(dataset_root)
    if local_total != ledger_count:
        raise RuntimeError(
            f"{dataset_root}: meta/info.json reports {local_total} episode(s), but "
            f"{ledger_label} has {ledger_count} row(s). Refusing to start collection "
            "because manifest resume order would be ambiguous."
        )
    return local_total


def _read_remote_total_episodes(
    repo_id: str,
    *,
    revision: str,
    remote_info_loader: Callable[[str, str], dict] | None = None,
) -> int:
    if remote_info_loader is None:

        def remote_info_loader(repo_id: str, revision: str) -> dict:
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="meta/info.json",
                revision=revision,
            )
            return json.loads(Path(path).read_text())

    info = remote_info_loader(repo_id, revision)
    return int(info["total_episodes"])


def assert_local_dataset_not_behind_hub(
    *,
    repo_id: str,
    dataset_path: str | Path,
    revision: str = "main",
    allow_empty_repo: bool = False,
    api: Any | None = None,
    remote_info_loader: Callable[[str, str], dict] | None = None,
) -> HubResumeCheck:
    """Fail loudly if local data is shorter than the same Hub dataset.

    Real collection scripts often push after a long session. If the local dataset
    folder is missing or restarted while the Hub copy is ahead, a later push from
    the shorter local folder can strand or overwrite hours of trajectories. This
    guard checks the cheap metadata path before collection/push and refuses to
    proceed when the Hub is the newer source of truth.

    ``repo_id`` must be a full ``NAMESPACE/NAME`` id (see
    :func:`mulligan.real.collect.hf_utils.resolve_push_repo_id`).
    """
    if "/" not in repo_id:
        raise ValueError(f"repo_id must be NAMESPACE/NAME, got {repo_id!r}")
    dataset_root = Path(dataset_path)
    local_total = read_local_total_episodes(dataset_root)

    api = api or HfApi()
    try:
        remote_total = _read_remote_total_episodes(
            repo_id,
            revision=revision,
            remote_info_loader=remote_info_loader,
        )
    except RepositoryNotFoundError:
        return HubResumeCheck(
            repo_id=repo_id,
            local_total_episodes=local_total,
            remote_total_episodes=None,
        )
    except EntryNotFoundError as exc:
        if allow_empty_repo:
            remote_files = set(
                api.list_repo_files(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                )
            )
            content_files = sorted(remote_files - _EMPTY_REPO_SENTINEL_FILES)
            if not content_files:
                return HubResumeCheck(
                    repo_id=repo_id,
                    local_total_episodes=local_total,
                    remote_total_episodes=0,
                )
            sample = ", ".join(content_files[:5])
            raise RuntimeError(
                f"{repo_id}@{revision} exists but has no meta/info.json; found "
                f"{len(content_files)} non-empty-repo file(s), including {sample!r}. "
                "Refusing to treat it as a safe resume target."
            ) from exc
        raise RuntimeError(
            f"{repo_id}@{revision} exists but has no meta/info.json; refusing to treat "
            "it as a safe resume target."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not verify Hub resume safety for {repo_id}@{revision} before "
            f"collection/push from {dataset_root}. Refusing to continue because a "
            "shorter local dataset could overwrite a longer Hub dataset. Fix network/auth "
            "and retry."
        ) from exc

    if remote_total > local_total:
        raise RuntimeError(
            f"Refusing to collect/push from {dataset_root}: local dataset has "
            f"{local_total} episode(s), but HuggingFace Hub {repo_id}@{revision} has "
            f"{remote_total}. Restore/download the Hub dataset locally before resuming, "
            "or choose a new DATASET_NAME. This prevents overwriting already-collected "
            "trajectories with a restarted local folder."
        )

    return HubResumeCheck(
        repo_id=repo_id,
        local_total_episodes=local_total,
        remote_total_episodes=remote_total,
    )
