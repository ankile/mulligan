"""Pinned dataset revisions and episode selectors for the real trainers.

Some public round repos merge several source sessions (e.g. ``mulligan/real-marker-d2-r02-eval`` holds
sessions b01-b03), so a repo id plus revision does not always pin the episodes a released checkpoint was
trained on. Both trainers therefore accept

    --dataset-revisions <repo>=<revision> [...]
    --dataset-episodes <repo>=session_id:<id>[,<id>...] [...]
    --dataset-episodes <repo>=episode_index:<i>[,<j>-<k>...] [...]

A ``session_id`` selector is resolved against the repo's public ``meta/episode_provenance.parquet`` at the
pinned revision. ``release/training-views.json`` records the selector of every released real checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SELECTOR_KINDS = ("session_id", "episode_index")
EPISODE_SOURCES_PATH = "meta/episode_provenance.parquet"


@dataclass(frozen=True)
class EpisodeSelector:
    kind: str
    values: tuple

    def __post_init__(self) -> None:
        if self.kind not in SELECTOR_KINDS:
            raise ValueError(
                f"Unknown episode selector kind {self.kind!r}; expected one of {SELECTOR_KINDS}"
            )
        if not self.values:
            raise ValueError(f"Empty {self.kind} selector")

    def to_cli(self) -> str:
        if self.kind == "episode_index":
            return f"episode_index:{_format_index_ranges(self.values)}"
        return f"session_id:{','.join(self.values)}"

    def to_json(self) -> dict[str, list]:
        return {self.kind: list(self.values)}

    @classmethod
    def from_json(cls, obj: Mapping[str, Sequence]) -> EpisodeSelector:
        if len(obj) != 1:
            raise ValueError(f"Episode selector must have exactly one key, got {sorted(obj)}")
        ((kind, values),) = obj.items()
        if kind == "episode_index":
            return cls(kind, tuple(sorted(int(v) for v in values)))
        return cls(kind, tuple(str(v) for v in values))


def _split_repo_assignment(item: str, flag: str) -> tuple[str, str]:
    repo, sep, value = item.partition("=")
    if not sep or not repo or not value:
        raise ValueError(f"{flag} expects <repo>=<value>, got {item!r}")
    return repo.strip(), value.strip()


def _parse_index_ranges(text: str) -> tuple[int, ...]:
    indices: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep:
            start, stop = int(lo), int(hi)
            if stop < start:
                raise ValueError(f"Descending episode range {part!r}")
            indices.update(range(start, stop + 1))
        else:
            indices.add(int(part))
    return tuple(sorted(indices))


def _format_index_ranges(values: Iterable[int]) -> str:
    ordered = sorted(set(int(v) for v in values))
    parts: list[str] = []
    start = prev = None
    for value in ordered:
        if start is None:
            start = prev = value
        elif value == prev + 1:
            prev = value
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = value
    if start is not None:
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def parse_episode_selector(text: str) -> EpisodeSelector:
    kind, sep, values = text.partition(":")
    if not sep:
        raise ValueError(
            f"Episode selector must be session_id:<ids> or episode_index:<ranges>, got {text!r}"
        )
    kind = kind.strip()
    if kind == "episode_index":
        return EpisodeSelector(kind, _parse_index_ranges(values))
    if kind == "session_id":
        return EpisodeSelector(kind, tuple(v.strip() for v in values.split(",") if v.strip()))
    raise ValueError(f"Unknown episode selector kind {kind!r}; expected one of {SELECTOR_KINDS}")


def parse_dataset_revisions(items: Sequence[str] | None) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for item in items or ():
        repo, revision = _split_repo_assignment(item, "--dataset-revisions")
        if repo in revisions and revisions[repo] != revision:
            raise ValueError(
                f"Conflicting --dataset-revisions for {repo}: {revisions[repo]} vs {revision}"
            )
        revisions[repo] = revision
    return revisions


def parse_dataset_episodes(items: Sequence[str] | None) -> dict[str, EpisodeSelector]:
    selectors: dict[str, EpisodeSelector] = {}
    for item in items or ():
        repo, text = _split_repo_assignment(item, "--dataset-episodes")
        if repo in selectors:
            raise ValueError(f"Repeated --dataset-episodes for {repo}")
        selectors[repo] = parse_episode_selector(text)
    return selectors


def validate_against_repo_ids(
    repo_ids: Sequence[str],
    revisions: Mapping[str, str],
    selectors: Mapping[str, EpisodeSelector],
) -> None:
    """Fail on revisions/selectors for repos the run does not read."""
    known = set(repo_ids)
    stray = sorted((set(revisions) | set(selectors)) - known)
    if stray:
        raise ValueError(
            f"--dataset-revisions/--dataset-episodes name repos not in the run's repo ids: {stray}"
        )
    unpinned = sorted(repo for repo in selectors if repo not in revisions)
    if unpinned:
        raise ValueError(
            f"--dataset-episodes needs a pinned --dataset-revisions entry for {unpinned}: "
            "episode indices are only stable at a fixed revision"
        )


def load_episode_provenance(repo_id: str, revision: str, *, local_root: str | Path | None = None):
    """Read ``meta/episode_provenance.parquet`` from a local copy or the Hub (anonymous read)."""
    import pandas as pd

    if local_root is not None:
        path = Path(local_root) / EPISODE_SOURCES_PATH
        if path.is_file():
            return pd.read_parquet(path)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, EPISODE_SOURCES_PATH, repo_type="dataset", revision=revision)
    return pd.read_parquet(path)


def resolve_episode_indices(
    repo_id: str,
    selector: EpisodeSelector,
    revision: str,
    *,
    local_root: str | Path | None = None,
    provenance=None,
) -> list[int]:
    """Return the sorted dataset episode indices a selector picks."""
    if selector.kind == "episode_index":
        return list(selector.values)
    if provenance is None:
        provenance = load_episode_provenance(repo_id, revision, local_root=local_root)
    if "session_id" not in provenance.columns or "episode_index" not in provenance.columns:
        raise ValueError(
            f"{repo_id}@{revision}: {EPISODE_SOURCES_PATH} lacks session_id/episode_index columns"
        )
    wanted = set(selector.values)
    missing = sorted(wanted - set(provenance["session_id"].astype(str)))
    if missing:
        raise ValueError(f"{repo_id}@{revision}: sessions {missing} not in {EPISODE_SOURCES_PATH}")
    picked = provenance.loc[provenance["session_id"].astype(str).isin(wanted), "episode_index"]
    return sorted(int(i) for i in picked)


def selectors_to_json(selectors: Mapping[str, EpisodeSelector]) -> str:
    """Stable JSON for run configs and checkpoint metadata."""
    return json.dumps(
        {repo: sel.to_json() for repo, sel in sorted(selectors.items())}, sort_keys=True
    )
