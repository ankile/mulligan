"""Download released Mulligan datasets and models at their pinned revisions.

Every repo is read at the revision in ``release/revisions.json`` (never ``main``). Selection
uses the release manifests ``release/datasets.json`` and ``release/models.json``.

    python -m mulligan.release.download datasets --task real-marker-d2 --role evaluation --round 5
    python -m mulligan.release.download datasets --task sim-square-narrow --method mulligan --meta-only
    python -m mulligan.release.download models --task real-square-d2 --kind idql-critic --dry-run
    python -m mulligan.release.download repo mulligan/real-routing-d2-r00-r05-eval --include "meta/*"

Filters combine with AND; each accepts several values (OR). ``--round N`` matches a dataset's
model round, collection round, or round-dataset round, and a model's round (``rNN``, ``cNN``,
``velocity-rNN``). ``--method`` matches a dataset's variant exactly, and a model's method exactly
or with a learner suffix (``mulligan`` matches ``mulligan-dp`` and ``mulligan-idql``, not the
ablation ``mulligan-no-cf-dp``).
Reads are anonymous unless ``HF_TOKEN`` is set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download

# model method = arm, or arm + learner for the real-robot checkpoints (mulligan-dp, mulligan-idql)
LEARNERS = ("dp", "idql")

RELEASE_DIR = Path(__file__).resolve().parents[2] / "release"
META_PATTERNS = ["README.md", "meta/**", "*.json", "*.csv"]


@cache
def _load(name: str) -> dict:
    path = RELEASE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run from a checkout of the release repo")
    return json.loads(path.read_text())


def load_revisions() -> dict[str, dict]:
    """``repo -> {type, revision, tag, ...}`` from ``release/revisions.json``."""
    return _load("revisions.json")["repos"]


def pinned_revision(repo: str) -> str:
    """The canonical revision of a released ``mulligan/*`` repo."""
    try:
        return load_revisions()[repo]["revision"]
    except KeyError:
        raise KeyError(f"{repo} is not a released repo (release/revisions.json)") from None


def repo_type(repo: str) -> str:
    try:
        return load_revisions()[repo]["type"]
    except KeyError:
        raise KeyError(f"{repo} is not a released repo (release/revisions.json)") from None


def _round_number(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    m = re.search(r"(\d+)$", value)
    return int(m[1]) if m else None


def _matches(value, wanted: Iterable | None) -> bool:
    return not wanted or value in set(wanted)


def select_datasets(
    task: Iterable[str] | None = None,
    role: Iterable[str] | None = None,
    round: Iterable[int] | None = None,
    method: Iterable[str] | None = None,
) -> list[dict]:
    """Rows of ``release/datasets.json`` that match every given filter."""
    rounds = set(round or ())
    out = []
    for row in _load("datasets.json")["datasets"]:
        if not (_matches(row["task"], task) and _matches(row["role"], role)):
            continue
        if not _matches(row["variant"], method):
            continue
        if rounds:
            have = set(row["model_rounds"])
            if row["collection_round"] is not None:
                have.add(row["collection_round"])
            rd = (row.get("round_dataset") or {}).get("round", "")
            have |= {int(x) for x in re.findall(r"R(\d+)", rd)}
            if rd == "R0-R5":
                have |= set(range(6))
            if not have & rounds:
                continue
        out.append(row)
    return out


def select_models(
    task: Iterable[str] | None = None,
    kind: Iterable[str] | None = None,
    round: Iterable[int] | None = None,
    method: Iterable[str] | None = None,
) -> list[dict]:
    """Rows of ``release/models.json`` with at least one checkpoint matching every filter."""
    rounds = set(round or ())
    out = []
    for row in _load("models.json")["models"]:
        for ck in row["checkpoints"]:
            if not (_matches(ck["task"], task) and _matches(ck["kind"], kind)):
                continue
            if method and not any(
                ck["method"] in (m, *(f"{m}-{learner}" for learner in LEARNERS)) for m in method
            ):
                continue
            if rounds and _round_number(ck["round"]) not in rounds:
                continue
            out.append(row)
            break
    return out


def download(
    repo: str,
    *,
    include: list[str] | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Path:
    """``snapshot_download`` of ``repo`` at its pinned revision; returns the snapshot path."""
    return Path(
        snapshot_download(
            repo,
            repo_type=repo_type(repo),
            revision=pinned_revision(repo),
            allow_patterns=include,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="what", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--include", action="append", help="glob of files to fetch (repeatable)")
    common.add_argument("--meta-only", action="store_true", help=f"only {', '.join(META_PATTERNS)}")
    common.add_argument("--cache-dir", type=Path, help="huggingface_hub cache directory")
    common.add_argument(
        "--local-dir", type=Path, help="write repos to <local-dir>/<repo name> instead of the cache"
    )
    common.add_argument("--dry-run", action="store_true", help="list repos and revisions only")
    d = sub.add_parser("datasets", parents=[common], help="released datasets")
    d.add_argument("--task", action="append")
    d.add_argument("--role", action="append")
    d.add_argument("--round", action="append", type=int)
    d.add_argument("--method", action="append", help="dataset variant, e.g. baseline, mulligan")
    m = sub.add_parser("models", parents=[common], help="released model repos")
    m.add_argument("--task", action="append")
    m.add_argument("--kind", action="append", help="dp-actor, idql-critic, idql-agent, divl-agent")
    m.add_argument("--round", action="append", type=int)
    m.add_argument("--method", action="append")
    r = sub.add_parser("repo", parents=[common], help="one or more repos by id")
    r.add_argument("repos", nargs="+")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.what == "datasets":
        repos = [r["repo"] for r in select_datasets(args.task, args.role, args.round, args.method)]
    elif args.what == "models":
        repos = [r["repo"] for r in select_models(args.task, args.kind, args.round, args.method)]
    else:
        repos = args.repos
    if not repos:
        print("no released repo matches the filters", file=sys.stderr)
        return 1
    include = (args.include or []) + (META_PATTERNS if args.meta_only else [])
    for repo in repos:
        rev = pinned_revision(repo)
        if args.dry_run:
            print(f"{repo}\t{repo_type(repo)}\t{rev}")
            continue
        local = args.local_dir / repo.split("/", 1)[1] if args.local_dir else None
        path = download(repo, include=include or None, cache_dir=args.cache_dir, local_dir=local)
        print(f"{repo}@{rev[:12]}\t{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
