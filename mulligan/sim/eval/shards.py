"""Sharded grid-eval outputs: status, merge of many directories, completeness check.

``python -m mulligan.sim.eval.grid_eval eval --shard-idx K --n-shards N`` writes
``results_shard_K_of_N.json`` plus the ``point_results_shard_K_of_N.json`` and
``cell_results_shard_K_of_N.json`` sidecars once shard K has evaluated its whole
point slice (an unfinished shard has only ``resume_shard_K_of_N.json``). A
directory is ready to merge when every shard 1..N is present and complete.

The command line is ``python -m mulligan.sim.eval.grid_eval merge`` (see its
``--help``); this module holds the logic behind it.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mulligan.sim.eval.grid_eval import RESULTS_SCHEMA, merge_shards

_SHARD_NAME = re.compile(r"^results_shard_(\d+)_of_(\d+)\.json$")


@dataclass
class ShardFileStatus:
    path: str
    idx: int | None
    n_shards: int | None
    expected_points: int | None
    actual_points: int
    complete: bool
    parse_error: str = ""
    # Sidecars the summary declares (point_results_path / cell_results_path) that are
    # missing on disk. The evaluator writes them before the summary, so a present
    # summary with a missing sidecar means the detail files were deleted later.
    missing_detail_sidecars: list[str] = field(default_factory=list)


@dataclass
class ShardDirStatus:
    directory: str
    shards: list[ShardFileStatus]
    expected_n_shards: int | None
    missing_shards: list[int]
    incomplete_shards: list[int]
    ready_to_merge: bool
    detail_loss_shards: dict[int, list[str]] = field(default_factory=dict)


def _shard_index_from_name(path: Path) -> tuple[int | None, int | None]:
    match = _SHARD_NAME.match(path.name)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _declared_sidecar(path: Path, value: Any) -> Path:
    declared = Path(str(value))
    if declared.is_absolute():
        return declared
    # The evaluator writes sidecars next to the summary; resolve relative to it,
    # never to the current directory.
    return path.parent / declared.name


def inspect_shard(path: Path) -> ShardFileStatus:
    name_idx, name_n = _shard_index_from_name(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ShardFileStatus(
            path=str(path),
            idx=name_idx,
            n_shards=name_n,
            expected_points=None,
            actual_points=0,
            complete=False,
            parse_error=repr(exc),
        )
    if data.get("schema") != RESULTS_SCHEMA:
        return ShardFileStatus(
            path=str(path),
            idx=name_idx,
            n_shards=name_n,
            expected_points=None,
            actual_points=0,
            complete=False,
            parse_error=f"unsupported schema {data.get('schema')!r}; expected {RESULTS_SCHEMA!r}",
        )

    config = data.get("config", {})
    idx = config.get("shard_idx", name_idx)
    n_shards = config.get("n_shards", name_n)
    expected_points = None
    if config.get("point_start_idx") is not None and config.get("point_end_idx") is not None:
        expected_points = int(config["point_end_idx"]) - int(config["point_start_idx"])

    sidecars = {
        key: _declared_sidecar(path, data[key])
        for key in ("point_results_path", "cell_results_path")
        if data.get(key)
    }
    missing = [sidecar.name for sidecar in sidecars.values() if not sidecar.exists()]
    actual_points = 0
    point_sidecar = sidecars.get("point_results_path")
    if point_sidecar is not None and point_sidecar.exists():
        rows = json.loads(point_sidecar.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{point_sidecar}: point results must be a JSON list")
        actual_points = len(rows)
    complete = (
        expected_points is not None
        and actual_points == expected_points
        and len(sidecars) == 2
        and not missing
    )
    return ShardFileStatus(
        path=str(path),
        idx=int(idx) if idx is not None else None,
        n_shards=int(n_shards) if n_shards is not None else None,
        expected_points=expected_points,
        actual_points=actual_points,
        complete=complete,
        missing_detail_sidecars=missing,
    )


def shard_paths_in(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.glob("results_shard_*_of_*.json")
            if _SHARD_NAME.match(path.name)
        ),
        key=lambda path: _shard_index_from_name(path)[0] or 0,
    )


def inspect_dir(directory: Path) -> ShardDirStatus:
    shards = [inspect_shard(path) for path in shard_paths_in(directory)]
    expected_values = {shard.n_shards for shard in shards if shard.n_shards is not None}
    if len(expected_values) > 1:
        raise ValueError(
            f"{directory}: shard files disagree on n_shards: {sorted(expected_values)}"
        )
    expected_n = expected_values.pop() if expected_values else None
    present = {shard.idx for shard in shards if shard.idx is not None}
    missing = sorted(set(range(1, expected_n + 1)) - present) if expected_n is not None else []
    incomplete = sorted(
        shard.idx for shard in shards if shard.idx is not None and not shard.complete
    )
    detail_loss = {
        shard.idx: shard.missing_detail_sidecars
        for shard in shards
        if shard.idx is not None and shard.missing_detail_sidecars
    }
    ready = bool(shards) and expected_n is not None and not missing and not incomplete
    return ShardDirStatus(
        directory=str(directory),
        shards=shards,
        expected_n_shards=expected_n,
        missing_shards=missing,
        incomplete_shards=incomplete,
        ready_to_merge=ready,
        detail_loss_shards=detail_loss,
    )


def format_counts(status: ShardDirStatus) -> str:
    """Per-shard ``evaluated/assigned`` point counts, e.g. ``[2000/2000, 1990/2000]``."""
    parts = []
    for shard in status.shards:
        expected = "?" if shard.expected_points is None else str(shard.expected_points)
        label = f"{shard.actual_points}/{expected}"
        if shard.parse_error:
            label += " unreadable"
        parts.append(label)
    return "[" + ", ".join(parts) + "]"


def recorded_manifest_path(directory: Path) -> Path:
    """The single point-manifest path recorded by the shards in ``directory``."""
    paths = {json.loads(path.read_text())["manifest_path"] for path in shard_paths_in(directory)}
    if len(paths) != 1:
        raise ValueError(f"{directory}: shards disagree on manifest_path: {sorted(paths)}")
    return Path(paths.pop())


def find_shard_dirs(roots: list[Path]) -> list[Path]:
    """Directories below ``roots`` that contain at least one results_shard_K_of_N.json."""
    return sorted(
        {shard.parent for root in roots for shard in root.rglob("results_shard_*_of_*.json")}
    )


def merge_dir(
    directory: Path,
    *,
    point_manifest: Path | None = None,
    n_shards: int | None = None,
) -> dict:
    """Merge one directory; the manifest and shard count default to what the shards record."""
    status = inspect_dir(directory)
    if status.expected_n_shards is None and n_shards is None:
        raise ValueError(f"{directory}: no shard results found")
    if n_shards is not None and status.expected_n_shards not in (None, n_shards):
        raise ValueError(
            f"{directory}: shards record n_shards={status.expected_n_shards}, not {n_shards}"
        )
    if not status.ready_to_merge:
        raise ValueError(
            f"{directory}: not ready to merge; shard point counts {format_counts(status)}, "
            f"missing shards {status.missing_shards}, incomplete shards {status.incomplete_shards}"
        )
    return merge_shards(
        manifest_path=point_manifest or recorded_manifest_path(directory),
        output_dir=directory,
        n_shards=n_shards or status.expected_n_shards,
    )


@dataclass
class IncompleteResult:
    path: Path
    expected: int
    actual: int


def check_results(path: Path) -> IncompleteResult | None:
    """An :class:`IncompleteResult` if ``path`` does not cover its whole manifest, else None."""
    results = json.loads(path.read_text())
    if results.get("schema") != RESULTS_SCHEMA:
        raise ValueError(f"{path}: not a grid-eval results file (schema {results.get('schema')!r})")
    expected = int(results["manifest_num_points"])
    point_results = path.parent / Path(results["point_results_path"]).name
    actual = len(json.loads(point_results.read_text())) if point_results.exists() else 0
    if results.get("partial") or actual != expected:
        return IncompleteResult(path, expected, actual)
    return None


def scan_results(roots: list[Path]) -> tuple[list[IncompleteResult], list[Path]]:
    """Check every results.json below ``roots``; returns (incomplete, not_grid_eval)."""
    bad: list[IncompleteResult] = []
    skipped: list[Path] = []
    for path in sorted({p for root in roots for p in root.rglob("results.json")}):
        try:
            result = check_results(path)
        except ValueError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            skipped.append(path)
            continue
        if result is not None:
            bad.append(result)
    return bad, skipped


def run_status(directories: list[Path]) -> int:
    statuses = [inspect_dir(directory) for directory in directories]
    for status in statuses:
        verdict = "ready" if status.ready_to_merge else "not ready"
        print(f"{status.directory}: {verdict}; shard point counts: {format_counts(status)}")
        if status.missing_shards:
            print(f"  missing shards: {status.missing_shards}")
        if status.incomplete_shards:
            print(f"  incomplete shards: {status.incomplete_shards}")
        if status.detail_loss_shards:
            print(f"  shards with missing sidecars: {status.detail_loss_shards}")
    return 0 if all(status.ready_to_merge for status in statuses) else 1


def run_verify(roots: list[Path]) -> int:
    bad, skipped = scan_results(roots)
    if skipped:
        print(f"Skipped {len(skipped)} results.json file(s) that are not grid-eval results.")
    if not bad:
        print("All scanned grid-eval results.json files are complete.")
        return 0
    print(f"Found {len(bad)} incomplete results.json file(s):")
    for result in bad:
        print(f"  {result.actual}/{result.expected}  {result.path}")
    return 1


def run_merge(
    paths: list[Path],
    *,
    auto_find: bool,
    point_manifest: Path | None,
    n_shards: int | None,
) -> int:
    dirs = find_shard_dirs(paths) if auto_find else paths
    if not dirs:
        print("no shard directories found", file=sys.stderr)
        return 1
    n_merged = n_skipped = 0
    for directory in dirs:
        if auto_find and (directory / "results.json").exists():
            print(f"skip (already merged): {directory}")
            n_skipped += 1
            continue
        try:
            merged = merge_dir(directory, point_manifest=point_manifest, n_shards=n_shards)
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"merged {directory}: num_points={merged['num_points']}, "
            f"overall_sr={merged['overall_success_rate']:.4f}"
        )
        n_merged += 1
    if len(dirs) > 1:
        print(f"\n{n_merged} merged, {n_skipped} skipped")
    return 0
