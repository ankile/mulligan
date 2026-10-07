from __future__ import annotations

import json
from pathlib import Path

import pytest

from mulligan.sim.eval import grid_eval
from mulligan.sim.eval.shards import format_counts, inspect_dir, merge_dir, scan_results

ARTIFACT = "hf://mulligan/sim-square-narrow-r01-mulligan-divl@05e15eb4f0324587337fc7a8e3374398c6044443/seed-1"


def _manifest(tmp_path: Path) -> tuple[dict, Path]:
    manifest = grid_eval.make_valid_sobol_manifest(
        task="square_narrow", num_points=40, seed=2026052402, require_all_cells=False
    )
    path = tmp_path / "manifest.json"
    grid_eval.write_json_atomic(path, manifest)
    return manifest, path


def _write_shard(
    output_dir: Path, manifest: dict, manifest_path: Path, idx: int, n_shards: int
) -> None:
    """Write shard outputs the way grid_eval.evaluate_manifest finalizes a shard."""
    start, end = grid_eval._shard_range(len(manifest["points"]), idx, n_shards)
    point_results = [
        grid_eval._point_result_from_rollout(
            point,
            success=point["point_idx"] % 2 == 0,
            reward=0.0,
            length=3,
            task_key="square_narrow",
        )
        for point in manifest["points"][start:end]
    ]
    paths = grid_eval._result_paths(output_dir, idx, n_shards)
    summary = grid_eval.summarize_results(
        manifest=manifest, point_results=point_results, require_complete=False
    )
    grid_eval.write_json_atomic(paths["point_results"], point_results)
    grid_eval.write_json_atomic(paths["cell_results"], summary["cells"])
    grid_eval.write_json_atomic(
        paths["results"],
        {
            "schema": grid_eval.RESULTS_SCHEMA,
            "artifact_path": ARTIFACT,
            "manifest_path": str(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
            "config": {
                "shard_idx": idx,
                "n_shards": n_shards,
                "point_start_idx": start,
                "point_end_idx": end,
            },
            **summary,
            "point_results_path": str(paths["point_results"]),
            "cell_results_path": str(paths["cell_results"]),
        },
    )


def test_inspect_dir_reports_missing_incomplete_and_deleted_sidecars(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    out = tmp_path / "out"
    for idx in (1, 2, 3):
        _write_shard(out, manifest, manifest_path, idx, 4)
    (out / "point_results_shard_2_of_4.json").unlink()

    status = inspect_dir(out)

    assert not status.ready_to_merge
    assert status.expected_n_shards == 4
    assert status.missing_shards == [4]
    assert status.incomplete_shards == [2]
    assert status.detail_loss_shards == {2: ["point_results_shard_2_of_4.json"]}
    assert format_counts(status) == "[10/10, 0/10, 10/10]"


def test_merge_dir_reads_manifest_and_shard_count_from_shards(tmp_path: Path, capsys) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    out = tmp_path / "results" / "seed1"
    for idx in (1, 2, 3, 4):
        _write_shard(out, manifest, manifest_path, idx, 4)
    assert inspect_dir(out).ready_to_merge

    assert grid_eval.main(["merge", "--auto-find", str(tmp_path / "results")]) == 0

    merged = json.loads((out / "results.json").read_text())
    assert merged["num_points"] == 40
    assert merged["partial"] is False
    assert merged["overall_success_rate"] == 0.5
    assert merged["artifact_path"] == ARTIFACT
    rows = json.loads((out / "point_results.json").read_text())
    assert [row["point_idx"] for row in rows] == list(range(40))

    # A second auto-find pass skips the merged directory; a direct merge rewrites it.
    capsys.readouterr()
    assert grid_eval.main(["merge", "--auto-find", str(tmp_path / "results")]) == 0
    assert "skip (already merged)" in capsys.readouterr().out
    assert grid_eval.main(["merge", "--output-dir", str(out)]) == 0
    assert json.loads((out / "results.json").read_text()) == merged


def test_merge_status_reports_readiness(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    out = tmp_path / "out"
    _write_shard(out, manifest, manifest_path, 1, 2)
    assert grid_eval.main(["merge", "--status", str(out)]) == 1
    _write_shard(out, manifest, manifest_path, 2, 2)
    assert grid_eval.main(["merge", "--status", str(out)]) == 0
    assert not (out / "results.json").exists()


def test_merge_dir_refuses_missing_shards(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    out = tmp_path / "out"
    for idx in (1, 3):
        _write_shard(out, manifest, manifest_path, idx, 3)

    with pytest.raises(ValueError, match=r"missing shards \[2\]"):
        merge_dir(out)
    assert grid_eval.main(["merge", str(out)]) == 2


def test_verify_flags_partial_results(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    good = tmp_path / "results" / "good"
    bad = tmp_path / "results" / "bad"
    for directory in (good, bad):
        for idx in (1, 2):
            _write_shard(directory, manifest, manifest_path, idx, 2)
        merge_dir(directory)
    rows = json.loads((bad / "point_results.json").read_text())
    grid_eval.write_json_atomic(bad / "point_results.json", rows[:30])

    incomplete, skipped = scan_results([tmp_path / "results"])
    assert skipped == []
    assert [(r.path, r.actual, r.expected) for r in incomplete] == [(bad / "results.json", 30, 40)]

    assert grid_eval.main(["merge", "--verify", str(tmp_path / "results")]) == 1
    assert grid_eval.main(["merge", "--verify", str(good)]) == 0
