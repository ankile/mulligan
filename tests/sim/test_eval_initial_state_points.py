import copy
import hashlib
import io
import json
import math
import random
import sys
import types

import numpy as np
import pytest
import mulligan.sim.eval.grid_eval as eval_points
import torch

from mulligan.utils.progress import (
    emit_completion,
    emit_progress,
    parse_progress_line,
)
from mulligan.sim.eval.grid_eval import (
    EqualTileGrid,
    _eta_s,
    _load_resume_results,
    _resume_config,
    _resume_path,
    _save_resume_results,
    _seed_eval_batch,
    evaluate_manifest,
    make_valid_sobol_manifest,
    make_equal_tile_manifest,
    merge_shards,
    point_to_placement,
    summarize_results,
    validate_manifest,
    validate_equal_tile_manifest,
    write_json_atomic,
)

# Locked eval grids (configs/sim/README.md, Evaluation grids; DIVL full_protocol.yaml locked_grid entries).
LOCKED_NARROW_MANIFEST_HASH = "45a9963f5592f495477c44beabe8d3c3e3173e920c2b2ba009c1e69532e53b17"
LOCKED_NARROW_FILE_SHA256 = "068b2deb11e553cdedfb0f789081b6fd7c4112aa7e22ecc3b655f90492168476"
LOCKED_BROAD_MANIFEST_HASH = "e0b5056648639f54189386a92384212d4fb09949bda23b622c10f8427aaa085c"
LOCKED_BROAD_FILE_SHA256 = "304210db9e7b3d175970f846753e2cd26ab59686fef3693c6f24f29c3f55a5e5"
# Round 0 Narrow equal-tile grid. The locked file predates the task_variant/env_name
# keys, so its hash covers the manifest without them.
LOCKED_NARROW_EQUAL_TILE_MANIFEST_HASH = (
    "41166760da933ccca99e5ac01694fbcb1ab2042312abcdded7ac7a67a724dbea"
)


def test_seed_eval_batch_replays_all_local_rng_streams() -> None:
    batch = [{"point_idx": 17}, {"point_idx": 18}]

    first_seed = _seed_eval_batch(20260831, batch)
    first = (random.random(), np.random.random(), torch.rand(3))
    second_seed = _seed_eval_batch(20260831, batch)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first_seed == second_seed
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])

    changed_seed = _seed_eval_batch(20260832, batch)
    assert changed_seed != first_seed


def test_resume_config_records_only_explicit_eval_seed() -> None:
    kwargs = {
        "artifact_path": "entity/project/run-final:v1",
        "manifest_hash_value": "abc123",
        "max_steps": 400,
        "num_envs": 2,
        "device": "cuda",
        "use_sync_envs": False,
        "num_action_samples": 4,
        "shard_idx": 1,
        "n_shards": 20,
        "point_start_idx": 0,
        "point_end_idx": 400,
    }

    unseeded = _resume_config(**kwargs)
    assert "eval_seed" not in unseeded
    assert "env_seed" not in unseeded
    seeded = _resume_config(**kwargs, eval_seed=20260831, env_seed=7)
    assert seeded["eval_seed"] == 20260831
    assert "ordered_batch_point_indices" in seeded["eval_seed_strategy"]
    assert seeded["env_seed"] == 7


def test_square_narrow_equal_tile_manifest_has_locked_counts_and_bounds() -> None:
    manifest = make_equal_tile_manifest(points_per_cell=3, seed=123)
    grid = EqualTileGrid()

    assert manifest["num_points"] == grid.n_cells * 3
    assert manifest["points_per_cell"] == 3
    validate_equal_tile_manifest(manifest)

    counts = np.zeros(grid.n_cells, dtype=np.int64)
    for point in manifest["points"]:
        cell_idx = point["cell_idx"]
        edges = grid.cell_edges(cell_idx)
        assert edges["x_lo"] <= point["nut_x"] < edges["x_hi"]
        assert edges["y_lo"] <= point["nut_y"] < edges["y_hi"]
        assert edges["yaw_lo"] <= point["nut_yaw"] < edges["yaw_hi"]
        assert -math.pi <= point["nut_yaw"] < math.pi
        counts[cell_idx] += 1

    np.testing.assert_array_equal(counts, np.full(grid.n_cells, 3))


def test_square_narrow_equal_tile_manifest_matches_locked_round0_grid() -> None:
    manifest = make_equal_tile_manifest(points_per_cell=100, seed=20260524)
    assert manifest["manifest_hash"] == LOCKED_NARROW_EQUAL_TILE_MANIFEST_HASH
    assert eval_points.manifest_hash(manifest) == LOCKED_NARROW_EQUAL_TILE_MANIFEST_HASH


def test_square_narrow_manifest_is_deterministic_and_hash_validated() -> None:
    first = make_equal_tile_manifest(points_per_cell=4, seed=20260524)
    second = make_equal_tile_manifest(points_per_cell=4, seed=20260524)
    assert first == second

    corrupted = copy.deepcopy(first)
    corrupted["points"][0]["nut_x"] += 1e-6
    try:
        validate_equal_tile_manifest(corrupted)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("corrupted manifest unexpectedly validated")


def test_global_valid_sobol_manifests_are_locked_and_occupied(tmp_path) -> None:
    narrow = make_valid_sobol_manifest(task="square_narrow", num_points=8000, seed=2026052402)
    broad = make_valid_sobol_manifest(task="square_broad", num_points=30000, seed=2026052499)

    assert narrow["num_points"] == 8000
    assert narrow["grid"]["n_cells"] == 80
    assert narrow["occupied_cells"] == 80
    assert narrow["base_seed"] == 2026052402
    assert broad["num_points"] == 30000
    assert broad["grid"]["n_cells"] == 720
    assert broad["occupied_cells"] == 720
    assert broad["base_seed"] == 2026052499
    assert broad["candidate_power"] == 16
    assert min(point["nut_peg_distance"] for point in broad["points"]) >= 0.13263

    validate_manifest(narrow)
    validate_manifest(broad)
    assert broad == make_valid_sobol_manifest(
        task="square_broad", num_points=30000, seed=2026052499
    )

    # The locked paper grids (manifest_hash and sha256 of the written file).
    assert narrow["manifest_hash"] == LOCKED_NARROW_MANIFEST_HASH
    assert broad["manifest_hash"] == LOCKED_BROAD_MANIFEST_HASH
    square_narrow_path = tmp_path / "square_narrow_valid_sobol8k_seed2026052402.json"
    square_broad_path = tmp_path / "square_broad_valid_sobol30k_seed2026052499.json"
    write_json_atomic(square_narrow_path, narrow)
    write_json_atomic(square_broad_path, broad)
    assert hashlib.sha256(square_narrow_path.read_bytes()).hexdigest() == LOCKED_NARROW_FILE_SHA256
    assert hashlib.sha256(square_broad_path.read_bytes()).hexdigest() == LOCKED_BROAD_FILE_SHA256
    assert (
        make_valid_sobol_manifest(task="square_narrow", num_points=8000, seed=2026052402) == narrow
    )

    corrupted = copy.deepcopy(broad)
    corrupted["points"][0]["peg_x"] += 1e-6
    try:
        validate_manifest(corrupted)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("corrupted Square-Broad manifest unexpectedly validated")


def test_square_broad_point_to_placement_adds_peg_body_pos() -> None:
    manifest = make_valid_sobol_manifest(
        task="square_broad",
        num_points=16,
        seed=2026052401,
        require_all_cells=False,
    )
    point = manifest["points"][0]
    placements = point_to_placement(point, 12, task_key="square_broad")

    assert placements[0] == ("body_pos", "peg1", [point["peg_x"], point["peg_y"]])
    assert placements[1][0] == "qpos"
    assert placements[1][1] == 12
    np.testing.assert_allclose(placements[1][2][0], point["nut_x"])
    np.testing.assert_allclose(placements[1][2][1], point["nut_y"])


def test_progress_events_match_monitor_contract() -> None:
    stream = io.StringIO()
    emit_progress(
        job_type="grid-eval",
        phase="evaluate",
        status="running",
        progress=0.5,
        current=4000,
        total=8000,
        eta_s=120.0,
        metrics={"points_done": 4000},
        file=stream,
    )
    emit_completion(
        job_type="grid-eval",
        success=True,
        exit_code=0,
        current=8000,
        total=8000,
        file=stream,
    )

    lines = stream.getvalue().strip().splitlines()
    running = parse_progress_line(lines[0])
    complete = parse_progress_line(lines[1])

    assert running is not None
    assert running.job_type == "grid-eval"
    assert running.status == "running"
    assert running.current == 4000
    assert running.total == 8000
    assert running.metrics["points_done"] == 4000

    assert complete is not None
    assert complete.status == "succeeded"
    assert complete.exit_code == 0
    assert complete.progress == 1.0


def test_eta_excludes_resumed_prefix() -> None:
    # This process completed 30 new points in 120 seconds.  The remaining
    # 7,070 points therefore take 28,280 seconds at the observed rate.  Using
    # all 430 points would incorrectly report less than 2,000 seconds.
    assert _eta_s(start_time=10.0, start_done=400, done=430, total=7500, now=130.0) == 28_280.0
    assert _eta_s(start_time=10.0, start_done=400, done=400, total=7500, now=130.0) is None
    assert _eta_s(start_time=10.0, start_done=0, done=7500, total=7500, now=130.0) == 0.0


def test_eval_resume_round_trip_requires_exact_config_and_prefix(tmp_path) -> None:
    points = [{"point_idx": idx, "cell_idx": idx // 2} for idx in range(6)]
    config = _resume_config(
        artifact_path="entity/project/run-final:v1",
        manifest_hash_value="abc123",
        max_steps=400,
        num_envs=2,
        device="cuda",
        use_sync_envs=False,
        num_action_samples=32,
        shard_idx=2,
        n_shards=4,
        point_start_idx=6,
        point_end_idx=12,
    )
    results = [
        {"point_idx": point["point_idx"], "cell_idx": point["cell_idx"], "success": 1}
        for point in points[:4]
    ]
    path = _resume_path(tmp_path, 2, 4)
    _save_resume_results(path=path, config=config, point_results=results)

    assert (
        _load_resume_results(
            path=path,
            expected_config=config,
            points=points,
            num_envs=2,
        )
        == results
    )

    changed_config = {**config, "artifact_path": "entity/project/other-final:v1"}
    try:
        _load_resume_results(
            path=path,
            expected_config=changed_config,
            points=points,
            num_envs=2,
        )
    except ValueError as exc:
        assert "config mismatch" in str(exc)
    else:
        raise AssertionError("mismatched eval resume config unexpectedly loaded")

    non_prefix = [*results]
    non_prefix[-1] = {"point_idx": 5, "cell_idx": 2, "success": 1}
    _save_resume_results(path=path, config=config, point_results=non_prefix)
    try:
        _load_resume_results(
            path=path,
            expected_config=config,
            points=points,
            num_envs=2,
        )
    except ValueError as exc:
        assert "contiguous prefix" in str(exc)
    else:
        raise AssertionError("non-prefix eval resume results unexpectedly loaded")


def test_summarize_results_requires_complete_cell_counts() -> None:
    manifest = make_equal_tile_manifest(points_per_cell=2, seed=11)
    point_results = [
        {
            "point_idx": point["point_idx"],
            "cell_idx": point["cell_idx"],
            "length": 3,
            "reward": 0.0,
            "success": 1,
        }
        for point in manifest["points"]
    ]
    summary = summarize_results(manifest=manifest, point_results=point_results)

    assert summary["num_points"] == 160
    assert summary["num_cells"] == 80
    assert summary["points_per_cell"] == 2
    assert summary["overall_success_rate"] == 1.0

    try:
        summarize_results(manifest=manifest, point_results=point_results[:-1])
    except ValueError as exc:
        assert "expected 160 point results" in str(exc)
    else:
        raise AssertionError("incomplete point results unexpectedly summarized")


def _write_shards(tmp_path, shard_config) -> tuple:
    """Four shards of a 40-point Square_D1 manifest; ``shard_config(k)`` is shard k's config."""
    manifest = make_valid_sobol_manifest(
        task="square_broad",
        num_points=40,
        seed=1234,
        require_all_cells=False,
    )
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    for shard_idx in range(1, 5):
        start = (manifest["num_points"] * (shard_idx - 1)) // 4
        end = (manifest["num_points"] * shard_idx) // 4
        point_results = [
            {
                "point_idx": point["point_idx"],
                "candidate_idx": point["candidate_idx"],
                "cell_idx": point["cell_idx"],
                "nut_x": point["nut_x"],
                "nut_y": point["nut_y"],
                "nut_yaw": point["nut_yaw"],
                "peg_x": point["peg_x"],
                "peg_y": point["peg_y"],
                "nut_peg_distance": point["nut_peg_distance"],
                "success": int(point["point_idx"] % 2 == 0),
                "reward": 0.0,
                "length": 3,
            }
            for point in manifest["points"][start:end]
        ]
        write_json_atomic(output_dir / f"point_results_shard_{shard_idx}_of_4.json", point_results)
        write_json_atomic(
            output_dir / f"results_shard_{shard_idx}_of_4.json",
            {
                "artifact_path": "entity/project/run-final-step:v0",
                "manifest_hash": manifest["manifest_hash"],
                "config": shard_config(shard_idx),
            },
        )
    return manifest_path, output_dir


SHARD_CONFIG = {"max_steps": 400, "num_action_samples": 32, "eval_seed": None, "env_seed": None}


def test_merge_shards_preserves_contiguous_prefix_coverage(tmp_path) -> None:
    manifest_path, output_dir = _write_shards(
        tmp_path, lambda k: {**SHARD_CONFIG, "shard_idx": k, "num_envs": 10 + k}
    )
    merged = merge_shards(manifest_path=manifest_path, output_dir=output_dir, n_shards=4)

    assert merged["num_points"] == 40
    assert merged["manifest_num_points"] == 40
    assert merged["partial"] is False
    assert merged["overall_success_rate"] == 0.5
    with (output_dir / "point_results.json").open() as f:
        rows = json.load(f)
    assert [row["point_idx"] for row in rows] == list(range(40))


@pytest.mark.parametrize(
    ("key", "other"),
    [("num_action_samples", 1), ("max_steps", 300), ("eval_seed", 7), ("env_seed", 0)],
)
def test_merge_shards_requires_equal_eval_settings(tmp_path, key, other) -> None:
    manifest_path, output_dir = _write_shards(
        tmp_path, lambda k: {**SHARD_CONFIG, "shard_idx": k, **({key: other} if k == 3 else {})}
    )
    with pytest.raises(ValueError, match=f"shards disagree on {key}"):
        merge_shards(manifest_path=manifest_path, output_dir=output_dir, n_shards=4)
    assert not (output_dir / "results.json").exists()


def test_eval_requires_num_envs_to_divide_manifest_size(tmp_path) -> None:
    manifest = make_equal_tile_manifest(points_per_cell=1, seed=17)
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    try:
        evaluate_manifest(
            artifact_path="entity/project/fake:v0",
            manifest_path=manifest_path,
            output_dir=tmp_path / "out",
            num_envs=3,
        )
    except ValueError as exc:
        assert "num_points must be divisible by num_envs" in str(exc)
    else:
        raise AssertionError("non-divisible num_envs unexpectedly accepted")


def test_eval_complete_resume_finalizes_without_policy_or_gpu(tmp_path) -> None:
    manifest = make_equal_tile_manifest(points_per_cell=1, seed=29)
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    artifact_path = "entity/project/fake-final-step-1:v0"
    num_envs = 10
    resume_path = _resume_path(output_dir, None, None)
    config = _resume_config(
        artifact_path=artifact_path,
        manifest_hash_value=manifest["manifest_hash"],
        max_steps=400,
        num_envs=num_envs,
        device="cuda",
        use_sync_envs=False,
        num_action_samples=32,
        shard_idx=None,
        n_shards=None,
        point_start_idx=0,
        point_end_idx=len(manifest["points"]),
    )
    point_results = [
        {
            **point,
            "success": int(point["point_idx"] % 2 == 0),
            "reward": 0.0,
            "length": 3,
        }
        for point in manifest["points"]
    ]
    _save_resume_results(path=resume_path, config=config, point_results=point_results)

    results = evaluate_manifest(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        num_envs=num_envs,
        device="cuda",
    )

    assert results["num_points"] == len(manifest["points"])
    assert results["partial"] is False
    assert results["overall_success_rate"] == 0.5
    assert (output_dir / "results.json").exists()
    assert (output_dir / "point_results.json").exists()
    assert (output_dir / "cell_results.json").exists()


class _FakePolicy:
    config = type("Config", (), {"num_action_samples": 1})()

    def eval(self) -> None:
        pass


class _FakeRefEnv:
    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def _install_fake_runtime(monkeypatch, vec_env_cls, *, loaded_paths=None, env_kwargs=None):
    """Replace the simulator, vector env and checkpoint loader with in-process fakes."""
    env_kwargs = [] if env_kwargs is None else env_kwargs

    def fake_step_rollouts_from_obs(**kwargs):
        n = len(kwargs["obs_list"])
        return [idx % 2 == 0 for idx in range(n)], [0.0] * n, [3] * n

    def fake_load_policy_from_checkpoint(*, checkpoint_path, device):
        if loaded_paths is not None:
            loaded_paths.append(checkpoint_path)
        return _FakePolicy(), object(), object()

    monkeypatch.setattr(eval_points, "_get_nut_qpos_start", lambda env: 0)
    monkeypatch.setattr(eval_points, "step_rollouts_from_obs", fake_step_rollouts_from_obs)
    fake_vec_env = types.ModuleType("mulligan.sim.vec_env")
    fake_vec_env.AsyncVectorEnv = vec_env_cls
    fake_vec_env.SyncVectorEnv = vec_env_cls
    fake_envs = types.ModuleType("mulligan.sim.envs")
    fake_envs.create_robosuite_env = lambda **kwargs: env_kwargs.append(kwargs) or _FakeRefEnv()
    fake_loader = types.ModuleType("mulligan.utils.load_pretrained")
    fake_loader.load_policy_from_checkpoint = fake_load_policy_from_checkpoint
    fake_idql = types.ModuleType("mulligan.agents.idql")
    fake_idql.IDQLPolicy = type("IDQLPolicy", (), {})
    for module in (fake_vec_env, fake_envs, fake_loader, fake_idql):
        monkeypatch.setitem(sys.modules, module.__name__, module)


class _RecordingVectorEnv:
    instances: list["_RecordingVectorEnv"] = []

    def __init__(self, env_fns, seed=None) -> None:
        self.num_envs = len(env_fns)
        self.seed = seed
        self.placements: list[list] = []
        self.seeds: list[list[int]] = []
        _RecordingVectorEnv.instances.append(self)

    def seed_envs(self, seeds) -> None:
        assert len(seeds) == self.num_envs
        self.seeds.append(list(seeds))

    def reset_with_placements(self, placements):
        assert len(placements) == self.num_envs
        self.placements.append(placements)
        return [object()] * self.num_envs, [], []

    def close(self) -> None:
        pass


def test_eval_persists_complete_results_before_async_close_failure(tmp_path, monkeypatch) -> None:
    manifest = make_equal_tile_manifest(points_per_cell=1, seed=31)
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    output_dir = tmp_path / "out"
    artifact_dir = tmp_path / "checkpoint"
    artifact_dir.mkdir()

    class FailingCloseVectorEnv(_RecordingVectorEnv):
        def close(self) -> None:
            raise RuntimeError("synthetic async close failure")

    completion_success: list[bool] = []
    _install_fake_runtime(monkeypatch, FailingCloseVectorEnv)
    monkeypatch.setattr(
        eval_points,
        "emit_completion",
        lambda *, success, **kwargs: completion_success.append(success),
    )

    try:
        evaluate_manifest(
            artifact_path=str(artifact_dir),
            manifest_path=manifest_path,
            output_dir=output_dir,
            num_envs=len(manifest["points"]),
            device="cpu",
            num_action_samples=1,
        )
    except RuntimeError as exc:
        assert str(exc) == "synthetic async close failure"
    else:
        raise AssertionError("async close failure unexpectedly succeeded")

    results = json.loads((output_dir / "results.json").read_text())
    assert results["num_points"] == len(manifest["points"])
    assert results["partial"] is False
    assert results["overall_success_rate"] == 0.5
    assert (output_dir / "point_results.json").exists()
    assert (output_dir / "cell_results.json").exists()

    assert completion_success == [False]


def test_eval_resolves_hf_artifact_and_forwards_env_seed(tmp_path, monkeypatch) -> None:
    manifest = make_valid_sobol_manifest(
        task="square_narrow", num_points=40, seed=2026052402, require_all_cells=False
    )
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    checkpoint_dir = tmp_path / "hf-cache" / "seed-1"
    checkpoint_dir.mkdir(parents=True)
    ref = "hf://mulligan/sim-square-narrow-r01-mulligan-divl@93356d9ead01cc93ee3e7f1d1cb9ea7e1f51f1f7/seed-1"

    resolved: list[str] = []

    def fake_resolve(artifact_ref: str):
        resolved.append(artifact_ref)
        return checkpoint_dir

    loaded_paths: list = []
    env_kwargs: list = []
    _RecordingVectorEnv.instances.clear()
    _install_fake_runtime(
        monkeypatch, _RecordingVectorEnv, loaded_paths=loaded_paths, env_kwargs=env_kwargs
    )
    monkeypatch.setattr(eval_points, "resolve_checkpoint", fake_resolve)

    results = evaluate_manifest(
        artifact_path=ref,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        num_envs=4,
        device="cpu",
        num_action_samples=32,
        shard_idx=1,
        n_shards=2,
        env_seed=11,
    )

    assert resolved == [ref]
    assert loaded_paths == [checkpoint_dir]
    (vec_env,) = _RecordingVectorEnv.instances
    # each start is reseeded from (env_seed, point_idx) before its batch
    assert vec_env.seed is None
    assert len(vec_env.placements) == 5
    points = manifest["points"][:20]
    assert vec_env.seeds == [
        eval_points._point_env_seeds(11, points[i : i + 4]) for i in range(0, 20, 4)
    ]
    # the reference env that reads the nut joint index is headless like the workers
    (ref_env_kwargs,) = env_kwargs
    assert ref_env_kwargs["use_render_wrapper"] is False
    assert results["config"]["env_seed_strategy"] == eval_points.ENV_SEED_STRATEGY
    assert results["artifact_path"] == ref
    assert results["config"]["env_seed"] == 11
    assert results["config"]["point_start_idx"] == 0
    assert results["config"]["point_end_idx"] == 20
    assert results["num_points"] == 20
    assert results["partial"] is True
    assert (tmp_path / "out" / "results_shard_1_of_2.json").exists()


def test_point_env_seeds_depend_only_on_env_seed_and_point() -> None:
    """A resumed or differently batched evaluation reseeds every start identically."""
    points = [{"point_idx": i} for i in range(8)]
    whole = eval_points._point_env_seeds(7, points)
    assert whole == eval_points._point_env_seeds(7, points[:3]) + eval_points._point_env_seeds(
        7, points[3:]
    )
    assert len(set(whole)) == 8 and all(0 <= s < 2**32 for s in whole)
    assert whole != eval_points._point_env_seeds(8, points)


def test_first_shard_is_the_manifest_prefix() -> None:
    # Shard 1 of 10 on the Narrow 8k grid is the first 800 starts.
    assert eval_points._shard_range(8000, 1, 10) == (0, 800)
    assert eval_points._shard_range(30000, 4, 4) == (22500, 30000)
    assert eval_points._shard_range(8000, None, None) == (0, 8000)
