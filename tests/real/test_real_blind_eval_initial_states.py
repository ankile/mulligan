import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from mulligan.real.eval.blind_eval_helpers import (
    _eval_initial_state_features,
    _eval_initial_state_frame_fields,
    _eval_initial_state_keys,
    _load_eval_initial_state_manifest,
    _load_round_plans,
)
from mulligan.real.eval.manifest_eval import (
    PolicyMapEntry,
    ResolvedPolicy,
    _entries_and_names,
    _ensure_all_round_plans,
    _format_eta,
    _make_round_plan,
    _parse_fixed_policy_num_action_samples,
    _pending_manifest_rounds,
    _save_results_file_quiet,
)
from mulligan.real.collect.initial_states import InitialStateTarget
from mulligan.real.eval.common import (
    PolicyEntry,
    RolloutRecord,
    _prepare_new_dataset_root,
    _restore_precreate_sidecars,
    cleanup_stale_image_episode_dirs,
    load_previous_results,
    print_results,
    save_results_file,
)
from mulligan.real.collect.rollout import (
    RecoverableRolloutError,
    _refresh_obs_until_robot_state_timestamp_advances,
)


_MANIFESTS = Path(__file__).resolve().parents[2] / "data" / "real" / "manifests"
_MARKER_D2_R0_MANIFEST = (
    _MANIFESTS
    / "marker_d2/r00/marker_d2_r0_three_arm_uniform100_sobol100_evalsobol50_scrambled.json"
)
_MARKER_D2_R0_EVAL_MANIFEST = _MANIFESTS / "marker_d2/r00/marker_d2_r0_eval_heldout_sobol50.json"


def test_eval_manifest_loader_requires_matching_task() -> None:
    with pytest.raises(ValueError, match=r"expected task in \['square_d2'\]"):
        _load_eval_initial_state_manifest(_MARKER_D2_R0_EVAL_MANIFEST, expected_task="square_d2")


def test_eval_manifest_loader_returns_ordered_targets() -> None:
    targets, meta = _load_eval_initial_state_manifest(
        _MARKER_D2_R0_EVAL_MANIFEST, expected_task="marker_d2"
    )
    assert meta["task"] == "marker_d2"
    assert [target.manifest_idx for target in targets] == list(range(len(targets)))
    assert len(targets) == 50


def test_eval_manifest_loader_accepts_committed_marker_d2_three_arm(tmp_path: Path) -> None:
    # blind_eval's loader accepts the committed marker_d2 5-key (pen + holder)
    # manifest by routing through the shared, marker_d2-aware collection loader.
    assert _MARKER_D2_R0_MANIFEST.exists(), _MARKER_D2_R0_MANIFEST

    targets, meta = _load_eval_initial_state_manifest(
        _MARKER_D2_R0_MANIFEST,
        expected_task="marker_d2",
    )

    assert meta["task"] == "marker_d2"
    assert meta["keys"] == ["pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y"]
    assert len(targets) == 250
    # Contiguous manifest_idx order is preserved by the shared loader.
    assert [t.manifest_idx for t in targets] == list(range(250))
    # All three arms appear as row sources (informational for eval; the shared
    # loader validated each against the manifest-declared arm whitelist).
    assert {t.source for t in targets} == {"baseline_uniform", "mulligan_sobol", "eval_heldout"}
    # The 5-key holder columns survive into the saved-dataset feature/field shape.
    assert _eval_initial_state_keys(meta) == [
        "pen_x",
        "pen_y",
        "pen_yaw",
        "holder_x",
        "holder_y",
    ]
    assert set(_eval_initial_state_features(meta)) == {
        "manifest_idx",
        "pen_x",
        "pen_y",
        "pen_yaw",
        "holder_x",
        "holder_y",
    }
    fields = _eval_initial_state_frame_fields(targets[0], meta)
    assert set(fields) == {
        "manifest_idx",
        "pen_x",
        "pen_y",
        "pen_yaw",
        "holder_x",
        "holder_y",
    }
    assert targets[0].holder_x is not None and targets[0].holder_y is not None
    assert fields["holder_x"].shape == (1,)
    assert float(fields["holder_x"][0]) == pytest.approx(targets[0].holder_x)


def test_eval_manifest_loader_rejects_off_grid_holder(tmp_path: Path) -> None:
    # An off-grid holder must fail loud (the shared marker_d2 validation snaps the
    # holder to a 1-inch registry grid; a hand-edited off-grid coord is rejected).
    payload = json.loads(_MARKER_D2_R0_MANIFEST.read_text())
    # Nudge the first row's holder_x off the 1-inch snap grid by half an inch.
    payload["states"][0]["holder_x"] = float(payload["states"][0]["holder_x"]) + 0.0127
    manifest = tmp_path / "off_grid.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="not an in-bounds grid point for marker_d2"):
        _load_eval_initial_state_manifest(manifest, expected_task="marker_d2")


def test_parse_fixed_policy_num_action_samples() -> None:
    assert _parse_fixed_policy_num_action_samples("iql_n4=4") == ("iql_n4", 4)
    with pytest.raises(ValueError, match="must be NAME=N"):
        _parse_fixed_policy_num_action_samples("iql_n4")
    with pytest.raises(ValueError, match="must be >= 1"):
        _parse_fixed_policy_num_action_samples("iql_n4=0")


def test_manifest_round_plan_allows_duplicate_fixed_model_slots() -> None:
    artifact = "wandb://entity/project/iql:v0"
    policy_map = {
        "fixed:0": PolicyMapEntry(policy_id=0, name="iql_n4", model_id=artifact),
        "fixed:1": PolicyMapEntry(policy_id=1, name="iql_n16", model_id=artifact),
    }
    round_plans: list[dict] = []
    round_plans_by_round: dict[int, dict] = {}

    _ensure_all_round_plans(
        manifest_targets=[
            SimpleNamespace(manifest_idx=0, pen_x=0.01, pen_y=-0.02, pen_yaw=0.3),
        ],
        round_plans=round_plans,
        round_plans_by_round=round_plans_by_round,
        fixed_policies=[
            ResolvedPolicy(
                name="iql_n4",
                model_id=artifact,
                slot_key="fixed:0",
                slot_type="fixed",
            ),
            ResolvedPolicy(
                name="iql_n16",
                model_id=artifact,
                slot_key="fixed:1",
                slot_type="fixed",
            ),
        ],
        policy_map=policy_map,
        random_seed=123,
    )

    assert len(round_plans) == 1
    planned = round_plans[0]["policy_order"]
    assert {row["slot_key"] for row in planned} == {"fixed:0", "fixed:1"}
    assert {row["policy_id"] for row in planned} == {0, 1}
    assert [
        entry.model_id for entry in _entries_and_names(records=[], policy_map=policy_map)[0]
    ] == [
        artifact,
        artifact,
    ]


def _record(round_num: int, policy_id: int, episode_index: int) -> RolloutRecord:
    return RolloutRecord(
        round_num=round_num,
        policy_id=policy_id,
        model_id=f"wandb://entity/project/{'ab'[policy_id]}:v0",
        anonymous_label="AB"[policy_id],
        outcome="success",
        num_steps=12,
        episode_index=episode_index,
        manifest_idx=round_num - 1,
    )


def test_pending_manifest_rounds_allows_final_partial_round() -> None:
    records = [
        _record(1, 0, 0),
        _record(1, 1, 1),
        _record(2, 0, 2),
    ]

    pending, completed_by_round = _pending_manifest_rounds(
        records,
        num_rounds=3,
        slots_per_round=2,
    )

    assert pending == [2, 3]
    assert completed_by_round == {2: [records[2]]}


def test_pending_manifest_rounds_rejects_mid_run_gap_without_flag() -> None:
    records = [
        _record(1, 0, 0),
        _record(2, 0, 1),
        _record(2, 1, 2),
    ]

    with pytest.raises(RuntimeError, match="--rerun-incomplete-rounds"):
        _pending_manifest_rounds(records, num_rounds=2, slots_per_round=2)


def test_pending_manifest_rounds_reruns_mid_run_gap_with_flag() -> None:
    records = [
        _record(1, 0, 0),
        _record(2, 0, 1),
        _record(2, 1, 2),
        _record(3, 0, 3),
        _record(3, 1, 4),
    ]

    pending, completed_by_round = _pending_manifest_rounds(
        records,
        num_rounds=3,
        slots_per_round=2,
        rerun_incomplete=True,
    )

    assert pending == [1]
    assert completed_by_round == {1: [records[0]]}


def test_pending_manifest_rounds_reruns_fully_missing_round_with_flag() -> None:
    records = [
        _record(1, 0, 0),
        _record(1, 1, 1),
        _record(3, 0, 2),
        _record(3, 1, 3),
    ]

    with pytest.raises(RuntimeError, match="incomplete non-final round"):
        _pending_manifest_rounds(records, num_rounds=3, slots_per_round=2)

    pending, completed_by_round = _pending_manifest_rounds(
        records,
        num_rounds=3,
        slots_per_round=2,
        rerun_incomplete=True,
    )

    assert pending == [2]
    assert completed_by_round == {}


def test_pending_manifest_rounds_complete_run_has_no_pending_rounds() -> None:
    records = [
        _record(1, 0, 0),
        _record(1, 1, 1),
        _record(2, 0, 2),
        _record(2, 1, 3),
    ]

    assert _pending_manifest_rounds(records, num_rounds=2, slots_per_round=2) == ([], {})


def test_results_round_trip_preserves_manifest_metadata(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    policies = [
        PolicyEntry("wandb://entity/project/a:v0", 0, policy=object(), results=[True]),
    ]
    records = [
        RolloutRecord(
            round_num=1,
            policy_id=0,
            model_id="wandb://entity/project/a:v0",
            anonymous_label="A",
            outcome="success",
            num_steps=12,
            episode_index=0,
            manifest_idx=0,
            pen_x=0.01,
            pen_y=-0.02,
            pen_yaw=0.3,
        )
    ]

    save_results_file(
        path,
        policies,
        records,
        argparse.Namespace(func=None, initial_states_manifest=tmp_path / "manifest.json"),
        "eval-dataset",
    )

    raw = json.loads(path.read_text())
    assert raw["args"]["initial_states_manifest"] == str(tmp_path / "manifest.json")

    loaded, _ = load_previous_results(path)
    assert loaded[0].manifest_idx == 0
    assert loaded[0].pen_x == pytest.approx(0.01)
    assert loaded[0].pen_y == pytest.approx(-0.02)
    assert loaded[0].pen_yaw == pytest.approx(0.3)


def test_manifest_eval_eta_formatting() -> None:
    assert _format_eta(None) == "unknown"
    assert _format_eta(8.9) == "8s"
    assert _format_eta(65) == "1m 05s"
    assert _format_eta(3665) == "1h 01m"


def test_manifest_eval_quiet_results_save_suppresses_status_line(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "results.json"
    policies = [
        PolicyEntry("wandb://entity/project/a:v0", 0, policy=object(), results=[]),
    ]

    _save_results_file_quiet(
        path,
        policies,
        [],
        argparse.Namespace(func=None),
        "eval-dataset",
        quiet=True,
    )

    captured = capsys.readouterr()
    assert "Results saved to:" not in captured.out
    assert path.exists()


def test_print_results_uses_compact_policy_table(capsys) -> None:
    policies = [
        PolicyEntry(
            "wandb://entity/project/full-model-name-that-should-not-print:v0",
            0,
            policy=object(),
            results=[True, False, True],
        ),
        PolicyEntry(
            "wandb://entity/project/unnamed-model:v0",
            1,
            policy=object(),
            results=[],
        ),
    ]

    print_results(policies, policy_names=["baseline_uniform_r2_no_cf_dp_25k", "short_b"])

    output = capsys.readouterr().out
    assert "baseline_uniform_r2_no_cf_dp_25k" in output
    assert "2/3 (66.7%)" in output
    assert "short_b" in output
    assert "0/0" in output
    assert "Model:" not in output
    assert "full-model-name-that-should-not-print" not in output


def test_manifest_eval_fixed_only_plan() -> None:
    round_plans: list[dict] = []
    round_plans_by_round: dict[int, dict] = {}
    policy_map = {
        "fixed:0": PolicyMapEntry(
            policy_id=0,
            name="fixed_policy",
            model_id="wandb://entity/project/fixed:v0",
        )
    }

    _ensure_all_round_plans(
        manifest_targets=[
            InitialStateTarget(
                manifest_idx=0,
                source="debug",
                source_index=0,
                pen_x=0.01,
                pen_y=-0.02,
                pen_yaw=0.3,
                raw={},
            )
        ],
        round_plans=round_plans,
        round_plans_by_round=round_plans_by_round,
        fixed_policies=[
            ResolvedPolicy(
                name="fixed_policy",
                model_id="wandb://entity/project/fixed:v0",
                slot_key="fixed:0",
                slot_type="fixed",
            )
        ],
        policy_map=policy_map,
        random_seed=123,
    )

    assert len(round_plans) == 1
    assert round_plans[0]["policy_order"][0]["name"] == "fixed_policy"


def test_dataset_create_preserves_false_start_results_sidecar(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval-dataset"
    dataset_path.mkdir()
    (dataset_path / "results.json").write_text('{"round_plans": []}')

    backup_path = _prepare_new_dataset_root(dataset_path)

    assert backup_path is not None
    assert not dataset_path.exists()
    dataset_path.mkdir()
    (dataset_path / "meta").mkdir()

    _restore_precreate_sidecars(dataset_path, backup_path)

    assert (dataset_path / "results.json").read_text() == '{"round_plans": []}'
    assert (backup_path / "results.json").read_text() == '{"round_plans": []}'


def test_dataset_create_preserves_precreate_chunk_info_sidecar_dir(tmp_path: Path) -> None:
    # The BoN chunk-info sidecar is written after each episode, but the dataset is created lazily after the FIRST
    # episode — so an IQL arm in round 1 slot A leaves chunk_info/ in the dataset
    # root and creation must move it aside and restore it, not refuse.
    dataset_path = tmp_path / "eval-dataset"
    (dataset_path / "chunk_info").mkdir(parents=True)
    sidecar_content = '{"round_num": 1}\n{"chunk_ordinal": 0}\n'
    (dataset_path / "chunk_info" / "round_0001_policy_a.jsonl").write_text(sidecar_content)
    (dataset_path / "results.json").write_text('{"round_plans": []}')

    backup_path = _prepare_new_dataset_root(dataset_path)

    assert backup_path is not None
    assert not dataset_path.exists()
    dataset_path.mkdir()
    (dataset_path / "meta").mkdir()

    _restore_precreate_sidecars(dataset_path, backup_path)

    restored = dataset_path / "chunk_info" / "round_0001_policy_a.jsonl"
    assert restored.read_text() == sidecar_content
    assert (dataset_path / "results.json").read_text() == '{"round_plans": []}'


def test_dataset_create_rejects_ambiguous_partial_dataset_root(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval-dataset"
    (dataset_path / "data").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="non-sidecar entries: data"):
        _prepare_new_dataset_root(dataset_path)


def test_dataset_create_drops_torn_results_tmp(tmp_path: Path) -> None:
    # A kill inside save_results_file leaves results.json.tmp next to the intact
    # results.json; the relaunch's first dataset creation must not refuse on it.
    dataset_path = tmp_path / "eval-dataset"
    dataset_path.mkdir()
    (dataset_path / "results.json").write_text('{"round_plans": []}')
    (dataset_path / "results.json.tmp").write_text('{"timestamp":')

    backup_path = _prepare_new_dataset_root(dataset_path)

    assert backup_path is not None
    assert sorted(p.name for p in backup_path.iterdir()) == ["results.json"]


def test_results_round_trip_keeps_local_checkpoint_model_ids(tmp_path: Path) -> None:
    # manifest_eval stores a local checkpoint dir as its absolute path; resume must read
    # back the same id the round plans carry. Bare W&B ids gain the wandb:// scheme.
    path = tmp_path / "results.json"
    local_id = str(tmp_path / "ckpt")
    policies = [
        PolicyEntry(local_id, 0, policy=object(), results=[True]),
        PolicyEntry("entity/project/b:v0", 1, policy=object(), results=[]),
    ]
    record = RolloutRecord(
        round_num=1,
        policy_id=0,
        model_id=local_id,
        anonymous_label="A",
        outcome="success",
        num_steps=12,
        episode_index=0,
    )
    save_results_file(path, policies, [record], argparse.Namespace(), "eval-dataset")

    loaded, policy_map = load_previous_results(path)

    assert loaded[0].model_id == local_id
    assert policy_map[local_id][0] == 0
    assert policy_map["wandb://entity/project/b:v0"][0] == 1


def _normalize_compiled_state_dict_keys(raw):
    from mulligan.real.policy.vision_idql import _normalize_compiled_state_dict_keys as fn

    return fn(raw)


def test_vision_iql_loader_strips_torch_compile_state_dict_prefix() -> None:
    raw = {"_orig_mod.layer.weight": torch.tensor([1.0])}

    normalized = _normalize_compiled_state_dict_keys(raw)

    assert list(normalized) == ["layer.weight"]
    assert torch.equal(normalized["layer.weight"], raw["_orig_mod.layer.weight"])


def test_vision_iql_loader_strips_nested_torch_compile_state_dict_prefix() -> None:
    raw = {
        "10000001_left._orig_mod.backbone.0.weight": torch.tensor([1.0]),
        "20000002_left._orig_mod.backbone.0.weight": torch.tensor([2.0]),
    }

    normalized = _normalize_compiled_state_dict_keys(raw)

    assert list(normalized) == [
        "10000001_left.backbone.0.weight",
        "20000002_left.backbone.0.weight",
    ]
    assert torch.equal(
        normalized["10000001_left.backbone.0.weight"],
        raw["10000001_left._orig_mod.backbone.0.weight"],
    )


def test_vision_iql_loader_rejects_mixed_compile_state_dict_prefixes() -> None:
    raw = {
        "_orig_mod.layer.weight": torch.tensor([1.0]),
        "layer.bias": torch.tensor([0.0]),
    }

    with pytest.raises(ValueError, match="mixes torch.compile-prefixed"):
        _normalize_compiled_state_dict_keys(raw)


def test_vision_iql_loader_rejects_mixed_nested_compile_state_dict_prefixes() -> None:
    raw = {
        "10000001_left._orig_mod.backbone.0.weight": torch.tensor([1.0]),
        "20000002_left.backbone.0.weight": torch.tensor([2.0]),
    }

    with pytest.raises(ValueError, match="mixes torch.compile-prefixed"):
        _normalize_compiled_state_dict_keys(raw)


def _obs_with_robot_state_timestamp(seconds: float) -> dict:
    whole_seconds = int(seconds)
    nanos = int(round((seconds - whole_seconds) * 1e9))
    return {
        "timestamp": {
            "robot_state": {
                "robot_timestamp_seconds": whole_seconds,
                "robot_timestamp_nanos": nanos,
            }
        }
    }


def test_rollout_refreshes_stale_post_step_robot_state_timestamp() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.calls = 0

        def get_observation(self) -> dict:
            self.calls += 1
            return _obs_with_robot_state_timestamp(1.2)

    env = FakeEnv()
    previous_obs = _obs_with_robot_state_timestamp(1.0)
    stale_obs = _obs_with_robot_state_timestamp(1.0)

    refreshed = _refresh_obs_until_robot_state_timestamp_advances(
        env,
        previous_obs=previous_obs,
        obs=stale_obs,
        max_wait_s=0.01,
        poll_interval_s=0.0,
    )

    assert refreshed["timestamp"]["robot_state"]["robot_timestamp_seconds"] == 1
    assert refreshed["timestamp"]["robot_state"]["robot_timestamp_nanos"] == 200000000
    assert env.calls == 1


def test_rollout_marks_persistently_stale_post_step_timestamp_recoverable() -> None:
    class FakeEnv:
        def get_observation(self) -> dict:
            return _obs_with_robot_state_timestamp(1.0)

    with pytest.raises(RecoverableRolloutError, match="timestamp did not advance"):
        _refresh_obs_until_robot_state_timestamp_advances(
            FakeEnv(),
            previous_obs=_obs_with_robot_state_timestamp(1.0),
            obs=_obs_with_robot_state_timestamp(1.0),
            max_wait_s=0.001,
            poll_interval_s=0.0,
        )


def test_manifest_eval_cleans_stale_image_episode_dirs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval-dataset"
    (dataset_path / "meta").mkdir(parents=True)
    (dataset_path / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2}))
    valid_dir = dataset_path / "images" / "observation.images.10000001_left" / "episode-000001"
    stale_dir = dataset_path / "images" / "observation.images.10000001_left" / "episode-000002"
    valid_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    (valid_dir / "frame-000000.png").write_bytes(b"valid")
    (stale_dir / "frame-000000.png").write_bytes(b"stale")

    removed = cleanup_stale_image_episode_dirs(dataset_path)

    assert removed == [stale_dir]
    assert valid_dir.exists()
    assert not stale_dir.exists()


def test_results_file_round_trips_round_plans(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    policies = [PolicyEntry("hf://org/a", 0, policy=object(), results=[True])]
    fixed = [ResolvedPolicy(name="a", model_id="hf://org/a", slot_key="fixed:0", slot_type="fixed")]
    target = SimpleNamespace(manifest_idx=0, pen_x=0.01, pen_y=-0.02, pen_yaw=0.3)
    import random

    plan = _make_round_plan(
        round_num=1,
        target=target,
        participants=fixed,
        policy_ids_by_slot_key={"fixed:0": 0},
        rng=random.Random(0),
    )
    save_results_file(
        path, policies, [], argparse.Namespace(func=None), "eval-dataset", round_plans=[plan]
    )
    raw = json.loads(path.read_text())
    assert raw["round_plans"] == [plan]
    assert "arena_session_id" not in raw
    assert _load_round_plans(path) == [plan]


def test_eval_initial_state_features_match_saved_fields() -> None:
    assert set(_eval_initial_state_features()) == {"manifest_idx"}
    features = _eval_initial_state_features(
        {"keys": ["pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y"]}
    )
    assert set(features) == {"manifest_idx", "pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y"}
    assert features["manifest_idx"]["dtype"] == "int64"
    assert features["pen_x"]["dtype"] == "float32"
