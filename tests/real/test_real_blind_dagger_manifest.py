import ast
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mulligan.data.constants import DataSource, EpisodeOutcome
from mulligan.real.collect.blind_dagger import (
    ArmState,
    _add_live_frame_to_dataset,
    _constant_extra_frame_fields,
    _initial_state_features,
    _patch_live_episode_buffer_for_final_outcome,
    _select_protocol_manifest_target,
    _select_manifest_target,
    _target_extra_frame_fields,
    _target_ledger_fields,
)
from mulligan.real.collect.initial_states import ArmSpec, _load_initial_state_manifest
from mulligan.real.operator_ui.cards import cleanup_unreferenced_initial_state_cards
from mulligan.sim.collect.quota import ProtocolQuotaLedger
from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeLiveDataset:
    def __init__(self) -> None:
        self.writer = SimpleNamespace(episode_buffer={"size": 0, "task": []})

    def add_frame(self, frame: dict) -> None:
        buffer = self.writer.episode_buffer
        buffer["task"].append(frame["task"])
        for key, value in frame.items():
            if key == "task":
                continue
            buffer.setdefault(key, []).append(value)
        buffer["size"] += 1


def _live_episode_data(n_frames: int = 3) -> dict:
    data = {
        "observations": [np.full((7,), i, dtype=np.float32) for i in range(n_frames)],
        "actions": [np.full((7,), i + 0.5, dtype=np.float32) for i in range(n_frames)],
        "steps_to_go": [n_frames - 1 - i for i in range(n_frames)],
        "rewards": [0.0, 1.0, 1.0],
        "dones": [0, 1, 1],
        "joint_positions": [np.zeros(7, dtype=np.float32) for _ in range(n_frames)],
        "joint_velocities": [np.zeros(7, dtype=np.float32) for _ in range(n_frames)],
        "cartesian_velocities": [np.zeros(6, dtype=np.float32) for _ in range(n_frames)],
        "action_info_cartesian_velocity": [np.zeros(6, dtype=np.float32) for _ in range(n_frames)],
        "action_info_cartesian_position": [np.zeros(6, dtype=np.float32) for _ in range(n_frames)],
        "action_info_joint_velocity": [np.zeros(7, dtype=np.float32) for _ in range(n_frames)],
        "action_info_joint_position": [np.zeros(7, dtype=np.float32) for _ in range(n_frames)],
        "action_info_gripper_position": [np.zeros(1, dtype=np.float32) for _ in range(n_frames)],
        "action_info_gripper_velocity": [np.zeros(1, dtype=np.float32) for _ in range(n_frames)],
        "image_20000002_left": [np.zeros((4, 4, 3), dtype=np.uint8) + i for i in range(n_frames)],
    }
    return data


def test_live_blind_dagger_buffer_patch_sets_final_dagger_labels() -> None:
    dataset = _FakeLiveDataset()
    arm_state = ArmState(
        spec=ArmSpec("blind_arm", "model"),
        arm_id=2,
        policy_id=7,
        policy=object(),
        camera_height=480,
        camera_width=640,
    )
    episode_data = _live_episode_data()
    extra_fields = _constant_extra_frame_fields(
        arm_state=arm_state,
        target=None,
        manifest_meta=None,
    )
    sources = [DataSource.AUTONOMOUS, DataSource.AUTONOMOUS, DataSource.HUMAN]

    for frame_index, source in enumerate(sources):
        _add_live_frame_to_dataset(
            dataset,
            episode_data,
            frame_index,
            camera_keys=["image_20000002_left"],
            task_name="marker_d2",
            source_id=int(source),
            extra_frame_fields=extra_fields,
        )

    buffer = dataset.writer.episode_buffer
    assert buffer["size"] == 3
    assert [int(v[0]) for v in buffer["success"]] == [EpisodeOutcome.FAILURE] * 3
    assert [int(v[0]) for v in buffer["intervention"]] == [0, 0, 0]

    intervention_flags = _patch_live_episode_buffer_for_final_outcome(
        dataset,
        episode_data,
        sources=sources,
        episode_success=True,
    )

    assert intervention_flags == [0, 1, 0]
    assert [int(v[0]) for v in buffer["success"]] == [EpisodeOutcome.SUCCESS] * 3
    assert [int(v[0]) for v in buffer["intervention"]] == intervention_flags
    assert [int(v[0]) for v in buffer["source"]] == [0, 0, 1]
    assert [int(v[0]) for v in buffer["is_valid"]] == [1, 1, 0]
    assert [float(v[0]) for v in buffer["reward"]] == [0.0, 1.0, 1.0]
    assert [int(v[0]) for v in buffer["done"]] == [0, 1, 1]
    assert [int(v[0]) for v in buffer["steps_to_go"]] == [2, 1, 0]
    assert [int(v[0]) for v in buffer["arm_id"]] == [2, 2, 2]
    assert "observation.images.side_1" in buffer


def test_vision_idql_policy_exposes_droid_action_space_contract() -> None:
    assert VisionIDQLRealWorldPolicy.action_space == "cartesian_velocity"
    assert VisionIDQLRealWorldPolicy.gripper_action_space is None


MARKER_D2_R0_MANIFEST = (
    REPO_ROOT
    / "data/real/manifests/marker_d2/r00"
    / "marker_d2_r0_three_arm_uniform100_sobol100_evalsobol50_scrambled.json"
)
# A holder placement on the marker_d2 registry grid (7 in, -2 in), in meters.
HOLDER = {"holder_x": 7 * 0.0254, "holder_y": -2 * 0.0254}


def _write_manifest(path: Path, states: list[dict]) -> None:
    """A marker_d2 manifest (the shipped R0 header) with ``states`` and an on-grid holder."""
    payload = json.loads(MARKER_D2_R0_MANIFEST.read_text())
    payload["states"] = [{**HOLDER, **row} for row in states]
    payload.pop("arms", None)
    path.write_text(json.dumps(payload))


def _write_nut_peg_manifest(path: Path, states: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "task": "fixed_peg_nut",
                "schema": "real_manual_initial_states_v1",
                "keys": ["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"],
                "units": {
                    "nut_x": "m",
                    "nut_y": "m",
                    "nut_yaw": "rad",
                    "peg_x": "m",
                    "peg_y": "m",
                },
                "bounds": {
                    "nut_x": [-0.1524, 0.1524],
                    "nut_y": [-0.254, 0.254],
                    "nut_yaw": [-math.pi, math.pi],
                    "peg_x": [0.2794, 0.2794],
                    "peg_y": [-0.0508, -0.0508],
                },
                "states": states,
            }
        )
    )


def test_manifest_loader_accepts_nut_peg_task_definition_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "nut_peg.json"
    _write_nut_peg_manifest(
        manifest,
        [
            {
                "manifest_idx": 0,
                "source": "pilot_taskdef",
                "source_index": 0,
                "nut_x": 0.01,
                "nut_y": -0.02,
                "nut_yaw": 0.3,
                "peg_x": 0.2794,
                "peg_y": -0.0508,
            }
        ],
    )

    targets, meta = _load_initial_state_manifest(
        manifest,
        [ArmSpec("pilot_taskdef", "teleop")],
        expected_task="fixed_peg_nut",
    )

    target = targets[0]
    assert target.nut_x == pytest.approx(0.01)
    assert target.nut_y == pytest.approx(-0.02)
    assert target.nut_yaw == pytest.approx(0.3)
    assert target.peg_x == pytest.approx(0.2794)
    assert target.peg_y == pytest.approx(-0.0508)
    assert target.pen_x == pytest.approx(target.nut_x)
    assert set(_initial_state_features(meta)) == {
        "manifest_idx",
        "nut_x",
        "nut_y",
        "nut_yaw",
        "peg_x",
        "peg_y",
    }
    extra_fields = _target_extra_frame_fields(target, meta)
    assert set(extra_fields) == set(_initial_state_features(meta))
    assert extra_fields["manifest_idx"].dtype.name == "int64"
    assert extra_fields["nut_x"].dtype.name == "float32"
    assert _target_ledger_fields(target, meta) == {
        "nut_x": pytest.approx(0.01),
        "nut_y": pytest.approx(-0.02),
        "nut_yaw": pytest.approx(0.3),
        "peg_x": pytest.approx(0.2794),
        "peg_y": pytest.approx(-0.0508),
    }


def test_manifest_loader_accepts_a_sources_list(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "manifest_idx": 0,
                "sources": ["baseline_uniform"],
                "source_index": 100,
                "pen_x": 0.0,
                "pen_y": 0.0,
                "pen_yaw": 0.0,
            }
        ],
    )

    targets, _ = _load_initial_state_manifest(
        manifest,
        [ArmSpec("baseline_uniform", "model")],
        expected_task="marker_d2",
    )

    assert targets[0].source == "baseline_uniform"
    assert targets[0].source_index == 100


def test_manifest_loader_rejects_source_sources_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "manifest_idx": 0,
                "source": "baseline_uniform",
                "sources": ["mulligan_sobol"],
                "source_index": 0,
                "pen_x": 0.0,
                "pen_y": 0.0,
                "pen_yaw": 0.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="has source"):
        _load_initial_state_manifest(
            manifest,
            [ArmSpec("baseline_uniform", "a"), ArmSpec("mulligan_sobol", "b")],
        )


def test_manifest_loader_rejects_unknown_source(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "manifest_idx": 0,
                "source": "missing_arm",
                "source_index": 0,
                "pen_x": 0.0,
                "pen_y": 0.0,
                "pen_yaw": 0.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="no matching --arm"):
        _load_initial_state_manifest(manifest, [ArmSpec("baseline_uniform", "model")])


def test_manifest_selection_uses_source_order_and_skips_consumed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "manifest_idx": 0,
                "source": "baseline_uniform",
                "source_index": 0,
                "pen_x": 0.0,
                "pen_y": 0.0,
                "pen_yaw": 0.0,
            },
            {
                "manifest_idx": 1,
                "source": "mulligan_sobol",
                "source_index": 0,
                "pen_x": 0.0,
                "pen_y": 0.0,
                "pen_yaw": math.pi / 2,
            },
        ],
    )
    targets, _ = _load_initial_state_manifest(
        manifest,
        [ArmSpec("baseline_uniform", "a"), ArmSpec("mulligan_sobol", "b")],
    )
    selected = _select_manifest_target(
        targets,
        consumed_manifest_idxs={0},
        success_counts=Counter({"baseline_uniform": 0, "mulligan_sobol": 0}),
        target_success_per_arm=1,
    )
    assert selected is not None
    assert selected.manifest_idx == 1
    assert selected.source == "mulligan_sobol"


def test_protocol_manifest_selection_follows_scrambled_manifest_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "manifest_idx": 0,
                "source": "baseline_uniform",
                "sources": ["baseline_uniform"],
                "source_index": 0,
                "pen_x": 0.0,
                "pen_y": 0.0,
                "pen_yaw": 0.0,
            },
            {
                "manifest_idx": 1,
                "source": "mulligan_sobol",
                "sources": ["mulligan_sobol"],
                "source_index": 0,
                "pen_x": 0.01,
                "pen_y": 0.0,
                "pen_yaw": 0.0,
            },
            {
                "manifest_idx": 2,
                "source": "baseline_uniform",
                "sources": ["baseline_uniform"],
                "source_index": 1,
                "pen_x": 0.02,
                "pen_y": 0.0,
                "pen_yaw": 0.0,
            },
        ],
    )
    targets, _ = _load_initial_state_manifest(
        manifest,
        [ArmSpec("baseline_uniform", "a"), ArmSpec("mulligan_sobol", "b")],
    )
    quota = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=tmp_path / "protocol.jsonl",
        arms_by_protocol={
            "no_cf": ["baseline_uniform", "mulligan_sobol"],
            "with_cf": ["baseline_uniform", "mulligan_sobol"],
        },
        balance_slack=10,
    )

    first = _select_protocol_manifest_target(targets, quota)
    assert first is not None
    assert first.manifest_idx == 0

    quota.credit_episode(
        manifest_idx=0,
        episode_index=0,
        success=False,
        is_counterfactual=False,
        quota_credit=True,
    )
    second = _select_protocol_manifest_target(targets, quota)
    assert second is not None
    assert second.manifest_idx == 1


LOCKED_DAGGER_MANIFESTS = [
    (task, rnd) for task in ("marker_d2", "square_d2") for rnd in range(1, 6)
]


def _max_same_arm_run(sources: list[str], arm: str) -> int:
    best = current = 0
    for source in sources:
        current = current + 1 if source == arm else 0
        best = max(best, current)
    return best


def _simulate_locked_protocol_manifest(
    tmp_path: Path, task: str, rnd: int, *, selection_mode: str, balance_slack: int, cf: bool
) -> tuple[ProtocolQuotaLedger, list[str], int, int]:
    """Run the paper's blind-DAgger protocol (no_cf 50/50 both arms, with_cf 50 Ours only) over a
    shipped locked manifest, crediting every fresh rollout and, with ``cf``, every allowed
    counterfactual."""
    manifest = (
        REPO_ROOT
        / "data/real/manifests"
        / task
        / f"r{rnd:02d}"
        / f"{task}_r{rnd}_promote25_fill25_blind_dagger.json"
    )
    targets, _ = _load_initial_state_manifest(
        manifest,
        [ArmSpec("baseline_uniform", "a"), ArmSpec("mulligan_sobol", "b")],
        expected_task=task,
    )
    quota = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 50, "with_cf": 50},
        ledger_path=tmp_path / "protocol.jsonl",
        arms_by_protocol={
            "no_cf": ["baseline_uniform", "mulligan_sobol"],
            "with_cf": ["mulligan_sobol"],
        },
        balance_slack=balance_slack,
        selection_mode=selection_mode,
    )
    fresh_sources: list[str] = []
    counterfactuals = max_spread = episode_index = 0
    while not quota.is_complete():
        target = _select_protocol_manifest_target(targets, quota)
        assert target is not None, quota.counts
        fresh_sources.append(target.source)
        row = quota.credit_episode(
            manifest_idx=target.manifest_idx,
            episode_index=episode_index,
            success=True,
            is_counterfactual=False,
        )
        episode_index += 1
        spread = list(quota.counts["no_cf"].values())
        max_spread = max(max_spread, max(spread) - min(spread))
        if target.source == "baseline_uniform":
            assert row["credited_protocol_arms"] == {"no_cf": ["baseline_uniform"]}
            assert not quota.can_accept_counterfactual(target.manifest_idx)
        else:
            assert row["credited_protocol_arms"]["no_cf"] == ["mulligan_sobol"]
        if cf and quota.can_accept_counterfactual(target.manifest_idx):
            cf_row = quota.credit_episode(
                manifest_idx=target.manifest_idx,
                episode_index=episode_index,
                success=True,
                is_counterfactual=True,
            )
            episode_index += 1
            counterfactuals += 1
            assert cf_row["credited_protocol_arms"] == {"with_cf": ["mulligan_sobol"]}
    return quota, fresh_sources, counterfactuals, max_spread


@pytest.mark.parametrize("cf", [False, True], ids=["no_cf", "aggressive_cf"])
@pytest.mark.parametrize(
    ("selection_mode", "balance_slack", "max_spread", "max_baseline_run"),
    [("hard_balance", 2, 2, 4), ("soft_weighted", 1, 5, 6)],
    ids=["hard_balance", "soft_weighted"],
)
@pytest.mark.parametrize(
    ("task", "rnd"), LOCKED_DAGGER_MANIFESTS, ids=[f"{t}-r{r}" for t, r in LOCKED_DAGGER_MANIFESTS]
)
def test_locked_protocol_manifest_stays_blind_balanced(
    tmp_path: Path, task, rnd, selection_mode, balance_slack, max_spread, max_baseline_run, cf
) -> None:
    quota, fresh, counterfactuals, spread = _simulate_locked_protocol_manifest(
        tmp_path, task, rnd, selection_mode=selection_mode, balance_slack=balance_slack, cf=cf
    )
    assert quota.counts == {
        "no_cf": {"baseline_uniform": 50, "mulligan_sobol": 50},
        "with_cf": {"mulligan_sobol": 50},
    }
    assert len(fresh) == 100
    assert (counterfactuals > 0) == cf
    assert spread <= max_spread
    assert set(fresh[-12:]) == {"baseline_uniform", "mulligan_sobol"}
    assert _max_same_arm_run(fresh, "baseline_uniform") <= max_baseline_run


def test_cleanup_unreferenced_initial_state_visualizations(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    keep = target_dir / "target_0000.png"
    remove = target_dir / "target_0001.png"
    unrelated = target_dir / "notes.txt"
    keep.write_bytes(b"keep")
    remove.write_bytes(b"remove")
    unrelated.write_text("not a target card")

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"initial_state_visualization_path": str(keep)}) + "\n")

    removed = cleanup_unreferenced_initial_state_cards(target_dir, ledger)

    assert removed == [remove]
    assert keep.exists()
    assert not remove.exists()
    assert unrelated.exists()


def test_initial_state_visualization_calls_pass_task_name() -> None:
    paths = [
        REPO_ROOT / "mulligan/real/eval/blind_eval_helpers.py",
        REPO_ROOT / "mulligan/real/collect/teleop.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            else:
                continue
            if func_name == "_write_initial_state_visualization":
                keyword_names = {kw.arg for kw in node.keywords}
                assert "task_name" in keyword_names, (
                    f"{path}:{node.lineno} must pass task_name to "
                    "_write_initial_state_visualization"
                )
