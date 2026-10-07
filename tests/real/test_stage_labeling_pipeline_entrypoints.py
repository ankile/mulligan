import dataclasses
from pathlib import Path

import pytest

from mulligan.real.stage_labeling.cascade_pipeline.common import (
    MarkerD2CascadeConfig,
    SquareD2CascadeConfig,
)
from mulligan.real.stage_labeling.cascade_pipeline.marker_d2 import (
    apply_marker_d2_cascade,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _adjust_square_d2_raw_results,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _apply_carry_endpoint_node_result,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _apply_peg_arrival_node_result,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _apply_s1_aligned_miss_timing_floor,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _apply_s1_subtype_node_result,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import _CARRY_ENDPOINT_NODE
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _carry_endpoint_candidate,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _raw_s1_aligned_miss_timing_floor,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _s1_subtype_timing_in_calibrated_band,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    _without_target_episode_few_shot,
)
from mulligan.real.stage_labeling.eval_sessions import MARKER_D2_R0_HELDOUT_DATASET_REPO_ID
from mulligan.real.stage_specs.square_d2 import apply_square_d2_r0_s1_failure_prior
from mulligan.real.stage_specs.square_d2 import apply_square_d2_endpoint_prior
from mulligan.real.stage_labeling.apply_cascade import build_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
SQUARE_D2_R0_DATASET_REPO_ID = "mulligan/real-square-d2-c00-teleop-mixed"


def test_stage_labeling_has_no_hardcoded_cloud_project() -> None:
    # The Gemini route is configured from the environment only (GEMINI_API_KEY, or
    # GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION for Vertex); nothing sets a project.
    root = REPO_ROOT / "mulligan/real/stage_labeling"
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in root.rglob("*.py")
        if "configure_institutional_vertex" in path.read_text()
        or '"GOOGLE_CLOUD_PROJECT":' in path.read_text()
    ]
    assert offenders == []


def test_marker_d2_cascade_default_model_is_gemini35_flash() -> None:
    config = MarkerD2CascadeConfig(
        input_run_name="input",
        output_run_name="output",
        dataset_repo_id=MARKER_D2_R0_HELDOUT_DATASET_REPO_ID,
        events_csv=Path("events.csv"),
    )

    assert config.model == "gemini-3.5-flash"


def test_marker_d2_non_r0_cascade_requires_repository_scoped_exemplars(
    tmp_path: Path,
) -> None:
    config = MarkerD2CascadeConfig(
        input_run_name="input",
        output_run_name="output",
        dataset_repo_id="org/a-later-marker-d2-eval",
        events_csv=tmp_path / "events.csv",
        runs_dir=tmp_path / "runs",
        build_dir=tmp_path / "target-assets",
    )

    with pytest.raises(ValueError, match="few-shot media must come from the reviewed R0"):
        apply_marker_d2_cascade(config)


class _StopAfterBuild(Exception):
    pass


def _stub_marker_cascade_inputs(monkeypatch, episodes: list[int]) -> list[set[int]]:
    import pandas as pd

    from mulligan.real.stage_labeling.cascade_pipeline import marker_d2 as cascade_marker

    built: list[set[int]] = []

    def fake_build_items(spec, build_dir, requested):
        built.append(set(requested))
        return [{"episode_index": ep} for ep in sorted(requested)]

    monkeypatch.setattr(
        cascade_marker, "load_raw", lambda _dir: [{"episode_index": ep} for ep in episodes]
    )
    monkeypatch.setattr(cascade_marker, "load_events", lambda _csv, eps: {ep: {} for ep in eps})
    monkeypatch.setattr(cascade_marker, "build_items", fake_build_items)
    monkeypatch.setattr(cascade_marker, "build_consensus", lambda raw, spec: pd.DataFrame())

    def no_gemini(*_a, **_k):
        raise AssertionError("no Gemini client may be created before the media preflight")

    monkeypatch.setattr(cascade_marker, "make_client", no_gemini)
    monkeypatch.setattr(cascade_marker, "configure_gemini", no_gemini)
    return built


def test_marker_d2_r0_cascade_builds_few_shot_exemplar_episodes(monkeypatch, tmp_path) -> None:
    from mulligan.real.stage_labeling.cascade import MARKER_D2_NODES

    built = _stub_marker_cascade_inputs(monkeypatch, [3, 5])
    config = MarkerD2CascadeConfig(
        input_run_name="input",
        output_run_name="output",
        dataset_repo_id=MARKER_D2_R0_HELDOUT_DATASET_REPO_ID,
        events_csv=tmp_path / "events.csv",
        runs_dir=tmp_path / "runs",
        build_dir=tmp_path / "assets",
    )
    # The stub builds no media, so the preflight must stop the run before any Gemini call.
    with pytest.raises(FileNotFoundError, match="few-shot media missing"):
        apply_marker_d2_cascade(config)

    exemplars = {entry[0] for node in MARKER_D2_NODES.values() for entry in node.few_shot}
    assert built == [{3, 5} | exemplars]


def test_marker_d2_non_r0_cascade_fails_before_gemini_on_missing_exemplars(
    monkeypatch, tmp_path
) -> None:
    built = _stub_marker_cascade_inputs(monkeypatch, [3, 5])
    config = MarkerD2CascadeConfig(
        input_run_name="input",
        output_run_name="output",
        dataset_repo_id="org/a-later-marker-d2-eval",
        events_csv=tmp_path / "events.csv",
        runs_dir=tmp_path / "runs",
        build_dir=tmp_path / "target-assets",
        exemplar_dataset_repo_id=MARKER_D2_R0_HELDOUT_DATASET_REPO_ID,
        exemplar_build_dir=tmp_path / "r0-assets",
    )
    with pytest.raises(FileNotFoundError, match="episode_008_wrist_final_crop.png"):
        apply_marker_d2_cascade(config)
    assert built == [{3, 5}]


def test_square_d2_cascade_config_has_no_fixed_peg_node_knobs() -> None:
    config = SquareD2CascadeConfig(
        input_run_name="input",
        output_run_name="output",
        dataset_repo_id=SQUARE_D2_R0_DATASET_REPO_ID,
        events_csv=Path("events.csv"),
    )

    assert config.model == "gemini-3.5-flash"
    assert config.included_episodes == frozenset()
    assert not hasattr(config, "apply_node_c")
    assert not hasattr(config, "override_min_frac")
    assert config.carry_endpoint_override_min_frac == 0.8
    assert config.peg_arrival_override_min_frac == 0.8


def test_square_d2_s1_failure_prior_matches_reviewed_taxonomy() -> None:
    label = {
        "max_stage": 1,
        "final_state": "nut_on_table",
        "failure_mode": "no_grasp_attempt",
        "grasp_acquired": False,
        "pregrasp_alignment_reached": True,
        "pregrasp_alignment_time_s": 3.0,
        "notes": "",
    }

    adjusted, prior = apply_square_d2_r0_s1_failure_prior(label)

    assert prior is not None
    assert adjusted["max_stage"] == 1
    assert adjusted["failure_mode"] == "pregrasp_misalignment"
    assert adjusted["pregrasp_alignment_reached"] is False
    assert adjusted["pregrasp_alignment_time_s"] is None


def test_square_d2_s1_failure_prior_canonicalizes_aligned_miss_backbone_votes() -> None:
    label = {
        "max_stage": 1,
        "final_state": "nut_on_table",
        "failure_mode": "aligned_grasp_miss",
        "grasp_acquired": False,
        "pregrasp_alignment_reached": True,
        "pregrasp_alignment_time_s": 3.0,
    }

    adjusted, prior = apply_square_d2_r0_s1_failure_prior(label)

    assert prior is not None
    assert adjusted["failure_mode"] == "pregrasp_misalignment"
    assert adjusted["pregrasp_alignment_reached"] is False


def test_square_d2_s1_failure_prior_preserves_node_decided_aligned_miss() -> None:
    label = {
        "max_stage": 1,
        "final_state": "nut_on_table",
        "failure_mode": "missed_grasp_after_alignment",
        "grasp_acquired": False,
        "pregrasp_alignment_reached": True,
        "pregrasp_alignment_time_s": 4.73,
    }

    adjusted, prior = apply_square_d2_r0_s1_failure_prior(label)

    assert prior is None
    assert adjusted == label


def test_square_d2_s1_subtype_node_result_sets_aligned_miss_fields() -> None:
    label = {
        "max_stage": 1,
        "final_state": "nut_on_table",
        "failure_mode": "pregrasp_misalignment",
        "grasp_acquired": False,
        "pregrasp_alignment_reached": False,
        "pregrasp_alignment_time_s": None,
        "grasp_attempt_time_s": 5.0,
        "needs_human_review": False,
        "notes": "",
    }

    adjusted, overrode, win_frac = _apply_s1_subtype_node_result(
        label,
        {
            "verdict": "missed_grasp_after_alignment",
            "stage": 1,
            "votes": [
                "missed_grasp_after_alignment",
                "missed_grasp_after_alignment",
                "pregrasp_misalignment",
                "missed_grasp_after_alignment",
                "missed_grasp_after_alignment",
            ],
        },
    )

    assert overrode is True
    assert win_frac == 0.8
    assert adjusted["max_stage"] == 1
    assert adjusted["failure_mode"] == "missed_grasp_after_alignment"
    assert adjusted["pregrasp_alignment_reached"] is True
    assert adjusted["pregrasp_alignment_time_s"] == 5.0


def test_square_d2_s1_subtype_timing_gate_matches_reviewed_boundary() -> None:
    assert _s1_subtype_timing_in_calibrated_band(
        {"pregrasp_alignment_time_s": 4.0, "grasp_attempt_time_s": 5.0}
    )
    assert not _s1_subtype_timing_in_calibrated_band(
        {"pregrasp_alignment_time_s": 4.8, "grasp_attempt_time_s": 5.2}
    )
    assert not _s1_subtype_timing_in_calibrated_band(
        {"pregrasp_alignment_time_s": 3.2, "grasp_attempt_time_s": 5.2}
    )


def test_square_d2_raw_s1_timing_floor_requires_multiple_aligned_samples_in_band() -> None:
    samples = [
        {
            "sample_idx": 0,
            "parsed": {
                "max_stage": 1,
                "final_state": "nut_on_table",
                "failure_mode": "aligned_grasp_miss",
                "grasp_acquired": False,
                "pregrasp_alignment_time_s": 3.6,
                "grasp_attempt_time_s": 5.2,
            },
        },
        {
            "sample_idx": 1,
            "parsed": {
                "max_stage": 1,
                "final_state": "nut_on_table",
                "failure_mode": "aligned_grasp_miss",
                "grasp_acquired": False,
                "pregrasp_alignment_time_s": 4.4,
                "grasp_attempt_time_s": 5.2,
            },
        },
        {
            "sample_idx": 2,
            "parsed": {
                "max_stage": 1,
                "final_state": "nut_on_table",
                "failure_mode": "pregrasp_misalignment",
                "grasp_acquired": False,
                "pregrasp_alignment_time_s": None,
                "grasp_attempt_time_s": 5.2,
            },
        },
    ]

    floor = _raw_s1_aligned_miss_timing_floor(samples)

    assert floor is not None
    assert floor["aligned_sample_count"] == 2
    assert floor["selected_sample"]["sample_idx"] == 1


def test_square_d2_raw_s1_timing_floor_rejects_near_miss_outside_band() -> None:
    samples = [
        {
            "sample_idx": idx,
            "parsed": {
                "max_stage": 1,
                "final_state": "nut_on_table",
                "failure_mode": "aligned_grasp_miss",
                "grasp_acquired": False,
                "pregrasp_alignment_time_s": 4.2,
                "grasp_attempt_time_s": 4.87,
            },
        }
        for idx in range(3)
    ]

    assert _raw_s1_aligned_miss_timing_floor(samples) is None


def test_square_d2_s1_timing_floor_sets_aligned_miss_fields() -> None:
    label = {
        "max_stage": 1,
        "final_state": "nut_on_table",
        "failure_mode": "pregrasp_misalignment",
        "pregrasp_alignment_reached": False,
        "pregrasp_alignment_time_s": None,
        "grasp_attempt_time_s": 5.2,
        "grasp_acquired": False,
        "notes": "",
    }
    floor = {
        "name": "square_d2_s1_aligned_miss_timing_floor",
        "selected_sample": {
            "sample_idx": 1,
            "pregrasp_alignment_time_s": 4.4,
            "grasp_attempt_time_s": 5.2,
            "delta_s": 0.8,
        },
    }

    adjusted, adjustment = _apply_s1_aligned_miss_timing_floor(label, floor)

    assert adjustment is not None
    assert adjusted["failure_mode"] == "missed_grasp_after_alignment"
    assert adjusted["pregrasp_alignment_reached"] is True
    assert adjusted["pregrasp_alignment_time_s"] == 4.4


def test_square_d2_endpoint_prior_demotes_non_success_success_endpoint() -> None:
    label = {
        "episode_index": 10,
        "max_stage": 4,
        "final_state": "nut_fully_seated_released",
        "failure_mode": "none",
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": True,
        "notes": "",
    }

    adjusted, prior = apply_square_d2_endpoint_prior(label, {"original_outcome": "failure"})

    assert prior is not None
    assert adjusted["max_stage"] == 5
    assert adjusted["final_state"] == "nut_partially_on_peg_released"
    assert adjusted["failure_mode"] == "released_partial_not_seated"
    assert adjusted["nut_fully_seated"] is False
    assert adjusted["nut_released"] is True


@pytest.mark.parametrize("outcome", ["", "nan", float("nan")])
def test_square_d2_carry_endpoint_candidate_skips_rows_without_outcome(outcome) -> None:
    # Teleop / collection events carry no rollout outcome; a seated+released S7 must
    # not be routed to the carry-endpoint override as a "non-success".
    label = {
        "episode_index": 9,
        "max_stage": 7,
        "final_state": "nut_fully_seated_released",
        "failure_mode": "none",
        "grasp_acquired": True,
        "nut_fully_seated": True,
        "nut_released": True,
    }

    assert not _carry_endpoint_candidate(label, {"original_outcome": outcome})
    assert _carry_endpoint_candidate(label, {"original_outcome": "failure"})
    assert not _carry_endpoint_candidate(label, {"original_outcome": "success"})


def test_square_d2_carry_endpoint_node_maps_lost_held_residual() -> None:
    label = {
        "episode_index": 40,
        "max_stage": 3,
        "final_state": "nut_in_gripper_at_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
        "grasp_lost_time_s": None,
        "peg_contact_time_s": 7.0,
        "peg_alignment_time_s": None,
        "nut_fully_seated_time_s": None,
        "nut_released_time_s": None,
    }

    assert _carry_endpoint_candidate(label, {"original_outcome": "failure"})
    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "passive_slip_before_controlled_peg_arrival_on_table",
            "votes": [
                "passive_slip_before_controlled_peg_arrival_on_table",
                "passive_slip_before_controlled_peg_arrival_on_table",
                "still_held_at_peg_not_seated",
            ],
        },
        event_row={"gripper_reopened_at_end": "True"},
        override_min_frac=0.6,
    )

    assert overrode is True
    assert win_frac == pytest.approx(2 / 3)
    assert adjusted["max_stage"] == 2
    assert adjusted["final_state"] == "nut_on_table"
    assert adjusted["failure_mode"] == "nut_slipped_from_gripper"
    assert adjusted["grasp_lost"] is True
    assert adjusted["nut_released"] is False
    assert adjusted["peg_contact_time_s"] is None


def test_square_d2_carry_endpoint_node_allows_structural_family_override() -> None:
    label = {
        "episode_index": 14,
        "max_stage": 3,
        "final_state": "nut_in_gripper_at_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
        "grasp_lost_time_s": None,
        "peg_contact_time_s": 7.5,
        "peg_alignment_time_s": None,
        "nut_fully_seated_time_s": None,
        "nut_released_time_s": None,
    }

    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "passive_slip_after_controlled_peg_arrival_on_table",
            "votes": [
                "passive_slip_after_controlled_peg_arrival_on_table",
                "dropped_during_transport_off_table",
                "passive_slip_after_controlled_peg_arrival_on_table",
                "dropped_during_transport_off_table",
                "passive_slip_after_controlled_peg_arrival_on_table",
            ],
        },
        event_row={"gripper_reopened_at_end": "True"},
        override_min_frac=0.8,
    )

    assert overrode is True
    assert win_frac == 1.0
    assert adjusted["max_stage"] == 3
    assert adjusted["final_state"] == "nut_on_table"
    assert adjusted["grasp_lost"] is True


def test_square_d2_carry_endpoint_node_blocks_family_override_with_current_family_vote() -> None:
    label = {
        "episode_index": 97,
        "max_stage": 3,
        "final_state": "nut_in_gripper_at_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
    }

    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "dropped_during_transport_off_table",
            "votes": [
                "dropped_during_transport_off_table",
                "passive_slip_before_controlled_peg_arrival_on_table",
                "dropped_during_transport_off_table",
                "dropped_during_transport_off_table",
                "still_held_at_peg_not_seated",
            ],
        },
        event_row={"gripper_reopened_at_end": "False"},
        override_min_frac=0.8,
    )

    assert overrode is False
    assert win_frac == pytest.approx(3 / 5)
    assert adjusted == label


def test_square_d2_carry_endpoint_fully_seated_verdict_does_not_set_seated_bool() -> None:
    label = {
        "episode_index": 8,
        "max_stage": 3,
        "final_state": "nut_in_gripper_at_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
        "grasp_lost_time_s": None,
        "peg_alignment_time_s": None,
        "nut_fully_seated_time_s": None,
        "nut_released_time_s": None,
    }

    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "still_held_at_peg_fully_seated",
            "votes": [
                "still_held_at_peg_fully_seated",
                "still_held_at_peg_not_seated",
                "still_held_at_peg_fully_seated",
                "still_held_at_peg_fully_seated",
                "still_held_at_peg_fully_seated",
            ],
        },
        event_row={"gripper_reopened_at_end": "False"},
        override_min_frac=0.8,
    )

    assert overrode is True
    assert win_frac == 0.8
    assert adjusted["max_stage"] == 3
    assert adjusted["final_state"] == "nut_in_gripper_at_peg"
    assert adjusted["nut_fully_seated"] is False
    assert adjusted["nut_fully_seated_time_s"] is None


def test_square_d2_carry_endpoint_aliases_old_cached_verdicts() -> None:
    label = {
        "episode_index": 94,
        "max_stage": 2,
        "final_state": "nut_in_gripper_away_from_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
        "grasp_lost_time_s": None,
        "nut_released_time_s": None,
        "peg_contact_time_s": None,
        "peg_alignment_time_s": None,
        "nut_fully_seated_time_s": None,
    }

    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "lost_before_controlled_peg_arrival_on_table",
            "votes": [
                "lost_before_controlled_peg_arrival_on_table",
                "lost_before_controlled_peg_arrival_on_table",
                "lost_before_controlled_peg_arrival_on_table",
            ],
        },
        event_row={"gripper_reopened_at_end": "True"},
        override_min_frac=0.6,
    )

    assert overrode is True
    assert win_frac == 1.0
    assert adjusted["max_stage"] == 2
    assert adjusted["final_state"] == "nut_on_table"
    assert adjusted["failure_mode"] == "nut_slipped_from_gripper"
    assert adjusted["grasp_lost"] is True


def test_square_d2_carry_endpoint_blocks_release_without_reopen() -> None:
    label = {
        "episode_index": 29,
        "max_stage": 3,
        "final_state": "nut_in_gripper_at_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
    }

    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "controlled_release_at_peg_not_seated_on_table",
            "votes": [
                "controlled_release_at_peg_not_seated_on_table",
                "controlled_release_at_peg_not_seated_on_table",
                "controlled_release_at_peg_not_seated_on_table",
                "controlled_release_at_peg_not_seated_on_table",
                "controlled_release_at_peg_not_seated_on_table",
            ],
        },
        event_row={"gripper_reopened_at_end": "False"},
        override_min_frac=0.8,
    )

    assert overrode is False
    assert win_frac == 1.0
    assert adjusted == label


def test_square_d2_carry_endpoint_maps_existing_release_family_to_table_release() -> None:
    label = {
        "episode_index": 10,
        "max_stage": 5,
        "final_state": "nut_partially_on_peg_released",
        "failure_mode": "released_partial_not_seated",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": True,
        "notes": "",
    }

    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "passive_slip_after_controlled_peg_arrival_on_table",
            "votes": [
                "passive_slip_after_controlled_peg_arrival_on_table",
                "passive_slip_after_controlled_peg_arrival_on_table",
                "passive_slip_after_controlled_peg_arrival_on_table",
                "passive_slip_after_controlled_peg_arrival_on_table",
                "passive_slip_after_controlled_peg_arrival_on_table",
            ],
        },
        event_row={"gripper_reopened_at_end": "True"},
        override_min_frac=0.8,
    )

    assert overrode is True
    assert win_frac == 1.0
    assert adjusted["max_stage"] == 3
    assert adjusted["final_state"] == "nut_on_table"
    assert adjusted["failure_mode"] == "released_partial_not_seated"
    assert adjusted["grasp_lost"] is False
    assert adjusted["nut_released"] is True


def test_square_d2_carry_endpoint_reopen_allows_family_majority_over_held() -> None:
    label = {
        "episode_index": 143,
        "max_stage": 3,
        "final_state": "nut_in_gripper_at_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
        "grasp_lost_time_s": None,
        "nut_released_time_s": None,
        "peg_contact_time_s": 7.3,
        "peg_alignment_time_s": None,
        "nut_fully_seated_time_s": None,
    }

    adjusted, overrode, win_frac = _apply_carry_endpoint_node_result(
        label,
        {
            "verdict": "dropped_during_transport_on_table",
            "votes": [
                "still_held_at_peg_fully_seated",
                "controlled_release_at_peg_not_seated_on_table",
                "dropped_during_transport_on_table",
                "dropped_during_transport_on_table",
                "dropped_during_transport_on_table",
            ],
        },
        event_row={"gripper_reopened_at_end": "True"},
        override_min_frac=0.8,
    )

    assert overrode is True
    assert win_frac == pytest.approx(3 / 5)
    assert adjusted["max_stage"] == 2
    assert adjusted["final_state"] == "nut_on_table"
    assert adjusted["failure_mode"] == "dropped_nut_during_transport"
    assert adjusted["grasp_lost"] is True
    assert adjusted["nut_released"] is False


def test_square_d2_peg_arrival_demotes_side_only_held_boundary() -> None:
    label = {
        "episode_index": 89,
        "max_stage": 3,
        "final_state": "nut_in_gripper_at_peg",
        "failure_mode": "timeout_holding_nut",
        "grasp_acquired": True,
        "grasp_lost": False,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
        "grasp_lost_time_s": None,
        "nut_released_time_s": None,
        "peg_contact_time_s": 5.2,
        "peg_alignment_time_s": None,
        "nut_fully_seated_time_s": None,
    }

    adjusted, overrode, win_frac = _apply_peg_arrival_node_result(
        label,
        {
            "verdict": "no_controlled_peg_arrival",
            "votes": [
                "no_controlled_peg_arrival",
                "no_controlled_peg_arrival",
                "no_controlled_peg_arrival",
                "no_controlled_peg_arrival",
                "controlled_peg_arrival",
            ],
        },
        override_min_frac=0.8,
    )

    assert overrode is True
    assert win_frac == 0.8
    assert adjusted["max_stage"] == 2
    assert adjusted["final_state"] == "nut_in_gripper_away_from_peg"
    assert adjusted["failure_mode"] == "timeout_holding_nut"
    assert adjusted["grasp_lost"] is False
    assert adjusted["peg_contact_time_s"] is None


def test_square_d2_peg_arrival_demotes_table_loss_to_transport_drop() -> None:
    label = {
        "episode_index": 143,
        "max_stage": 3,
        "final_state": "nut_on_table",
        "failure_mode": "nut_slipped_from_gripper",
        "grasp_acquired": True,
        "grasp_lost": True,
        "nut_fully_seated": False,
        "nut_released": False,
        "notes": "",
        "nut_released_time_s": None,
        "peg_contact_time_s": 7.3,
        "peg_alignment_time_s": None,
        "nut_fully_seated_time_s": None,
    }

    adjusted, overrode, win_frac = _apply_peg_arrival_node_result(
        label,
        {
            "verdict": "no_controlled_peg_arrival",
            "votes": ["no_controlled_peg_arrival"] * 5,
        },
        override_min_frac=0.8,
    )

    assert overrode is True
    assert win_frac == 1.0
    assert adjusted["max_stage"] == 2
    assert adjusted["final_state"] == "nut_on_table"
    assert adjusted["failure_mode"] == "dropped_nut_during_transport"
    assert adjusted["grasp_lost"] is True
    assert adjusted["nut_released"] is False


def test_square_d2_target_episode_omitted_from_own_few_shot() -> None:
    node = dataclasses.replace(
        _CARRY_ENDPOINT_NODE,
        few_shot=(
            (10, "controlled_release_at_peg_not_seated_on_table", "same target"),
            (18, "passive_slip_after_controlled_peg_arrival_on_table", "other target"),
        ),
    )

    adjusted = _without_target_episode_few_shot(node, 10)

    assert [entry[0] for entry in adjusted.few_shot] == [18]


def test_apply_stage_label_cascade_parser_accepts_square_d2() -> None:
    args = build_parser().parse_args(
        [
            "--task",
            "square_d2",
            "--input-run-name",
            "input",
            "--output-run-name",
            "output",
            "--dataset-repo-id",
            SQUARE_D2_R0_DATASET_REPO_ID,
            "--events-csv",
            "events.csv",
        ]
    )

    assert args.task == "square_d2"
    assert args.episode_indices == ""


def test_apply_stage_label_cascade_parser_accepts_square_d2_episode_indices() -> None:
    args = build_parser().parse_args(
        [
            "--task",
            "square_d2",
            "--input-run-name",
            "input",
            "--output-run-name",
            "output",
            "--dataset-repo-id",
            SQUARE_D2_R0_DATASET_REPO_ID,
            "--events-csv",
            "events.csv",
            "--episode-indices",
            "0-2,10",
        ]
    )

    assert args.task == "square_d2"
    assert args.episode_indices == "0-2,10"


def test_square_d2_cascade_preserves_backbone_samples_when_prior_applies() -> None:
    raw = [
        {
            "episode_index": 17,
            "sample_idx": 0,
            "parsed": {"max_stage": 1, "failure_mode": "pregrasp_misalignment"},
        },
        {
            "episode_index": 17,
            "sample_idx": 1,
            "parsed": {"max_stage": 1, "failure_mode": "aligned_grasp_miss"},
        },
    ]
    labels = {17: {"max_stage": 1, "failure_mode": "pregrasp_misalignment"}}
    refinements = {17: {"label_prior_applied": True}}

    adjusted = _adjust_square_d2_raw_results(raw, labels, refinements)

    assert [r["parsed"]["failure_mode"] for r in adjusted] == [
        "pregrasp_misalignment",
        "pregrasp_misalignment",
    ]
    assert [r["backbone_parsed"]["failure_mode"] for r in adjusted] == [
        "pregrasp_misalignment",
        "aligned_grasp_miss",
    ]
