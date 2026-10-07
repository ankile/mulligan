"""Characterization guards for ``mulligan.real.stage_labeling.consensus``.

Small SYNTHETIC inputs (a few episodes x 3 self-consistency samples; a tiny events
DataFrame) exercise ``_consensus_episode`` / ``_majority`` and the adjudication flag
logic on the marker_d2 field vocabulary. ``release_contradicts_sensor`` must not
KeyError when the events CSV lacks the ``gripper_reopened_at_end`` column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mulligan.real.stage_specs import get_label_task_spec
from mulligan.real.stage_labeling.consensus import (
    _consensus_episode,
    adjudicate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "marker_d2"


@pytest.fixture(scope="module")
def marker_spec():
    return get_label_task_spec(MARKER)


# --------------------------------------------------------------------------- #
# Synthetic sample builders (parsed-label dicts, marker field vocabulary).
# --------------------------------------------------------------------------- #


def _parsed(
    *,
    stage: int,
    final_state: str = "marker_fully_seated_released",
    failure_mode: str = "none",
    confidence: str = "high",
    needs_review: bool = False,
    notes: str = "n",
    bools: dict | None = None,
    times: dict | None = None,
) -> dict:
    """One parsed-label dict in the marker (v2) field vocabulary."""
    spec = get_label_task_spec(MARKER)
    bools = bools or {}
    times = times or {}
    p: dict = {
        "max_stage_v2": stage,
        "final_state": final_state,
        "early_failure_mode_v2": failure_mode,
        "confidence": confidence,
        "needs_human_review": needs_review,
        "notes": notes,
    }
    for f in spec.bool_fields:
        p[f] = bools.get(f, False)
    for f in spec.time_fields:
        p[f] = times.get(f, None)
    return p


def _samples(episode_index: int, parsed_list: list[dict]) -> list[dict]:
    """Wrap parsed dicts as the labeler's raw per-sample records."""
    return [{"episode_index": episode_index, "parsed": p} for p in parsed_list]


# Representative synthetic episodes covering: unanimous stage, tie->lower,
# majority bool, median times (incl. None), min-confidence, review OR, and a
# stage-7-with-mismatched-final-state.
SYNTHETIC_EPISODES = {
    # unanimous S7 full success, all bools true, all samples agree
    0: [
        _parsed(
            stage=7,
            bools={"marker_released": True, "marker_fully_seated": True, "grasp_acquired": True},
            times={"approach_time_s": 1.0, "marker_released_time_s": 9.0},
        ),
        _parsed(
            stage=7,
            bools={"marker_released": True, "marker_fully_seated": True, "grasp_acquired": True},
            times={"approach_time_s": 1.2, "marker_released_time_s": 9.2},
        ),
        _parsed(
            stage=7,
            bools={"marker_released": True, "marker_fully_seated": True, "grasp_acquired": True},
            times={"approach_time_s": 1.1, "marker_released_time_s": 9.1},
        ),
    ],
    # stage tie 3,3,5 -> majority 3 (no tie here) but bool split 2/3, mixed conf
    1: [
        _parsed(
            stage=3,
            confidence="high",
            bools={"grasp_acquired": True},
            times={"approach_time_s": 2.0},
        ),
        _parsed(
            stage=3,
            confidence="low",
            bools={"grasp_acquired": True},
            times={"approach_time_s": 3.0},
        ),
        _parsed(
            stage=5,
            confidence="medium",
            bools={"grasp_acquired": False},
            times={"approach_time_s": None},
        ),
    ],
    # genuine tie 2 vs 6 (one each + one of 2) -> majority 2; disagreement true
    2: [
        _parsed(stage=2, needs_review=False, times={"grasp_acquired_time_s": 4.0}),
        _parsed(stage=6, needs_review=True, times={"grasp_acquired_time_s": 6.0}),
        _parsed(stage=2, needs_review=False, times={"grasp_acquired_time_s": 8.0}),
    ],
    # perfect 2-way tie on stage (4,4,2,2) is even; use 4,2 -> tie breaks to 2
    3: [
        _parsed(stage=4, final_state="marker_on_table", failure_mode="slip"),
        _parsed(stage=2, final_state="marker_on_table", failure_mode="missed_grasp"),
    ],
    # S7 stage but final_state NOT the success final state -> mismatch flag later
    4: [
        _parsed(stage=7, final_state="marker_partially_in_holder_released"),
        _parsed(stage=7, final_state="marker_partially_in_holder_released"),
        _parsed(stage=7, final_state="marker_fully_seated_released"),
    ],
}


# --------------------------------------------------------------------------- #
# _majority parity.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# _consensus_episode parity (per-episode and via build_consensus).
# --------------------------------------------------------------------------- #


def test_consensus_voting_details(marker_spec):
    """Spot-check the documented voting semantics directly."""
    # episode 2: stage tie 2,6,2 -> 2, disagreement true, review OR'd true
    row = _consensus_episode(_samples(2, SYNTHETIC_EPISODES[2]), marker_spec)
    assert row["max_stage_v2"] == 2
    assert row["sample_stage_disagreement"] is True
    assert row["model_requested_review"] is True
    assert row["stage_min"] == 2 and row["stage_max"] == 6
    # median of grasp_acquired_time_s over [4,6,8] = 6.0
    assert row["grasp_acquired_time_s"] == 6.0

    # episode 1: min confidence over high/low/medium -> low; bool 2/3 true -> True
    row1 = _consensus_episode(_samples(1, SYNTHETIC_EPISODES[1]), marker_spec)
    assert row1["gemini_confidence"] == "low"
    assert row1["grasp_acquired"] is True  # 2 of 3
    # median approach_time_s over present [2.0, 3.0] (None dropped) = 2.5
    assert row1["approach_time_s"] == 2.5

    # episode 3: even tie 4 vs 2 -> 2 (smallest)
    row3 = _consensus_episode(_samples(3, SYNTHETIC_EPISODES[3]), marker_spec)
    assert row3["max_stage_v2"] == 2
    assert row3["sample_stage_disagreement"] is True


# --------------------------------------------------------------------------- #
# Adjudication-flag parity.
#
# We replicate the reference flag block exactly on the same tiny joined
# frame and compare review_reason strings against adjudicate().
# --------------------------------------------------------------------------- #


def _reference_review_reason(ref, joined_in: pd.DataFrame, spec) -> pd.Series:
    """Run the reference flag sequence on a copy; return review_reason.

    This is an independent transcription of the flag block, operating on the same
    column names, so equality of the resulting review_reason proves adjudicate()
    matches it.
    """
    joined = joined_in.copy()
    joined["gemini_success"] = joined["max_stage_v2"].astype(int).eq(7)
    joined["v1_success"] = joined["original_outcome"].astype(str).eq("success")
    if "max_stage_achieved" in joined.columns:
        joined["stage_delta_vs_v1"] = joined["max_stage_v2"].astype(int) - joined[
            "max_stage_achieved"
        ].astype(int)
    else:
        joined["stage_delta_vs_v1"] = pd.NA

    joined["review_reason"] = ""
    ref._flag(
        joined,
        joined["marker_released"] & ~joined["gripper_reopened_at_end"].astype(bool),
        "release_contradicts_sensor",
    )
    if "final_insert_depth" in joined.columns:
        ref._flag(
            joined,
            joined["max_stage_v2"].astype(int).ge(6)
            & joined["final_insert_depth"].astype(str).ne("full"),
            "seated_contradicts_depth",
        )
        ref._flag(
            joined,
            joined["final_insert_depth"].astype(str).eq("full")
            & joined["max_stage_v2"].astype(int).lt(6),
            "depth_contradicts_stage",
        )
    has_v1_outcome = ~joined["original_outcome"].astype(str).isin(["", "nan", "None"])
    ref._flag(
        joined,
        has_v1_outcome & (joined["gemini_success"] != joined["v1_success"]),
        "success_endpoint_disagreement",
    )
    ref._flag(
        joined,
        joined["gemini_success"]
        & joined["final_state"].astype(str).ne("marker_fully_seated_released"),
        "stage7_final_state_mismatch",
    )
    ref._flag(joined, joined["sample_stage_disagreement"], "sample_stage_disagreement")
    ref._flag(joined, joined["model_requested_review"], "model_requested_review")
    ref._flag(joined, joined["gemini_confidence"].eq("low"), "low_confidence")
    if joined["stage_delta_vs_v1"].notna().any():
        ref._flag(joined, joined["stage_delta_vs_v1"].abs().ge(2), "large_stage_delta_vs_v1")
    return joined["review_reason"]


def _full_joined() -> pd.DataFrame:
    """A tiny joined frame exercising every flag, with all event columns present.

    Rows (one per episode), columns are the union of events-CSV + consensus
    columns the flag block reads.
    """
    return pd.DataFrame(
        [
            # 0: clean S7 full success, sensor + depth consistent -> NO flags
            dict(
                episode_index=0,
                policy_short="mulligan",
                original_outcome="success",
                max_stage_achieved=7,
                final_insert_depth="full",
                gripper_reopened_at_end=True,
                max_stage_v2=7,
                final_state="marker_fully_seated_released",
                marker_released=True,
                sample_stage_disagreement=False,
                model_requested_review=False,
                gemini_confidence="high",
            ),
            # 1: release claimed but jaws never reopened -> release_contradicts_sensor
            dict(
                episode_index=1,
                policy_short="mulligan",
                original_outcome="failure",
                max_stage_achieved=5,
                final_insert_depth="partial",
                gripper_reopened_at_end=False,
                max_stage_v2=5,
                final_state="marker_partially_in_holder_released",
                marker_released=True,
                sample_stage_disagreement=False,
                model_requested_review=False,
                gemini_confidence="medium",
            ),
            # 2: seated stage (S6) but depth not full -> seated_contradicts_depth
            dict(
                episode_index=2,
                policy_short="base",
                original_outcome="failure",
                max_stage_achieved=6,
                final_insert_depth="partial",
                gripper_reopened_at_end=False,
                max_stage_v2=6,
                final_state="marker_fully_seated_held",
                marker_released=False,
                sample_stage_disagreement=False,
                model_requested_review=False,
                gemini_confidence="high",
            ),
            # 3: depth full but stage < seated -> depth_contradicts_stage; also a
            # large stage delta (v1=3, v2=5 -> 2) -> large_stage_delta_vs_v1
            dict(
                episode_index=3,
                policy_short="base",
                original_outcome="failure",
                max_stage_achieved=3,
                final_insert_depth="full",
                gripper_reopened_at_end=True,
                max_stage_v2=5,
                final_state="marker_partially_in_holder_released",
                marker_released=True,
                sample_stage_disagreement=False,
                model_requested_review=False,
                gemini_confidence="high",
            ),
            # 4: S7 (gemini success) but v1 outcome failure -> endpoint disagreement;
            # final_state not success state -> stage7_final_state_mismatch
            dict(
                episode_index=4,
                policy_short="mulligan",
                original_outcome="failure",
                max_stage_achieved=7,
                final_insert_depth="full",
                gripper_reopened_at_end=True,
                max_stage_v2=7,
                final_state="marker_partially_in_holder_released",
                marker_released=True,
                sample_stage_disagreement=False,
                model_requested_review=False,
                gemini_confidence="high",
            ),
            # 5: disagreement + model review + low confidence triad
            dict(
                episode_index=5,
                policy_short="base",
                original_outcome="success",
                max_stage_achieved=7,
                final_insert_depth="full",
                gripper_reopened_at_end=True,
                max_stage_v2=7,
                final_state="marker_fully_seated_released",
                marker_released=True,
                sample_stage_disagreement=True,
                model_requested_review=True,
                gemini_confidence="low",
            ),
            # 6: empty original_outcome -> no endpoint flag despite v2 success/v1 mismatch
            dict(
                episode_index=6,
                policy_short="base",
                original_outcome="",
                max_stage_achieved=4,
                final_insert_depth="partial",
                gripper_reopened_at_end=True,
                max_stage_v2=4,
                final_state="marker_in_gripper_at_holder",
                marker_released=False,
                sample_stage_disagreement=False,
                model_requested_review=False,
                gemini_confidence="high",
            ),
        ]
    )


def test_adjudicate_expected_reasons(marker_spec):
    """Lock the exact review_reason content per row (documents the semantics)."""
    out = adjudicate(_full_joined(), marker_spec)
    reasons = dict(zip(out["episode_index"], out["review_reason"], strict=True))
    assert reasons[0] == ""
    assert reasons[1] == "release_contradicts_sensor;"
    assert reasons[2] == "seated_contradicts_depth;"
    assert "depth_contradicts_stage;" in reasons[3]
    assert "large_stage_delta_vs_v1;" in reasons[3]
    assert "success_endpoint_disagreement;" in reasons[4]
    assert "stage7_final_state_mismatch;" in reasons[4]
    assert reasons[5] == ("sample_stage_disagreement;model_requested_review;low_confidence;")
    # episode 6: blank original_outcome suppresses the endpoint flag
    assert "success_endpoint_disagreement" not in reasons[6]


# --------------------------------------------------------------------------- #
# Column-presence guards: leaner events CSVs skip the object-physics rules.
# This is where an unguarded release_contradicts_sensor rule would raise KeyError.
# --------------------------------------------------------------------------- #


def _lean_joined() -> pd.DataFrame:
    """A minimal events join with NO gripper/depth/seed columns at all."""
    return pd.DataFrame(
        [
            dict(
                episode_index=0,
                policy_short="mulligan",
                original_outcome="success",
                max_stage_v2=7,
                final_state="marker_fully_seated_released",
                marker_released=True,
                sample_stage_disagreement=False,
                model_requested_review=False,
                gemini_confidence="high",
            ),
            dict(
                episode_index=1,
                policy_short="mulligan",
                original_outcome="failure",
                max_stage_v2=7,
                final_state="marker_fully_seated_released",
                marker_released=True,
                sample_stage_disagreement=True,
                model_requested_review=False,
                gemini_confidence="low",
            ),
        ]
    )


def test_adjudicate_no_keyerror_without_gripper_column(marker_spec):
    """adjudicate() guards the gripper column and does not raise."""
    out = adjudicate(_lean_joined(), marker_spec)
    reasons = dict(zip(out["episode_index"], out["review_reason"], strict=True))
    # release_contradicts_sensor / depth rules skipped (no columns); structural
    # rules still fire (endpoint disagreement on ep1, disagreement+low-conf).
    assert "release_contradicts_sensor" not in reasons[0]
    assert "release_contradicts_sensor" not in reasons[1]
    assert reasons[0] == ""  # S7 + v1 success agree, high conf, no disagreement
    assert "success_endpoint_disagreement;" in reasons[1]
    assert "sample_stage_disagreement;" in reasons[1]
    assert "low_confidence;" in reasons[1]


def test_adjudicate_skips_large_delta_without_max_stage_achieved(marker_spec):
    """Without max_stage_achieved, stage_delta is NA and the delta flag is skipped."""
    out = adjudicate(_lean_joined(), marker_spec)
    assert out["stage_delta_vs_v1"].isna().all()
    assert not out["review_reason"].str.contains("large_stage_delta_vs_v1").any()


def test_failure_mode_disagreement_flag_and_confidence_tiebreak():
    # Three samples at the SAME stage but three different modes (routing ep68 pattern):
    # the consensus must flag the disagreement and NOT pick alphabetically when confidences differ.
    from mulligan.real.stage_labeling import consensus as C
    from mulligan.real.stage_specs import get_label_task_spec

    spec = get_label_task_spec("routing_d2")
    sf, ff = spec.stage_field, spec.failure_mode_field

    def _s(idx, mode, conf):
        p = {
            sf: 6,
            spec.final_state_field: "rope_in_first_clip_at_second_unseated",
            ff: mode,
            "confidence": conf,
            "needs_human_review": False,
            "notes": "x",
        }
        for f in spec.bool_fields:
            p[f] = f == "rope_grasped" or f == "first_clip_seated"
        for f in spec.time_fields:
            p[f] = None
        return {"episode_index": 68, "sample_idx": idx, "parsed": p}

    # "timeout_holding" is alphabetically last but has the highest confidence -> must win.
    samples = [
        _s(0, "regrasp_failed", "low"),
        _s(1, "stopped_after_first", "low"),
        _s(2, "timeout_holding", "high"),
    ]
    row = C._consensus_episode(samples, spec)
    assert row["sample_failure_mode_disagreement"] is True
    assert row[ff] == "timeout_holding"  # confidence tiebreak, not alphabetical
