"""Guards for mulligan.real.stage_labeling.events.

Unit-tests the gripper close/release detection and the event assembly with
stubbed frame I/O.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import mulligan.real.stage_specs as sl
from mulligan.real.stage_labeling import events

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "marker_d2"


def _battery() -> list[list[float]]:
    return [
        [0, 0, 0.8, 0.8, 0.8, 0.0],  # close then clean release near end
        [0, 0, 0.8, 0.8, 0.8, 0.8],  # held to end, no release
        [0, 0, 0.1, 0.15],  # never closes
        [0, 0, 0.8, 0.85, 0.82, 0.8, 0.6, 0.4, 0.3],  # release-in-progress at end
        [0, 0, 0, 0, 0, 0.9],  # late close, last frame
        [0, 0.8, 0.0, 0, 0.8, 0.8, 0.0],  # multi-close, last-below release
        [0.9, 0.9, 0.9],  # closed from the start, never opens
        [0, 0, 0.8, 0.85, 0.83, 0.82, 0.81, 0.45],  # plateau then single drop at end
    ]


def test_release_params_shallow_partial_open_recover():
    """A shallow partial-open-then-reclose off the hold plateau (the earlier fixed-peg
    Nut task's release: gripper opens just enough to drop a seated nut, then closes on air,
    ending the trace below the plateau) is MISSED by marker's strict defaults
    (final >= 0.5 or < 0.3 below plateau) but caught by square's relaxed params."""
    # hold plateau ~0.63, brief dip to 0.59, recover to 0.64, end at 0.57 (ep29-like)
    g = pd.Series([0.0, 0.0, 0.5, 0.63, 0.63, 0.63, 0.63, 0.59, 0.59, 0.63, 0.64, 0.57])
    assert events.release_after_hold(g, 2, 0.2) is None  # marker defaults (0.5, 0.3) miss it
    assert events.release_after_hold(g, 2, 0.2, final_abs_max=0.6, plateau_margin=0.05) is not None
    # a genuinely held trace (ends flat at the plateau) is NOT a release either way
    held = pd.Series([0.0, 0.0, 0.5, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63])
    assert events.release_after_hold(held, 2, 0.2, final_abs_max=0.6, plateau_margin=0.05) is None


def test_derive_gripper_events_threshold_and_fps():
    g = pd.Series([0, 0, 0.8, 0.8, 0.8, 0.0])
    out = events.derive_gripper_events(g, 0.2, 15.0)
    assert out["episode_length"] == 6
    assert out["gripper_hold_frame"] == 2
    assert out["gripper_hold_time_s"] == round(2 / 15.0, 2)
    assert out["gripper_release_frame"] == 5
    assert out["gripper_reopened_at_end"] is True
    # fps scales the times; threshold gates the close
    out30 = events.derive_gripper_events(g, 0.2, 30.0)
    assert out30["gripper_hold_time_s"] == round(2 / 30.0, 2)
    never = events.derive_gripper_events(pd.Series([0.0, 0.1, 0.15]), 0.2, 15.0)
    assert never["gripper_hold_frame"] is None and never["gripper_reopened_at_end"] is False


def test_two_close_trace_is_single_cycle_first_close_only():
    """DOCUMENTED LIMITATION for routing_d2 (multi-grasp episodes): the minimal
    events derivation is SINGLE-CYCLE. Given a rope trace with two grasp cycles
    (close, open, re-close, final open), the hold is pinned to the FIRST close and
    the release collapses to the TERMINAL reopen; the intermediate re-grasp
    (open@5, reclose@8) is invisible. This is fine for the labeler's cap physics
    (a jaw close DID happen; the terminal settle IS a reopen) but means the events
    CSV does not represent routing's regrasp — the VLM prompt/labels carry that.
    """
    # close@2, open@5, re-close@8, final open@12 (episode ends at frame 13).
    g = pd.Series([0.0, 0.0, 0.85, 0.85, 0.85, 0.0, 0.0, 0.0, 0.85, 0.85, 0.85, 0.85, 0.0, 0.0])
    out = events.derive_gripper_events(g, 0.2, 15.0)
    assert out["gripper_hold_frame"] == 2  # FIRST cycle's close, not the re-close@8
    assert out["gripper_release_frame"] == 13  # terminal reopen, not the mid-episode open@5
    assert out["gripper_reopened_at_end"] is True
    # first_crossing/release_after_hold see exactly one close + one (terminal) release.
    assert events.first_crossing(g, 0.2) == 2
    assert events.release_after_hold(g, 2, 0.2) == 13


def _stub_frames(monkeypatch, per_episode: dict[int, list[float]]):
    spec = sl.get_label_task_spec(MARKER)
    col = spec.gripper_state_column
    rows = []
    for ep, series in per_episode.items():
        for fi, v in enumerate(series):
            rows.append({"episode_index": ep, "frame_index": fi, col: v})
    frames = pd.DataFrame(rows)
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: frames)
    return spec


def test_load_frame_gripper_reads_parquet_with_and_without_is_valid(monkeypatch, tmp_path):
    """Pins the real frame-parquet read path (column subset + is_valid retry),
    which the other events tests bypass by stubbing load_frame_gripper itself."""
    spec = sl.get_label_task_spec(MARKER)
    col = spec.gripper_state_column
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            col: [0.0, 0.8],
            "is_valid": [1, 1],
            "extra_column": ["a", "b"],  # must be dropped by the column subset
        }
    ).to_parquet(tmp_path / "with_valid.parquet")
    pd.DataFrame(
        {
            "episode_index": [1, 1],
            "frame_index": [0, 1],
            col: [0.1, 0.9],
            "extra_column": ["c", "d"],
        }
    ).to_parquet(tmp_path / "without_valid.parquet")
    monkeypatch.setattr(
        events,
        "frame_parquet_rel_paths",
        lambda spec: ["with_valid.parquet", "without_valid.parquet"],
    )
    monkeypatch.setattr(events, "repo_file", lambda spec, rel_path: tmp_path / rel_path)

    frames = events.load_frame_gripper(spec)

    assert list(frames.columns) == ["episode_index", "frame_index", col, "is_valid"]
    assert frames.index.tolist() == [0, 1, 2, 3]  # ignore_index concat
    assert list(frames["episode_index"]) == [0, 0, 1, 1]
    assert list(frames["frame_index"]) == [0, 1, 0, 1]
    assert list(frames[col]) == [0.0, 0.8, 0.1, 0.9]
    assert list(frames["is_valid"].iloc[:2]) == [1, 1]
    # the parquet without is_valid falls back to the 3-column read -> NaN after concat
    assert frames["is_valid"].iloc[2:].isna().all()


def test_build_minimal_events_columns_and_values(monkeypatch):
    spec = _stub_frames(monkeypatch, {0: [0, 0, 0.8, 0.8, 0.0], 1: [0, 0, 0.1]})
    df = events.build_minimal_events(spec, policy_steps=events.FULL_RECORDED_EPISODE)
    assert list(df["episode_index"]) == [0, 1]
    row0 = df[df["episode_index"] == 0].iloc[0]
    assert row0["gripper_hold_frame"] == 2 and bool(row0["gripper_reopened_at_end"]) is True
    row1 = df[df["episode_index"] == 1].iloc[0]
    assert pd.isna(row1["gripper_hold_frame"]) and bool(row1["gripper_reopened_at_end"]) is False
    # mandatory labeler columns + blank summarizer context are present even when
    # the dataset has no results.json.
    for c in ("episode_index", "policy_short", "original_outcome", "episode_length",
              "gripper_hold_time_s", "gripper_release_time_s", "gripper_reopened_at_end"):  # fmt: skip
        assert c in df.columns
    assert set(df["policy_short"]) == {""}
    assert set(df["original_outcome"]) == {""}


def test_build_minimal_events_uses_is_valid_prefix(monkeypatch):
    spec = sl.get_label_task_spec(MARKER)
    col = spec.gripper_state_column
    frames = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            col: [0.0, 0.8, 0.8, 0.0],
            "is_valid": [1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: frames)

    df = events.build_minimal_events(spec, policy_steps=events.FULL_RECORDED_EPISODE)
    row = df.iloc[0]
    assert row["episode_length"] == 2
    assert row["num_steps"] == 2
    assert row["gripper_hold_frame"] == 1
    # The invalid tail contains the reopen, but it must not affect stage events.
    assert pd.isna(row["gripper_release_frame"])
    assert bool(row["gripper_reopened_at_end"]) is False


def test_build_minimal_events_uses_rollout_num_steps_cap(monkeypatch):
    spec = sl.get_label_task_spec(MARKER)
    col = spec.gripper_state_column
    frames = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            col: [0.0, 0.8, 0.8, 0.0],
            "is_valid": [1, 1, 1, 1],
        }
    )
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: frames)

    df = events.build_minimal_events(
        spec,
        rollout_meta={
            0: {"policy_short": "mulligan", "original_outcome": "failure", "num_steps": 2}
        },
        policy_steps={0: 2},
    )
    row = df.iloc[0]
    assert row["episode_length"] == 2
    assert row["num_steps"] == 2
    assert row["gripper_hold_frame"] == 1
    # The reconciled outcome ended before the later open, so this must not be a release.
    assert pd.isna(row["gripper_release_frame"])
    assert bool(row["gripper_reopened_at_end"]) is False


def test_build_minimal_events_rejects_rollout_num_steps_past_valid_prefix(monkeypatch):
    spec = sl.get_label_task_spec(MARKER)
    col = spec.gripper_state_column
    frames = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            col: [0.0, 0.8, 0.8],
            "is_valid": [1, 1, 0],
        }
    )
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: frames)
    with pytest.raises(RuntimeError, match="exceeds dataset is_valid prefix"):
        events.build_minimal_events(
            spec,
            rollout_meta={
                0: {"policy_short": "mulligan", "original_outcome": "failure", "num_steps": 3}
            },
            policy_steps={0: 3},
        )


def test_build_minimal_events_rejects_non_prefix_is_valid(monkeypatch):
    spec = sl.get_label_task_spec(MARKER)
    col = spec.gripper_state_column
    frames = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            col: [0.0, 0.8, 0.8],
            "is_valid": [1, 0, 1],
        }
    )
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: frames)
    with pytest.raises(RuntimeError, match="is_valid is not a valid prefix"):
        events.build_minimal_events(spec, policy_steps=events.FULL_RECORDED_EPISODE)


def test_build_minimal_events_merges_rollout_meta(monkeypatch):
    spec = _stub_frames(monkeypatch, {0: [0, 0, 0.8, 0.0], 1: [0, 0.9, 0.0]})
    meta = {
        0: {"policy_short": "mulligan_sobol", "original_outcome": "failure"},
        1: {"policy_short": "baseline_uniform", "original_outcome": "success"},
    }
    df = events.build_minimal_events(
        spec, rollout_meta=meta, policy_steps=events.FULL_RECORDED_EPISODE
    )
    assert set(df["policy_short"]) == {"mulligan_sobol", "baseline_uniform"}
    assert df[df["episode_index"] == 1].iloc[0]["original_outcome"] == "success"


def test_build_minimal_events_fail_loud_on_missing_meta(monkeypatch):
    spec = _stub_frames(monkeypatch, {0: [0, 0.8, 0.0], 1: [0, 0.9, 0.0]})
    with pytest.raises(KeyError, match="missing episode 1"):
        events.build_minimal_events(
            spec,
            rollout_meta={0: {"policy_short": "x"}},
            policy_steps=events.FULL_RECORDED_EPISODE,
        )


# --------------------------------------------------------------------------- #
# The policy-phase boundary contract (`policy_steps`).
#
# Regression guard: stage-events CSVs derived over the WHOLE recording would let the
# operator's physical-reset jaw re-open — up to tens of seconds past the policy — land in
# `gripper_reopened_at_end`, which is the VLM labeler's "the robot released" signal.
# The boundary must be impossible to omit, and impossible to omit SILENTLY.
# --------------------------------------------------------------------------- #


def _reset_tail_frames(spec):
    """One episode: policy grips and holds for 4 frames, operator reset opens the jaws."""
    col = spec.gripper_state_column
    return pd.DataFrame(
        {
            "episode_index": [0] * 7,
            "frame_index": list(range(7)),
            #        <-- policy phase (4) -->   <-- operator reset -->
            col: [0.0, 0.85, 0.85, 0.85, 0.85, 0.4, 0.0],
            "is_valid": [1] * 7,
        }
    )


def test_build_minimal_events_requires_an_explicit_policy_boundary(monkeypatch):
    """Omitting the boundary is a TypeError, not a silent full-recording derivation."""
    spec = sl.get_label_task_spec(MARKER)
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: _reset_tail_frames(spec))
    with pytest.raises(TypeError, match="policy_steps"):
        events.build_minimal_events(spec)  # type: ignore[call-arg]


def test_policy_steps_clips_the_operator_reset_tail(monkeypatch):
    """THE regression: the reset re-open must not be read as a policy release."""
    spec = sl.get_label_task_spec(MARKER)
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: _reset_tail_frames(spec))

    clipped = events.build_minimal_events(spec, policy_steps={0: 4}).iloc[0]
    assert clipped["episode_length"] == 4
    assert clipped["num_steps"] == 4
    assert clipped["gripper_hold_frame"] == 1
    assert pd.isna(clipped["gripper_release_frame"])
    assert bool(clipped["gripper_reopened_at_end"]) is False

    # ... and the un-clipped derivation is exactly the bug that shipped.
    full = events.build_minimal_events(spec, policy_steps=events.FULL_RECORDED_EPISODE).iloc[0]
    assert full["episode_length"] == 7
    assert bool(full["gripper_reopened_at_end"]) is True


def test_policy_steps_mapping_fails_loud_on_a_missing_episode(monkeypatch):
    spec = _stub_frames(monkeypatch, {0: [0, 0.8, 0.0], 1: [0, 0.9, 0.0]})
    with pytest.raises(KeyError, match="policy_steps missing episode 1"):
        events.build_minimal_events(spec, policy_steps={0: 2})


def test_policy_steps_rejects_a_boundary_past_the_valid_prefix(monkeypatch):
    spec = sl.get_label_task_spec(MARKER)
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: _reset_tail_frames(spec))
    with pytest.raises(RuntimeError, match="exceeds dataset is_valid prefix"):
        events.build_minimal_events(spec, policy_steps={0: 8})
    with pytest.raises(RuntimeError, match="policy_steps must be positive"):
        events.build_minimal_events(spec, policy_steps={0: 0})


def test_policy_steps_must_agree_with_rollout_meta_num_steps(monkeypatch):
    """Two independent records of where the policy stopped; disagreement is loud."""
    spec = sl.get_label_task_spec(MARKER)
    monkeypatch.setattr(events, "load_frame_gripper", lambda spec: _reset_tail_frames(spec))
    with pytest.raises(RuntimeError, match="disagrees with policy_steps"):
        events.build_minimal_events(spec, rollout_meta={0: {"num_steps": 5}}, policy_steps={0: 4})
    agreeing = events.build_minimal_events(
        spec, rollout_meta={0: {"num_steps": 4}}, policy_steps={0: 4}
    ).iloc[0]
    assert agreeing["num_steps"] == 4


def test_policy_steps_rejects_a_bare_string_that_is_not_the_sentinel(monkeypatch):
    spec = _stub_frames(monkeypatch, {0: [0, 0.8, 0.0]})
    with pytest.raises(TypeError, match="FULL_RECORDED_EPISODE"):
        events.build_minimal_events(spec, policy_steps="whole thing")


def test_policy_steps_from_rollout_meta_fails_loud_without_num_steps():
    assert events.policy_steps_from_rollout_meta({0: {"num_steps": 7}}) == {0: 7}
    with pytest.raises(KeyError, match="has no num_steps"):
        events.policy_steps_from_rollout_meta({0: {"policy_short": "mulligan"}})


def test_load_rollout_meta_uses_canonical_outcome_results(monkeypatch):
    from types import SimpleNamespace

    from mulligan.real.eval.outcome_results import FrameOutcome
    from mulligan.real.stage_labeling import prepare_events

    calls = []

    def fake_load_hf_json(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "summary": [
                {
                    "policy_id": 0,
                    "name": "baseline_policy",
                    "num_rounds": 1,
                    "successes": 0,
                    "failures": 1,
                }
            ],
            "rollouts": [
                {
                    "episode_index": 83,
                    "policy_id": 0,
                    "outcome": "timeout",
                    "num_steps": 3,
                    "nut_x": 0.01,
                    "nut_y": 0.02,
                    "nut_yaw": 0.03,
                    "peg_x": 0.24,
                    "peg_y": -0.01,
                }
            ],
        }

    def fake_load_outcome_edit_record(*args, **kwargs):
        return {
            "changed_episodes": {
                "83": {"new_outcome": "success", "outcome_frame": 2, "soft_truncate": False}
            }
        }

    monkeypatch.setattr(prepare_events, "load_hf_json", fake_load_hf_json)
    monkeypatch.setattr(prepare_events, "load_outcome_edit_record", fake_load_outcome_edit_record)
    monkeypatch.setattr(
        prepare_events,
        "load_frame_outcomes_from_hf",
        lambda *args, **kwargs: {83: FrameOutcome("success", 3)},
    )
    task_spec = SimpleNamespace(dataset_repo_id="org/example", lifecycle_task="square_d2")

    meta = prepare_events.load_rollout_meta(
        task_spec,
        {"baseline_policy": "baseline"},
        outcome_overrides_filename=".outcome_edit_progress.json",
        require_outcome_overrides=True,
    )

    assert calls == [(("org/example", "results.json"), {"revision": "main"})]
    assert meta[83]["policy_short"] == "baseline"
    assert meta[83]["original_outcome"] == "success"
    assert meta[83]["peg_y"] == -0.01
    # the policy-phase boundary the events builder clips to
    assert meta[83]["num_steps"] == 3
    assert events.policy_steps_from_rollout_meta(meta) == {83: 3}


def test_load_rollout_meta_required_override_missing_fails(monkeypatch):
    from types import SimpleNamespace

    from huggingface_hub.errors import EntryNotFoundError

    from mulligan.real.stage_labeling import prepare_events

    monkeypatch.setattr(
        prepare_events,
        "load_hf_json",
        lambda *args, **kwargs: {
            "summary": [{"policy_id": 0, "name": "baseline_policy"}],
            "rollouts": [],
        },
    )

    def missing_override(*args, **kwargs):
        raise EntryNotFoundError("missing .outcome_edit_progress.json")

    monkeypatch.setattr(prepare_events, "load_outcome_edit_record", missing_override)
    task_spec = SimpleNamespace(dataset_repo_id="org/example", lifecycle_task="square_d2")

    with pytest.raises(EntryNotFoundError, match="outcome_edit_progress"):
        prepare_events.load_rollout_meta(
            task_spec,
            None,
            outcome_overrides_filename=".outcome_edit_progress.json",
            require_outcome_overrides=True,
        )
