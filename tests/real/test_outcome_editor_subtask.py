"""Subtask-reward extension of the outcome editor.

Covers ``apply_outcome_edits`` reward-spike writes + validation, the progress
record round-trip, and a cheap Monte-Carlo-return integration proving a done=0
single-frame spike lifts pre-spike return targets by ``gamma**dist``. All
fixtures are synthetic; no HF/network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mulligan.tools import outcome_review as outcome_editor
from mulligan.tools.outcome_review import (
    apply_outcome_edits,
    detect_existing_outcome_frame,
    episode_fully_processed,
    load_progress,
    mark_episode_changed,
    save_progress,
)


def _episode_df(
    *,
    ep_idx: int = 0,
    n_frames: int = 6,
    is_valid: list[int] | None = None,
) -> pd.DataFrame:
    """A single fresh (all-zero reward/done) episode with a terminal pad frame."""
    if is_valid is None:
        # last frame is invalid terminal padding, everything else valid
        is_valid = [1] * (n_frames - 1) + [0]
    assert len(is_valid) == n_frames
    return pd.DataFrame(
        [
            {
                "episode_index": ep_idx,
                "frame_index": f,
                "reward": 0.0,
                "done": 0,
                "success": 0,
                "is_valid": is_valid[f],
            }
            for f in range(n_frames)
        ]
    )


def _progress(ep_idx: int, **entry) -> dict:
    return {"changed_episodes": {str(ep_idx): entry}, "skipped_episodes": []}


def test_apply_subtask_spike_sets_reward_only_at_marked_frame() -> None:
    df = _episode_df(n_frames=6)
    progress = _progress(
        0,
        new_outcome="success",
        outcome_frame=4,
        soft_truncate=False,
        subtask_frames=[2],
    )

    assert apply_outcome_edits(df, progress, subtask_marks=1) is True

    by_frame = df.set_index("frame_index")
    # reward: spike at the marked subtask frame, 0 elsewhere pre-terminal.
    assert by_frame.loc[0, "reward"] == 0.0
    assert by_frame.loc[1, "reward"] == 0.0
    assert by_frame.loc[2, "reward"] == 1.0  # subtask spike
    assert by_frame.loc[3, "reward"] == 0.0
    # terminal plateau (outcome frame onward) unchanged for a success episode.
    assert by_frame.loc[4, "reward"] == 1.0
    assert by_frame.loc[5, "reward"] == 1.0
    # done / is_valid untouched at the spike (done=0, is_valid=1).
    assert by_frame.loc[2, "done"] == 0
    assert by_frame.loc[2, "is_valid"] == 1
    # success is the episode-level constant.
    assert (df["success"] == 1).all()


def _collection_style_success_df(*, ep_idx: int = 0, n_frames: int = 7, onset: int = 4):
    """An episode as the collection stack writes it: success=1 everywhere,
    reward=1/done=1 plateau from the operator-marked onset, is_valid prefix."""
    df = _episode_df(ep_idx=ep_idx, n_frames=n_frames)
    df.loc[df["frame_index"] >= onset, ["reward", "done"]] = [1.0, 1]
    df["success"] = 1
    return df


def test_detect_existing_outcome_frame_finds_plateau_onset() -> None:
    df = _collection_style_success_df(onset=4)
    assert detect_existing_outcome_frame(df) == 4
    # A timeout-style episode (no done anywhere) has no onset to prefill.
    assert detect_existing_outcome_frame(_episode_df()) is None


def test_add_mark_only_preserves_existing_terminal_labels() -> None:
    """The routing R0 flow: prefilled onset + one subtask mark must rewrite the
    existing collection-time terminal plateau to IDENTICAL values, changing only
    the spike frame's reward."""
    df = _collection_style_success_df(n_frames=7, onset=4)
    original = df.copy(deep=True)
    onset = detect_existing_outcome_frame(df)
    assert onset == 4
    progress = _progress(
        0,
        new_outcome="success",
        outcome_frame=onset,
        soft_truncate=False,
        subtask_frames=[2],
    )

    apply_outcome_edits(df, progress, subtask_marks=1)

    # Everything except the spike frame's reward is identical to the
    # pre-edit collection labels.
    expected = original.copy(deep=True)
    expected.loc[expected["frame_index"] == 2, "reward"] = 1.0
    pd.testing.assert_frame_equal(df, expected, check_dtype=False)


def test_n0_golden_matches_expected_single_mark_behavior() -> None:
    """With no subtask machinery the writes match the pre-subtask editor exactly."""
    df = _episode_df(n_frames=6)
    progress = _progress(0, new_outcome="success", outcome_frame=4, soft_truncate=False)

    apply_outcome_edits(df, progress, subtask_marks=0)

    expected = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3, 4, 5],
            "reward": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "done": [0, 0, 0, 0, 1, 1],
            "success": [1, 1, 1, 1, 1, 1],
            "is_valid": [1, 1, 1, 1, 1, 0],
        }
    )
    pd.testing.assert_frame_equal(df, expected, check_dtype=False)


def test_n0_golden_unaffected_by_default_apply_call() -> None:
    """Default apply_outcome_edits (no subtask_marks kwarg) equals subtask_marks=0."""
    df_default = _episode_df(n_frames=6)
    df_explicit = _episode_df(n_frames=6)
    apply_outcome_edits(
        df_default, _progress(0, new_outcome="failure", outcome_frame=3, soft_truncate=True)
    )
    apply_outcome_edits(
        df_explicit,
        _progress(0, new_outcome="failure", outcome_frame=3, soft_truncate=True),
        subtask_marks=0,
    )
    pd.testing.assert_frame_equal(df_default, df_explicit)


def test_validation_rejects_subtask_at_or_after_outcome_frame() -> None:
    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[4]
    )
    with pytest.raises(ValueError, match="strictly before"):
        apply_outcome_edits(df, progress, subtask_marks=1)


def test_timeout_subtask_on_last_valid_frame_survives_padding_normalization() -> None:
    """A timeout boundary on terminal padding normalizes onto the last valid frame.

    The subtask may have completed on that exact final frame: timeout has no
    terminal done transition, so reward=1/done=0 remains unambiguous.
    """
    df = _episode_df(n_frames=6)
    progress = _progress(
        0,
        new_outcome="timeout",
        outcome_frame=5,
        soft_truncate=False,
        subtask_frames=[4],
    )

    assert apply_outcome_edits(df, progress, subtask_marks=1) is True

    by_frame = df.set_index("frame_index")
    assert by_frame.loc[4, "reward"] == 1.0
    assert by_frame.loc[4, "done"] == 0
    assert by_frame.loc[4, "is_valid"] == 1
    assert by_frame.loc[5, "reward"] == 0.0
    assert by_frame.loc[5, "done"] == 0
    assert by_frame.loc[5, "is_valid"] == 0
    assert outcome_editor._episode_outcomes_by_index(  # noqa: SLF001
        df,
        subtask_frames_by_episode={0: [4]},
    ) == {0: "timeout"}
    # Regression: the review queue classified this state as an error,
    # so any later session on the dataset crashed in get_filtered_episodes.
    assert outcome_editor.detect_episode_outcome(df) == "timeout"


def test_reedit_restores_previously_invalidated_frame_and_writes_spike() -> None:
    """Re-edit scenario: a prior soft-truncate left a mid-episode is_valid=0 frame.

    Validity is judged POST-edit: apply_outcome_edits re-validates every frame
    before the terminal pad, so re-marking the previously-invalid frame as a
    subtask spike must succeed and land on a re-validated (is_valid=1) frame.
    (The outcome frame itself must be valid pre-edit — normalize_outcome_frame
    enforces that separately — so only the subtask frame is invalid here.)
    """
    df = _episode_df(n_frames=6, is_valid=[1, 1, 0, 1, 1, 0])  # frame 2 invalid pre-edit
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[2]
    )

    assert apply_outcome_edits(df, progress, subtask_marks=1) is True

    by_frame = df.set_index("frame_index")
    assert by_frame.loc[2, "reward"] == 1.0
    assert by_frame.loc[2, "done"] == 0
    assert by_frame.loc[2, "is_valid"] == 1  # restored by the edit
    assert by_frame.loc[4, "reward"] == 1.0  # terminal plateau intact
    assert by_frame.loc[5, "is_valid"] == 0  # terminal pad stays invalid


def test_apply_tolerates_record_without_marks_in_subtask_mode() -> None:
    """A progress file written by a session without subtask marks (all episodes
    changed, no subtask_frames key) must re-apply cleanly under --subtask-marks N — terminal edits applied,
    no spike, no crash. The exactly-N contract binds only to records that carry
    marks."""
    df = _collection_style_success_df(n_frames=7, onset=4)
    original = df.copy(deep=True)
    # Record shape as found on the hub for sessions without marks: no subtask_frames key.
    progress = _progress(0, new_outcome="success", outcome_frame=4, soft_truncate=False)

    apply_outcome_edits(df, progress, subtask_marks=1)

    # Terminal labels rewritten to identical values; no spike anywhere.
    pd.testing.assert_frame_equal(df, original, check_dtype=False)


def test_subtask_mark_count_error_outcome_aware() -> None:
    # Success must carry EXACTLY N; failure/timeout may carry 0..N. N=0 disables.
    err = outcome_editor.subtask_mark_count_error
    assert err("success", 1, 1) is None
    assert err("success", 0, 1) is not None  # a success that reached the sub-goal must mark it
    assert err("success", 2, 1) is not None
    assert err("failure", 0, 1) is None  # never seated a clip -> legitimate 0-mark failure
    assert err("failure", 1, 1) is None  # seated then fumbled -> one spike, still a failure
    assert err("timeout", 0, 1) is None
    assert err("failure", 2, 1) is not None  # over budget
    assert err("success", 0, 0) is None  # N=0 disables entirely


def test_apply_rejects_success_reviewed_with_wrong_mark_count() -> None:
    df = _episode_df(n_frames=8)
    progress = _progress(
        0, new_outcome="success", outcome_frame=6, soft_truncate=False, subtask_frames=[2, 4]
    )
    with pytest.raises(ValueError, match="must carry exactly 1 subtask mark"):
        apply_outcome_edits(df, progress, subtask_marks=1)


def test_apply_zero_mark_failure_review_writes_no_spike() -> None:
    """The routing_d2 bug: a failure that never reached the sub-goal (0 clips seated)
    is a legitimate 0-mark review — key present but empty. It must apply terminal
    edits, write no spike, and count as processed (not re-queue forever)."""
    df = _episode_df(n_frames=8)
    # Reviewed failure with an explicit EMPTY subtask_frames key.
    progress = _progress(
        0, new_outcome="failure", outcome_frame=6, soft_truncate=False, subtask_frames=[]
    )
    apply_outcome_edits(df, progress, subtask_marks=1)
    ep = df[df["episode_index"] == 0]
    assert (ep["reward"] == 0.0).all(), "a 0-mark failure must have no reward spike anywhere"
    assert episode_fully_processed(progress["changed_episodes"]["0"], 1) is True


def test_apply_failure_with_one_mark_writes_spike() -> None:
    """Seated the first clip then fumbled -> failure terminal + one mid-episode spike."""
    df = _episode_df(n_frames=8)
    progress = _progress(
        0, new_outcome="failure", outcome_frame=6, soft_truncate=False, subtask_frames=[3]
    )
    apply_outcome_edits(df, progress, subtask_marks=1)
    ep = df[df["episode_index"] == 0].set_index("frame_index")
    assert ep.loc[3, "reward"] == 1.0 and ep.loc[3, "done"] == 0
    assert ep.loc[6, "reward"] == 0.0 and ep.loc[6, "done"] == 1  # failure terminal, no reward


def test_apply_rejects_marks_when_subtask_mode_off() -> None:
    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[2]
    )
    with pytest.raises(ValueError, match="--subtask-marks is 0"):
        apply_outcome_edits(df, progress, subtask_marks=0)


def test_episode_fully_processed_is_key_presence_in_subtask_mode() -> None:
    unmarked = {"new_outcome": "success", "outcome_frame": 313, "soft_truncate": False}
    marked = {**unmarked, "subtask_frames": [120]}
    reviewed_zero = {"new_outcome": "failure", "outcome_frame": 313, "subtask_frames": []}
    # Sessions without marks: any changed record is processed.
    assert episode_fully_processed(unmarked, 0) is True
    # Subtask mode: reviewed <=> subtask_frames KEY present (even empty). Unmarked re-enters.
    assert episode_fully_processed(unmarked, 1) is False
    assert episode_fully_processed(marked, 1) is True
    assert episode_fully_processed(reviewed_zero, 1) is True  # the 0-mark failure review
    # Never-changed episodes are always remaining.
    assert episode_fully_processed(None, 0) is False
    assert episode_fully_processed(None, 1) is False


def test_mark_episode_changed_stamps_empty_key_when_reviewed() -> None:
    # A 0-mark reviewed failure must persist an EMPTY subtask_frames key so it is
    # distinguishable from an unmarked record; N=0 sessions never stamp the key.
    progress = {"changed_episodes": {}, "skipped_episodes": []}
    mark_episode_changed(
        progress,
        5,
        new_outcome="failure",
        outcome_frame=10,
        soft_truncate=False,
        subtask_frames=[],
        subtask_reviewed=True,
    )
    assert progress["changed_episodes"]["5"]["subtask_frames"] == []
    progress2 = {"changed_episodes": {}, "skipped_episodes": []}
    mark_episode_changed(
        progress2,
        5,
        new_outcome="failure",
        outcome_frame=10,
        soft_truncate=False,
    )
    assert "subtask_frames" not in progress2["changed_episodes"]["5"]


def test_resolve_subtask_marks_from_task_config() -> None:
    # routing_d2 is the single source of truth: num_subtask_marks=1 on its RealTaskSpec.
    assert outcome_editor.resolve_subtask_marks({"routing_d2"}, None) == 1
    # The pen lines carry the default 0 (no subtask marks).
    assert outcome_editor.resolve_subtask_marks({"marker_d2"}, None) == 0
    assert outcome_editor.resolve_subtask_marks({"square_d2"}, None) == 0


def test_resolve_subtask_marks_unregistered_task_defaults_zero() -> None:
    # An unknown/sim/ad-hoc task name resolves to 0 (never crashes).
    assert outcome_editor.resolve_subtask_marks({"Not_A_Real_Task"}, None) == 0
    assert outcome_editor.resolve_subtask_marks(set(), None) == 0


def test_resolve_subtask_marks_override_wins() -> None:
    # An explicit CLI value overrides the task config in both directions.
    assert outcome_editor.resolve_subtask_marks({"routing_d2"}, 0) == 0
    assert outcome_editor.resolve_subtask_marks({"marker_d2"}, 2) == 2
    assert outcome_editor.resolve_subtask_marks(set(), 3) == 3


def test_resolve_subtask_marks_mixed_tasks_fail_loud() -> None:
    # A dataset mixing a 1-mark and a 0-mark task has no well-defined count -> raise.
    with pytest.raises(ValueError, match="differing num_subtask_marks"):
        outcome_editor.resolve_subtask_marks({"routing_d2", "marker_d2"}, None)
    # ...unless an explicit override disambiguates.
    assert outcome_editor.resolve_subtask_marks({"routing_d2", "marker_d2"}, 1) == 1


def test_validation_rejects_wrong_mark_count() -> None:
    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[2]
    )
    with pytest.raises(ValueError, match="must carry exactly 2 subtask mark"):
        apply_outcome_edits(df, progress, subtask_marks=2)


def test_validation_rejects_subtask_at_or_after_outcome_with_soft_truncate() -> None:
    # With soft truncation enabled, a mark at/after the outcome frame (here 4
    # with outcome_frame=3) violates the strictly-before rule, which is also
    # what keeps every spike inside the soft-truncate valid prefix
    # (0..outcome_frame).
    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=3, soft_truncate=True, subtask_frames=[4]
    )
    with pytest.raises(ValueError, match="strictly before"):
        apply_outcome_edits(df, progress, subtask_marks=1)


def test_validation_rejects_subtask_on_terminal_padding_frame() -> None:
    # Frame 5 is the invalid terminal padding row; it is always at/after the
    # outcome frame so the strictly-before rule rejects it.
    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[5]
    )
    with pytest.raises(ValueError, match="strictly before"):
        apply_outcome_edits(df, progress, subtask_marks=1)


def test_validation_rejects_nonexistent_subtask_frame() -> None:
    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[99]
    )
    with pytest.raises(ValueError, match="not found"):
        apply_outcome_edits(df, progress, subtask_marks=1)


def test_duplicate_subtask_frames_do_not_double_count() -> None:
    # A hand-edited duplicate like [2, 2] dedups to one mark: it must not
    # satisfy N=2, and it must apply cleanly as a single mark under N=1.
    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[2, 2]
    )
    with pytest.raises(ValueError, match="must carry exactly 2 subtask mark"):
        apply_outcome_edits(df, progress, subtask_marks=2)

    df = _episode_df(n_frames=6)
    progress = _progress(
        0, new_outcome="success", outcome_frame=4, soft_truncate=False, subtask_frames=[2, 2]
    )
    apply_outcome_edits(df, progress, subtask_marks=1)
    assert df.set_index("frame_index").loc[2, "reward"] == 1.0


def test_progress_record_round_trips_subtask_frames(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    progress = {"changed_episodes": {}, "skipped_episodes": []}
    mark_episode_changed(
        progress,
        7,
        new_outcome="success",
        outcome_frame=10,
        soft_truncate=False,
        subtask_frames=[4, 2, 4],
    )
    save_progress(root, progress)

    reloaded = load_progress(root)
    entry = reloaded["changed_episodes"]["7"]
    assert entry["subtask_frames"] == [2, 4]  # sorted + deduplicated on write

    on_disk = json.loads((root / outcome_editor.PROGRESS_FILENAME).read_text())
    assert on_disk["changed_episodes"]["7"]["subtask_frames"] == [2, 4]


def test_progress_record_omits_key_when_no_subtask_marks(tmp_path: Path) -> None:
    """N=0 progress files stay byte-identical: no subtask_frames key is written."""
    root = tmp_path / "dataset"
    root.mkdir()
    progress = {"changed_episodes": {}, "skipped_episodes": []}
    mark_episode_changed(progress, 3, new_outcome="failure", outcome_frame=5, soft_truncate=True)
    entry = progress["changed_episodes"]["3"]
    assert "subtask_frames" not in entry
    # A record without the key reads as an empty subtask list.
    assert entry.get("subtask_frames", []) == []


class _FakeHF:
    def __init__(self, cols):
        self._cols = cols

    @property
    def column_names(self):
        return list(self._cols)

    def __getitem__(self, key):
        return self._cols[key]


class _FakeSub:
    def __init__(self, cols, n):
        self.hf_dataset = _FakeHF(cols)
        self._n = n

    def __len__(self):
        return self._n


def test_done0_subtask_spike_raises_pre_spike_mc_returns_by_gamma_dist() -> None:
    """A done=0 single-frame reward spike lifts earlier return-to-go targets.

    Exercises the real MC-return code path (no formula copied here). Episode
    reward [0,1,0,1] with done [F,F,F,T] and gamma=0.9:
      G3=1.0 (terminal), G2=0.9, G1=1+0.9*0.9=1.81, G0=0.9*1.81=1.629.
    Frame 0 (dist 1 before the spike) is lifted by gamma**1 = 0.9 vs the
    no-spike baseline (0.729 -> 1.629).
    """
    from mulligan.real.train.critic import discounted_mc_returns_by_dataset

    common = {
        "done": [False, False, False, True],
        "episode_index": [0, 0, 0, 0],
        "is_valid": [1, 1, 1, 1],
    }
    baseline = _FakeSub({"reward": [0.0, 0.0, 0.0, 1.0], **common}, 4)
    spiked = _FakeSub({"reward": [0.0, 1.0, 0.0, 1.0], **common}, 4)

    (base_returns,) = discounted_mc_returns_by_dataset(
        [baseline],
        ["r"],
        gamma=0.9,
        reward_shift=0.0,
        intervention_negative_reward=None,
        intervention_values_by_dataset=[],
    )
    (spiked_returns,) = discounted_mc_returns_by_dataset(
        [spiked],
        ["r"],
        gamma=0.9,
        reward_shift=0.0,
        intervention_negative_reward=None,
        intervention_values_by_dataset=[],
    )

    assert base_returns.tolist() == pytest.approx([0.729, 0.81, 0.9, 1.0])
    assert spiked_returns.tolist() == pytest.approx([1.629, 1.81, 0.9, 1.0])
    # Pre-spike frame 0 lifted by exactly gamma**dist (dist=1) * spike(1.0).
    assert (spiked_returns[0] - base_returns[0]).item() == pytest.approx(0.9)


def test_only_episodes_filter_and_fail_loud():
    """--episodes restricts to an explicit index set and fails loud on a missing one."""
    import pandas as pd
    import pytest

    from mulligan.tools.outcome_review import _parse_episode_indices, get_filtered_episodes

    df = pd.DataFrame(
        [
            {"episode_index": ep, "is_valid": 1, "reward": 0.0, "done": 0}
            for ep in (60, 62, 65, 67, 70)
        ]
    )
    assert get_filtered_episodes(df, "all", only_episodes=frozenset({60, 65, 67})) == [60, 65, 67]
    # combines with the outcome filter (all here are timeouts)
    assert get_filtered_episodes(df, "timeout", only_episodes=frozenset({65})) == [65]
    # absent index -> loud
    with pytest.raises(ValueError, match="not present in dataset"):
        get_filtered_episodes(df, "all", only_episodes=frozenset({60, 999}))
    # parser helper
    assert _parse_episode_indices("60,65,67") == frozenset({60, 65, 67})
    with pytest.raises(Exception):
        _parse_episode_indices("60,notanint")


def test_force_redo_reopens_finished_episodes():
    """--episodes (force_redo) re-includes already-confirmed episodes; default skips them."""
    from mulligan.tools.outcome_review import remaining_episodes_to_review

    # ep60 was confirmed with 0 marks (subtask_frames key present) -> normally done.
    changed = {"60": {"outcome": "timeout", "subtask_frames": []}}
    filtered = [60, 65, 67]
    # default: ep60 is skipped as fully processed (65/67 have no record -> still pending)
    assert remaining_episodes_to_review(filtered, changed, set(), 1) == [65, 67]
    # force_redo: all listed episodes re-open regardless of prior completion
    assert remaining_episodes_to_review(filtered, changed, set(), 1, force_redo=True) == [
        60,
        65,
        67,
    ]
    # force_redo even overrides an explicit skip
    assert remaining_episodes_to_review(filtered, changed, {65}, 1, force_redo=True) == [60, 65, 67]
    assert remaining_episodes_to_review(filtered, changed, {65}, 1) == [67]


def test_timeout_prefill_falls_back_to_last_valid_frame():
    """The timeout g->c fix: no done==1 plateau, so prefill uses the last valid frame."""
    from mulligan.tools.outcome_review import (
        detect_existing_outcome_frame,
        last_valid_frame_index,
    )

    df = _episode_df()  # all done=0 (timeout) with an is_valid=0 terminal pad frame
    assert detect_existing_outcome_frame(df) is None  # what left marked_frame None before
    expected = int(df[df["is_valid"] == 1]["frame_index"].max())
    assert last_valid_frame_index(df) == expected  # fallback -> a valid outcome frame, g->c works
