"""Valid-prefix (soft-truncation) filtering in the real trainers.

Outcome-edited real LeRobot episodes store a leading valid prefix (frames
``0..outcome`` inclusive) followed by an invalid suffix (post-outcome
retract/reset junk + the terminal padding frame), all marked ``is_valid==0``.
Training must exclude that suffix so DP anchors / RL transitions never land on
junk and action chunks never overrun into it, while sim datasets without
an ``is_valid`` column keep the plain ``drop_n_last`` behaviour.

These tests pin:
  (a) DP anchor indices (via ``EpisodeAwareSampler`` over the clamped boundaries)
      exclude the invalid suffix and count ``drop_n_last`` back from the valid end;
  (b) a dataset WITHOUT ``is_valid`` yields indices IDENTICAL to the raw path;
  (c) IQL train/holdout index construction excludes the suffix and the MC
      return-to-go pass preserves the terminal transition at the outcome frame
      while never leaking the invalid suffix into valid-frame targets.
"""

from __future__ import annotations

import pytest
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler

from mulligan.real.train.critic import (
    discounted_mc_returns_by_dataset,
    iql_episode_frame_indices,
)
from mulligan.data.transforms import (
    clamp_soft_truncated_anchor_ends,
    compute_multidataset_valid_boundaries,
    episode_anchor_exclusive_end,
)


class _FakeHF:
    """Minimal HF-dataset stand-in: column access + ``column_names``."""

    def __init__(self, columns: dict[str, list]) -> None:
        self._columns = {k: list(v) for k, v in columns.items()}

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __getitem__(self, key: str) -> list:
        return self._columns[key]


class _FakeSub:
    def __init__(self, columns: dict[str, list]) -> None:
        self.hf_dataset = _FakeHF(columns)
        self.features = set(columns)

    def __len__(self) -> int:
        return len(self.hf_dataset["episode_index"])


def _sampler_anchor_indices(from_indices, to_indices, *, drop_n_last: int) -> list[int]:
    sampler = EpisodeAwareSampler(
        dataset_from_indices=from_indices,
        dataset_to_indices=to_indices,
        drop_n_last_frames=drop_n_last,
        shuffle=False,
    )
    return sorted(sampler.indices)


# ---------------------------------------------------------------------------
# (a) DP anchors exclude the invalid suffix; drop_n_last counts from valid end
# ---------------------------------------------------------------------------


def test_dp_anchors_exclude_invalid_suffix_and_count_from_valid_end() -> None:
    # One episode, 8 stored frames. Valid prefix = first 5 (outcome at frame 4);
    # frames 5,6 are soft-truncated junk, frame 7 is the terminal pad -> is_valid=0.
    is_valid = [1, 1, 1, 1, 1, 0, 0, 0]
    sub = _FakeSub({"episode_index": [0] * 8, "is_valid": is_valid})

    from_idx, valid_to, raw_to, n_excluded = compute_multidataset_valid_boundaries([sub])
    assert from_idx == [0]
    assert valid_to == [5]  # ep_from + valid_prefix_length(=5)
    assert raw_to == [8]
    assert n_excluded == 3  # frames 5,6,7

    drop = 2
    anchors = _sampler_anchor_indices(from_idx, valid_to, drop_n_last=drop)
    # Rule: [ep_from, ep_from + max(0, L_valid - drop_n_last)) = [0, 5-2) = [0,3)
    assert anchors == [0, 1, 2]
    # No anchor ever lands on an invalid-suffix frame.
    assert all(is_valid[a] == 1 for a in anchors)


def test_dp_two_episodes_clamped_independently() -> None:
    # ep0 stored 4 frames, valid prefix 3; ep1 stored 6 frames, valid prefix 5.
    sub = _FakeSub(
        {
            "episode_index": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "is_valid": [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        }
    )
    from_idx, valid_to, _raw, _n = compute_multidataset_valid_boundaries([sub])
    assert from_idx == [0, 4]
    assert valid_to == [3, 9]  # ep0 prefix 3 -> 0+3; ep1 prefix 5 -> 4+5
    anchors = _sampler_anchor_indices(from_idx, valid_to, drop_n_last=2)
    # ep0: [0, 0+3-2) = [0,1); ep1: [4, 4+5-2) = [4,7)
    assert anchors == [0, 4, 5, 6]


# ---------------------------------------------------------------------------
# (a2) Forward action-chunk overrun: supervised timesteps never hit junk frames
# ---------------------------------------------------------------------------
#
# LeRobot's loss masks only action_is_pad (frames beyond the RAW episode end), never
# in-episode is_valid==0 frames. An anchor supervises action timesteps
# anchor + d for d in action_delta_indices, so on soft-truncated episodes the anchor
# range must additionally retreat max(action_delta_indices) from the valid end.


def _real_dp_recipe_geometry():
    """(drop_n_last, action_delta_indices) for the ACTUAL real-DP recipe.

    Reads the r18 recipe (``DPTrainingRecipe``: predict 12 / exec 6 / drop 2) and the
    trainer's single observation step through the REAL LeRobot
    ``DiffusionConfig``, so the pinned geometry tracks the code, not this test's
    assumptions.
    """
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    from mulligan.real.lifecycle.tasks import DPTrainingRecipe

    recipe = DPTrainingRecipe()
    cfg = DiffusionConfig(
        horizon=recipe.chunk_size,
        n_action_steps=recipe.n_action_steps,
        n_obs_steps=1,  # the real DP trainer's single observation step
        down_dims=recipe.down_dims,
    )
    return recipe.drop_n_last_frames, cfg.action_delta_indices


def test_dp_chunk_never_supervised_on_junk_under_real_recipe() -> None:
    drop, action_delta = _real_dp_recipe_geometry()
    assert drop == 2  # r18 recipe drop_n_last_frames
    max_fwd = max(action_delta)
    assert max_fwd == 11  # n_obs_steps=1, horizon=12 -> deltas 0..11

    # Soft-truncated episode: 40 stored frames, valid prefix 20 (outcome at frame 19),
    # frames 20..38 hover/timeout junk, frame 39 terminal pad.
    is_valid = [1] * 20 + [0] * 20
    sub = _FakeSub({"episode_index": [0] * 40, "is_valid": is_valid})
    from_idx, valid_to, raw_to, _n = compute_multidataset_valid_boundaries([sub])
    assert (from_idx, valid_to, raw_to) == ([0], [20], [40])

    # The plain rule (valid_to - drop) would supervise junk: its deepest anchor's chunk
    # overruns the valid prefix by max_fwd - drop frames.
    plain_deepest_anchor = valid_to[0] - drop - 1
    assert plain_deepest_anchor + max_fwd > valid_to[0] - 1

    sampler_to, n_excluded = clamp_soft_truncated_anchor_ends(
        from_idx, valid_to, raw_to, drop_n_last=drop, max_forward_action_offset=max_fwd
    )
    assert n_excluded == max_fwd - drop  # 9 tail anchors dropped on this episode
    anchors = _sampler_anchor_indices(from_idx, sampler_to, drop_n_last=drop)
    assert anchors == list(range(0, valid_to[0] - max_fwd))  # [0, 9)
    for a in anchors:
        for d in action_delta:
            t = a + d
            assert t <= valid_to[0] - 1, f"anchor {a} supervises junk timestep {t}"
            assert is_valid[t] == 1


def test_overrun_clamp_byte_identical_without_soft_truncation() -> None:
    # Pad-only episodes (the collection writer marks exactly one terminal pad frame
    # is_valid=0) and datasets without is_valid keep identical sampler to-indices
    # and anchor sets.
    drop, action_delta = _real_dp_recipe_geometry()
    max_fwd = max(action_delta)

    # Two pad-only episodes (10 + 8 stored frames) and one no-is_valid dataset.
    pad_only = _FakeSub(
        {
            "episode_index": [0] * 10 + [1] * 8,
            "is_valid": [1] * 9 + [0] + [1] * 7 + [0],
        }
    )
    no_is_valid = _FakeSub({"episode_index": [0] * 6})

    from_idx, valid_to, raw_to, _n = compute_multidataset_valid_boundaries([pad_only, no_is_valid])
    assert (from_idx, valid_to, raw_to) == ([0, 10, 18], [9, 17, 24], [10, 18, 24])

    sampler_to, n_excluded = clamp_soft_truncated_anchor_ends(
        from_idx, valid_to, raw_to, drop_n_last=drop, max_forward_action_offset=max_fwd
    )
    # Byte-identical: the clamp returns the valid_to list unchanged, removes nothing.
    assert sampler_to == valid_to
    assert n_excluded == 0
    # And the anchor index sets are exactly the unclamped ones
    # ([from, valid_to - drop) per episode with drop=2).
    anchors = _sampler_anchor_indices(from_idx, sampler_to, drop_n_last=drop)
    assert anchors == list(range(0, 7)) + list(range(10, 15)) + list(range(18, 22))
    assert anchors == _sampler_anchor_indices(from_idx, valid_to, drop_n_last=drop)


def test_episode_anchor_exclusive_end_rules() -> None:
    # Pad-only suffix (<=1): the plain drop_n_last rule.
    assert episode_anchor_exclusive_end(0, 9, 10, drop_n_last=2, max_forward_action_offset=11) == 7
    # No invalid frames at all (raw == valid): the plain rule.
    assert episode_anchor_exclusive_end(0, 10, 10, drop_n_last=2, max_forward_action_offset=11) == 8
    # Soft-truncated (junk suffix > 1): retreat by the full forward action extent.
    assert episode_anchor_exclusive_end(0, 20, 40, drop_n_last=2, max_forward_action_offset=11) == 9
    # Forward extent already covered by drop_n_last (e.g. ACT drop=chunk_size-1): unchanged.
    assert (
        episode_anchor_exclusive_end(0, 20, 40, drop_n_last=19, max_forward_action_offset=19) == 1
    )
    # Valid prefix shorter than the required retreat: clamps at ep_from (no anchors).
    assert episode_anchor_exclusive_end(5, 10, 30, drop_n_last=2, max_forward_action_offset=11) == 5
    # Non-zero episode start offsets translate.
    assert (
        episode_anchor_exclusive_end(100, 120, 140, drop_n_last=2, max_forward_action_offset=11)
        == 109
    )


# ---------------------------------------------------------------------------
# (b) No is_valid column -> identical to the pre-existing raw boundary path
# ---------------------------------------------------------------------------


def test_no_is_valid_column_is_identical_to_raw_boundaries() -> None:
    sub = _FakeSub({"episode_index": [0, 0, 0, 0, 1, 1, 1]})
    raw_from, raw_to = [0, 4], [4, 7]
    v_from, v_to, v_raw, n_excluded = compute_multidataset_valid_boundaries([sub])

    assert v_from == raw_from
    assert v_to == raw_to == v_raw
    assert n_excluded == 0
    for drop in (0, 1, 2):
        assert _sampler_anchor_indices(raw_from, raw_to, drop_n_last=drop) == (
            _sampler_anchor_indices(v_from, v_to, drop_n_last=drop)
        )


# ---------------------------------------------------------------------------
# (c) IQL: holdout index construction + MC return-to-go terminal semantics
# ---------------------------------------------------------------------------


def test_iql_episode_frame_indices_exclude_invalid_suffix() -> None:
    sub = _FakeSub(
        {
            "episode_index": [0] * 8,
            "is_valid": [1, 1, 1, 1, 1, 0, 0, 0],
            "done": [0] * 8,
        }
    )
    from_idx, valid_to, _raw, _n = compute_multidataset_valid_boundaries(
        [sub], stop_at_first_done=False
    )
    frames = iql_episode_frame_indices(
        from_idx,
        valid_to,
        sub.hf_dataset["done"],
        episode_set={0},
        horizon=2,
    )
    # Includes final complete window [3,4] and reads invalid frame 5 only as s'.
    assert frames == [0, 1, 2, 3]


def _mc_returns_single_sub(columns, *, gamma, reward_shift):
    sub = _FakeSub(columns)
    return discounted_mc_returns_by_dataset(
        [sub],
        ["synthetic/test"],
        gamma=gamma,
        reward_shift=reward_shift,
        intervention_negative_reward=None,
        intervention_values_by_dataset=[torch.zeros(len(sub), dtype=torch.long)],
    )[0]


def test_mc_returns_timeout_do_not_leak_invalid_suffix() -> None:
    # Timeout episode: done==0 throughout, reward 0. Soft-truncated suffix (frames
    # 3,4) is is_valid==0. With a nonzero living-cost shift, leaving the suffix in
    # the backward pass would leak its shaped reward into valid frames; truncation
    # must stop the RTG accumulation at the last valid frame (frame 2).
    gamma, shift = 0.9, -0.1
    returns = _mc_returns_single_sub(
        {
            "episode_index": [0] * 5,
            "reward": [0.0] * 5,
            "done": [0, 0, 0, 0, 0],
            "is_valid": [1, 1, 1, 0, 0],
        },
        gamma=gamma,
        reward_shift=shift,
    )
    # Backward over valid rows only: r2=-0.1, r1=-0.1+0.9*-0.1, r0=-0.1+0.9*r1
    exp2 = shift
    exp1 = shift + gamma * exp2
    exp0 = shift + gamma * exp1
    assert returns[:3].tolist() == pytest.approx([exp0, exp1, exp2], abs=1e-6)
    # Invalid suffix rows are never anchored -> intentionally NaN, not leaked values.
    assert bool(torch.isnan(returns[3])) and bool(torch.isnan(returns[4]))


def test_mc_returns_preserve_terminal_transition_at_outcome_frame() -> None:
    # Success episode: outcome at frame 2 (reward=1, done=1 from there); suffix
    # frames 3,4 are is_valid==0. The terminal reward at the last valid frame must
    # flow back into earlier valid frames (bootstrap reset by done at the outcome).
    gamma = 0.9
    returns = _mc_returns_single_sub(
        {
            "episode_index": [0] * 5,
            "reward": [0.0, 0.0, 1.0, 1.0, 1.0],
            "done": [0, 0, 1, 1, 1],
            "is_valid": [1, 1, 1, 0, 0],
        },
        gamma=gamma,
        reward_shift=0.0,
    )
    # r2 = 1.0 (terminal, done resets), r1 = 0.9*1.0, r0 = 0.9*r1
    assert returns[:3].tolist() == pytest.approx([gamma * gamma * 1.0, gamma * 1.0, 1.0], abs=1e-6)
    assert bool(torch.isnan(returns[3])) and bool(torch.isnan(returns[4]))


def test_mc_returns_without_is_valid_span_full_episode() -> None:
    # No is_valid column (sim): behaviour unchanged -> every row filled.
    gamma = 0.9
    returns = _mc_returns_single_sub(
        {
            "episode_index": [0] * 3,
            "reward": [0.0, 0.0, 1.0],
            "done": [0, 0, 1],
        },
        gamma=gamma,
        reward_shift=0.0,
    )
    assert torch.isfinite(returns).all()
    assert returns.tolist() == pytest.approx([gamma * gamma * 1.0, gamma * 1.0, 1.0], abs=1e-6)


def test_multi_subdataset_offsets_accumulate() -> None:
    # Two sub-datasets, each with its own is_valid pattern, verify global offsets
    # and per-sub column caching stay aligned.
    sub_a = _FakeSub({"episode_index": [0, 0, 0], "is_valid": [1, 1, 0]})
    sub_b = _FakeSub({"episode_index": [0, 0, 0, 0], "is_valid": [1, 1, 1, 0]})
    from_idx, valid_to, raw_to, n_excluded = compute_multidataset_valid_boundaries([sub_a, sub_b])
    assert from_idx == [0, 3]  # sub_b starts at global offset 3
    assert valid_to == [2, 6]  # a: 0+2 ; b: 3+3
    assert raw_to == [3, 7]
    assert n_excluded == 2  # one pad each
