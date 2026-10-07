"""Integration-style check of valid-prefix filtering across BOTH real trainers.

Unlike ``test_valid_prefix_training_indices`` (which pins each helper in
isolation), this drives the *actual* boundary + index-construction entry points
that the DP and critic trainers call, over a tiny synthetic
two-sub-dataset MultiLeRobotDataset that mixes the two hard cases:

  * ``sub_a`` — a soft-truncated SUCCESS episode whose invalid suffix (5 frames)
    is longer than any trainer's ``drop_n_last`` (2 for DP, ``k`` for IQL).
  * ``sub_b`` — a TIMEOUT episode with ``done==0`` throughout and a soft-truncated
    suffix (3 frames), the case where an un-clamped backward RTG pass would leak
    living-cost shaped reward from the junk suffix into valid-frame targets.

Asserts the end-to-end invariants that matter for the overnight runs:
  1. No DP anchor lands in an invalid suffix, AND no supervised action timestep of
     any anchor's forward chunk (``clamp_soft_truncated_anchor_ends``) does either —
     LeRobot masks only ``action_is_pad`` (beyond the RAW episode end), so the anchor
     range itself must retreat the full forward action extent on soft-truncated
     episodes. Same for IQL train/holdout indices via ``drop_n_last == k``.
  2. The IQL k-step reward/action chunk AND its k-offset next-state read stay
     entirely within the valid prefix (``drop_n_last == k`` fully protects the
     chunk horizon — the strong "chunk never crosses the valid end" guarantee).
  3. Every sampled index maps to a FINITE MC return; suffix rows are NaN.
  4. A sim/legacy sub-dataset WITHOUT an ``is_valid`` column is identical to
     the raw-boundary path.
"""

from __future__ import annotations

import torch
import pytest
from lerobot.datasets.sampler import EpisodeAwareSampler

from mulligan.real.train.critic import (
    apply_intervention_reward_shaping,
    audit_iql_anchor_supervision,
    canonicalize_post_terminal_steps,
    discounted_mc_returns_by_dataset,
    iql_episode_frame_indices,
)
from mulligan.data.transforms import (
    clamp_soft_truncated_anchor_ends,
    compute_multidataset_valid_boundaries,
)


class _FakeHF:
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


def _build_multidataset():
    # sub_a: soft-truncated SUCCESS. 15 stored frames, valid prefix 10 (outcome at
    # local frame 9); frames 10..14 are invalid suffix (junk + terminal pad).
    sub_a = _FakeSub(
        {
            "episode_index": [0] * 15,
            "is_valid": [1] * 10 + [0] * 5,
            "done": [0] * 9 + [1] * 6,
            "reward": [0.0] * 9 + [1.0] * 6,
        }
    )
    # sub_b: TIMEOUT, done==0 throughout. 11 stored frames, valid prefix 8;
    # frames 8..10 are invalid suffix.
    sub_b = _FakeSub(
        {
            "episode_index": [0] * 11,
            "is_valid": [1] * 8 + [0] * 3,
            "done": [0] * 11,
            "reward": [0.0] * 11,
        }
    )
    return sub_a, sub_b


def test_integration_dp_and_iql_valid_prefix_end_to_end():
    sub_a, sub_b = _build_multidataset()
    subs = [sub_a, sub_b]
    repo_ids = ["synthetic/sub_a", "synthetic/sub_b"]

    from_idx, valid_to, raw_to, n_excluded = compute_multidataset_valid_boundaries(subs)
    assert from_idx == [0, 15]
    assert valid_to == [10, 23]  # 0+10 ; 15+8
    assert raw_to == [15, 26]
    assert n_excluded == 8  # 5 (sub_a) + 3 (sub_b)

    # Per-episode (from, valid_to, last_valid_global_index, is_valid_column)
    episodes = [
        (0, 10, sub_a.hf_dataset["is_valid"], 0),
        (15, 23, sub_b.hf_dataset["is_valid"], 15),
    ]

    # ---- (1) DP anchors: valid boundaries + forward-chunk overrun clamp ----
    dp_drop = 2
    # Forward action extent (max action_delta_indices) sized to these tiny episodes;
    # the real-recipe geometry (drop 2 / max offset 11) is pinned in
    # test_valid_prefix_training_indices.
    dp_max_fwd = 4
    dp_to, n_overrun = clamp_soft_truncated_anchor_ends(
        from_idx,
        valid_to,
        raw_to,
        drop_n_last=dp_drop,
        max_forward_action_offset=dp_max_fwd,
    )
    # Both episodes are soft-truncated (junk suffix > terminal pad): each loses
    # max_fwd - drop = 2 tail anchors relative to a plain valid_to - drop rule.
    assert n_overrun == 4
    dp_sampler = EpisodeAwareSampler(
        dataset_from_indices=from_idx,
        dataset_to_indices=dp_to,
        drop_n_last_frames=dp_drop,
        shuffle=False,
    )
    dp_anchors = sorted(dp_sampler.indices)
    # ep0: [0, 10-4) ; ep1: [15, 23-4)
    assert dp_anchors == list(range(0, 6)) + list(range(15, 19))
    for a in dp_anchors:
        gstart, vend, isvalid, gbase = next(e for e in episodes if e[0] <= a < e[1])
        # No anchor lands in an invalid suffix.
        assert isvalid[a - gbase] == 1
        # The FULL supervised action chunk stays within the valid prefix.
        assert a + dp_max_fwd <= vend - 1
        assert all(isvalid[a - gbase + d] == 1 for d in range(dp_max_fwd + 1))

    # ---- (2) IQL train + holdout indices via the outcome-aware constructor ----
    k = 3
    done_values = sub_a.hf_dataset["done"] + sub_b.hf_dataset["done"]
    iql_train = iql_episode_frame_indices(
        from_idx, valid_to, done_values, episode_set={0, 1}, horizon=k
    )
    # Terminal ep0 retains its valid done anchor; timeout ep1 retains the final
    # complete [20,21,22] window whose successor is first-invalid frame 23.
    assert iql_train == list(range(0, 10)) + list(range(15, 21))
    for a in iql_train:
        gstart, vend, isvalid, gbase = next(e for e in episodes if e[0] <= a < e[1])
        assert isvalid[a - gbase] == 1
        if not any(done_values[a : min(a + k, vend)]):
            # Nonterminal anchors keep a complete valid action/reward window.
            assert a + (k - 1) <= vend - 1
        # Terminal anchors may use the stored terminal tail/pad at later offsets;
        # their return masks bootstrap immediately at the current done row.

    # ---- (3) MC returns: finite on every sampled index, NaN on the suffix ----
    returns = discounted_mc_returns_by_dataset(
        subs,
        repo_ids,
        gamma=0.9,
        reward_shift=-0.1,  # nonzero living cost: would leak from suffix if un-clamped
        intervention_negative_reward=None,
        intervention_values_by_dataset=[
            torch.zeros(len(sub_a), dtype=torch.long),
            torch.zeros(len(sub_b), dtype=torch.long),
        ],
    )
    ret_a, ret_b = returns
    # Valid prefixes are finite; invalid suffixes are NaN (skipped, never anchored).
    assert torch.isfinite(ret_a[:10]).all() and torch.isnan(ret_a[10:]).all()
    assert torch.isfinite(ret_b[:8]).all() and torch.isnan(ret_b[8:]).all()
    # Timeout suffix did NOT leak: last valid timeout return is exactly the single
    # living-cost step (-0.1), not accumulated over the junk suffix.
    assert float(ret_b[7]) == torch.tensor(-0.1).item()

    # Every DP and IQL anchor (mapped to its sub-local index) resolves to a finite
    # MC target — i.e. no sampling path can fetch a NaN row.
    for a in dp_anchors + iql_train:
        if a < 15:
            assert torch.isfinite(ret_a[a])
        else:
            assert torch.isfinite(ret_b[a - 15])


def test_integration_sim_legacy_without_is_valid_matches_raw():
    # Sim/legacy sub-dataset carries no is_valid column: must be byte-identical to
    # the pre-existing raw-boundary path, including sampler anchors.
    sub = _FakeSub({"episode_index": [0, 0, 0, 0, 1, 1, 1]})
    raw_from, raw_to = [0, 4], [4, 7]
    v_from, v_to, v_raw, n_excluded = compute_multidataset_valid_boundaries([sub])
    assert v_from == raw_from
    assert v_to == raw_to == v_raw
    assert n_excluded == 0
    for drop in (0, 1, 2):
        raw_anchors = sorted(
            EpisodeAwareSampler(
                dataset_from_indices=raw_from,
                dataset_to_indices=raw_to,
                drop_n_last_frames=drop,
                shuffle=False,
            ).indices
        )
        v_anchors = sorted(
            EpisodeAwareSampler(
                dataset_from_indices=v_from,
                dataset_to_indices=v_to,
                drop_n_last_frames=drop,
                shuffle=False,
            ).indices
        )
        assert raw_anchors == v_anchors


def test_repeated_done_tail_is_retained_for_iql_boundaries_and_returns():
    sub = _FakeSub(
        {
            "episode_index": [0] * 7,
            "is_valid": [1, 1, 1, 1, 1, 1, 0],
            "done": [0, 0, 1, 1, 1, 1, 1],
            "reward": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    # DP/default semantics still stop at first done.
    from_idx, dp_to, raw_to, n_excluded = compute_multidataset_valid_boundaries([sub])
    assert from_idx == [0]
    assert dp_to == [3]
    assert raw_to == [7]
    assert n_excluded == 4
    # IQL uses is_valid alone for current-state eligibility.
    _, iql_to, _, iql_excluded = compute_multidataset_valid_boundaries(
        [sub], stop_at_first_done=False
    )
    assert iql_to == [6]
    assert iql_excluded == 1
    returns = discounted_mc_returns_by_dataset(
        [sub],
        ["synthetic/repeated-terminal"],
        gamma=0.9,
        reward_shift=0.0,
        intervention_negative_reward=None,
        intervention_values_by_dataset=[torch.zeros(len(sub), dtype=torch.long)],
    )[0]
    assert returns[:6].tolist() == pytest.approx([1.71, 1.9, 1.0, 1.0, 1.0, 1.0])
    assert torch.isnan(returns[6:]).all()


def test_iql_supervision_audit_fails_when_success_reward_tail_is_corrupt():
    with pytest.raises(ValueError, match="terminal reward tail must equal success=1"):
        audit_iql_anchor_supervision(
            anchor_indices=[0, 1, 2],
            global_episode_indices=torch.tensor([0, 0, 0]),
            global_dataset_indices=torch.tensor([0, 0, 0]),
            from_indices=[0],
            to_indices=[3],
            reward_values=[0.0, 1.0, 0.0],
            done_values=[0, 1, 1],
            success_values=[1, 1, 1],
            repo_ids=["synthetic/corrupt"],
            horizon=2,
        )


def test_iql_supervision_audit_allows_nonterminal_subtask_reward():
    rows = audit_iql_anchor_supervision(
        anchor_indices=[0, 1],
        global_episode_indices=torch.tensor([0, 0, 0]),
        global_dataset_indices=torch.tensor([0, 0, 0]),
        from_indices=[0],
        to_indices=[3],
        reward_values=[0.0, 1.0, 0.0],
        done_values=[0, 0, 0],
        success_values=[0, 0, 0],
        repo_ids=["synthetic/graded-timeout"],
        horizon=2,
    )
    assert rows[0]["reward_windows"] == 2
    assert rows[0]["timeout_anchors"] == 2


def test_post_terminal_chunk_steps_are_canonicalized():
    batch = {
        "action": torch.tensor([[[1.0], [2.0], [999.0]]]),
        "reward": torch.tensor([[0.0, 1.0, -5.0]]),
        # The invalid suffix may return to done=False; canonicalization exists
        # to overwrite it, while valid-prefix continuity is checked at startup.
        "done": torch.tensor([[False, True, False]]),
    }
    canonicalize_post_terminal_steps(batch)
    assert batch["action"].squeeze(-1).tolist() == [[1.0, 2.0, 2.0]]
    assert batch["reward"].tolist() == [[0.0, 1.0, 1.0]]
    assert batch["done"].tolist() == [[False, True, True]]


def test_intervention_gather_clamps_at_loaded_table_end():
    batch = {
        "index": torch.tensor([2]),
        "reward": torch.zeros(1, 3),
    }
    apply_intervention_reward_shaping(
        batch,
        intervention_by_frame=torch.tensor([0, 0, 1]),
        intervention_by_dataset=None,
        intervention_negative_reward=-1.0,
        reward_shift=0.0,
        horizon=3,
    )
    assert batch["reward"].tolist() == [[-1.0, -1.0, -1.0]]
