#!/usr/bin/env python3
"""
Test suite for fast dataset loading and episode boundary handling.

All datasets MUST use the padded frame format (written by mulligan.sim.collect.teleop):
  - Each episode has T+1 frames: T real + 1 padded
  - Frames 0 to T-1: Real (s, a, r, done) with is_valid=1
  - Frame T: Padded frame with terminal observation s_T and is_valid=0

The padded frame exists solely to provide the correct next_state for the
final transition. This is validated at load time - datasets without the
padded format will raise an error.

The load_dataset_fast tests run the loader on tiny local LeRobot datasets
(tests/unit/sim_dataset.py): padded frames, mid-episode invalid frames, missing
padding, episode limits, repeated success rewards, reward shifts.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.unit.sim_dataset import ENV_STATE_DIM, STATE_DIM, build_sim_dataset


# =============================================================================
# load_dataset_fast on tiny local LeRobot datasets
# =============================================================================


def _state(ep, t, length=None):
    return {"observation.state": np.full(STATE_DIM, 100 * ep + t, np.float32)}


def _load(tmp_path, name="ds", **kw):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from mulligan.training.fast_dataset_loader import load_dataset_fast

    loader_keys = ("intervention_negative_reward", "reward_shift")
    loader_kw = {k: kw.pop(k) for k in loader_keys if k in kw}
    root = build_sim_dataset(tmp_path / name, **kw)
    return load_dataset_fast(LeRobotDataset(name, root=root), **loader_kw)


class TestLoadDatasetFast:
    def test_padded_frames_give_next_state_and_are_excluded(self, tmp_path):
        data = _load(tmp_path, episode_lengths=(5, 3), overrides=_state)
        # 5 + 3 valid transitions; the padded frame of each episode is dropped
        assert len(data["states"]) == 8
        assert data["states"][:, 0].tolist() == [0, 1, 2, 3, 4, 100, 101, 102]
        # next_state of the last real frame is the padded frame's terminal observation
        assert data["next_states"][:, 0].tolist() == [1, 2, 3, 4, 5, 101, 102, 103]
        assert data["states"].shape[1] == STATE_DIM + ENV_STATE_DIM
        assert data["dones"].tolist() == [0, 0, 0, 0, 1, 0, 0, 1]
        assert data["episode_indices"].tolist() == [0] * 5 + [1] * 3
        assert data["intervention"].tolist() == [0] * 8  # optional column absent

    def test_mid_episode_invalid_frames_are_excluded(self, tmp_path):
        def overrides(ep, t, length):
            out = _state(ep, t)
            if t == 2:
                out["is_valid"] = np.array([0], np.int64)
            return out

        data = _load(tmp_path, episode_lengths=(5,), overrides=overrides)
        assert data["states"][:, 0].tolist() == [0, 1, 3, 4]

    @pytest.mark.parametrize("unpadded", [(0, 1), (1,)])
    def test_missing_padded_frame_raises(self, tmp_path, unpadded):
        def overrides(ep, t, length):
            if ep in unpadded and t == length:
                return {"is_valid": np.array([1], np.int64)}
            return {}

        with pytest.raises(ValueError, match=f"Episode {unpadded[0]} does not end with is_valid=0"):
            _load(tmp_path, episode_lengths=(4, 4), overrides=overrides)

    def test_missing_required_column_raises(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from mulligan.training.fast_dataset_loader import load_dataset_fast

        root = build_sim_dataset(tmp_path / "ds", episode_lengths=(3,))
        hf = LeRobotDataset("ds", root=root).hf_dataset.remove_columns(["success"])
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset_fast(SimpleNamespace(hf_dataset=hf))

    def test_repeated_success_rewards_are_collapsed(self, tmp_path):
        def overrides(ep, t, length):
            # reward 1 on the last three real frames, done only on the last
            return {"reward": np.array([1.0 if length - 3 <= t < length else 0.0], np.float32)}

        data = _load(tmp_path, episode_lengths=(6,), overrides=overrides)
        assert data["rewards"].tolist() == [0, 0, 0, 1, 0, 0]
        assert data["dones"].tolist() == [0, 0, 0, 1, 0, 1]


class TestRewardShift:
    """intervention_negative_reward applies to intervention frames, then reward_shift to all."""

    INTERVENTION = {"intervention": {"dtype": "int64", "shape": (1,), "names": None}}

    def test_intervention_penalty_then_reward_shift_compose(self, tmp_path):
        def overrides(ep, t, length):
            return {"intervention": np.array([int(t == 1)], np.int64)}

        kw = dict(episode_lengths=(3,), extra_features=self.INTERVENTION, overrides=overrides)
        base = _load(tmp_path, "base", **kw)
        assert base["rewards"].tolist() == [0.0, 0.0, 1.0]
        assert base["intervention"].tolist() == [0, 1, 0]
        shifted = _load(
            tmp_path, "shifted", intervention_negative_reward=-2.0, reward_shift=-1.0, **kw
        )
        assert shifted["rewards"].tolist() == [-1.0, -3.0, 0.0]

    def test_reward_shift_zero_is_a_strict_noop(self, tmp_path):
        a = _load(tmp_path, "a", episode_lengths=(4,))
        b = _load(tmp_path, "b", episode_lengths=(4,), reward_shift=0.0)
        assert torch.equal(a["rewards"], b["rewards"])


# =============================================================================
# prepare_chunked_data: dones → masks contract
# =============================================================================
#
# These tests pin down how `prepare_chunked_data` translates `dones` into
# `masks`. Masks are baked from the dones at call time, and downstream training
# reads `batch["masks"]` (not `dones`) for TD bootstrapping, so train.py applies
# the `done_on_intervention_and_failure` dones modification before the chunking
# call. The tests lock in that contract.


class TestPrepareChunkedDataDonesMasks:
    """Pin down how prepare_chunked_data translates dones into masks."""

    @staticmethod
    def _build_data(dones_pattern):
        """Construct a minimal `data` dict for prepare_chunked_data.

        One episode with len(dones_pattern) frames. State/action/reward/
        next_state are all simple sentinels — only `dones` matters for masks.
        """
        import torch

        n = len(dones_pattern)
        state_dim, action_dim = 2, 2
        return {
            "states": torch.arange(n * state_dim, dtype=torch.float32).reshape(n, state_dim),
            "actions": torch.zeros((n, action_dim), dtype=torch.float32),
            "rewards": torch.zeros(n, dtype=torch.float32),
            "next_states": torch.zeros((n, state_dim), dtype=torch.float32),
            "dones": torch.tensor(dones_pattern, dtype=torch.float32),
            "episode_indices": torch.zeros(n, dtype=torch.int64),
            "dataset_indices": torch.zeros(n, dtype=torch.int64),
        }

    def test_chunk_size_1_masks_are_one_minus_dones(self):
        """For chunk_size=1, masks == 1 - dones (per-frame)."""
        import torch

        from mulligan.training.fast_dataset_loader import prepare_chunked_data

        data = self._build_data([0, 0, 0, 0, 1])
        result = prepare_chunked_data(data, chunk_size=1, gamma=0.99)
        expected = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0]])
        torch.testing.assert_close(result["masks"], expected)

    def test_chunk_size_3_mask_is_zero_when_any_done_in_window(self):
        """For chunk_size>1, mask[t] is 0 if any done within [t, t+chunk_size) is 1.

        Episode of 6 frames with done=1 at index 4 (the natural terminal):
          frame 0: window [0,1,2] → no done → mask=1
          frame 1: window [1,2,3] → no done → mask=1
          frame 2: window [2,3,4] → done at 4 → mask=0
          frame 3: window [3,4,5] → done at 4 → mask=0
          frame 4: window [4,5]   → done at 4 → mask=0   (truncated, n_valid=2)
          frame 5: window [5]     → no done   → mask=1   (truncated, n_valid=1)
        """
        import torch

        from mulligan.training.fast_dataset_loader import prepare_chunked_data

        data = self._build_data([0, 0, 0, 0, 1, 0])
        result = prepare_chunked_data(data, chunk_size=3, gamma=0.99)
        expected = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [1.0]])
        torch.testing.assert_close(result["masks"], expected)

    def test_pre_call_dones_modification_propagates_to_masks(self):
        """Modifying `data['dones']` before prepare_chunked_data changes masks.

        Callers that need masks to reflect a custom done-flag pattern must
        modify dones before calling.
        """
        import torch

        from mulligan.training.fast_dataset_loader import prepare_chunked_data

        data = self._build_data([0, 0, 0, 0, 0, 0])  # no dones initially
        # Mark frame 2 as done before chunking (e.g. intervention-terminal)
        data["dones"][2] = 1.0
        result = prepare_chunked_data(data, chunk_size=3, gamma=0.99)
        # With chunk_size=3, the done at frame 2 zeros the masks at frames 0, 1, 2
        # (the first window not containing it is frame 3, [3,4,5]).
        expected = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
        torch.testing.assert_close(result["masks"], expected)

    def test_post_call_dones_modification_does_not_change_masks(self):
        """Modifying `data['dones']` after prepare_chunked_data leaves masks frozen.

        The masks below are unchanged from the all-zeros-dones case (all 1s),
        even though dones[2] is now 1.

        Note for chunk_size=1: the function returns
        ``(1.0 - data['dones']).unsqueeze(-1)``. This is a fresh tensor
        produced by a tensor op (not a view), so post-call dones writes
        do not propagate to masks regardless of chunk_size.
        """
        import torch

        from mulligan.training.fast_dataset_loader import prepare_chunked_data

        # chunk_size=3 case
        data = self._build_data([0, 0, 0, 0, 0, 0])
        result = prepare_chunked_data(data, chunk_size=3, gamma=0.99)
        masks_before = result["masks"].clone()
        # Caller now tries to mark a frame as done — too late, masks already baked
        data["dones"][2] = 1.0
        torch.testing.assert_close(result["masks"], masks_before)
        # Sanity: masks are all 1 (no done was visible at chunk time)
        torch.testing.assert_close(result["masks"], torch.ones((6, 1)))

        # chunk_size=1 case (separate code path inside prepare_chunked_data)
        data = self._build_data([0, 0, 0, 0, 0, 0])
        result = prepare_chunked_data(data, chunk_size=1, gamma=0.99)
        masks_before = result["masks"].clone()
        data["dones"][2] = 1.0
        torch.testing.assert_close(result["masks"], masks_before)
        torch.testing.assert_close(result["masks"], torch.ones((6, 1)))


class TestPrepareChunkedDataRewards:
    """Chunk rewards stop at the first terminal inside the chunk."""

    @staticmethod
    def _data(rewards, dones, episodes):
        n = len(rewards)
        return {
            "states": torch.arange(n * 2, dtype=torch.float32).reshape(n, 2),
            "actions": torch.zeros((n, 2)),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "next_states": torch.zeros((n, 2)),
            "dones": torch.tensor(dones, dtype=torch.float32),
            "episode_indices": torch.tensor(episodes, dtype=torch.int64),
            "dataset_indices": torch.zeros(n, dtype=torch.int64),
        }

    def test_rewards_after_an_in_chunk_done_are_not_summed(self):
        from mulligan.training.fast_dataset_loader import prepare_chunked_data

        # done at frame 1 (e.g. done_on_intervention_and_failure), -1 rewards after it
        data = self._data([0, -1, -1, -1, -1, -1], [0, 1, 0, 0, 0, 0], [0] * 6)
        out = prepare_chunked_data(data, chunk_size=4, gamma=0.5)
        # anchor 0: r0 + 0.5 * r1, nothing after the done
        assert out["rewards"][0].item() == -0.5
        assert out["rewards"][1].item() == -1.0
        assert out["masks"][0].item() == 0.0
        # anchor 2 has no done in its window: all four rewards
        assert out["rewards"][2].item() == -(1 + 0.5 + 0.25 + 0.125)

    def test_terminal_only_at_episode_end_is_unchanged(self):
        """Every paper dataset has its only done on the last valid frame; there the
        reward is the full in-episode sum, bit for bit."""
        from mulligan.training.fast_dataset_loader import prepare_chunked_data

        gen = torch.Generator().manual_seed(0)
        lengths = [3, 9, 17, 1, 12]
        episodes = [e for e, n in enumerate(lengths) for _ in range(n)]
        rewards = torch.randn(len(episodes), generator=gen).tolist()
        dones = [float(i == n - 1 and e % 2 == 0) for e, n in enumerate(lengths) for i in range(n)]
        data = self._data(rewards, dones, episodes)
        out = prepare_chunked_data(data, chunk_size=8, gamma=0.99)

        powers = torch.tensor([0.99**i for i in range(8)], dtype=torch.float32)
        r = data["rewards"]
        start = 0
        for n in lengths:
            for t in range(n):
                k = min(8, n - t)
                expected = (r[start + t : start + t + k] * powers[:k]).sum()
                assert torch.equal(out["rewards"][start + t, 0], expected)
            start += n


# =============================================================================
# Run tests manually
# =============================================================================


if __name__ == "__main__":
    print("Running fast dataset loader tests...\n")
    pytest.main([__file__, "-v"])
