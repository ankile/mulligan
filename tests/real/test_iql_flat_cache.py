"""Contracts for the horizon-independent IQL trajectory cache."""

import json

import pytest
import torch

import torch.nn as nn

from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.networks.vision_iql import VisionIQL
from mulligan.data.constants import DataSource
from mulligan.real.train.iql_flat_cache import (
    FLAT_CACHE_DYNAMIC_METADATA_KEYS,
    FlatEncodedTrajectoryCache,
    build_flat_cache_from_physical_frames,
    derive_flat_cache_rows,
    load_flat_encoded_cache,
    save_flat_encoded_cache,
)
from mulligan.real.train.critic import (
    build_multidataset_frame_metadata,
    build_prebuilt_eval_metadata_dataset,
    build_prebuilt_train_metadata_dataset,
    flat_cache_expected_metadata,
    iql_episode_frame_indices,
    intervention_values_for_subdataset,
    resolve_independent_target_samples,
)


def _cache(*, terminal: bool = False, views: int = 3) -> FlatEncodedTrajectoryCache:
    # Two contiguous episodes. Episode 0 has frames 0..9; episode 1 has 0..7.
    episode = torch.tensor([0] * 10 + [1] * 8)
    frame = torch.tensor(list(range(10)) + list(range(8)))
    n = len(frame)
    base = torch.arange(n, dtype=torch.float32)
    visual = torch.stack([base + 100 * view for view in range(views)], dim=1).unsqueeze(-1)
    done = torch.zeros(n, dtype=torch.long)
    if terminal:
        done[6:10] = 1
    is_valid = torch.ones(n, dtype=torch.bool)
    # Episode 1's final row is the boundary successor, not an anchor.
    is_valid[-1] = False
    anchor = is_valid.clone()
    anchor[4:10] = False
    anchor[-7:] = False
    return FlatEncodedTrajectoryCache(
        visual=visual,
        proprio=torch.stack((base, base + 0.5), dim=1),
        action=torch.stack((base, -base), dim=1),
        reward=base / 10,
        done=done,
        is_valid=is_valid,
        anchor_eligible=anchor,
        success=(episode == 0).long(),
        source=torch.zeros(n, dtype=torch.long),
        intervention=torch.zeros(n, dtype=torch.long),
        dataset_index=torch.zeros(n, dtype=torch.long),
        episode_index=episode,
        frame_index=frame,
    )


def test_flat_cache_rejects_duplicate_and_noncontiguous_frame_keys() -> None:
    cache = _cache()
    tensors = cache.tensor_payload()
    tensors["frame_index"][1] = 0
    with pytest.raises(ValueError, match="duplicate"):
        FlatEncodedTrajectoryCache.from_tensor_payload(tensors)

    tensors = cache.tensor_payload()
    tensors["frame_index"][1] = 20
    with pytest.raises(ValueError, match="noncontiguous"):
        FlatEncodedTrajectoryCache.from_tensor_payload(tensors)


def test_h6_transition_is_gathered_from_flat_rows() -> None:
    cache = _cache()
    batch = cache.build_transitions(
        torch.tensor([0, 10]), action_horizon=6, td_horizon=6, current_view=1
    )

    torch.testing.assert_close(batch["visual_curr"].flatten(), torch.tensor([100.0, 110.0]))
    torch.testing.assert_close(batch["visual_next"].flatten(), torch.tensor([106.0, 116.0]))
    torch.testing.assert_close(batch["action"][0, :, 0], torch.arange(6.0))
    torch.testing.assert_close(batch["reward"][0], torch.arange(6.0) / 10)
    assert batch["valid_steps"].tolist() == [6, 6]


def test_long_transition_truncates_at_timeout_boundary() -> None:
    cache = _cache()
    batch = cache.build_transitions(
        torch.tensor([10]), action_horizon=6, td_horizon=12, current_view=0
    )

    # Six valid action/reward steps and boundary successor at episode frame 7.
    assert batch["valid_steps"].item() == 7
    assert batch["frame_index"].item() == 0
    assert batch["visual_next"].item() == 17
    # Positions beyond the valid prefix repeat its final valid transition row.
    torch.testing.assert_close(batch["reward"][0, 7:].unique(), torch.tensor([1.6]))


def test_h96_transition_keeps_six_step_action_chunk() -> None:
    n = 110
    base = torch.arange(n, dtype=torch.float32)
    cache = FlatEncodedTrajectoryCache(
        visual=base[:, None],
        proprio=base[:, None],
        action=base[:, None],
        reward=base,
        done=torch.zeros(n, dtype=torch.long),
        is_valid=torch.ones(n, dtype=torch.bool),
        anchor_eligible=torch.tensor([True] + [False] * (n - 1)),
        success=torch.ones(n, dtype=torch.long),
        source=torch.zeros(n, dtype=torch.long),
        intervention=torch.zeros(n, dtype=torch.long),
        dataset_index=torch.zeros(n, dtype=torch.long),
        episode_index=torch.zeros(n, dtype=torch.long),
        frame_index=torch.arange(n),
    )

    batch = cache.build_transitions(
        torch.tensor([0]), action_horizon=6, td_horizon=96, current_view=0
    )

    assert batch["action"].shape == (1, 6, 1)
    assert batch["reward"].shape == (1, 96)
    assert batch["visual_next"].item() == 96
    assert batch["valid_steps"].item() == 96


def test_terminal_transition_repeats_first_terminal_and_bootstraps_from_it() -> None:
    cache = _cache(terminal=True)
    batch = cache.build_transitions(
        torch.tensor([2]), action_horizon=6, td_horizon=12, current_view=0
    )

    assert batch["valid_steps"].item() == 5  # offsets 0..4, with done at frame 6
    assert batch["done"][0].tolist() == [0, 0, 0, 0, 1] + [1] * 7
    assert batch["action"][0, :, 0].tolist() == [2, 3, 4, 5, 6, 6]
    assert batch["visual_next"].item() == 6


def test_chunk_bootstraps_are_chunk_aligned_and_boundary_truncated() -> None:
    cache = _cache()
    batch = cache.build_transitions(
        torch.tensor([10]),
        action_horizon=6,
        td_horizon=12,
        current_view=2,
        include_chunk_bootstraps=True,
    )

    assert batch["td_horizons"].tolist() == [6, 12]
    assert batch["td_horizon_valid"].tolist() == [[True, False]]
    # H6 uses frame 6. Invalid H12 is safely clamped to the boundary successor.
    assert batch["visual_bootstrap"].flatten().tolist() == [216.0, 217.0]
    assert batch["observation.bootstrap_state"][0, :, 0].tolist() == [16.0, 17.0]


def test_chunk_bootstraps_truncate_terminal_inside_first_chunk() -> None:
    cache = _cache(terminal=True)
    batch = cache.build_transitions(
        torch.tensor([2]),
        action_horizon=6,
        td_horizon=12,
        current_view=0,
        include_chunk_bootstraps=True,
    )

    assert batch["valid_steps"].item() == 5
    assert batch["td_horizon_valid"].tolist() == [[True, False]]
    # Both safe gather rows are terminal frame 6; only H6 participates.
    assert batch["visual_bootstrap"].flatten().tolist() == [6.0, 6.0]


def test_chunk_bootstraps_keep_terminal_anchor_in_bounds() -> None:
    tensors = _cache(terminal=True).tensor_payload()
    tensors["anchor_eligible"][6] = True
    cache = FlatEncodedTrajectoryCache.from_tensor_payload(tensors)
    batch = cache.build_transitions(
        torch.tensor([6]),
        action_horizon=6,
        td_horizon=12,
        current_view=0,
        include_chunk_bootstraps=True,
    )

    assert batch["valid_steps"].item() == 1
    assert batch["td_horizon_valid"].tolist() == [[True, False]]
    assert batch["visual_bootstrap"].flatten().tolist() == [6.0, 6.0]


def test_transition_rejects_cross_episode_and_ineligible_anchor() -> None:
    cache = _cache()
    with pytest.raises(ValueError, match="ineligible"):
        cache.build_transitions(torch.tensor([7]), action_horizon=6, td_horizon=6)

    tensors = cache.tensor_payload()
    tensors["anchor_eligible"][14] = True
    invalid = FlatEncodedTrajectoryCache.from_tensor_payload(tensors)
    with pytest.raises(ValueError, match="complete action chunk"):
        invalid.build_transitions(torch.tensor([14]), action_horizon=6, td_horizon=6)


def test_clean_and_independent_successor_view_selection() -> None:
    cache = _cache()
    matched = cache.build_transitions(
        torch.tensor([0]), action_horizon=6, td_horizon=6, current_view=2
    )
    clean_target = cache.build_transitions(
        torch.tensor([0]),
        action_horizon=6,
        td_horizon=6,
        current_view=2,
        successor_view=0,
    )
    assert matched["visual_curr"].item() == 200
    assert matched["visual_next"].item() == 206
    assert clean_target["visual_curr"].item() == 200
    assert clean_target["visual_next"].item() == 6


def test_independent_target_views_are_separate_per_position_and_horizon(monkeypatch) -> None:
    cache = _cache(views=4)
    queued = iter(
        (
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[3, 2, 1, 0]]),
            torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]]),
        )
    )

    def fake_randint(high, size, *, device):
        assert high == 4
        value = next(queued).to(device)
        assert tuple(value.shape) == tuple(size)
        return value

    monkeypatch.setattr(torch, "randint", fake_randint)
    batch = cache.build_transitions(
        torch.tensor([0]),
        action_horizon=6,
        td_horizon=12,
        current_view=2,
        include_chunk_bootstraps=True,
        independent_target_samples=4,
    )

    assert batch["visual_curr"].item() == 200
    assert batch["visual_next"].item() == 209
    assert batch["target_curr_view_indices"].tolist() == [[0, 1, 2, 3]]
    assert batch["target_next_view_indices"].tolist() == [[3, 2, 1, 0]]
    assert batch["target_bootstrap_view_indices"].tolist() == [[[0, 1, 2, 3], [3, 2, 1, 0]]]
    assert batch["visual_curr_target"].flatten().tolist() == [0.0, 100.0, 200.0, 300.0]
    assert batch["visual_next_target"].flatten().tolist() == [309.0, 209.0, 109.0, 9.0]
    assert batch["visual_bootstrap_target"].shape == (1, 2, 4, 1)
    assert batch["visual_bootstrap_target"][0, 0, :, 0].tolist() == [
        6.0,
        106.0,
        206.0,
        306.0,
    ]
    assert batch["visual_bootstrap_target"][0, 1, :, 0].tolist() == [
        309.0,
        209.0,
        109.0,
        9.0,
    ]


def test_flat_cache_sampling_routes_independent_targets() -> None:
    batch = _cache().sample(2, action_horizon=6, td_horizon=6, independent_target_samples=1)
    assert batch["visual_curr_target"].shape == (2, 1, 1)
    assert batch["visual_next_target"].shape == (2, 1, 1)


def test_target_view_config_fails_loud_on_invalid_combinations(tmp_path) -> None:
    valid = {
        "target_view_sampling": "independent",
        "target_view_samples": 4,
        "use_embedding_cache": True,
        "embedding_cache_input": tmp_path / "cache.pt",
        "cache_augmented_views": 10,
    }
    assert resolve_independent_target_samples(**valid) == 4
    with pytest.raises(ValueError, match="prebuilt schema-v3"):
        resolve_independent_target_samples(**{**valid, "embedding_cache_input": None})
    with pytest.raises(ValueError, match="multi-view"):
        resolve_independent_target_samples(**{**valid, "cache_augmented_views": 0})
    with pytest.raises(ValueError, match="must be 1"):
        resolve_independent_target_samples(**{**valid, "target_view_sampling": "matched"})


def test_flat_cache_satisfies_training_replay_buffer_interface() -> None:
    # The training loop treats the flat cache as a drop-in replay buffer and reads
    # these members generically (e.g. logging `replay_buffer.fill_pct * 100`). A
    # missing member surfaces only as a mid-run AttributeError on the cluster, so
    # pin the whole surface the precompute-embeddings training path touches.
    cache = _cache()
    for attr in (
        "fill_pct",
        "memory_gb",
        "capacity",
        "size",
        "num_visual_views",
        "anchor_rows",
        "to_device",
        "compute_stats",
        "build_transitions",
        "sample",
    ):
        assert hasattr(cache, attr), f"flat cache missing replay-buffer member {attr!r}"
    # A fully materialized in-memory cache is always 100% filled, matching
    # EncodedReplayBuffer.fill_pct so the training data/buffer_fill_pct log is sane.
    assert cache.fill_pct == 1.0
    assert cache.memory_gb > 0.0


def test_prebuilt_metadata_adapter_reproduces_cache_anchor_rows() -> None:
    tensors = _cache().tensor_payload()
    # Mirror production timeout episodes: an invalid boundary-successor row is
    # retained after each valid prefix, and every complete H6 window is eligible.
    tensors["is_valid"][9] = False
    tensors["anchor_eligible"][11] = True
    cache = FlatEncodedTrajectoryCache.from_tensor_payload(tensors)
    dataset = build_prebuilt_train_metadata_dataset(cache, {"repo_ids": ["repo"]})
    sub_datasets = dataset._datasets
    _episodes, _datasets, _frames, from_indices, to_indices = build_multidataset_frame_metadata(
        sub_datasets, ["repo"]
    )
    done = sub_datasets[0].hf_dataset["done"].tolist()
    anchors = iql_episode_frame_indices(
        from_indices,
        to_indices,
        done,
        episode_set=set(range(len(from_indices))),
        horizon=6,
    )

    assert anchors == cache.anchor_rows.tolist()
    assert dataset.num_frames == cache.size
    assert dataset.num_episodes == 2
    with pytest.raises(RuntimeError, match="cannot decode"):
        dataset[0]


def test_prebuilt_eval_metadata_adapter_preserves_every_holdout_row() -> None:
    holdout = {
        "dataset_indices": torch.tensor([0, 0, 1]),
        "success": torch.tensor([1, 0, 1]),
        "source": torch.tensor([0, 1, 0]),
    }
    dataset = build_prebuilt_eval_metadata_dataset(holdout, ["eval-a", "eval-b"])
    sub_datasets = dataset._datasets
    _episodes, _datasets, _frames, from_indices, to_indices = build_multidataset_frame_metadata(
        sub_datasets, ["eval-a", "eval-b"]
    )
    done = torch.cat([sub.hf_dataset["done"] for sub in sub_datasets]).tolist()
    anchors = iql_episode_frame_indices(
        from_indices,
        to_indices,
        done,
        episode_set=set(range(len(from_indices))),
        horizon=6,
    )

    assert anchors == [0, 1, 2]
    assert dataset.num_episodes == 3


def test_truncated_transition_feeds_loss_end_to_end() -> None:
    # Interface-alignment pin: build_transitions output (td_horizon=12, anchor
    # truncated at 7 valid steps by its episode boundary) assembled exactly the
    # way the training loop does, through forward_encoded_states. Padded reward
    # positions repeat the last valid row (1.6 each) — the hand-computed target
    # proves they are excluded and the bootstrap uses gamma**7 at row t+7.
    cache = _cache()
    batch = cache.build_transitions(
        torch.tensor([10]), action_horizon=6, td_horizon=12, current_view=0
    )
    assert batch["valid_steps"].item() == 7

    gamma = 0.99
    state_dim = 1 + 2  # toy visual feature dim + proprio dim
    action_dim = batch["action"].numel()
    model = VisionIQL(
        encoder=nn.Identity(),
        q1=QNetwork(state_dim, action_dim, hidden_dims=[8], use_layer_norm=False),
        q2=QNetwork(state_dim, action_dim, hidden_dims=[8], use_layer_norm=False),
        v_net=VNetwork(state_dim, hidden_dims=[8]),
        camera_keys=["cam"],
        separate_encoders=False,
        expectile=0.7,
        gamma=gamma,
        tau=0.005,
    )
    proprio = batch["observation.state"]
    curr_state = torch.cat([batch["visual_curr"], proprio[:, 0]], dim=-1)
    next_state = torch.cat([batch["visual_next"], proprio[:, 1]], dim=-1)
    discount_powers = torch.tensor([[gamma**i for i in range(12)]])
    with torch.no_grad():
        v_next = model.v_net(next_state)
        out = model.forward_encoded_states(
            curr_state,
            next_state,
            batch["action"].reshape(1, -1),
            batch["reward"],
            batch["done"],
            discount_powers,
            valid_steps=batch["valid_steps"],
        )
    expected = (
        sum(batch["reward"][0, i].item() * gamma**i for i in range(7)) + gamma**7 * v_next.item()
    )
    assert torch.allclose(out["td_target_mean"], torch.tensor(expected), rtol=1e-5)

    # Malformed valid_steps fail loud: wrong shape (silent broadcast hazard)
    # and fractional dtype.
    with pytest.raises(ValueError, match="shape"):
        model.forward_encoded_states(
            curr_state.repeat(2, 1),
            next_state.repeat(2, 1),
            batch["action"].reshape(1, -1).repeat(2, 1),
            batch["reward"].repeat(2, 1),
            batch["done"].repeat(2, 1),
            discount_powers,
            valid_steps=batch["valid_steps"],
        )
    with pytest.raises(ValueError, match="integer"):
        model.forward_encoded_states(
            curr_state,
            next_state,
            batch["action"].reshape(1, -1),
            batch["reward"],
            batch["done"],
            discount_powers,
            valid_steps=batch["valid_steps"].float(),
        )


def test_flat_cache_roundtrip_and_strict_metadata(tmp_path) -> None:
    cache = _cache()
    path = tmp_path / "flat.pt"
    metadata = {"repo_ids": ["example/repo"], "encoder_state_sha256": "abc"}
    save_flat_encoded_cache(path, cache, metadata)
    loaded = load_flat_encoded_cache(path, metadata)

    assert loaded.size == cache.size
    assert torch.equal(loaded.visual, cache.visual)
    manifest = json.loads(path.with_suffix(".pt.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["eligible_anchors"] == cache.anchor_rows.numel()
    assert manifest["normalization_action_horizon"] == 6
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_flat_encoded_cache(path, {**metadata, "encoder_state_sha256": "different"})


class _FakeHfDataset:
    """Minimal Arrow-column stand-in for intervention_values_for_subdataset."""

    def __init__(self, columns: dict[str, list]) -> None:
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, name: str) -> list:
        return self._columns[name]


def test_flat_cache_dynamic_metadata_keys_single_sourced() -> None:
    # The trainer strips exactly the shared recipe keys (no re-listed copy that could drift).
    base = {key: 1 for key in FLAT_CACHE_DYNAMIC_METADATA_KEYS}
    base["encoder_state_sha256"] = "abc"
    base["repo_ids"] = ["example/repo"]

    train_stripped = flat_cache_expected_metadata(base)
    assert set(train_stripped) & FLAT_CACHE_DYNAMIC_METADATA_KEYS == set()
    assert train_stripped["encoder_state_sha256"] == "abc"


def test_intervention_column_required_when_recipe_shapes_reward() -> None:
    # Recipe with intervention_negative_reward set => the trainer derives
    # require_intervention=True. A source
    # subdataset with autonomous rollouts but no 'intervention' column must fail
    # loud instead of synthesizing zero-intervention rewards.
    metadata = {"intervention_negative_reward": -1.0}
    require_intervention = metadata["intervention_negative_reward"] is not None
    assert require_intervention is True

    source_values = [int(DataSource.HUMAN), int(DataSource.AUTONOMOUS)]
    hf = _FakeHfDataset({"source": source_values, "success": [1, 0]})
    with pytest.raises(ValueError, match="intervention"):
        intervention_values_for_subdataset(
            repo_id="example/policy-rollouts",
            column_names=hf.column_names,
            source_values=source_values,
            hf_dataset=hf,
            require_intervention=require_intervention,
        )

    # Same source with the recipe disabled (intervention_negative_reward None)
    # derives require_intervention=False and synthesizes zeros without raising.
    disabled = {"intervention_negative_reward": None}
    assert intervention_values_for_subdataset(
        repo_id="example/policy-rollouts",
        column_names=hf.column_names,
        source_values=source_values,
        hf_dataset=hf,
        require_intervention=disabled["intervention_negative_reward"] is not None,
    ) == [0, 0]


def test_flat_cache_selector_metadata_is_compared_both_ways():
    from mulligan.real.train.iql_flat_cache import validate_flat_encoded_cache_metadata

    subset = {"dataset_episodes": {"a/b": {"episode_index": [0, 1]}}}
    validate_flat_encoded_cache_metadata({"k": 1, **subset}, {"k": 1, **subset})
    validate_flat_encoded_cache_metadata({"k": 1, "extra": 2}, {"k": 1})
    with pytest.raises(ValueError, match="dataset_episodes"):
        validate_flat_encoded_cache_metadata({"k": 1, **subset}, {"k": 1})
    with pytest.raises(ValueError, match="dataset_episodes"):
        validate_flat_encoded_cache_metadata({"k": 1}, {"k": 1, **subset})


def test_one_pass_frame_set_covers_terminal_tail_and_timeout_successors() -> None:
    # Timeout boundary AND terminal tail: the physical-frame set derive_flat_cache_rows
    # keys the cache on.
    action_horizon = 3
    episode = torch.tensor([0] * 10 + [1] * 8, dtype=torch.long)
    frame = torch.tensor(list(range(10)) + list(range(8)), dtype=torch.long)
    n = int(frame.numel())
    base = torch.arange(n, dtype=torch.float32)
    done = torch.zeros(n, dtype=torch.long)
    done[6:10] = 1  # episode 0 terminates at frame 6 with a terminal tail 6..9
    frame_table = {
        "proprio": torch.stack((base, base + 0.5), dim=1),
        "action": torch.stack((base, -base), dim=1),
        "reward": base / 10,
        "done": done,
        "is_valid": torch.ones(n, dtype=torch.bool),
        "success": (episode == 0).long(),
        "source": torch.zeros(n, dtype=torch.long),
        "intervention": torch.zeros(n, dtype=torch.long),
        "dataset_index": torch.zeros(n, dtype=torch.long),
        "episode_index": episode,
        "frame_index": frame,
    }
    # Episode 0: anchors up to the terminal chunk; episode 1: pure timeout episode.
    ep0_anchors = [0, 1, 2, 3]  # frame 3 terminates via done at frame 6 (>=3 valid)
    ep1_anchors = [10 + f for f in range(8) if f + action_horizon <= 7]
    anchors = torch.tensor(ep0_anchors + ep1_anchors, dtype=torch.long)

    rows, anchor_eligible, boundary_added = derive_flat_cache_rows(
        source_anchor_indices=anchors,
        frame_table=frame_table,
        action_horizon=action_horizon,
    )

    # Each non-terminal anchor adds its within-episode successor at anchor + H:
    # episode 0 adds 4..6, episode 1 (timeout) adds 15..17.
    expected_rows = list(range(0, 7)) + list(range(10, 18))
    assert rows.tolist() == expected_rows
    assert boundary_added == 6
    assert rows[anchor_eligible].tolist() == anchors.tolist()


def test_one_pass_build_aligns_physical_frames_with_rows() -> None:
    # Two timeout episodes (10 + 8 frames); anchors keep a full 3-step action chunk.
    action_horizon = 3
    episode = torch.tensor([0] * 10 + [1] * 8, dtype=torch.long)
    frame = torch.tensor(list(range(10)) + list(range(8)), dtype=torch.long)
    n = int(frame.numel())
    base = torch.arange(n, dtype=torch.float32)
    frame_table = {
        "proprio": torch.stack((base, base + 0.5), dim=1),
        "action": torch.stack((base, -base), dim=1),
        "reward": base / 10,
        "done": torch.zeros(n, dtype=torch.long),
        "is_valid": torch.ones(n, dtype=torch.bool),
        "success": (episode == 0).long(),
        "source": torch.zeros(n, dtype=torch.long),
        "intervention": torch.zeros(n, dtype=torch.long),
        "dataset_index": torch.zeros(n, dtype=torch.long),
        "episode_index": episode,
        "frame_index": frame,
    }
    anchors = torch.tensor(
        [g for g in range(10) if g + action_horizon <= 9]
        + [10 + f for f in range(8) if f + action_horizon <= 7],
        dtype=torch.long,
    )
    rows, anchor_eligible, _ = derive_flat_cache_rows(
        source_anchor_indices=anchors, frame_table=frame_table, action_horizon=action_horizon
    )
    visual = torch.stack((rows.float(), rows.float() + 0.25, rows.float() * 2.0), dim=1)

    cache = build_flat_cache_from_physical_frames(
        visual=visual,
        rows=rows,
        anchor_eligible=anchor_eligible,
        frame_table=frame_table,
        action_horizon=action_horizon,
    )

    assert cache.size == rows.numel()
    torch.testing.assert_close(cache.visual, visual)
    torch.testing.assert_close(cache.proprio, frame_table["proprio"][rows])
    assert rows[cache.anchor_rows].tolist() == anchors.tolist()
    assert set(cache.tensor_payload()) == set(FlatEncodedTrajectoryCache._TENSOR_NAMES)
    with pytest.raises(ValueError, match="physical visual bank"):
        build_flat_cache_from_physical_frames(
            visual=visual[:-1],
            rows=rows,
            anchor_eligible=anchor_eligible,
            frame_table=frame_table,
            action_horizon=action_horizon,
        )
