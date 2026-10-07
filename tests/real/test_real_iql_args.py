"""Argument parsing guards for real-world Vision-IQL training."""

from __future__ import annotations

from pathlib import Path

import pytest

from mulligan.real.train.iql_args import parse_args
from mulligan.real.train.critic import (
    resolve_camera_keys,
    resolve_io_resources,
    resolve_td_horizon,
    validate_auto_resume_frequencies,
    validate_iql_resume_topology,
    validate_target_clipping_config,
)


def test_real_iql_io_resources_split_workers_across_loader_pools() -> None:
    assert resolve_io_resources(buffer_capacity_gb=24.0, num_workers=16) == (8, 16)
    assert resolve_io_resources(buffer_capacity_gb=24.0, num_workers=15) == (7, 14)
    assert resolve_io_resources(buffer_capacity_gb=24.0, num_workers=0) == (0, 0)


def test_real_iql_io_resources_reject_undersubscribed_worker_budget() -> None:
    with pytest.raises(ValueError, match="must be 0 or at least"):
        resolve_io_resources(buffer_capacity_gb=24.0, num_workers=1)
    with pytest.raises(ValueError, match="--buffer-capacity-gb must be > 0"):
        resolve_io_resources(buffer_capacity_gb=0.0, num_workers=16)


def test_real_iql_auto_resume_args_default_enabled_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "mulligan.real.train.critic",
            "--repo-ids",
            "mulligan/example",
            "--encoder-artifact",
            "entity/project/artifact:v0",
        ],
    )

    args = parse_args()

    assert args.log_freq == 5000
    assert args.resume_checkpoint_freq == 5000
    assert args.no_auto_resume is False


def test_real_iql_auto_resume_args_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "mulligan.real.train.critic",
            "--repo-ids",
            "mulligan/example",
            "--encoder-artifact",
            "entity/project/artifact:v0",
            "--resume-checkpoint-freq",
            "500",
            "--no-auto-resume",
        ],
    )

    args = parse_args()

    assert args.resume_checkpoint_freq == 500
    assert args.no_auto_resume is True


def test_real_iql_embedding_cache_persistence_args(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "mulligan.real.train.critic",
            "--repo-ids",
            "mulligan/example",
            "--encoder-artifact",
            "entity/project/artifact:v0",
            "--precompute-embeddings",
            "--embedding-cache-augmented-views",
            "10",
            "--embedding-cache-output",
            "outputs/cache.pt",
            "--embedding-cache-only",
            "--embedding-eval-cache-output",
            "outputs/eval-cache.pt",
            "--embedding-eval-cache-only",
        ],
    )

    args = parse_args()

    assert args.embedding_cache_output == Path("outputs/cache.pt")
    assert args.embedding_cache_only is True
    assert args.embedding_eval_cache_output == Path("outputs/eval-cache.pt")
    assert args.embedding_eval_cache_only is True


def test_real_iql_auto_resume_rejects_more_frequent_wandb_logging() -> None:
    with pytest.raises(ValueError, match="--log-freq >= --resume-checkpoint-freq"):
        validate_auto_resume_frequencies(
            auto_resume_enabled=True,
            log_freq=100,
            resume_checkpoint_freq=5000,
        )
    # Without W&B there is no logged history to keep behind the resume state.
    validate_auto_resume_frequencies(
        auto_resume_enabled=True, log_freq=100, resume_checkpoint_freq=5000, use_wandb=False
    )


def test_real_iql_td_horizon_is_chunk_aligned_and_decoupled() -> None:
    assert resolve_td_horizon(action_horizon=6, td_horizon_steps=None) == 6
    assert resolve_td_horizon(action_horizon=6, td_horizon_steps=12) == 12
    assert resolve_td_horizon(action_horizon=6, td_horizon_steps=18) == 18


@pytest.mark.parametrize("td_horizon", [5, 7, 10])
def test_real_iql_td_horizon_rejects_short_or_non_chunk_aligned(td_horizon: int) -> None:
    with pytest.raises(ValueError, match="td-horizon-steps"):
        resolve_td_horizon(action_horizon=6, td_horizon_steps=td_horizon)


def test_zero_intervention_penalty_allows_sparse_reward_target_clipping() -> None:
    validate_target_clipping_config(
        intervention_negative_reward=0.0,
        clip_targets_min=0.0,
        clip_targets_max=1.0,
    )


def test_nonzero_intervention_penalty_rejects_incompatible_target_clipping() -> None:
    with pytest.raises(ValueError, match="changes the reward range"):
        validate_target_clipping_config(
            intervention_negative_reward=-1.0,
            clip_targets_min=0.0,
            clip_targets_max=1.0,
        )


def test_resume_topology_rejects_target_v_state() -> None:
    with pytest.raises(RuntimeError, match="target-V network"):
        validate_iql_resume_topology(
            {
                "v_target.mlp.0.weight": object(),
                "q1.mlp.0.weight": object(),
            }
        )


def test_resume_topology_accepts_target_q_state() -> None:
    validate_iql_resume_topology(
        {
            "q1_target.mlp.0.weight": object(),
            "q2_target.mlp.0.weight": object(),
        }
    )


def test_real_iql_resolves_explicit_role_camera_keys() -> None:
    camera_keys, excluded = resolve_camera_keys(
        all_camera_keys=[
            "observation.images.wrist_left",
            "observation.images.wrist_right",
            "observation.images.side_1",
            "observation.images.side_2",
        ],
        camera_filter="_left",
        camera_keys_arg="side_1,wrist_left",
    )

    assert camera_keys == ["observation.images.side_1", "observation.images.wrist_left"]
    assert excluded == ["observation.images.wrist_right", "observation.images.side_2"]


def test_vision_idql_artifact_name_dedups_prefix_and_fits_cap() -> None:
    from mulligan.real.train.critic import vision_idql_artifact_name

    # Run names already carry the vision-idql- prefix. The
    # longest real run name must fit W&B's 128-char cap with the longest
    # checkpoint suffix (this name exceeded the cap when the
    # prefix was doubled).
    r1 = (
        "vision-idql-insert-marker-d1-mulligan-sobol-with-cf-r1-replacement-"
        "r0e23-b025-r0o100-dag50-with-cf-no-cf-100k-seed-1"
    )
    name = vision_idql_artifact_name(r1, "step_100000")
    assert name == f"{r1}-step_100000"
    assert len(name) <= 128
    assert vision_idql_artifact_name(r1, "final").endswith("-final")

    # Legacy run names without the prefix still get it.
    legacy = vision_idql_artifact_name("iql-tql-direct-aug", "final")
    assert legacy == "vision-idql-iql-tql-direct-aug-final"

    # Over-long names fail loudly instead of dying at the first upload.
    with pytest.raises(ValueError, match="128"):
        vision_idql_artifact_name("vision-idql-" + "x" * 120, "step_100000")
