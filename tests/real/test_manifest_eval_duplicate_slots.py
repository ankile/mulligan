"""Duplicate-model slots in ``manifest_eval._load_scheduled_policies``.

Two --fixed-policy labels may name the same checkpoint to compare eval-time settings.
A slot with its own DP override must get its own policy object, and slots that share
one object must each run with their own num_action_samples.
"""

from __future__ import annotations

from mulligan.real.eval.common import PolicyEntry
from mulligan.real.eval.manifest_eval import (
    PolicyMapEntry,
    _load_scheduled_policies,
    _maybe_override_idql_samples,
)
from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

CRITIC = "hf://mulligan/critic"
CHECKPOINT_SAMPLES = 16


def _fake_loader(calls: list[dict]):
    def loader(**kwargs):
        calls.append(kwargs)
        policy = VisionIDQLRealWorldPolicy.__new__(VisionIDQLRealWorldPolicy)
        policy.num_action_samples = CHECKPOINT_SAMPLES
        policy.dp_override = kwargs["dp_artifact_override"]
        return PolicyEntry(
            model_id=kwargs["model_id"], policy_id=kwargs["policy_id"], policy=policy
        )

    return loader


def _load(tmp_path, *, samples_by_slot, dp_override_by_slot=None, num_action_samples=None):
    calls: list[dict] = []
    policy_map = {
        "fixed:0": PolicyMapEntry(policy_id=0, name="A", model_id=CRITIC),
        "fixed:1": PolicyMapEntry(policy_id=1, name="B", model_id=CRITIC),
    }
    loaded = _load_scheduled_policies(
        policy_map=policy_map,
        num_action_samples_by_slot_key=samples_by_slot,
        device="cpu",
        noise_scheduler=None,
        num_inference_steps=None,
        camera_height=480,
        camera_width=640,
        n_action_steps=6,
        num_action_samples=num_action_samples,
        load_log_path=tmp_path / "policy_loading.log",
        dp_override_by_slot_key=dp_override_by_slot,
        policy_loader=_fake_loader(calls),
    )
    return loaded, calls


def _samples_before_each_rollout(loaded, samples_by_slot, order):
    """What the per-rollout re-apply in main() leaves on the policy before each rollout."""
    seen = []
    for slot_key in order:
        entry = loaded[slot_key]
        _maybe_override_idql_samples(entry, samples_by_slot.get(slot_key))
        seen.append(entry.policy.num_action_samples)
    return seen


def test_dp_override_slot_gets_its_own_policy_object(tmp_path):
    loaded, calls = _load(
        tmp_path,
        samples_by_slot={},
        dp_override_by_slot={"fixed:1": "hf://mulligan/newer-dp"},
    )
    assert [c["dp_artifact_override"] for c in calls] == [None, "hf://mulligan/newer-dp"]
    assert loaded["fixed:0"].policy is not loaded["fixed:1"].policy
    assert loaded["fixed:1"].policy.dp_override == "hf://mulligan/newer-dp"


def test_same_model_and_override_still_share_one_object(tmp_path):
    loaded, calls = _load(tmp_path, samples_by_slot={"fixed:0": 64})
    assert len(calls) == 1
    assert loaded["fixed:0"].policy is loaded["fixed:1"].policy


def test_shared_slot_without_override_keeps_checkpoint_samples(tmp_path):
    samples_by_slot = {"fixed:0": 64}
    loaded, _ = _load(tmp_path, samples_by_slot=samples_by_slot)
    seen = _samples_before_each_rollout(
        loaded, samples_by_slot, ["fixed:0", "fixed:1", "fixed:1", "fixed:0"]
    )
    assert seen == [64, CHECKPOINT_SAMPLES, CHECKPOINT_SAMPLES, 64]


def test_shared_slot_without_override_uses_cli_samples(tmp_path):
    samples_by_slot = {"fixed:1": 64}
    loaded, _ = _load(tmp_path, samples_by_slot=samples_by_slot, num_action_samples=32)
    seen = _samples_before_each_rollout(loaded, samples_by_slot, ["fixed:1", "fixed:0"])
    assert seen == [64, 32]
