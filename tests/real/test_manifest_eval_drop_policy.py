"""Unit tests for ``--drop-fixed-policy`` retire-on-resume accounting in
``mulligan.real.eval.manifest_eval``.

These cover the pure scheduling/accounting helpers (no robot, no policy load):
retiring one arm on resume must keep its already-saved records, count earlier
fully-complete rounds as still-complete, and run only the surviving arms in
pending rounds while preserving the blind paired design.
"""

from __future__ import annotations

import random

import pytest

from mulligan.real.eval.common import RolloutRecord
from mulligan.real.eval.manifest_eval import (
    ResolvedPolicy,
    _apply_round_cap,
    _maybe_override_idql_samples,
    _complete_round_numbers,
    _make_round_plan,
    _parse_fixed_policy_dp_override,
    _pending_manifest_rounds,
    _remaining_from_plan,
)


def test_manifest_eval_cli_has_no_arena_or_speech_flags(monkeypatch):
    from mulligan.real.eval.manifest_eval import parse_args

    argv = [
        "manifest_eval",
        "--environment",
        "routing_d2",
        "--initial-states-manifest",
        "manifest.json",
        "--fixed-policy",
        "a=hf://org/a",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = vars(parse_args())
    assert args["fixed_policy"] == ["a=hf://org/a"]
    assert not [k for k in args if "arena" in k or "speech" in k or "openpi" in k]


def test_fixed_policy_model_ids_need_a_scheme_or_a_local_dir(tmp_path, monkeypatch):
    from mulligan.real.eval.manifest_eval import _normalize_model_id

    assert _normalize_model_id("hf://org/repo@abc") == "hf://org/repo@abc"
    assert _normalize_model_id("wandb://e/p/a:v0") == "wandb://e/p/a:v0"
    (tmp_path / "ckpt").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _normalize_model_id("ckpt") == str(tmp_path.resolve() / "ckpt")
    with pytest.raises(ValueError, match="Unsupported model ID"):
        _normalize_model_id("org/repo")


def test_parse_fixed_policy_dp_override_valid():
    name, dp = _parse_fixed_policy_dp_override("dpiql_arm=wandb://ent/proj/dp-final:v0")
    assert name == "dpiql_arm"
    assert dp == "wandb://ent/proj/dp-final:v0"
    # whitespace-tolerant
    assert _parse_fixed_policy_dp_override("  a = b ") == ("a", "b")


def test_parse_fixed_policy_dp_override_rejects_malformed():
    for bad in ("noequals", "=onlydp", "onlyname=", "  =  "):
        with pytest.raises(ValueError):
            _parse_fixed_policy_dp_override(bad)


class _Target:
    def __init__(self, manifest_idx: int) -> None:
        self.manifest_idx = manifest_idx
        self.pen_x = 0.0
        self.pen_y = 0.0
        self.pen_yaw = 0.0


def _rec(round_num: int, policy_id: int, anonymous_label: str | None = None) -> RolloutRecord:
    return RolloutRecord(
        round_num=round_num,
        policy_id=policy_id,
        model_id=f"model-{policy_id}",
        # Real runs stamp the plan's shuffled label; "?" matches any plan slot. The
        # completeness helpers ignore labels, only _remaining_from_plan checks them.
        anonymous_label=anonymous_label or "?",
        outcome="success",
        num_steps=10,
        episode_index=round_num * 10 + policy_id,
        manifest_idx=round_num - 1,
        pen_x=0.0,
        pen_y=0.0,
        pen_yaw=0.0,
    )


def _five_arm_plan(round_num: int) -> dict:
    participants = [
        ResolvedPolicy(
            name=f"name-{pid}",
            model_id=f"model-{pid}",
            slot_key=f"fixed:{pid}",
            slot_type="fixed",
        )
        for pid in range(5)
    ]
    policy_ids_by_slot_key = {f"fixed:{pid}": pid for pid in range(5)}
    return _make_round_plan(
        round_num=round_num,
        target=_Target(round_num - 1),
        participants=participants,
        policy_ids_by_slot_key=policy_ids_by_slot_key,
        rng=random.Random(1234 + round_num),
    )


# Rounds 1-2 fully complete (5 arms each), round 3 not started. policy_id 3 is the dropped arm.
DROPPED = frozenset({3})
COMPLETE_RECORDS = [_rec(r, pid) for r in (1, 2) for pid in range(5)]


def test_baseline_no_drop_marks_full_rounds_complete():
    pending, completed_by_round = _pending_manifest_rounds(
        COMPLETE_RECORDS, num_rounds=3, slots_per_round=5
    )
    assert pending == [3]
    assert completed_by_round == {}
    assert _complete_round_numbers(COMPLETE_RECORDS, slots_per_round=5) == {1, 2}


def test_drop_keeps_earlier_full_rounds_complete():
    # Rounds 1-2 have 5 records incl the dropped arm; with one arm retired they
    # still count complete (4 surviving == effective 4), only round 3 pends.
    pending, completed_by_round = _pending_manifest_rounds(
        COMPLETE_RECORDS,
        num_rounds=3,
        slots_per_round=5,
        dropped_policy_ids=DROPPED,
    )
    assert pending == [3]
    assert completed_by_round == {}
    assert _complete_round_numbers(
        COMPLETE_RECORDS, slots_per_round=5, dropped_policy_ids=DROPPED
    ) == {1, 2}


def test_drop_excludes_retired_arm_from_pending_round_slots():
    plan = _five_arm_plan(3)
    remaining = _remaining_from_plan(
        plan=plan,
        round_num=3,
        completed_records=[],
        slots_per_round=5,
        dropped_policy_ids=DROPPED,
    )
    assert len(remaining) == 4
    assert all(int(entry["policy_id"]) != 3 for entry in remaining)
    # Surviving arms keep their pre-baked anonymous labels (blind design intact).
    assert {int(e["policy_id"]) for e in remaining} == {0, 1, 2, 4}


def test_drop_round_completes_with_only_surviving_records():
    # Round 3 ran the 4 surviving arms (not the dropped one). It must read as complete.
    survivor_records = COMPLETE_RECORDS + [_rec(3, pid) for pid in (0, 1, 2, 4)]
    pending, _ = _pending_manifest_rounds(
        survivor_records,
        num_rounds=3,
        slots_per_round=5,
        dropped_policy_ids=DROPPED,
    )
    assert pending == []
    assert _complete_round_numbers(
        survivor_records, slots_per_round=5, dropped_policy_ids=DROPPED
    ) == {1, 2, 3}
    # And a fully-pending round 3 plan now has nothing left to run.
    plan = _five_arm_plan(3)
    remaining = _remaining_from_plan(
        plan=plan,
        round_num=3,
        completed_records=[_rec(3, pid) for pid in (0, 1, 2, 4)],
        slots_per_round=5,
        dropped_policy_ids=DROPPED,
    )
    assert remaining == []


def test_drop_rejects_extra_active_record():
    # If a "complete" round somehow has 5 surviving (non-dropped) records, that is
    # corruption and must fail loudly, not silently truncate.
    bad = [_rec(1, pid) for pid in range(5)]
    bad.append(
        RolloutRecord(
            round_num=1,
            policy_id=7,
            model_id="model-7",
            anonymous_label="H",
            outcome="success",
            num_steps=10,
            episode_index=99,
            manifest_idx=0,
            pen_x=0.0,
            pen_y=0.0,
            pen_yaw=0.0,
        )
    )
    with pytest.raises(RuntimeError):
        _pending_manifest_rounds(bad, num_rounds=1, slots_per_round=5, dropped_policy_ids=DROPPED)


# ---- Phased eval: retire arms now (--drop-fixed-policy), un-drop them later on the same starts.
# Phase 1 ran rounds 1-2 with policy_id 3 retired (4 records each); phase 2 resumes with NO drops.
PHASE1_RECORDS = [_rec(r, pid) for r in (1, 2) for pid in (0, 1, 2, 4)]


def test_undrop_marks_earlier_rounds_pending_and_requires_rerun_flag():
    with pytest.raises(RuntimeError, match="rerun-incomplete-rounds"):
        _pending_manifest_rounds(PHASE1_RECORDS, num_rounds=3, slots_per_round=5)
    pending, completed_by_round = _pending_manifest_rounds(
        PHASE1_RECORDS, num_rounds=3, slots_per_round=5, rerun_incomplete=True
    )
    assert pending == [1, 2, 3]
    assert {r: len(v) for r, v in completed_by_round.items()} == {1: 4, 2: 4}
    assert _complete_round_numbers(PHASE1_RECORDS, slots_per_round=5) == set()


def test_undrop_schedules_only_the_retired_arm_in_its_prebaked_slot():
    for round_num in (1, 2):
        plan = _five_arm_plan(round_num)
        remaining = _remaining_from_plan(
            plan=plan,
            round_num=round_num,
            completed_records=[r for r in PHASE1_RECORDS if r.round_num == round_num],
            slots_per_round=5,
        )
        assert [int(e["policy_id"]) for e in remaining] == [3]
        planned = next(e for e in plan["policy_order"] if int(e["policy_id"]) == 3)
        assert remaining[0]["anonymous_label"] == planned["anonymous_label"]
    # Untouched round 3 runs every arm in the pre-baked shuffle.
    remaining = _remaining_from_plan(
        plan=_five_arm_plan(3), round_num=3, completed_records=[], slots_per_round=5
    )
    assert len(remaining) == 5


def test_round_cap_applies_before_collection():
    assert _apply_round_cap([1, 2, 3], 0) == [1, 2, 3]
    assert _apply_round_cap([1, 2, 3], 2) == [1, 2]
    # A restart after the cap was reached collects NOTHING.
    assert _apply_round_cap([21, 22, 50], 20) == []
    with pytest.raises(ValueError):
        _apply_round_cap([1], -1)


def test_idql_sample_override_is_silent(capsys):
    # BLINDING: the override runs before every anonymous rollout header; any stdout here
    # would identify the DP+IQL arm to the operator.
    from mulligan.real.eval.common import PolicyEntry
    from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

    policy = VisionIDQLRealWorldPolicy.__new__(VisionIDQLRealWorldPolicy)
    entry = PolicyEntry(model_id="m", policy_id=0, policy=policy, results=[])
    _maybe_override_idql_samples(entry, 32)
    _maybe_override_idql_samples(entry, 32)
    assert policy.num_action_samples == 32
    assert capsys.readouterr().out == ""


def test_save_results_file_is_atomic(tmp_path, monkeypatch):
    # A hard kill mid-write must leave the previous valid
    # results.json untouched (resume ledger + visit provenance live there).
    import argparse
    import json

    import mulligan.real.eval.common as ec
    from mulligan.real.eval.common import PolicyEntry, save_results_file

    out = tmp_path / "results.json"
    entries = [PolicyEntry("m0", 0, policy=None, results=[True])]
    args = argparse.Namespace(environment="routing_d2")
    save_results_file(out, entries, [_rec(1, 0)], args, "ds", policy_names=["a0"])
    good = out.read_text()
    assert json.loads(good)["rollouts"][0]["round"] == 1
    assert not out.with_name("results.json.tmp").exists()

    real_dump = json.dump

    def dying_dump(obj, fh, **kw):
        fh.write('{"timestamp":')
        raise KeyboardInterrupt

    monkeypatch.setattr(ec.json, "dump", dying_dump)
    with pytest.raises(KeyboardInterrupt):
        save_results_file(
            out, entries, [_rec(1, 0), _rec(1, 1)], args, "ds", policy_names=["a0", "a1"]
        )
    assert out.read_text() == good  # previous snapshot intact
    monkeypatch.setattr(ec.json, "dump", real_dump)
