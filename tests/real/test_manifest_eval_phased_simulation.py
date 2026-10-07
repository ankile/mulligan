"""Three-phase simulation of the Cable 15-arm lineage eval protocol against the REAL
``manifest_eval`` scheduling, results-file, and resume code (no hardware).

Phase 1: 15 ``--fixed-policy`` specs, ids 9..14 retired (``--drop-fixed-policy``), rounds 1..20
(``--stop-after-round 20``).
Phase 2: same specs, no drops, ``--rerun-incomplete-rounds``: rounds 1..20 get ONLY the 6 missing
slots in their pre-baked order/labels, then rounds 21..50 run all 15; an interruption mid catch-up
is resumed. results.json round-trips between phases and ends with every (round, policy) exactly once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mulligan.real.eval.common import (
    RolloutRecord,
    load_previous_results,
    save_results_file,
)
from mulligan.real.eval.manifest_eval import (
    PolicyMapEntry,
    ResolvedPolicy,
    _complete_round_numbers,
    _ensure_all_round_plans,
    _entries_and_names,
    _load_phase_stops,
    _load_round_plans,
    _pending_manifest_rounds,
    _remaining_from_plan,
    _round_plans_by_round,
)

N_ARMS = 15
N_ROUNDS = 50
DROPPED = frozenset(range(9, 15))
PHASE1_CAP = 20
SEED = 2026090702


class _Target:
    def __init__(self, manifest_idx: int) -> None:
        self.manifest_idx = manifest_idx
        self.pen_x = 0.0
        self.pen_y = 0.0
        self.pen_yaw = 0.0


def _fixed() -> list[ResolvedPolicy]:
    return [
        ResolvedPolicy(
            name=f"arm{i}",
            model_id=f"hf://org/arm{i}",
            slot_key=f"fixed:{i}",
            slot_type="fixed",
        )
        for i in range(N_ARMS)
    ]


def _policy_map() -> dict[str, PolicyMapEntry]:
    return {
        p.slot_key: PolicyMapEntry(policy_id=i, name=p.name, model_id=p.model_id)
        for i, p in enumerate(_fixed())
    }


def _rec(round_num: int, entry: dict, episode_index: int, visit_id: str) -> RolloutRecord:
    return RolloutRecord(
        round_num=round_num,
        policy_id=int(entry["policy_id"]),
        model_id=str(entry["model_id"]),
        anonymous_label=str(entry["anonymous_label"]),
        outcome="success" if (round_num + int(entry["policy_id"])) % 3 else "failure",
        num_steps=100,
        episode_index=episode_index,
        manifest_idx=round_num - 1,
        pen_x=0.0,
        pen_y=0.0,
        pen_yaw=0.0,
        visit_id=visit_id,
    )


class Session:
    """One manifest_eval invocation over a shared results.json, driven by the real helpers."""

    def __init__(self, results_path: Path, dropped: frozenset[int], rerun_incomplete: bool):
        self.results_path = results_path
        self.dropped = dropped
        self.policy_map = _policy_map()
        self.previous_records, previous_policy_map = load_previous_results(results_path)
        for model_id, (policy_id, _name) in previous_policy_map.items():
            assert (
                self.policy_map[f"fixed:{policy_id}"].model_id == model_id
            )  # ids are list positions
        self.round_plans = _load_round_plans(results_path)
        self.phase_stops = _load_phase_stops(results_path)
        self.cap_triggered = False
        self.visit_id = f"visit-{len(self.phase_stops) + 1}"
        self.round_plans_by_round = _round_plans_by_round(self.round_plans)
        self.pending, self.completed_by_round = _pending_manifest_rounds(
            self.previous_records,
            num_rounds=N_ROUNDS,
            slots_per_round=N_ARMS,
            rerun_incomplete=rerun_incomplete,
            dropped_policy_ids=dropped,
        )
        _ensure_all_round_plans(
            manifest_targets=[_Target(i) for i in range(N_ROUNDS)],
            round_plans=self.round_plans,
            round_plans_by_round=self.round_plans_by_round,
            fixed_policies=_fixed(),
            policy_map=self.policy_map,
            random_seed=SEED,
        )
        self.new_records: list[RolloutRecord] = []
        self.next_episode = max((r.episode_index for r in self.previous_records), default=-1) + 1

    def remaining(self, round_num: int) -> list[dict]:
        return _remaining_from_plan(
            plan=self.round_plans_by_round[round_num],
            round_num=round_num,
            completed_records=self.completed_by_round.get(round_num, [])
            + [r for r in self.new_records if r.round_num == round_num],
            slots_per_round=N_ARMS,
            dropped_policy_ids=self.dropped,
        )

    def run(
        self, *, stop_after_round: int = 0, interrupt_at: tuple[int, int] | None = None
    ) -> None:
        for round_num in self.pending:
            for k, entry in enumerate(self.remaining(round_num)):
                if interrupt_at == (round_num, k):
                    return
                self.new_records.append(_rec(round_num, entry, self.next_episode, self.visit_id))
                self.next_episode += 1
            if stop_after_round and round_num >= stop_after_round:
                self.cap_triggered = True
                return

    def shutdown(self) -> None:
        all_records = self.previous_records + self.new_records
        self.complete_rounds = _complete_round_numbers(
            all_records, slots_per_round=N_ARMS, dropped_policy_ids=self.dropped
        )
        if all_records:
            self.phase_stops.append(
                {
                    "visit_id": self.visit_id,
                    "stop_after_round": 0,
                    "cap_triggered": self.cap_triggered,
                    "dropped_policy_ids": sorted(self.dropped),
                    "max_round_with_records": max(r.round_num for r in all_records),
                    "n_records": len(all_records),
                    "record_keys": sorted(
                        [int(r.round_num), int(r.policy_id)] for r in all_records
                    ),
                }
            )
        entries, names = _entries_and_names(records=all_records, policy_map=self.policy_map)
        save_results_file(
            self.results_path,
            entries,
            all_records,
            argparse.Namespace(environment="routing_d2", random_seed=SEED),
            "sim-dataset",
            policy_names=names,
            round_plans=self.round_plans,
            phase_stops=self.phase_stops,
        )


def test_three_phase_protocol_end_to_end(tmp_path: Path) -> None:
    results = tmp_path / "results.json"

    # ---- Phase 1: ids 9..14 retired, cap at round 20.
    p1 = Session(results, DROPPED, rerun_incomplete=True)
    assert p1.pending == list(range(1, N_ROUNDS + 1))
    plans_after_p1 = {int(p["round"]): [dict(e) for e in p["policy_order"]] for p in p1.round_plans}
    for plan in p1.round_plans:  # every plan is a full 15-arm shuffle, dropped arms included
        assert sorted(int(e["policy_id"]) for e in plan["policy_order"]) == list(range(N_ARMS))
    assert all(len(p1.remaining(r)) == 9 for r in (1, 20, 50))
    p1.run(stop_after_round=PHASE1_CAP)
    assert len(p1.new_records) == PHASE1_CAP * 9
    assert {r.policy_id for r in p1.new_records} == set(range(9))
    p1.shutdown()
    assert p1.complete_rounds == set(range(1, PHASE1_CAP + 1))  # complete under the active arms

    # ---- Phase 2a: drops removed. Without the rerun flag the resume fails loud.
    with pytest.raises(RuntimeError, match="rerun-incomplete-rounds"):
        Session(results, frozenset(), rerun_incomplete=False)
    p2 = Session(results, frozenset(), rerun_incomplete=True)
    assert len(p2.previous_records) == PHASE1_CAP * 9
    assert p2.pending == list(range(1, N_ROUNDS + 1))
    assert {
        int(p["round"]): [dict(e) for e in p["policy_order"]] for p in p2.round_plans
    } == plans_after_p1  # plans untouched
    for r in range(
        1, PHASE1_CAP + 1
    ):  # catch-up = exactly the 6 retired arms, pre-baked order + labels
        rem = p2.remaining(r)
        assert [int(e["policy_id"]) for e in rem] == [
            int(e["policy_id"])
            for e in sorted(plans_after_p1[r], key=lambda e: e["slot_idx"])
            if int(e["policy_id"]) in DROPPED
        ]
        assert all(int(e["policy_id"]) in DROPPED for e in rem)
    assert len(p2.remaining(PHASE1_CAP + 1)) == N_ARMS
    # Interrupted mid catch-up: rounds 1..10 caught up, round 11 has 3 of its 6.
    p2.run(interrupt_at=(11, 3))
    p2.shutdown()
    assert p2.complete_rounds == set(range(1, 11))  # only the caught-up rounds are complete
    caught_up = {
        (r.round_num, r.policy_id)
        for r in load_previous_results(results)[0]
        if r.policy_id in DROPPED
    }
    round11_first3 = [
        int(e["policy_id"])
        for e in sorted(plans_after_p1[11], key=lambda e: e["slot_idx"])
        if int(e["policy_id"]) in DROPPED
    ][:3]
    assert caught_up == {(r, pid) for r in range(1, 11) for pid in DROPPED} | {
        (11, pid) for pid in round11_first3
    }

    # ---- Phase 2b: resume after the interruption; finish everything.
    p3 = Session(results, frozenset(), rerun_incomplete=True)
    assert p3.pending == list(range(11, N_ROUNDS + 1))
    assert len(p3.remaining(11)) == 3 and all(
        int(e["policy_id"]) in DROPPED for e in p3.remaining(11)
    )
    p3.run()
    p3.shutdown()

    # ---- Invariants
    final_records, _ = load_previous_results(results)
    assert len(final_records) == N_ARMS * N_ROUNDS
    pairs = [(r.round_num, r.policy_id) for r in final_records]
    assert len(set(pairs)) == len(pairs) == 750
    assert sorted(r.episode_index for r in final_records) == list(range(750))
    assert set(pairs) == {(r, pid) for r in range(1, N_ROUNDS + 1) for pid in range(N_ARMS)}
    # Labels of every record match its pre-baked plan slot; head rounds kept phase-1 labels.
    by_round = _round_plans_by_round(_load_round_plans(results))
    for rec in final_records:
        planned = next(
            e
            for e in by_round[rec.round_num]["policy_order"]
            if int(e["policy_id"]) == rec.policy_id
        )
        assert rec.anonymous_label == planned["anonymous_label"]
    # Durable visit provenance: every record carries its invocation's visit_id, and every
    # graceful shutdown appended a phase_stops entry (3 invocations here).
    stops = _load_phase_stops(results)
    assert [s["visit_id"] for s in stops] == ["visit-1", "visit-2", "visit-3"]
    assert stops[0]["cap_triggered"] and stops[0]["dropped_policy_ids"] == sorted(DROPPED)
    assert {tuple(k) for k in stops[0]["record_keys"]} == {
        (r.round_num, r.policy_id) for r in final_records if r.visit_id == "visit-1"
    }
    visits_by_round: dict[int, set[str]] = {}
    for r in final_records:
        visits_by_round.setdefault(r.round_num, set()).add(r.visit_id)
    # Head rounds = two visits (phase-1 front arms, catch-up back arms); round 11's catch-up
    # was itself split by the interruption (three visits); tail rounds = one visit each.
    assert all(visits_by_round[r] == {"visit-1", "visit-2"} for r in range(1, 11))
    assert visits_by_round[11] == {"visit-1", "visit-2", "visit-3"}
    assert all(visits_by_round[r] == {"visit-1", "visit-3"} for r in range(12, PHASE1_CAP + 1))
    assert all(visits_by_round[r] == {"visit-3"} for r in range(PHASE1_CAP + 1, N_ROUNDS + 1))
    assert {r for r, v in visits_by_round.items() if len(v) == 1} == set(
        range(PHASE1_CAP + 1, N_ROUNDS + 1)
    )
    # A resume after completion has nothing pending and a repeated shutdown only appends a stop.
    p4 = Session(results, frozenset(), rerun_incomplete=True)
    assert p4.pending == []
    p4.shutdown()
    assert load_previous_results(results)[0] == final_records
    assert len(_load_phase_stops(results)) == 4
