# ruff: noqa: E402
"""Manifest-driven blind real-world policy evaluation.

- every ``--fixed-policy NAME=MODEL_ID`` (``hf://...`` or ``wandb://...``) is rolled
  out once per manifest round;
- a locked initial-state manifest (``data/real/manifests/<task>/...``) defines the
  start every policy is tested from in that round;
- anonymous rollout labels are shuffled per round and saved in ``round_plans``
  before any rollout starts, so interrupted runs resume the same assignment;
- phased evals retire arms with ``--drop-fixed-policy`` and cap a phase with
  ``--stop-after-round``; each rollout record carries its invocation's
  ``visit_id`` and every graceful shutdown appends a ``phase_stops`` entry;
- ``results.json`` (round plans, rollouts, per-policy summary) is the resume ledger
  and is pushed to the eval dataset repo with the episodes.
"""

from __future__ import annotations

# HighGUI must initialize before lerobot/av load (see mulligan.real.operator_ui.display).
# The parser uses only light imports and runs first, so --help and argument errors exit
# before the prewarm and before torch/lerobot load.
from mulligan.real.operator_ui.display import prewarm_highgui

import argparse
from pathlib import Path

from mulligan.real.eval.inference_server import add_remote_inference_args
from mulligan.real.operator_ui.cli import add_operator_ui_args
from mulligan.real.policy.dp import REAL_PROTOCOL_N_ACTION_STEPS
from mulligan.real.robot.cameras import DEFAULT_CAMERA_KEYS
from mulligan.real.robot.cli import add_robot_reset_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifest-driven blind real-robot eval")
    parser.add_argument("--environment", required=True)
    parser.add_argument(
        "--fixed-policy",
        action="append",
        default=[],
        metavar="NAME=MODEL_ID",
        help="Policy rolled out every round; MODEL_ID is hf://<repo> or wandb://...",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026052701,
        help="Seed of the per-round anonymous slot shuffle (round r uses seed + r).",
    )
    parser.add_argument("--initial-states-manifest", type=Path, required=True)
    add_operator_ui_args(parser, cards=True)
    parser.add_argument("--freq", type=int, default=15)
    parser.add_argument("--camera-filter", default="_left")
    parser.add_argument("--camera-keys", default=DEFAULT_CAMERA_KEYS)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--noise-scheduler", choices=["DDPM", "DDIM"], default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--num-action-samples", type=int, default=None)
    parser.add_argument(
        "--fixed-policy-num-action-samples",
        action="append",
        default=[],
        metavar="NAME=N",
        help=(
            "Override Vision-IQL num_action_samples for one fixed-policy label. "
            "This allows the same IQL artifact to appear in multiple fixed slots "
            "with different re-rank sample counts."
        ),
    )
    parser.add_argument(
        "--fixed-policy-dp-override",
        action="append",
        default=[],
        metavar="NAME=DP_ARTIFACT",
        help=(
            "Make a Vision-IQL fixed-policy re-rank a DIFFERENT DP actor than the one "
            "in its metadata (NAME=DP model ID). The IQL critic keeps its own "
            "fine-tuned encoder + Q/V; only the candidate-action sampler is swapped. Use "
            "for proxy evals (e.g. an older critic re-ranking a newer-round DP actor)."
        ),
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=REAL_PROTOCOL_N_ACTION_STEPS,
        help="Action chunk steps to execute before re-planning. Defaults to the "
        f"real-robot protocol exec horizon ({REAL_PROTOCOL_N_ACTION_STEPS}); "
        "prediction horizon comes from the checkpoint.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--no-auto-timeout",
        action="store_true",
        help=(
            "Do not auto-mark an episode 'timeout' at --max-steps. The policy keeps "
            "running until the operator presses 1=SUCCESS / 0=TIMEOUT / 9=FAILURE."
        ),
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-path", default="./data")
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument(
        "--max-recoverable-rollout-errors-per-slot",
        type=int,
        default=6,
        help=(
            "Maximum stale-observation or similar recoverable rollout faults to "
            "discard and retry for one anonymous policy slot before failing loudly."
        ),
    )
    parser.add_argument(
        "--recoverable-rollout-retry-delay-s",
        type=float,
        default=2.0,
        help="Initial delay before retrying a discarded recoverable rollout (default: 2.0)",
    )
    parser.add_argument(
        "--recoverable-rollout-retry-backoff",
        type=float,
        default=2.0,
        help="Multiplier for recoverable rollout retry delay after each fault (default: 2.0)",
    )
    parser.add_argument(
        "--recoverable-rollout-max-retry-delay-s",
        type=float,
        default=30.0,
        help="Maximum delay before retrying a discarded recoverable rollout (default: 30.0)",
    )
    parser.add_argument(
        "--checkpoint-interval-rounds",
        type=int,
        default=0,
        help=(
            "Finalize/reopen the dataset every N completed manifest rounds. "
            "Default 0 checkpoints only at shutdown, allowing background saves "
            "to overlap later rounds. Set 1 for maximum crash resilience."
        ),
    )
    parser.add_argument(
        "--rerun-incomplete-rounds",
        action="store_true",
        help=(
            "Allow resuming when previous results contain incomplete non-final "
            "rounds (e.g. after surgically removing a bad rollout record); only "
            "the missing policy slots of those rounds are re-run, keeping their "
            "planned anonymous labels."
        ),
    )
    parser.add_argument(
        "--progress-total-all-arms",
        action="store_true",
        help=(
            "Report progress / ETA against the FULL design (manifest rounds x every "
            "--fixed-policy slot, retired arms included) instead of the surviving arms only. "
            "For a PHASED eval whose first phase retires arms that catch up later on the same "
            "starts: the operator sees e.g. 180/750 rather than 180/450."
        ),
    )
    parser.add_argument(
        "--stop-after-round",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Outcome-independent cap for a PHASED eval: after manifest round N completes (and "
            "its checkpoint ran), stop through the normal graceful-quit path (seal drain, "
            "finalize, push, results.json). 0 = no cap. Lets the earlier phase leave a "
            "pre-declared block of untouched rounds for the full-arm-set phase."
        ),
    )
    parser.add_argument(
        "--drop-fixed-policy",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Retire a previously-included --fixed-policy on resume: keep passing its "
            "--fixed-policy spec (so completed-round policy_id mapping stays consistent), "
            "but skip loading and rolling it out for every remaining round. Already-saved "
            "records for it are preserved; pending rounds run only the surviving arms. "
            "Repeatable. The retired arm is not re-shuffled into pending rounds, so the "
            "blind paired design continues over the surviving arms."
        ),
    )
    add_robot_reset_args(parser)
    add_remote_inference_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    _CLI_ARGS = parse_args()
    prewarm_highgui()

import contextlib
import io
import json
import logging
import os
import random
import threading
import uuid
import time
from dataclasses import dataclass
from datetime import datetime

import torch

from mulligan.real.eval.blind_eval_helpers import (
    _copy_eval_initial_state_manifest,
    _eval_initial_state_frame_fields,
    _eval_initial_state_features,
    _load_eval_initial_state_manifest,
    _load_round_plans,
    _round_plans_by_round,
)
from mulligan.real.robot.cameras import remove_excluded_camera_features_from_lerobot_dataset
from mulligan.real.policy.loader import load_policy_by_model_id
from mulligan.real.eval.common import (
    PolicyEntry,
    RolloutRecord,
    checkpoint_dataset,
    cleanup_stale_image_episode_dirs,
    create_dataset,
    hub_push_timeout_s,
    load_previous_results,
    num_subtask_marks_for_task,
    print_results,
    reopen_dataset_for_append,
    rollout_outcome_line,
    run_with_timeout,
    save_episode_to_dataset,
    save_results_file,
    wait_until_ready,
)
from mulligan.real.eval.inference_server import remote_inference_session_from_args
from mulligan.real.collect.rollout import (
    RecoverableRolloutError,
    auto_detect_device,
    camera_serials_from_keys,
    parse_camera_keys,
    restrict_zed_cameras_to_serials,
    rollout_episode,
    verified_reset,
)
from mulligan.real.operator_ui.monitor import shared_policy_crop_boxes
from mulligan.real.operator_ui.session import OperatorUI
from mulligan.real.operator_ui.progress import EvalScene
from mulligan.real.collect.seal_pipeline import SealPipeline, run_seal_chain

logger = logging.getLogger(__name__)

_NATIVE_STDERR_REDIRECT_LOCK = threading.Lock()


@dataclass(frozen=True)
class FixedPolicySpec:
    name: str
    spec: str


@dataclass(frozen=True)
class PolicyMapEntry:
    policy_id: int
    name: str
    model_id: str


@dataclass(frozen=True)
class ResolvedPolicy:
    name: str
    model_id: str
    slot_key: str
    slot_type: str


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _save_results_file_quiet(*args, quiet: bool, **kwargs) -> None:
    if not quiet:
        save_results_file(*args, **kwargs)
        return
    with contextlib.redirect_stdout(io.StringIO()):
        save_results_file(*args, **kwargs)


@contextlib.contextmanager
def _quiet_native_stderr_to_file(path: Path):
    """Redirect native fd-2 chatter to a log while preserving Python exceptions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _NATIVE_STDERR_REDIRECT_LOCK:
        saved_stderr_fd = os.dup(2)
        try:
            with path.open("a") as log_file:
                os.dup2(log_file.fileno(), 2)
                yield
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)


@contextlib.contextmanager
def _quiet_background_save_context(native_stderr_log_path: Path):
    """Quiet known noisy dataset-save progress while keeping failures loud."""
    try:
        from datasets.utils import logging as datasets_logging
    except ImportError:
        datasets_logging = None

    progress_was_enabled = (
        datasets_logging.is_progress_bar_enabled() if datasets_logging is not None else None
    )
    if datasets_logging is not None and progress_was_enabled:
        datasets_logging.disable_progress_bar()
    try:
        with _quiet_native_stderr_to_file(native_stderr_log_path):
            yield
    finally:
        if datasets_logging is not None and progress_was_enabled:
            datasets_logging.enable_progress_bar()


def _normalize_model_id(raw_id: str) -> str:
    """Model IDs name their source: ``hf://<repo>``, ``wandb://<artifact>`` or an existing local
    checkpoint directory (returned as an absolute path)."""
    if raw_id.startswith(("wandb://", "hf://")):
        return raw_id
    local = Path(raw_id).expanduser()
    if local.is_dir():
        return str(local.resolve())
    raise ValueError(
        f"Unsupported model ID {raw_id!r}; use hf://<repo>, wandb://<artifact> or an existing "
        "local checkpoint directory"
    )


def _parse_fixed_policy(value: str) -> FixedPolicySpec:
    if "=" not in value:
        raise ValueError(f"--fixed-policy entries must be NAME=MODEL_ID, got {value!r}")
    name, spec = value.split("=", 1)
    name = name.strip()
    spec = spec.strip()
    if not name or not spec:
        raise ValueError(f"--fixed-policy entry has empty name or spec: {value!r}")
    return FixedPolicySpec(name=name, spec=spec)


def _parse_fixed_policy_num_action_samples(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise ValueError(f"--fixed-policy-num-action-samples entries must be NAME=N, got {value!r}")
    name, raw_count = value.split("=", 1)
    name = name.strip()
    raw_count = raw_count.strip()
    if not name or not raw_count:
        raise ValueError(
            f"--fixed-policy-num-action-samples entry has empty name or count: {value!r}"
        )
    count = int(raw_count)
    if count < 1:
        raise ValueError(f"num_action_samples override for {name!r} must be >= 1")
    return name, count


def _parse_fixed_policy_dp_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(
            f"--fixed-policy-dp-override entries must be NAME=DP_ARTIFACT, got {value!r}"
        )
    name, dp = value.split("=", 1)
    name, dp = name.strip(), dp.strip()
    if not name or not dp:
        raise ValueError(
            f"--fixed-policy-dp-override entry has empty name or DP artifact: {value!r}"
        )
    return name, dp


def _pending_manifest_rounds(
    records: list[RolloutRecord],
    *,
    num_rounds: int,
    slots_per_round: int,
    rerun_incomplete: bool = False,
    dropped_policy_ids: frozenset[int] = frozenset(),
) -> tuple[list[int], dict[int, list[RolloutRecord]]]:
    """Validate previous records and return the rounds that still need rollouts.

    Returns ``(pending_round_nums, completed_records_by_round)`` where
    *pending_round_nums* is every round (1-indexed) that does not yet have all
    *effective* (non-dropped) records, in ascending order, and
    *completed_records_by_round* holds the already-saved records for any pending
    round that was partially run.

    *dropped_policy_ids* are policies retired on resume (``--drop-fixed-policy``):
    their already-saved records (from earlier complete rounds) are kept but no
    longer counted toward round completeness, so a round that previously held all
    *slots_per_round* records still counts as complete with one arm retired, and a
    fresh round only needs the ``slots_per_round - len(dropped)`` surviving arms.

    Sequential resume only ever produces one partial round (the final one); a
    pending round before the final recorded round means a record was removed or
    lost mid-run, which is an error unless *rerun_incomplete* is set (used to
    redo a surgically-removed bad rollout in place).
    """
    effective_slots = slots_per_round - len(dropped_policy_ids)

    def _active(round_records: list[RolloutRecord]) -> list[RolloutRecord]:
        return [r for r in round_records if r.policy_id not in dropped_policy_ids]

    by_round: dict[int, list[RolloutRecord]] = {}
    for record in records:
        by_round.setdefault(record.round_num, []).append(record)
    for round_num, round_records in sorted(by_round.items()):
        policy_ids = {record.policy_id for record in round_records}
        if len(policy_ids) != len(round_records):
            raise RuntimeError(f"Previous results duplicate a policy in round {round_num}")
        if len(round_records) > slots_per_round:
            raise RuntimeError(
                f"Previous results contain {len(round_records)} records for round "
                f"{round_num}, but this eval has only {slots_per_round} slots"
            )
        if len(_active(round_records)) > effective_slots:
            raise RuntimeError(
                f"Previous results contain {len(_active(round_records))} active records for "
                f"round {round_num}, but this eval has only {effective_slots} surviving slots"
            )
        expected_manifest_idx = round_num - 1
        manifest_idxs = {record.manifest_idx for record in round_records}
        if manifest_idxs != {expected_manifest_idx}:
            raise RuntimeError(
                f"Round {round_num} has manifest_idx values {sorted(manifest_idxs)}; "
                f"expected {expected_manifest_idx}"
            )
    pending_round_nums = [
        round_num
        for round_num in range(1, num_rounds + 1)
        if len(_active(by_round.get(round_num, []))) < effective_slots
    ]
    final_round_num = max(by_round, default=0)
    mid_run_pending = [round_num for round_num in pending_round_nums if round_num < final_round_num]
    if mid_run_pending and not rerun_incomplete:
        raise RuntimeError(
            f"Previous results contain incomplete non-final round(s) {mid_run_pending}. "
            "Pass --rerun-incomplete-rounds to redo just the missing policy slots of "
            "those rounds."
        )
    completed_records_by_round = {
        round_num: by_round[round_num] for round_num in pending_round_nums if round_num in by_round
    }
    return pending_round_nums, completed_records_by_round


def _make_round_plan(
    *,
    round_num: int,
    target,
    participants: list[ResolvedPolicy],
    policy_ids_by_slot_key: dict[str, int],
    rng: random.Random,
) -> dict:
    order = list(range(len(participants)))
    rng.shuffle(order)
    policy_order = []
    for slot_idx, participant_idx in enumerate(order):
        participant = participants[participant_idx]
        policy_order.append(
            {
                "slot_idx": slot_idx,
                "anonymous_label": chr(ord("A") + slot_idx),
                "policy_id": policy_ids_by_slot_key[participant.slot_key],
                "model_id": participant.model_id,
                "name": participant.name,
                "slot_key": participant.slot_key,
                "slot_type": participant.slot_type,
            }
        )
    return {
        "round": round_num,
        "manifest_idx": target.manifest_idx,
        "pen_x": target.pen_x,
        "pen_y": target.pen_y,
        "pen_yaw": target.pen_yaw,
        "policy_order": policy_order,
    }


def _ensure_policy_map_entry(
    policy_map: dict[str, PolicyMapEntry],
    *,
    slot_key: str,
    model_id: str,
    policy_id: int,
    name: str,
) -> None:
    existing = policy_map.get(slot_key)
    if existing is not None:
        if existing.policy_id != policy_id:
            raise RuntimeError(
                f"Policy slot {slot_key} has policy_id={existing.policy_id} in saved "
                f"results, but round plan uses policy_id={policy_id}"
            )
        if existing.name and existing.name != name:
            raise RuntimeError(
                f"Policy slot {slot_key} has name {existing.name!r} in saved results, "
                f"but round plan uses {name!r}"
            )
        if existing.model_id != model_id:
            raise RuntimeError(
                f"Policy slot {slot_key} has model_id={existing.model_id}, "
                f"but round plan uses {model_id}"
            )
        return

    for existing_slot_key, existing in policy_map.items():
        if existing.policy_id == policy_id:
            raise RuntimeError(
                f"Round plan reuses policy_id={policy_id} for slot {slot_key}, "
                f"already assigned to {existing_slot_key}"
            )
    policy_map[slot_key] = PolicyMapEntry(policy_id=policy_id, name=name, model_id=model_id)


def _apply_round_cap(pending_round_nums: list[int], stop_after_round: int) -> list[int]:
    """Rounds this invocation may run under ``--stop-after-round`` (0 = no cap).

    Enforced BEFORE policy loading / robot init so a restart after the cap was reached
    (e.g. to retry publication) collects nothing instead of one more round.
    """
    if stop_after_round < 0:
        raise ValueError(f"--stop-after-round must be >= 0, got {stop_after_round}")
    if not stop_after_round:
        return list(pending_round_nums)
    return [round_num for round_num in pending_round_nums if round_num <= stop_after_round]


def _load_phase_stops(results_path: Path) -> list[dict]:
    """Durable invocation provenance (``phase_stops`` in results.json): one entry per
    graceful shutdown of every invocation (visit_id, retired arms, cap, the exact (round,
    policy_id) record set at that moment). Per-record visit identity lives on each rollout
    record (``visit_id``); this list is the invocation-level ledger next to it."""
    if not results_path.exists():
        return []
    data = json.loads(results_path.read_text())
    stops = data.get("phase_stops", [])
    if stops is None:
        return []
    if not isinstance(stops, list):
        raise ValueError(f"{results_path}: phase_stops must be a list")
    return stops


def _sync_policy_map_from_round_plan(
    policy_map: dict[str, PolicyMapEntry],
    round_plan: dict,
) -> None:
    policy_order = round_plan.get("policy_order")
    if not isinstance(policy_order, list):
        raise RuntimeError(f"Round plan {round_plan.get('round')} is missing policy_order")
    for planned in policy_order:
        model_id = str(planned["model_id"])
        # slot_key is genuinely optional in a persisted round plan: manifest_eval
        # resumes over the shared blind_eval results.json format, and
        # blind_eval._make_round_plan writes policy_order entries WITHOUT slot_key
        # (only slot_idx/anonymous_label/policy_id/model_id). For those plans the
        # model_id IS the slot key -- requiring planned["slot_key"] would crash a
        # legitimate resume.
        slot_key = str(planned.get("slot_key") or model_id)
        _ensure_policy_map_entry(
            policy_map,
            slot_key=slot_key,
            model_id=model_id,
            policy_id=int(planned["policy_id"]),
            name=str(planned.get("name") or model_id),
        )


def _ensure_all_round_plans(
    *,
    manifest_targets,
    round_plans: list[dict],
    round_plans_by_round: dict[int, dict],
    fixed_policies: list[ResolvedPolicy],
    policy_map: dict[str, PolicyMapEntry],
    random_seed: int,
) -> None:
    """Materialize every round plan before rollout starts.

    This keeps the anonymous A/B/C assignments stable even if the process is
    interrupted before data is saved, and it lets us preload all scheduled
    policies once before the operator starts the physical eval.
    """
    for existing_plan in sorted(round_plans, key=lambda plan: int(plan["round"])):
        _sync_policy_map_from_round_plan(policy_map, existing_plan)

    for round_num, target in enumerate(manifest_targets, start=1):
        existing_plan = round_plans_by_round.get(round_num)
        if existing_plan is not None:
            if int(existing_plan["manifest_idx"]) != target.manifest_idx:
                raise RuntimeError(
                    f"Round {round_num} plan has manifest_idx={existing_plan['manifest_idx']}, "
                    f"expected {target.manifest_idx}"
                )
            continue

        participants = list(fixed_policies)
        rng = random.Random(random_seed + round_num)
        policy_ids_by_slot_key = {
            slot_key: entry.policy_id for slot_key, entry in policy_map.items()
        }
        round_plan = _make_round_plan(
            round_num=round_num,
            target=target,
            participants=participants,
            policy_ids_by_slot_key=policy_ids_by_slot_key,
            rng=rng,
        )
        round_plans.append(round_plan)
        round_plans_by_round[round_num] = round_plan


# manifest_eval loads ALL scheduled policies up front and keeps them resident (see
# DESIGN.md). Warn when free VRAM after a load drops below this margin so a heavy
# mixed manifest surfaces the OOM risk loudly instead of via a cryptic CUDA error.
_MIN_FREE_GB_AFTER_POLICY_LOAD = 6.0

# Per-episode footering writes one episode-meta parquet per seal; merge them back
# into a single file this often so each reopen's metadata load stays cheap even
# when checkpoint_interval_rounds=0 (the merge is the only thing that bounds file
# count). A bound of one round's worth of episodes keeps the reopen near-O(1).
_SEAL_CONSOLIDATE_EVERY = 12


def _load_scheduled_policies(
    *,
    policy_map: dict[str, PolicyMapEntry],
    num_action_samples_by_slot_key: dict[str, int],
    device: str,
    noise_scheduler: str | None,
    num_inference_steps: int | None,
    camera_height: int,
    camera_width: int,
    n_action_steps: int,
    num_action_samples: int | None,
    load_log_path: Path,
    skip_model_ids: frozenset[str] = frozenset(),
    dp_override_by_slot_key: dict[str, str] | None = None,
    policy_loader=None,
) -> dict[str, PolicyEntry]:
    loaded: dict[str, PolicyEntry] = {}
    # Slots share one policy object per (model_id, DP override). Their num_action_samples
    # is re-applied before every rollout.
    cached_by_load_key: dict[tuple[str, str | None], PolicyEntry] = {}
    checkpoint_samples_by_load_key: dict[tuple[str, str | None], int | None] = {}
    slot_keys_by_load_key: dict[tuple[str, str | None], list[str]] = {}
    loader = policy_loader or load_policy_by_model_id
    scheduled = [
        item
        for item in sorted(policy_map.items(), key=lambda item: item[1].policy_id)
        if item[1].model_id not in skip_model_ids
    ]
    print(f"Loading {len(scheduled)} scheduled policy/policies before rollout...")
    load_log_path.parent.mkdir(parents=True, exist_ok=True)
    load_log_path.write_text("")
    for load_idx, (slot_key, map_entry) in enumerate(scheduled, start=1):
        model_id = map_entry.model_id
        policy_id = map_entry.policy_id
        dp_override = (dp_override_by_slot_key or {}).get(slot_key)
        load_key = (model_id, dp_override)
        print(f"  Loading scheduled policy {load_idx}/{len(scheduled)}")
        previous_disable_level = logging.root.manager.disable
        with load_log_path.open("a") as load_log:
            print(f"\n=== scheduled policy {load_idx}/{len(scheduled)} ===", file=load_log)
            print(f"slot_key={slot_key}", file=load_log)
            print(f"model_id={model_id}", file=load_log)
            try:
                if load_key in cached_by_load_key:
                    cached = cached_by_load_key[load_key]
                    entry = PolicyEntry(
                        model_id=model_id,
                        policy_id=policy_id,
                        policy=cached.policy,
                        camera_height=cached.camera_height,
                        camera_width=cached.camera_width,
                    )
                    print(f"  Reusing already-loaded policy object for duplicate slot {slot_key}")
                else:
                    logging.disable(logging.INFO)
                    with contextlib.redirect_stdout(load_log), contextlib.redirect_stderr(load_log):
                        entry = loader(
                            model_id=model_id,
                            policy_id=policy_id,
                            device=device,
                            noise_scheduler=noise_scheduler,
                            num_inference_steps=num_inference_steps,
                            default_camera_height=camera_height,
                            default_camera_width=camera_width,
                            n_action_steps=n_action_steps,
                            dp_artifact_override=dp_override,
                        )
                    cached_by_load_key[load_key] = entry
                    checkpoint_samples_by_load_key[load_key] = getattr(
                        entry.policy, "num_action_samples", None
                    )
                slot_keys_by_load_key.setdefault(load_key, []).append(slot_key)
                _maybe_override_idql_samples(
                    entry,
                    num_action_samples_by_slot_key.get(slot_key, num_action_samples),
                )
            except Exception:
                print(f"Policy loading failed; full log: {load_log_path}")
                _print_file_tail(load_log_path, line_count=50)
                raise
            finally:
                logging.disable(previous_disable_level)
        loaded[slot_key] = entry
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if free_gb < _MIN_FREE_GB_AFTER_POLICY_LOAD:
                print(
                    f"  WARNING: only {free_gb:.1f} GB GPU free after loading "
                    f"{load_idx}/{len(scheduled)} scheduled policies. manifest_eval keeps "
                    f"ALL policies resident; a large mixed DP/IDQL manifest can OOM. "
                    f"Split heavy manifests across separate runs.",
                    flush=True,
                )
    # A shared object keeps the count of the slot configured last. Record every sharing
    # slot's own count (its override, else --num-action-samples, else the checkpoint's)
    # for the per-rollout re-apply.
    for load_key, slot_keys in slot_keys_by_load_key.items():
        if len(slot_keys) < 2:
            continue
        for slot_key in slot_keys:
            count = num_action_samples_by_slot_key.get(slot_key, num_action_samples)
            if count is None:
                count = checkpoint_samples_by_load_key[load_key]
            if count is not None:
                num_action_samples_by_slot_key[slot_key] = count
    print(f"All scheduled policies loaded. Detailed load log: {load_log_path}")
    return loaded


def _print_file_tail(path: Path, *, line_count: int) -> None:
    lines = path.read_text(errors="replace").splitlines()
    print(f"Last {min(line_count, len(lines))} line(s) from {path}:")
    for line in lines[-line_count:]:
        print(line)


def _remaining_from_plan(
    *,
    plan: dict,
    round_num: int,
    completed_records: list[RolloutRecord],
    slots_per_round: int,
    dropped_policy_ids: frozenset[int] = frozenset(),
) -> list[dict]:
    if int(plan["round"]) != round_num:
        raise RuntimeError(f"Round plan has round={plan['round']}, expected {round_num}")
    policy_order = plan.get("policy_order")
    if not isinstance(policy_order, list) or len(policy_order) != slots_per_round:
        raise RuntimeError(
            f"Round {round_num} plan has {None if policy_order is None else len(policy_order)} "
            f"slot(s), expected {slots_per_round}"
        )
    completed_by_policy = {record.policy_id: record for record in completed_records}
    if len(completed_by_policy) != len(completed_records):
        raise RuntimeError(f"Round {round_num} has duplicate completed policy IDs")
    remaining = []
    for entry in sorted(policy_order, key=lambda row: int(row["slot_idx"])):
        policy_id = int(entry["policy_id"])
        if policy_id in dropped_policy_ids:
            # Arm retired on resume (--drop-fixed-policy): never roll out, even if
            # the pre-baked plan still schedules it. Earlier-round records for it
            # are preserved; pending rounds simply skip its slot.
            continue
        completed = completed_by_policy.get(policy_id)
        if completed is not None:
            if completed.model_id != entry["model_id"]:
                raise RuntimeError(
                    f"Round {round_num} completed policy_id={policy_id} has "
                    f"{completed.model_id}, but plan has {entry['model_id']}"
                )
            if completed.anonymous_label not in {entry["anonymous_label"], "?"}:
                raise RuntimeError(
                    f"Round {round_num} completed policy_id={policy_id} used label "
                    f"{completed.anonymous_label!r}, but plan has {entry['anonymous_label']!r}"
                )
            continue
        remaining.append(entry)
    return remaining


def _complete_round_numbers(
    records: list[RolloutRecord],
    *,
    slots_per_round: int,
    dropped_policy_ids: frozenset[int] = frozenset(),
) -> set[int]:
    effective_slots = slots_per_round - len(dropped_policy_ids)
    by_round: dict[int, list[RolloutRecord]] = {}
    for record in records:
        by_round.setdefault(record.round_num, []).append(record)
    complete: set[int] = set()
    for round_num, round_records in by_round.items():
        if len({record.policy_id for record in round_records}) != len(round_records):
            raise RuntimeError(f"Round {round_num} duplicates a policy record")
        active = [r for r in round_records if r.policy_id not in dropped_policy_ids]
        if len(active) == effective_slots:
            complete.add(round_num)
        elif len(active) > effective_slots:
            raise RuntimeError(
                f"Round {round_num} has {len(active)} active records for "
                f"{effective_slots} surviving slots"
            )
    return complete


def _entries_and_names(
    *,
    records: list[RolloutRecord],
    policy_map: dict[str, PolicyMapEntry],
) -> tuple[list[PolicyEntry], list[str]]:
    results_by_policy: dict[int, list[bool]] = {}
    for record in records:
        results_by_policy.setdefault(record.policy_id, []).append(record.outcome == "success")
    entries: list[PolicyEntry] = []
    names: list[str] = []
    for _slot_key, map_entry in sorted(policy_map.items(), key=lambda row: row[1].policy_id):
        entries.append(
            PolicyEntry(
                model_id=map_entry.model_id,
                policy_id=map_entry.policy_id,
                policy=None,
                results=results_by_policy.get(map_entry.policy_id, []),
            )
        )
        names.append(map_entry.name)
    return entries, names


def _maybe_override_idql_samples(entry: PolicyEntry, num_action_samples: int | None) -> None:
    if num_action_samples is None:
        return
    # BLINDING: this runs right before every anonymous rollout header, so it must not
    # print -- a "num_action_samples -> 32" line would tell the operator which anonymous
    # label is the DP+IQL arm. Configuration is logged at DEBUG only; failures still raise.
    if hasattr(entry.policy, "set_num_action_samples"):
        changed = entry.policy.set_num_action_samples(num_action_samples)
        if changed:
            logging.getLogger(__name__).debug("remote num_action_samples -> %s", num_action_samples)
        return
    from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

    if isinstance(entry.policy, VisionIDQLRealWorldPolicy):
        entry.policy.num_action_samples = num_action_samples
        logging.getLogger(__name__).debug("num_action_samples -> %s", num_action_samples)


def resolve_hf_repo_id(args: argparse.Namespace, dataset_name: str) -> str:
    """Repo id of the eval dataset: ``--hf-repo-id``, else ``<HF user>/<dataset name>``.

    With ``--no-push`` the id only names the local LeRobot dataset, so no HF login is needed.
    """
    if args.hf_repo_id:
        return args.hf_repo_id
    if args.no_push:
        return f"local/{dataset_name}"
    from huggingface_hub import HfApi

    return f"{HfApi().whoami()['name']}/{dataset_name}"


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    fixed_specs = [_parse_fixed_policy(value) for value in args.fixed_policy]
    fixed_names = [spec.name for spec in fixed_specs]
    if len(set(fixed_names)) != len(fixed_names):
        raise ValueError(f"--fixed-policy labels must be unique, got {fixed_names}")
    dropped_names = set(args.drop_fixed_policy)
    unknown_dropped_names = sorted(dropped_names - set(fixed_names))
    if unknown_dropped_names:
        raise ValueError(
            "--drop-fixed-policy references label(s) not present in --fixed-policy: "
            f"{unknown_dropped_names}. Keep passing the retired arm's --fixed-policy spec "
            "so completed-round policy_id mapping stays consistent."
        )
    # Fixed-policy slot index IS its policy_id (slot_key=f"fixed:{idx}"), so the
    # dropped policy_ids are exactly the indices of the dropped fixed names.
    dropped_policy_ids = frozenset(
        idx for idx, spec in enumerate(fixed_specs) if spec.name in dropped_names
    )
    fixed_sample_overrides_by_name = dict(
        _parse_fixed_policy_num_action_samples(value)
        for value in args.fixed_policy_num_action_samples
    )
    unknown_sample_override_names = sorted(set(fixed_sample_overrides_by_name) - set(fixed_names))
    if unknown_sample_override_names:
        raise ValueError(
            "--fixed-policy-num-action-samples references unknown fixed-policy label(s): "
            f"{unknown_sample_override_names}"
        )
    fixed_dp_overrides_by_name = dict(
        _parse_fixed_policy_dp_override(value) for value in args.fixed_policy_dp_override
    )
    unknown_dp_override_names = sorted(set(fixed_dp_overrides_by_name) - set(fixed_names))
    if unknown_dp_override_names:
        raise ValueError(
            "--fixed-policy-dp-override references unknown fixed-policy label(s): "
            f"{unknown_dp_override_names}"
        )
    if len(fixed_specs) < 2:
        raise ValueError("Provide at least two --fixed-policy arms for a paired blind eval")
    if args.max_recoverable_rollout_errors_per_slot < 0:
        raise ValueError("--max-recoverable-rollout-errors-per-slot must be non-negative")
    if args.recoverable_rollout_retry_delay_s < 0:
        raise ValueError("--recoverable-rollout-retry-delay-s must be non-negative")
    if args.recoverable_rollout_retry_backoff < 1:
        raise ValueError("--recoverable-rollout-retry-backoff must be >= 1")
    if args.recoverable_rollout_max_retry_delay_s < 0:
        raise ValueError("--recoverable-rollout-max-retry-delay-s must be non-negative")
    if args.checkpoint_interval_rounds < 0:
        raise ValueError("--checkpoint-interval-rounds must be non-negative")
    if args.reset_max_retries < 1:
        raise ValueError("--reset-max-retries must be >= 1")
    if args.reset_retry_delay_s < 0:
        raise ValueError("--reset-retry-delay-s must be non-negative")
    if args.reset_retry_backoff < 1:
        raise ValueError("--reset-retry-backoff must be >= 1")
    if args.reset_max_retry_delay_s < 0:
        raise ValueError("--reset-max-retry-delay-s must be non-negative")
    if args.robot_state_refresh_max_wait_s < 0:
        raise ValueError("--robot-state-refresh-max-wait-s must be non-negative")
    if args.robot_state_refresh_poll_interval_s <= 0:
        raise ValueError("--robot-state-refresh-poll-interval-s must be > 0")

    device = args.device or auto_detect_device()
    requested_camera_keys = parse_camera_keys(args.camera_keys, args.camera_filter)
    remote_inference = remote_inference_session_from_args(args, default_workdir=Path.cwd())
    policy_loader = remote_inference.load_policy_entry if remote_inference is not None else None
    dataset_name = args.dataset_name or f"manifest-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # One id per launch, stamped on every rollout record this invocation collects (durable
    # visit provenance for phased / resumed evals; see RolloutRecord.visit_id).
    visit_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    print(f"visit_id: {visit_id}")
    ui = OperatorUI.from_args(
        args, cards=True, default_card_dir=Path("/tmp") / f"{dataset_name}_initial_state_targets"
    )
    dataset_path = Path(args.dataset_path) / dataset_name
    results_path = Path(args.results_path) if args.results_path else dataset_path / "results.json"
    num_subtask_marks = num_subtask_marks_for_task(args.environment)
    if num_subtask_marks > 0:
        print(
            f"Task {args.environment} defines {num_subtask_marks} mid-episode sub-goal mark(s): "
            f"press 'g'/numpad'3' when each sub-goal is reached (graded score max "
            f"{num_subtask_marks + 1}/episode)."
        )
    slots_per_round = len(fixed_specs)
    effective_slots_per_round = slots_per_round - len(dropped_policy_ids)
    if dropped_policy_ids and effective_slots_per_round < 2:
        raise ValueError(
            f"--drop-fixed-policy would leave {effective_slots_per_round} surviving slot(s); "
            "the paired eval needs at least 2 arms per round."
        )
    if dropped_policy_ids:
        print(
            f"Retiring {len(dropped_policy_ids)} arm(s) on resume "
            f"(--drop-fixed-policy {sorted(dropped_names)}): policy_id(s) "
            f"{sorted(dropped_policy_ids)}; {effective_slots_per_round} surviving slot(s)/round."
        )

    manifest_targets, manifest_meta = _load_eval_initial_state_manifest(
        args.initial_states_manifest,
        expected_task=args.environment,
    )
    previous_records, previous_policy_map = load_previous_results(results_path)
    round_plans = _load_round_plans(results_path)
    phase_stops = _load_phase_stops(results_path)
    round_plans_by_round = _round_plans_by_round(round_plans)
    pending_round_nums, completed_records_by_round = _pending_manifest_rounds(
        previous_records,
        num_rounds=len(manifest_targets),
        slots_per_round=slots_per_round,
        rerun_incomplete=args.rerun_incomplete_rounds,
        dropped_policy_ids=dropped_policy_ids,
    )
    all_pending_round_nums = list(pending_round_nums)
    pending_round_nums = _apply_round_cap(pending_round_nums, args.stop_after_round)
    if args.stop_after_round and len(pending_round_nums) != len(all_pending_round_nums):
        print(
            f"--stop-after-round {args.stop_after_round}: {len(pending_round_nums)} of "
            f"{len(all_pending_round_nums)} pending round(s) are eligible in this invocation."
        )
    if args.stop_after_round and not pending_round_nums and all_pending_round_nums:
        print(
            f"--stop-after-round {args.stop_after_round}: every pending round is above the cap; "
            "nothing to collect in this invocation (no policy loading, no robot motion)."
        )
        return
    if previous_records:
        print(
            f"Loaded {len(previous_records)} previous rollouts "
            f"({len(manifest_targets) - len(pending_round_nums)} complete rounds) "
            f"from {results_path}"
        )

    fixed_policies: list[ResolvedPolicy] = []
    policy_map: dict[str, PolicyMapEntry] = {}
    num_action_samples_by_slot_key: dict[str, int] = {}
    dp_override_by_slot_key: dict[str, str] = {}
    resolved_fixed_specs = [(spec, _normalize_model_id(spec.spec)) for spec in fixed_specs]
    # Model ids of retired arms (resolved from the kept --fixed-policy specs): skip
    # loading their weights even though they remain in policy_map for naming/history.
    dropped_model_ids = frozenset(resolved_fixed_specs[idx][1] for idx in dropped_policy_ids)
    # _load_scheduled_policies skips by model_id, so a surviving slot sharing a dropped
    # arm's model_id would be wrongly skipped (and later fail at loaded_policies[...]).
    # That can only happen with a duplicate-model eval; fail loudly at startup if so.
    surviving_dropped_models = sorted(
        {
            resolved_fixed_specs[idx][1]
            for idx in range(len(resolved_fixed_specs))
            if idx not in dropped_policy_ids and resolved_fixed_specs[idx][1] in dropped_model_ids
        }
    )
    if surviving_dropped_models:
        raise ValueError(
            "--drop-fixed-policy would also retire a surviving slot sharing the same "
            f"model_id(s): {surviving_dropped_models}. Drop by an unambiguous arm or give the "
            "duplicate-model arms distinct checkpoints."
        )
    fixed_model_counts: dict[str, int] = {}
    for _spec, model_id in resolved_fixed_specs:
        fixed_model_counts[model_id] = fixed_model_counts.get(model_id, 0) + 1
    for idx, (spec, model_id) in enumerate(resolved_fixed_specs):
        name = spec.name
        slot_key = f"fixed:{idx}"
        if model_id in previous_policy_map:
            previous_policy_id, previous_name = previous_policy_map[model_id]
            if fixed_model_counts[model_id] == 1 and previous_policy_id != idx:
                raise RuntimeError(
                    f"Fixed policy {name!r} previously used policy_id={previous_policy_id}, "
                    f"but current fixed slot is {idx}. Use the same --fixed-policy order or "
                    "a new DATASET_NAME."
                )
            if fixed_model_counts[model_id] == 1 and previous_name and previous_name != name:
                raise RuntimeError(
                    f"Fixed policy {model_id} was previously named {previous_name!r}, "
                    f"current name is {name!r}. Use the same name or a new DATASET_NAME."
                )
        policy_map[slot_key] = PolicyMapEntry(policy_id=idx, name=name, model_id=model_id)
        if name in fixed_sample_overrides_by_name:
            num_action_samples_by_slot_key[slot_key] = fixed_sample_overrides_by_name[name]
        if name in fixed_dp_overrides_by_name:
            dp_override_by_slot_key[slot_key] = fixed_dp_overrides_by_name[name]
        fixed_policies.append(
            ResolvedPolicy(
                name=name,
                model_id=model_id,
                slot_key=slot_key,
                slot_type="fixed",
            )
        )

    for model_id, (policy_id, name) in previous_policy_map.items():
        if any(entry.policy_id == policy_id for entry in policy_map.values()):
            continue
        policy_map.setdefault(
            model_id,
            PolicyMapEntry(policy_id=policy_id, name=name, model_id=model_id),
        )

    _ensure_all_round_plans(
        manifest_targets=manifest_targets,
        round_plans=round_plans,
        round_plans_by_round=round_plans_by_round,
        fixed_policies=fixed_policies,
        policy_map=policy_map,
        random_seed=args.random_seed,
    )

    hf_repo_id = resolve_hf_repo_id(args, dataset_name)
    from huggingface_hub.utils import validate_repo_id

    # Validate before the target window, policy loading, or robot initialization.
    # Otherwise an invalid ID is discovered only during the final Hub push, after
    # physical rollouts have already been collected.
    validate_repo_id(hf_repo_id)

    cleanup_stale_image_episode_dirs(dataset_path)

    all_entries_so_far, all_names_so_far = _entries_and_names(
        records=previous_records,
        policy_map=policy_map,
    )
    _save_results_file_quiet(
        results_path,
        all_entries_so_far,
        previous_records,
        args,
        dataset_name,
        policy_names=all_names_so_far,
        round_plans=round_plans,
        phase_stops=phase_stops,
        quiet=True,
    )

    progress_slots = slots_per_round if args.progress_total_all_arms else effective_slots_per_round
    ui.configure_progress(
        total=len(manifest_targets) * progress_slots,
        completed=(
            len(previous_records)
            if args.progress_total_all_arms
            else sum(r.policy_id not in dropped_policy_ids for r in previous_records)
        ),
    )

    def eval_scene(round_num: int, planned: dict) -> EvalScene:
        active = [
            p
            for p in sorted(
                round_plans_by_round[round_num]["policy_order"], key=lambda p: p["slot_idx"]
            )
            if int(p["policy_id"]) not in dropped_policy_ids
        ]
        return EvalScene(
            round_num,
            len(manifest_targets),
            planned["anonymous_label"],
            active.index(planned) + 1,
            len(active),
            manifest_targets[round_num - 1],
        )

    def first_pending_scene(next_round_num: int | None) -> EvalScene | None:
        if next_round_num is None:
            return None
        next_remaining = _remaining_from_plan(
            plan=round_plans_by_round[next_round_num],
            round_num=next_round_num,
            completed_records=completed_records_by_round.get(next_round_num, []),
            slots_per_round=slots_per_round,
            dropped_policy_ids=dropped_policy_ids,
        )
        return eval_scene(next_round_num, next_remaining[0])

    if pending_round_nums:
        ui.show_eval_scene(
            first_pending_scene(pending_round_nums[0]), manifest_meta, task_name=args.environment
        )
        print("First target shown before policy loading and robot initialization.")

    print(
        f"[protocol] Effective n_action_steps (exec horizon) = {args.n_action_steps} "
        f"(real-robot protocol default = {REAL_PROTOCOL_N_ACTION_STEPS}; "
        "prediction horizon is checkpoint-specific)."
    )
    print(f"[protocol] Inference backend = {args.inference_backend}")
    loaded_policies = _load_scheduled_policies(
        policy_map=policy_map,
        num_action_samples_by_slot_key=num_action_samples_by_slot_key,
        dp_override_by_slot_key=dp_override_by_slot_key,
        device=device,
        noise_scheduler=args.noise_scheduler,
        num_inference_steps=args.num_inference_steps,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        n_action_steps=args.n_action_steps,
        num_action_samples=args.num_action_samples,
        load_log_path=dataset_path / "policy_loading.log",
        skip_model_ids=dropped_model_ids,
        policy_loader=policy_loader,
    )
    # Blinded session: the monitor may show the policies' own crops only if they all agree.
    if ui.monitor_camera_keys:
        ui.monitor_crop_boxes = shared_policy_crop_boxes(
            [entry.policy for entry in loaded_policies.values()]
        )

    if requested_camera_keys is not None:
        requested_serials = camera_serials_from_keys(requested_camera_keys)
        print(f"Restricting ZED cameras to serials: {requested_serials}")
        restrict_zed_cameras_to_serials(requested_serials)

    import mulligan.real.robot.camera_config  # noqa: F401
    from mulligan.real.robot.droid_compat import RobotEnv

    print("=" * 72)
    print("MANIFEST EVAL")
    print("=" * 72)
    print(f"Environment: {args.environment}")
    print(f"Dataset: {dataset_name}")
    print(f"Manifest states: {len(manifest_targets)}")
    if dropped_policy_ids:
        print(
            f"Policies per round: {effective_slots_per_round} surviving "
            f"({slots_per_round} scheduled, {len(dropped_policy_ids)} retired)"
        )
    else:
        print(f"Policies per round: {slots_per_round}")
    print(f"Device: {device}")
    print(f"Background save log: {dataset_path / 'background_save.log'}")
    print()

    print("\nInitializing robot environment...")
    env = RobotEnv(action_space="cartesian_velocity")
    ui.observe = env.get_observation
    print("Initializing keyboard listener...")
    _ = ui.keyboard  # enter cbreak mode now, before the first prompt

    def reset_robot() -> None:
        verified_reset(
            env,
            max_retries=args.reset_max_retries,
            retry_delay_s=args.reset_retry_delay_s,
            retry_backoff=args.reset_retry_backoff,
            max_retry_delay_s=args.reset_max_retry_delay_s,
        )

    ui.set_phase("Resetting robot", "First setup is ready. Keep clear of the arm during reset.")
    print("Performing initial robot reset...")
    verified_reset(
        env,
        max_retries=args.reset_max_retries,
        retry_delay_s=args.reset_retry_delay_s,
        retry_backoff=args.reset_retry_backoff,
        max_retry_delay_s=args.reset_max_retry_delay_s,
    )
    print("Robot reset complete.")

    # Concurrency model (see mulligan/real/collect/seal_pipeline.py): each completed episode's
    # seal chain (save_episode -> parquet footer via checkpoint_dataset ->
    # rollout_records bookkeeping -> results.json rewrite) runs on ONE serial
    # background worker so it overlaps the NEXT rollout instead of blocking the
    # operator. Shared-state discipline:
    #   * `dataset` / `saved_episode_count` / `episodes_since_consolidate` are
    #     mutated ONLY inside seal chains (serial executor), except on the main
    #     thread when provably no chain is in flight (first-episode create, and
    #     after a blocking seal_pipeline.drain()). The main thread's bare
    #     `dataset is None` checks are safe: chains only ever replace one
    #     non-None dataset object with another.
    #   * `rollout_records` appends and every results.json write take
    #     `records_lock`; main-thread readers (progress/ETA prints) snapshot
    #     under the lock and tolerate lagging one in-flight episode.
    #   * A record enters `rollout_records`/results.json only INSIDE a chain,
    #     AFTER its episode's parquet footer succeeded (run_seal_chain ordering),
    #     so results.json can never reference an episode that is not durable.
    dataset = None
    saved_episode_count = 0
    episodes_since_consolidate = 0
    seal_pipeline = SealPipeline()
    records_lock = threading.Lock()
    all_camera_keys: list[str] = []
    rollout_records: list[RolloutRecord] = []
    background_save_log_path = dataset_path / "background_save.log"
    # Headroom/drain timeout: must cover a full seal chain started from scratch
    # (streaming-encoder drain of a whole episode + parquet footer + occasional
    # consolidate + results.json), not just the streaming-encoder slack.
    seal_wait_timeout_s = 240

    def print_manifest_progress(prefix: str) -> None:
        with records_lock:
            all_records = previous_records + list(rollout_records)
        complete_round_nums = _complete_round_numbers(
            all_records,
            slots_per_round=slots_per_round,
            dropped_policy_ids=dropped_policy_ids,
        )
        if args.progress_total_all_arms:
            completed_rollouts = len(all_records)
            total_rollouts = len(manifest_targets) * slots_per_round
        else:
            completed_rollouts = len(
                [r for r in all_records if r.policy_id not in dropped_policy_ids]
            )
            total_rollouts = len(manifest_targets) * effective_slots_per_round
        session_rollouts = len(all_records) - len(previous_records)
        elapsed = ui.progress.elapsed
        rate = ui.progress.session_completed / elapsed if elapsed > 0 else None
        eta = ui.progress.eta
        rate_text = f"{rate * 60:.1f} rollouts/min" if rate else "unknown"

        print()
        print(prefix)
        rounds_note = " (under active arms)" if dropped_policy_ids else ""
        print(
            f"  Rounds complete: {len(complete_round_nums)}/{len(manifest_targets)}{rounds_note}  "
            f"Rollouts saved: {completed_rollouts}/{total_rollouts}"
        )
        print(
            f"  This launch: {session_rollouts} saved in {_format_eta(elapsed)}  "
            f"Rate: {rate_text}  ETA: {_format_eta(eta)}"
        )
        if seal_pipeline.in_flight:
            print("  (a background episode seal is in flight; counts may lag by one)")

    def _write_results_snapshot() -> None:
        """Rewrite results.json from the current records (thread-safe).

        Called from BOTH the main thread (round boundaries, unsaved-rollout
        records) and background seal chains; `records_lock` serializes the
        snapshot + file write so two writers can never interleave on the file.
        """
        with records_lock:
            all_records_now = previous_records + list(rollout_records)
            all_entries_now, all_names_now = _entries_and_names(
                records=all_records_now,
                policy_map=policy_map,
            )
            _save_results_file_quiet(
                results_path,
                all_entries_now,
                all_records_now,
                args,
                dataset_name,
                policy_names=all_names_now,
                round_plans=round_plans,
                phase_stops=phase_stops,
                quiet=True,
            )

    def _append_rollout_record(record: RolloutRecord) -> None:
        with records_lock:
            rollout_records.append(record)

    def checkpoint_due_after_round(completed_round_num: int) -> bool:
        interval = args.checkpoint_interval_rounds
        return interval > 0 and completed_round_num % interval == 0

    def recoverable_rollout_retry_delay_s(error_count: int) -> float:
        if args.recoverable_rollout_retry_delay_s <= 0:
            return 0.0
        delay = args.recoverable_rollout_retry_delay_s * (
            args.recoverable_rollout_retry_backoff ** max(0, error_count - 1)
        )
        return min(args.recoverable_rollout_max_retry_delay_s, delay)

    cap_triggered = False
    try:
        print_manifest_progress("MANIFEST PROGRESS")
        for pending_pos, round_num in enumerate(pending_round_nums):
            seal_pipeline.poll_completed(description="Completed episode background seal")
            target_idx = round_num - 1
            target = manifest_targets[target_idx]
            next_round_num = (
                pending_round_nums[pending_pos + 1]
                if pending_pos + 1 < len(pending_round_nums)
                else None
            )
            completed_for_round = completed_records_by_round.get(round_num, [])

            print(f"\nROUND {round_num}/{len(manifest_targets)}")
            print_manifest_progress("Progress before round")

            round_plan = round_plans_by_round.get(round_num)
            if round_plan is None:
                raise RuntimeError(f"Missing precomputed round plan for round {round_num}")

            _write_results_snapshot()

            # NB: `remaining` comes from `completed_records_by_round`, which is
            # computed ONCE at startup from results.json (previous sessions only)
            # -- it never reads this session's `rollout_records`, so no seal-chain
            # drain is needed before re-planning a round.
            remaining = _remaining_from_plan(
                plan=round_plan,
                round_num=round_num,
                completed_records=completed_for_round,
                slots_per_round=slots_per_round,
                dropped_policy_ids=dropped_policy_ids,
            )
            ui.show_eval_scene(
                eval_scene(round_num, remaining[0]), manifest_meta, task_name=args.environment
            )

            slot_labels = [
                f"Policy {planned['anonymous_label']}"
                for planned in sorted(round_plan["policy_order"], key=lambda row: row["slot_idx"])
                if int(planned["policy_id"]) not in dropped_policy_ids
            ]
            print(f"Anonymous rollout slots: {', '.join(slot_labels)}")
            wait_until_ready(ui, prompt="Set up the target", on_reset=reset_robot)

            for rollout_idx, planned in enumerate(remaining):
                # DURABILITY CONTRACT: the
                # previous episode's seal chain keeps running in the background
                # THROUGH this rollout's reset + execution -- we rely on the
                # streaming encoder and only block at submit time if the previous
                # chain is still unfinished (SealPipeline headroom wait). A crash
                # during this rollout/reset can therefore lose the PREVIOUS
                # completed episode (un-footered -> quarantined by
                # reconcile_resumed_dataset; or unsaved -> simply missing). Either
                # way its record never reached results.json (records are written
                # only after the footer succeeds), so its round shows incomplete
                # on resume and is re-collected.
                label = str(planned["anonymous_label"])
                model_id = str(planned["model_id"])
                slot_key = str(planned.get("slot_key") or model_id)
                policy_id = int(planned["policy_id"])
                entry = loaded_policies[slot_key]
                if entry.policy_id != policy_id:
                    raise RuntimeError(
                        f"Loaded policy slot {slot_key} has policy_id={entry.policy_id}, "
                        f"but round plan uses policy_id={policy_id}"
                    )
                _maybe_override_idql_samples(
                    entry,
                    num_action_samples_by_slot_key.get(slot_key),
                )
                print(f"\n--- Policy {label} (rollout {rollout_idx + 1}/{len(remaining)}) ---")

                recoverable_error_count = 0
                current_scene = eval_scene(round_num, planned)
                upcoming_scene = (
                    eval_scene(round_num, remaining[rollout_idx + 1])
                    if rollout_idx + 1 < len(remaining)
                    else first_pending_scene(next_round_num)
                )

                def preview_before_reset(outcome: str) -> None:
                    ui.preview_after_rollout(
                        outcome,
                        current=current_scene,
                        upcoming=upcoming_scene,
                        manifest_meta=manifest_meta,
                        task_name=args.environment,
                    )

                while True:
                    try:
                        num_steps, outcome, episode_data, subtask_frames = rollout_episode(
                            env,
                            entry.policy,
                            ui,
                            freq=args.freq,
                            camera_height=entry.camera_height,
                            camera_width=entry.camera_width,
                            save_camera_height=args.camera_height,
                            save_camera_width=args.camera_width,
                            camera_filter=args.camera_filter,
                            max_steps=args.max_steps,
                            auto_timeout_at_max_steps=not args.no_auto_timeout,
                            save_data=True,
                            all_camera_keys=all_camera_keys,
                            requested_camera_keys=requested_camera_keys,
                            print_timing_summary=False,
                            reset_max_retries=args.reset_max_retries,
                            reset_retry_delay_s=args.reset_retry_delay_s,
                            reset_retry_backoff=args.reset_retry_backoff,
                            reset_max_retry_delay_s=args.reset_max_retry_delay_s,
                            robot_state_refresh_max_wait_s=args.robot_state_refresh_max_wait_s,
                            robot_state_refresh_poll_interval_s=(
                                args.robot_state_refresh_poll_interval_s
                            ),
                            subtask_marks=num_subtask_marks,
                            pre_reset_callback=preview_before_reset,
                        )
                    except RecoverableRolloutError as exc:
                        recoverable_error_count += 1
                        limit = args.max_recoverable_rollout_errors_per_slot
                        print(
                            f"Policy {label}: recoverable rollout error "
                            f"{recoverable_error_count}/{limit}: {exc}"
                        )
                        print("Discarding the partial rollout and resetting before retry.")
                        # No terminal outcome was recorded. Keep this slot and target;
                        # show the retry helper before recovery motion, even on errors.
                        ui.progress.rollout_started = None
                        try:
                            preview_before_reset("restart")
                        finally:
                            reset_robot()
                        entry.policy.reset()
                        if recoverable_error_count > limit:
                            raise RuntimeError(
                                f"Policy {label} hit {recoverable_error_count} recoverable "
                                "rollout errors for the same slot; robot observation stream "
                                "is not healthy enough to continue safely."
                            ) from exc
                        retry_delay_s = recoverable_rollout_retry_delay_s(recoverable_error_count)
                        if retry_delay_s > 0:
                            print(
                                "Waiting "
                                f"{retry_delay_s:g}s before retrying this anonymous policy..."
                            )
                            time.sleep(retry_delay_s)
                        wait_until_ready(
                            ui,
                            prompt="Reset the scene to retry this anonymous policy",
                            on_reset=reset_robot,
                        )
                        continue
                    if outcome != "restart":
                        break
                    print(f"Policy {label}: RESTART requested ({num_steps} steps discarded).")
                    wait_until_ready(
                        ui,
                        prompt="Reset the scene to retry this anonymous policy",
                        on_reset=reset_robot,
                    )

                if outcome == "quit":
                    raise KeyboardInterrupt

                is_success = outcome == "success"
                print(
                    rollout_outcome_line(
                        label, outcome, num_steps, subtask_frames, num_subtask_marks
                    )
                )

                # ---- Persist per-chunk IDQL candidate diagnostics (sidecar) --
                # Mirrors the sidecar block of mulligan.real.collect.rollout:
                # manifest_eval calls rollout_episode() directly and bypasses that
                # loop. Keyed by round + anonymous label (no policy
                # identity beyond what results.json already records). Loud-warn
                # only: a logging failure must not kill a robot episode.
                chunk_infos = getattr(entry.policy, "last_episode_chunk_infos", None)
                if chunk_infos:
                    try:
                        # Inside the dataset dir: the hub push carries extra
                        # files (results.json, *.log already ride along), so
                        # sidecars persist to the HF repo with zero extra steps.
                        sidecar_root = dataset_path / "chunk_info"
                        sidecar_root.mkdir(parents=True, exist_ok=True)
                        sidecar_path = sidecar_root / f"round_{round_num:04d}_{label}.jsonl"
                        with open(sidecar_path, "w") as sidecar_f:
                            header = {
                                "round_num": round_num,
                                "anonymous_label": label,
                                "outcome": outcome,
                                "num_steps": num_steps,
                                "num_chunks": len(chunk_infos),
                            }
                            sidecar_f.write(json.dumps(header) + "\n")
                            for info in chunk_infos:
                                sidecar_f.write(json.dumps(info) + "\n")
                    except Exception as exc:  # noqa: BLE001
                        print(f"WARNING: failed to write chunk-info sidecar: {exc!r}")
                    entry.policy.last_episode_chunk_infos = []

                episode_index = -1
                if episode_data is not None and num_steps > 0:
                    episode_length = len(episode_data["actions"])
                    episode_data["steps_to_go"] = [
                        episode_length - 1 - j for j in range(episode_length)
                    ]
                    cam_data_keys = sorted(k for k in episode_data if k.startswith("image_"))
                    if dataset is None:
                        dataset_meta_file = dataset_path / "meta" / "info.json"
                        if dataset_meta_file.exists():
                            removed_camera_features = (
                                remove_excluded_camera_features_from_lerobot_dataset(dataset_path)
                            )
                            if removed_camera_features:
                                print(
                                    "Removed excluded camera feature(s) from existing dataset: "
                                    f"{removed_camera_features}"
                                )
                            print(f"Loading existing dataset from {dataset_path}")
                            dataset = reopen_dataset_for_append(
                                hf_repo_id,
                                dataset_path,
                                episode_data,
                                cam_data_keys=cam_data_keys,
                            )
                            saved_episode_count = dataset.num_episodes
                            print(f"Loaded existing dataset with {saved_episode_count} episodes")
                        else:
                            dataset = create_dataset(
                                dataset_path=dataset_path,
                                hf_repo_id=hf_repo_id,
                                episode_data=episode_data,
                                cam_data_keys=cam_data_keys,
                                freq=args.freq,
                                extra_features=_eval_initial_state_features(manifest_meta),
                            )
                        manifest_snapshot_path = _copy_eval_initial_state_manifest(
                            args.initial_states_manifest,
                            dataset_path,
                        )
                        print(f"Dataset manifest snapshot: {manifest_snapshot_path}")

                    # The ENTIRE seal chain runs in the background: save_episode
                    # (streaming-encoder drain) -> parquet footer (finalize+reopen,
                    # consolidate every _SEAL_CONSOLIDATE_EVERY seals to bound
                    # per-episode meta-file growth) -> rollout_records/results.json.
                    # Per-episode values are bound as defaults at submit time;
                    # `dataset`/`saved_episode_count`/`episodes_since_consolidate`
                    # are read via nonlocal at CHAIN RUN time, which is safe because
                    # submit_chain drains the previous chain first (serial executor
                    # => the previous chain's checkpoint_dataset handoff is visible).
                    def _seal_episode_chain(
                        *,
                        episode_data_value=episode_data,
                        episode_success_value=is_success,
                        episode_camera_keys=cam_data_keys,
                        episode_policy_id=policy_id,
                        episode_round_num=round_num,
                        episode_task=args.environment,
                        episode_target=target,
                        episode_model_id=model_id,
                        episode_label=label,
                        episode_outcome=outcome,
                        episode_num_steps=num_steps,
                        episode_subtask_frames=tuple(subtask_frames),
                    ) -> None:
                        nonlocal dataset, saved_episode_count, episodes_since_consolidate
                        timings: dict[str, float] = {}
                        chain_t0 = time.time()
                        chain_state = {"episode_index": -1, "consolidated": False}

                        def _save() -> None:
                            chain_state["episode_index"] = saved_episode_count
                            with _quiet_background_save_context(background_save_log_path):
                                save_episode_to_dataset(
                                    dataset=dataset,
                                    episode_data=episode_data_value,
                                    episode_success=episode_success_value,
                                    camera_keys=episode_camera_keys,
                                    policy_id=episode_policy_id,
                                    round_num=episode_round_num,
                                    task=episode_task,
                                    extra_frame_fields=_eval_initial_state_frame_fields(
                                        episode_target,
                                        manifest_meta,
                                    ),
                                    verbose=False,
                                )
                            timings["save"] = time.time() - chain_t0

                        def _footer() -> None:
                            # Footer the just-saved episode's parquet via a cheap
                            # finalize+reopen so info.json's episode counter stays in
                            # lockstep with the durably-footered data on disk.
                            nonlocal dataset, saved_episode_count
                            nonlocal episodes_since_consolidate
                            footer_t0 = time.time()
                            episodes_since_consolidate += 1
                            do_consolidate = episodes_since_consolidate >= _SEAL_CONSOLIDATE_EVERY
                            if do_consolidate:
                                episodes_since_consolidate = 0
                            chain_state["consolidated"] = do_consolidate
                            dataset, saved_episode_count = checkpoint_dataset(
                                dataset,
                                None,
                                hf_repo_id,
                                dataset_path,
                                verbose=False,
                                consolidate=do_consolidate,
                            )
                            timings["footer"] = time.time() - footer_t0

                        def _record() -> None:
                            # Runs ONLY after the footer succeeded (run_seal_chain
                            # ordering): results.json can never list a record whose
                            # episode is not durably on disk.
                            record_t0 = time.time()
                            _append_rollout_record(
                                RolloutRecord(
                                    round_num=episode_round_num,
                                    policy_id=episode_policy_id,
                                    model_id=episode_model_id,
                                    anonymous_label=episode_label,
                                    outcome=episode_outcome,
                                    num_steps=episode_num_steps,
                                    episode_index=chain_state["episode_index"],
                                    manifest_idx=episode_target.manifest_idx,
                                    pen_x=episode_target.pen_x,
                                    pen_y=episode_target.pen_y,
                                    pen_yaw=episode_target.pen_yaw,
                                    subtask_frames=episode_subtask_frames,
                                    visit_id=visit_id,
                                )
                            )
                            _write_results_snapshot()
                            timings["results"] = time.time() - record_t0

                        run_seal_chain(save=_save, footer=_footer, record=_record)
                        print(
                            f"  [seal-timing] encode/save {timings['save']:.1f}s | "
                            f"parquet footer+reopen {timings['footer']:.1f}s"
                            f"{' (consolidate)' if chain_state['consolidated'] else ''} | "
                            f"results.json {timings['results']:.1f}s | "
                            f"total {time.time() - chain_t0:.1f}s "
                            f"({saved_episode_count} eps on disk)"
                        )

                    # Headroom wait: at most ONE seal chain in flight (a buffered
                    # episode holds all camera frames in RAM); blocks only if the
                    # PREVIOUS chain is still running.
                    seal_pipeline.submit_chain(
                        _seal_episode_chain,
                        description="Previous episode background seal",
                        timeout=seal_wait_timeout_s,
                    )
                    print("Episode queued for background seal (save -> footer -> results.json)...")
                else:
                    _append_rollout_record(
                        RolloutRecord(
                            round_num=round_num,
                            policy_id=policy_id,
                            model_id=model_id,
                            anonymous_label=label,
                            outcome=outcome,
                            num_steps=num_steps,
                            episode_index=episode_index,
                            manifest_idx=target.manifest_idx,
                            pen_x=target.pen_x,
                            pen_y=target.pen_y,
                            pen_yaw=target.pen_yaw,
                            subtask_frames=tuple(subtask_frames),
                        )
                    )
                    _write_results_snapshot()

                if rollout_idx == len(remaining) - 1:
                    seal_pipeline.poll_completed(
                        description="Final episode background seal for round"
                    )

                if rollout_idx < len(remaining) - 1:
                    wait_until_ready(
                        ui, prompt="Reset the scene for the next policy", on_reset=reset_robot
                    )

            print(f"\nRound {round_num} complete.")
            print_manifest_progress("MANIFEST PROGRESS")

            seal_pipeline.poll_completed(description="Final episode background seal for round")
            _write_results_snapshot()
            if dataset is not None and checkpoint_due_after_round(round_num):
                # SYNC POINT: the round-boundary hub checkpoint mutates `dataset`
                # on the MAIN thread, so the full seal queue must drain first.
                seal_pipeline.drain(
                    description="Final episode background seal before round checkpoint",
                    timeout=seal_wait_timeout_s,
                )
                dataset, saved_episode_count = checkpoint_dataset(
                    dataset,
                    None,
                    hf_repo_id,
                    dataset_path,
                    verbose=False,
                )
            elif seal_pipeline.in_flight:
                print(
                    "Background episode seal still running; overlapping it with the "
                    "next round (it drains before any conflicting dataset write)."
                )
            if args.stop_after_round and round_num >= args.stop_after_round:
                print(
                    f"--stop-after-round {args.stop_after_round}: round {round_num} complete; "
                    "stopping through the graceful shutdown path."
                )
                cap_triggered = True
                raise KeyboardInterrupt

        print(f"Completed all {len(manifest_targets)} manifest initial states.")

    except KeyboardInterrupt:
        print("\n\nEvaluation ended.")
    finally:
        background_save_error: RuntimeError | None = None
        try:
            # SYNC POINT: end-of-session shutdown drains the full seal queue so
            # the final episode's footer + results.json record are durable before
            # the finalize/consolidate/push below.
            seal_pipeline.drain(
                description="Final background episode seal",
                timeout=seal_wait_timeout_s,
            )
        except RuntimeError as exc:
            background_save_error = exc
            print(f"ERROR: {exc}")
            print(
                "The final episode's background seal did not complete cleanly. Its "
                "rollout record only reached results.json if its parquet footer "
                "succeeded, so an unrecorded episode leaves its round incomplete "
                "and it is re-collected on resume."
            )
        # Blocks on a still-running (timed-out) chain rather than finalizing the
        # dataset underneath it; returns immediately if the chain is done/failed.
        seal_pipeline.shutdown(wait=True)
        if background_save_error is not None:
            print("Cleaning stale image scratch directories so the dataset can be resumed.")
            cleanup_stale_image_episode_dirs(dataset_path)
        if dataset is not None:
            print(f"Finalizing dataset... ({saved_episode_count} episodes saved)")
            dataset.stop_image_writer()
            dataset.finalize()
            from mulligan.data.recording import consolidate_episodes_parquet

            consolidate_episodes_parquet(dataset_path)
        if dataset is not None and not args.no_push and background_save_error is None:
            print(f"Pushing dataset to HuggingFace Hub: {hf_repo_id}")
            from mulligan.tools.lerobot_hub import push_lerobot_dataset_replacing_remote

            # Replacing push: a --rerun-incomplete-rounds redo rewrites existing
            # data/video shards, so stale remote shards must be pruned, not just
            # overlaid. For append-only runs this is identical to push_to_hub and
            # it moves the v3.0 tag onto the pushed commit either way.
            run_with_timeout(
                lambda: push_lerobot_dataset_replacing_remote(dataset, private=False),
                seconds=hub_push_timeout_s(dataset.root),
                description="HuggingFace Hub push",
            )
        ui.close()
        if remote_inference is not None:
            remote_inference.close()

        all_records = previous_records + rollout_records
        all_entries, all_names = _entries_and_names(records=all_records, policy_map=policy_map)
        complete_round_nums = _complete_round_numbers(
            all_records,
            slots_per_round=slots_per_round,
            dropped_policy_ids=dropped_policy_ids,
        )
        manifest_complete = set(range(1, len(manifest_targets) + 1)).issubset(complete_round_nums)

        if all_records:
            # Durable invocation provenance (see _load_phase_stops): one entry per graceful
            # shutdown, every invocation. Each record also carries its own visit_id, so a
            # hard death still leaves per-record provenance in the per-seal snapshots.
            phase_stops.append(
                {
                    "visit_id": visit_id,
                    "recorded_at": datetime.now().isoformat(timespec="seconds"),
                    "stop_after_round": int(args.stop_after_round),
                    "cap_triggered": bool(cap_triggered),
                    "dropped_policy_ids": sorted(dropped_policy_ids),
                    "max_round_with_records": max(r.round_num for r in all_records),
                    "n_records": len(all_records),
                    "n_records_this_visit": sum(r.visit_id == visit_id for r in all_records),
                    "record_keys": sorted(
                        [int(r.round_num), int(r.policy_id)] for r in all_records
                    ),
                }
            )
        if all_records:
            save_results_file(
                results_path,
                all_entries,
                all_records,
                args,
                dataset_name,
                policy_names=all_names,
                round_plans=round_plans,
                phase_stops=phase_stops,
            )

        if (
            all_records
            and dataset is not None
            and not args.no_push
            and background_save_error is None
        ):
            from huggingface_hub import HfApi

            print(f"Uploading results.json to HF repo: {hf_repo_id}")
            run_with_timeout(
                lambda: HfApi().upload_file(
                    path_or_fileobj=str(results_path),
                    path_in_repo="results.json",
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                ),
                seconds=60,
                description="results.json upload to HF Hub",
            )
            from mulligan.tools.lerobot_hub import advance_lerobot_version_tag

            # upload_file commits past the tag the dataset push just placed;
            # re-advance so v3.0 tracks the final state incl. results.json.
            run_with_timeout(
                lambda: advance_lerobot_version_tag(hf_repo_id),
                seconds=60,
                description="v3.0 tag re-advance after results.json upload",
            )

        # BLINDING: never print per-arm results while arms are still retired (a phased
        # eval's first phase can exhaust the manifest under its active arms).
        if manifest_complete and not dropped_policy_ids:
            print_results(
                all_entries,
                policy_names=all_names,
                rollout_records=all_records,
                num_subtask_marks=num_subtask_marks,
            )
        elif all_records:
            print()
            print("=" * 72)
            print("EVALUATION INCOMPLETE - keeping policy identities blinded in console")
            print("=" * 72)
            print(f"Completed rounds: {len(complete_round_nums)}/{len(manifest_targets)}")
            print(f"Unblinded accounting is saved in: {results_path}")
        if background_save_error is not None:
            raise background_save_error


if __name__ == "__main__":
    main(_CLI_ARGS)
