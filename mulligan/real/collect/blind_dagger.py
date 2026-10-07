# ruff: noqa: E402
"""Blinded multi-arm DAgger collection on the real Franka robot.

This is the real-world analogue of the simulator mixed-list collection pattern:
load policy arms, scramble which arm controls each episode, hide the active
arm from the operator, save one mixed LeRobot dataset, and write a JSONL ledger
that deterministically splits the mixed data back into arm-specific datasets.

Example (Marker round-2 collection with the round-1 actors and a locked manifest)::

    python -m mulligan.real.collect.blind_dagger \
        --task-name marker_d2 \
        --arm baseline_uniform=hf://mulligan/real-marker-d2-r01-baseline-dp \
        --arm mulligan_sobol=hf://mulligan/real-marker-d2-r01-mulligan-dp \
        --initial-states-manifest data/real/manifests/marker_d2/r02/<manifest>.json \
        --protocol-quota-targets no_cf=50,with_cf=50 \
        --protocol-quota-arms 'no_cf=baseline_uniform,mulligan_sobol;with_cf=mulligan_sobol' \
        --protocol-quota-selection-mode soft_weighted \
        --protocol-quota-ledger ./data/<dataset>/meta/protocol_quota_ledger.jsonl \
        --dataset-name <dataset> \
        --push-to-hub --hf-namespace <your-hf-user-or-org>
"""

from __future__ import annotations

# HighGUI must initialize before lerobot/av load (see mulligan.real.operator_ui.display).
# The parser uses only light imports and runs first, so --help and argument errors exit
# before the prewarm and before torch/lerobot load.
from mulligan.real.operator_ui.display import prewarm_highgui

import argparse
from pathlib import Path

from mulligan.real.collect.hf_utils import (
    add_hf_namespace_arg,
    add_license_arg,
    resolve_push_repo_id,
)
from mulligan.real.collect.initial_states import ArmSpec
from mulligan.real.lifecycle.tasks import TASK_NAME_HELP, get_task_spec, task_name_choices
from mulligan.real.operator_ui.cli import add_operator_ui_args
from mulligan.real.policy.dp import REAL_PROTOCOL_N_ACTION_STEPS
from mulligan.real.robot.cameras import DEFAULT_CAMERA_KEYS
from mulligan.sim.collect.quota import parse_arm_target_caps


def _parse_arm(spec: str) -> ArmSpec:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--arm must be KEY=MODEL_ID")
    key, model_id = spec.split("=", 1)
    key = key.strip()
    model_id = model_id.strip()
    if not key or not model_id:
        raise argparse.ArgumentTypeError(f"--arm has empty key or model id: {spec!r}")
    if any(ch.isspace() for ch in key):
        raise argparse.ArgumentTypeError(f"arm key must not contain whitespace: {key!r}")
    return ArmSpec(key=key, model_id=model_id)


def _parse_dp_artifact_override(spec: str) -> tuple[str, str]:
    """Parse an ``ARM_KEY=DP_ARTIFACT`` entry for --fixed-policy-dp-override.

    Mirrors manifest_eval's flag: forces a Vision-IQL arm to re-rank this exact
    DP actor artifact instead of the critic's baked-in ``dp_artifact`` (which can
    point at a non-canonical auto-resume version, e.g. ``-final:v4`` when the
    canonical clean actor is ``-final:v0``)."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--fixed-policy-dp-override must be ARM_KEY=DP_ARTIFACT, got {spec!r}"
        )
    key, dp_artifact = spec.split("=", 1)
    key = key.strip()
    dp_artifact = dp_artifact.strip()
    if not key or not dp_artifact:
        raise argparse.ArgumentTypeError(
            f"--fixed-policy-dp-override has empty arm key or DP artifact: {spec!r}"
        )
    return key, dp_artifact


def _parse_protocol_targets(spec: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--protocol-quota-targets entries must be PROTOCOL=INT; got {item!r}"
            )
        protocol, value = item.split("=", 1)
        protocol = protocol.strip()
        if not protocol:
            raise argparse.ArgumentTypeError(
                f"--protocol-quota-targets has empty protocol in {item!r}"
            )
        targets[protocol] = int(value)
    if not targets:
        raise argparse.ArgumentTypeError("--protocol-quota-targets must not be empty")
    return targets


def _parse_protocol_arms(spec: str | None) -> dict[str, list[str]] | None:
    if spec is None:
        return None
    arms_by_protocol: dict[str, list[str]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--protocol-quota-arms entries must be PROTOCOL=ARM[,ARM...]; got {item!r}"
            )
        protocol, raw_arms = item.split("=", 1)
        protocol = protocol.strip()
        arms = [arm.strip() for arm in raw_arms.split(",") if arm.strip()]
        if not protocol or not arms:
            raise argparse.ArgumentTypeError(
                f"--protocol-quota-arms has empty protocol or arms in {item!r}"
            )
        if protocol in arms_by_protocol:
            raise argparse.ArgumentTypeError(
                f"duplicate protocol in --protocol-quota-arms: {protocol}"
            )
        if len(set(arms)) != len(arms):
            raise argparse.ArgumentTypeError(
                f"duplicate arm in --protocol-quota-arms for {protocol!r}: {arms}"
            )
        arms_by_protocol[protocol] = arms
    if not arms_by_protocol:
        raise argparse.ArgumentTypeError("--protocol-quota-arms must not be empty if set")
    return arms_by_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blinded two-arm DAgger collection on the real Franka robot"
    )
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        required=True,
        help=(
            "Arm spec KEY=MODEL_ID, where MODEL_ID is hf://NAMESPACE/REPO[@REV] or a "
            "local checkpoint dir. Provide two or more arms."
        ),
    )
    parser.add_argument(
        "--target-success-per-arm",
        type=int,
        default=None,
        help="Stop after this many successful saved episodes per arm.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260525)
    parser.add_argument("--freq", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-filter", type=str, default="_left")
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=DEFAULT_CAMERA_KEYS,
        help=(
            "Comma-separated camera keys or bare serials to open and save. "
            "Empty string uses --camera-filter discovery."
        ),
    )
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-path", default="./data")
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument(
        "--protocol-quota-targets",
        type=_parse_protocol_targets,
        default=None,
        help=(
            "Enable protocol quota collection with PROTOCOL=COUNT entries, "
            "e.g. no_cf=50,with_cf=50. Counts are per targeted arm."
        ),
    )
    parser.add_argument(
        "--protocol-quota-ledger",
        type=Path,
        default=None,
        help="JSONL ledger for protocol quota accounting.",
    )
    parser.add_argument(
        "--protocol-quota-arms",
        type=_parse_protocol_arms,
        default=None,
        help=(
            "Optional protocol arm allowlist, e.g. "
            "'no_cf=baseline_uniform,mulligan_sobol;with_cf=mulligan_sobol'."
        ),
    )
    parser.add_argument(
        "--protocol-quota-arm-caps",
        type=parse_arm_target_caps,
        default=None,
        help=(
            "Optional per-arm quota caps 'PROTOCOL.ARM=INT|collected[;...]' (a ledgered mid-"
            "collection accounting amendment): 'collected' freezes that arm's target at the "
            "count already in the ledger, so the protocol completes without it and fresh "
            "selection draws from the union of protocols each arm still owes. The manifest "
            "bytes are untouched; the caps are recorded in every quota-ledger row."
        ),
    )
    parser.add_argument("--protocol-quota-balance-slack", type=int, default=1)
    parser.add_argument("--protocol-quota-progress-window", type=int, default=20)
    parser.add_argument(
        "--protocol-quota-selection-mode",
        choices=["hard_balance", "soft_weighted"],
        default="soft_weighted",
        help=(
            "Fresh manifest-row selection policy. soft_weighted (default; the paper's "
            "collections) samples an arm first with exponential weights over the quota each "
            "arm still owes, then consumes that arm's next queued row. hard_balance takes the "
            "first eligible row in manifest order whose arm keeps the per-arm counts within "
            "--protocol-quota-balance-slack."
        ),
    )
    parser.add_argument(
        "--protocol-quota-softmax-beta",
        type=float,
        default=1.0,
        help="Soft-weighted arm sampler beta; 0 gives uniform eligible-arm sampling.",
    )
    parser.add_argument(
        "--protocol-quota-sampling-seed",
        type=int,
        default=None,
        help="Seed for soft-weighted protocol quota arm sampling. Defaults to --shuffle-seed.",
    )
    parser.add_argument(
        "--quota-credit-mode",
        choices=["success", "saved"],
        default="success",
        help=(
            "Whether protocol quotas count only successful saved episodes or every "
            "saved valid trajectory while preserving success labels."
        ),
    )
    parser.add_argument(
        "--initial-states-manifest",
        type=Path,
        default=None,
        help=(
            "Manual initial-state manifest with pen_x, pen_y, pen_yaw and "
            "source labels matching --arm keys. When set, manifest source "
            "selects the policy arm."
        ),
    )
    parser.add_argument(
        "--task-name",
        choices=task_name_choices(),
        required=True,
        help=TASK_NAME_HELP,
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--push-repo-id",
        default=None,
        help=(
            "Explicit HuggingFace dataset repo id (NAMESPACE/NAME) for --push-to-hub. "
            "Defaults to DATASET_NAME, qualified with --hf-namespace when bare."
        ),
    )
    add_hf_namespace_arg(parser)
    add_license_arg(parser)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--noise-scheduler", choices=["DDPM", "DDIM"], default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=None,
        help="Override Vision-IQL action candidates for all Vision-IQL arms.",
    )
    parser.add_argument(
        "--fixed-policy-dp-override",
        action="append",
        type=_parse_dp_artifact_override,
        default=[],
        metavar="ARM_KEY=DP_ARTIFACT",
        help=(
            "Force a Vision-IQL arm (by arm key) to re-rank this exact DP actor "
            "artifact instead of the critic's baked-in dp_artifact (which can pin a "
            "non-canonical auto-resume version). Repeatable. Fails loud if the arm "
            "key is unknown or the arm is not a Vision-IQL policy."
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
    add_operator_ui_args(parser, cards=True)
    args = parser.parse_args()
    args.task_name = get_task_spec(args.task_name).task_name
    if args.push_to_hub:
        try:
            resolve_push_repo_id(args.push_repo_id or args.dataset_name, args.hf_namespace)
        except ValueError as exc:
            parser.error(str(exc))
    return args


if __name__ == "__main__":
    _CLI_ARGS = parse_args()
    prewarm_highgui()

import json
import logging
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import mulligan.real.policy.lerobot_patches  # noqa: F401  (h264 video codec)
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.constants import DataSource, EpisodeOutcome
from mulligan.data.task_names import validate_lerobot_task_name
from mulligan.real.collect.dataset_features import (
    build_supplementary_frame_fields,
    build_real_lerobot_features,
    camera_feature_name,
    compute_intervention_flags,
    ensure_dataset_can_store_episode_telemetry,
    finalize_episode_data,
    record_or_verify_camera_role_serials,
)
from mulligan.real.collect.dagger import (
    accumulate_segment_data,
    auto_detect_device,
    policy_rollout_segment,
    process_image,
    spacemouse_correction_segment,
)
from mulligan.real.policy.loader import load_policy_by_model_id
from mulligan.tools.dataset_lineage import sha256_file as _sha256_file
from mulligan.real.eval.common import (
    _streaming_encoding_enabled,
    cleanup_stale_image_episode_dirs,
    num_subtask_marks_for_task,
)
from mulligan.real.robot.cameras import (
    DEFAULT_EXCLUDED_CAMERA_KEYS,
    policy_live_camera_keys,
    remove_excluded_camera_features_from_lerobot_dataset,
    select_image_camera_keys,
    serial_key_to_role,
)
from mulligan.real.collect.hf_utils import ensure_dataset_repo
from mulligan.real.collect.rollout import (
    camera_serials_from_keys,
    parse_camera_keys,
    record_subtask_mark,
    restrict_zed_cameras_to_serials,
    verified_reset,
)
from mulligan.real.collect.initial_states import (
    InitialStateTarget,
    _format_initial_state_target,
    _initial_state_setup_subject,
    _load_initial_state_manifest,
    _load_manifest_payload,
    _manifest_keys,
    _target_value,
)
from mulligan.real.operator_ui.cards import (
    CardStyle,
    cleanup_unreferenced_initial_state_cards,
)
from mulligan.real.operator_ui.gates import GateOutcome
from mulligan.real.operator_ui.keys import key_label
from mulligan.real.operator_ui.monitor import shared_policy_crop_boxes
from mulligan.real.operator_ui.progress import (
    CollectionScene,
    CollectionStatus,
    duration,
)
from mulligan.real.operator_ui.session import OperatorUI
from mulligan.sim.collect.quota import ProtocolQuotaLedger

logger = logging.getLogger(__name__)


class _DeferredResetCoordinator:
    """Defer a verified reset until after the next operator target card is visible.

    DROID's zerorpc/gevent client is not safe to drive from a background thread:
    RPC calls can fail with gevent LoopExit when the thread has no active gevent
    hub. This coordinator intentionally keeps every RobotEnv call on the main
    thread while still letting the UI render the next schematic before reset
    motion starts.
    """

    def __init__(
        self,
        env: object,
        *,
        randomize: bool,
        reset_fn: Callable[..., dict] = verified_reset,
        before_reset: Callable[[str], None] | None = None,
    ) -> None:
        self._env = env
        self._randomize = randomize
        self._reset_fn = reset_fn
        # Runs with the queued reason right before the motion starts (the operator panel
        # switches to "Resetting robot" here, after the next target card is on screen).
        self._before_reset = before_reset
        self._pending_reason: str | None = None

    @property
    def has_pending(self) -> bool:
        return self._pending_reason is not None

    @property
    def in_progress(self) -> bool:
        return False

    def start(self, reason: str) -> None:
        if self._pending_reason is not None:
            raise RuntimeError(
                f"Cannot queue robot reset for {reason}: previous reset "
                f"{self._pending_reason!r} is still pending. "
                "Call finish() before scheduling another reset."
            )
        print(f"Queued robot reset ({reason}); it will run after the target card is shown.")
        self._pending_reason = reason

    def finish(self, reason: str) -> dict:
        if self._pending_reason is None:
            raise RuntimeError(f"No pending robot reset to finish for {reason}")
        pending_reason = self._pending_reason
        self._pending_reason = None
        print(f"Running robot reset ({pending_reason}; {reason})...", flush=True)
        if self._before_reset is not None:
            self._before_reset(pending_reason)
        return self._reset_fn(self._env, randomize=self._randomize)

    def finish_if_pending(self, reason: str) -> dict | None:
        if self._pending_reason is None:
            return None
        return self.finish(reason)

    def shutdown(self) -> None:
        return None


@dataclass
class ArmState:
    spec: ArmSpec
    arm_id: int
    policy_id: int
    policy: object
    camera_height: int
    camera_width: int


def _parse_expected_counts(spec: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--expected-success-per-arm entries must be ARM=INT; got {item!r}"
            )
        arm, value = item.split("=", 1)
        counts[arm.strip()] = int(value)
    if not counts:
        raise argparse.ArgumentTypeError("--expected-success-per-arm must not be empty")
    return counts


def _read_ledger(ledger_path: Path, arms: list[ArmSpec]) -> tuple[list[dict], Counter[str]]:
    arm_keys = {arm.key for arm in arms}
    rows: list[dict] = []
    success_counts: Counter[str] = Counter({arm.key: 0 for arm in arms})
    if not ledger_path.exists():
        return rows, success_counts

    seen_episode_idxs: set[int] = set()
    for line_no, line in enumerate(ledger_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        arm_key = row["arm_key"]
        if arm_key not in arm_keys:
            raise ValueError(
                f"{ledger_path}:{line_no}: unknown arm_key {arm_key!r}; "
                f"expected one of {sorted(arm_keys)}"
            )
        episode_index = int(row["episode_index"])
        if episode_index in seen_episode_idxs:
            raise ValueError(f"{ledger_path}:{line_no}: duplicate episode_index {episode_index}")
        seen_episode_idxs.add(episode_index)
        if bool(row["success"]):
            success_counts[arm_key] += 1
        rows.append(row)

    expected = list(range(len(rows)))
    actual = sorted(seen_episode_idxs)
    if actual != expected:
        raise ValueError(
            f"{ledger_path}: episode_index values must be contiguous {expected[:3]}...; "
            f"got {actual[:10]}"
        )
    return rows, success_counts


def _initial_state_features(manifest_meta: dict) -> dict:
    features = {
        "manifest_idx": {"dtype": "int64", "shape": (1,), "names": ["manifest_idx"]},
    }
    for key in _manifest_keys(manifest_meta):
        features[key] = {"dtype": "float32", "shape": (1,), "names": [key]}
    return features


def _target_extra_frame_fields(
    target: InitialStateTarget,
    manifest_meta: dict,
) -> dict[str, np.ndarray]:
    fields = {
        "manifest_idx": np.array([target.manifest_idx], dtype=np.int64),
    }
    for key in _manifest_keys(manifest_meta):
        fields[key] = np.array([_target_value(target, key)], dtype=np.float32)
    return fields


def _target_ledger_fields(target: InitialStateTarget, manifest_meta: dict) -> dict[str, float]:
    return {key: _target_value(target, key) for key in _manifest_keys(manifest_meta)}


def _append_ledger_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _manifest_snapshot_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "initial_states_manifest.json"


def _is_lerobot_dataset_initialized(dataset_path: Path) -> bool:
    return (dataset_path / "meta" / "info.json").exists()


def _prepare_lerobot_create_root(dataset_path: Path) -> Path | None:
    """Make room for LeRobotDataset.create while preserving pre-dataset artifacts."""
    if not dataset_path.exists():
        return None
    if _is_lerobot_dataset_initialized(dataset_path):
        raise RuntimeError(f"Dataset is already initialized at {dataset_path}")

    meta_dir = dataset_path / "meta"
    for ledger_name in (
        "teleop_manifest_ledger.jsonl",
        "blind_dagger_ledger.jsonl",
        "protocol_quota_ledger.jsonl",
    ):
        ledger_path = meta_dir / ledger_name
        if ledger_path.exists() and ledger_path.stat().st_size > 0:
            raise RuntimeError(
                f"{dataset_path} exists without meta/info.json but contains non-empty "
                f"{ledger_path}. Refusing to create over a possibly inconsistent dataset."
            )

    backup_path = dataset_path.with_name(
        f"{dataset_path.name}.pre_lerobot_create_{int(time.time())}"
    )
    shutil.move(str(dataset_path), str(backup_path))
    print(f"Moved pre-dataset artifacts to {backup_path}")
    return backup_path


def _restore_lerobot_precreate_artifacts(backup_path: Path | None, dataset_path: Path) -> None:
    if backup_path is None:
        return
    source_targets = backup_path / "meta" / "initial_state_targets"
    if source_targets.exists():
        target_targets = dataset_path / "meta" / "initial_state_targets"
        shutil.copytree(source_targets, target_targets, dirs_exist_ok=True)
        print(f"Restored pre-created initial-state target cards to {target_targets}")


def _check_existing_manifest_snapshot(source_path: Path, dataset_path: Path) -> None:
    snapshot_path = _manifest_snapshot_path(dataset_path)
    if not snapshot_path.exists():
        return
    source_hash = _sha256_file(source_path)
    snapshot_hash = _sha256_file(snapshot_path)
    if snapshot_hash != source_hash:
        if _is_append_only_manifest_extension(snapshot_path, source_path):
            print(
                f"Manifest {source_path} is an append-only extension of dataset snapshot "
                f"{snapshot_path}; collection can resume with the extended manifest."
            )
            return
        _raise_manifest_mismatch(snapshot_path, source_path, snapshot_hash, source_hash)


def _ensure_manifest_snapshot(source_path: Path, dataset_path: Path) -> tuple[Path, str]:
    """Copy the locked initial-state manifest into dataset meta and verify resumes."""
    source_hash = _sha256_file(source_path)
    snapshot_path = _manifest_snapshot_path(dataset_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        snapshot_hash = _sha256_file(snapshot_path)
        if snapshot_hash != source_hash:
            if not _is_append_only_manifest_extension(snapshot_path, source_path):
                _raise_manifest_mismatch(snapshot_path, source_path, snapshot_hash, source_hash)
            shutil.copyfile(source_path, snapshot_path)
            print(f"Updated dataset manifest snapshot to append-only extension {source_path}")
    else:
        shutil.copyfile(source_path, snapshot_path)
    return snapshot_path, source_hash


def _is_append_only_manifest_extension(old_path: Path, new_path: Path) -> bool:
    old_payload = _load_manifest_payload(old_path)
    new_payload = _load_manifest_payload(new_path)

    # An append-only extension may GROW the state list (and arm_counts / samplers.n_states),
    # but the protocol/geometry identity must be invariant -- otherwise a resume could
    # silently overwrite the dataset snapshot with a differently-designed manifest that
    # merely shares a state prefix. Compare every protocol-defining top-level field;
    # absent-on-both keys (e.g. a task without sampled_placements) compare equal.
    required_equal_keys = [
        "task",
        "schema",
        "keys",
        "units",
        "operator_frame",
        "bounds",
        "bounds_display",
        "sources",
        "seeds",
        "shuffle_block_size",
        "phase",
        "source",
        "arms",
        "sampled_placements",
        "scrambled_manifest_order",
    ]
    for key in required_equal_keys:
        if old_payload.get(key) != new_payload.get(key):
            return False

    old_states = old_payload.get("states")
    new_states = new_payload.get("states")
    if not isinstance(old_states, list) or not isinstance(new_states, list):
        return False
    if len(new_states) < len(old_states):
        return False
    return new_states[: len(old_states)] == old_states


def _raise_manifest_mismatch(
    snapshot_path: Path,
    source_path: Path,
    snapshot_hash: str,
    source_hash: str,
) -> None:
    raise RuntimeError(
        f"Existing dataset manifest snapshot {snapshot_path} does not match "
        f"{source_path}. snapshot_sha256={snapshot_hash}, source_sha256={source_hash}. "
        "Only exact matches or append-only extensions with an identical state prefix are allowed."
    )


def _select_arm(
    arms: list[ArmSpec],
    success_counts: Counter[str],
    target_success_per_arm: int,
    *,
    shuffle_seed: int,
    episode_index: int,
) -> ArmSpec | None:
    eligible = [arm for arm in arms if success_counts[arm.key] < target_success_per_arm]
    if not eligible:
        return None
    min_count = min(success_counts[arm.key] for arm in eligible)
    lowest = [arm for arm in eligible if success_counts[arm.key] == min_count]
    rng = random.Random(shuffle_seed + episode_index)
    return rng.choice(lowest)


def _select_manifest_target(
    targets: list[InitialStateTarget],
    consumed_manifest_idxs: set[int],
    success_counts: Counter[str],
    target_success_per_arm: int,
) -> InitialStateTarget | None:
    incomplete = {arm for arm, count in success_counts.items() if count < target_success_per_arm}
    if not incomplete:
        return None
    for target in targets:
        if target.manifest_idx in consumed_manifest_idxs:
            continue
        if target.source in incomplete:
            return target
    remaining = {arm: target_success_per_arm - success_counts[arm] for arm in incomplete}
    raise RuntimeError(
        "Initial-state manifest is exhausted before all arms reached the "
        f"success target. Remaining successes needed: {remaining}"
    )


def _select_protocol_manifest_target(
    targets: list[InitialStateTarget],
    quota: ProtocolQuotaLedger,
) -> InitialStateTarget | None:
    if quota.is_complete():
        return None
    manifest_idx = quota.select_fresh_manifest_idx()
    if manifest_idx is not None:
        target_by_idx = {target.manifest_idx: target for target in targets}
        if manifest_idx not in target_by_idx:
            raise RuntimeError(
                f"Protocol quota selected manifest_idx={manifest_idx}, but the "
                "loaded initial-state targets do not contain that row."
            )
        return target_by_idx[manifest_idx]
    raise RuntimeError(
        "Protocol manifest is exhausted before all protocol quotas were filled. "
        f"Remaining={quota.remaining()}"
    )


def _count_protocol_remaining_targets(
    targets: list[InitialStateTarget],
    quota: ProtocolQuotaLedger,
) -> tuple[int, int]:
    remaining = [
        target
        for target in targets
        if target.manifest_idx not in quota.fresh_consumed_manifest_idxs
    ]
    eligible = [
        target for target in remaining if quota.is_fresh_state_eligible(target.manifest_idx)
    ]
    return len(remaining), len(eligible)


def _collection_status(
    *,
    protocol_quota: ProtocolQuotaLedger | None,
    success_counts: Counter[str],
    target_success_per_arm: int | None,
    session_successes: int,
    elapsed_s: float,
) -> CollectionStatus:
    """The operator panel's header cells for the two collection modes.

    Protocol-quota sessions show the ledger's own quota / pace / ETA (single source of
    truth, same numbers as the terminal readout). Per-arm sessions show pooled successes
    with a whole-session-pace ETA (the eval panel's rule).
    """
    if protocol_quota is not None:
        return CollectionStatus(**protocol_quota.operator_panel_status())
    if target_success_per_arm is None:
        raise ValueError("per-arm collection status needs target_success_per_arm")
    total = target_success_per_arm * len(success_counts)
    done = sum(min(count, target_success_per_arm) for count in success_counts.values())
    remaining = total - done
    if remaining == 0:
        eta = 0.0
    elif session_successes:
        eta = remaining * elapsed_s / session_successes
    else:
        eta = None
    return CollectionStatus(
        metrics=(
            ("SUCCESSES", f"{done} / {total}"),
            ("REMAINING", str(remaining)),
            ("EST. TIME LEFT", duration(eta)),
        ),
        fraction=done / total,
        summary=f"{session_successes} successes this session",
    )


def _format_protocol_quota_totals(quota: ProtocolQuotaLedger) -> str:
    parts = []
    for protocol in quota.protocols:
        parts.append(
            f"{protocol}={quota.protocol_count(protocol)}/{quota.protocol_total(protocol)}"
        )
    return ", ".join(parts)


def _assert_protocol_cli_matches_manifest(
    *,
    manifest_meta: dict,
    targets: dict[str, int],
    arms_by_protocol: dict[str, list[str]] | None,
) -> None:
    manifest_targets = manifest_meta.get("protocol_targets")
    if manifest_targets is not None:
        normalized_targets = {str(k): int(v) for k, v in manifest_targets.items()}
        if normalized_targets != targets:
            raise ValueError(
                "Protocol quota targets disagree with the locked manifest: "
                f"cli={targets}, manifest={normalized_targets}"
            )
    manifest_arms = manifest_meta.get("protocol_arms")
    if manifest_arms is not None:
        normalized_manifest_arms = {
            str(protocol): [str(arm) for arm in arms] for protocol, arms in manifest_arms.items()
        }
        if arms_by_protocol is None:
            raise ValueError(
                "Locked manifest defines protocol_arms, but --protocol-quota-arms "
                "was not provided. Pass the manifest's exact protocol arm map."
            )
        normalized_cli_arms = {
            str(protocol): [str(arm) for arm in arms] for protocol, arms in arms_by_protocol.items()
        }
        if normalized_cli_arms != normalized_manifest_arms:
            raise ValueError(
                "Protocol quota arms disagree with the locked manifest: "
                f"cli={normalized_cli_arms}, manifest={normalized_manifest_arms}"
            )


def _make_features(
    episode_data: dict,
    cam_data_keys: list[str],
    manifest_meta: dict | None,
) -> dict:
    extra_features = {
        "arm_id": {"dtype": "int64", "shape": (1,), "names": ["arm_id"]},
        "policy_id": {"dtype": "int64", "shape": (1,), "names": ["policy_id"]},
    }
    if manifest_meta:
        extra_features.update(_initial_state_features(manifest_meta))
    else:
        extra_features["manifest_idx"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["manifest_idx"],
        }
    return build_real_lerobot_features(
        episode_data,
        cam_data_keys,
        camera_name_fn=serial_key_to_role,
        extra_features=extra_features,
    )


def _ensure_dataset_has_manifest_features(
    dataset: LeRobotDataset,
    *,
    manifest_meta: dict,
    dataset_path: Path,
) -> None:
    if not manifest_meta:
        return
    expected = _initial_state_features(manifest_meta)
    features = dataset.meta.features
    bad: dict[str, str] = {}
    for key, spec in expected.items():
        got = features.get(key)
        if got is None:
            bad[key] = "missing"
            continue
        if got.get("dtype") != spec["dtype"] or tuple(got.get("shape", ())) != tuple(spec["shape"]):
            bad[key] = f"got dtype={got.get('dtype')!r} shape={got.get('shape')!r}"
    if not bad:
        return
    raise RuntimeError(
        f"Existing dataset at {dataset_path} does not carry the manifest-derived "
        f"initial-state feature schema required for this run: {bad}. Refusing to "
        "append ambiguous marker_d2 data. If it has no useful episodes, move it "
        "aside and restart collection; otherwise backfill/migrate these columns "
        "from meta/initial_states_manifest.json by manifest_idx first."
    )


def _constant_extra_frame_fields(
    *,
    arm_state: ArmState,
    target: InitialStateTarget | None,
    manifest_meta: dict | None,
) -> dict[str, np.ndarray]:
    fields = {
        "arm_id": np.array([arm_state.arm_id], dtype=np.int64),
        "policy_id": np.array([arm_state.policy_id], dtype=np.int64),
    }
    if target is not None:
        if manifest_meta is None:
            raise RuntimeError("initial-state target is set but manifest_meta is missing")
        fields.update(_target_extra_frame_fields(target, manifest_meta))
        return fields

    fields["manifest_idx"] = np.array([-1], dtype=np.int64)
    for key in _manifest_keys(manifest_meta) if manifest_meta else []:
        fields[key] = np.array([np.nan], dtype=np.float32)
    return fields


def _build_dataset_schema_probe(
    obs: dict,
    camera_keys: list[str],
    *,
    camera_height: int,
    camera_width: int,
) -> dict:
    state = np.concatenate(
        [
            np.array(obs["robot_state"]["cartesian_position"], dtype=np.float32),
            np.array([obs["robot_state"]["gripper_position"]], dtype=np.float32),
        ]
    )
    episode_data = {
        "observations": [state],
        "actions": [np.zeros(7, dtype=np.float32)],
    }
    # build_real_lerobot_features declares ALL Franka telemetry columns
    # unconditionally, and the live recording path (policy_rollout_segment ->
    # append_franka_telemetry) NaN-fills any the server does not expose, so schema
    # and frames always agree without probing which signals are present here.
    for cam_key in camera_keys:
        episode_data[f"image_{cam_key}"] = [
            process_image(obs["image"][cam_key], camera_height, camera_width)
        ]
    return episode_data


def _assert_writer_episode_size(dataset: LeRobotDataset, expected_size: int) -> dict:
    writer = getattr(dataset, "writer", None)
    if writer is None:
        raise RuntimeError("LeRobotDataset has no writer; open it with create/resume before saving")
    buffer = writer.episode_buffer
    actual_size = int(buffer["size"])
    if actual_size != expected_size:
        raise RuntimeError(
            f"LeRobot live episode buffer has {actual_size} frames, expected {expected_size}. "
            "Refusing to save misaligned DAgger labels."
        )
    return buffer


def _add_live_frame_to_dataset(
    dataset: LeRobotDataset,
    episode_data: dict,
    frame_index: int,
    *,
    camera_keys: list[str],
    task_name: str,
    source_id: int,
    extra_frame_fields: dict[str, np.ndarray],
) -> None:
    rewards = episode_data.get("rewards")
    dones = episode_data.get("dones")
    steps_to_go = episode_data.get("steps_to_go")
    frame = {
        "task": task_name,
        "observation.state": episode_data["observations"][frame_index].astype(np.float32),
        "action": episode_data["actions"][frame_index].astype(np.float32),
        "steps_to_go": np.array(
            [0 if steps_to_go is None else steps_to_go[frame_index]],
            dtype=np.int64,
        ),
        "source": np.array([source_id], dtype=np.int64),
        "intervention": np.array([0], dtype=np.int64),
        "success": np.array([EpisodeOutcome.FAILURE], dtype=np.int64),
        "is_valid": np.array([1], dtype=np.int64),
        "reward": np.array([0.0 if rewards is None else rewards[frame_index]], dtype=np.float32),
        "done": np.array([0 if dones is None else dones[frame_index]], dtype=np.int64),
    }
    frame.update(extra_frame_fields)
    frame.update(build_supplementary_frame_fields(episode_data, frame_index))
    for cam_data_key in camera_keys:
        frame[camera_feature_name(cam_data_key, serial_key_to_role)] = episode_data[cam_data_key][
            frame_index
        ]
    dataset.add_frame(frame)


def _patch_live_episode_buffer_for_final_outcome(
    dataset: LeRobotDataset,
    episode_data: dict,
    *,
    sources: list[int],
    episode_success: bool,
) -> list[int]:
    episode_length = len(episode_data["actions"])
    if len(sources) != episode_length:
        raise ValueError(
            "sources length must match live episode length before saving "
            f"({len(sources)=}, {episode_length=})"
        )
    buffer = _assert_writer_episode_size(dataset, episode_length)
    intervention_flags = compute_intervention_flags(sources)
    success_value = EpisodeOutcome.SUCCESS if episode_success else EpisodeOutcome.FAILURE

    replacements: dict[str, list[np.ndarray]] = {
        "steps_to_go": [
            np.array([episode_data["steps_to_go"][i]], dtype=np.int64)
            for i in range(episode_length)
        ],
        "source": [np.array([sources[i]], dtype=np.int64) for i in range(episode_length)],
        "intervention": [
            np.array([intervention_flags[i]], dtype=np.int64) for i in range(episode_length)
        ],
        "success": [np.array([success_value], dtype=np.int64) for _ in range(episode_length)],
        "is_valid": [
            np.array([0 if i == episode_length - 1 else 1], dtype=np.int64)
            for i in range(episode_length)
        ],
        "reward": [
            np.array([episode_data["rewards"][i]], dtype=np.float32) for i in range(episode_length)
        ],
        "done": [
            np.array([episode_data["dones"][i]], dtype=np.int64) for i in range(episode_length)
        ],
    }
    for key, values in replacements.items():
        existing = buffer.get(key)
        if existing is None:
            raise RuntimeError(f"LeRobot episode buffer is missing expected key {key!r}")
        if len(existing) != episode_length:
            raise RuntimeError(
                f"LeRobot episode buffer key {key!r} has length {len(existing)}, "
                f"expected {episode_length}"
            )
        buffer[key] = values
    return intervention_flags


def _load_or_create_dataset(
    dataset: LeRobotDataset | None,
    *,
    dataset_name: str,
    dataset_path: Path,
    episode_data: dict,
    cam_data_keys: list[str],
    freq: int,
    manifest_meta: dict | None = None,
    streaming_encoding: bool = True,
) -> LeRobotDataset:
    if dataset is not None:
        return dataset
    if _is_lerobot_dataset_initialized(dataset_path):
        removed_camera_features = remove_excluded_camera_features_from_lerobot_dataset(dataset_path)
        if removed_camera_features:
            print(
                "Removed excluded camera feature(s) from existing dataset: "
                f"{removed_camera_features}"
            )
        print(f"Loading existing dataset from {dataset_path}")
        dataset = LeRobotDataset(repo_id=dataset_name, root=str(dataset_path))
        dataset = ensure_dataset_can_store_episode_telemetry(
            dataset,
            episode_data,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            image_writer_threads=0 if streaming_encoding else 4,
            streaming_encoding=streaming_encoding,
        )
        if manifest_meta:
            _ensure_dataset_has_manifest_features(
                dataset,
                manifest_meta=manifest_meta,
                dataset_path=dataset_path,
            )
        print(f"Loaded existing dataset with {dataset.num_episodes} episodes")
        # Resume guard: the live serial<->role cabling must still match what this
        # dataset was created with, or appended frames would be mislabeled.
        record_or_verify_camera_role_serials(dataset, cam_data_keys)
        return dataset

    print(f"Creating new dataset at {dataset_path}")
    precreate_backup_path = _prepare_lerobot_create_root(dataset_path)
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=freq,
        root=str(dataset_path),
        robot_type="franka",
        features=_make_features(episode_data, cam_data_keys, manifest_meta),
        image_writer_threads=0 if streaming_encoding else 4,
        streaming_encoding=streaming_encoding,
    )
    _restore_lerobot_precreate_artifacts(precreate_backup_path, dataset_path)
    # Record the serial<->role provenance map.
    record_or_verify_camera_role_serials(dataset, cam_data_keys)
    print(f"Dataset created at {dataset_path}")
    return dataset


def configure_operator_logging() -> None:
    """Make ``logging.INFO`` actually reach the operator's terminal.

    ``mulligan.real.collect.dagger`` (imported by this module) calls
    ``logging.basicConfig`` after importing ``mulligan.real.policy.lerobot_patches``,
    which transitively installs a root handler, so that ``basicConfig`` is a no-op and
    every ``logger.info`` was invisible on the robot. ``force=True`` replaces whatever
    handler got there first.

    A separate function so the regression test can exercise the real setup rather than a
    copy of it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def install_graceful_termination_handlers() -> None:
    """Turn ``SIGTERM``/``SIGHUP`` into an exception so the shutdown ``finally`` runs.

    LeRobot v3 only makes episodes durable at ``finalize()``. The default handler exits
    the interpreter without unwinding, so a plain ``kill`` (or a closed ssh session)
    skipped the collector's ``finally``, which finalizes the dataset. Raising
    ``KeyboardInterrupt`` routes both signals into the exact path Ctrl-C already takes.
    ``SIGKILL`` cannot be handled.
    """
    import signal

    def _raise_interrupt(signum, _frame):
        name = signal.Signals(signum).name
        print(
            f"\n\nReceived {name}: finalizing the dataset before exiting "
            "(do NOT kill -9 now -- that is what loses the un-finalized tail).",
            flush=True,
        )
        raise KeyboardInterrupt(name)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _raise_interrupt)


def collection_env_kwargs(action_space: str, gripper_action_space: str | None) -> dict:
    """RobotEnv constructor kwargs for a homogeneous collection cohort.

    The constructor spaces are only the per-step FALLBACK — both DAgger segments pass
    their spaces explicitly on every ``env.step`` — so this mirrors ``manifest_eval``:
    UMI-relative cohorts (``cartesian_position``) get the position gripper, and the
    cartesian_velocity default is unchanged from every prior velocity round.
    """
    if action_space == "cartesian_position":
        return {"action_space": "cartesian_position", "gripper_action_space": "position"}
    if action_space == "cartesian_velocity":
        return {"action_space": "cartesian_velocity"}
    raise ValueError(f"Unsupported collection cohort action_space {action_space!r}")


def main(args: argparse.Namespace | None = None) -> None:
    configure_operator_logging()
    install_graceful_termination_handlers()
    if args is None:
        args = parse_args()
    args.task_name = validate_lerobot_task_name(args.task_name, context="--task-name")
    arms: list[ArmSpec] = list(args.arm)
    if len(arms) < 2:
        raise ValueError(f"Blind real DAgger expects at least two --arm entries, got {len(arms)}")
    if len({arm.key for arm in arms}) != len(arms):
        raise ValueError(f"Duplicate arm keys: {[arm.key for arm in arms]}")
    protocol_quota_enabled = args.protocol_quota_targets is not None
    if protocol_quota_enabled:
        if args.initial_states_manifest is None:
            raise ValueError("--protocol-quota-targets requires --initial-states-manifest")
        if args.protocol_quota_ledger is None:
            raise ValueError("--protocol-quota-targets requires --protocol-quota-ledger")
        if (
            args.protocol_quota_selection_mode == "soft_weighted"
            and args.protocol_quota_balance_slack != 1
        ):
            raise ValueError(
                "--protocol-quota-balance-slack is only used by "
                "--protocol-quota-selection-mode=hard_balance; leave it at the "
                "default when using soft_weighted."
            )
        if args.protocol_quota_softmax_beta < 0:
            raise ValueError("--protocol-quota-softmax-beta must be >= 0")
    else:
        if args.target_success_per_arm is None:
            raise ValueError(
                "--target-success-per-arm is required without --protocol-quota-targets"
            )
        if args.target_success_per_arm <= 0:
            raise ValueError("--target-success-per-arm must be positive")

    device = args.device or auto_detect_device()
    dataset_name = args.dataset_name
    dataset_path = Path(args.dataset_path) / dataset_name
    ledger_path = args.ledger_path or dataset_path / "meta" / "blind_dagger_ledger.jsonl"
    requested_camera_keys = parse_camera_keys(args.camera_keys, args.camera_filter)
    # Cards (and so the card window) exist for manifest-driven collection only.
    ui = OperatorUI.from_args(
        args,
        cards=args.initial_states_manifest is not None,
        default_card_dir=dataset_path / "meta" / "initial_state_targets",
    )
    cleanup_stale_image_episode_dirs(dataset_path)

    protocol_quota: ProtocolQuotaLedger | None = None
    if protocol_quota_enabled:
        ledger_rows = []
        success_counts: Counter[str] = Counter({arm.key: 0 for arm in arms})
        saved_episode_count = 0
        consumed_manifest_idxs: set[int] = set()
    else:
        ledger_rows, success_counts = _read_ledger(ledger_path, arms)
        saved_episode_count = len(ledger_rows)
        consumed_manifest_idxs = {
            int(row["manifest_idx"]) for row in ledger_rows if row.get("manifest_idx") is not None
        }
    manifest_targets: list[InitialStateTarget] | None = None
    manifest_meta: dict = {}
    manifest_snapshot_path: Path | None = None
    manifest_sha256: str | None = None
    if args.initial_states_manifest is not None:
        missing_manifest_rows = [
            int(row["episode_index"]) for row in ledger_rows if row.get("manifest_idx") is None
        ]
        if missing_manifest_rows:
            raise ValueError(
                f"{ledger_path}: cannot resume manifest-driven collection from "
                f"ledger rows without manifest_idx: {missing_manifest_rows[:20]}"
            )
        manifest_targets, manifest_meta = _load_initial_state_manifest(
            args.initial_states_manifest,
            arms,
            expected_task=args.task_name,
        )
        if protocol_quota_enabled:
            _assert_protocol_cli_matches_manifest(
                manifest_meta=manifest_meta,
                targets=args.protocol_quota_targets,
                arms_by_protocol=args.protocol_quota_arms,
            )
            protocol_quota = ProtocolQuotaLedger(
                manifest_path=args.initial_states_manifest,
                targets_by_protocol=args.protocol_quota_targets,
                ledger_path=args.protocol_quota_ledger,
                arms_by_protocol=args.protocol_quota_arms,
                arm_target_caps=args.protocol_quota_arm_caps,
                balance_slack=args.protocol_quota_balance_slack,
                progress_window=args.protocol_quota_progress_window,
                selection_mode=args.protocol_quota_selection_mode,
                softmax_beta=args.protocol_quota_softmax_beta,
                sampling_seed=(
                    args.shuffle_seed
                    if args.protocol_quota_sampling_seed is None
                    else args.protocol_quota_sampling_seed
                ),
            )
            if protocol_quota.protocol_arm_targets:
                print(
                    "Protocol quota per-arm caps: "
                    f"{protocol_quota.protocol_arm_targets}; those arms are done for that "
                    "protocol and fresh selection draws from the union of protocols each arm "
                    "still owes."
                )
            saved_episode_count = protocol_quota.n_saved_rows
            consumed_manifest_idxs = set(protocol_quota.fresh_consumed_manifest_idxs)
            for arm in arms:
                success_counts[arm.key] = int(
                    protocol_quota.counts.get("no_cf", Counter()).get(arm.key, 0)
                )
        manifest_snapshot_path = _manifest_snapshot_path(dataset_path)
        manifest_sha256 = _sha256_file(args.initial_states_manifest)
        _check_existing_manifest_snapshot(args.initial_states_manifest, dataset_path)
        if not protocol_quota_enabled:
            duplicated_consumed = len(consumed_manifest_idxs) != sum(
                1 for row in ledger_rows if row.get("manifest_idx") is not None
            )
            if duplicated_consumed:
                raise ValueError(f"{ledger_path}: duplicate manifest_idx values in existing ledger")
        manifest_indexes = {target.manifest_idx for target in manifest_targets}
        unknown_consumed = consumed_manifest_idxs - manifest_indexes
        if unknown_consumed:
            resume_ledger_path = args.protocol_quota_ledger or ledger_path
            raise ValueError(
                f"{resume_ledger_path}: ledger contains manifest_idx values not present "
                f"in {args.initial_states_manifest}: {sorted(unknown_consumed)[:20]}"
            )

    print("=" * 72)
    print("Real Robot Blinded Multi-Arm DAgger Collection")
    print("=" * 72)
    print(f"Task: {args.task_name}")
    print(f"Dataset: {dataset_name}")
    if protocol_quota is None:
        print(f"Ledger: {ledger_path}")
        print(f"Target successes per arm: {args.target_success_per_arm}")
        print(f"Current successful counts: {dict(success_counts)}")
    else:
        print(f"Protocol quota ledger: {args.protocol_quota_ledger}")
        print(f"Protocol quotas: {_format_protocol_quota_totals(protocol_quota)}")
        print(
            "Protocol quota selection: "
            f"{protocol_quota.selection_mode}"
            + (
                ""
                if protocol_quota.selection_mode == "hard_balance"
                else (f" (beta={protocol_quota.softmax_beta}, seed={protocol_quota.sampling_seed})")
            )
        )
        print(f"Quota credit mode: {args.quota_credit_mode}")
    print(f"Device: {device}")
    if args.initial_states_manifest is not None:
        print("Initial-state manifest: loaded (path hidden for blinding)")
        print(f"Manifest rows consumed: {len(consumed_manifest_idxs)}")
    if requested_camera_keys is not None:
        print(f"Camera keys: {requested_camera_keys}")
    print()

    # ---- Load policies -----------------------------------------------------
    print(
        f"[protocol] Effective n_action_steps (exec horizon) = {args.n_action_steps} "
        f"(real-robot protocol default = {REAL_PROTOCOL_N_ACTION_STEPS}; "
        "prediction horizon is checkpoint-specific)."
    )
    dp_overrides = dict(args.fixed_policy_dp_override)
    unknown_override_keys = set(dp_overrides) - {arm.key for arm in arms}
    if unknown_override_keys:
        raise SystemExit(
            f"--fixed-policy-dp-override references unknown arm key(s) "
            f"{sorted(unknown_override_keys)}; arms are {sorted(arm.key for arm in arms)}"
        )
    arm_states: dict[str, ArmState] = {}
    for arm_id, arm in enumerate(arms):
        print(f"Loading arm {arm_id + 1}/{len(arms)}")
        entry = load_policy_by_model_id(
            arm.model_id,
            policy_id=arm_id,
            device=device,
            noise_scheduler=args.noise_scheduler,
            num_inference_steps=args.num_inference_steps,
            default_camera_height=args.camera_height,
            default_camera_width=args.camera_width,
            n_action_steps=args.n_action_steps,
            dp_artifact_override=dp_overrides.get(arm.key),
        )
        if args.num_action_samples is not None or arm.key in dp_overrides:
            from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

            if arm.key in dp_overrides and not isinstance(entry.policy, VisionIDQLRealWorldPolicy):
                # The non-IDQL load path silently ignores dp_artifact_override --
                # refusing here keeps a mis-keyed override from becoming a no-op.
                raise SystemExit(
                    f"--fixed-policy-dp-override given for arm {arm.key!r}, but its "
                    f"loaded policy is {type(entry.policy).__name__}, not a Vision-IQL "
                    "policy; the override would be silently ignored."
                )
            if args.num_action_samples is not None and isinstance(
                entry.policy, VisionIDQLRealWorldPolicy
            ):
                entry.policy.num_action_samples = args.num_action_samples
        arm_states[arm.key] = ArmState(
            spec=arm,
            arm_id=arm_id,
            policy_id=entry.policy_id,
            policy=entry.policy,
            camera_height=entry.camera_height,
            camera_width=entry.camera_width,
        )

    # env_action_space: cartesian_velocity, or cartesian_position for the relative arm.
    action_spaces = {
        getattr(state.policy, "env_action_space", state.policy.action_space)
        for state in arm_states.values()
    }
    gripper_spaces = {
        getattr(state.policy, "gripper_action_space", None) for state in arm_states.values()
    }
    if len(action_spaces) != 1 or len(gripper_spaces) != 1:
        raise ValueError(
            "All blinded collection arms must use the same RobotEnv action spaces; "
            f"action_spaces={action_spaces}, gripper_spaces={gripper_spaces}"
        )
    # Position-space cohorts (UMI-relative policies, which command DROID's
    # cartesian_position space) ARE supported here, alongside the VELOCITY
    # SpaceMouse human-correction segments: RobotEnv.step takes action_space /
    # gripper_action_space PER CALL and the NUC recomputes the joint target from the
    # current robot state on every command (no controller-mode switch), so
    # policy_rollout_segment steps with the policy's own spaces and
    # spacemouse_correction_segment steps with cartesian_velocity + velocity gripper
    # inside one episode (the exact mix the non-blind collect.dagger has always run).
    # `active.policy.reset()` after every 'continue' correction re-anchors the
    # relative arm's chunk at the post-correction pose. The constructor default
    # below mirrors manifest_eval and only governs fallbacks.

    # Operator camera monitor must show the POLICIES' actual trained crops, not the
    # station defaults: per-task overrides ride in each checkpoint's camera_crop_boxes.
    # Only a crop map SHARED by ALL arms can be shown -- rendering the active arm's crop
    # would leak arm identity through the blind protocol.
    if ui.monitor_camera_keys:
        ui.monitor_crop_boxes = shared_policy_crop_boxes(
            [state.policy for state in arm_states.values()]
        )

    if requested_camera_keys is not None:
        requested_serials = camera_serials_from_keys(requested_camera_keys)
        print(f"Restricting ZED cameras to serials: {requested_serials}")
        restrict_zed_cameras_to_serials(requested_serials)

    spacemouse = None

    import mulligan.real.robot.camera_config  # noqa: F401  (patches ZED to 15fps)
    from mulligan.real.robot.droid_compat import RobotEnv
    from mulligan.teleop.spacemouse import RobosuiteSpaceMouse

    env_kwargs = collection_env_kwargs(next(iter(action_spaces)), next(iter(gripper_spaces)))
    print(f"Initializing robot environment ({env_kwargs})...")
    env = RobotEnv(**env_kwargs)

    print("Initializing SpaceMouse...")
    spacemouse = RobosuiteSpaceMouse(
        pos_sensitivity=1.3,
        rot_sensitivity=1.6,
    )

    print("Initializing keyboard listener...")
    _ = ui.keyboard  # enter cbreak mode now, before the first prompt
    print("Keyboard listener ready (terminal mode, works over SSH)")
    ui.observe = env.get_observation

    num_subtask_marks = num_subtask_marks_for_task(args.task_name)
    if num_subtask_marks > 0:
        print(
            f"Task {args.task_name} defines {num_subtask_marks} mid-episode sub-goal mark(s): "
            f"press {key_label('g')} when each sub-goal is reached (reward=1.0 spike at save, "
            "the same record the outcome review writes)."
        )
    session_successes = 0

    def refresh_collection_status() -> None:
        ui.update_status(
            _collection_status(
                protocol_quota=protocol_quota,
                success_counts=success_counts,
                target_success_per_arm=args.target_success_per_arm,
                session_successes=session_successes,
                elapsed_s=ui.progress.elapsed,
            )
        )

    ui.configure_progress(total=None, completed=0)
    refresh_collection_status()

    # Documented rollback knob (MULLIGAN_STREAMING_ENCODING=0), honoured by every real
    # entrypoint through this one helper: turn streaming off if the encoder cannot keep up.
    streaming_encoding = _streaming_encoding_enabled()
    if not streaming_encoding:
        print("MULLIGAN_STREAMING_ENCODING=0: recording with the buffered (non-streaming) encoder.")

    dataset = None
    camera_keys: list[str] = []
    all_camera_keys: list[str] = []
    counterfactual_target: InitialStateTarget | None = None
    reset_coordinator = _DeferredResetCoordinator(
        env,
        randomize=args.randomize_reset,
        before_reset=lambda _reason: ui.set_phase(
            "Resetting robot", "Keep clear of the arm during reset."
        ),
    )

    session_ended_intentionally = False
    try:
        # One-time session-resume CF prompt. Mirrors the sim collector's
        # resume_counterfactual_manifest_idx flow in robosuite_dagger_state.py:
        # when relaunching to continue a collection, the previous session's last
        # credited start may still owe a With-CF replay (the in-session CF prompt
        # only fires immediately after the episode, so a quit/restart drops it).
        # Offer that pending CF before sampling a fresh target from the queue.
        resume_quit = False
        resume_target: InitialStateTarget | None = None
        if protocol_quota is not None and manifest_targets is not None:
            resume_idx = protocol_quota.resume_counterfactual_manifest_idx()
            if resume_idx is not None:
                resume_target = next(t for t in manifest_targets if t.manifest_idx == resume_idx)
            elif protocol_quota.last_successful_manifest_idx is not None:
                print(
                    "\nPrevious successful start cannot accept another With-CF "
                    "replay; starting from the next eligible fresh start."
                )
        if resume_target is not None:
            choice = ui.choose(
                prompt=(
                    f"\nPrevious successful start (#{resume_target.manifest_idx + 1:03d}) can "
                    "still accept a With-CF replay"
                ),
                choices={
                    "c": "collect that With-CF now",
                    "n": "draw a fresh target from the queue",
                    "q": "quit",
                },
                default="n",
                phase="Resume session",
            )
            if choice == "q":
                resume_quit = True
            elif choice == "c":
                counterfactual_target = resume_target

        while not resume_quit:
            current_initial_target = None
            visualization_path = None
            if manifest_targets is not None:
                if protocol_quota is not None:
                    if counterfactual_target is not None:
                        current_initial_target = counterfactual_target
                    else:
                        current_initial_target = _select_protocol_manifest_target(
                            manifest_targets,
                            protocol_quota,
                        )
                else:
                    current_initial_target = _select_manifest_target(
                        manifest_targets,
                        consumed_manifest_idxs,
                        success_counts,
                        args.target_success_per_arm,
                    )
                next_arm = (
                    None
                    if current_initial_target is None
                    else next(arm for arm in arms if arm.key == current_initial_target.source)
                )
            else:
                next_arm = _select_arm(
                    arms,
                    success_counts,
                    args.target_success_per_arm,
                    shuffle_seed=args.shuffle_seed,
                    episode_index=saved_episode_count,
                )
            if next_arm is None:
                reset_coordinator.finish_if_pending("before collection stop")
                if protocol_quota is None:
                    print("Target successful counts reached for all arms.")
                else:
                    print("Target protocol quotas reached.")
                ui.set_phase("Finishing session", "All targets reached. Finalizing the dataset.")
                break
            if not reset_coordinator.has_pending:
                reset_coordinator.start("before episode")
            active = arm_states[next_arm.key]
            is_counterfactual = counterfactual_target is not None
            # Operator-facing start label: a running count of UNIQUE fresh starts
            # done so far + this one, rather than the (non-sequential, possibly
            # scrambled) manifest row index. A CF replay re-runs an already-counted
            # start, so it shows the count without the +1 (the "RE-ROLLOUT / CF"
            # mode label disambiguates).
            if current_initial_target is not None:
                n_unique_starts_done = (
                    len(protocol_quota.fresh_consumed_manifest_idxs)
                    if protocol_quota is not None
                    else len(consumed_manifest_idxs)
                )
                start_ordinal = (
                    n_unique_starts_done if is_counterfactual else n_unique_starts_done + 1
                )
                target_display_label = f"start {start_ordinal:03d}"
            else:
                target_display_label = f"target {saved_episode_count + 1:03d}"

            print("\n" + "=" * 60)
            print(f"Episode {saved_episode_count + 1}")
            if protocol_quota is None:
                print(
                    "Successful counts: "
                    + ", ".join(
                        f"arm{idx}={success_counts[arm.key]}/{args.target_success_per_arm}"
                        for idx, arm in enumerate(arms)
                    )
                )
            else:
                print(f"Protocol quotas: {_format_protocol_quota_totals(protocol_quota)}")
                print(f"Episode mode: {'counterfactual replay' if is_counterfactual else 'fresh'}")
            if current_initial_target is not None:
                print("Manual initial-state setup:")
                print(
                    "  "
                    + _format_initial_state_target(
                        current_initial_target,
                        display_label=target_display_label,
                    )
                )
            # Coordinate-free card (the operator places objects from the diagram); the
            # panel header around it carries the live quota / pace / ETA cells.
            visualization_path = ui.show_collection_scene(
                CollectionScene(
                    start_label=target_display_label,
                    mode="CF replay" if is_counterfactual else "First rollout",
                    episode_num=saved_episode_count + 1,
                    target=current_initial_target,
                ),
                manifest_meta,
                task_name=args.task_name,
                style=CardStyle(
                    mode_label="RE-ROLLOUT / CF" if is_counterfactual else "FIRST ROLLOUT",
                    display_label=target_display_label,
                    show_coordinates=False,
                ),
            )
            print("=" * 60)

            while True:
                episode_outcome = None
                accumulated_data = None
                accumulated_sources: list[int] = []
                current_gripper_action = None
                segment_count = 0
                had_intervention = False
                intervention_count = 0
                policy_steps_per_segment: list[int] = []
                subtask_frames: list[int] = []
                live_episode_started = False
                live_frame_count = 0
                live_cam_data_keys: list[str] = []

                def mark_subgoal() -> None:
                    # live_frame_count is the number of frames of THIS attempt already in
                    # the live buffer, across every policy / human segment so far: the mark
                    # lands on the most recent recorded frame, as in the eval rollout.
                    if record_subtask_mark(
                        subtask_frames, step=live_frame_count, subtask_marks=num_subtask_marks
                    ):
                        ui.progress.marks = len(subtask_frames)

                def initialize_live_dataset(obs_for_schema: dict) -> None:
                    nonlocal dataset, manifest_sha256, manifest_snapshot_path
                    nonlocal live_cam_data_keys
                    live_cam_data_keys = sorted(f"image_{cam_key}" for cam_key in all_camera_keys)
                    if not live_cam_data_keys:
                        raise RuntimeError("Cannot initialize live dataset before camera discovery")

                    schema_episode_data = _build_dataset_schema_probe(
                        obs_for_schema,
                        all_camera_keys,
                        camera_height=args.camera_height,
                        camera_width=args.camera_width,
                    )
                    schema_cam_data_keys = sorted(
                        key for key in schema_episode_data if key.startswith("image_")
                    )
                    if schema_cam_data_keys != live_cam_data_keys:
                        raise RuntimeError(
                            "Live schema camera keys disagree with selected camera keys: "
                            f"schema={schema_cam_data_keys}, selected={live_cam_data_keys}"
                        )

                    dataset = _load_or_create_dataset(
                        dataset,
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                        episode_data=schema_episode_data,
                        cam_data_keys=live_cam_data_keys,
                        freq=args.freq,
                        manifest_meta=manifest_meta if current_initial_target is not None else None,
                        streaming_encoding=streaming_encoding,
                    )
                    if args.initial_states_manifest is not None:
                        manifest_snapshot_path, manifest_sha256 = _ensure_manifest_snapshot(
                            args.initial_states_manifest,
                            dataset_path,
                        )
                    if dataset.num_episodes != saved_episode_count:
                        raise RuntimeError(
                            f"Dataset episode count {dataset.num_episodes} does not match "
                            f"ledger count {saved_episode_count}; refusing to append ambiguous data"
                        )
                    _assert_writer_episode_size(dataset, 0)

                def discard_live_episode_buffer(reason: str) -> None:
                    nonlocal live_episode_started, live_frame_count
                    if dataset is not None and live_episode_started:
                        print(f"Discarding live LeRobot episode buffer ({reason}).")
                        dataset.clear_episode_buffer(delete_images=True)
                    live_episode_started = False
                    live_frame_count = 0

                def record_live_step(
                    segment_data: dict,
                    frame_index: int,
                    source_id: int,
                ) -> None:
                    nonlocal live_episode_started, live_frame_count
                    if dataset is None:
                        raise RuntimeError("Live dataset was not initialized before recording")
                    current_cam_keys = sorted(
                        key for key in segment_data if key.startswith("image_")
                    )
                    if current_cam_keys != live_cam_data_keys:
                        raise RuntimeError(
                            "Segment camera keys changed during collection: "
                            f"current={current_cam_keys}, expected={live_cam_data_keys}"
                        )
                    _assert_writer_episode_size(dataset, live_frame_count)
                    extra_frame_fields = _constant_extra_frame_fields(
                        arm_state=active,
                        target=current_initial_target,
                        manifest_meta=manifest_meta if current_initial_target is not None else None,
                    )
                    _add_live_frame_to_dataset(
                        dataset,
                        segment_data,
                        frame_index,
                        camera_keys=live_cam_data_keys,
                        task_name=args.task_name,
                        source_id=int(source_id),
                        extra_frame_fields=extra_frame_fields,
                    )
                    live_episode_started = True
                    live_frame_count += 1

                if reset_coordinator.has_pending and current_initial_target is not None:
                    print(
                        "Target card is ready. Running robot reset now; keep clear of "
                        "the arm until reset finishes.",
                        flush=True,
                    )
                obs = reset_coordinator.finish("before episode start")
                if current_initial_target is not None:
                    setup_subject = _initial_state_setup_subject(
                        current_initial_target, args.task_name
                    )

                    def _operator_setup_reset() -> None:
                        nonlocal obs
                        reset_coordinator.start("operator setup reset")
                        obs = reset_coordinator.finish("operator setup reset")

                    # THE gate the drain inside ui.gate exists for: the episode's frames
                    # are stamped with the target pose, so a key buffered during
                    # save_episode/reset that is consumed here starts the rollout before
                    # the object has been moved -- poisoning the ledger with a start state
                    # the robot never saw.
                    gate = ui.gate(
                        prompt=f"Place the {setup_subject} at the shown target",
                        on_reset=_operator_setup_reset,
                    )
                    if gate is GateOutcome.QUIT:
                        episode_outcome = "quit"
                        break
                    obs = env.get_observation()

                active.policy.reset()

                if not camera_keys:
                    all_cams = sorted(obs.get("image", {}).keys())
                    if requested_camera_keys is None:
                        selected, all_cams = select_image_camera_keys(
                            obs,
                            args.camera_filter,
                            context="blind DAgger camera discovery",
                        )
                    else:
                        missing = [k for k in requested_camera_keys if k not in all_cams]
                        if missing:
                            raise RuntimeError(
                                f"Requested cameras missing: {missing}; available={all_cams}"
                            )
                        excluded = sorted(
                            set(requested_camera_keys) & set(DEFAULT_EXCLUDED_CAMERA_KEYS)
                        )
                        if excluded:
                            raise RuntimeError(
                                f"Requested cameras include excluded camera(s) {excluded}; "
                                "these cameras must not be stored in real-world datasets."
                            )
                        selected = list(requested_camera_keys)
                    camera_keys.extend(selected)
                    all_camera_keys.extend(selected)
                    print(f"Cameras discovered: {all_cams}")
                    print(f"Cameras selected: {camera_keys}")
                    for state in arm_states.values():
                        if hasattr(state.policy, "set_camera_keys"):
                            state.policy.set_camera_keys(
                                policy_live_camera_keys(state.policy, camera_keys)
                            )

                initialize_live_dataset(obs)
                # max_steps bounds each POLICY segment, not the episode; the panel counts
                # every recorded frame of the attempt, so it shows a plain step count.
                ui.begin_rollout(
                    0,
                    num_subtask_marks,
                    phase="Human correction" if is_counterfactual else "Policy running",
                )

                if is_counterfactual:
                    segment_count += 1
                    print("Counterfactual replay: starting with human control.")
                    seg_data, action_str, gripper = spacemouse_correction_segment(
                        env,
                        spacemouse,
                        ui,
                        all_camera_keys,
                        freq=args.freq,
                        save_camera_height=args.camera_height,
                        save_camera_width=args.camera_width,
                        initial_gripper_action=current_gripper_action,
                        allow_reset=True,
                        record_step_fn=record_live_step,
                        on_subgoal=mark_subgoal,
                    )
                    if action_str == "reset":
                        episode_outcome = "reset"
                    else:
                        if len(seg_data["actions"]) > 0:
                            accumulated_data, seg_sources = accumulate_segment_data(
                                accumulated_data,
                                seg_data,
                                DataSource.HUMAN,
                            )
                            accumulated_sources.extend(seg_sources)
                        current_gripper_action = gripper
                        if action_str in {"success", "timeout", "failure", "discard", "quit"}:
                            episode_outcome = action_str

                while episode_outcome is None:
                    segment_count += 1
                    seg_data, action_str, gripper = policy_rollout_segment(
                        env,
                        active.policy,
                        ui,
                        all_camera_keys,
                        freq=args.freq,
                        save_camera_height=args.camera_height,
                        save_camera_width=args.camera_width,
                        max_steps=args.max_steps,
                        initial_gripper_action=current_gripper_action,
                        allow_reset=True,
                        record_step_fn=record_live_step,
                        on_subgoal=mark_subgoal,
                    )
                    if action_str == "reset":
                        episode_outcome = "reset"
                        break
                    if len(seg_data["actions"]) > 0:
                        policy_steps_per_segment.append(len(seg_data["actions"]))
                        accumulated_data, seg_sources = accumulate_segment_data(
                            accumulated_data,
                            seg_data,
                            DataSource.AUTONOMOUS,
                        )
                        accumulated_sources.extend(seg_sources)
                    current_gripper_action = gripper

                    if action_str == "intervention":
                        had_intervention = True
                        intervention_count += 1
                        seg_data, action_str, gripper = spacemouse_correction_segment(
                            env,
                            spacemouse,
                            ui,
                            all_camera_keys,
                            freq=args.freq,
                            save_camera_height=args.camera_height,
                            save_camera_width=args.camera_width,
                            initial_gripper_action=current_gripper_action,
                            allow_reset=True,
                            record_step_fn=record_live_step,
                            on_subgoal=mark_subgoal,
                        )
                        if action_str == "reset":
                            episode_outcome = "reset"
                            break
                        if len(seg_data["actions"]) > 0:
                            accumulated_data, seg_sources = accumulate_segment_data(
                                accumulated_data,
                                seg_data,
                                DataSource.HUMAN,
                            )
                            accumulated_sources.extend(seg_sources)
                        current_gripper_action = gripper
                        if action_str == "continue":
                            active.policy.reset()
                            continue

                    if action_str in {"success", "timeout", "failure", "discard", "quit"}:
                        episode_outcome = action_str
                        break

                if episode_outcome == "reset":
                    discard_live_episode_buffer("operator reset")
                    print(
                        "\nTrajectory reset requested. Discarding the in-memory attempt "
                        "and retrying the same target without saving."
                    )
                    ui.end_rollout(
                        "reset",
                        phase="Retrying the same target",
                        detail="Attempt discarded, nothing saved. Robot reset queued.",
                    )
                    reset_coordinator.start("operator trajectory reset")
                    continue
                break

            if episode_outcome == "quit" and segment_count == 0:
                break
            if episode_outcome in {"discard", "quit"}:
                discard_live_episode_buffer(str(episode_outcome))

            is_success = episode_outcome == "success"
            should_save = (
                episode_outcome in {"success", "timeout", "failure"}
                and accumulated_data is not None
                and len(accumulated_sources) > 0
            )
            total_steps = len(accumulated_sources) if accumulated_data else 0
            print(
                f"\nEpisode {saved_episode_count + 1}: "
                f"{(episode_outcome or 'unknown').upper()} "
                f"({total_steps} steps, {segment_count} segments)"
            )
            if segment_count:
                ui.end_rollout(
                    str(episode_outcome),
                    phase="Saving episode" if should_save else "Episode not saved",
                    detail=(
                        f"Episode {saved_episode_count + 1}: {str(episode_outcome).upper()} "
                        f"({total_steps} steps, {segment_count} segments)"
                        + (". Encoding video..." if should_save else "")
                    ),
                )
            end_reset_started = False

            if should_save:
                final_obs = env.get_observation()
                reset_coordinator.start("end of episode")
                end_reset_started = True
                final_state = np.concatenate(
                    [
                        np.array(final_obs["robot_state"]["cartesian_position"], dtype=np.float32),
                        np.array([final_obs["robot_state"]["gripper_position"]], dtype=np.float32),
                    ]
                )
                accumulated_data["observations"].append(final_state.copy())
                accumulated_data["joint_positions"].append(
                    np.array(final_obs["robot_state"]["joint_positions"], dtype=np.float32)
                )
                for cam_key in all_camera_keys:
                    img = process_image(
                        final_obs["image"][cam_key], args.camera_height, args.camera_width
                    )
                    accumulated_data[f"image_{cam_key}"].append(img)

                finalize_episode_data(
                    accumulated_data,
                    final_obs,
                    is_success=is_success,
                    is_terminal=episode_outcome in {"success", "failure"},
                    subtask_frames=subtask_frames,
                )
                accumulated_sources.append(accumulated_sources[-1])
                cam_data_keys = sorted(k for k in accumulated_data if k.startswith("image_"))
                if dataset is None:
                    raise RuntimeError("Live dataset was not initialized before episode save")
                if not live_episode_started:
                    raise RuntimeError(
                        "No live LeRobot frames were recorded for a non-empty saved episode"
                    )
                if cam_data_keys != live_cam_data_keys:
                    raise RuntimeError(
                        "Final episode camera keys disagree with live buffer keys: "
                        f"final={cam_data_keys}, live={live_cam_data_keys}"
                    )

                terminal_frame_index = len(accumulated_data["actions"]) - 1
                _assert_writer_episode_size(dataset, live_frame_count)
                _add_live_frame_to_dataset(
                    dataset,
                    accumulated_data,
                    terminal_frame_index,
                    camera_keys=live_cam_data_keys,
                    task_name=args.task_name,
                    source_id=int(accumulated_sources[-1]),
                    extra_frame_fields=_constant_extra_frame_fields(
                        arm_state=active,
                        target=current_initial_target,
                        manifest_meta=manifest_meta if current_initial_target is not None else None,
                    ),
                )
                live_frame_count += 1
                intervention_flags = _patch_live_episode_buffer_for_final_outcome(
                    dataset,
                    accumulated_data,
                    sources=accumulated_sources,
                    episode_success=is_success,
                )

                episode_index = saved_episode_count
                if current_initial_target is not None and (
                    manifest_snapshot_path is None or manifest_sha256 is None
                ):
                    raise RuntimeError(
                        "Initial-state episode is ready to save but manifest snapshot metadata "
                        "is missing"
                    )
                print("Finalizing episode with LeRobot streaming encoder...")
                dataset.save_episode(parallel_encoding=False)
                live_episode_started = False
                live_frame_count = 0

                policy_frames = sum(
                    1 for source in accumulated_sources if source == DataSource.AUTONOMOUS
                )
                human_frames = sum(
                    1 for source in accumulated_sources if source == DataSource.HUMAN
                )
                has_human_frames = human_frames > 0
                status_str = "SUCCESS" if is_success else "FAILURE"
                print(
                    f"Episode saved as {status_str} "
                    f"(policy={policy_frames}, human={human_frames}, "
                    f"interventions={sum(intervention_flags)}, "
                    f"total saved={saved_episode_count + 1})"
                )

                if protocol_quota is not None:
                    if current_initial_target is None:
                        raise RuntimeError("protocol quota save requires an initial-state target")
                    quota_row = protocol_quota.credit_episode(
                        manifest_idx=current_initial_target.manifest_idx,
                        episode_index=episode_index,
                        success=bool(is_success),
                        is_counterfactual=bool(is_counterfactual),
                        matched_distance=0.0,
                        quota_credit=(
                            episode_outcome != "failure"
                            if args.quota_credit_mode == "saved"
                            else None
                        ),
                        extra={
                            "arm_key": next_arm.key,
                            "arm_id": active.arm_id,
                            "model_id": next_arm.model_id,
                            "outcome": episode_outcome,
                            "steps": int(total_steps),
                            "segments": int(segment_count),
                            "had_intervention": bool(had_intervention),
                            "human_frames": int(human_frames),
                            "has_human_frames": bool(has_human_frames),
                            "intervention_count": int(intervention_count),
                            "policy_steps_per_segment": [int(x) for x in policy_steps_per_segment],
                            "subtask_frames": [int(f) for f in subtask_frames],
                            "task_name": args.task_name,
                            "dataset_name": dataset_name,
                            "shuffle_seed": int(args.shuffle_seed),
                            "manifest_source_index": current_initial_target.source_index,
                            "manifest_snapshot_path": str(manifest_snapshot_path),
                            "manifest_sha256": manifest_sha256,
                            "protocol_arm_caps": dict(protocol_quota.protocol_arm_targets),
                            "initial_state_visualization_path": str(visualization_path),
                            **_target_ledger_fields(current_initial_target, manifest_meta),
                        },
                        write_ledger=False,
                    )
                    protocol_quota.append_reserved_row(quota_row)
                    credited_units = sum(
                        len(arms) for arms in quota_row["credited_protocol_arms"].values()
                    )
                    print(
                        "Protocol quota ledger: "
                        f"credited {credited_units} protocol-arm unit(s), "
                        f"{_format_protocol_quota_totals(protocol_quota)}"
                    )
                else:
                    row = {
                        "episode_index": episode_index,
                        "arm_key": next_arm.key,
                        "arm_id": active.arm_id,
                        "model_id": next_arm.model_id,
                        "success": bool(is_success),
                        "outcome": episode_outcome,
                        "steps": int(total_steps),
                        "segments": int(segment_count),
                        "had_intervention": bool(had_intervention),
                        "human_frames": int(human_frames),
                        "has_human_frames": bool(has_human_frames),
                        "task_name": args.task_name,
                        "dataset_name": dataset_name,
                        "shuffle_seed": int(args.shuffle_seed),
                        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "is_counterfactual": bool(is_counterfactual),
                        "intervention_count": int(intervention_count),
                        "policy_steps_per_segment": [int(x) for x in policy_steps_per_segment],
                        "subtask_frames": [int(f) for f in subtask_frames],
                    }
                    if current_initial_target is not None:
                        row.update(
                            {
                                "manifest_idx": current_initial_target.manifest_idx,
                                "manifest_source_index": current_initial_target.source_index,
                                "manifest_snapshot_path": str(manifest_snapshot_path),
                                "manifest_sha256": manifest_sha256,
                                "initial_state_visualization_path": str(visualization_path),
                                **_target_ledger_fields(current_initial_target, manifest_meta),
                            }
                        )
                    _append_ledger_row(ledger_path, row)

                saved_episode_count += 1
                if current_initial_target is not None and protocol_quota is None:
                    consumed_manifest_idxs.add(current_initial_target.manifest_idx)
                if is_success and protocol_quota is None:
                    success_counts[next_arm.key] += 1
                if is_success:
                    session_successes += 1
                refresh_collection_status()

            elif episode_outcome == "discard":
                print("Episode discarded -- not saved")

            if not end_reset_started:
                reset_coordinator.start("end of episode")

            if episode_outcome == "quit":
                reset_coordinator.finish("before quit")
                break

            if protocol_quota is not None:
                remaining_total, remaining_eligible = _count_protocol_remaining_targets(
                    manifest_targets,
                    protocol_quota,
                )
                for line in protocol_quota.progress_lines(
                    saved_episode_count=saved_episode_count,
                    total_manifest_remaining=remaining_total,
                    eligible_manifest_remaining=remaining_eligible,
                ):
                    print(line)

                if episode_outcome == "discard":
                    if is_counterfactual:
                        counterfactual_target = current_initial_target
                    else:
                        counterfactual_target = None
                    choice = ui.choose(
                        prompt="\nEpisode discarded",
                        choices={"n": "retry the same target", "q": "quit"},
                        default="n",
                        phase="Episode discarded",
                    )
                    if choice == "q":
                        episode_outcome = "quit"
                        reset_coordinator.finish("before quit")
                        break
                    continue

                can_cf = (
                    is_success
                    and should_save
                    and current_initial_target is not None
                    and protocol_quota.can_accept_counterfactual(
                        current_initial_target.manifest_idx
                    )
                )
                choices = {}
                if can_cf:
                    choices["c"] = "collect a counterfactual replay of this start"
                choices["n"] = "move on to the next target"
                choices["q"] = "quit"
                choice = ui.choose(prompt="\nEpisode saved", choices=choices, default="n")
                counterfactual_target = current_initial_target if choice == "c" else None
                if choice == "q":
                    episode_outcome = "quit"
                    reset_coordinator.finish("before quit")
                    break
                continue

            if current_initial_target is not None:
                continue

            choice = ui.choose(
                prompt="\nReady for the next episode",
                choices={"n": "start the next episode", "q": "quit"},
                default="n",
            )
            if choice == "q":
                episode_outcome = "quit"
                reset_coordinator.finish("before quit")
                break

        # Reached only by falling out of the collection loop, i.e. through one of its
        # deliberate breaks (operator quit, targets/quotas reached).
        # A KeyboardInterrupt / SIGTERM / crash jumps straight to `finally` instead, and
        # end-of-session consolidation is gated on this flag.
        session_ended_intentionally = True

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
    finally:
        active_exc = sys.exc_info()[1]
        reset_shutdown_exc: BaseException | None = None
        try:
            reset_coordinator.finish_if_pending("shutdown")
        except BaseException as exc:
            if reset_coordinator.has_pending:
                print(
                    "WARNING: interrupted while waiting for pending robot reset during "
                    "shutdown; waiting again so the reset result is not dropped."
                )
                try:
                    reset_coordinator.finish_if_pending("shutdown after interrupted wait")
                except BaseException as retry_exc:
                    reset_shutdown_exc = retry_exc
                    print(f"ERROR: pending robot reset failed during shutdown: {retry_exc}")
                else:
                    reset_shutdown_exc = exc
            else:
                reset_shutdown_exc = exc
                print(f"ERROR: pending robot reset failed during shutdown: {exc}")
        finally:
            reset_coordinator.shutdown()

        cleanup_exc: BaseException | None = None
        try:
            ui.set_phase(
                "Finishing session",
                f"{saved_episode_count} episode(s) saved. Finalizing the dataset.",
            )
            if dataset is not None:
                writer = getattr(dataset, "writer", None)
                if writer is not None and writer.episode_buffer is not None:
                    pending_frames = int(writer.episode_buffer.get("size", 0))
                    if pending_frames > 0:
                        print(
                            "Discarding incomplete live LeRobot episode buffer before finalizing "
                            f"({pending_frames} frame(s))."
                        )
                        dataset.clear_episode_buffer(delete_images=True)

                print(f"Finalizing dataset... ({saved_episode_count} episodes saved)")
                removed_targets = cleanup_unreferenced_initial_state_cards(
                    ui.card_dir,
                    args.protocol_quota_ledger if protocol_quota is not None else ledger_path,
                )
                if removed_targets:
                    print(f"Removed {len(removed_targets)} unreferenced initial-state target cards")
                dataset.stop_image_writer()
                dataset.finalize()

                # Consolidation DELETES the per-episode parquet shards, so it runs only
                # when the session really is over (a deliberate exit, not an interrupt).
                if session_ended_intentionally:
                    from mulligan.data.recording import consolidate_episodes_parquet

                    consolidate_episodes_parquet(dataset_path)
                else:
                    print(
                        "Skipping end-of-session parquet consolidation: this exit was "
                        "interrupted. The per-episode shards are kept so a restart "
                        "resumes cleanly."
                    )

                if args.push_to_hub:
                    print("\nPushing mixed dataset to HuggingFace Hub...")
                    hub_api = HfApi()
                    repo_id = resolve_push_repo_id(
                        args.push_repo_id or dataset_name, args.hf_namespace
                    )
                    if ensure_dataset_repo(hub_api, repo_id, private=args.private):
                        print(f"Created repository: {repo_id}")
                    dataset.repo_id = repo_id
                    from mulligan.tools.lerobot_hub import push_lerobot_dataset_tagged_main

                    try:
                        push_lerobot_dataset_tagged_main(
                            dataset, private=args.private, license=args.license
                        )
                    except Exception as exc:  # noqa: BLE001 — the DATA is already safe on disk
                        # The push is the LAST thing a robot day does; failing it hard here
                        # buries the local dataset's own "everything is finalized" report
                        # under a traceback and leaves the operator with no next step.
                        print(
                            f"\nERROR: HuggingFace push failed: {exc!r}\n"
                            "The local dataset is finalized and intact -- nothing was lost. "
                            "Re-push it by hand with:\n"
                            f"  .venv/bin/python -m mulligan.tools.push_dataset --root "
                            f"{dataset_path} --repo-id {repo_id}"
                            f"{' --private' if args.private else ''}\n"
                        )
                    else:
                        print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")

            if protocol_quota is None:
                print(f"Final successful counts: {dict(success_counts)}")
            else:
                print(f"Final protocol quotas: {_format_protocol_quota_totals(protocol_quota)}")
            ui.close()
            if spacemouse is not None:
                spacemouse.close()
        except BaseException as exc:
            cleanup_exc = exc
            if active_exc is None and reset_shutdown_exc is None:
                raise
            print(f"ERROR: cleanup failed while another exception is pending: {exc}")
        if reset_shutdown_exc is not None:
            if cleanup_exc is not None and hasattr(reset_shutdown_exc, "add_note"):
                reset_shutdown_exc.add_note(f"cleanup also failed: {cleanup_exc!r}")
            raise reset_shutdown_exc


if __name__ == "__main__":
    main(_CLI_ARGS)
