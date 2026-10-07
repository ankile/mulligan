#!/usr/bin/env python3
"""Split a protocol-quota DAgger collection into protocol/arm datasets.

The source of truth is the JSONL ledger written by
``mulligan.sim.collect.dagger --adaptive-protocol-quota-*`` or the real robot
blind DAgger collector. Fresh credited episodes may be copied into both ``no_cf``
and ``with_cf`` targets; CF replay episodes are copied only into the credited
``with_cf`` targets. By default, only successful episodes may have credits; real
robot collections can pass ``--credit-outcome-mode saved`` to split every saved
valid trajectory while preserving the success/failure labels.

Example:
    python -m mulligan.data.split_protocol_quota \\
        --source-repo <user>/square-narrow-dagger-r5 \\
        --manifest path/to/manifest.json \\
        --ledger path/to/protocol_quota_ledger.jsonl \\
        --target no_cf.baseline_uniform=<user>/square-narrow-baseline-r5-no-cf \\
        --target with_cf.baseline_uniform=<user>/square-narrow-baseline-r5-with-cf \\
        --output-root ./data/protocol_quota_split \\
        --push
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.split_copy import (
    VISUAL_DTYPES,
    copy_episode,
    create_target_dataset,
    last_frame_success,
)
from mulligan.data.task_names import validate_lerobot_task_name
from mulligan.real.eval.outcome_results import valid_prefix_length
from mulligan.real.lifecycle.tasks import (
    RealTaskSpec,
    find_task_spec_by_task_name,
    registered_task_specs,
)
from mulligan.tools.dataset_lineage import (
    make_parent_lineage,
    make_split_lineage,
    push_local_lineage,
    sha256_file,
    write_local_lineage,
)
from mulligan.tools.lerobot_hub import (
    push_lerobot_dataset_replacing_remote,
    refresh_lerobot_dataset_from_main,
)
from mulligan.tools.lerobot_fast_split import fast_split_dataset, split_video_encoder
from mulligan.sim.collect.quota import ProtocolQuotaLedger, parse_arm_target_caps
from mulligan.sim.envs import SIM_TASK_ENV_NAMES
from mulligan.utils.state_to_grid import extract_sampler_state_from_env_state


TASK_NAME_FOR_EXTRACTOR = SIM_TASK_ENV_NAMES


def _parse_protocol_targets(spec: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--expected-per-protocol entries must be NAME=INT; got {item!r}"
            )
        name, value = item.split("=", 1)
        targets[name.strip()] = int(value)
    if not targets:
        raise argparse.ArgumentTypeError("--expected-per-protocol must not be empty")
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
                f"--expected-protocol-arms entries must be NAME=ARM[,ARM...]; got {item!r}"
            )
        protocol, raw_arms = item.split("=", 1)
        protocol = protocol.strip()
        arms = [arm.strip() for arm in raw_arms.split(",") if arm.strip()]
        if not protocol or not arms:
            raise argparse.ArgumentTypeError(
                f"--expected-protocol-arms has empty protocol or arms; got {item!r}"
            )
        if protocol in arms_by_protocol:
            raise argparse.ArgumentTypeError(
                f"duplicate --expected-protocol-arms protocol {protocol!r}"
            )
        if len(set(arms)) != len(arms):
            raise argparse.ArgumentTypeError(
                f"duplicate arm in --expected-protocol-arms for {protocol!r}: {arms}"
            )
        arms_by_protocol[protocol] = arms
    if not arms_by_protocol:
        raise argparse.ArgumentTypeError("--expected-protocol-arms must not be empty if set")
    return arms_by_protocol


def _parse_target(spec: str) -> tuple[str, str, str, str | None]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--target must be PROTOCOL.ARM=REPO[:ROOT]; got {spec!r}")
    lhs, rhs = spec.split("=", 1)
    if "." not in lhs:
        raise argparse.ArgumentTypeError(f"--target left side must be PROTOCOL.ARM; got {lhs!r}")
    protocol, arm = [x.strip() for x in lhs.split(".", 1)]
    if ":" in rhs:
        repo, root = rhs.split(":", 1)
    else:
        repo, root = rhs, None
    repo = repo.strip()
    root = root.strip() if root else None
    if not protocol or not arm or not repo:
        raise argparse.ArgumentTypeError(f"--target has empty protocol, arm, or repo; got {spec!r}")
    return protocol, arm, repo, root


def _dataset_task_names(source: LeRobotDataset) -> set[str]:
    tasks = getattr(source.meta, "tasks", None)
    if tasks is None or len(tasks) == 0:
        return set()
    if "name" in tasks:
        return {str(value) for value in tasks["name"].tolist()}
    return {str(value) for value in tasks.index.tolist()}


def _ensure_target_episode_visual_dirs(target: LeRobotDataset) -> None:
    """Pre-create temporary frame dirs before LeRobot image writes."""
    writer = getattr(target, "writer", None)
    if writer is None:
        raise RuntimeError(
            "split target dataset is read-only (missing writer); create targets with "
            "LeRobotDataset.create before copying episodes"
        )
    if getattr(writer, "_streaming_encoder", None) is not None:
        return
    episode_buffer = writer.episode_buffer
    episode_index = episode_buffer["episode_index"]
    if isinstance(episode_index, np.ndarray):
        episode_index = int(episode_index.item() if episode_index.size == 1 else episode_index[0])
    else:
        episode_index = int(episode_index)
    for key, fmeta in target.features.items():
        if fmeta.get("dtype") in VISUAL_DTYPES:
            writer._get_image_file_dir(episode_index, key).mkdir(parents=True, exist_ok=True)


def _copy_episode(
    source: LeRobotDataset,
    ep_idx: int,
    target: LeRobotDataset,
    *,
    drop_visual_features: bool = False,
) -> None:
    if not drop_visual_features:
        _ensure_target_episode_visual_dirs(target)
    # Non-visual rows are read by position (no global-index lookup).
    copy_episode(
        source, ep_idx, target, drop_visual_features=drop_visual_features, frame_lookup=None
    )


def _summarize_saved_trajectory(
    ep_idx: int,
    *,
    source_values: list[int],
    intervention_values: list[int],
    valid_values: list[int],
    done_values: list[int],
    success_values: set[int],
) -> dict:
    """Reduce an episode's per-frame arrays to saved-trajectory accounting.

    Frame counts are truncated to the outcome-edited ``is_valid`` prefix: only
    frames up to the task-completion / done point are counted for the human vs
    policy vs intervention accounting. The invalid suffix (terminal padding plus
    any soft-truncated post-outcome retract/reset junk) is physically retained
    in the split dataset for video rendering but excluded from every count here.
    ``saved_frames`` stays the raw stored span so the dataset size is still
    visible; ``saved_valid_frames`` and the policy/human/intervention counts are
    the effective (truncated) accounting quantities.
    """
    n_frames = len(valid_values)
    assert (
        len(source_values) == n_frames
        and len(intervention_values) == n_frames
        and len(done_values) == n_frames
    ), f"episode {ep_idx}: per-frame array length mismatch"

    if success_values not in ({0}, {1}):
        raise ValueError(f"episode {ep_idx}: mixed success values {success_values}")

    valid_len = valid_prefix_length(valid_values, episode_index=ep_idx)

    # Count human/policy/intervention over the valid prefix only (cut off at the
    # done/outcome frame), never over the retained invalid suffix.
    source_counts: Counter[int] = Counter(source_values[:valid_len])
    intervention_count = int(sum(intervention_values[:valid_len]))

    success = success_values == {1}
    human_frames = int(source_counts[1])
    policy_frames = int(source_counts[0])
    if human_frames + policy_frames != valid_len:
        raise ValueError(
            f"episode {ep_idx}: policy+human frames {human_frames + policy_frames} "
            f"!= valid frames {valid_len}"
        )
    if human_frames == 0 and intervention_count != 0:
        raise ValueError(
            f"episode {ep_idx}: policy-only saved trajectory has "
            f"{intervention_count} saved intervention flags"
        )
    if human_frames == 0:
        trajectory_kind = "policy_only_success" if success else "policy_only_failure"
    else:
        trajectory_kind = "mixed_human_success" if success else "mixed_human_failure"

    return {
        "saved_frames": int(n_frames),
        "saved_valid_frames": int(valid_len),
        "saved_policy_frames": policy_frames,
        "saved_human_frames": human_frames,
        "saved_intervention_count": intervention_count,
        "saved_success": success,
        "saved_final_done": bool(done_values[valid_len - 1]),
        "saved_trajectory_kind": trajectory_kind,
    }


def _episode_saved_trajectory_stats(source: LeRobotDataset, ep_idx: int) -> dict:
    ep_meta = source.meta.episodes[ep_idx]
    start = int(ep_meta["dataset_from_index"])
    end = int(ep_meta["dataset_to_index"])
    if end <= start:
        raise ValueError(f"episode {ep_idx} has empty frame range [{start}, {end})")

    source_values: list[int] = []
    intervention_values: list[int] = []
    success_values: set[int] = set()
    valid_values: list[int] = []
    done_values: list[int] = []
    for frame_idx in range(start, end):
        item = source.hf_dataset[frame_idx]
        source_id = int(np.asarray(item["source"]).item())
        if source_id not in {0, 1}:
            raise ValueError(f"episode {ep_idx} frame {frame_idx}: invalid source {source_id}")
        source_values.append(source_id)

        intervention = int(np.asarray(item["intervention"]).item())
        if intervention not in {0, 1}:
            raise ValueError(
                f"episode {ep_idx} frame {frame_idx}: invalid intervention {intervention}"
            )
        intervention_values.append(intervention)

        success_values.add(int(np.asarray(item["success"]).item()))
        valid_values.append(int(np.asarray(item["is_valid"]).item()))
        done_values.append(int(np.asarray(item["done"]).item()))

    return _summarize_saved_trajectory(
        ep_idx,
        source_values=source_values,
        intervention_values=intervention_values,
        valid_values=valid_values,
        done_values=done_values,
        success_values=success_values,
    )


def _load_credited_rows(path: Path, *, credit_outcome_mode: str) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        # These keys are written unconditionally by ProtocolQuotaLedger.credit_episode;
        # a missing one means a malformed/foreign ledger row, which must fail loudly
        # (never silently drop a credited episode from the split).
        credited = row["credited_protocol_arms"]
        success = bool(row["success"])
        if credit_outcome_mode == "success" and not success and credited:
            raise SystemExit(f"{path}:{line_no}: non-success row has credits: {credited}")
        if credited and (credit_outcome_mode == "saved" or success):
            rows.append(row)
    return rows


def _unique_string_values(rows: list[dict], key: str) -> list[str]:
    return sorted(
        {
            str(row[key])
            for row in rows
            if key in row and row[key] is not None and str(row[key]).strip()
        }
    )


def _arm_to_model_id(rows: list[dict]) -> dict[str, str]:
    models_by_arm: dict[str, set[str]] = {}
    for row in rows:
        arm = row.get("arm_key")
        model_id = row.get("model_id")
        if arm is None or model_id is None:
            continue
        models_by_arm.setdefault(str(arm), set()).add(str(model_id))
    ambiguous = {arm: sorted(models) for arm, models in models_by_arm.items() if len(models) > 1}
    if ambiguous:
        raise SystemExit(f"Ledger has multiple model_id values for arm_key(s): {ambiguous}")
    return {arm: next(iter(models)) for arm, models in models_by_arm.items()}


def _validate_real_manifest_metadata(
    source: LeRobotDataset,
    ep_idx: int,
    row: dict,
    quota: ProtocolQuotaLedger,
    spec: RealTaskSpec,
) -> None:
    first = int(source.meta.episodes[ep_idx]["dataset_from_index"])
    item = source.hf_dataset[first]
    expected_idx = int(row["manifest_idx"])
    dataset_idx = int(np.asarray(item["manifest_idx"]).item())
    if dataset_idx != expected_idx:
        raise SystemExit(
            f"Ledger row episode_index={ep_idx} expected manifest_idx {expected_idx}, "
            f"but dataset first frame has manifest_idx {dataset_idx}"
        )
    state = quota.states[expected_idx]

    item_keys = set(item.keys())
    # Free 3-DOF state: real manifest-backed datasets store the task's own
    # manifest keys as per-frame features, e.g. square_d2 uses nut_*.
    for manifest_key in spec.state_keys:
        if manifest_key not in item_keys:
            raise SystemExit(
                f"Ledger row episode_index={ep_idx} manifest_idx={expected_idx} "
                f"cannot validate manifest key {manifest_key!r}: dataset first frame "
                "is missing that required per-frame feature"
            )
        dataset_value = float(np.asarray(item[manifest_key]).item())
        manifest_value = float(state[manifest_key])
        if not np.isclose(dataset_value, manifest_value, rtol=0.0, atol=1e-5):
            raise SystemExit(
                f"Ledger row episode_index={ep_idx} manifest_idx={expected_idx} "
                f"has dataset {manifest_key}={dataset_value:.8g}, expected "
                f"{manifest_value:.8g} from manifest key {manifest_key}"
            )
    # Sampled scene placements are part of the manifest-backed dataset schema.
    # Validate them when present so square_d2's sampled peg cannot drift.
    for placement in spec.sampled_placements.values():
        for manifest_key in placement.keys:
            if manifest_key not in item_keys:
                raise SystemExit(
                    f"Ledger row episode_index={ep_idx} manifest_idx={expected_idx} "
                    f"cannot validate sampled-placement key {manifest_key!r}: "
                    "dataset first frame is missing that feature"
                )
            dataset_value = float(np.asarray(item[manifest_key]).item())
            manifest_value = float(state[manifest_key])
            if not np.isclose(dataset_value, manifest_value, rtol=0.0, atol=1e-5):
                raise SystemExit(
                    f"Ledger row episode_index={ep_idx} manifest_idx={expected_idx} "
                    f"has dataset {manifest_key}={dataset_value:.8g}, expected "
                    f"{manifest_value:.8g} from manifest"
                )
    # Fixed scene objects are not stored per-frame; pin the manifest values to
    # the registry constants so a drifted manifest fails loudly here.
    for obj, (key_x, key_y) in spec.fixed_object_keys.items():
        expected_xy = spec.fixed_objects[obj]
        for key, expected_value in zip((key_x, key_y), expected_xy):
            manifest_value = float(state[key])
            if not np.isclose(manifest_value, expected_value, rtol=0.0, atol=1e-9):
                raise SystemExit(
                    f"Ledger row episode_index={ep_idx} manifest_idx={expected_idx} "
                    f"has manifest {key}={manifest_value:.8g}, expected fixed "
                    f"{obj} value {expected_value:.8g} from the task registry"
                )


def build_parser() -> argparse.ArgumentParser:
    """CLI surface, built separately so tests can assert on it without running a split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--expected-per-protocol",
        type=_parse_protocol_targets,
        default=_parse_protocol_targets("no_cf=100,with_cf=100"),
        help="Comma-separated protocol targets used to validate the ledger.",
    )
    parser.add_argument(
        "--protocol-quota-arm-caps",
        type=parse_arm_target_caps,
        default=None,
        help=(
            "Per-arm quota caps the collection ran with ('PROTOCOL.ARM=INT|collected[;...]'); "
            "'collected' resolves to that arm's credited count in the ledger."
        ),
    )
    parser.add_argument(
        "--expected-protocol-arms",
        type=_parse_protocol_arms,
        default=None,
        help=(
            "Optional semicolon-separated protocol arm allowlist used to "
            "validate asymmetric ledgers, e.g. "
            "'no_cf=baseline_uniform,sobol,mulligan;"
            "with_cf=sobol,mulligan'."
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        required=True,
        help="PROTOCOL.ARM=REPO[:ROOT]. Provide one per protocol/arm output.",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        default=None,
        help=(
            "Protocol(s) to split. Default: all protocols in the manifest/ledger. "
            "Use e.g. --protocol with_cf to emit only the with-CF arms while "
            "no-CF collection is still in progress."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--drop-visual-features",
        action="store_true",
        help=(
            "Create split datasets without image/video observation features. "
            "This skips source video decoding and target visual encoding."
        ),
    )
    parser.add_argument(
        "--credit-outcome-mode",
        choices=["success", "saved"],
        default="success",
        help=(
            "Validate credits as success-only (sim default) or any saved trajectory "
            "with success labels preserved (real robot protocol collections)."
        ),
    )
    parser.add_argument(
        "--copy-backend",
        choices=["native", "frame"],
        default="frame",
        help=(
            "Copy engine. 'frame' (default) re-writes each episode frame by frame "
            "and supports --drop-visual-features. 'native' uses the faster file-level "
            "LeRobot splitter (parquet copy + video segment re-encode), supports "
            "overlapping protocol views and --drop-columns / --fill-null-columns."
        ),
    )
    parser.add_argument(
        "--drop-columns",
        default=None,
        help="Comma-separated feature columns to drop from native split outputs.",
    )
    parser.add_argument(
        "--fill-null-columns",
        default=None,
        help="Comma-separated feature columns whose null cells are filled with NaN.",
    )
    parser.add_argument(
        "--forbid-video-padding",
        action="store_true",
        help=(
            "Native backend only: fail if any video timestamp span or decoded segment "
            "would require padding/truncation to match episode row length."
        ),
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete local target roots before creating split datasets.",
    )
    parser.add_argument(
        "--replace-remote-codec",
        action="store_true",
        help=(
            "Allow --push to overwrite an existing Hub dataset whose video codec differs "
            "from the codec this split writes (LeRobot's AV1 default). Off by default: a "
            "codec change rewrites every remote pixel under the same repo id."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    drop_columns = (
        [c.strip() for c in args.drop_columns.split(",") if c.strip()]
        if args.drop_columns
        else None
    )
    fill_null_columns = (
        [c.strip() for c in args.fill_null_columns.split(",") if c.strip()]
        if args.fill_null_columns
        else None
    )
    if args.copy_backend == "native" and args.drop_visual_features:
        raise SystemExit("--drop-visual-features requires --copy-backend frame")
    if (drop_columns or fill_null_columns) and args.copy_backend != "native":
        raise SystemExit("--drop-columns / --fill-null-columns require --copy-backend native")
    if drop_columns and fill_null_columns and (set(drop_columns) & set(fill_null_columns)):
        raise SystemExit(
            f"Columns cannot be both dropped and filled: "
            f"{sorted(set(drop_columns) & set(fill_null_columns))}"
        )

    quota = ProtocolQuotaLedger(
        manifest_path=args.manifest,
        targets_by_protocol=args.expected_per_protocol,
        ledger_path=args.ledger,
        arms_by_protocol=args.expected_protocol_arms,
        arm_target_caps=args.protocol_quota_arm_caps,
    )
    selected_protocols = list(args.protocol) if args.protocol else list(quota.protocols)
    unknown_protocols = set(selected_protocols) - set(quota.protocols)
    if unknown_protocols:
        raise SystemExit(
            f"Unknown --protocol value(s): {sorted(unknown_protocols)}. "
            f"Known protocols: {quota.protocols}"
        )
    if len(set(selected_protocols)) != len(selected_protocols):
        raise SystemExit(f"Duplicate --protocol values: {selected_protocols}")
    rows = _load_credited_rows(args.ledger, credit_outcome_mode=args.credit_outcome_mode)
    if not rows:
        raise SystemExit(f"No credited rows found in {args.ledger}")
    model_id_by_arm = _arm_to_model_id(rows)
    ledger_sha256 = sha256_file(args.ledger)
    manifest_sha256 = sha256_file(args.manifest)

    targets_by_key: dict[tuple[str, str], tuple[str, str | None]] = {}
    for protocol, arm, repo, root in args.target:
        key = (protocol, arm)
        if key in targets_by_key:
            raise SystemExit(f"Duplicate --target for {protocol}.{arm}")
        targets_by_key[key] = (repo, root)

    expected_keys = {
        (protocol, arm)
        for protocol in selected_protocols
        for arm in quota.arms_by_protocol[protocol]
    }
    missing = expected_keys - set(targets_by_key)
    extra = set(targets_by_key) - expected_keys
    if missing or extra:
        raise SystemExit(
            f"Target protocol/arm keys do not match ledger manifest. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

    ds = (
        LeRobotDataset(
            repo_id=args.source_repo,
            root=args.source_root,
            download_videos=not args.drop_visual_features,
        )
        if args.source_root
        else refresh_lerobot_dataset_from_main(
            repo_id=args.source_repo,
            download_videos=not args.drop_visual_features,
        )
    )
    for task_name in _dataset_task_names(ds):
        validate_lerobot_task_name(task_name, context=f"{args.source_repo} LeRobot task metadata")
    real_spec = find_task_spec_by_task_name(quota.task)
    if real_spec is not None:
        validate_lerobot_task_name(real_spec.task_name, context=f"{args.manifest} task")
        if tuple(quota.keys) != real_spec.manifest_keys:
            raise SystemExit(
                f"{args.manifest} keys {list(quota.keys)} do not match the "
                f"{real_spec.name} registry manifest keys {list(real_spec.manifest_keys)}"
            )
        extractor_task = None
    else:
        if quota.task not in TASK_NAME_FOR_EXTRACTOR:
            real_task_names = {spec.task_name for spec in registered_task_specs()}
            valid_tasks = sorted({*TASK_NAME_FOR_EXTRACTOR, *real_task_names})
            raise SystemExit(
                f"{args.manifest} task must be one of {valid_tasks}; got {quota.task!r}"
            )
        extractor_task = TASK_NAME_FOR_EXTRACTOR[quota.task]

    credited_counts: Counter[tuple[str, str]] = Counter()
    saved_trajectory_stats_by_episode: dict[int, dict] = {}
    for row in rows:
        ep_idx = int(row["episode_index"])
        if ep_idx < 0 or ep_idx >= ds.num_episodes:
            raise SystemExit(
                f"Ledger episode_index {ep_idx} out of range for source "
                f"dataset with {ds.num_episodes} episodes"
            )
        saved_stats = _episode_saved_trajectory_stats(ds, ep_idx)
        saved_trajectory_stats_by_episode[ep_idx] = saved_stats
        # CORE-INTEGRITY INVARIANT: a counterfactual replay must NEVER credit a
        # no_cf protocol. The live ledger guarantees this (preview_credit only adds
        # no_cf when not is_counterfactual), but the split is the FINAL isolation
        # gate before actor training — enforce it here so a poisoned/buggy ledger
        # row cannot silently route a CF episode into a no_cf actor repo and
        # invalidate the Ours-vs-baseline comparison. (Hard-read: keys are written
        # unconditionally by ProtocolQuotaLedger.credit_episode.)
        if bool(row["is_counterfactual"]) and row["credited_protocol_arms"].get("no_cf"):
            raise SystemExit(
                f"Ledger row episode_index={ep_idx}: counterfactual episode credits "
                f"the no_cf protocol ({row['credited_protocol_arms']['no_cf']}). CF "
                "replays must never enter a no_cf training repo."
            )
        row_success = bool(row["success"])
        if last_frame_success(ds, ep_idx, frame_lookup=None) != row_success:
            raise SystemExit(
                f"Ledger row episode_index={ep_idx} success={row_success} but "
                "dataset last frame has a different success value"
            )
        if saved_stats["saved_success"] != row_success:
            raise SystemExit(
                f"Ledger row episode_index={ep_idx} success={row_success} but "
                "dataset contains a different episode success label"
            )
        operator_interventions = int(row["intervention_count"])
        if saved_stats["saved_intervention_count"] > operator_interventions:
            raise SystemExit(
                f"Ledger row episode_index={ep_idx} has fewer operator interventions "
                f"({operator_interventions}) than saved policy-to-human transitions "
                f"({saved_stats['saved_intervention_count']})"
            )

        if real_spec is not None:
            _validate_real_manifest_metadata(ds, ep_idx, row, quota, real_spec)
        else:
            first = int(ds.meta.episodes[ep_idx]["dataset_from_index"])
            env_state = np.asarray(ds.hf_dataset[first]["observation.environment_state"])
            vec = np.asarray(
                extract_sampler_state_from_env_state(env_state, task=extractor_task),
                dtype=np.float64,
            )
            matched_idx, dist = quota.match(vec)
            expected_idx = int(row["manifest_idx"])
            if matched_idx != expected_idx:
                raise SystemExit(
                    f"Ledger row episode_index={ep_idx} expected manifest_idx "
                    f"{expected_idx}, but dataset first state matches {matched_idx} "
                    f"(dist={dist:.2e})"
                )

        for protocol, arms in row["credited_protocol_arms"].items():
            for arm in arms:
                credited_counts[(protocol, arm)] += 1

    expected_count = {
        (protocol, arm): quota.target_for(protocol, arm)  # honours manifest protocol_arm_targets
        for protocol in selected_protocols
        for arm in quota.arms_by_protocol[protocol]
    }
    bad = {
        key: credited_counts.get(key, 0)
        for key, expected in expected_count.items()
        if credited_counts.get(key, 0) != expected
    }
    if bad:
        raise SystemExit(
            f"FAIL: credited counts do not match expected targets for "
            f"selected protocol(s) {selected_protocols}. "
            f"Bad counts: {bad}. All counts: {dict(credited_counts)}"
        )
    print(f"Credited counts verified: {dict(credited_counts)}")

    # Producer lineage (arm_key/arm_id/model_id) is written per credited row by
    # the real-world collector (mulligan.real.collect.blind_dagger extra=). The sim collector
    # (mulligan.sim.collect.dagger + quota) does not record it in the
    # ledger: policy routing lives in the manifest's policy_source instead.
    # Branch on the ledger schema once: hard-read when the ledger carries
    # lineage, refuse a mixed schema, and only then allow the sim None path.
    lineage_keys = ("arm_key", "arm_id", "model_id")
    rows_with_lineage = [r for r in rows if any(k in r for k in lineage_keys)]
    if rows_with_lineage and len(rows_with_lineage) != len(rows):
        raise SystemExit(
            f"FAIL: mixed ledger schema: {len(rows_with_lineage)}/{len(rows)} rows have "
            f"producer lineage fields {lineage_keys}"
        )
    has_producer_lineage = bool(rows_with_lineage)
    if has_producer_lineage:
        incomplete = [
            int(r["episode_index"]) for r in rows if any(k not in r for k in lineage_keys)
        ]
        if incomplete:
            raise SystemExit(
                f"FAIL: ledger rows missing some producer lineage fields: episodes {incomplete[:5]}"
            )
    else:
        print(
            "Ledger has no producer lineage fields (sim collector); sidecar producer_* "
            "fields will be None and lineage stays with the manifest policy_source."
        )

    def _sidecar_row(protocol: str, arm: str, row: dict, new_episode_index: int) -> dict:
        ep_idx = int(row["episode_index"])
        saved_stats = saved_trajectory_stats_by_episode[ep_idx]
        return {
            "episode_index": new_episode_index,
            "source_episode_index": ep_idx,
            "manifest_idx": (None if row.get("manifest_idx") is None else int(row["manifest_idx"])),
            "manifest_sources": row.get("manifest_sources", []),
            "protocol": protocol,
            "arm": arm,
            # Lineage is load-bearing for the real-world audit trail;
            # hard-read when the ledger schema carries it (validated above),
            # None for sim ledgers that route policies via the manifest instead.
            "producer_arm_key": row["arm_key"] if has_producer_lineage else None,
            "producer_arm_id": row["arm_id"] if has_producer_lineage else None,
            "producer_model_id": row["model_id"] if has_producer_lineage else None,
            "credited_protocol_arms": row["credited_protocol_arms"],
            "is_counterfactual": bool(row["is_counterfactual"]),
            "source_success": bool(row["success"]),
            "manifest_hash": row.get("manifest_hash"),
            "operator_intervention_count": int(row["intervention_count"]),
            "operator_policy_steps_per_segment": [int(x) for x in row["policy_steps_per_segment"]],
            **saved_stats,
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    targets: dict[tuple[str, str], LeRobotDataset] = {}
    written: Counter[tuple[str, str]] = Counter()
    split_sidecars: dict[tuple[str, str], list[dict]] = {}
    copied_source_rows: dict[tuple[str, str], list[dict]] = {}

    if args.copy_backend == "native":
        groups: dict[str, dict] = {}
        group_to_key: dict[str, tuple[str, str]] = {}
        rows_by_key: dict[tuple[str, str], list[dict]] = {}
        for key, (repo, root_override) in targets_by_key.items():
            protocol, arm = key
            root_path = (
                Path(root_override) if root_override else args.output_root / repo.split("/")[-1]
            )
            if args.overwrite and root_path.exists():
                print(f"  removing existing target root for overwrite: {root_path}")
                shutil.rmtree(root_path)
            rows_for_key = [
                row
                for row in rows
                if protocol in selected_protocols
                and arm in row["credited_protocol_arms"].get(protocol, [])
            ]
            expected = expected_count[key]
            if len(rows_for_key) != expected:
                raise SystemExit(
                    f"{protocol}.{arm}: assigned {len(rows_for_key)} source episodes, "
                    f"expected {expected}"
                )
            group_name = f"{protocol}.{arm}"
            groups[group_name] = {
                "repo_id": repo,
                "root": root_path,
                "episodes": [int(row["episode_index"]) for row in rows_for_key],
            }
            group_to_key[group_name] = key
            rows_by_key[key] = rows_for_key
        print(
            f"Native split backend: encoder={split_video_encoder()} "
            f"forbid_video_padding={args.forbid_video_padding}"
            + (f" fill_null={fill_null_columns}" if fill_null_columns else "")
            + (f" drop={drop_columns}" if drop_columns else "")
        )
        native_targets = fast_split_dataset(
            ds,
            groups,
            drop_columns=drop_columns,
            fill_null_columns=fill_null_columns,
            allow_video_padding=not args.forbid_video_padding,
        )
        for group_name, target_ds in native_targets.items():
            key = group_to_key[group_name]
            protocol, arm = key
            rows_for_key = rows_by_key[key]
            targets[key] = target_ds
            copied_source_rows[key] = rows_for_key
            split_sidecars[key] = [
                _sidecar_row(protocol, arm, row, idx) for idx, row in enumerate(rows_for_key)
            ]
            written[key] = int(target_ds.num_episodes)
            if written[key] != len(split_sidecars[key]):
                raise SystemExit(
                    f"{protocol}.{arm}: native split wrote {written[key]} episodes but "
                    f"sidecar has {len(split_sidecars[key])} rows"
                )
    else:
        for key, (repo, root_override) in targets_by_key.items():
            root_path = (
                Path(root_override) if root_override else args.output_root / repo.split("/")[-1]
            )
            if args.overwrite and root_path.exists():
                print(f"  removing existing target root for overwrite: {root_path}")
                shutil.rmtree(root_path)
            targets[key] = create_target_dataset(
                ds,
                repo,
                root_path,
                drop_visual_features=args.drop_visual_features,
            )
            print(f"  target {key[0]}.{key[1]}: repo={repo}, root={root_path}")

        split_sidecars = {key: [] for key in targets}
        copied_source_rows = {key: [] for key in targets}
        total_done = 0
        for row in rows:
            ep_idx = int(row["episode_index"])
            for protocol, arms in row["credited_protocol_arms"].items():
                if protocol not in selected_protocols:
                    continue
                for arm in arms:
                    key = (protocol, arm)
                    new_episode_index = int(written[key])
                    _copy_episode(
                        ds,
                        ep_idx,
                        targets[key],
                        drop_visual_features=args.drop_visual_features,
                    )
                    split_sidecars[key].append(_sidecar_row(protocol, arm, row, new_episode_index))
                    copied_source_rows[key].append(row)
                    written[key] += 1
                    total_done += 1
                    if total_done % 25 == 0:
                        print(f"  copied {total_done} protocol-arm episodes")

    for key, target_ds in targets.items():
        sidecar_path = target_ds.root / "meta" / "protocol_split_manifest.jsonl"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_sidecars[key])
        )
        print(f"  wrote sidecar {sidecar_path}")
        protocol, arm = key
        target_repo = targets_by_key[key][0]
        producer_model_ids = _unique_string_values(copied_source_rows[key], "model_id")
        target_model_id = model_id_by_arm.get(arm)
        mutually_exclusive = [
            repo
            for (other_protocol, other_arm), (repo, _root) in targets_by_key.items()
            if other_arm == arm and other_protocol != protocol
        ]
        lineage = make_split_lineage(
            repo_id=target_repo,
            task=quota.task,
            source_type="dagger",
            parent_repo_id=args.source_repo,
            derivation_script="mulligan.data.split_protocol_quota",
            sidecar_path="meta/protocol_split_manifest.jsonl",
            ledger_path=str(args.ledger),
            ledger_sha256=ledger_sha256,
            manifest_path=str(args.manifest),
            manifest_sha256=manifest_sha256,
            split_key="protocol+arm",
            view_family_id=f"{args.source_repo}::{arm}",
            view_id=protocol,
            mutually_exclusive_with=mutually_exclusive,
            extra={
                "protocol": protocol,
                "arm": arm,
                "target_arm_key": arm,
                "target_model_id": target_model_id,
                "producer_model_ids": producer_model_ids,
            },
        )
        lineage_path = write_local_lineage(target_ds.root, lineage)
        print(f"  wrote lineage {lineage_path}")
        print(f"\n{key[0]}.{key[1]} split: {target_ds.num_episodes} episodes at {target_ds.root}")
        if args.copy_backend == "frame":
            target_ds.finalize()

    parent_lineage = make_parent_lineage(
        repo_id=args.source_repo,
        task=quota.task,
        source_type="dagger",
        ledger_path=str(args.ledger),
        ledger_sha256=ledger_sha256,
        manifest_path=str(args.manifest),
        manifest_sha256=manifest_sha256,
        derived_repos=[repo for repo, _root in targets_by_key.values()],
        derivation_script="mulligan.data.split_protocol_quota",
    )
    parent_lineage_path = write_local_lineage(ds.root, parent_lineage)
    print(f"  wrote parent lineage {parent_lineage_path}")

    if args.push:
        print("\nPushing splits to HuggingFace Hub...")
        for key, target_ds in targets.items():
            push_lerobot_dataset_replacing_remote(
                target_ds, replace_remote_codec=args.replace_remote_codec
            )
            print(f"  pushed {key[0]}.{key[1]}: {targets_by_key[key][0]}")
        push_local_lineage(args.source_repo, ds.root)
        print(f"  pushed parent lineage: {args.source_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
