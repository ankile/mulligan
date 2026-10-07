#!/usr/bin/env python3
# ruff: noqa: E402
"""
Train IQL value functions (Q/V) on real-world vision data using LeRobot DataLoaders.

Loads images directly from LeRobot datasets via MultiLeRobotDataset + DataLoader
with delta_timestamps, encodes them on-the-fly through VisionIQL (unified encoder +
Q/V module), and trains IQL with a distributional value head (DIVL: categorical V
fit to the HL-Gauss projection of the target-Q minimum, TD for Q bootstrapped from
an entropy-adaptive quantile of V) with interleaved holdout evaluation. The DP encoder is frozen; only the Q/V heads train.

The parser defaults are the settings every released critic shares; the per-run
configs in ``configs/real/`` (run through ``mulligan.real.train.launch``) set the rest.

Checkpoints are written under ``<output-dir>/checkpoints/<step_N|final>/`` as
``iql_checkpoint.pt`` + ``metadata.json`` (the layout of the released critic repos);
``metadata.json`` records the resolved DP encoder source as ``dp_artifact``.

Usage:
    python -m mulligan.real.train.critic \
        --repo-ids mulligan/real-marker-d2-c00-teleop-sobol,mulligan/real-marker-d2-c01-dagger-mulligan \
        --eval-repo-ids mulligan/real-marker-d2-c00-teleop-validation \
        --dataset-revisions mulligan/real-marker-d2-c00-teleop-sobol=<sha> ... \
        --encoder-artifact hf://mulligan/real-marker-d2-r01-mulligan-dp \
        --camera-keys side_1,wrist_left --gamma 0.997 --output-dir outputs/marker_critic
"""

import hashlib
import json
import math
import os
import queue
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Suppress pydantic warnings from wandb's incorrect Field(frozen=True, repr=False) usage
# (only relevant with --use-wandb; a known wandb bug with no functional impact).
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
)
from lerobot.utils.feature_utils import dataset_to_policy_features
from torch.utils.data import DataLoader, RandomSampler, Subset

from mulligan.data.constants import DataSource
from mulligan.networks.distributional_v import DistributionalVNetwork
from mulligan.networks.q_network import QNetwork
from mulligan.networks.vision_encoder import load_frozen_encoder_from_dp
from mulligan.networks.vision_iql import VisionIQL
from mulligan.real.robot.cameras import STATION_CAMERA_KEYS_BY_ROLE
from mulligan.real.policy.relative_pose import POSE_DIM as RELATIVE_POSE_DIM
from mulligan.real.policy.relative_pose import command_pose_chunk_to_relative_torch
from mulligan.real.train.iql_args import parse_args
from mulligan.real.train.iql_flat_cache import (
    FLAT_CACHE_DYNAMIC_METADATA_KEYS,
    FLAT_ENCODED_CACHE_SCHEMA_VERSION,
    FlatEncodedTrajectoryCache,
    build_flat_cache_from_physical_frames,
    derive_flat_cache_rows,
    load_flat_encoded_cache,
    load_flat_encoded_cache_payload,
    save_flat_encoded_cache,
    validate_flat_encoded_cache_metadata,
)
from mulligan.utils.seeding import set_seed
from mulligan.real.train.iql_eval import (
    create_augmented_image_grid,
    encode_holdout_images,
    evaluate_on_holdout,
    loaded_frame_index,
    local_frame_index,
)
from mulligan.real.policy.image_preprocess import (
    GpuImagePreprocessor,
    PolicyImagePreprocessTransform,
    quantize_float01_to_uint8,
)
from mulligan.real.policy.side_crop import (
    build_crop_feature_map,
    install_per_camera_crop,
    normalize_crop_map,
    set_raw_uint8_mode_on_subdatasets,
)
from mulligan.data.fast_lerobot_reader import (
    FAST_READER_ENV,
    enable_fast_reader_on_subdatasets,
    fast_reader_enabled_by_env,
)
from mulligan.real.policy.vision_idql import _normalize_compiled_state_dict_keys
from mulligan.real.train.dataset_selectors import selectors_to_json
from mulligan.real.train.hub_data import (
    dataset_dir,
    dataset_revision,
    is_commit_sha,
    load_multi_dataset,
    parse_dataset_pins,
    prepare_datasets,
    resolve_dataset_commit,
    resolve_encoder_source,
    resolve_selected_episodes,
)
from mulligan.real.train.run_logging import RunLogger
from mulligan.training.resume import AutoResumeManager, should_save_resume_checkpoint
from mulligan.training.precision import configure_torch_precision
from mulligan.training.timer import TrainingTimer
from mulligan.real.eval.outcome_results import valid_prefix_length
from mulligan.data.transforms import (
    compute_multidataset_valid_boundaries,
    raw_metadata_column,
    remove_features_from_lerobot_subdatasets,
    require_multilerobot_subdatasets,
)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str | torch.device):
    """Move optimizer tensor state after loading a CPU resume checkpoint."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class _Uint8RefreshDataset(torch.utils.data.Dataset):
    """Quantize camera frames to the replay buffer's packed uint8 IN the worker.

    Applies the buffer's own quantization formula after the (unchanged) float
    crop-resize transform, so the bytes reaching the buffer are bit-identical
    to the float32 pipeline while worker→main IPC shrinks 4×. Module-level so
    spawn-context DataLoader workers can pickle it.
    """

    def __init__(self, base: torch.utils.data.Dataset, camera_keys: list[str]):
        self.base = base
        self.camera_keys = camera_keys

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        from mulligan.data.vision_replay_buffer import VisionReplayBuffer

        item = self.base[idx]
        for cam in self.camera_keys:
            item[cam] = VisionReplayBuffer.pack_images_uint8(item[cam])
        return item


class _RawUint8SubsetView(torch.utils.data.Dataset):
    """Worker-local raw-uint8 native view over ``Subset(MultiLeRobotDataset, idx)``.

    PICKLING-ISOLATION INVARIANT (critical correctness): DataLoader spawn/fork
    workers each receive an independent PICKLED copy of this view and, through it,
    of the MultiLeRobotDataset and every per-camera crop proxy. The first
    ``__getitem__`` in each worker flips raw-uint8 mode on THAT worker's OWN proxy
    copies only. The main-process dataset object and every OTHER loader — in
    particular the holdout encode loader, which must keep yielding fully-processed
    float32 224^2 images — are never mutated, because they hold different
    (pre-flip) pickled copies. NEVER wrap this around a dataset whose proxies are
    shared LIVE with an in-process consumer (e.g. num_workers=0), or the lazy flip
    would corrupt that consumer's image format; the trainer guards num_workers>0.
    """

    def __init__(self, base_subset):
        self.base = base_subset
        self._flipped = False

    def _ensure_flipped(self) -> None:
        if self._flipped:
            return
        # ``base`` is a torch Subset; ``base.dataset`` is the MultiLeRobotDataset
        # whose ``_datasets`` are the per-camera crop proxies to flip.
        set_raw_uint8_mode_on_subdatasets(self.base.dataset._datasets, True)
        self._flipped = True

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        self._ensure_flipped()
        return self.base[idx]


def replay_pixel_roundtrip(images: torch.Tensor) -> torch.Tensor:
    """Match the online replay buffer's float -> uint8 -> float image path."""
    return quantize_float01_to_uint8(images).float().div_(255.0)


IQL_AUGMENTATION_SPEC_VERSION = 1
IQL_BRIGHTNESS_SCALE = 0.1
IQL_CONTRAST_SCALE = 0.2
IQL_SATURATION_SCALE = 0.5
IQL_SHARPNESS_MAX = 2.0
IQL_ROTATION_DEGREES = 5.0
IQL_BLUR_KERNEL = ((1, 2, 1), (2, 4, 2), (1, 2, 1))
IQL_BLUR_KERNEL_DIVISOR = 16.0


def iql_augmentation_spec(*, shift_frac: float) -> dict[str, object]:
    """Return the complete, versioned stochastic image-transform contract."""
    return {
        "version": IQL_AUGMENTATION_SPEC_VERSION,
        "brightness_scale": IQL_BRIGHTNESS_SCALE,
        "contrast_scale": IQL_CONTRAST_SCALE,
        "saturation_scale": IQL_SATURATION_SCALE,
        "sharpness_range": [0.0, IQL_SHARPNESS_MAX],
        "rotation_degrees": IQL_ROTATION_DEGREES,
        "translation_fraction": shift_frac,
        "blur_kernel": [list(row) for row in IQL_BLUR_KERNEL],
        "blur_kernel_divisor": IQL_BLUR_KERNEL_DIVISOR,
        "affine_align_corners": False,
        "affine_padding_mode": "zeros",
    }


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and bytes from a loaded module state."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Module state entry {name!r} is not a tensor: {type(tensor)!r}")
        value = tensor.detach().contiguous().cpu()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def augment_iql_images(
    images: torch.Tensor,
    *,
    blur_kernel: torch.Tensor,
    shift_frac: float,
) -> torch.Tensor:
    """Apply the image augmentation used by online and cached Vision-IQL.

    Random parameters are drawn independently for every row passed to this
    function. Callers invoke it independently for each camera and timestamp,
    matching the established online replay path.
    """
    batch, channels, height, width = images.shape
    if channels != 3:
        raise ValueError(f"Expected RGB input, got shape {tuple(images.shape)}")
    device = images.device
    images = images * (
        1.0 + (torch.rand(batch, 3, 1, 1, device=device) * 2 - 1) * IQL_BRIGHTNESS_SCALE
    )
    contrast = 1.0 + (torch.rand(batch, 1, 1, 1, device=device) * 2 - 1) * IQL_CONTRAST_SCALE
    images = contrast * images + (1 - contrast) * images.mean(dim=(-2, -1), keepdim=True)
    saturation = 1.0 + (torch.rand(batch, 1, 1, 1, device=device) * 2 - 1) * IQL_SATURATION_SCALE
    images = saturation * images + (1 - saturation) * images.mean(dim=-3, keepdim=True)
    blurred = F.conv2d(
        F.pad(images.clamp(0, 1), (1, 1, 1, 1), mode="reflect"),
        blur_kernel.to(device=device, dtype=images.dtype),
        groups=3,
    )
    sharpness = torch.rand(batch, 1, 1, 1, device=device) * IQL_SHARPNESS_MAX
    images = ((1 - sharpness) * blurred + sharpness * images).clamp(0, 1)
    angle = (torch.rand(batch, device=device) * 2 - 1) * (IQL_ROTATION_DEGREES * torch.pi / 180)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    tx = (torch.rand(batch, device=device) * 2 - 1) * shift_frac
    ty = (torch.rand(batch, device=device) * 2 - 1) * shift_frac
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, [batch, channels, height, width], align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


def prepare_cached_iql_images(
    images: torch.Tensor,
    *,
    augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Reproduce replay-buffer pixels before frozen-encoder cache encoding."""
    images = replay_pixel_roundtrip(images)
    if augment_fn is not None:
        images = augment_fn(images).clone()
    return images


ENCODED_EVAL_CACHE_SCHEMA_VERSION = 2


def flat_cache_expected_metadata(metadata: dict) -> dict:
    """Return horizon/reward-independent provenance required by schema v3.

    ``seed`` is informational for a REUSED cache, not binding: the cached
    augmented views were drawn under the builder's seed, and a different
    training seed legitimately reuses them (MLP init + sampling still vary by
    the run seed). Frame identity is pinned by the frame-list sha and repo
    revisions. Note the caveat this accepts: a seed-N cached run is NOT the
    cached analog of a seed-N conventional run, whose augmentations would have
    been re-drawn under seed N.
    """
    result = {
        key: value
        for key, value in metadata.items()
        if key not in FLAT_CACHE_DYNAMIC_METADATA_KEYS and key != "seed"
    }
    result.update(
        {
            "cache_layout": "flat_frame_trajectory",
            "reward_semantics": "raw_dataset_reward",
            "action_semantics": "raw_dataset_action",
        }
    )
    return result


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


IQL_MAX_CONCURRENT_WORKER_POOLS = 2  # persistent holdout + refresh/cache loaders
REAL_IQL_CACHE_FPS = 15


class _TensorColumnTable(dict[str, torch.Tensor]):
    """Minimal HF-column surface backed entirely by cache tensors."""

    @property
    def column_names(self) -> list[str]:
        return list(self)


class _CachedMetadataSubDataset:
    def __init__(self, columns: dict[str, torch.Tensor]) -> None:
        self.hf_dataset = _TensorColumnTable(columns)
        self.features = set(columns)

    def __len__(self) -> int:
        return int(self.hf_dataset["done"].shape[0])

    def __getitem__(self, _index):
        raise RuntimeError(
            "Prebuilt-cache metadata adapters cannot decode samples; training must read "
            "states/actions/rewards exclusively from the loaded cache tensors"
        )


class _CachedMetadataMultiDataset:
    """Dataset metadata facade used only by the existing split/audit machinery."""

    def __init__(self, sub_datasets: list[_CachedMetadataSubDataset]) -> None:
        self._datasets = sub_datasets
        self.num_frames = sum(len(dataset) for dataset in sub_datasets)
        self.num_episodes = sum(
            int(torch.unique(dataset.hf_dataset["episode_index"]).numel())
            for dataset in sub_datasets
        )
        self.disabled_features: set[str] = set()

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, _index):
        raise RuntimeError(
            "Prebuilt-cache metadata adapters cannot decode samples; a raw-dataset path "
            "was reached unexpectedly"
        )


def build_prebuilt_train_metadata_dataset(
    cache: FlatEncodedTrajectoryCache,
    metadata: dict,
) -> _CachedMetadataMultiDataset:
    """Reconstruct episode-prefix columns from a flat cache without LeRobot I/O."""
    repo_ids = metadata["repo_ids"]
    if not isinstance(repo_ids, list) or not repo_ids:
        raise ValueError("flat cache metadata repo_ids must be a nonempty list")
    found_dataset_ids = set(int(value) for value in torch.unique(cache.dataset_index).tolist())
    expected_dataset_ids = set(range(len(repo_ids)))
    if found_dataset_ids != expected_dataset_ids:
        raise ValueError(
            "flat cache dataset_index coverage does not match repo_ids: "
            f"found={sorted(found_dataset_ids)}, expected={sorted(expected_dataset_ids)}"
        )

    sub_datasets = []
    for dataset_index in range(len(repo_ids)):
        rows = torch.where(cache.dataset_index == dataset_index)[0]
        global_episodes = cache.episode_index[rows]
        unique_episodes = torch.unique(global_episodes, sorted=True)
        local_episode_by_global = {
            int(episode): local_index
            for local_index, episode in enumerate(unique_episodes.tolist())
        }
        local_episodes = torch.tensor(
            [local_episode_by_global[int(episode)] for episode in global_episodes.tolist()],
            dtype=torch.long,
        )
        sub_datasets.append(
            _CachedMetadataSubDataset(
                {
                    "episode_index": local_episodes,
                    "frame_index": cache.frame_index[rows].cpu(),
                    "is_valid": cache.is_valid[rows].cpu(),
                    "source": cache.source[rows].cpu(),
                    "success": cache.success[rows].cpu(),
                    "reward": cache.reward[rows].cpu(),
                    "done": cache.done[rows].cpu(),
                    "intervention": cache.intervention[rows].cpu(),
                }
            )
        )
    return _CachedMetadataMultiDataset(sub_datasets)


def build_prebuilt_eval_metadata_dataset(
    holdout_data: dict[str, torch.Tensor],
    repo_ids: list[str],
) -> _CachedMetadataMultiDataset:
    """Build a metadata-only eval facade whose every cached anchor remains eligible."""
    dataset_indices = holdout_data["dataset_indices"].long().cpu()
    success = holdout_data["success"].long().cpu()
    source = holdout_data["source"].long().cpu()
    if dataset_indices.shape != success.shape or source.shape != success.shape:
        raise ValueError("encoded eval cache metadata tensors have inconsistent lengths")
    found_dataset_ids = set(int(value) for value in torch.unique(dataset_indices).tolist())
    expected_dataset_ids = set(range(len(repo_ids)))
    if found_dataset_ids != expected_dataset_ids:
        raise ValueError(
            "encoded eval dataset_index coverage does not match repo_ids: "
            f"found={sorted(found_dataset_ids)}, expected={sorted(expected_dataset_ids)}"
        )

    sub_datasets = []
    for dataset_index in range(len(repo_ids)):
        rows = torch.where(dataset_indices == dataset_index)[0]
        n = int(rows.numel())
        # Each row is represented as a terminal one-frame episode. The facade is
        # consumed only to preserve the existing all-heldout index contract; actual
        # states/actions/outcomes come directly from holdout_data.
        sub_datasets.append(
            _CachedMetadataSubDataset(
                {
                    "episode_index": torch.arange(n),
                    "frame_index": torch.zeros(n, dtype=torch.long),
                    "is_valid": torch.ones(n, dtype=torch.bool),
                    "source": source[rows],
                    "success": success[rows],
                    "reward": torch.zeros(n),
                    "done": torch.ones(n, dtype=torch.long),
                    "intervention": torch.zeros(n, dtype=torch.long),
                }
            )
        )
    return _CachedMetadataMultiDataset(sub_datasets)


def dataset_commit_revisions(repo_ids: list[str], revisions: dict[str, str]) -> dict[str, str]:
    """Commit sha each repo is read at (the pinned revision, else the default tag)."""
    return {
        repo_id: resolve_dataset_commit(repo_id, dataset_revision(revisions, repo_id))
        for repo_id in repo_ids
    }


def validate_cache_revisions(
    cache_revisions: dict[str, str],
    revisions: dict[str, str],
    *,
    label: str,
) -> None:
    """Refuse a prebuilt cache encoded from a different revision than ``--dataset-revisions``."""
    for repo_id, revision in sorted(revisions.items()):
        if repo_id not in cache_revisions:
            continue
        pinned = revision if is_commit_sha(revision) else resolve_dataset_commit(repo_id, revision)
        if cache_revisions[repo_id] != pinned:
            raise ValueError(
                f"{label} was built from {repo_id}@{cache_revisions[repo_id]}, but "
                f"--dataset-revisions pins {revision} ({pinned})"
            )


def selector_metadata(selectors: dict, repo_ids: list[str]) -> dict[str, dict]:
    """Episode selectors of ``repo_ids`` as recorded in cache and checkpoint metadata."""
    return json.loads(
        selectors_to_json({repo: sel for repo, sel in selectors.items() if repo in repo_ids})
    )


def resolve_io_resources(*, buffer_capacity_gb: float, num_workers: int) -> tuple[int, int]:
    """Split the peak worker budget across the overlapping loader pools.

    Returns ``(workers_per_loader, peak_workers)``.
    """
    if buffer_capacity_gb <= 0:
        raise ValueError(f"--buffer-capacity-gb must be > 0, got {buffer_capacity_gb}")
    if num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {num_workers}")
    if 0 < num_workers < IQL_MAX_CONCURRENT_WORKER_POOLS:
        raise ValueError(
            "--num-workers is a peak budget and must be 0 or at least "
            f"{IQL_MAX_CONCURRENT_WORKER_POOLS} for the overlapping persistent loader "
            f"pools; got num_workers={num_workers}"
        )
    workers_per_loader = num_workers // IQL_MAX_CONCURRENT_WORKER_POOLS
    return workers_per_loader, workers_per_loader * IQL_MAX_CONCURRENT_WORKER_POOLS


def save_encoded_eval_cache(
    path: Path,
    *,
    states: torch.Tensor,
    holdout_data: dict[str, torch.Tensor],
    metadata: dict,
) -> None:
    """Atomically persist clean holdout states and their scoring metadata."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = {
        "schema_version": ENCODED_EVAL_CACHE_SCHEMA_VERSION,
        "metadata": metadata,
        "states": states.detach().cpu(),
        "holdout_data": {
            key: value.detach().cpu() for key, value in holdout_data.items() if key != "states"
        },
    }
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    manifest_path = path.with_suffix(path.suffix + ".json")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    manifest = {
        "schema_version": ENCODED_EVAL_CACHE_SCHEMA_VERSION,
        "cache_path": str(path),
        "size_bytes": path.stat().st_size,
        "samples": int(states.shape[0]),
        "state_dim": int(states.shape[1]),
        "metadata": metadata,
    }
    try:
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(manifest_tmp, manifest_path)
    finally:
        if manifest_tmp.exists():
            manifest_tmp.unlink()


def validate_encoded_eval_cache_payload(payload: dict, expected_metadata: dict) -> dict:
    """Validate an already-loaded eval payload against exact experiment provenance.

    The training ``seed`` is informational, not binding: the clean eval cache is a
    deterministic encode of the holdout split, and the split identity is already
    pinned exactly by ``holdout_indices_sha256`` + ``repo_revisions_v3_0`` (a
    seed-2 training against a seed-1-built cache would mismatch ONLY on ``seed``,
    with every content-bearing key identical).
    """
    if payload.get("schema_version") != ENCODED_EVAL_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Encoded eval cache schema mismatch: "
            f"found={payload.get('schema_version')!r}, "
            f"expected={ENCODED_EVAL_CACHE_SCHEMA_VERSION}"
        )
    actual_metadata = payload.get("metadata")
    actual_cmp = dict(actual_metadata) if isinstance(actual_metadata, dict) else {}
    expected_cmp = dict(expected_metadata)
    cache_seed = actual_cmp.pop("seed", None)
    run_seed = expected_cmp.pop("seed", None)
    if cache_seed != run_seed:
        print(
            f"Encoded eval cache was built under seed={cache_seed}, this run uses "
            f"seed={run_seed} — allowed (cache content is seed-invariant; split "
            "identity is sha-pinned)."
        )
    if actual_cmp != expected_cmp:
        keys = sorted(set(actual_cmp) | set(expected_cmp))
        mismatches = {
            key: {"found": actual_cmp.get(key), "expected": expected_cmp.get(key)}
            for key in keys
            if actual_cmp.get(key) != expected_cmp.get(key)
        }
        raise ValueError(f"Encoded eval cache metadata mismatch: {mismatches}")
    states = payload.get("states")
    holdout_data = payload.get("holdout_data")
    if not isinstance(states, torch.Tensor) or not isinstance(holdout_data, dict):
        raise ValueError("Encoded eval cache is missing states or holdout metadata")
    required_keys = {
        "actions",
        "success",
        "source",
        "done",
        "is_valid",
        "frame_indices",
        "episode_indices",
        "dataset_indices",
        "original_frame_indices",
    }
    missing = sorted(required_keys - set(holdout_data))
    if missing:
        raise ValueError(f"Encoded eval cache is missing holdout keys: {missing}")
    n = int(expected_metadata["n_holdout"])
    state_dim = int(expected_metadata["state_dim"])
    action_flat_dim = int(expected_metadata["action_flat_dim"])
    if states.shape != (n, state_dim):
        raise ValueError(
            f"Encoded eval states shape mismatch: found={tuple(states.shape)}, "
            f"expected={(n, state_dim)}"
        )
    if holdout_data["actions"].shape != (n, action_flat_dim):
        raise ValueError(
            "Encoded eval actions shape mismatch: "
            f"found={tuple(holdout_data['actions'].shape)}, "
            f"expected={(n, action_flat_dim)}"
        )
    for key in required_keys:
        value = holdout_data[key]
        if not isinstance(value, torch.Tensor) or value.shape[0] != n:
            raise ValueError(
                f"Encoded eval holdout key {key!r} must be a tensor with leading dim {n}"
            )
    if not torch.isfinite(states).all() or not torch.isfinite(holdout_data["actions"]).all():
        raise ValueError("Encoded eval cache contains non-finite states or actions")
    return {"states": states, "holdout_data": holdout_data}


def load_encoded_eval_cache_payload(path: Path) -> tuple[dict, dict]:
    """Load an eval cache once so fast initialization can reuse its tensors."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Encoded eval cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    if payload.get("schema_version") != ENCODED_EVAL_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Encoded eval cache schema mismatch: "
            f"found={payload.get('schema_version')!r}, "
            f"expected={ENCODED_EVAL_CACHE_SCHEMA_VERSION}"
        )
    if not isinstance(metadata, dict):
        raise ValueError("Encoded eval cache is missing metadata")
    return payload, metadata


def load_encoded_eval_cache(path: Path, expected_metadata: dict) -> dict:
    """Load clean holdout states only when all experiment provenance matches."""
    payload, _metadata = load_encoded_eval_cache_payload(path)
    return validate_encoded_eval_cache_payload(payload, expected_metadata)


def validate_auto_resume_frequencies(
    *,
    auto_resume_enabled: bool,
    log_freq: int,
    resume_checkpoint_freq: int,
    use_wandb: bool = True,
) -> None:
    if not auto_resume_enabled:
        return
    if resume_checkpoint_freq <= 0:
        raise ValueError("Automatic resume requires --resume-checkpoint-freq > 0.")
    if use_wandb and log_freq < resume_checkpoint_freq:
        raise ValueError(
            "Automatic resume requires --log-freq >= --resume-checkpoint-freq so W&B "
            "does not contain training metrics newer than the last local resume state."
        )


def validate_target_clipping_config(
    *,
    intervention_negative_reward: float | None,
    clip_targets_min: float | None,
    clip_targets_max: float | None,
) -> None:
    """Reject clipping only when intervention shaping actually changes rewards."""
    if (
        intervention_negative_reward is not None
        and intervention_negative_reward != 0.0
        and (clip_targets_min is not None or clip_targets_max is not None)
    ):
        raise ValueError(
            "--intervention-negative-reward changes the reward range; disable "
            "--clip-targets-min/--clip-targets-max for this run."
        )


def resolve_independent_target_samples(
    *,
    target_view_sampling: str,
    target_view_samples: int,
    use_embedding_cache: bool,
    embedding_cache_input: Path | None,
    cache_augmented_views: int,
) -> int:
    """Validate target-view controls and return zero for the default path."""
    if target_view_sampling not in ("matched", "independent"):
        raise ValueError(f"unknown target-view sampling mode: {target_view_sampling!r}")
    if target_view_samples < 1:
        raise ValueError(f"--target-view-samples must be >= 1, got {target_view_samples}")
    if target_view_sampling == "matched":
        if target_view_samples != 1:
            raise ValueError("--target-view-samples must be 1 with matched target views")
        return 0
    if not use_embedding_cache:
        raise ValueError("independent target views require --precompute-embeddings")
    if embedding_cache_input is None:
        raise ValueError(
            "independent target views require a prebuilt schema-v3 flat cache "
            "(--embedding-cache-input)"
        )
    total_cache_views = 1 + cache_augmented_views
    if cache_augmented_views == 0:
        raise ValueError("independent target views require a multi-view cache")
    if target_view_samples > total_cache_views:
        raise ValueError(
            "--target-view-samples cannot exceed the configured cache view count: "
            f"samples={target_view_samples}, views={total_cache_views}"
        )
    return target_view_samples


def validate_iql_resume_topology(policy_state_dict: dict[str, torch.Tensor]) -> None:
    """Refuse a resume state that is not the target-Q / online-V topology."""
    keys = tuple(policy_state_dict)
    has_target_v = any(key.startswith("v_target.") for key in keys)
    has_target_q1 = any(key.startswith("q1_target.") for key in keys)
    has_target_q2 = any(key.startswith("q2_target.") for key in keys)
    if has_target_v or not (has_target_q1 and has_target_q2):
        raise RuntimeError(
            "Local auto-resume state has a target-V network instead of target Q1/Q2; "
            "this trainer (target-Q / online-V) cannot resume it. Launch with a new run "
            "name/output directory."
        )


def _get_actions(batch: dict[str, torch.Tensor], action_columns: list[str]) -> torch.Tensor:
    """Concatenate action columns from batch into a single tensor."""
    if len(action_columns) == 1:
        return batch[action_columns[0]]
    parts = []
    for col in action_columns:
        t = batch[col]
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        parts.append(t)
    return torch.cat(parts, dim=-1)


# --action-mode relative sources the absolute per-step COMMAND pose from these
# columns ([xyz, rpy] euler + absolute gripper = 7D per frame); the frame table /
# flat cache stores them raw and command_pose_chunk_to_relative_torch relativizes
# the gathered chunk against the anchor frame's proprio pose at batch time.
RELATIVE_ACTION_SOURCE_COLUMNS = ("action.cartesian_position", "action.gripper_position")


def _relativize_action_chunk(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Relativize a (B, H, 7) command-pose chunk against the ANCHOR step's proprio.

    ``state`` is the batch's ``observation.state``: either ``(B, D)`` (single step) or
    ``(B, T, D)`` windowed, where index 0 along T is the anchor/current step (the flat
    cache stacks [current, successor]; the v2 stream windows [t, t+k]). Returns the
    ``(B, H, 10)`` physical relative chunk. Fails loudly on any other layout.
    """
    if state.ndim == 3:
        anchor = state[:, 0, :]
    elif state.ndim == 2:
        anchor = state
    else:
        raise ValueError(
            f"observation.state must be (B, D) or (B, T, D) for the relative anchor; "
            f"got {tuple(state.shape)}"
        )
    return command_pose_chunk_to_relative_torch(action, anchor)


def _hf_column_to_int_list(column) -> list[int]:
    """Convert a vectorized HF Dataset column read to plain Python ints."""
    if isinstance(column, torch.Tensor):
        column = column.tolist()
    elif isinstance(column, np.ndarray):
        column = column.tolist()
    return [int(value.item() if hasattr(value, "item") else value) for value in column]


def resolve_camera_keys(
    *,
    all_camera_keys: list[str],
    camera_filter: str,
    camera_keys_arg: str | None,
) -> tuple[list[str], list[str]]:
    """Resolve explicit or suffix-filtered visual keys from dataset metadata."""
    if camera_keys_arg:
        requested = [item.strip() for item in camera_keys_arg.split(",") if item.strip()]
        if not requested:
            raise ValueError("--camera-keys was provided but no non-empty keys were parsed")
        normalized = [
            key if key.startswith("observation.images.") else f"observation.images.{key}"
            for key in requested
        ]
        missing = [key for key in normalized if key not in all_camera_keys]
        if missing:
            raise ValueError(
                f"Requested camera keys {missing} are not present. "
                f"Available cameras: {all_camera_keys}"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"--camera-keys contains duplicates: {normalized}")
        camera_keys = normalized
    else:
        camera_keys = [key for key in all_camera_keys if key.endswith(camera_filter)]
        if not camera_keys:
            raise ValueError(
                f"No cameras match filter '*{camera_filter}'. Available cameras: {all_camera_keys}"
            )

    excluded_camera_keys = [key for key in all_camera_keys if key not in camera_keys]
    return camera_keys, excluded_camera_keys


def resolve_iql_horizons(
    prediction_horizon: int,
    n_action_steps: int | None,
) -> tuple[int, int]:
    """Return (DP prediction horizon, IQL critic/execution horizon)."""
    if prediction_horizon < 1:
        raise ValueError(f"--chunk-size must be positive, got {prediction_horizon}")
    critic_horizon = prediction_horizon if n_action_steps is None else n_action_steps
    if critic_horizon < 1:
        raise ValueError(f"--n-action-steps must be positive, got {critic_horizon}")
    if critic_horizon > prediction_horizon:
        raise ValueError(
            f"--n-action-steps ({critic_horizon}) must be <= --chunk-size ({prediction_horizon})"
        )
    return prediction_horizon, critic_horizon


def resolve_td_horizon(action_horizon: int, td_horizon_steps: int | None) -> int:
    """Resolve a chunk-aligned TD horizon without changing the ranked action chunk."""
    td_horizon = action_horizon if td_horizon_steps is None else td_horizon_steps
    if td_horizon < action_horizon:
        raise ValueError(
            f"--td-horizon-steps ({td_horizon}) must be >= --n-action-steps ({action_horizon})"
        )
    if td_horizon % action_horizon != 0:
        raise ValueError(
            f"--td-horizon-steps ({td_horizon}) must be an integer multiple of "
            f"--n-action-steps ({action_horizon})"
        )
    return td_horizon


def td_lambda_at_step(
    step: int,
    *,
    initial_lambda: float,
    hold_steps: int,
    cosine_end_step: int,
) -> float:
    """Held then cosine-decayed TD(lambda) schedule."""
    if step < 0:
        raise ValueError(f"step must be nonnegative, got {step}")
    if not 0.0 <= initial_lambda <= 1.0:
        raise ValueError(f"initial_lambda must be in [0, 1], got {initial_lambda}")
    if hold_steps < 0 or cosine_end_step <= hold_steps:
        raise ValueError(
            "TD(lambda) schedule requires 0 <= hold_steps < cosine_end_step; "
            f"got {hold_steps=} and {cosine_end_step=}"
        )
    if step <= hold_steps:
        return initial_lambda
    if step >= cosine_end_step:
        return 0.0
    progress = (step - hold_steps) / (cosine_end_step - hold_steps)
    return initial_lambda * 0.5 * (1.0 + math.cos(math.pi * progress))


def critic_lr_at_step(
    step: int,
    *,
    base_lr: float,
    schedule: str,
    warmup_steps: int,
    total_steps: int,
    min_frac: float,
) -> float:
    """Per-step Q/V learning rate. 'constant' keeps the base learning rate."""
    if schedule == "constant":
        return base_lr
    if schedule != "warmup_cosine":
        raise ValueError(f"unknown critic LR schedule: {schedule!r}")
    if step < 0:
        raise ValueError(f"step must be nonnegative, got {step}")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(
            "warmup_cosine requires 0 <= warmup_steps < total_steps; "
            f"got {warmup_steps=} and {total_steps=}"
        )
    if not 0.0 <= min_frac <= 1.0:
        raise ValueError(f"--critic-lr-min-frac must be in [0, 1], got {min_frac}")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    floor = base_lr * min_frac
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def validate_td_lambda_config(
    *,
    enabled: bool,
    use_embedding_cache: bool,
    action_horizon: int,
    max_horizon: int,
    initial_lambda: float,
    hold_steps: int,
    cosine_end_step: int,
    training_steps: int,
) -> None:
    """Fail loudly on TD(lambda) configurations that cannot supply exact targets."""
    if not enabled:
        return
    if not use_embedding_cache:
        raise ValueError(
            "--td-lambda-curriculum requires --precompute-embeddings with a "
            "schema-v3 flat trajectory cache"
        )
    if max_horizon <= action_horizon:
        raise ValueError(
            "--td-lambda-curriculum requires --td-horizon-steps greater than "
            f"--n-action-steps ({max_horizon} <= {action_horizon})"
        )
    if max_horizon % action_horizon != 0:
        raise ValueError("TD(lambda) maximum horizon must be action-chunk aligned")
    if not 0.0 <= initial_lambda <= 1.0:
        raise ValueError(f"--td-lambda-initial must be in [0, 1], got {initial_lambda}")
    if hold_steps < 0:
        raise ValueError("--td-lambda-hold-steps must be nonnegative")
    if cosine_end_step <= hold_steps:
        raise ValueError("--td-lambda-cosine-end-step must exceed --td-lambda-hold-steps")
    if cosine_end_step > training_steps:
        raise ValueError(
            "--td-lambda-cosine-end-step cannot exceed --training-steps; "
            f"got {cosine_end_step} > {training_steps}"
        )


def intervention_values_for_subdataset(
    *,
    repo_id: str,
    column_names: list[str],
    source_values: list[int],
    hf_dataset,
    require_intervention: bool,
) -> list[int]:
    """Return frame-level intervention flags, synthesizing zeros only when valid."""
    if "intervention" in column_names:
        return _hf_column_to_int_list(hf_dataset["intervention"])
    if not require_intervention:
        return [0] * len(source_values)

    non_human_sources = sorted({int(value) for value in source_values if value != DataSource.HUMAN})
    is_eval_dataset = {"policy_id", "round_id"}.issubset(set(column_names)) and (
        "eval" in repo_id or "blind-eval" in repo_id
    )
    if non_human_sources and is_eval_dataset:
        warnings.warn(
            f"Dataset {repo_id!r} has autonomous eval rollouts but no 'intervention' "
            "column; treating all frames as zero-intervention for reward shaping.",
            stacklevel=2,
        )
        return [0] * len(source_values)
    if non_human_sources:
        raise ValueError(
            "Real-world Vision-IQL intervention reward shaping requires an "
            f"'intervention' column for non-teleop data. Dataset {repo_id!r} "
            f"is missing it and contains non-human source ids {non_human_sources}."
        )
    return [0] * len(source_values)


def apply_intervention_reward_shaping(
    batch: dict[str, torch.Tensor],
    *,
    intervention_by_frame: torch.Tensor,
    intervention_by_dataset: list[torch.Tensor] | None = None,
    intervention_negative_reward: float | None,
    reward_shift: float,
    horizon: int,
) -> None:
    """Modify batch['reward'] in-place using frame-indexed intervention metadata."""
    if intervention_negative_reward is None and reward_shift == 0.0:
        return
    if "reward" not in batch:
        raise KeyError("Cannot shape rewards because batch is missing required 'reward' key")

    rewards = batch["reward"]
    shaped = rewards

    if intervention_negative_reward is not None:
        if "index" not in batch:
            raise KeyError(
                "Cannot apply intervention_negative_reward because batch is missing "
                "required 'index' key"
            )
        frame_idx = batch["index"]
        if frame_idx.ndim > 1:
            frame_idx = frame_idx.reshape(frame_idx.shape[0], -1)[:, 0]
        frame_idx = frame_idx.long()
        offsets = torch.arange(horizon, device=frame_idx.device, dtype=torch.long)
        gather_idx = frame_idx[:, None] + offsets
        if intervention_by_dataset is not None and "dataset_index" in batch:
            dataset_idx = batch["dataset_index"]
            if dataset_idx.ndim > 1:
                dataset_idx = dataset_idx.reshape(dataset_idx.shape[0], -1)[:, 0]
            dataset_idx = dataset_idx.long().to(frame_idx.device)
            intervention_mask = torch.empty(
                gather_idx.shape,
                device=frame_idx.device,
                dtype=torch.long,
            )
            handled_rows = torch.zeros_like(dataset_idx, dtype=torch.bool)
            for ds_i, lookup_cpu in enumerate(intervention_by_dataset):
                rows = dataset_idx == ds_i
                if rows.any():
                    lookup = lookup_cpu.to(frame_idx.device)
                    # Bound metadata reads to the loaded table. Episode-terminal
                    # positions are canonicalized after shaping, so offsets past
                    # the first done are overwritten with the terminal step.
                    clamped_idx = gather_idx[rows].clamp(max=len(lookup) - 1)
                    intervention_mask[rows] = lookup[clamped_idx]
                    handled_rows |= rows
            if not handled_rows.all():
                missing = sorted({int(value) for value in dataset_idx[~handled_rows].tolist()})
                raise IndexError(f"batch contains unknown dataset_index value(s): {missing}")
        else:
            lookup = intervention_by_frame.to(frame_idx.device)
            intervention_mask = lookup[gather_idx.clamp(max=len(lookup) - 1)]
        intervention_mask = intervention_mask.to(device=rewards.device, dtype=rewards.dtype)
        while intervention_mask.ndim < rewards.ndim:
            intervention_mask = intervention_mask.unsqueeze(-1)
        shaped = shaped + intervention_negative_reward * intervention_mask

    if reward_shift != 0.0:
        shaped = shaped + reward_shift

    batch["reward"] = shaped


def make_episode_holdout_split(
    num_episodes: int,
    holdout_pct: float,
    *,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return train and holdout episode indices with non-empty sides."""
    if num_episodes < 2:
        raise ValueError(
            f"Real-world Vision-IQL training needs at least 2 episodes for "
            f"train/holdout split, got {num_episodes}."
        )
    if not 0.0 < holdout_pct < 1.0:
        raise ValueError(f"--holdout-pct must be in (0, 1), got {holdout_pct}")

    rng = np.random.RandomState(seed)
    episode_order = rng.permutation(num_episodes).tolist()
    n_holdout = max(1, int(num_episodes * holdout_pct))
    if n_holdout >= num_episodes:
        raise ValueError(
            f"--holdout-pct={holdout_pct} leaves no training episodes "
            f"for num_episodes={num_episodes}"
        )
    return episode_order[n_holdout:], episode_order[:n_holdout]


def parse_repo_id_list(raw: str | None) -> list[str]:
    """Parse comma-separated repo IDs and reject empty entries."""
    if raw is None:
        return []
    repo_ids = [repo.strip() for repo in raw.split(",")]
    empty_positions = [i for i, repo in enumerate(repo_ids) if not repo]
    if empty_positions:
        raise ValueError(f"Repo ID list contains empty entries at positions {empty_positions}")
    return repo_ids


def build_multidataset_frame_metadata(
    sub_datasets,
    repo_ids: list[str],
    *,
    log_label: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Build global frame-to-episode/dataset mappings for a MultiLeRobotDataset.

    Episode ``to`` indices are clamped to each episode's outcome-edited valid prefix
    (see :func:`mulligan.data.transforms.compute_multidataset_valid_boundaries`): the
    RL replay/holdout index lists built from these boundaries therefore never include
    the invalid suffix (soft-truncated retract/reset junk + terminal pad). Datasets
    without an ``is_valid`` column keep their raw ends. When ``log_label`` is
    given and any suffix frame was excluded, a loud one-line summary is printed.
    """
    cum_frames = [0]
    global_ep_offset = 0
    global_episode_idx_arr = []
    global_dataset_idx_arr = []

    for ds_i, sub_ds in enumerate(sub_datasets):
        n = len(sub_ds)
        cum_frames.append(cum_frames[-1] + n)
        ep_indices = raw_metadata_column(sub_ds.hf_dataset, "episode_index")
        if isinstance(ep_indices, torch.Tensor):
            ep_indices = ep_indices.tolist()
        if len(ep_indices) < n:
            raise RuntimeError(
                f"Dataset {repo_ids[ds_i]!r} has fewer episode_index rows than its "
                f"reported length ({len(ep_indices)=}, {n=})"
            )
        ep_indices = ep_indices[:n]
        # Global episode ids are positions in episode order, the same indexing the
        # from/to boundary lists use. For a whole repo (episode_index 0..N-1) this is
        # episode_index + offset; for a --dataset-episodes subset it stays dense.
        local_position: dict[int, int] = {}
        for e in ep_indices:
            if e not in local_position:
                local_position[e] = len(local_position)
            global_episode_idx_arr.append(local_position[e] + global_ep_offset)
        # A whole repo must carry episode_index 0..N-1 in row order (the LeRobot v3 layout),
        # so its positional ids equal episode_index.
        if getattr(sub_ds, "episodes", None) is None and any(
            pos != e for e, pos in local_position.items()
        ):
            bad = next(e for e, pos in local_position.items() if pos != e)
            raise ValueError(
                f"Dataset {repo_ids[ds_i]!r}: episode_index values are not 0..N-1 in row "
                f"order (episode_index {bad} is episode #{local_position[bad]}); re-index the "
                "dataset before training on it."
            )
        global_dataset_idx_arr.extend([ds_i] * n)
        global_ep_offset += len(local_position)

    global_episode_idx_t = torch.tensor(global_episode_idx_arr, dtype=torch.long)
    global_dataset_idx_t = torch.tensor(global_dataset_idx_arr, dtype=torch.long)

    cum_frames_t = torch.tensor(cum_frames[:-1], dtype=torch.long)
    original_frame_idx_t = (
        torch.arange(len(global_dataset_idx_t), dtype=torch.long)
        - cum_frames_t[global_dataset_idx_t]
    )

    from_indices, to_indices, _raw_to_indices, n_suffix_excluded = (
        compute_multidataset_valid_boundaries(sub_datasets, stop_at_first_done=False)
    )
    if log_label is not None and n_suffix_excluded > 0:
        print(
            f"[VALID-PREFIX] {log_label}: excluded {n_suffix_excluded} invalid-suffix frame(s) "
            f"(is_valid==0 only; valid terminal tails are retained) across "
            f"{len(from_indices)} episodes from replay/holdout sampling — expected for "
            "outcome-edited real data."
        )
    return (
        global_episode_idx_t,
        global_dataset_idx_t,
        original_frame_idx_t,
        from_indices,
        to_indices,
    )


def _hf_column_1d(hf_dataset, repo_id: str, column: str, *, dtype: torch.dtype) -> torch.Tensor:
    """Read a required HF dataset column as a flat 1-D tensor (loud on absence)."""
    if column not in hf_dataset.column_names:
        raise KeyError(
            f"Dataset {repo_id!r} has no {column!r} column; cannot derive discounted "
            "return-to-go values. Pass --v-min/--v-max explicitly if this only affects "
            "DIVL support derivation."
        )
    values = hf_dataset[column]
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype).reshape(-1)
    return torch.as_tensor(values, dtype=dtype).reshape(-1)


def discounted_mc_returns_by_dataset(
    sub_datasets,
    repo_ids: list[str],
    *,
    gamma: float,
    reward_shift: float,
    intervention_negative_reward: float | None,
    intervention_values_by_dataset: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Done-aware discounted Monte Carlo return-to-go for every loaded frame."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1] to derive MC returns, got {gamma}")

    returns_by_dataset: list[torch.Tensor] = []
    total_frames = 0
    for ds_i, sub_ds in enumerate(sub_datasets):
        repo_id = repo_ids[ds_i]
        hf = sub_ds.hf_dataset
        n = len(sub_ds)
        rewards = _hf_column_1d(hf, repo_id, "reward", dtype=torch.float32)
        dones = _hf_column_1d(hf, repo_id, "done", dtype=torch.bool)
        ep_indices = _hf_column_1d(hf, repo_id, "episode_index", dtype=torch.long)
        if len(rewards) < n or len(dones) < n or len(ep_indices) < n:
            raise RuntimeError(
                f"Dataset {repo_id!r} has fewer reward/done/episode_index rows than its "
                f"reported length ({len(rewards)=}, {len(dones)=}, "
                f"{len(ep_indices)=}, {n=})"
            )
        rewards, dones, ep_indices = rewards[:n], dones[:n], ep_indices[:n]
        shaped = rewards + float(reward_shift)
        if intervention_negative_reward is not None:
            if ds_i >= len(intervention_values_by_dataset):
                raise RuntimeError(
                    f"Missing intervention lookup for dataset {repo_id!r}; cannot derive "
                    "shaped MC return targets."
                )
            interv = intervention_values_by_dataset[ds_i][:n].to(dtype=torch.float32)
            if len(interv) != n:
                raise RuntimeError(
                    f"intervention lookup length mismatch for {repo_id!r} ({len(interv)=}, {n=})"
                )
            shaped = shaped + float(intervention_negative_reward) * interv

        # Restrict the per-episode backward RTG pass to the outcome-edited valid prefix.
        # The invalid suffix (soft-truncated retract/reset junk + terminal pad) is never
        # anchored, so its returns stay NaN; leaving it in the backward pass would leak
        # its shaped reward into valid-frame targets for TIMEOUT episodes (done==0
        # throughout means no reset). Datasets without is_valid use the whole episode.
        has_is_valid = "is_valid" in hf.column_names
        is_valid_np = (
            _hf_column_1d(hf, repo_id, "is_valid", dtype=torch.long).numpy()[:n]
            if has_is_valid
            else None
        )
        returns = torch.full((n,), float("nan"), dtype=torch.float32)
        shaped_np = shaped.numpy()
        dones_np = dones.numpy()
        ep_np = ep_indices.numpy()
        for ep in np.unique(ep_np):
            rows = np.flatnonzero(ep_np == ep)
            if is_valid_np is not None:
                prefix = valid_prefix_length(is_valid_np[rows], episode_index=int(ep))
                rows = rows[:prefix]
                terminal = np.flatnonzero(dones_np[rows])
                if len(terminal):
                    first = int(terminal[0])
                    if not dones_np[rows][first:].all():
                        raise ValueError(f"{repo_id!r} episode {int(ep)}: done returns to zero")
            running = 0.0
            for row in rows[::-1]:
                running = (
                    float(shaped_np[row])
                    if bool(dones_np[row])
                    else float(shaped_np[row]) + gamma * running
                )
                returns[row] = running
                total_frames += 1
        # NaN marks the intentionally-skipped invalid suffix (never anchored, never
        # looked up); only ±inf indicates a real computation blowup and must fail loud.
        inf_mask = torch.isinf(returns)
        if n > 0 and inf_mask.any():
            bad = torch.nonzero(inf_mask, as_tuple=False).flatten()[:10].tolist()
            raise ValueError(f"Infinite MC return targets in {repo_id!r}; examples={bad}")
        returns_by_dataset.append(returns)

    if total_frames == 0:
        raise ValueError("discounted_mc_returns_by_dataset: no frames found")
    return returns_by_dataset


def empirical_discounted_rtg_range(
    sub_datasets,
    repo_ids: list[str],
    *,
    gamma: float,
    reward_shift: float,
    intervention_negative_reward: float | None,
    intervention_values_by_dataset: list[torch.Tensor],
) -> tuple[float, float]:
    """Empirical done-aware discounted return-to-go (RTG) range over the training data.

    This is the quantity the distributional (DIVL) value head actually regresses
    toward, so the C51 atom support should bracket THIS (with margin). An
    undiscounted per-episode reward SUM would be mis-scaled and, under reward shaping /
    done-padded terminal success windows, could miss the true suffix-return tails (e.g.
    negative intervention-penalty states). For each episode the RTG matches VisionIQL TD,
    ``G_t = shaped_r_t + gamma * (1 - done_t) * G_{t+1}`` (gamma is per env-frame; cf.
    mulligan/networks/vision_iql.py), where ``shaped_r = reward + reward_shift +
    intervention_negative_reward * intervention`` (matching apply_intervention_reward_shaping).
    Returns ``(min_t G_t, max_t G_t)`` over all training frames.
    """
    returns_by_dataset = discounted_mc_returns_by_dataset(
        sub_datasets,
        repo_ids,
        gamma=gamma,
        reward_shift=reward_shift,
        intervention_negative_reward=intervention_negative_reward,
        intervention_values_by_dataset=intervention_values_by_dataset,
    )
    all_returns = torch.cat([returns for returns in returns_by_dataset if returns.numel() > 0])
    # Invalid-suffix rows carry NaN (skipped in the backward pass); exclude them so the
    # DIVL value support brackets only real anchored returns.
    all_returns = all_returns[torch.isfinite(all_returns)]
    if all_returns.numel() == 0:
        raise ValueError("empirical_discounted_rtg_range: no frames found to derive value support")
    return float(all_returns.min().item()), float(all_returns.max().item())


def resolve_divl_value_support(
    v_min: float | None,
    v_max: float | None,
    return_range: tuple[float, float],
) -> tuple[float, float]:
    """Resolve the DIVL value support, mirroring sim's _resolve_divl_value_support.

    If both v_min and v_max are given they are kept. Otherwise both are derived
    from the empirical return range with a symmetric 5%-of-span margin so the
    outermost atoms bracket observed returns. Partial overrides are rejected.
    """
    if (v_min is None) != (v_max is None):
        raise ValueError(
            "DIVL value support requires BOTH --v-min and --v-max or NEITHER; "
            f"got v_min={v_min}, v_max={v_max}"
        )
    if v_min is not None and v_max is not None:
        if not v_max > v_min:
            raise ValueError(f"--v-max ({v_max}) must be greater than --v-min ({v_min})")
        return float(v_min), float(v_max)

    lo, hi = float(return_range[0]), float(return_range[1])
    if not hi > lo:
        raise ValueError(
            f"empirical return range must have max > min to derive DIVL support, got "
            f"({lo}, {hi}); pass --v-min/--v-max explicitly"
        )
    margin = 0.05 * (hi - lo)
    return lo - margin, hi + margin


def iql_episode_frame_indices(
    from_indices: list[int],
    to_indices: list[int],
    done_values: list[int],
    *,
    episode_set: set[int],
    horizon: int,
) -> list[int]:
    """Return IQL anchors under the real outcome-editing transition contract.

    is_valid determines whether a row may be the current state. Terminal
    episodes retain every valid row, including the repeated done=1 tail:
    those rows are deliberate exact reward/value anchors and their bootstrap is
    masked immediately. Timeout episodes have no terminal row, so they retain
    only anchors with a complete horizon-step action/reward window; the last
    such anchor reads the first is_valid=0 row only as its successor state.
    """
    if horizon < 1:
        raise ValueError(f"IQL horizon must be positive, got {horizon}")
    if len(from_indices) != len(to_indices):
        raise ValueError(
            f"episode boundary length mismatch: {len(from_indices)=}, {len(to_indices)=}"
        )
    if to_indices and len(done_values) < max(to_indices):
        raise ValueError(
            f"done column has {len(done_values)} rows but boundaries reach {max(to_indices)}"
        )

    frame_indices: list[int] = []
    for ep_i, (fi, ti) in enumerate(zip(from_indices, to_indices, strict=True)):
        if ep_i not in episode_set:
            continue
        ep_done = np.asarray(done_values[fi:ti], dtype=bool)
        terminal = np.flatnonzero(ep_done)
        if len(terminal):
            first = int(terminal[0])
            if not ep_done[first:].all():
                raise ValueError(f"episode {ep_i}: done returns to zero after terminal frame")
            anchor_end = ti
        else:
            # range() is exclusive: +1 retains the final complete horizon-step
            # window [ti-horizon, ..., ti-1], whose successor is row ti.
            anchor_end = max(fi, ti - horizon + 1)
        frame_indices.extend(range(fi, anchor_end))
    return frame_indices


def canonicalize_post_terminal_steps(batch: dict[str, torch.Tensor]) -> None:
    """Repeat the first terminal step across later chunk positions in-place.

    Retaining a terminal anchor near the valid-prefix end can make later delta
    offsets land on invalid retract/reset rows. Those positions are beyond the
    first done and therefore absent from the TD return, but the critic still sees
    the full action chunk. Repeating the terminal action/reward/done removes that
    irrelevant junk dependence and matches ordinary end-of-episode padding.
    """
    for key in ("action", "reward", "done"):
        if key not in batch:
            raise KeyError(f"terminal-window canonicalization requires batch key {key!r}")
    actions = batch["action"]
    rewards = batch["reward"]
    dones = batch["done"]
    if actions.ndim != 3:
        raise ValueError(f"expected action shape (B, K, A), got {tuple(actions.shape)}")
    if rewards.shape[:2] != actions.shape[:2] or dones.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "action/reward/done horizon mismatch: "
            f"{tuple(actions.shape)=}, {tuple(rewards.shape)=}, {tuple(dones.shape)=}"
        )

    done_2d = dones.reshape(dones.shape[0], dones.shape[1], -1).bool().any(dim=-1)
    has_done = done_2d.any(dim=1)
    first_done = done_2d.to(torch.int64).argmax(dim=1)
    steps = torch.arange(actions.shape[1], device=actions.device)
    post_terminal = has_done[:, None] & (steps[None, :] > first_done[:, None])

    def _repeat_first_terminal(tensor: torch.Tensor) -> None:
        gather_shape = [tensor.shape[0], 1] + [1] * (tensor.ndim - 2)
        gather_idx = first_done.reshape(gather_shape).expand(tensor.shape[0], 1, *tensor.shape[2:])
        terminal_value = tensor.gather(1, gather_idx).expand_as(tensor)
        mask = post_terminal.reshape(tensor.shape[0], tensor.shape[1], *([1] * (tensor.ndim - 2)))
        tensor.copy_(torch.where(mask, terminal_value, tensor))

    _repeat_first_terminal(actions)
    _repeat_first_terminal(rewards)
    _repeat_first_terminal(dones)


def audit_iql_anchor_supervision(
    *,
    anchor_indices: list[int],
    global_episode_indices: torch.Tensor,
    global_dataset_indices: torch.Tensor,
    from_indices: list[int],
    to_indices: list[int],
    reward_values: list[float],
    done_values: list[int],
    success_values: list[int],
    repo_ids: list[str],
    horizon: int,
) -> list[dict[str, int | str]]:
    """Validate and summarize the reward/done supervision reachable by anchors."""
    n_frames = len(global_episode_indices)
    columns = {
        "dataset_index": len(global_dataset_indices),
        "reward": len(reward_values),
        "done": len(done_values),
        "success": len(success_values),
    }
    bad_lengths = {name: length for name, length in columns.items() if length != n_frames}
    if bad_lengths:
        raise ValueError(
            f"IQL supervision audit column length mismatch: {n_frames=}, {bad_lengths=}"
        )

    rows = [
        {
            "repo_id": repo_id,
            "anchors": 0,
            "reward_windows": 0,
            "done_windows": 0,
            "terminal_tail_anchors": 0,
            "timeout_anchors": 0,
            "timeout_subtask_reward_episodes": 0,
            "success_terminal_episodes": 0,
            "failure_terminal_episodes": 0,
        }
        for repo_id in repo_ids
    ]
    episode_has_done = [False] * len(from_indices)
    anchored_episodes = {int(global_episode_indices[idx]) for idx in anchor_indices}
    for ep_i, (fi, ti) in enumerate(zip(from_indices, to_indices, strict=True)):
        if ep_i not in anchored_episodes:
            continue
        if fi == ti:
            continue
        ds_i = int(global_dataset_indices[fi])
        if not 0 <= ds_i < len(rows):
            raise IndexError(f"episode {ep_i} has invalid dataset index {ds_i}")
        if not torch.all(global_dataset_indices[fi:ti] == ds_i):
            raise ValueError(f"episode {ep_i} crosses dataset boundaries")
        ep_done = np.asarray(done_values[fi:ti], dtype=bool)
        terminal = np.flatnonzero(ep_done)
        if len(terminal):
            episode_has_done[ep_i] = True
            first = int(terminal[0])
            if not ep_done[first:].all():
                raise ValueError(f"episode {ep_i}: done returns to zero after terminal frame")
            outcome = int(success_values[fi])
            if outcome not in (0, 1):
                raise ValueError(f"episode {ep_i}: success must be binary, got {outcome}")
            if any(int(value) != outcome for value in success_values[fi:ti]):
                raise ValueError(f"episode {ep_i}: success is not constant within the episode")
            expected_reward = float(outcome)
            actual_tail = np.asarray(reward_values[fi + first : ti], dtype=float)
            if not np.allclose(actual_tail, expected_reward, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"episode {ep_i}: terminal reward tail must equal success={outcome}; "
                    f"observed range=({actual_tail.min()}, {actual_tail.max()})"
                )
            key = "success_terminal_episodes" if outcome == 1 else "failure_terminal_episodes"
            rows[ds_i][key] += 1
        # A nonterminal timeout may still contain a semantically valid subtask
        # reward (for example routing_d2's first clip). Task-specific reward
        # audits validate those traces; the generic anchor audit must not
        # assume that every nonterminal episode is reward-free.

    for idx in anchor_indices:
        if not 0 <= idx < n_frames:
            raise IndexError(f"IQL anchor {idx} outside [0, {n_frames})")
        ep_i = int(global_episode_indices[idx])
        if not 0 <= ep_i < len(to_indices):
            raise IndexError(f"IQL anchor {idx} has invalid episode index {ep_i}")
        ti = to_indices[ep_i]
        ds_i = int(global_dataset_indices[idx])
        window_end = min(idx + horizon, ti)
        window_rewards = reward_values[idx:window_end]
        window_dones = done_values[idx:window_end]
        row = rows[ds_i]
        row["anchors"] += 1
        row["reward_windows"] += int(any(abs(float(value)) > 1e-8 for value in window_rewards))
        row["done_windows"] += int(any(bool(value) for value in window_dones))
        row["terminal_tail_anchors"] += int(bool(done_values[idx]))
        row["timeout_anchors"] += int(not episode_has_done[ep_i])

    for row in rows:
        if row["success_terminal_episodes"] and not row["reward_windows"]:
            raise RuntimeError(
                f"{row['repo_id']}: successful terminal episodes exist but no sampled "
                "IQL window contains reward supervision"
            )
        if row["failure_terminal_episodes"] and not row["done_windows"]:
            raise RuntimeError(
                f"{row['repo_id']}: failed terminal episodes exist but no sampled "
                "IQL window contains done supervision"
            )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def vision_idql_artifact_name(run_name: str, suffix: str) -> str:
    """Checkpoint artifact name for a run; W&B caps artifact names at 128 chars.

    Run names for this script conventionally already start with "vision-idql-";
    doubling the prefix would push long run names over the cap and fail at the
    first checkpoint upload.
    """
    base = run_name if run_name.startswith("vision-idql") else f"vision-idql-{run_name}"
    name = f"{base}-{suffix}".replace("/", "-").replace(":", "-")
    if len(name) > 128:
        raise ValueError(
            f"W&B artifact name is {len(name)} chars (cap 128): {name!r}. Shorten the run name."
        )
    return name


# --------------------------------------------------------------------------- #
# Reusable Vision-IQL training primitives
#
# One training step, the model/optimizer builds and the frame classification.
# --------------------------------------------------------------------------- #


@dataclass
class IQLFrameClassification:
    """Per-frame supervision columns gathered across a dataset mix.

    All list fields are indexed by GLOBAL frame index (sub-datasets concatenated
    in ``repo_ids`` order), which is the index space the replay buffer and the
    anchor-eligibility helpers use.
    """

    source_values: list[int]
    success_values: list[int]
    reward_values: list[float]
    done_values: list[int]
    intervention_values: list[int]
    intervention_values_by_dataset: list[torch.Tensor]
    intervention_by_frame: torch.Tensor


def iql_classify_frames(
    repo_ids: list[str],
    sub_datasets: list,
    *,
    n_global_frames: int,
    require_intervention: bool,
) -> IQLFrameClassification:
    """Pre-compute the per-frame source/success/reward/done/intervention columns.

    Fails loudly on a dataset missing ``source``/``success`` (without them the
    human-success vs other replay split and the holdout metrics are invalid) and
    on any column shorter than its sub-dataset.
    """
    source_values: list[int] = []
    success_values: list[int] = []
    reward_values: list[float] = []
    done_values: list[int] = []
    intervention_values: list[int] = []
    intervention_values_by_dataset: list[torch.Tensor] = []
    for repo_id, sub_ds in zip(repo_ids, sub_datasets, strict=True):
        missing_required_features = [
            name for name in ("source", "success") if name not in sub_ds.hf_dataset.column_names
        ]
        if missing_required_features:
            raise ValueError(
                "Real-world Vision-IQL training requires every LeRobot dataset to include "
                f"{missing_required_features} feature(s); {repo_id!r} is missing them. "
                "Without them, human-success vs other replay splits and holdout metrics "
                "would be invalid."
            )
        local_source_values = _hf_column_to_int_list(sub_ds.hf_dataset["source"])
        local_success_values = _hf_column_to_int_list(sub_ds.hf_dataset["success"])
        local_reward_values = _hf_column_1d(
            sub_ds.hf_dataset, repo_id, "reward", dtype=torch.float32
        ).tolist()
        local_done_values = _hf_column_1d(
            sub_ds.hf_dataset, repo_id, "done", dtype=torch.long
        ).tolist()
        local_intervention_values = intervention_values_for_subdataset(
            repo_id=repo_id,
            column_names=list(sub_ds.hf_dataset.column_names),
            source_values=local_source_values,
            hf_dataset=sub_ds.hf_dataset,
            require_intervention=require_intervention,
        )
        local_dataset_len = len(sub_ds)
        if (
            len(local_source_values) < local_dataset_len
            or len(local_success_values) < local_dataset_len
            or len(local_reward_values) < local_dataset_len
            or len(local_done_values) < local_dataset_len
        ):
            raise RuntimeError(
                f"Dataset {repo_id!r} has fewer required metadata rows than its reported "
                f"length ({len(local_source_values)=}, {len(local_success_values)=}, "
                f"{len(local_reward_values)=}, {len(local_done_values)=}, {local_dataset_len=})"
            )
        if len(local_intervention_values) != len(local_source_values):
            raise RuntimeError(
                f"intervention column length does not match source column length for "
                f"{repo_id!r} ({len(local_intervention_values)=}, "
                f"{len(local_source_values)=})"
            )
        local_source_values = local_source_values[:local_dataset_len]
        local_success_values = local_success_values[:local_dataset_len]
        local_reward_values = local_reward_values[:local_dataset_len]
        local_done_values = local_done_values[:local_dataset_len]
        local_intervention_values = local_intervention_values[:local_dataset_len]
        source_values.extend(local_source_values)
        success_values.extend(local_success_values)
        reward_values.extend(local_reward_values)
        done_values.extend(local_done_values)
        intervention_values.extend(local_intervention_values)
        intervention_values_by_dataset.append(
            torch.tensor(local_intervention_values, dtype=torch.long)
        )

    if (
        len(source_values) != n_global_frames
        or len(success_values) != n_global_frames
        or len(reward_values) != n_global_frames
        or len(done_values) != n_global_frames
        or len(intervention_values) != n_global_frames
    ):
        raise RuntimeError(
            "source/success/intervention column lengths do not match the global frame index "
            f"({len(source_values)=}, {len(success_values)=}, "
            f"{len(reward_values)=}, {len(done_values)=}, "
            f"{len(intervention_values)=}, frames={n_global_frames})"
        )
    return IQLFrameClassification(
        source_values=source_values,
        success_values=success_values,
        reward_values=reward_values,
        done_values=done_values,
        intervention_values=intervention_values,
        intervention_values_by_dataset=intervention_values_by_dataset,
        intervention_by_frame=torch.tensor(intervention_values, dtype=torch.long),
    )


def iql_partition_anchor_indices(
    candidate_indices,
    source_values: list[int],
    success_values: list[int],
    *,
    validate_disjoint: bool = False,
) -> tuple[list[int], list[int]]:
    """Split anchor-eligible frames into (human-success, other).

    ``validate_disjoint`` is off by default because the offline loop runs that
    check at a later point in ``main()``; keeping it opt-in preserves the
    original error ordering.
    """
    human_success_indices: list[int] = []
    other_indices: list[int] = []
    for idx in candidate_indices:
        is_human_success = source_values[idx] == DataSource.HUMAN and success_values[idx] == 1
        if is_human_success:
            human_success_indices.append(idx)
        else:
            other_indices.append(idx)
    if validate_disjoint:
        assert_disjoint_anchor_partitions(human_success_indices, other_indices)
    return human_success_indices, other_indices


def assert_disjoint_anchor_partitions(
    human_success_indices: list[int], other_indices: list[int]
) -> None:
    """Fail loudly if the two train partitions would double-sample a frame."""
    overlap = set(human_success_indices) & set(other_indices)
    if overlap:
        example_indices = sorted(overlap)[:10]
        raise RuntimeError(
            "Human-success and other training index partitions overlap; this would "
            f"double-sample {len(overlap)} frame(s). Example global indices: {example_indices}"
        )


# VisionIQL takes an expectile for its scalar-V loss; the distributional value head the
# trainer builds does not read it.
VISION_IQL_EXPECTILE = 0.7


def iql_build_model(
    *,
    encoder,
    q1,
    q2,
    v_net,
    camera_keys: list[str],
    separate_encoders: bool,
    args,
    image_norm_mean,
    image_norm_std,
    action_dim: int,
    state_dim: int,
    device,
) -> VisionIQL:
    """Construct the VisionIQL wrapper around already-built encoder/Q/V modules."""
    return VisionIQL(
        encoder=encoder,
        q1=q1,
        q2=q2,
        v_net=v_net,
        camera_keys=camera_keys,
        separate_encoders=separate_encoders,
        expectile=VISION_IQL_EXPECTILE,
        gamma=args.gamma,
        tau=args.tau,
        clip_targets_min=args.clip_targets_min,
        clip_targets_max=args.clip_targets_max,
        distributional=True,
        hl_gauss_sigma_ratio=args.hl_gauss_sigma_ratio,
        tau_base=args.tau_base,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_entropy_alpha=args.tau_entropy_alpha,
        image_norm_mean=image_norm_mean,
        image_norm_std=image_norm_std,
        encoder_autocast_bf16=args.encoder_autocast_bf16,
        channels_last=args.channels_last,
        # Action FiLM head: step_dim is the PER-STEP action dim (arm + gripper);
        # state_dim is the fused critic state the head and Q both consume.
        action_film_head=args.action_film_head,
        action_film_hidden=args.action_film_hidden,
        action_film_arm_dims=args.action_film_arm_dims,
        action_film_step_dim=action_dim,
        action_film_state_dim=state_dim,
    ).to(device)


def iql_build_optimizer(mod, args) -> torch.optim.Optimizer:
    """Build the single AdamW over the named ``q`` / ``v`` param groups of VisionIQL ``mod``."""
    optimizer_param_groups = []
    q_params = list(mod.q1.parameters()) + list(mod.q2.parameters())
    # The FiLM head trains through the Q loss, so it rides the q group.
    if mod.action_film_head:
        q_params += list(mod.film_head.parameters())
    optimizer_param_groups.append({"name": "q", "params": q_params, "lr": args.lr})
    optimizer_param_groups.append(
        {"name": "v", "params": list(mod.v_net.parameters()), "lr": args.lr}
    )
    return torch.optim.AdamW(
        optimizer_param_groups,
        weight_decay=args.weight_decay,
    )


@dataclass
class IQLStepTensors:
    """Device-resident inputs for one IQL update (output of the transfer block).

    The raw-image branch populates ``cam_imgs_raw``; the cached-embedding branch
    populates the ``*_visual`` / ``td_lambda_*`` fields instead.
    """

    proprio: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    valid_steps: torch.Tensor | None = None
    # Raw-image branch
    cam_imgs_raw: dict[str, torch.Tensor] = field(default_factory=dict)
    # Cached-embedding branch
    curr_visual: torch.Tensor | None = None
    next_visual: torch.Tensor | None = None
    curr_visual_target: torch.Tensor | None = None
    next_visual_target: torch.Tensor | None = None
    td_lambda_visual: torch.Tensor | None = None
    td_lambda_visual_target: torch.Tensor | None = None
    td_lambda_proprio: torch.Tensor | None = None
    td_lambda_horizons: torch.Tensor | None = None
    td_lambda_horizon_valid: torch.Tensor | None = None


@dataclass
class IQLStepContext:
    """Loop-invariant state consumed by one Vision-IQL training step."""

    args: object
    timer: TrainingTimer
    device: object
    camera_keys: list[str]
    use_embedding_cache: bool
    augment_fn: Callable | None
    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    discount_powers: torch.Tensor
    independent_target_samples: int
    log_augmented_batch: Callable | None = None


def iql_train_step(
    model,
    batch: IQLStepTensors,
    optimizer,
    ctx: IQLStepContext,
    *,
    step: int,
    td_lambda_active: bool,
    current_td_lambda: float | None = None,
    sampling_td_horizon: int | None = None,
) -> dict:
    """One Vision-IQL optimization step.

    ``td_lambda_active`` is REQUIRED (keyword-only, no default). It selects the
    critic's REGRESSION TARGET — plain n-step TD versus the TD(lambda) mixture —
    so a caller that forgets it silently trains a different objective than the
    one its config declares, and the loss dict looks identical either way.

    image convert -> augment -> proprio/action z-score -> forward -> backward ->
    clip -> optimizer.step -> target update, returning the loss dict.

    The caller still owns replay sampling, the host->device transfer, the replay
    buffer refresh, logging and checkpointing. Both critic-input regimes run
    through this one function: ``ctx.use_embedding_cache`` selects the cached
    ``forward_encoded_states`` path (which also owns the TD(lambda) /
    independent-target-sample machinery) versus the raw-image ``model(...)`` path,
    which also Polyak-averages ``encoder_target`` toward the encoder.
    """
    args = ctx.args
    timer = ctx.timer
    device = ctx.device
    camera_keys = ctx.camera_keys
    use_embedding_cache = ctx.use_embedding_cache
    proprio = batch.proprio

    with timer("img_convert"):
        curr_imgs: dict[str, torch.Tensor] = {}
        next_imgs: dict[str, torch.Tensor] = {}
        if not use_embedding_cache:
            for cam_key in camera_keys:
                img = batch.cam_imgs_raw[cam_key]
                curr_imgs[cam_key] = img[:, 0]
                next_imgs[cam_key] = img[:, 1]

    bsz = proprio.shape[0]

    # Augment images
    with timer("augment"):
        if ctx.augment_fn is not None and not use_embedding_cache:
            for cam_key in camera_keys:
                curr_imgs[cam_key] = ctx.augment_fn(curr_imgs[cam_key]).clone()
                next_imgs[cam_key] = ctx.augment_fn(next_imgs[cam_key]).clone()
    if not use_embedding_cache and ctx.log_augmented_batch is not None:
        ctx.log_augmented_batch(
            torch.stack([curr_imgs[cam_key] for cam_key in camera_keys], dim=1), step
        )

    # Z-score normalize proprio and actions
    proprio_norm = (proprio - ctx.proprio_mean) / ctx.proprio_std
    td_lambda_proprio_norm = None
    if td_lambda_active:
        td_lambda_proprio_norm = (batch.td_lambda_proprio - ctx.proprio_mean) / ctx.proprio_std
    if args.proprio_dropout > 0:
        if args.proprio_dropout >= 1.0:
            proprio_norm = torch.zeros_like(proprio_norm)
            if td_lambda_proprio_norm is not None:
                td_lambda_proprio_norm = torch.zeros_like(td_lambda_proprio_norm)
        else:
            mask = torch.bernoulli(
                torch.full((bsz, 1, 1), 1.0 - args.proprio_dropout, device=device)
            )
            proprio_norm = proprio_norm * mask
            if td_lambda_proprio_norm is not None:
                td_lambda_proprio_norm = td_lambda_proprio_norm * mask
    b_actions_norm = ((batch.actions - ctx.action_mean) / ctx.action_std).reshape(bsz, -1)

    # Forward + loss (VisionIQL handles everything)
    with timer("forward"):
        if use_embedding_cache:
            curr_state = torch.cat([batch.curr_visual, proprio_norm[:, 0]], dim=-1)
            next_state = torch.cat([batch.next_visual, proprio_norm[:, 1]], dim=-1)
            target_curr_state = None
            bootstrap_next_state = None
            if ctx.independent_target_samples:
                target_proprio_curr = proprio_norm[:, 0, None, :].expand(
                    -1, ctx.independent_target_samples, -1
                )
                target_proprio_next = proprio_norm[:, 1, None, :].expand(
                    -1, ctx.independent_target_samples, -1
                )
                target_curr_state = torch.cat(
                    [batch.curr_visual_target, target_proprio_curr], dim=-1
                )
                bootstrap_next_state = torch.cat(
                    [batch.next_visual_target, target_proprio_next], dim=-1
                )
            td_lambda_bootstrap_states = None
            if td_lambda_active:
                if ctx.independent_target_samples:
                    target_td_proprio = td_lambda_proprio_norm[:, :, None, :].expand(
                        -1, -1, ctx.independent_target_samples, -1
                    )
                    td_lambda_bootstrap_states = torch.cat(
                        [batch.td_lambda_visual_target, target_td_proprio], dim=-1
                    )
                else:
                    td_lambda_bootstrap_states = torch.cat(
                        [batch.td_lambda_visual, td_lambda_proprio_norm], dim=-1
                    )
            losses = model.forward_encoded_states(
                curr_state,
                next_state,
                b_actions_norm,
                batch.rewards,
                batch.dones,
                ctx.discount_powers[:sampling_td_horizon],
                target_curr_state=target_curr_state,
                bootstrap_next_state=bootstrap_next_state,
                valid_steps=batch.valid_steps,
                td_lambda_bootstrap_states=td_lambda_bootstrap_states,
                td_lambda_horizons=(batch.td_lambda_horizons if td_lambda_active else None),
                td_lambda_horizon_valid=(
                    batch.td_lambda_horizon_valid if td_lambda_active else None
                ),
                td_lambda=(current_td_lambda if td_lambda_active else None),
            )
        else:
            losses = model(
                curr_imgs,
                next_imgs,
                proprio_norm[:, 0],
                proprio_norm[:, 1],
                b_actions_norm,
                batch.rewards,
                batch.dones,
                ctx.discount_powers,
            )

    # Backward + optimize
    with timer("backward"):
        optimizer.zero_grad()
        losses["total"].backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

    # Target update
    with timer("target_update"):
        model.update_targets(update_encoder=not use_embedding_cache)

    return losses


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.dry_run:
        print("[DRY RUN] Running 100 steps with 1 eval cycle")
        args.training_steps = 100
        args.eval_freq = 50
        args.eval_video_freq = 50
        args.log_freq = 10
        # A 100-step dry run has no resumable state worth protecting; without
        # this, log_freq=10 < resume_checkpoint_freq trips
        # validate_auto_resume_frequencies and no dry run ever reaches step 1.
        args.no_auto_resume = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.enable_tf32:
        configure_torch_precision(True)
        print("  TF32 matmul + cudnn TF32 enabled (--enable-tf32)")
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("  cudnn.benchmark enabled (--cudnn-benchmark)")

    set_seed(args.seed)
    num_workers_per_loader, peak_num_workers = resolve_io_resources(
        buffer_capacity_gb=args.buffer_capacity_gb,
        num_workers=args.num_workers,
    )
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    repo_ids = parse_repo_id_list(args.repo_ids)
    eval_repo_ids = parse_repo_id_list(args.eval_repo_ids)
    use_external_eval = len(eval_repo_ids) > 0
    overlapping_eval_repos = sorted(set(repo_ids) & set(eval_repo_ids))
    if overlapping_eval_repos:
        raise ValueError(
            "--eval-repo-ids must be disjoint from --repo-ids to avoid train/eval leakage; "
            f"overlap={overlapping_eval_repos}"
        )
    dataset_revisions, dataset_selectors = parse_dataset_pins(
        args.dataset_revisions, args.dataset_episodes, [*repo_ids, *eval_repo_ids]
    )
    root = Path(args.root) if args.root else None
    prediction_horizon, critic_horizon = resolve_iql_horizons(
        args.chunk_size,
        args.n_action_steps,
    )
    k = critic_horizon
    td_k = resolve_td_horizon(k, args.td_horizon_steps)
    td_lambda_enabled = bool(args.td_lambda_curriculum)
    if args.num_action_samples < 1:
        raise ValueError(f"--num-action-samples must be >= 1, got {args.num_action_samples}")
    validate_target_clipping_config(
        intervention_negative_reward=args.intervention_negative_reward,
        clip_targets_min=args.clip_targets_min,
        clip_targets_max=args.clip_targets_max,
    )
    # "absolute" (default) scores the dataset ``action`` chunk directly. The UMI-relative
    # critic (proprio-anchored) scores physical relative-pose chunks instead: the frame
    # table / flat cache stores the ABSOLUTE per-frame command pose columns
    # (relativization is anchor-dependent, so it runs at batch-assembly time via
    # command_pose_chunk_to_relative_torch).
    action_mode = args.action_mode
    action_columns = (
        list(RELATIVE_ACTION_SOURCE_COLUMNS) if action_mode == "relative" else ["action"]
    )
    if action_mode == "relative":
        if not args.precompute_embeddings:
            raise NotImplementedError(
                "--action-mode relative is implemented for the flat-cache regime only "
                "(--precompute-embeddings); the streaming VisionReplayBuffer path does "
                "not carry the command-pose columns."
            )
    use_embedding_cache = bool(args.precompute_embeddings)
    cache_augmented_views = args.embedding_cache_augmented_views
    target_view_sampling = args.target_view_sampling
    target_view_samples = args.target_view_samples
    independent_target_samples = resolve_independent_target_samples(
        target_view_sampling=target_view_sampling,
        target_view_samples=target_view_samples,
        use_embedding_cache=use_embedding_cache,
        embedding_cache_input=args.embedding_cache_input,
        cache_augmented_views=cache_augmented_views,
    )
    validate_td_lambda_config(
        enabled=td_lambda_enabled,
        use_embedding_cache=use_embedding_cache,
        action_horizon=k,
        max_horizon=td_k,
        initial_lambda=args.td_lambda_initial,
        hold_steps=args.td_lambda_hold_steps,
        cosine_end_step=args.td_lambda_cosine_end_step,
        training_steps=args.training_steps,
    )
    if args.embedding_cache_only and not use_embedding_cache:
        raise ValueError("--embedding-cache-only requires --precompute-embeddings")
    if args.embedding_cache_only and args.embedding_cache_output is None:
        raise ValueError("--embedding-cache-only requires --embedding-cache-output")
    if args.embedding_cache_only and args.embedding_cache_input is not None:
        raise ValueError("--embedding-cache-only cannot be combined with --embedding-cache-input")
    if args.embedding_cache_input is not None and not use_embedding_cache:
        raise ValueError("--embedding-cache-input requires --precompute-embeddings")
    if args.embedding_cache_output is not None and not use_embedding_cache:
        raise ValueError("--embedding-cache-output requires --precompute-embeddings")
    if args.embedding_eval_cache_input is not None and not use_embedding_cache:
        raise ValueError("--embedding-eval-cache-input requires --precompute-embeddings")
    if args.embedding_eval_cache_input is not None and args.embedding_cache_input is None:
        raise ValueError("--embedding-eval-cache-input requires --embedding-cache-input")
    if args.embedding_eval_cache_output is not None and not use_embedding_cache:
        raise ValueError("--embedding-eval-cache-output requires --precompute-embeddings")
    if args.embedding_eval_cache_output is not None and args.embedding_cache_input is None:
        raise ValueError("--embedding-eval-cache-output requires --embedding-cache-input")
    if args.embedding_eval_cache_only:
        if not use_embedding_cache:
            raise ValueError("--embedding-eval-cache-only requires --precompute-embeddings")
        if args.embedding_cache_input is None:
            raise ValueError("--embedding-eval-cache-only requires --embedding-cache-input")
        if args.embedding_eval_cache_output is None:
            raise ValueError("--embedding-eval-cache-only requires --embedding-eval-cache-output")
        if args.embedding_eval_cache_input is not None:
            raise ValueError(
                "--embedding-eval-cache-only cannot be combined with --embedding-eval-cache-input"
            )
    if cache_augmented_views < 0:
        raise ValueError(
            f"--embedding-cache-augmented-views must be >= 0, got {cache_augmented_views}"
        )
    if cache_augmented_views > 0 and not use_embedding_cache:
        raise ValueError("--embedding-cache-augmented-views requires --precompute-embeddings")
    if use_embedding_cache:
        if args.augmentation == "gpu" and cache_augmented_views == 0:
            raise ValueError(
                "--precompute-embeddings with --augmentation gpu requires "
                "--embedding-cache-augmented-views > 0"
            )
        if args.augmentation == "none" and cache_augmented_views > 0:
            raise ValueError("--embedding-cache-augmented-views > 0 requires --augmentation gpu")
    prebuilt_cache_fast_init = bool(
        args.embedding_cache_input is not None
        and args.embedding_eval_cache_input is not None
        and not args.embedding_cache_only
        and args.embedding_cache_output is None
        and args.embedding_eval_cache_output is None
    )
    preloaded_train_cache: FlatEncodedTrajectoryCache | None = None
    preloaded_train_metadata: dict | None = None
    preloaded_eval_payload: dict | None = None
    preloaded_eval_metadata: dict | None = None
    if prebuilt_cache_fast_init:
        cache_load_t0 = time.perf_counter()
        print("[CACHE-FAST-INIT] Loading prebuilt train cache before dataset initialization")
        preloaded_train_cache, preloaded_train_metadata = load_flat_encoded_cache_payload(
            args.embedding_cache_input
        )
        if preloaded_train_metadata["repo_ids"] != repo_ids:
            raise ValueError(
                "prebuilt train cache repo_ids do not match --repo-ids: "
                f"cache={preloaded_train_metadata['repo_ids']}, cli={repo_ids}"
            )
        if preloaded_train_metadata["action_columns"] != action_columns:
            raise ValueError(
                "prebuilt train cache action_columns do not match this run's: "
                f"cache={preloaded_train_metadata['action_columns']}, cli={action_columns}"
            )
        if int(preloaded_train_metadata["augmented_views"]) != cache_augmented_views:
            raise ValueError(
                "prebuilt train cache augmented-view count mismatch: "
                f"cache={preloaded_train_metadata['augmented_views']}, "
                f"cli={cache_augmented_views}"
            )
        validate_cache_revisions(
            preloaded_train_metadata["repo_revisions_v3_0"],
            dataset_revisions,
            label="prebuilt train cache",
        )
        dataset = build_prebuilt_train_metadata_dataset(
            preloaded_train_cache, preloaded_train_metadata
        )
        if args.embedding_eval_cache_input is not None:
            preloaded_eval_payload, preloaded_eval_metadata = load_encoded_eval_cache_payload(
                args.embedding_eval_cache_input
            )
            if preloaded_eval_metadata["repo_ids"] != eval_repo_ids:
                raise ValueError(
                    "prebuilt eval cache repo_ids do not match --eval-repo-ids: "
                    f"cache={preloaded_eval_metadata['repo_ids']}, cli={eval_repo_ids}"
                )
            validate_cache_revisions(
                preloaded_eval_metadata["repo_revisions_v3_0"],
                dataset_revisions,
                label="prebuilt eval cache",
            )
            eval_dataset = build_prebuilt_eval_metadata_dataset(
                preloaded_eval_payload["holdout_data"], eval_repo_ids
            )
        else:
            eval_dataset = None
        print(
            "[CACHE-FAST-INIT] Train/eval cache tensors loaded in "
            f"{time.perf_counter() - cache_load_t0:.1f}s; raw LeRobot refresh and "
            "construction will be skipped"
        )

    print("=" * 60)
    print("Real-World Vision IQL Training (Direct LeRobot Loading)")
    print("=" * 60)
    print(f"Training datasets: {repo_ids}")
    if use_external_eval:
        print(f"Eval-only datasets: {eval_repo_ids}")
    else:
        print(f"Eval split: {args.holdout_pct:.1%} episode holdout from training datasets")
    print(f"Root: {root or 'default (HuggingFace cache)'}")
    print(f"Device: {device}")
    print(f"DP prediction horizon (--chunk-size): {prediction_horizon}")
    print(f"IQL critic/action horizon (--n-action-steps): {k}")
    print(f"TD reward/bootstrap horizon (--td-horizon-steps): {td_k} ({td_k // k} action chunk(s))")
    print(f"Exported num_action_samples: {args.num_action_samples}")
    print(f"Gamma: {args.gamma}")
    print(f"Tau: {args.tau}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(
        f"Grad clipping: {'max_norm=' + str(args.max_grad_norm) if args.max_grad_norm > 0 else 'disabled'}"
    )
    print(f"Training steps: {args.training_steps}")
    if args.checkpoint_freq > 0:
        print(f"Checkpointing: every {args.checkpoint_freq} steps + final")
    else:
        print("Checkpointing: final only")
    worker_budget = (
        f"{args.num_workers} requested peak "
        f"({num_workers_per_loader}/loader, {peak_num_workers} allocated)"
    )
    print(
        "Loader: "
        f"backend={args.video_backend}, "
        f"num_workers={worker_budget}, "
        f"prefetch_factor={args.prefetch_factor}, "
        f"persistent_workers={args.persistent_workers}, "
        f"multiprocessing_context={args.multiprocessing_context}"
    )
    print(
        f"Replay buffer: {args.buffer_capacity_gb:.1f} GB, refresh {args.buffer_refresh_rate}/step"
    )
    if args.clip_targets_min is not None or args.clip_targets_max is not None:
        print(f"Target clipping: [{args.clip_targets_min}, {args.clip_targets_max}]")
    if args.intervention_negative_reward is not None:
        print(f"Intervention reward penalty: {args.intervention_negative_reward:+.3f}")
    if args.reward_shift != 0.0:
        print(f"Reward shift: {args.reward_shift:+.3f}")
    if args.proprio_dropout > 0:
        print(f"Proprio dropout: {args.proprio_dropout}")
    print("Encoder: frozen")
    print("=" * 60)
    print()

    if not prebuilt_cache_fast_init:
        root = prepare_datasets(
            [*repo_ids, *eval_repo_ids], root, dataset_revisions, sync=not args.no_dataset_sync
        )
    # Episodes each repo contributes (None = all). The prebuilt-cache path takes its
    # frames from the cache, whose metadata records the selectors it was built with.
    selected_episodes: dict[str, list[int] | None] = {}
    if not prebuilt_cache_fast_init:
        selected_episodes = resolve_selected_episodes(
            [*repo_ids, *eval_repo_ids], root, dataset_revisions, dataset_selectors
        )

    # ---- 1. Load dataset metadata ----
    if prebuilt_cache_fast_init:
        if preloaded_train_metadata is None or preloaded_train_cache is None:
            raise RuntimeError("prebuilt cache fast initialization lost its loaded train payload")
        # Real-world datasets and robot evaluation use the station's fixed 15 Hz
        # contract; the flat cache does not record an fps.
        fps = REAL_IQL_CACHE_FPS
        cache_camera_keys = list(preloaded_train_metadata["camera_keys"])
        camera_keys, _cache_excluded = resolve_camera_keys(
            all_camera_keys=cache_camera_keys,
            camera_filter=args.camera_filter,
            camera_keys_arg=args.camera_keys,
        )
        if camera_keys != cache_camera_keys:
            raise ValueError(
                "prebuilt cache camera order does not match the resolved camera selection: "
                f"cache={cache_camera_keys}, resolved={camera_keys}"
            )
        all_camera_keys = camera_keys
        excluded_camera_keys = []
        proprio_dim = int(preloaded_train_cache.proprio.shape[1])
        action_dim = int(preloaded_train_cache.action.shape[1])
        if action_dim != int(preloaded_train_metadata["action_dim"]):
            raise ValueError(
                "prebuilt cache action tensor/metadata dimension mismatch: "
                f"tensor={action_dim}, metadata={preloaded_train_metadata['action_dim']}"
            )
        print("[CACHE-FAST-INIT] Dataset metadata sourced from cache payload")
    else:
        print("Loading dataset metadata...")
        # LeRobotDatasetMetadata expects the dataset root itself. MultiLeRobotDataset
        # expects the parent root and appends each repo_id internally.
        ds_meta = LeRobotDatasetMetadata(
            repo_ids[0],
            root=dataset_dir(root, repo_ids[0]),
            revision=dataset_revision(dataset_revisions, repo_ids[0]),
        )
        fps = ds_meta.fps
        print(f"  Primary dataset: {repo_ids[0]}")
        print(f"  FPS: {fps}")

        # ---- 2. Discover and filter cameras ----
        all_features = dataset_to_policy_features(ds_meta.features)
        all_camera_keys = [key for key, ft in all_features.items() if ft.type is FeatureType.VISUAL]
        camera_keys, excluded_camera_keys = resolve_camera_keys(
            all_camera_keys=all_camera_keys,
            camera_filter=args.camera_filter,
            camera_keys_arg=args.camera_keys,
        )
        proprio_dim = ds_meta.features["observation.state"]["shape"][0]
        action_dim = sum(ds_meta.features[col]["shape"][0] for col in action_columns)

    # The frame table / flat cache stores actions at the RAW column width; the critic
    # consumes the relativized 10D pose per step. raw_action_dim keys every
    # cache-storage contract, action_dim every critic-input contract (Q nets, stats,
    # shape checks, eval-cache flat width). Identical under action_mode=absolute.
    raw_action_dim = action_dim
    if action_mode == "relative":
        if raw_action_dim != 7:
            raise ValueError(
                "--action-mode relative expects the 7D [xyz, rpy, grip] command-pose "
                f"columns {list(RELATIVE_ACTION_SOURCE_COLUMNS)}; resolved raw width "
                f"{raw_action_dim}. Inspect the datamix features."
            )
        action_dim = RELATIVE_POSE_DIM

    if args.camera_keys:
        print(
            "  Camera selection: explicit --camera-keys "
            f"({args.camera_keys}); --camera-filter ignored"
        )

    n_cameras = len(camera_keys)
    print(f"  All cameras: {all_camera_keys}")
    print(f"  Selected cameras: {camera_keys}")
    print(f"  Excluded cameras: {excluded_camera_keys}")

    print(f"  Proprio dim: {proprio_dim}")
    print(f"  Action columns: {action_columns}")
    print(f"  Action dim: {action_dim}")
    print()

    # ---- 3. Load encoder metadata before dataset construction ----
    print("Loading pretrained encoder...")

    encoder_dir, encoder_source = resolve_encoder_source(args.encoder_artifact)
    encoder, encoder_meta = load_frozen_encoder_from_dp(encoder_dir, device=device)
    print(f"  DP encoder: {encoder_source}")
    encoder_state_sha256 = module_state_sha256(encoder) if use_embedding_cache else None

    feature_dim = encoder_meta["feature_dim"]
    # Mirrored DP image normalization — the frozen encoder was trained behind
    # this transform; encode paths must apply it to raw [0,1] frames. Loud
    # KeyError if the loader did not extract it.
    image_norm_mean = encoder_meta["image_norm_mean"]
    image_norm_std = encoder_meta["image_norm_std"]
    print(
        "  IQL image normalization (mirrored from DP preprocessor): "
        f"mean={image_norm_mean.flatten().tolist()}, "
        f"std={image_norm_std.flatten().tolist()}"
    )
    missing_encoder_meta = [
        key for key in ("separate_encoders", "camera_key_order") if key not in encoder_meta
    ]
    if missing_encoder_meta:
        raise KeyError(
            "Encoder artifact metadata is missing required key(s): "
            f"{missing_encoder_meta}. Regenerate the encoder artifact metadata."
        )
    separate_encoders = encoder_meta["separate_encoders"]
    encoder_camera_key_order = encoder_meta["camera_key_order"]

    action_target = encoder_meta.get("action_target", "cartesian_velocity")
    cartesian_action_frame = encoder_meta.get("cartesian_action_frame", "base")
    if action_target != "cartesian_velocity" or cartesian_action_frame != "base":
        raise NotImplementedError(
            "Vision-IQL training only supports DP encoders with "
            "action_target=cartesian_velocity and cartesian_action_frame=base; "
            f"got action_target={action_target!r}, "
            f"cartesian_action_frame={cartesian_action_frame!r}."
        )

    # Dual-stream ROI crops are a DP input this trainer does not support.
    if encoder_meta.get("dual_side_crop_boxes"):
        raise NotImplementedError(
            "DP encoders with dual_side_crop_boxes are not supported; got "
            f"{encoder_meta['dual_side_crop_boxes']}"
        )

    dp_camera_crops = normalize_crop_map(
        encoder_meta.get("camera_crop_boxes", {}),
        context="encoder artifact camera_crop_boxes",
    )
    selected_raw_camera_keys = [
        cam_key.removeprefix("observation.images.") for cam_key in camera_keys
    ]
    selected_raw_set = set(selected_raw_camera_keys)
    unknown_crop_keys = sorted(set(dp_camera_crops) - selected_raw_set)
    if unknown_crop_keys:
        raise ValueError(
            "Encoder artifact camera_crop_boxes contains camera(s) not selected for "
            f"IQL training: {unknown_crop_keys}; selected={selected_raw_camera_keys}."
        )
    role_named_cameras = any(key in STATION_CAMERA_KEYS_BY_ROLE for key in selected_raw_camera_keys)
    crop_reference_hw = (480, 640) if role_named_cameras else None

    print(f"  Feature dim: {feature_dim}")
    print(f"  Separate encoders: {separate_encoders}")
    if encoder_camera_key_order:
        print(f"  DP camera key order: {encoder_camera_key_order}")
    print(
        "  DP action contract: "
        f"action_target={action_target}, cartesian_action_frame={cartesian_action_frame}"
    )
    if dp_camera_crops:
        print(
            "  DP camera_crop_boxes: "
            f"{dict(dp_camera_crops)}"
            + (f" (crop_reference_hw={crop_reference_hw})" if crop_reference_hw is not None else "")
        )
    else:
        print("  DP camera_crop_boxes: none")

    if separate_encoders:
        encoder_keys = set(encoder.keys())
        missing = [cam_key for cam_key in selected_raw_camera_keys if cam_key not in encoder_keys]
        if missing:
            raise ValueError(
                f"Camera keys {missing} have no matching encoder. "
                f"Available encoder keys: {sorted(encoder_keys)}"
            )
        print(f"  Encoder keys: {sorted(encoder_keys)}")
        print(f"  Dataset camera keys: {camera_keys}")
    print()

    # ---- 4. Build delta_timestamps for IQL ----
    delta_timestamps = {}
    if not prebuilt_cache_fast_init:
        delta_timestamps = {
            "observation.state": [0, td_k / fps],
            "reward": [i / fps for i in range(td_k)],
            "done": [i / fps for i in range(td_k)],
        }
        for col in action_columns:
            delta_timestamps[col] = [i / fps for i in range(k)]
        for cam_key in camera_keys:
            delta_timestamps[cam_key] = [0, td_k / fps]

    if not prebuilt_cache_fast_init:
        print("Delta timestamps:")
        for key, ts in delta_timestamps.items():
            if len(ts) <= 3:
                print(f"  {key}: {ts}")
            else:
                print(f"  {key}: [{ts[0]}, ..., {ts[-1]}] ({len(ts)} steps)")
        print()

    # ---- 5. Create MultiLeRobotDataset ----
    resize_transform = PolicyImagePreprocessTransform((args.image_height, args.image_width))
    if not prebuilt_cache_fast_init:
        print("Loading dataset(s)...")
        train_episodes_by_repo = {repo_id: selected_episodes[repo_id] for repo_id in repo_ids}
        dataset = load_multi_dataset(
            repo_ids,
            root,
            dataset_revisions,
            episodes=(
                train_episodes_by_repo
                if any(v is not None for v in train_episodes_by_repo.values())
                else None
            ),
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
            image_transforms=resize_transform,
        )
        eval_dataset = None
        if use_external_eval:
            eval_episodes_by_repo = {
                repo_id: selected_episodes[repo_id] for repo_id in eval_repo_ids
            }
            eval_dataset = load_multi_dataset(
                eval_repo_ids,
                root,
                dataset_revisions,
                episodes=(
                    eval_episodes_by_repo
                    if any(v is not None for v in eval_episodes_by_repo.values())
                    else None
                ),
                delta_timestamps=delta_timestamps,
                video_backend=args.video_backend,
                image_transforms=resize_transform,
            )

    sub_datasets = require_multilerobot_subdatasets(dataset)
    eval_sub_datasets = (
        require_multilerobot_subdatasets(eval_dataset) if eval_dataset is not None else sub_datasets
    )

    # Column-cache fast reader: removes HF-datasets Python overhead (~14-18 ms/
    # sample, dominated by Features-deepcopy churn in delta-key queries) from
    # every __getitem__ — refresh workers, seed fill, holdout extraction.
    # Outputs are identical to the stock reader (tests/unit/test_fast_lerobot_reader.py).
    if prebuilt_cache_fast_init:
        print("  Fast reader: unnecessary (metadata columns are cache-backed tensors)")
    elif fast_reader_enabled_by_env():
        cached_bytes = enable_fast_reader_on_subdatasets(sub_datasets)
        if eval_dataset is not None:
            cached_bytes += enable_fast_reader_on_subdatasets(eval_sub_datasets)
        print(f"  Fast reader ON: {cached_bytes / 1e6:.1f} MB non-video columns cached in RAM")
    else:
        print(f"  Fast reader DISABLED via {FAST_READER_ENV}=0")

    print(f"  Loaded {dataset.num_frames} frames across {dataset.num_episodes} episodes")
    if eval_dataset is not None:
        print(
            f"  Loaded eval-only {eval_dataset.num_frames} frames across "
            f"{eval_dataset.num_episodes} episodes"
        )
    if dataset.disabled_features:
        print(f"  Disabled features (not common): {dataset.disabled_features}")
    if eval_dataset is not None and eval_dataset.disabled_features:
        print(f"  Disabled eval features (not common): {eval_dataset.disabled_features}")

    # Remove excluded camera features from sub-datasets to prevent video decoding
    if excluded_camera_keys and not prebuilt_cache_fast_init:
        remove_features_from_lerobot_subdatasets(sub_datasets, excluded_camera_keys)
        if eval_dataset is not None:
            remove_features_from_lerobot_subdatasets(eval_sub_datasets, excluded_camera_keys)
        print(f"  Removed {len(excluded_camera_keys)} excluded cameras from video pipeline")
    image_hw = (args.image_height, args.image_width)
    crop_feature_map = build_crop_feature_map(dp_camera_crops) if dp_camera_crops else {}
    if dp_camera_crops and not prebuilt_cache_fast_init:
        sub_datasets = install_per_camera_crop(
            dataset,
            crop_feature_map,
            resize_transform,
            crop_resize_hw=image_hw,
            crop_post_transform=None,
            crop_reference_hw=crop_reference_hw,
        )
        if eval_dataset is not None:
            eval_sub_datasets = install_per_camera_crop(
                eval_dataset,
                crop_feature_map,
                resize_transform,
                crop_resize_hw=image_hw,
                crop_post_transform=None,
                crop_reference_hw=crop_reference_hw,
            )
        else:
            eval_sub_datasets = sub_datasets
        print(f"  Applied DP camera_crop_boxes to IQL datasets (crop-then-resize to {image_hw})")
    print()

    # ---- 6b. uint8-native throughput path (crop+resize+float moved to GPU) ----
    # When on, DataLoader workers return RAW uint8 native frames and the crop +
    # antialias-resize + float machinery runs batched on GPU in the main process
    # via this single GpuImagePreprocessor (constructed from the SAME crop map /
    # target / reference used at dataset construction above). The buffer and the
    # end-to-end 224^2 pixel format are unchanged, so deploy/eval need no change.
    uint8_native = args.uint8_native_images and not use_embedding_cache
    gpu_preprocessor: GpuImagePreprocessor | None = None
    if uint8_native:
        # Preconditions. This flag is default-ON, so an unmet precondition
        # downgrades LOUDLY to the float path instead of crashing the run. Perf knob: the
        # experiment is
        # identical either way, so the downgrade passes the "is the experiment
        # still valid?" fallback test. Opt out deliberately with
        # --no-uint8-native-images to silence the WARN.
        precondition_problem: str | None = None
        if num_workers_per_loader <= 0:
            precondition_problem = (
                "--uint8-native-images requires at least one resolved loader worker: "
                "the raw-uint8 flip is "
                "isolated per DataLoader worker via pickling. With num_workers=0 the "
                "refresh loader would run in-process and mutate the shared crop proxies, "
                "corrupting the holdout/seed image format."
            )
        elif not args.persistent_workers:
            precondition_problem = (
                "--uint8-native-images requires --persistent-workers: the holdout encode "
                "loader must keep its PRE-flip pickled worker copies (float32 224^2 images) "
                "alive across the seed-loop raw-mode flip."
            )
        elif not dp_camera_crops:
            precondition_problem = (
                "--uint8-native-images requires DP camera_crop_boxes so the per-camera crop "
                "proxy is installed; the GPU preprocessor defers that proxy's crop+resize. "
                "Encoder artifact has no camera_crop_boxes."
            )
        if precondition_problem is not None:
            uint8_native = False
            args.uint8_native_images = False
            print(
                "WARNING: uint8-native images DOWNGRADED to the float path "
                f"(precondition unmet): {precondition_problem}"
            )
    if uint8_native:
        gpu_preprocessor = GpuImagePreprocessor(
            crop_feature_map,
            image_hw,
            crop_reference_hw=crop_reference_hw,
        )
        print(
            "  uint8-native images ENABLED: workers return raw uint8 native frames; "
            "crop+resize+float run on GPU (main process)"
        )
        print()

    # ---- 6. Build global metadata mappings ----
    (
        global_episode_idx_arr,
        global_dataset_idx_arr,
        original_frame_idx_arr,
        from_indices,
        to_indices,
    ) = build_multidataset_frame_metadata(sub_datasets, repo_ids, log_label="Train datasets")
    frame_index_by_dataset = loaded_frame_index(sub_datasets)
    if use_external_eval:
        (
            eval_episode_idx_arr,
            eval_dataset_idx_arr,
            eval_original_frame_idx_arr,
            eval_from_indices,
            eval_to_indices,
        ) = build_multidataset_frame_metadata(
            eval_sub_datasets, eval_repo_ids, log_label="Eval datasets"
        )
    else:
        eval_episode_idx_arr = global_episode_idx_arr
        eval_dataset_idx_arr = global_dataset_idx_arr
        eval_original_frame_idx_arr = original_frame_idx_arr
        eval_from_indices = from_indices
        eval_to_indices = to_indices

    # ---- 6. Compute episode boundaries ----
    num_episodes = len(from_indices)
    eval_num_episodes = len(eval_from_indices)

    print(
        f"Episode boundaries: {num_episodes} training episodes, IQL horizon={k} "
        "(terminal tails retained; timeouts require a complete horizon)"
    )
    if use_external_eval:
        print(f"Eval boundaries: {eval_num_episodes} eval-only episodes")

    # ---- 7. Episode-level train/holdout split ----
    if use_external_eval:
        train_episodes = list(range(num_episodes))
        holdout_episodes = list(range(eval_num_episodes))
    else:
        train_episodes, holdout_episodes = make_episode_holdout_split(
            num_episodes, args.holdout_pct, seed=args.seed
        )
    holdout_ep_set = set(holdout_episodes)
    train_ep_set = set(train_episodes)

    print(f"  Train: {len(train_ep_set)} episodes")
    if use_external_eval:
        print(f"  Eval-only holdout: {len(holdout_ep_set)} episodes")
    else:
        print(f"  Holdout: {len(holdout_ep_set)} episodes")

    for repo_id, sub_ds in zip(
        eval_repo_ids if use_external_eval else repo_ids,
        eval_sub_datasets,
        strict=True,
    ):
        missing_eval_features = [
            name for name in ("source", "success") if name not in sub_ds.hf_dataset.column_names
        ]
        if missing_eval_features:
            raise ValueError(
                "Real-world Vision-IQL evaluation requires every eval LeRobot dataset to "
                f"include {missing_eval_features} feature(s); {repo_id!r} is missing them."
            )

    # ---- 8. Pre-compute frame classifications for sampling ----
    holdout_indices = []

    frame_classification = iql_classify_frames(
        repo_ids,
        sub_datasets,
        n_global_frames=len(global_dataset_idx_arr),
        require_intervention=args.intervention_negative_reward is not None,
    )
    source_values = frame_classification.source_values
    success_values = frame_classification.success_values
    reward_values = frame_classification.reward_values
    done_values = frame_classification.done_values
    intervention_values_by_dataset = frame_classification.intervention_values_by_dataset
    intervention_by_frame = frame_classification.intervention_by_frame
    train_candidate_indices = iql_episode_frame_indices(
        from_indices,
        to_indices,
        done_values,
        episode_set=train_ep_set,
        horizon=k,
    )
    human_success_train_indices, other_train_indices = iql_partition_anchor_indices(
        train_candidate_indices, source_values, success_values
    )

    if not use_external_eval:
        holdout_indices = iql_episode_frame_indices(
            from_indices,
            to_indices,
            done_values,
            episode_set=holdout_ep_set,
            horizon=k,
        )

    assert_disjoint_anchor_partitions(human_success_train_indices, other_train_indices)

    if use_external_eval:
        eval_done_values: list[int] = []
        for repo_id, sub_ds in zip(eval_repo_ids, eval_sub_datasets, strict=True):
            local_done = _hf_column_1d(
                sub_ds.hf_dataset, repo_id, "done", dtype=torch.long
            ).tolist()
            eval_done_values.extend(local_done[: len(sub_ds)])
        holdout_indices = iql_episode_frame_indices(
            eval_from_indices,
            eval_to_indices,
            eval_done_values,
            episode_set=holdout_ep_set,
            horizon=k,
        )

    all_train_idx = human_success_train_indices + other_train_indices
    n_train = len(all_train_idx)
    n_holdout_frames = len(holdout_indices)
    if prebuilt_cache_fast_init:
        if preloaded_train_cache is None or preloaded_train_metadata is None:
            raise RuntimeError("prebuilt cache fast initialization lost train cache state")
        derived_anchor_rows = torch.tensor(sorted(all_train_idx), dtype=torch.long)
        if not torch.equal(derived_anchor_rows, preloaded_train_cache.anchor_rows.cpu()):
            raise ValueError(
                "cache-backed episode reconstruction does not reproduce anchor_eligible; "
                "refusing to train on a shifted transition set"
            )
        if n_train != int(preloaded_train_metadata["n_train"]):
            raise ValueError(
                "cache-backed anchor count does not match metadata n_train: "
                f"derived={n_train}, metadata={preloaded_train_metadata['n_train']}"
            )
    if n_train == 0:
        raise ValueError("Vision-IQL training split produced zero train frames")
    if n_holdout_frames == 0:
        raise ValueError("Vision-IQL evaluation split produced zero holdout frames")
    supervision_audit = audit_iql_anchor_supervision(
        anchor_indices=all_train_idx,
        global_episode_indices=global_episode_idx_arr,
        global_dataset_indices=global_dataset_idx_arr,
        from_indices=from_indices,
        to_indices=to_indices,
        reward_values=reward_values,
        done_values=done_values,
        success_values=success_values,
        repo_ids=repo_ids,
        horizon=k,
    )
    print(
        f"  Train frames: {n_train:,} "
        f"(human-success: {len(human_success_train_indices):,}, "
        f"other: {len(other_train_indices):,})"
    )
    print(
        "  Train index partitions: disjoint "
        f"({len(human_success_train_indices):,} + {len(other_train_indices):,})"
    )
    print(f"  Holdout frames: {n_holdout_frames:,}")
    print("  IQL supervision audit (sampled train anchors):")
    for row in supervision_audit:
        print(
            f"    {row['repo_id']}: anchors={row['anchors']:,}, "
            f"reward_windows={row['reward_windows']:,}, "
            f"done_windows={row['done_windows']:,}, "
            f"terminal_tail={row['terminal_tail_anchors']:,}, "
            f"timeouts={row['timeout_anchors']:,}, "
            f"timeout_subtask_rewards={row['timeout_subtask_reward_episodes']:,}"
        )
    if args.intervention_negative_reward is not None:
        print(f"  Intervention frames: {int(intervention_by_frame.sum().item()):,}")
    print()

    # ---- 8b. Set up replay buffer ----
    from mulligan.data.vision_replay_buffer import VisionReplayBuffer

    replay_buffer: VisionReplayBuffer | FlatEncodedTrajectoryCache | None = None
    if use_embedding_cache:
        print("Setting up encoded replay cache (full train set, no image replay buffer)...")
        print()
    else:
        print(
            f"Setting up replay buffer (budget={args.buffer_capacity_gb:.1f} GB, "
            f"refresh={args.buffer_refresh_rate}/step)..."
        )
        replay_buffer = VisionReplayBuffer.from_budget_gb(
            budget_gb=args.buffer_capacity_gb,
            camera_keys=camera_keys,
            n_image_timestamps=2,
            img_h=args.image_height,
            img_w=args.image_width,
            action_chunk_size=k,
            action_dim=action_dim,
            state_dim=proprio_dim,
            n_state_timestamps=2,
            reward_horizon_size=td_k,
        )
        print(f"  Buffer memory: {replay_buffer.memory_gb:.2f} GB")
        print()

    # ---- 9. Sampling strategy ----
    use_workers = num_workers_per_loader > 0
    print(f"Sampler: uniform replay buffer refresh ({n_train} frames)")
    print()

    # ---- 10. Augmentation ----
    # NOTE: torch.compile is applied AFTER VisionIQL creation so targets remain uncompiled.
    aug_mode = args.augmentation
    augment_fn = None
    if aug_mode == "gpu":
        blur_kernel = (
            torch.tensor(IQL_BLUR_KERNEL, device=device, dtype=torch.float32)
            .div_(IQL_BLUR_KERNEL_DIVISOR)
            .view(1, 1, 3, 3)
            .expand(3, 1, 3, 3)
        )

        shift_frac = args.aug_shift_frac

        def _gpu_augment(img: torch.Tensor) -> torch.Tensor:
            return augment_iql_images(
                img,
                blur_kernel=blur_kernel,
                shift_frac=shift_frac,
            )

        augment_fn = torch.compile(_gpu_augment, mode="reduce-overhead")
        print(
            "  Augmentation: gpu "
            "(compiled per-chan bright + contrast + sat + sharpness + affine, "
            f"shift_frac={shift_frac})"
        )
    else:
        print("  Augmentation: none")

    visual_feature_dim = n_cameras * feature_dim
    if prebuilt_cache_fast_init:
        if preloaded_train_cache is None:
            raise RuntimeError("prebuilt cache fast initialization lost train cache state")
        cached_visual_dim = int(preloaded_train_cache.visual.shape[-1])
        if cached_visual_dim != visual_feature_dim:
            raise ValueError(
                "prebuilt cache visual width does not match encoder/camera metadata: "
                f"cache={cached_visual_dim}, expected={visual_feature_dim}"
            )
    state_dim = visual_feature_dim + proprio_dim
    chunked_action_dim = k * action_dim

    print(f"  Visual feature dim: {visual_feature_dim}")
    print(f"  State dim: {state_dim}")
    print(f"  Chunked action dim: {chunked_action_dim}")
    print()

    discount_powers_gpu = torch.tensor(
        [args.gamma**i for i in range(td_k)], dtype=torch.float32, device=device
    )

    # ---- 11. Streaming data preparation ----
    N = dataset.num_frames
    holdout_idx = torch.tensor(holdout_indices, dtype=torch.long)
    holdout_dataset = eval_dataset if eval_dataset is not None else dataset
    holdout_repo_ids = eval_repo_ids if use_external_eval else repo_ids
    encoded_eval_cache_metadata = {}
    if args.embedding_eval_cache_input is not None or args.embedding_eval_cache_output is not None:
        if prebuilt_cache_fast_init:
            if preloaded_eval_metadata is None:
                raise RuntimeError("prebuilt cache fast initialization lost eval metadata")
            holdout_repo_revisions = preloaded_eval_metadata["repo_revisions_v3_0"]
        else:
            holdout_repo_revisions = dataset_commit_revisions(holdout_repo_ids, dataset_revisions)
        holdout_indices_sha256 = (
            preloaded_eval_metadata["holdout_indices_sha256"]
            if prebuilt_cache_fast_init and preloaded_eval_metadata is not None
            else hashlib.sha256(np.asarray(holdout_indices, dtype=np.int64).tobytes()).hexdigest()
        )
        train_indices_sha256_for_eval = (
            preloaded_train_metadata["train_indices_sha256"]
            if prebuilt_cache_fast_init and preloaded_train_metadata is not None
            else hashlib.sha256(np.asarray(all_train_idx, dtype=np.int64).tobytes()).hexdigest()
        )
        encoded_eval_cache_metadata = {
            "repo_ids": holdout_repo_ids,
            "repo_revisions_v3_0": holdout_repo_revisions,
            "holdout_indices_sha256": holdout_indices_sha256,
            "n_holdout": len(holdout_indices),
            "state_dim": state_dim,
            "action_flat_dim": k * action_dim,
            "train_indices_sha256": train_indices_sha256_for_eval,
            "train_embedding_cache_sha256": sha256_file(args.embedding_cache_input),
            "encoder_artifact": args.encoder_artifact,
            "encoder_state_sha256": encoder_state_sha256,
            "camera_keys": camera_keys,
            "camera_crop_boxes": {key: list(value) for key, value in dp_camera_crops.items()},
            "crop_reference_hw": list(crop_reference_hw) if crop_reference_hw is not None else None,
            "image_hw": list(image_hw),
            "image_norm_mean": image_norm_mean.tolist(),
            "image_norm_std": image_norm_std.tolist(),
            "encoder_autocast_bf16": args.encoder_autocast_bf16,
            "channels_last": args.channels_last,
            # The action FiLM head is applied at Q-scoring time and does not
            # change the cached embeddings or actions, so it is not cache metadata.
            "action_columns": action_columns,
            "action_horizon": k,
            "action_dim": action_dim,
            "proprio_dropout": args.proprio_dropout,
            "seed": args.seed,
        }
        holdout_selectors = selector_metadata(dataset_selectors, holdout_repo_ids)
        if holdout_selectors:
            # Binding only when a selection is active, so caches of whole repos keep
            # their metadata unchanged.
            encoded_eval_cache_metadata["dataset_episodes"] = holdout_selectors
        if action_mode == "relative":
            # The eval cache stores the TRANSFORMED (relativized, z-scored) flat
            # actions, so the mode is a binding provenance key. Recorded only when
            # non-default: the eval-cache validator compares the full metadata dict
            # symmetrically, so absolute caches (no key) keep matching
            # absolute runs, and any relative/absolute crossing mismatches loudly.
            encoded_eval_cache_metadata["action_mode"] = action_mode

    encode_dl_kwargs = dict(
        batch_size=args.encoding_batch_size,
        shuffle=False,
        num_workers=num_workers_per_loader,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if use_workers else None,
        persistent_workers=args.persistent_workers if use_workers else False,
    )
    if use_workers and args.multiprocessing_context != "none":
        encode_dl_kwargs["multiprocessing_context"] = args.multiprocessing_context

    if uint8_native and not encode_dl_kwargs["persistent_workers"]:
        # Invariant: the holdout encode loader's workers are spawned NOW (pre-flip)
        # and must survive the later seed-loop raw-mode flip so they keep yielding
        # fully-processed float32 224^2 images. Without persistent workers they
        # would re-spawn and could pickle a mid-flip proxy state.
        raise ValueError(
            "--uint8-native-images requires the holdout DataLoader to use persistent "
            f"workers, but persistent_workers={encode_dl_kwargs['persistent_workers']} "
            f"(num_workers_per_loader={num_workers_per_loader}, "
            f"--persistent-workers={args.persistent_workers})."
        )

    # Cached training with holdout eval fully disabled never consumes holdout
    # data either: evaluate_on_holdout is gated on eval_freq > 0 and there is no
    # eval cache to build. Skipping the stream (over an hour on a large holdout)
    # lets a cached training run start its loop immediately; holdout diagnostics
    # can run afterwards from an eval cache built separately.
    skip_holdout_no_eval = (
        use_embedding_cache
        and not args.embedding_cache_only
        and args.eval_freq == 0
        and args.embedding_eval_cache_input is None
        and args.embedding_eval_cache_output is None
    )
    encoded_eval_payload = None
    if args.embedding_cache_only or skip_holdout_no_eval:
        # A train embedding-cache-only build returns before it ever reads holdout
        # metadata (holdout is only consumed by Q/V eval and the eval-cache path,
        # both of which run after the early return below), so the holdout stream is
        # skipped for this mode. The eval-cache build
        # (--embedding-eval-cache-only) is a distinct mode — mutually exclusive with
        # --embedding-cache-only — and still streams its holdout below.
        holdout_dl = None
        holdout_data = None
        reason = (
            "train embedding-cache-only build"
            if args.embedding_cache_only
            else "cached training with --eval-freq 0"
        )
        print(f"Skipping holdout metadata stream ({reason}).")
    elif args.embedding_eval_cache_input is not None:
        print(f"Loading encoded eval cache: {args.embedding_eval_cache_input}")
        if prebuilt_cache_fast_init:
            if preloaded_eval_payload is None:
                raise RuntimeError("prebuilt cache fast initialization lost eval cache payload")
            encoded_eval_payload = validate_encoded_eval_cache_payload(
                preloaded_eval_payload, encoded_eval_cache_metadata
            )
        else:
            encoded_eval_payload = load_encoded_eval_cache(
                args.embedding_eval_cache_input,
                encoded_eval_cache_metadata,
            )
        holdout_data = encoded_eval_payload["holdout_data"]
        holdout_data["states"] = encoded_eval_payload["states"]
        holdout_dl = None
        print(
            f"  Loaded {holdout_data['states'].shape[0]:,} clean holdout states "
            f"(dim={holdout_data['states'].shape[1]})"
        )
    else:
        print(f"Streaming clean holdout data from LeRobotDataset ({N:,} frames)...")

        holdout_subset = Subset(holdout_dataset, holdout_indices)
        holdout_dl = DataLoader(holdout_subset, **encode_dl_kwargs)
        holdout_actions_list = []
        holdout_success_list = []
        holdout_source_list = []
        holdout_done_list = []
        holdout_is_valid_list = []
        holdout_frame_index_list = []

        holdout_t0 = time.perf_counter()
        for batch in holdout_dl:
            batch["action"] = _get_actions(batch, action_columns)
            canonicalize_post_terminal_steps(batch)
            if action_mode == "relative":
                # After canonicalization so copied post-terminal steps relativize
                # against the SAME anchor as the live steps (per-sample anchor).
                batch["action"] = _relativize_action_chunk(
                    batch["action"], batch["observation.state"]
                )
            B_h = batch["action"].shape[0]
            if batch["action"].shape[1:] != (k, action_dim):
                raise ValueError(
                    f"Holdout action shape must be (B, {k}, {action_dim}), got "
                    f"{tuple(batch['action'].shape)}"
                )
            holdout_actions_list.append(batch["action"].reshape(B_h, -1).cpu())
            holdout_success_list.append(batch["success"].long().cpu())
            holdout_source_list.append(batch["source"].long().cpu())
            for key, output in (
                ("done", holdout_done_list),
                ("is_valid", holdout_is_valid_list),
                ("frame_index", holdout_frame_index_list),
            ):
                if key not in batch:
                    raise KeyError(f"Holdout batch is missing required trajectory key {key!r}")
                value = batch[key]
                if value.ndim > 1:
                    value = value.reshape(B_h, -1)[:, 0]
                output.append(value.cpu())

        holdout_elapsed = time.perf_counter() - holdout_t0
        print(f"  Holdout metadata extracted in {holdout_elapsed:.1f}s")

        holdout_data = {
            "actions": torch.cat(holdout_actions_list, dim=0).to(device),
            "success": torch.cat(holdout_success_list, dim=0),
            "source": torch.cat(holdout_source_list, dim=0),
            "done": torch.cat(holdout_done_list, dim=0).bool(),
            "is_valid": torch.cat(holdout_is_valid_list, dim=0).bool(),
            "frame_indices": torch.cat(holdout_frame_index_list, dim=0).long(),
            "episode_indices": eval_episode_idx_arr[holdout_idx],
            "dataset_indices": eval_dataset_idx_arr[holdout_idx],
            "original_frame_indices": eval_original_frame_idx_arr[holdout_idx],
        }
        del (
            holdout_actions_list,
            holdout_success_list,
            holdout_source_list,
            holdout_done_list,
            holdout_is_valid_list,
            holdout_frame_index_list,
        )

    def _shape_refresh_batch(batch: dict) -> dict:
        """Apply intervention/shift reward shaping and terminal canonicalization to a
        freshly decoded batch — the single shaping path for buffer seeding and
        per-step refresh (sync or async)."""
        if action_mode == "relative":
            raise NotImplementedError(
                "--action-mode relative reached the streaming replay-buffer refresh "
                "path; it is gated to the flat-cache regime at arg resolution — this "
                "is a wiring bug, not a user error."
            )
        # The per-frame lookups below are positional over each sub-dataset's loaded rows;
        # LeRobot's batch['index'] is the absolute frame index (they differ under
        # --dataset-episodes). From here on batch['index'] is the row position.
        batch["index"] = local_frame_index(
            batch["index"], batch["dataset_index"], frame_index_by_dataset
        )
        apply_intervention_reward_shaping(
            batch,
            intervention_by_frame=intervention_by_frame,
            intervention_by_dataset=intervention_values_by_dataset,
            intervention_negative_reward=args.intervention_negative_reward,
            reward_shift=args.reward_shift,
            horizon=td_k,
        )
        # Run after reward shaping: offsets strictly after the first done become
        # exact copies of the shaped terminal action/reward/done step, regardless
        # of what invalid retract/reset rows contain.
        canonicalize_post_terminal_steps(batch)
        return batch

    def _gpu_pack_native_images(batch: dict) -> dict:
        """Crop+resize+float raw uint8 native camera frames on GPU, then quantize.

        Only used on the --uint8-native-images path. Mutates and returns ``batch``
        with each camera key replaced by a uint8 GPU tensor at the buffer's 224^2
        resolution — exactly what VisionReplayBuffer.refresh() consumes (its
        _prep_images passes an already-packed uint8-on-device tensor through). The
        crop map / target / reference match the per-worker proxy, so the buffer
        pixel content is identical up to the CPU->GPU resize LSB.
        """
        gpu_imgs = {cam: batch[cam].to(device, non_blocking=True) for cam in camera_keys}
        processed = gpu_preprocessor(gpu_imgs)
        for cam in camera_keys:
            batch[cam] = VisionReplayBuffer.pack_images_uint8(processed[cam])
        return batch

    @torch.no_grad()
    def _encode_cached_visual(
        batch: dict[str, torch.Tensor], *, timestamp_idx: int, augment: bool = False
    ) -> torch.Tensor:
        features = []
        norm_mean = image_norm_mean.reshape(1, 3, 1, 1).float().to(device)
        norm_std = image_norm_std.reshape(1, 3, 1, 1).float().to(device)
        for cam_key in camera_keys:
            img = batch[cam_key].to(device, non_blocking=True)[:, timestamp_idx]
            if augment:
                if augment_fn is None:
                    raise RuntimeError("Cached augmented encoding requires a GPU augment function")
                img = prepare_cached_iql_images(img, augment_fn=augment_fn)
            else:
                # VisionReplayBuffer always stores images as uint8, regardless
                # of whether augmentation is enabled. Clean-only and multi-view
                # caches therefore share this pixel contract.
                img = prepare_cached_iql_images(img)
            img = (img - norm_mean) / norm_std
            if args.channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
            cam_encoder = (
                encoder[cam_key.removeprefix("observation.images.")]
                if separate_encoders
                else encoder
            )
            if args.encoder_autocast_bf16 and img.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feat = cam_encoder(img)
                feat = feat.float()
            else:
                feat = cam_encoder(img)
            features.append(feat)
        return torch.cat(features, dim=-1)

    def _build_flat_frame_table() -> dict[str, torch.Tensor]:
        """Read immutable frame-level fields without decoding any images."""

        def column(hf_dataset, name: str, *, dtype: torch.dtype) -> torch.Tensor:
            values = hf_dataset[name]
            if isinstance(values, torch.Tensor):
                return values.to(dtype=dtype)
            values = list(values)
            if values and (
                isinstance(values[0], (torch.Tensor, list, tuple)) or hasattr(values[0], "shape")
            ):
                return torch.stack([torch.as_tensor(value, dtype=dtype) for value in values])
            return torch.tensor(values, dtype=dtype)

        parts: dict[str, list[torch.Tensor]] = defaultdict(list)
        for ds_i, sub_ds in enumerate(sub_datasets):
            hf = sub_ds.hf_dataset
            n = len(sub_ds)
            state = column(hf, "observation.state", dtype=torch.float32)[:n]
            action_values = [
                column(hf, action_column, dtype=torch.float32)[:n]
                for action_column in action_columns
            ]
            action = torch.cat(
                [value.unsqueeze(-1) if value.ndim == 1 else value for value in action_values],
                dim=-1,
            )
            parts["proprio"].append(state)
            parts["action"].append(action)
            parts["reward"].append(column(hf, "reward", dtype=torch.float32)[:n])
            parts["done"].append(column(hf, "done", dtype=torch.long)[:n])
            parts["is_valid"].append(
                column(hf, "is_valid", dtype=torch.bool)[:n]
                if "is_valid" in hf.column_names
                else torch.ones(n, dtype=torch.bool)
            )
            parts["success"].append(column(hf, "success", dtype=torch.long)[:n])
            parts["source"].append(column(hf, "source", dtype=torch.long)[:n])
            parts["intervention"].append(intervention_values_by_dataset[ds_i][:n].long())
            parts["frame_index"].append(column(hf, "frame_index", dtype=torch.long)[:n])
        table = {name: torch.cat(values) for name, values in parts.items()}
        table["dataset_index"] = global_dataset_idx_arr
        table["episode_index"] = global_episode_idx_arr
        return table

    def _encode_physical_frame_slice(row_slice: torch.Tensor) -> torch.Tensor:
        """Encode each physical frame in ``row_slice`` EXACTLY ONCE (clean + augmented
        views) at timestamp 0. Halves encoder forwards + CPU decode vs the two-timestamp
        transition pass. Returns an (M, [V,] D) CPU tensor aligned to ``row_slice``."""
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        if args.channels_last:
            encoder.to(memory_format=torch.channels_last)
        slice_subset = Subset(dataset, row_slice.tolist())
        slice_dl_kwargs = dict(
            batch_size=args.encoding_batch_size,
            shuffle=False,
            num_workers=num_workers_per_loader,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=args.prefetch_factor if use_workers else None,
            persistent_workers=args.persistent_workers if use_workers else False,
        )
        if use_workers and args.multiprocessing_context != "none":
            slice_dl_kwargs["multiprocessing_context"] = args.multiprocessing_context
        slice_dl = DataLoader(slice_subset, **slice_dl_kwargs)
        visual_parts: list[torch.Tensor] = []
        encoded = 0
        total = int(row_slice.numel())
        for batch in slice_dl:
            clean = _encode_cached_visual(batch, timestamp_idx=0)
            if cache_augmented_views > 0:
                views = [clean]
                for _ in range(cache_augmented_views):
                    views.append(_encode_cached_visual(batch, timestamp_idx=0, augment=True))
                visual_parts.append(torch.stack(views, dim=1).cpu())
            else:
                visual_parts.append(clean.cpu())
            encoded += int(clean.shape[0])
            if encoded == total or encoded % max(args.encoding_batch_size * 10, 1) == 0:
                print(f"  Encoded {encoded:,}/{total:,} physical frames")
        if encoded != total:
            raise RuntimeError(f"one-pass encode saw {encoded:,} frame(s), expected {total:,}")
        return torch.cat(visual_parts, dim=0)

    def _build_flat_cache_one_pass() -> FlatEncodedTrajectoryCache:
        """One-pass PHYSICAL-FRAME encoder for the schema-v3 flat cache.

        Encodes every unique physical frame in the flat-cache frame set exactly once
        (vs the old two-timestamp transition pass that double-encoded successors).
        The frame set and anchor eligibility come from ``derive_flat_cache_rows``."""
        frame_table = _build_flat_frame_table()
        anchors = torch.as_tensor(all_train_idx, dtype=torch.long)
        rows, anchor_eligible, boundary_added = derive_flat_cache_rows(
            source_anchor_indices=anchors,
            frame_table=frame_table,
            action_horizon=k,
        )
        print(
            f"One-pass physical-frame encode: {int(rows.numel()):,} unique frame(s) "
            f"(anchors={anchors.numel():,}, boundary_successors={boundary_added:,}; "
            f"batch_size={args.encoding_batch_size})..."
        )
        cache_t0 = time.perf_counter()
        visual = _encode_physical_frame_slice(rows)
        cache = build_flat_cache_from_physical_frames(
            visual=visual,
            rows=rows,
            anchor_eligible=anchor_eligible,
            frame_table=frame_table,
            action_horizon=k,
        )
        elapsed = time.perf_counter() - cache_t0
        print(
            f"  Encoded flat trajectory cache: {cache.size:,} frames in {elapsed:.1f}s "
            f"x {cache.num_visual_views} visual view(s) "
            f"({cache.memory_gb:.3f} GB on CPU before device move; "
            f"boundary_successors_added={boundary_added})"
        )
        return cache

    encoded_cache_metadata: dict = {}
    if use_embedding_cache:
        train_indices_sha256 = (
            preloaded_train_metadata["train_indices_sha256"]
            if prebuilt_cache_fast_init and preloaded_train_metadata is not None
            else hashlib.sha256(np.asarray(all_train_idx, dtype=np.int64).tobytes()).hexdigest()
        )
        if prebuilt_cache_fast_init:
            if preloaded_train_metadata is None:
                raise RuntimeError("prebuilt cache fast initialization lost train metadata")
            repo_revisions = preloaded_train_metadata["repo_revisions_v3_0"]
        else:
            repo_revisions = dataset_commit_revisions(repo_ids, dataset_revisions)
        encoded_cache_metadata = {
            "repo_ids": repo_ids,
            "repo_revisions_v3_0": repo_revisions,
            "train_indices_sha256": train_indices_sha256,
            "n_train": n_train,
            "encoder_artifact": args.encoder_artifact,
            "encoder_state_sha256": encoder_state_sha256,
            "camera_keys": camera_keys,
            "camera_crop_boxes": {key: list(value) for key, value in dp_camera_crops.items()},
            "crop_reference_hw": (
                list(crop_reference_hw) if crop_reference_hw is not None else None
            ),
            "image_hw": list(image_hw),
            "image_norm_mean": image_norm_mean.tolist(),
            "image_norm_std": image_norm_std.tolist(),
            "encoder_autocast_bf16": args.encoder_autocast_bf16,
            "channels_last": args.channels_last,
            "augmentation": args.augmentation,
            "aug_shift_frac": args.aug_shift_frac,
            "augmentation_spec": iql_augmentation_spec(shift_frac=args.aug_shift_frac),
            "augmented_views": cache_augmented_views,
            "seed": args.seed,
            "action_columns": action_columns,
            # Dynamic key (stripped from the flat-cache static contract): the cache
            # stores raw command-pose columns either way; relativization happens at
            # batch time. Cross-mode safety is enforced by action_columns/action_dim.
            "action_mode": action_mode,
            # The FiLM head is NOT a cache-metadata field (applied at Q-scoring
            # time, not cache-build time — does not change the cached embeddings /
            # actions). See the eval-cache metadata note above.
            "action_horizon": k,
            "td_horizon": td_k,
            # RAW stored-column width (== the cache action tensor width), NOT the
            # critic step dim — the prebuilt-cache loader checks tensor==metadata.
            "action_dim": raw_action_dim,
            "gamma": args.gamma,
            "intervention_negative_reward": args.intervention_negative_reward,
            "reward_shift": args.reward_shift,
        }
        train_selectors = selector_metadata(dataset_selectors, repo_ids)
        if train_selectors:
            # Binding only when a selection is active (see the eval-cache note above).
            encoded_cache_metadata["dataset_episodes"] = train_selectors

    # ---- 11b. Seed replay buffer ----
    if use_embedding_cache:
        cache_build_t0 = time.perf_counter()
        if args.embedding_cache_input is not None:
            print(f"Loading encoded replay cache: {args.embedding_cache_input}")
            expected_flat_metadata = flat_cache_expected_metadata(encoded_cache_metadata)
            if prebuilt_cache_fast_init:
                if preloaded_train_cache is None or preloaded_train_metadata is None:
                    raise RuntimeError(
                        "prebuilt cache fast initialization lost loaded train payload"
                    )
                validate_flat_encoded_cache_metadata(
                    preloaded_train_metadata, expected_flat_metadata
                )
                replay_buffer = preloaded_train_cache
            else:
                cache_header = torch.load(
                    args.embedding_cache_input, map_location="cpu", weights_only=True
                )
                cache_schema = cache_header.get("schema_version")
                del cache_header
                if cache_schema != FLAT_ENCODED_CACHE_SCHEMA_VERSION:
                    raise ValueError(
                        "Cached IQL training requires the horizon-independent schema-v3 flat "
                        f"cache; found schema {cache_schema!r}. Rebuild it with "
                        "--precompute-embeddings --embedding-cache-output."
                    )
                replay_buffer = load_flat_encoded_cache(
                    args.embedding_cache_input,
                    expected_flat_metadata,
                )
            print(
                f"  Loaded {replay_buffer.size:,} samples x "
                f"{replay_buffer.num_visual_views} view(s) "
                f"({replay_buffer.memory_gb:.3f} GB)"
            )
        else:
            replay_buffer = _build_flat_cache_one_pass()
        if independent_target_samples:
            if not isinstance(replay_buffer, FlatEncodedTrajectoryCache):
                raise TypeError("independent target views require a schema-v3 flat cache")
            if replay_buffer.num_visual_views < 2:
                raise ValueError("independent target views require a multi-view flat cache")
            if independent_target_samples > replay_buffer.num_visual_views:
                raise ValueError(
                    "--target-view-samples exceeds the loaded cache view count: "
                    f"samples={independent_target_samples}, "
                    f"views={replay_buffer.num_visual_views}"
                )
        cache_elapsed = time.perf_counter() - cache_build_t0
        if args.embedding_cache_output is not None:
            print(f"Saving encoded replay cache: {args.embedding_cache_output}")
            save_flat_encoded_cache(
                args.embedding_cache_output,
                replay_buffer,
                flat_cache_expected_metadata(encoded_cache_metadata),
                normalization_action_horizon=k,
            )
            size_gib = args.embedding_cache_output.resolve().stat().st_size / (1024**3)
            print(f"  Saved encoded replay cache ({size_gib:.3f} GiB on disk)")
        normalization_sample_frac = 1.0
    else:
        if replay_buffer is None:
            raise RuntimeError("Image replay buffer was not initialized")
        seed_size = min(
            replay_buffer.capacity,
            n_train,
            max(args.batch_size, 1024, int(0.1 * n_train)),
        )
        print(f"Seeding replay buffer with {seed_size:,} samples (rest fills during training)...")
        fill_t0 = time.perf_counter()
        seed_subset = Subset(dataset, all_train_idx)
        seed_dl = DataLoader(
            seed_subset,
            batch_size=64,
            sampler=RandomSampler(seed_subset, replacement=False),
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        # uint8-native seed loop runs num_workers=0 in the MAIN process, so it flips
        # raw-uint8 mode on the SHARED crop proxies and MUST restore it afterwards.
        # The holdout encode loader (already spawned above with persistent_workers)
        # holds PRE-flip pickled worker copies, so its float32 image path is unaffected
        # for the life of those workers; the async refresh producer that flips per
        # worker starts only AFTER this block. Restore in a try/finally so an exception
        # mid-seed can never leave the main-process proxies stuck in raw mode.
        filled = 0
        try:
            # Enable INSIDE the try so a partial flip (exception mid-walk) is still
            # unconditionally restored by the finally below (disable is idempotent).
            if uint8_native:
                set_raw_uint8_mode_on_subdatasets(dataset._datasets, True)
            for batch in seed_dl:
                if uint8_native:
                    filled += replay_buffer.refresh(
                        _gpu_pack_native_images(_shape_refresh_batch(batch))
                    )
                else:
                    filled += replay_buffer.refresh(_shape_refresh_batch(batch))
                if filled >= seed_size:
                    break
        finally:
            if uint8_native:
                set_raw_uint8_mode_on_subdatasets(dataset._datasets, False)
        del seed_dl, seed_subset

        fill_elapsed = time.perf_counter() - fill_t0
        cache_elapsed = fill_elapsed
        normalization_sample_frac = replay_buffer.size / n_train
        print(f"  Buffer seed: {replay_buffer.size:,} samples in {fill_elapsed:.1f}s")
        print(
            f"  Buffer fill: {replay_buffer.fill_pct * 100:.1f}% ({replay_buffer.memory_gb:.2f} GB)"
        )

    if args.embedding_cache_only:
        print(
            f"Embedding-cache-only preprocessing complete: {replay_buffer.size:,} samples x "
            f"{replay_buffer.num_visual_views} view(s) -> {args.embedding_cache_output.resolve()}"
        )
        return

    # ---- Compute normalization statistics from buffer ----
    print("Computing normalization statistics from buffer...")
    if replay_buffer.size < 0.05 * n_train:
        print(
            "  WARNING: normalization stats use the initial replay-buffer stats seed "
            f"({replay_buffer.size:,}/{n_train:,} randomly sampled frames); "
            "this is a startup-speed tradeoff."
        )
    buffer_stats = (
        replay_buffer.compute_stats(action_horizon=k)
        if use_embedding_cache
        else replay_buffer.compute_stats()
    )
    if use_embedding_cache and action_mode == "relative":
        # Recompute action stats over the CRITIC-INPUT action space: the cache's
        # own compute_stats reduces the raw stored columns, which is wrong when a
        # relativization sits between cache and critic.
        stats_batch = replay_buffer.build_transitions(
            replay_buffer.anchor_rows,
            action_horizon=k,
            td_horizon=k,
            current_view=0,
        )
        relative = _relativize_action_chunk(
            stats_batch["action"], stats_batch["observation.state"]
        ).reshape(-1, action_dim)
        buffer_stats["action"] = {
            "mean": relative.mean(dim=0).cpu(),
            "std": relative.std(dim=0).clamp(min=1e-6).cpu(),
        }
        del stats_batch, relative
    proprio_mean = buffer_stats["state"]["mean"].to(device)
    proprio_std = buffer_stats["state"]["std"].to(device)
    action_mean = buffer_stats["action"]["mean"].to(device)
    action_std = buffer_stats["action"]["std"].to(device)
    min_proprio_std = float(proprio_std.min().item())
    if min_proprio_std <= 1e-4:
        raise ValueError(
            "proprio_std has near-zero component "
            f"(min={min_proprio_std:.2e}); cannot normalize. Check data quality."
        )
    min_action_std = float(action_std.min().item())
    if min_action_std <= 1e-4:
        raise ValueError(
            "action_std has near-zero component "
            f"(min={min_action_std:.2e}); cannot normalize. Check data quality."
        )

    print("  Proprio stats (per-dim):")
    print(f"    mean: {[f'{v:.4f}' for v in proprio_mean.tolist()]}")
    print(f"    std:  {[f'{v:.4f}' for v in proprio_std.tolist()]}")
    print("  Action stats (per-dim):")
    print(f"    mean: {[f'{v:.4f}' for v in action_mean.tolist()]}")
    print(f"    std:  {[f'{v:.4f}' for v in action_std.tolist()]}")

    if holdout_data is None:
        print("  Holdout disabled (--eval-freq 0, no eval cache): no holdout actions.")
    elif encoded_eval_payload is None:
        holdout_actions_raw = holdout_data["actions"]
        holdout_actions_reshaped = holdout_actions_raw.reshape(-1, k, action_dim)
        holdout_data["actions"] = (
            (holdout_actions_reshaped - action_mean) / action_std
        ).reshape_as(holdout_actions_raw)
        print(f"  Holdout actions normalized (shape={holdout_actions_raw.shape})")
    else:
        print(f"  Holdout actions loaded pre-normalized (shape={holdout_data['actions'].shape})")

    if args.embedding_eval_cache_output is not None:
        if holdout_dl is None:
            raise RuntimeError(
                "Cannot build an encoded eval cache without a clean holdout DataLoader"
            )
        print("Encoding clean holdout states for reusable evaluation cache...")
        encoder.eval()
        if args.channels_last:
            encoder.to(memory_format=torch.channels_last)
        holdout_data["states"] = encode_holdout_images(
            holdout_dl=holdout_dl,
            encoder=encoder,
            camera_keys=camera_keys,
            separate_encoders=separate_encoders,
            device=device,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            image_norm_mean=image_norm_mean,
            image_norm_std=image_norm_std,
            encoder_autocast_bf16=args.encoder_autocast_bf16,
            channels_last=args.channels_last,
        )
        save_encoded_eval_cache(
            args.embedding_eval_cache_output,
            states=holdout_data["states"],
            holdout_data=holdout_data,
            metadata=encoded_eval_cache_metadata,
        )
        size_mib = args.embedding_eval_cache_output.resolve().stat().st_size / (1024**2)
        print(
            f"  Saved clean eval cache: {holdout_data['states'].shape[0]:,} states "
            f"({size_mib:.1f} MiB)"
        )
        if args.embedding_eval_cache_only:
            print(
                "Embedding-eval-cache-only preprocessing complete: "
                f"{len(holdout_indices):,} states -> "
                f"{args.embedding_eval_cache_output.resolve()}"
            )
            return

    # Move buffer to GPU for fast sampling
    replay_buffer.to_device(device)

    refresh_queue: queue.Queue | None = None
    refresh_stop: threading.Event | None = None
    refresh_thread: threading.Thread | None = None
    refresh_producer_error: list[BaseException] = []
    buffer_refresh_iter = None
    if use_embedding_cache:
        print("  Refresh DataLoader: disabled (--precompute-embeddings)")
        print()
    else:
        # Create refresh DataLoader for per-step buffer refresh
        if n_train < args.buffer_refresh_rate:
            raise ValueError(
                f"--buffer-refresh-rate {args.buffer_refresh_rate} exceeds the train set "
                f"size {n_train}; the refresh DataLoader (drop_last=True) would never "
                "yield a batch and rotation would silently stop."
            )
        refresh_subset = Subset(dataset, all_train_idx)
        if uint8_native:
            # uint8-native supersedes the post-resize worker quantize: workers return
            # RAW uint8 native frames (crop+resize+float+pack all deferred to GPU). The
            # view lazily flips raw mode on each WORKER's OWN pickled proxy copies, so
            # the main-process dataset and holdout loader are untouched.
            refresh_subset = _RawUint8SubsetView(refresh_subset)
        elif args.async_buffer_refresh:
            # Bit-identical uint8 packing moved into the workers (4× less IPC);
            # the producer's pack step passes uint8 through unchanged and pins.
            refresh_subset = _Uint8RefreshDataset(refresh_subset, camera_keys)
        refresh_dl_kwargs = dict(
            batch_size=args.buffer_refresh_rate,
            num_workers=num_workers_per_loader,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=args.prefetch_factor if use_workers else None,
            persistent_workers=args.persistent_workers if use_workers else False,
        )
        if use_workers and args.multiprocessing_context != "none":
            refresh_dl_kwargs["multiprocessing_context"] = args.multiprocessing_context
        if args.async_buffer_refresh:
            # The producer thread draws i.i.d. refresh batches from the full train set
            # with its own seeded generator.
            refresh_generator = torch.Generator()
            refresh_generator.manual_seed(args.seed * 100_003)
            buffer_refresh_loader = DataLoader(
                refresh_subset,
                sampler=RandomSampler(
                    refresh_subset,
                    replacement=True,
                    generator=refresh_generator,
                ),
                **refresh_dl_kwargs,
            )
        else:
            buffer_refresh_loader = DataLoader(
                refresh_subset,
                sampler=RandomSampler(refresh_subset, replacement=True),
                **refresh_dl_kwargs,
            )
        buffer_refresh_iter = iter(buffer_refresh_loader)

        # Async producer: owns the (blocking) DataLoader iterator + CPU-side batch
        # shaping. The main thread stays the ONLY mutator of replay-buffer tensors,
        # so a training-batch sample can never observe a half-written slot.
        if args.async_buffer_refresh:
            refresh_queue = queue.Queue(maxsize=4)
            refresh_stop = threading.Event()
            pin_refresh_batches = torch.cuda.is_available()

            def _refresh_producer() -> None:
                # Any producer failure is stored and re-raised by the training loop
                # on its next poll — a dead producer must never silently stop
                # buffer rotation.
                try:
                    it = buffer_refresh_iter
                    while not refresh_stop.is_set():
                        try:
                            raw = next(it)
                        except StopIteration:
                            it = iter(buffer_refresh_loader)
                            continue
                        shaped = _shape_refresh_batch(raw)
                        # Image handling depends on the throughput path:
                        # - uint8-native: workers already returned RAW uint8 native
                        #   frames. Do NOT pack/resize here (crop+resize+float+pack run
                        #   on GPU in the main-thread poll); just pin the native uint8
                        #   for a non-blocking H2D. Native frames are ~8x the 224^2
                        #   footprint, but still uint8 and no CPU resize on this thread.
                        # - async path without raw-uint8: workers returned float32 224^2 resized
                        # frames,
                        #   pre-pack to the buffer's uint8 format + pin,
                        #   so the poll does a small non-blocking H2D instead of a
                        #   4x-larger pageable copy behind queued forward kernels.
                        for cam in camera_keys:
                            img = shaped[cam]
                            if not uint8_native:
                                img = VisionReplayBuffer.pack_images_uint8(img)
                            if pin_refresh_batches:
                                img = img.pin_memory()
                            shaped[cam] = img
                        while not refresh_stop.is_set():
                            try:
                                refresh_queue.put(shaped, timeout=1.0)
                                break
                            except queue.Full:
                                continue
                except BaseException as exc:  # noqa: BLE001 - surfaced in main loop
                    refresh_producer_error.append(exc)

            # NOTE: the thread is STARTED just before the training loop (after the
            # auto-resume early-return) so a completed-resume exit never leaves a
            # producer running against a torn-down loader.
            refresh_thread = threading.Thread(
                target=_refresh_producer, daemon=True, name="buffer-refresh-producer"
            )

        print(
            f"  Refresh DataLoader: batch_size={args.buffer_refresh_rate}, "
            f"num_workers_per_loader={num_workers_per_loader}"
            + (", async producer thread" if args.async_buffer_refresh else "")
        )
        print()

    torch.cuda.empty_cache()

    if holdout_data is None:
        holdout_success_frames = 0
        holdout_failure_frames = 0
    else:
        holdout_success_frames = (holdout_data["success"] == 1).sum().item()
        holdout_failure_frames = (holdout_data["success"] == 0).sum().item()
    if holdout_data is None:
        print("  Holdout: disabled (--eval-freq 0, no eval cache)")
    else:
        print(
            f"  Holdout: {n_holdout_frames} frames "
            f"(success={holdout_success_frames}, failure={holdout_failure_frames})"
        )
    print()

    # ---- 12. Create VisionIQL (unified encoder + Q/V module) ----
    print("Creating VisionIQL module...")
    q1 = QNetwork(
        state_dim,
        chunked_action_dim,
        hidden_dims=hidden_dims,
        use_layer_norm=True,
    ).to(device)
    q2 = QNetwork(
        state_dim,
        chunked_action_dim,
        hidden_dims=hidden_dims,
        use_layer_norm=True,
    ).to(device)
    # DIVL value support: derived from the empirical SHAPED, done-aware discounted
    # return-to-go range (the quantity the distributional V regresses toward) unless
    # --v-min/--v-max are given. divl_v_min/divl_v_max are persisted in checkpoint
    # metadata so eval reconstructs the same categorical support.
    return_range = empirical_discounted_rtg_range(
        sub_datasets,
        repo_ids,
        gamma=args.gamma,
        reward_shift=args.reward_shift,
        intervention_negative_reward=args.intervention_negative_reward,
        intervention_values_by_dataset=intervention_values_by_dataset,
    )
    divl_v_min, divl_v_max = resolve_divl_value_support(args.v_min, args.v_max, return_range)
    print(
        f"  DIVL value support: [{divl_v_min:.3f}, {divl_v_max:.3f}] "
        f"(num_atoms={args.num_atoms}, returns=[{return_range[0]:.3f}, "
        f"{return_range[1]:.3f}], tau_base={args.tau_base}, "
        f"tau_entropy_alpha={args.tau_entropy_alpha})"
    )
    v_net = DistributionalVNetwork(
        state_dim,
        hidden_dims=hidden_dims,
        use_layer_norm=True,
        num_atoms=args.num_atoms,
        v_min=divl_v_min,
        v_max=divl_v_max,
    ).to(device)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    model = iql_build_model(
        encoder=encoder,
        q1=q1,
        q2=q2,
        v_net=v_net,
        camera_keys=camera_keys,
        separate_encoders=separate_encoders,
        args=args,
        image_norm_mean=image_norm_mean,
        image_norm_std=image_norm_std,
        action_dim=action_dim,
        state_dim=state_dim,
        device=device,
    )

    if args.channels_last:
        # Convert conv weights for online + target encoders; encode() converts
        # the image tensors so conv kernels hit the NHWC tensor-core path.
        model.encoder.to(memory_format=torch.channels_last)
        model.encoder_target.to(memory_format=torch.channels_last)
        print("  Encoder memory format: channels_last (--channels-last)")

    # torch.compile the online encoder after VisionIQL creation; the target
    # encoder stays eager.
    if use_embedding_cache:
        print("  Online encoder torch.compile skipped (--precompute-embeddings)")
    else:
        print("  Compiling online encoder with torch.compile...")
    if not use_embedding_cache and separate_encoders:
        for key in list(model.encoder.keys()):
            model.encoder[key] = torch.compile(model.encoder[key], mode="default")
    elif not use_embedding_cache:
        model.encoder = torch.compile(model.encoder, mode="default")

    n_q_params = sum(p.numel() for p in model.q1.parameters())
    n_v_params = sum(p.numel() for p in model.v_net.parameters())
    n_enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Q-network params: {n_q_params:,} (x2)")
    print(f"  V-network params: {n_v_params:,} (online only; target critics are Q1/Q2)")
    print(f"  Encoder params: {n_enc_params:,} (frozen)")
    print(f"  Total trainable: {2 * n_q_params + n_v_params:,}")
    print()

    # ---- 13. Single optimizer (Q + V param groups) ----
    optimizer = iql_build_optimizer(model, args)

    # ---- 14. Initialize W&B ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_clipping = args.clip_targets_min is not None or args.clip_targets_max is not None
    if has_clipping:
        cmin = args.clip_targets_min if args.clip_targets_min is not None else ""
        cmax = args.clip_targets_max if args.clip_targets_max is not None else ""
        clip_str = f"clip{cmin}{cmax}"
    else:
        clip_str = "noclip"

    enc_type_str = "sep" if separate_encoders else "shared"
    aug_str = {"none": "noaug", "gpu": "gpuaug"}.get(aug_mode, aug_mode)
    steps_k = args.training_steps // 1000
    auto_run_name = f"iql-{enc_type_str}-direct-{aug_str}-{clip_str}-bs{args.batch_size}-{steps_k}k"

    if args.wandb_notes is None:
        auto_notes = (
            f"Vision IQL - direct LeRobot loading, "
            f"{'separate' if separate_encoders else 'shared'} encoder, "
            f"{aug_mode} augmentation, "
            f"{'clip [' + str(args.clip_targets_min) + ',' + str(args.clip_targets_max) + ']' if has_clipping else 'no clip'}, "
            f"{num_episodes} episodes, {n_train} train frames"
        )
    else:
        auto_notes = args.wandb_notes

    run_name = args.wandb_run_name or auto_run_name
    # Resume state lives under <output-dir>/_resume/<hash(run name)>: rerunning the same command
    # into the same --output-dir resumes, with or without W&B.
    auto_resume_enabled = not args.no_auto_resume
    validate_auto_resume_frequencies(
        auto_resume_enabled=auto_resume_enabled,
        log_freq=args.log_freq,
        resume_checkpoint_freq=args.resume_checkpoint_freq,
        use_wandb=args.use_wandb,
    )

    resume_base_dir = output_dir
    resume_manager = AutoResumeManager(
        resume_base_dir,
        run_name=run_name,
        enabled=auto_resume_enabled,
    )
    resume_meta = None
    resume_manager.acquire_lock()
    resume_meta = resume_manager.load_metadata()
    if resume_meta is not None and resume_meta.completed:
        raise RuntimeError(
            f"Run '{run_name}' in {resume_base_dir} is already marked completed locally. "
            "Use a new --output-dir (or --wandb-run-name) for a fresh training run."
        )
    if resume_meta is not None:
        output_dir = Path(resume_meta.run_dir)
        print(f"  Auto-resume: using existing run directory {output_dir}")
    elif auto_resume_enabled:
        resume_manager.initialize_fresh_run(output_dir)
        print(f"  Auto-resume: initialized local state under {resume_manager.resume_dir}")

    wandb_config = {
        "data": {
            "repo_ids": repo_ids,
            "eval_repo_ids": eval_repo_ids,
            "dataset_revisions": {
                repo_id: dataset_revision(dataset_revisions, repo_id)
                for repo_id in [*repo_ids, *eval_repo_ids]
            },
            "dataset_episodes": selector_metadata(dataset_selectors, [*repo_ids, *eval_repo_ids]),
            "train_episode_filter": None,
            "use_external_eval": use_external_eval,
            "root": str(root) if root else None,
            "camera_keys": camera_keys,
            "camera_crop_boxes": dict(dp_camera_crops),
            "crop_reference_hw": list(crop_reference_hw) if crop_reference_hw is not None else None,
            "n_cameras": n_cameras,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "action_columns": action_columns,
            "fps": fps,
            "n_episodes": num_episodes,
            "n_train_frames": n_train,
            "n_holdout_frames": n_holdout_frames,
        },
        "encoder": {
            "artifact": args.encoder_artifact,
            "feature_dim": feature_dim,
            "separate_encoders": separate_encoders,
            "camera_key_order": encoder_camera_key_order,
            "action_target": action_target,
            "cartesian_action_frame": cartesian_action_frame,
            "artifact_resolved": encoder_source,
            "camera_crop_boxes": dict(dp_camera_crops),
            "frozen": True,
            "encoder_grad_source": "q-only",
            "compiled": not use_embedding_cache,
        },
        "augmentation": {
            "mode": aug_mode,
            "enabled": aug_mode != "none",
            "online_streaming": not use_embedding_cache,
            "embedding_cache_augmented_views": cache_augmented_views,
            "embedding_cache_total_views": 1 + cache_augmented_views,
            "embedding_cache_replay_pixel_roundtrip": cache_augmented_views > 0,
            "target_view_sampling": target_view_sampling,
            "target_view_samples": target_view_samples,
            "target_view_iid_with_replacement": independent_target_samples > 0,
            "target_view_reduce": (
                "mean_scalar_qv_outputs" if independent_target_samples > 1 else "single"
            ),
        },
        "iql": {
            "chunk_size": k,
            "prediction_horizon": prediction_horizon,
            "n_action_steps": k,
            "td_horizon_steps": td_k,
            "td_horizon_chunks": td_k // k,
            "td_lambda_curriculum": td_lambda_enabled,
            "td_lambda_initial": args.td_lambda_initial if td_lambda_enabled else None,
            "td_lambda_hold_steps": args.td_lambda_hold_steps if td_lambda_enabled else None,
            "td_lambda_cosine_end_step": (
                args.td_lambda_cosine_end_step if td_lambda_enabled else None
            ),
            "clean_target_features": False,
            "target_view_sampling": target_view_sampling,
            "target_view_samples": target_view_samples,
            "num_action_samples": args.num_action_samples,
            "gamma": args.gamma,
            "tau": args.tau,
            "target_topology": "target_q_online_v",
            "target_encoder_for_q_labels": True,
            "transition_boundary_contract": "is_valid_current_terminal_tails_retained",
            "clip_targets_min": args.clip_targets_min,
            "clip_targets_max": args.clip_targets_max,
            "intervention_negative_reward": args.intervention_negative_reward,
            "reward_shift": args.reward_shift,
            "value_training_mode": "iql",
            "q_heads_trained": True,
            "candidate_action_reranking_valid": True,
            "iql_recipe": getattr(args, "iql_recipe", None),
            "distributional": True,
            "num_atoms": args.num_atoms,
            "v_min": divl_v_min,
            "v_max": divl_v_max,
            "hl_gauss_sigma_ratio": args.hl_gauss_sigma_ratio,
            "tau_base": args.tau_base,
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "tau_entropy_alpha": args.tau_entropy_alpha,
        },
        "network": {
            "hidden_dims": hidden_dims,
            "use_layer_norm": True,
            "state_dim": state_dim,
            "num_q_networks": 2,
            "redq_target_subset": 2,
            "visual_feature_dim": visual_feature_dim,
            "proprio_dim": proprio_dim,
            "base_proprio_dim": proprio_dim,
            "chunked_action_dim": chunked_action_dim,
            # Critic-input options this trainer does not offer, at their fixed values.
            "visual_input_layernorm": False,
            "action_contrast_weight": 0.0,
            "action_contrast_num_negatives": 8,
            "action_contrast_sigma": 0.5,
            "action_contrast_temperature": 0.1,
            "action_input_transform": "none",
            # Observation-conditioned learned action-scaling (FiLM) head.
            # Deploy/scorer must apply the same head to z-scored candidate actions
            # before Q when action_film_head is True. step_dim = per-step action
            # dim; state_dim above is the FiLM head's input.
            "action_film_head": args.action_film_head,
            "action_film_hidden": args.action_film_hidden,
            "action_film_arm_dims": args.action_film_arm_dims,
            "action_film_step_dim": action_dim,
            "critic_extra_state": "none",
            "critic_extra_state_dim": 0,
            "critic_extra_state_mask_included": False,
            "critic_extra_state_seed": args.seed,
            "critic_extra_state_availability": {},
        },
        "training": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "critic_lr_schedule": args.critic_lr_schedule,
            "critic_lr_warmup_steps": args.critic_lr_warmup_steps,
            "critic_lr_min_frac": args.critic_lr_min_frac,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "training_steps": args.training_steps,
            "checkpoint_freq": args.checkpoint_freq,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "num_workers_per_loader": num_workers_per_loader,
            "peak_num_workers": peak_num_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
            "multiprocessing_context": args.multiprocessing_context,
            "video_backend": args.video_backend,
            "async_buffer_refresh": args.async_buffer_refresh,
            "enable_tf32": args.enable_tf32,
            "cudnn_benchmark": args.cudnn_benchmark,
            "encoder_autocast_bf16": args.encoder_autocast_bf16,
            "channels_last": args.channels_last,
            "uint8_native_images": args.uint8_native_images,
            "fast_reader": fast_reader_enabled_by_env(),
            "precompute_embeddings": use_embedding_cache,
            "embedding_cache_memory_gb": replay_buffer.memory_gb,
            "embedding_cache_size": replay_buffer.size if use_embedding_cache else None,
            "embedding_cache_visual_views": replay_buffer.num_visual_views
            if use_embedding_cache
            else None,
            "buffer_capacity_gb": args.buffer_capacity_gb,
            "buffer_refresh_rate": args.buffer_refresh_rate,
            "resume_checkpoint_freq": args.resume_checkpoint_freq,
            "auto_resume": auto_resume_enabled,
            "normalize_input": True,
            "normalization_stats_input": None,
            "normalization_stats_sha256": None,
            "proprio_dropout": args.proprio_dropout,
            "inference_proprio_dropout": 0.0,
            "value_training_mode": "iql",
            "q_heads_trained": True,
            # Q/V heads have no dropout; scorers rebuild the MLP layout from this key.
            "qv_dropout": None,
        },
        "eval": {
            "holdout_pct": args.holdout_pct,
            "external_eval": use_external_eval,
            "eval_repo_ids": eval_repo_ids,
            "eval_freq": args.eval_freq,
            "proprio_dropout": 0.0,
            "n_train_episodes": len(train_ep_set),
            "n_holdout_episodes": len(holdout_ep_set),
        },
    }
    run_logger = RunLogger(args.use_wandb)
    # Local auto-resume restores the training state; a resumed run logs to a new W&B run.
    run_logger.init(
        project=args.wandb_project,
        name=run_name,
        notes=auto_notes,
        config=wandb_config,
    )
    run_logger.log(
        {
            "timing/cache_build_s": cache_elapsed,
            "data/buffer_memory_gb": replay_buffer.memory_gb,
            "data/buffer_capacity": replay_buffer.capacity,
            "data/buffer_fill_pct": replay_buffer.fill_pct * 100,
        },
        step=0,
    )
    print(f"W&B: {'enabled' if args.use_wandb else 'disabled'}")
    print(f"Run name: {run_name}")
    print()
    if args.use_wandb:
        # Fail fast on the 128-char W&B artifact-name cap with the longest
        # checkpoint suffix this run will produce, instead of dying at the
        # first checkpoint upload deep into training.
        vision_idql_artifact_name(run_name, f"step_{args.training_steps}")
    # ---- 15. Restore local resume state, if present ----
    start_step = 1
    last_resume_checkpoint_step = 0
    resume_state = resume_manager.load_training_state()
    if resume_state is not None:
        resume_policy_state = resume_state["policy_state_dict"]
        validate_iql_resume_topology(resume_policy_state)
        model.load_state_dict(resume_policy_state)
        optimizer_state = resume_state["optimizer_state_dicts"]["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
        last_resume_checkpoint_step = int(resume_state["last_checkpoint_step"])
        start_step = last_resume_checkpoint_step + 1
        print(
            f"  Auto-resume: restored local training state from step {last_resume_checkpoint_step}"
        )
    if start_step > args.training_steps:
        resume_manager.mark_completed()
        print(
            f"  Auto-resume: local state is already at step {last_resume_checkpoint_step}; "
            "marking completed."
        )
        return

    if resume_state is None:
        print("  Auto-resume: no prior local training state found")

    def save_resume_checkpoint(current_step: int, reason: str) -> None:
        nonlocal last_resume_checkpoint_step
        if not auto_resume_enabled:
            return
        if current_step <= last_resume_checkpoint_step:
            return
        resume_manager.save_training_state(
            policy_state_dict=model.state_dict(),
            optimizer_state_dicts={"optimizer": optimizer.state_dict()},
            step=current_step,
            best_success_rate=0.0,
        )
        last_resume_checkpoint_step = current_step
        print(f"  Auto-resume: saved local state at step {current_step} ({reason})")

    # Annotated eval videos index the loaded (possibly episode-subset) holdout
    # datasets; the prebuilt-cache path resolves the selection only when needed.
    video_episodes_by_repo = {
        repo_id: selected_episodes.get(repo_id) for repo_id in holdout_repo_ids
    }
    if (
        prebuilt_cache_fast_init
        and args.eval_video_freq > 0
        and any(repo_id in dataset_selectors for repo_id in holdout_repo_ids)
    ):
        video_episodes_by_repo = resolve_selected_episodes(
            holdout_repo_ids, root, dataset_revisions, dataset_selectors
        )

    # ---- 16. Training loop ----
    if refresh_thread is not None:
        refresh_thread.start()
    print("Starting training...")
    print("=" * 60)

    training_start = time.perf_counter()
    loop_start = time.perf_counter()
    loss_accum = defaultdict(float)
    accum_count = 0
    refresh_samples_since_log = 0

    # MULLIGAN_REAL_TIMER_CUDA_SYNC=1 makes phase timers device-synced so GPU time is
    # attributed to the phase that queued it (profiling only; adds per-phase syncs).
    timer = TrainingTimer(cuda_sync=os.environ.get("MULLIGAN_REAL_TIMER_CUDA_SYNC", "") == "1")
    cumulative_timings = defaultdict(float)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    B = args.batch_size

    def _vision_idql_metadata(step: int) -> dict:
        """``metadata.json`` of a Vision-IDQL checkpoint (read by the deploy loader)."""
        metadata = {
            "policy_type": "vision_idql",
            "dp_artifact": encoder_source,
            "num_action_samples": args.num_action_samples,
            "state_dim": state_dim,
            "action_dim": chunked_action_dim,
            "prediction_horizon": prediction_horizon,
            "n_action_steps": k,
            "td_horizon_steps": td_k,
            "td_horizon_chunks": td_k // k,
            "td_lambda_curriculum": td_lambda_enabled,
            "td_lambda_initial": args.td_lambda_initial if td_lambda_enabled else None,
            "td_lambda_hold_steps": (args.td_lambda_hold_steps if td_lambda_enabled else None),
            "td_lambda_cosine_end_step": (
                args.td_lambda_cosine_end_step if td_lambda_enabled else None
            ),
            "clean_target_features": False,
            "target_view_sampling": target_view_sampling,
            "target_view_samples": target_view_samples,
            "hidden_dims": hidden_dims,
            "use_layer_norm": True,
            "value_training_mode": "iql",
            "q_heads_trained": True,
            "candidate_action_reranking_valid": True,
            "iql_recipe": getattr(args, "iql_recipe", None),
            # DIVL (distributional V) reconstruction fields read by the deploy loader.
            "distributional": True,
            "num_atoms": args.num_atoms,
            "v_min": divl_v_min,
            "v_max": divl_v_max,
            "hl_gauss_sigma_ratio": args.hl_gauss_sigma_ratio,
            "tau_base": args.tau_base,
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "tau_entropy_alpha": args.tau_entropy_alpha,
            "camera_keys": camera_keys,
            "iql_camera_key_order": camera_keys,
            "dp_camera_key_order": encoder_camera_key_order,
            "camera_crop_boxes": dict(dp_camera_crops),
            "crop_reference_hw": (
                list(crop_reference_hw) if crop_reference_hw is not None else None
            ),
            "action_target": action_target,
            "cartesian_action_frame": cartesian_action_frame,
            # Critic action representation: "relative" critics score the UMI
            # proprio-anchored physical relative-pose chunk (10D/step) and can only
            # rerank an action_mode=relative DP; "absolute" is the raw
            # dataset-action chunk. vision_idql_policy reads this to enforce
            # the DP<->critic pairing and pick the candidate representation.
            "action_mode": action_mode,
            "encoder_frozen": True,
            # Deploy/eval consumers must reproduce the training-time encoder
            # numerics/layout (fp32 vs bf16-autocast, NCHW vs channels_last).
            "encoder_autocast_bf16": args.encoder_autocast_bf16,
            "channels_last": args.channels_last,
            # Provenance only: the buffer/e2e 224^2 pixel format is unchanged,
            # so deploy/eval consumers need no runtime change for these flags.
            "uint8_native_images": args.uint8_native_images,
            "fast_reader": fast_reader_enabled_by_env(),
            "precompute_embeddings": use_embedding_cache,
            "embedding_cache_size": replay_buffer.size if use_embedding_cache else None,
            "embedding_cache_visual_views": replay_buffer.num_visual_views
            if use_embedding_cache
            else None,
            "embedding_cache_memory_gb": replay_buffer.memory_gb if use_embedding_cache else None,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "step": step,
            "repo_ids": repo_ids,
            "eval_repo_ids": eval_repo_ids,
            "dataset_revisions": wandb_config["data"]["dataset_revisions"],
            "dataset_episodes": wandb_config["data"]["dataset_episodes"],
            "normalize_input": True,
            # Critic-input options the release trainer does not offer keep their fixed
            # values (the deploy loader reads these keys with the same defaults).
            "visual_input_layernorm": False,
            "action_contrast_weight": 0.0,
            "action_contrast_num_negatives": 8,
            "action_contrast_sigma": 0.5,
            "action_contrast_temperature": 0.1,
            "action_input_transform": "none",
            # Observation-conditioned learned action-scaling (FiLM) head.
            # DEPLOY/scorer applies the same head to z-scored candidate actions
            # before Q when action_film_head is True (vision_idql_policy.py
            # loads it from this metadata and fails loud on a missing state dict).
            "action_film_head": args.action_film_head,
            "action_film_hidden": args.action_film_hidden,
            "action_film_arm_dims": args.action_film_arm_dims,
            "action_film_step_dim": action_dim,
            "critic_extra_state": "none",
            "critic_extra_state_dim": 0,
            "critic_proprio_dim": proprio_dim,
            "base_proprio_dim": proprio_dim,
            "critic_extra_state_mask_included": False,
            "critic_extra_state_seed": args.seed,
            "critic_extra_state_availability": {},
            "inference_proprio_dropout": 0.0,
            "proprio_mean": proprio_mean.cpu().tolist(),
            "proprio_std": proprio_std.cpu().tolist(),
            "action_mean": action_mean.cpu().tolist(),
            "action_std": action_std.cpu().tolist(),
            "training_config": wandb_config,
        }
        return metadata

    def _save_checkpoint(step: int, suffix: str) -> Path:
        """Write ``checkpoints/<suffix>/{iql_checkpoint.pt,metadata.json}``."""
        m = model
        ckpt_data = {
            "checkpoint_format_version": 2,
            "q1_state_dict": m.q1.state_dict(),
            "q2_state_dict": m.q2.state_dict(),
            "q1_target_state_dict": m.q1_target.state_dict(),
            "q2_target_state_dict": m.q2_target.state_dict(),
            # The FiLM head is a VisionIQL submodule but the checkpoint
            # saves per-component state_dicts (not the whole-module dict), so it
            # does NOT ride automatically — save it explicitly, gated so non-FiLM
            # checkpoints are unchanged (absent key => scorer reads identity).
            **(
                {
                    "film_head_state_dict": m.film_head.state_dict(),
                    "film_head_target_state_dict": m.film_head_target.state_dict(),
                }
                if m.action_film_head
                else {}
            ),
            "v_state_dict": m.v_net.state_dict(),
            "encoder_state_dict": _normalize_compiled_state_dict_keys(m.encoder.state_dict()),
            "encoder_target_state_dict": _normalize_compiled_state_dict_keys(
                m.encoder_target.state_dict()
            ),
            "config": wandb_config,
            "step": step,
            "normalization": {
                "proprio_mean": proprio_mean.cpu(),
                "proprio_std": proprio_std.cpu(),
                "action_mean": action_mean.cpu(),
                "action_std": action_std.cpu(),
            },
            # Deploy/probe consumers must apply this to raw [0,1] frames before
            # the IQL encoder.
            "image_normalization": {
                "mean": image_norm_mean.cpu(),
                "std": image_norm_std.cpu(),
            },
        }
        package_dir = ckpt_dir / suffix
        package_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_data, package_dir / "iql_checkpoint.pt")
        metadata = _vision_idql_metadata(step)
        (package_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if args.use_wandb:
            artifact_name = vision_idql_artifact_name(run_logger.run_name, suffix)
            run_logger.log_artifact_dir(
                package_dir, name=artifact_name, artifact_type="policy", metadata=metadata
            )
            print(f"  Uploaded vision-idql artifact: {artifact_name}")
        return package_dir

    logged_start_augmented_batch = False

    def _maybe_log_start_augmented_batch(curr_images: torch.Tensor, step: int) -> None:
        nonlocal logged_start_augmented_batch
        if logged_start_augmented_batch or not args.use_wandb:
            return
        grid = create_augmented_image_grid(curr_images, num_samples=8)
        if grid is None:
            return
        run_logger.log({"train/augmented_images": run_logger.image(grid)}, step=step)
        logged_start_augmented_batch = True

    step_ctx = IQLStepContext(
        args=args,
        timer=timer,
        device=device,
        camera_keys=camera_keys,
        use_embedding_cache=use_embedding_cache,
        augment_fn=augment_fn,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        action_mean=action_mean,
        action_std=action_std,
        discount_powers=discount_powers_gpu,
        independent_target_samples=independent_target_samples,
        log_augmented_batch=_maybe_log_start_augmented_batch,
    )

    for step in range(start_step, args.training_steps + 1):
        # -- Q/V LR schedule (constant path leaves the optimizer untouched) --
        if args.critic_lr_schedule != "constant":
            critic_lr = critic_lr_at_step(
                step,
                base_lr=args.lr,
                schedule=args.critic_lr_schedule,
                warmup_steps=args.critic_lr_warmup_steps,
                total_steps=args.training_steps,
                min_frac=args.critic_lr_min_frac,
            )
            for group in optimizer.param_groups:
                if group.get("name") in ("q", "v"):
                    group["lr"] = critic_lr

        current_td_lambda = (
            td_lambda_at_step(
                step,
                initial_lambda=args.td_lambda_initial,
                hold_steps=args.td_lambda_hold_steps,
                cosine_end_step=args.td_lambda_cosine_end_step,
            )
            if td_lambda_enabled
            else None
        )
        td_lambda_active = td_lambda_enabled and current_td_lambda > 0.0
        sampling_td_horizon = td_k if td_lambda_active else k

        # -- Sample batch from replay buffer --
        with timer("sampling"):
            if use_embedding_cache:
                batch = replay_buffer.sample(
                    B,
                    action_horizon=k,
                    td_horizon=sampling_td_horizon,
                    include_chunk_bootstraps=td_lambda_active,
                    independent_target_samples=independent_target_samples,
                )
            else:
                batch = replay_buffer.sample(B, image_format="float32")

        # Declared up front so the (branch-populated) transfer block below can be
        # packed into an IQLStepTensors without a NameError on the unused arms.
        cam_imgs_raw: dict[str, torch.Tensor] = {}
        curr_visual = next_visual = None
        curr_visual_target = next_visual_target = None
        td_lambda_visual = td_lambda_visual_target = None
        td_lambda_proprio = td_lambda_horizons_gpu = td_lambda_horizon_valid_gpu = None

        with timer("img_transfer"):
            if use_embedding_cache:
                curr_visual = batch["visual_curr"].to(device, non_blocking=True)
                next_visual = batch["visual_next"].to(device, non_blocking=True)
                if independent_target_samples:
                    curr_visual_target = batch["visual_curr_target"].to(device, non_blocking=True)
                    next_visual_target = batch["visual_next_target"].to(device, non_blocking=True)
                if td_lambda_active:
                    td_lambda_visual = batch["visual_bootstrap"].to(device, non_blocking=True)
                    td_lambda_proprio = batch["observation.bootstrap_state"].to(
                        device, non_blocking=True
                    )
                    td_lambda_horizons_gpu = batch["td_horizons"].to(device, non_blocking=True)
                    td_lambda_horizon_valid_gpu = batch["td_horizon_valid"].to(
                        device, non_blocking=True
                    )
                    if independent_target_samples:
                        td_lambda_visual_target = batch["visual_bootstrap_target"].to(
                            device, non_blocking=True
                        )
            else:
                cam_imgs_raw = {
                    cam: batch[cam].to(device, non_blocking=True) for cam in camera_keys
                }
            actions_gpu = batch["action"].to(device, non_blocking=True)
            rewards_gpu = batch["reward"].to(device, non_blocking=True)
            if use_embedding_cache:
                if action_mode == "relative":
                    # Flat-cache batches carry the raw (B, H, 7) command-pose chunk;
                    # relativize against the anchor step's proprio (observation.state
                    # is the [current, successor] stack — index 0 is the anchor).
                    actions_gpu = _relativize_action_chunk(
                        actions_gpu,
                        batch["observation.state"].to(device, non_blocking=True),
                    )
                interventions_gpu = batch["intervention"].to(device, non_blocking=True)
                if args.intervention_negative_reward is not None:
                    rewards_gpu = (
                        rewards_gpu
                        + args.intervention_negative_reward
                        * interventions_gpu.to(rewards_gpu.dtype)
                    )
                if args.reward_shift != 0:
                    rewards_gpu = rewards_gpu + args.reward_shift
            dones_gpu = batch["done"].to(device, non_blocking=True)
            valid_steps_gpu = None
            if use_embedding_cache and sampling_td_horizon > k and not td_lambda_active:
                # Multi-step targets: a timeout anchor near its episode boundary
                # carries valid_steps < td_k. The loss masks its padded reward
                # positions and bootstraps with a per-sample gamma**valid_steps
                # from the truncated successor. At td_k == k this is left None
                # by construction (eligible non-terminal anchors always carry a
                # complete action chunk), keeping the fixed-horizon path
                # bit-identical.
                valid_steps_gpu = batch["valid_steps"].to(device, non_blocking=True)
            proprio = batch["observation.state"].to(device, non_blocking=True)

        # Refresh replay buffer while GPU transfers are in flight
        with timer("buffer_refresh"):
            if use_embedding_cache:
                pass
            elif args.async_buffer_refresh:
                if refresh_producer_error:
                    raise RuntimeError(
                        "Async buffer-refresh producer thread died; failing loudly "
                        "instead of training without rotation."
                    ) from refresh_producer_error[0]
                # Non-blocking poll: rotate in a decoded batch only when the
                # producer has one ready; otherwise rotation slips this step.
                try:
                    ready_batch = refresh_queue.get_nowait()
                except queue.Empty:
                    ready_batch = None
                if ready_batch is not None:
                    if uint8_native:
                        # Producer pinned RAW uint8 native frames; do the GPU
                        # crop+resize+float+pack here (main thread owns all CUDA).
                        ready_batch = _gpu_pack_native_images(ready_batch)
                    refresh_samples_since_log += replay_buffer.refresh(ready_batch)
            else:
                try:
                    refresh_batch = next(buffer_refresh_iter)
                except StopIteration:
                    buffer_refresh_iter = iter(buffer_refresh_loader)
                    refresh_batch = next(buffer_refresh_iter)
                shaped_refresh = _shape_refresh_batch(refresh_batch)
                if uint8_native:
                    # Sync path: workers returned RAW uint8 native frames; run the
                    # GPU crop+resize+float+pack before writing into the buffer.
                    shaped_refresh = _gpu_pack_native_images(shaped_refresh)
                refresh_samples_since_log += replay_buffer.refresh(shaped_refresh)

        step_tensors = IQLStepTensors(
            proprio=proprio,
            actions=actions_gpu,
            rewards=rewards_gpu,
            dones=dones_gpu,
            valid_steps=valid_steps_gpu,
            cam_imgs_raw=cam_imgs_raw,
            curr_visual=curr_visual,
            next_visual=next_visual,
            curr_visual_target=curr_visual_target,
            next_visual_target=next_visual_target,
            td_lambda_visual=td_lambda_visual,
            td_lambda_visual_target=td_lambda_visual_target,
            td_lambda_proprio=td_lambda_proprio,
            td_lambda_horizons=td_lambda_horizons_gpu,
            td_lambda_horizon_valid=td_lambda_horizon_valid_gpu,
        )
        losses = iql_train_step(
            model,
            step_tensors,
            optimizer,
            step_ctx,
            step=step,
            td_lambda_active=td_lambda_active,
            current_td_lambda=current_td_lambda,
            sampling_td_horizon=sampling_td_horizon,
        )

        # ---- Accumulate losses for logging ----
        loss_accum["train/value_loss"] += losses["value_loss"].item()
        loss_accum["train/critic_loss"] += losses["critic_loss"].item()
        loss_accum["train/q1_mean"] += losses["q1_mean"].item()
        loss_accum["train/q2_mean"] += losses["q2_mean"].item()
        loss_accum["train/v_mean"] += losses["v_mean"].item()
        loss_accum["train/advantage_mean"] += losses["advantage_mean"].item()
        loss_accum["train/td_target_mean"] += losses["td_target_mean"].item()
        for loss_key, loss_value in losses.items():
            if loss_key.startswith("td_") and loss_key != "td_target_mean":
                if torch.isfinite(loss_value):
                    loss_accum[f"train/{loss_key}"] += loss_value.item()
        if td_lambda_enabled and not td_lambda_active:
            loss_accum["train/td_lambda"] += 0.0
        accum_count += 1

        # ---- Logging ----
        if step % args.log_freq == 0:
            now = time.perf_counter()
            steps_since_loop_start = step - start_step + 1
            avg_step_time = (now - loop_start) / max(steps_since_loop_start, 1)
            remaining = avg_step_time * (args.training_steps - step)
            eta_h, eta_rem = divmod(int(remaining), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m}m{eta_s:02d}s"

            log_dict = {k_: v / accum_count for k_, v in loss_accum.items()}
            log_dict["timing/step_time_ms"] = avg_step_time * 1000
            log_dict["timing/eta_hours"] = remaining / 3600
            log_dict["data/buffer_fill_pct"] = replay_buffer.fill_pct * 100
            log_dict["data/buffer_size"] = replay_buffer.size
            log_dict["data/normalization_sample_frac"] = normalization_sample_frac
            log_dict["data/refresh_samples_per_step"] = refresh_samples_since_log / max(
                accum_count, 1
            )
            refresh_samples_since_log = 0

            timing_stats = timer.get_stats()
            for phase, avg_s in timing_stats.items():
                log_dict[f"timing/{phase}_ms"] = avg_s * 1000

            for phase in timer.timings:
                cumulative_timings[phase] += timer.timings[phase]
            total_cumulative = sum(cumulative_timings.values())
            if total_cumulative > 0:
                for phase, total_s in cumulative_timings.items():
                    log_dict[f"timing_share/{phase}"] = total_s / total_cumulative

            if args.critic_lr_schedule != "constant":
                log_dict["train/critic_lr"] = next(
                    group["lr"]
                    for group in optimizer.param_groups
                    if group.get("name") in ("q", "v")
                )

            v_loss_avg = log_dict["train/value_loss"]
            c_loss_avg = log_dict["train/critic_loss"]
            q_avg = log_dict["train/q1_mean"]
            v_avg = log_dict["train/v_mean"]
            print(
                f"step {step}/{args.training_steps}  "
                f"v_loss={v_loss_avg:.4f}  c_loss={c_loss_avg:.4f}  "
                f"Q={q_avg:.3f}  V={v_avg:.3f}  "
                f"ETA={eta_str}"
            )
            timer.print_stats(prefix="  ")
            print()

            run_logger.log(log_dict, step=step)
            loss_accum.clear()
            accum_count = 0
            timer.reset()

        # ---- Evaluation ----
        eval_due = args.eval_freq > 0 and step % args.eval_freq == 0
        if eval_due:
            print(f"\n  [EVAL] Step {step}...")
            eval_start = time.perf_counter()

            model.encoder.eval()
            if "states" not in holdout_data:
                if holdout_dl is None:
                    raise RuntimeError(
                        "Holdout states are absent but no holdout DataLoader is available"
                    )
                holdout_data["states"] = encode_holdout_images(
                    holdout_dl=holdout_dl,
                    encoder=model.encoder,
                    camera_keys=camera_keys,
                    separate_encoders=separate_encoders,
                    device=device,
                    proprio_mean=proprio_mean,
                    proprio_std=proprio_std,
                    image_norm_mean=image_norm_mean,
                    image_norm_std=image_norm_std,
                    encoder_autocast_bf16=args.encoder_autocast_bf16,
                    channels_last=args.channels_last,
                )
            else:
                holdout_data["states"] = holdout_data["states"].to(device)
            should_render_videos = args.eval_video_freq > 0 and step % args.eval_video_freq == 0
            eval_metrics = evaluate_on_holdout(
                q_networks=[model.q1, model.q2],
                v_network=model.v_net,
                holdout_data=holdout_data,
                device=device,
                step=step,
                output_dir=output_dir,
                repo_ids=holdout_repo_ids if should_render_videos else None,
                camera_keys=camera_keys if should_render_videos else None,
                max_videos=args.max_eval_videos,
                chunk_size=k,
                root=root,
                gamma=args.gamma,
                episodes_by_repo=video_episodes_by_repo,
                revisions=dataset_revisions,
            )

            eval_elapsed = time.perf_counter() - eval_start
            v_gap = eval_metrics["eval/v_s0_gap"]
            v_succ = eval_metrics["eval/v_s0_success_mean"]
            v_fail = eval_metrics["eval/v_s0_failure_mean"]
            print(
                f"  [EVAL] V(s0) gap: {v_gap:.3f} "
                f"(success={v_succ:.3f}, failure={v_fail:.3f})  "
                f"({eval_elapsed:.1f}s)"
            )
            print()

            run_logger.log(eval_metrics, step=step)

        if (
            args.checkpoint_freq > 0
            and step % args.checkpoint_freq == 0
            and step < args.training_steps
        ):
            ckpt_path = _save_checkpoint(step, f"step_{step}")
            print(f"Saved checkpoint to {ckpt_path}")

        if should_save_resume_checkpoint(
            step=step,
            resume_checkpoint_freq=args.resume_checkpoint_freq,
        ):
            save_resume_checkpoint(step, "step_complete")

    # Stop the async refresh producer before teardown so it cannot touch the
    # DataLoader/CUDA context mid-destruction.
    if refresh_stop is not None:
        refresh_stop.set()
        if refresh_thread is not None:
            refresh_thread.join(timeout=10.0)

    # ---- 17. Save final checkpoint ----
    print("\nSaving final checkpoint...")
    final_ckpt_path = _save_checkpoint(args.training_steps, "final")
    print(f"Saved to {final_ckpt_path}")
    save_resume_checkpoint(args.training_steps, "final")
    resume_manager.mark_completed()
    run_logger.finish()

    total_time = time.perf_counter() - training_start
    print(f"\nDone! Total time: {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
