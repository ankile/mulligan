#!/usr/bin/env python3
# ruff: noqa: E402
"""Train a Diffusion Policy on real-robot LeRobot datasets.

Real-robot data has no simulator, so there is no rollout evaluation during training: the
trainer logs a validation loss on held-out repos and saves LeRobot checkpoints
(``config.json`` + ``model.safetensors`` + pre/post-processors) under ``--output-dir``.

Supports:
- camera selection (``--task`` default roles, ``--camera-keys`` or a suffix filter)
- multi-dataset training (base teleop + DAgger rounds)
- DAgger filtering (train only on human frames of successful episodes)
- pinned dataset revisions and per-repo episode selectors (``--dataset-revisions``,
  ``--dataset-episodes``; see ``mulligan.real.train.dataset_selectors``)

Usage:
    python -m mulligan.real.train.policy \
        --task marker_d2 \
        --repo-ids mulligan/real-marker-d2-c00-teleop-baseline,mulligan/real-marker-d2-c01-dagger-baseline \
        --eval-repo-ids mulligan/real-marker-d2-c00-teleop-validation \
        --dataset-revisions mulligan/real-marker-d2-c00-teleop-baseline=<sha> ... \
        --output-dir outputs/marker_dp
"""

import argparse
import dataclasses
import json
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from mulligan import apply_runtime_patches

apply_runtime_patches()

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import (
    get_policy_class,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.utils.constants import ACTION, OBS_PREFIX
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.utils import cycle
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import make_grid

from mulligan.data.constants import DataSource
from mulligan.data.fast_lerobot_reader import (
    FAST_READER_ENV,
    enable_fast_reader_on_subdatasets,
    fast_reader_enabled_by_env,
)
from mulligan.data.transforms import (
    clamp_soft_truncated_anchor_ends,
    compute_multidataset_valid_boundaries,
    compute_relative_pose_pertimestep_stats,
    fill_null_floats_in_lerobot_subdatasets,
    filter_episodes,
    remap_action_to_position_r6_in_subdatasets,
    remove_features_from_lerobot_subdatasets,
    require_multilerobot_subdatasets,
)
from mulligan.real.lifecycle.tasks import find_task_spec_by_task_name, get_task_spec
from mulligan.real.policy.image_preprocess import (
    PolicyImagePreprocessTransform,
    QuantizeUint8PostTransform,
    uint8_to_float01,
)
from mulligan.real.policy.side_crop import (
    build_crop_feature_map,
    install_per_camera_crop,
    merge_default_crops,
    parse_side_crop,
)
from mulligan.real.policy.visual_norm import (
    apply_imagenet_visual_stats,
    assert_image_std_physical,
)
from mulligan.real.robot.cameras import (
    STATION_CAMERA_DEFAULT_CROPS,
    STATION_IMAGE_HW,
    STATION_VIDEO_BACKEND,
    select_camera_feature_keys,
)
from mulligan.real.train.dataset_selectors import selectors_to_json
from mulligan.real.train.hub_data import (
    add_dataset_revision_args,
    dataset_commits,
    dataset_dir,
    dataset_revision,
    load_multi_dataset,
    parse_dataset_pins,
    prepare_datasets,
    resolve_selected_episodes,
)
from mulligan.real.train.run_logging import RunLogger
from mulligan.training.straddled_sampler import (
    InfiniteSubsetRandomSampler,
    SubsetRandomSampler,
)
from mulligan.training.timer import TrainingTimer
from mulligan.utils.seeding import set_seed


TRAINING_STATE_FILENAME = "training_state.pt"
# Data provenance (repos, pinned revisions, episode selectors) and the run config.
TRAIN_METADATA_FILENAME = "train_metadata.json"
# LeRobot policy type this trainer builds.
POLICY_TYPE = "diffusion"


def save_training_state(
    checkpoint_path: Path,
    *,
    step: int,
    optimizer,
    lr_scheduler,
    wandb_run_id: str | None,
    precision_regime: dict | None = None,
    decoded_frame_cache: bool | None = None,
    decoded_frame_cache_mixed: bool = False,
) -> None:
    """Persist full resumable training state next to the weights-only save_pretrained.

    save_pretrained stays weights-only (for deployment); this writes optimizer/scheduler/
    RNG/step + the W&B run id so a preempted job can resume on the same LR curve rather
    than restarting from scratch.
    """
    state = {
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        },
        "wandb_run_id": wandb_run_id or None,
        # Training-forward precision regime (--channels-last):
        # checked on resume so a relaunch cannot silently switch regime mid-run
        # (the checkpoint metadata records only the final flags).
        "precision_regime": precision_regime,
        # Effective image-path state (--decoded-frame-cache incl. budget
        # fallbacks). Accepted perf-knob class (<=1/255 pre-aug quantization), so
        # a mismatch on resume WARNs + marks the run mixed instead of refusing —
        # the mixed marker is sticky across resumes and lands in the checkpoint
        # metadata so a mixed-regime run cannot masquerade as a pure one.
        "decoded_frame_cache": decoded_frame_cache,
        "decoded_frame_cache_mixed": decoded_frame_cache_mixed,
    }
    torch.save(state, checkpoint_path / TRAINING_STATE_FILENAME)


def find_latest_resumable_checkpoint(checkpoint_root: Path) -> tuple[Path, int] | None:
    """Return (checkpoint_dir, step) for the highest-step checkpoint_<N> dir that has a
    training_state.pt, or None. The 'final' dir is intentionally ignored — a run that
    reached 'final' is complete and should not be auto-resumed."""
    if not checkpoint_root.is_dir():
        return None
    best: tuple[Path, int] | None = None
    for child in checkpoint_root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint_"):
            continue
        suffix = child.name[len("checkpoint_") :]
        if not suffix.isdigit():
            continue
        if not (child / TRAINING_STATE_FILENAME).is_file():
            continue
        step = int(suffix)
        if best is None or step > best[1]:
            best = (child, step)
    return best


def restore_rng_state(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])


def _resolve_task_spec(task: str):
    """Resolve a --task string to a RealTaskSpec, trying the lifecycle key first
    (e.g. ``"marker_d2"``) then the collection task name (e.g. ``"square_d2"``).

    Returns ``None`` for an unregistered task so a non-real (sim) --task can still be
    passed for other purposes without breaking camera selection; the only effect of a
    resolved spec here is supplying default consumed_camera_roles.
    """
    try:
        return get_task_spec(task)
    except KeyError:
        return find_task_spec_by_task_name(task)


def _apply_real_training_recipe(args) -> None:
    """Resolve per-task + per-station training defaults into ``args`` IN PLACE.

    Precedence: explicit CLI flag > ``--task`` recipe (+ robot-station capture defaults)
    > built-in fallback. This is what lets a launch wrapper omit the whole DP recipe and
    pass only ``--task`` plus the data/deviation it is actually testing. Runs once at the
    top of :func:`train` so every downstream consumer sees concrete, resolved values.

    Only fields the per-task recipe sets are routed through here (their argparse default
    is a ``None`` sentinel). Fields with one value for every task (``--vision-backbone``
    resnet18, ``--visual-normalization`` imagenet) are plain argparse defaults. The per-task DP
    recipe is :class:`mulligan.real.lifecycle.tasks.DPTrainingRecipe`; the per-station capture
    constants are ``camera_utils.STATION_IMAGE_HW`` / ``STATION_VIDEO_BACKEND``.

    chunk_size / n_action_steps / down_dims / drop_n_last_frames / video_backend keep their
    ``None`` sentinel for non-task callers: the existing per-policy downstream resolution
    (DP horizon=16, exec=6, down=(512,1024,2048), config drop default, backend auto-detect)
    handles those unchanged for a bare (no ``--task``) invocation.
    """
    spec = _resolve_task_spec(args.task) if args.task else None
    recipe = spec.training if spec is not None else None

    if recipe is not None:
        # Per-robot-station capture defaults (resize target + video decode backend).
        if args.image_height is None:
            args.image_height = STATION_IMAGE_HW[0]
        if args.image_width is None:
            args.image_width = STATION_IMAGE_HW[1]
        if args.video_backend is None:
            args.video_backend = STATION_VIDEO_BACKEND
        # Per-task DP learning recipe.
        if args.chunk_size is None:
            args.chunk_size = recipe.chunk_size
        if args.n_action_steps is None:
            args.n_action_steps = recipe.n_action_steps
        if args.down_dims is None:
            args.down_dims = ",".join(str(d) for d in recipe.down_dims)
        if args.batch_size is None:
            args.batch_size = recipe.batch_size
        if args.training_steps is None:
            args.training_steps = recipe.training_steps
        if args.save_freq is None:
            args.save_freq = recipe.save_freq
        if args.eval_freq is None:
            args.eval_freq = recipe.eval_freq
        if args.drop_n_last_frames is None:
            args.drop_n_last_frames = str(recipe.drop_n_last_frames)
        print(
            f"--task {args.task!r} resolved real-DP recipe (spec {spec.name!r}): "
            f"chunk_size={args.chunk_size} n_action_steps={args.n_action_steps} "
            f"down_dims={args.down_dims} batch_size={args.batch_size} "
            f"training_steps={args.training_steps} save_freq={args.save_freq} "
            f"eval_freq={args.eval_freq} drop_n_last_frames={args.drop_n_last_frames} "
            f"image={args.image_height}x{args.image_width} video_backend={args.video_backend}"
        )

    # Built-in fallbacks for the args whose argparse default was made a None sentinel, so a
    # non-task caller (or any field a task recipe leaves unset) gets the
    # generic defaults.
    if args.image_height is None:
        args.image_height = 240
    if args.image_width is None:
        args.image_width = 320
    if args.batch_size is None:
        args.batch_size = 8
    if args.training_steps is None:
        args.training_steps = 100_000
    if args.save_freq is None:
        args.save_freq = 25_000
    if args.eval_freq is None:
        args.eval_freq = 2_500
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.eval_batch_size <= 0:
        raise ValueError(f"--eval-batch-size must be positive, got {args.eval_batch_size}")


def resolve_delta_timestamps(cfg, ds_meta):
    """Resolve delta_timestamps from policy config's delta_indices properties.

    Inlined from lerobot.datasets.factory to avoid deep import chain
    that pulls in hardware dependencies (serial, etc.).
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
    return delta_timestamps if delta_timestamps else None


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the real DP trainer (no side effects)."""
    parser = argparse.ArgumentParser(
        prog="python -m mulligan.real.train.policy",
        description="Train policies on real robot teleoperation data",
    )

    # Dataset
    parser.add_argument(
        "--repo-ids",
        type=str,
        required=True,
        help="Comma-separated dataset repo IDs (e.g., 'mulligan/<teleop>,mulligan/<dagger>')",
    )
    parser.add_argument(
        "--eval-repo-ids",
        type=str,
        default="",
        help=(
            "Optional comma-separated dataset repo IDs used only for validation loss. "
            "When set, the main --repo-ids are used entirely for training and --val-pct "
            "is ignored. Eval repos must be repo-id-disjoint from training repos."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Optional local LeRobot root containing repo-id subdirectories. "
            "Used for cached or re-encoded dataset copies."
        ),
    )
    add_dataset_revision_args(parser)
    parser.add_argument(
        "--no-dataset-sync",
        action="store_true",
        help=(
            "Read the local dataset copies under --dataset-root as they are: no Hub sync (which "
            "prunes local parquet shards and re-downloads the pinned revision). For datasets "
            "that are not on the Hub; the run records no dataset commits."
        ),
    )
    parser.add_argument(
        "--filter-dagger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train on DAgger datasets' human-correction frames of successful episodes only "
            "(default: on; --no-filter-dagger trains on every frame)"
        ),
    )

    # Policy
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path of a checkpoint to warm-start from",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Number of action steps to execute per prediction (distinct from "
        "chunk_size). For diffusion, unset resolves to the real-robot protocol "
        "exec horizon (6); --chunk-size controls the prediction horizon.",
    )
    parser.add_argument(
        "--camera-filter",
        type=str,
        default="_left",
        help="Suffix to filter camera names (default: '_left'). "
        "Only cameras matching this suffix are used for training.",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=None,
        help="Explicit comma-separated cameras to CONSUME (role names like "
        "'side_1,wrist_left', or full observation.images.* "
        "keys). Overrides "
        "--camera-filter; the dataset may store MORE cameras (kept for reuse/future "
        "tasks) but training uses only these.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Lifecycle task key (e.g. 'marker_d2') or collection task name (e.g. "
        "'square_d2'). When set AND --camera-keys is not given, the task's "
        "consumed_camera_roles (mulligan.real.lifecycle.tasks) become the default "
        "--camera-keys (e.g. marker_d2 -> 'side_1,wrist_left'). Explicit --camera-keys "
        "always overrides; tasks with no consumed_camera_roles keep the suffix-filter path.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Resize images to this height before training. Unset: the --task station "
        "default (camera_utils.STATION_IMAGE_HW, 224), else 240.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Resize images to this width before training. Unset: the --task station "
        "default (camera_utils.STATION_IMAGE_HW, 224), else 320.",
    )
    parser.add_argument(
        "--side-crop",
        "--camera-crop",
        dest="side_crop",
        action="append",
        default=None,
        metavar="CAM=x0,y0,x1,y1",
        help=(
            "Per-camera static replacement crop applied to the camera frame before resize "
            "(repeatable, one per camera). Half-open box. E.g. "
            "--camera-crop side_1=138,0,580,447. Cropped cameras slice "
            "[y0:y1, x0:x1] then resize to --image-height/--image-width. Boxes are pixels "
            "of the frames as the dataset stores them (640x480 for the released, role-keyed "
            "datasets). Selected cameras without a box get the station config's default "
            "crop (and task overrides). The box is saved in policy config.json "
            "(camera_crop_boxes) so eval crops identically with no extra eval flags. "
            "--side-crop is an alias."
        ),
    )
    parser.add_argument(
        "--pretrained-backbone-weights",
        type=str,
        default=None,
        help=(
            "Torchvision pretrained weights for the diffusion RGB backbone "
            "(e.g. DEFAULT or IMAGENET1K_V1). Diffusion policy only."
        ),
    )
    parser.add_argument(
        "--vision-backbone",
        type=str,
        default="resnet18",
        help=("Diffusion visual backbone: a torchvision ResNet name (default: resnet18)."),
    )
    parser.add_argument(
        "--visual-normalization",
        choices=("dataset", "imagenet", "identity"),
        default="imagenet",
        help=(
            "Visual normalization of the image inputs. 'imagenet' (default) normalizes every "
            "image stream by the ImageNet mean/std, which the ImageNet-pretrained ResNet "
            "backbone expects; 'dataset' uses the per-dataset LeRobot mean/std; 'identity' "
            "is for encoders that normalize internally."
        ),
    )

    # Training
    parser.add_argument(
        "--training-steps",
        type=int,
        default=None,
        help="Total gradient steps. Unset: the --task recipe (DPTrainingRecipe, 50000), "
        "else 100000.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Unset: the --task recipe (DPTrainingRecipe, 64), else 8.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Checkpoint save interval in steps. Unset: the --task recipe "
        "(DPTrainingRecipe, 25000), else 25000.",
    )
    parser.add_argument(
        "--log-freq", type=int, default=100, help="W&B log interval in steps (default: 100)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override (default: use policy preset)",
    )
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default=None,
        choices=["cosine", "constant", "constant_with_warmup"],
        help="LR schedule, forwarded to DiffusionConfig.scheduler_name "
        "and thence to diffusers' get_scheduler (default: None = keep the policy preset's "
        "'cosine'). Use 'constant' for warm-start continuation runs, where a fresh cosine "
        "would re-raise the LR to its peak.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Diffusion prediction horizon. Unset: the --task recipe, else 16.",
    )
    parser.add_argument(
        "--down-dims",
        type=str,
        default=None,
        help="Diffusion UNet down_dims as comma-separated ints (e.g., '256,512,1024'). "
        "Default: (512,1024,2048) ~266M params. Use '256,512,1024' for ~60M params.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/train_real",
        help="Checkpoint directory (default: ./outputs/train_real)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--model-init-seed",
        type=int,
        default=None,
        help=(
            "Optional fixed seed used only while constructing a fresh policy, then the "
            "training RNG state is restored. Use this for paired ablations with identical "
            "initial weights while --seed still controls training/data-loader randomness. "
            "Ignored when resuming or loading --pretrained-path."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable auto-resume from the latest checkpoint in --output-dir. By default "
        "(resume ON) the trainer detects the latest checkpoint_<N> dir with a "
        "training_state.pt and restores optimizer/scheduler/EMA/RNG/step + the same W&B run "
        "so an interrupted job continues instead of restarting from scratch.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--uint8-native-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ship uint8 (not float32) camera frames from DataLoader workers, finishing "
            "image preprocessing on GPU. DEFAULT-ON (opt out with "
            "--no-uint8-native-images). The full worker crop+resize+aug stack runs "
            "unchanged; post-aug frames are quantized to uint8 in the worker with the "
            "pack_images_uint8 formula and re-floated on GPU (<=1/255 round-trip error). "
            "Requires --num-workers>0. When it is unmet the trainer WARNs loudly and "
            "downgrades to the float path (perf knob; the experiment is identical either way)."
        ),
    )
    parser.add_argument(
        "--decoded-frame-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Decode every served TRAIN frame once at startup (sequential per-file "
            "sweep) into shared policy-res uint8 tensors; DataLoader workers then "
            "serve frames from RAM instead of re-decoding video per sample "
            "(~10 ms/frame random-access AV1 decode, re-paid ~120x per frame over "
            "100k steps). DEFAULT-ON (opt out with "
            "--no-decoded-frame-cache). Cached bytes are the cropped, resized, uint8-quantized "
            "frames (<=1/255 pre-aug quantization; "
            "build + serve paths are self-checked against the stock "
            "decode). Requires the fast reader + per-camera crop boxes + "
            "--image-height; unmet preconditions or an over-budget datamix WARN "
            "loudly and fall back to per-sample decode (pure perf knob). External "
            "eval datasets are cached too when active."
        ),
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Move all 4D conv weights (the ResNet image encoders) to channels_last so "
            "cudnn selects NHWC kernels; activations convert at the first conv and stay "
            "NHWC through the backbone. Layout-only (identity numerics class, same as "
            "the IQL --channels-last flag). Diffusion-only pending validation elsewhere."
        ),
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default=None,
        help="Video decoding backend (default: auto-detect). Options: torchcodec, pyav",
    )
    # Recipe knobs. All default to the production recipe.
    parser.add_argument(
        "--action-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help="Action representation. 'absolute' "
        "(default) is the recorded per-frame base-frame cartesian velocity target. 'relative' "
        "implements UMI's relative trajectory: the canonical 'action' is remapped to "
        "the ABSOLUTE proprio EE-pose trajectory (10D xyz+r6+gripper, sourced from "
        "observation.state), windowed by LeRobot, then re-expressed relative to the "
        "current EE pose (anchor = chunk[n_obs_steps-1]) with UMI 6D rotation. Each "
        "future timestep is normalized with its OWN (T,D) statistics "
        "(temporally_independent_normalization=True). Requires the proprio "
        "cartesian_position + gripper_position columns.",
    )
    parser.add_argument(
        "--drop-n-last-frames",
        type=str,
        default=None,
        help="Override drop_n_last_frames for anchor sampling. 'auto' computes "
        "horizon - n_action_steps - n_obs_steps + 1 for the actual geometry (the LeRobot "
        "config default of 7 is stale for predict-8/exec-6/obs-1); an integer sets it "
        "explicitly; unset keeps the policy-config default.",
    )
    parser.add_argument(
        "--spatial-softmax-num-keypoints",
        type=int,
        default=None,
        help="Diffusion ResNet SpatialSoftmax keypoint count. Unset keeps LeRobot's default of 32.",
    )
    parser.add_argument(
        "--sync-timing",
        action="store_true",
        help="Synchronize CUDA around timing blocks so async kernels are attributed correctly.",
    )

    # Validation
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Steps between val loss evaluations (0 = disable). Unset: the --task recipe "
        "(DPTrainingRecipe, 10000), else 2500.",
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=8,
        help="Number of batches to average for val loss (default: 8)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Validation/eval DataLoader batch size. Default: use --batch-size.",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=0.05,
        help="Fraction of episodes held out for validation (default: 0.05)",
    )

    # W&B
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mulligan",
        help="W&B project name (default: mulligan)",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-notes", type=str, default=None, help="W&B run notes")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


@contextmanager
def _temporary_seed(seed: int | None):
    """Temporarily seed all local RNGs, then restore the caller's stream state."""
    if seed is None:
        yield
        return

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def apply_temporal_dim_handling(batch):
    """Give every image tensor a time dimension: (B, C, H, W) -> (B, 1, C, H, W).

    Diffusion expects (B, T, C, H, W) images; images that already carry T are kept.
    """
    for key in list(batch.keys()):
        if key.startswith("observation.images.") and not key.endswith("_is_pad"):
            if isinstance(batch[key], torch.Tensor):
                if batch[key].ndim not in {4, 5}:
                    raise ValueError(
                        f"{key} for diffusion policy must be 4D or 5D, "
                        f"got shape {tuple(batch[key].shape)}"
                    )
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
    return batch


def flatten_loss_dict(loss_dict: dict) -> dict[str, float]:
    """Flatten a policy loss_dict into scalar-only metrics for logging/accumulation.

    Policies return loss_dict with two kinds of values:
      - scalar (int/float or 0-d tensor): logged directly (e.g. "loss", "l1_loss")
      - list of floats: per-action-dimension losses, expanded into "loss_dim_0", "loss_dim_1", etc.

    Any other type crashes immediately — we never silently skip unknown data.
    """
    flat = {}
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            flat[k] = v.item()
        elif isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                flat[f"{k}_{i}"] = float(x)
        else:
            raise TypeError(
                f"Unexpected type in loss_dict: key={k!r}, type={type(v).__name__}, value={v!r}. "
                f"Expected scalar (int/float/0-d tensor) or list of scalars."
            )
    return flat


def append_split_index(
    global_idx: int,
    frame_to_episode: dict[int, int],
    val_episode_set: set[int],
    train_indices: list[int],
    val_indices: list[int],
) -> None:
    """Append a filtered frame to train/validation lists using a required mapping."""
    ep_i = frame_to_episode[global_idx]
    if ep_i in val_episode_set:
        val_indices.append(global_idx)
    else:
        train_indices.append(global_idx)


def make_episode_val_split(
    num_episodes: int,
    val_pct: float,
    *,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return train and validation episode indices with non-empty sides."""
    if num_episodes < 2:
        raise ValueError(f"Validation split needs at least 2 episodes, got {num_episodes}.")
    if not 0.0 < val_pct < 1.0:
        raise ValueError(f"--val-pct must be in (0, 1) when validation is enabled, got {val_pct}")

    rng = np.random.RandomState(seed)
    episode_order = rng.permutation(num_episodes).tolist()
    num_val = max(1, int(num_episodes * val_pct))
    if num_val >= num_episodes:
        raise ValueError(
            f"--val-pct={val_pct} leaves no training episodes for num_episodes={num_episodes}"
        )
    return sorted(episode_order[num_val:]), sorted(episode_order[:num_val])


# no_grad, NOT inference_mode: the returned tensors feed the training forward,
# and inference tensors crash backward if the normalizer passes them through
# unchanged (e.g. IDENTITY visual normalization).
@torch.no_grad()
def finish_camera_images_on_device(batch, camera_keys, device):
    """Move worker-emitted uint8 camera frames to ``device`` and re-float them.

    Shared ``--uint8-native-images`` image-finishing path, used by the train step
    loop, ``compute_val_loss``, and the debug image grid (with ``device="cpu"``).
    Frames are post-aug uint8 at the policy resolution (crop+resize+aug already ran
    in the worker). ``uint8_to_float01`` fails loudly on non-uint8 input — a float
    frame here means the worker-side quantize did not run.

    Mutates and returns ``batch`` with each camera key replaced by a float32
    [0,1] tensor on ``device``. Leading batch/temporal dims pass through.
    """
    for cam in camera_keys:
        batch[cam] = uint8_to_float01(batch[cam].to(device, non_blocking=True))
    return batch


def compute_val_loss(
    val_dataloader,
    val_dl_iter,
    policy,
    preprocessor,
    policy_cfg,
    device,
    num_batches,
    uint8_native=False,
    camera_keys=None,
):
    """Compute average validation loss over num_batches from the val dataloader.

    Returns (val_metrics_dict, updated_val_dl_iter).
    """
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive for validation, got {num_batches}")

    was_training = policy.training
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        policy.eval()
        total_loss = 0.0
        loss_accum = {}
        count = 0

        for _ in range(num_batches):
            try:
                batch = next(val_dl_iter)
            except StopIteration:
                val_dl_iter = iter(val_dataloader)
                batch = next(val_dl_iter)

            # uint8-native: re-float the worker-emitted uint8 camera frames on GPU
            # before the lerobot preprocessor, matching the training step's image
            # path exactly.
            if uint8_native:
                batch = finish_camera_images_on_device(batch, camera_keys, device)

            batch = preprocessor(batch)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            batch = apply_temporal_dim_handling(batch)

            loss, loss_dict = policy.forward(batch)
            total_loss += loss.item()
            if loss_dict is not None:
                for k, v in flatten_loss_dict(loss_dict).items():
                    loss_accum[k] = loss_accum.get(k, 0.0) + v
            count += 1

        metrics = {"val/loss": total_loss / count}
        for k, v in loss_accum.items():
            metrics[f"val/{k}"] = v / count
        return metrics, val_dl_iter
    finally:
        policy.train(was_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def save_augmented_image_grid(batch, camera_keys, output_dir, step, num_samples=8):
    """Save a grid of augmented images from a raw batch (before normalization).

    For each of `num_samples` batch items, shows all cameras side-by-side.
    Layout: 2 rows of (num_samples/2) groups, each group = [cam1, cam2, ...].
    Returns (pil_image, save_path) or None if no images found.
    """
    # Collect per-camera tensors: (B, C, H, W)
    cam_tensors = []
    for cam_key in camera_keys:
        if cam_key not in batch:
            continue
        img_tensor = batch[cam_key]
        if not isinstance(img_tensor, torch.Tensor):
            continue
        if img_tensor.ndim == 5:
            img_tensor = img_tensor[:, 0]
        cam_tensors.append(img_tensor)

    if not cam_tensors:
        return None

    num_cams = len(cam_tensors)
    batch_size = cam_tensors[0].shape[0]
    n = min(num_samples, batch_size)

    # Interleave cameras: [s0_c0, s0_c1, s1_c0, s1_c1, ...]
    images = []
    for i in range(n):
        for cam_t in cam_tensors:
            images.append(cam_t[i])

    # nrow = cameras per sample * samples per row (half the samples per row for 2 rows)
    samples_per_row = max(1, n // 2)
    nrow = num_cams * samples_per_row

    grid = make_grid(torch.stack(images), nrow=nrow, normalize=True, value_range=(0, 1))
    # Convert to PIL
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).clip(0, 255).astype("uint8")
    pil_img = Image.fromarray(grid_np)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"augmented_images_step{step}.png"
    pil_img.save(save_path)
    return pil_img, save_path


def save_first_val_image_grid(val_dataloader, camera_keys, output_dir, step, *, uint8_native):
    """Save the augmented-image grid of the first validation; returns (grid, val_dl_iter).

    The grid is drawn under a seed from system entropy so it varies across runs (the
    train/val split stays deterministic via np.random.RandomState). ``torch.seed()``
    reseeds the CPU and every CUDA generator, so both are restored afterwards.
    """
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(torch.seed())
        val_dl_iter = iter(val_dataloader)

        # Collect enough batches to get 8 distinct samples
        num_grid_samples = 8
        collected = {k: [] for k in camera_keys}
        total_collected = 0
        while total_collected < num_grid_samples:
            try:
                raw_val_batch = next(val_dl_iter)
            except StopIteration:
                val_dl_iter = iter(val_dataloader)
                raw_val_batch = next(val_dl_iter)
            for k in camera_keys:
                if k in raw_val_batch and isinstance(raw_val_batch[k], torch.Tensor):
                    collected[k].append(raw_val_batch[k])
            total_collected += raw_val_batch[camera_keys[0]].shape[0]

        # Concatenate into a single "big batch" for the grid function
        merged_batch = {}
        for k in camera_keys:
            if collected[k]:
                merged_batch[k] = torch.cat(collected[k], dim=0)

        # uint8-native: worker frames are post-aug policy-res uint8; finish them on CPU
        # with the SAME shared path the train/val loops use so the grid shows the
        # float [0,1] frames the policy actually consumes.
        if uint8_native:
            merged_batch = finish_camera_images_on_device(
                merged_batch, list(merged_batch.keys()), "cpu"
            )

        grid = save_augmented_image_grid(
            merged_batch, camera_keys, output_dir, step, num_samples=num_grid_samples
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    return grid, val_dl_iter


def check_resume_frame_cache_state(resume_state: dict, frame_cache_active: bool) -> bool:
    """Sticky mixed-image-path marker across resume; returns the new mixed flag.

    The decoded frame cache is an accepted perf knob (<=1/255 pre-aug
    quantization), so unlike the strict precision regime a mismatch WARNs and
    marks the run mixed rather than refusing — refusal would crash-loop every
    run resumed across a change of the default. Checkpoints without the key are
    treated as uncached so far.
    """
    prior_mixed = bool(resume_state.get("decoded_frame_cache_mixed", False))
    saved = resume_state.get("decoded_frame_cache")
    saved_active = bool(saved) if saved is not None else False
    mixed = prior_mixed or (saved_active != frame_cache_active)
    if mixed and not prior_mixed:
        print(
            "WARNING: decoded-frame-cache state changed across resume "
            f"(checkpoint={'ON' if saved_active else 'off'} -> now "
            f"{'ON' if frame_cache_active else 'off'}): the image path is mixed-regime "
            "from here on (<=1/255 pre-aug quantization class). Recorded in the "
            "checkpoint metadata as decoded_frame_cache_mixed."
        )
    return mixed


def check_resume_precision_regime(resume_state: dict, args) -> None:
    """Refuse resume when the checkpoint's precision regime differs from the current flags.

    A run must keep ONE training-forward precision regime end-to-end: the
    checkpoint metadata records only the final flags, so a resume that
    flips --channels-last mid-run would mislabel the arm (and leave restored
    optimizer/EMA tensors in the other memory layout). A checkpoint without the
    key is read as fp32/NCHW.
    """
    saved = dict(resume_state.get("precision_regime") or {"channels_last": False})
    # A checkpoint may record autocast_bf16=False; this trainer has no bf16 autocast regime,
    # so only a checkpoint that trained under it mismatches.
    if saved.get("autocast_bf16") is False:
        del saved["autocast_bf16"]
    current = {"channels_last": bool(args.channels_last)}
    if saved != current:
        raise ValueError(
            f"Resume precision-regime mismatch: checkpoint was trained with {saved} but "
            f"current flags resolve to {current}. Relaunch with matching flags, or start "
            "a fresh --output-dir for the new regime."
        )


# --------------------------------------------------------------------------- #
# DP training primitives: one optimization step, the optimizer/scheduler build and
# the deployable checkpoint layout.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class DPStepContext:
    """Loop-invariant state consumed by one Diffusion-Policy training step."""

    policy: object
    optimizer: torch.optim.Optimizer
    lr_scheduler: object | None
    preprocessor: object
    policy_cfg: object
    device: str
    selected_camera_keys: list[str]
    uint8_native: bool
    grad_clip_norm: float
    args: argparse.Namespace
    timer: TrainingTimer


@dataclasses.dataclass
class DPStepResult:
    """What the caller needs back from one DP step (all of it for logging)."""

    loss: torch.Tensor
    loss_dict: dict | None
    grad_norm: torch.Tensor | float


def dp_train_step(batch, ctx: DPStepContext, *, step: int) -> DPStepResult:
    """One Diffusion-Policy optimization step.

    preprocess -> forward -> backward -> clip -> step -> scheduler.

    The caller still owns data loading, logging, validation and checkpointing;
    ``step`` is the 1-based global step (the EMA warmup schedule is a function of it).
    """
    timer = ctx.timer
    policy = ctx.policy
    optimizer = ctx.optimizer
    device = ctx.device

    with timer("preprocessing"):
        # uint8-native: re-float the worker-emitted uint8 camera frames on GPU
        # BEFORE the lerobot preprocessor (its DeviceProcessorStep is then a no-op
        # move and NormalizerProcessorStep normalizes the float32 policy-res tensor
        # on GPU); crop+resize+aug already ran in the worker.
        if ctx.uint8_native:
            batch = finish_camera_images_on_device(batch, ctx.selected_camera_keys, device)

        # Preprocess (normalization)
        batch = ctx.preprocessor(batch)

        # Move to device (non_blocking works because DataLoader uses pin_memory)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # Temporal dimension handling
        batch = apply_temporal_dim_handling(batch)

    with timer("forward"):
        # Forward pass
        loss, loss_dict = policy.forward(batch)

    with timer("backward"):
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), ctx.grad_clip_norm)

        optimizer.step()
        if ctx.lr_scheduler is not None:
            ctx.lr_scheduler.step()

    return DPStepResult(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)


@dataclasses.dataclass
class DPOptimizerBundle:
    """Everything the training loop needs out of the optimizer/scheduler build."""

    optimizer: torch.optim.Optimizer
    lr_scheduler: object | None
    grad_clip_norm: float


def dp_build_optimizer(policy, policy_cfg, args) -> DPOptimizerBundle:
    """Build the DP optimizer + LR scheduler from the policy presets."""
    optimizer_cfg = policy_cfg.get_optimizer_preset()
    if args.lr is not None:
        optimizer_cfg.lr = args.lr

    if hasattr(policy, "get_optim_params"):
        params = policy.get_optim_params()
    else:
        params = policy.parameters()
    optimizer = optimizer_cfg.build(params)

    scheduler_cfg = policy_cfg.get_scheduler_preset()
    lr_scheduler = (
        scheduler_cfg.build(optimizer, args.training_steps) if scheduler_cfg is not None else None
    )

    grad_clip_norm = optimizer_cfg.grad_clip_norm

    print(f"Optimizer: {optimizer.__class__.__name__} (lr={optimizer_cfg.lr})")
    if lr_scheduler is not None:
        print(
            f"LR Scheduler: {scheduler_cfg.__class__.__name__} "
            f"(name={getattr(scheduler_cfg, 'name', None)})"
        )
    print(f"Gradient clip norm: {grad_clip_norm}")

    return DPOptimizerBundle(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_clip_norm=grad_clip_norm,
    )


def save_dp_deployment_dir(policy, preprocessor, postprocessor, save_dir) -> Path:
    """Write a self-contained, deployable DP checkpoint directory.

    Weights + pre/post-processors only — the resumable optimizer/RNG state is a
    separate concern (``save_training_state``). Crop/action policy-contract state is
    encoded in config.json so checkpoints are self-contained and eval needs no hidden files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)
    return save_dir


def train(args):
    # Resolve per-task (DPTrainingRecipe) + per-station (camera_utils) training defaults
    # into args BEFORE any guard/consumer reads them, so a --task launch can omit the whole
    # recipe. Explicit CLI flags always win; bare (no --task) callers are unchanged.
    _apply_real_training_recipe(args)
    if not args.vision_backbone.startswith("resnet"):
        raise ValueError(
            f"--vision-backbone must be a torchvision ResNet, got {args.vision_backbone!r}"
        )
    # Resolve the backbone's pretrained weights once, pinned explicitly rather than
    # inherited from the installed LeRobot version: ImageNet weights (paired with
    # BatchNorm) by default; --pretrained-backbone-weights takes a torchvision weights
    # string, or 'none'/'random'/'' for a random-init GroupNorm backbone.
    _pbw = args.pretrained_backbone_weights
    if _pbw is None:
        resolved_pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1"
    elif _pbw.lower() in ("none", "random", ""):
        resolved_pretrained_backbone_weights = None
    else:
        resolved_pretrained_backbone_weights = _pbw
    if args.spatial_softmax_num_keypoints is not None:
        if args.spatial_softmax_num_keypoints <= 0:
            raise ValueError(
                "--spatial-softmax-num-keypoints must be positive, got "
                f"{args.spatial_softmax_num_keypoints}"
            )
    # Setup
    set_seed(args.seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Enable CUDA performance optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    print(
        f"Precision: channels_last={'ON' if args.channels_last else 'off'} "
        "(fp32 + tf32 matmul; conv-weight layout only)"
    )

    # Parse comma-separated repo IDs
    repo_ids = [r.strip() for r in args.repo_ids.split(",")]
    eval_repo_ids = [r.strip() for r in args.eval_repo_ids.split(",") if r.strip()]
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

    print("=" * 60)
    print("Training Diffusion Policy on Real Robot Data")
    print("=" * 60)
    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else None

    print(f"Datasets: {', '.join(repo_ids)}")
    if use_external_eval:
        print(f"Eval-only datasets: {', '.join(eval_repo_ids)}")
    if dataset_root is not None:
        print(f"Dataset root: {dataset_root}")
    print(f"Camera filter: *{args.camera_filter}")
    if args.image_height is not None:
        print(f"Image resize: {args.image_height}x{args.image_width}")
    print(f"Filter DAgger: {args.filter_dagger}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Train batch size: {args.batch_size}")
    print(
        f"Eval batch size: {args.eval_batch_size} "
        f"({args.num_eval_batches} batches = "
        f"{args.eval_batch_size * args.num_eval_batches} frame anchors per eval log)"
    )
    print("=" * 60)
    print()

    dataset_root = prepare_datasets(
        [*repo_ids, *eval_repo_ids],
        dataset_root,
        dataset_revisions,
        sync=not args.no_dataset_sync,
    )
    # Episodes each repo contributes before any DAgger filtering (None = all episodes).
    selected_episodes = resolve_selected_episodes(
        [*repo_ids, *eval_repo_ids], dataset_root, dataset_revisions, dataset_selectors
    )

    # 1. Episode-level filtering (before dataset load)
    episodes_dict = {}
    dataset_info = {}
    if args.filter_dagger:
        print("Filtering datasets...")
        for repo_id in repo_ids:
            print(f"Processing dataset: {repo_id}")
            filtered_episodes, matching_frames = filter_episodes(
                repo_id,
                root=dataset_dir(dataset_root, repo_id),
                revision=dataset_revision(dataset_revisions, repo_id),
            )
            selection = selected_episodes[repo_id]
            if selection is not None:
                keep = set(selection)
                filtered_episodes = (
                    list(selection)
                    if filtered_episodes is None
                    else [ep for ep in filtered_episodes if ep in keep]
                )
                if not filtered_episodes:
                    raise ValueError(
                        f"{repo_id}: --dataset-episodes selection leaves no episode after "
                        "--filter-dagger"
                    )
            episodes_dict[repo_id] = filtered_episodes

            temp_meta = LeRobotDatasetMetadata(
                repo_id,
                root=dataset_dir(dataset_root, repo_id),
                revision=dataset_revision(dataset_revisions, repo_id),
            )
            num_episodes = (
                len(filtered_episodes)
                if filtered_episodes is not None
                else temp_meta.total_episodes
            )
            dataset_info[repo_id] = {
                "total_episodes": temp_meta.total_episodes,
                "used_episodes": num_episodes,
                "total_frames": temp_meta.total_frames,
                "matching_frames": matching_frames,
            }
            print()
    else:
        if any(selected_episodes[repo_id] is not None for repo_id in repo_ids):
            episodes_dict = {repo_id: selected_episodes[repo_id] for repo_id in repo_ids}
        for repo_id in repo_ids:
            temp_meta = LeRobotDatasetMetadata(
                repo_id,
                root=dataset_dir(dataset_root, repo_id),
                revision=dataset_revision(dataset_revisions, repo_id),
            )
            selection = selected_episodes[repo_id]
            dataset_info[repo_id] = {
                "total_episodes": temp_meta.total_episodes,
                "used_episodes": temp_meta.total_episodes if selection is None else len(selection),
                "total_frames": temp_meta.total_frames,
                "matching_frames": temp_meta.total_frames,
            }
    # 2. Load dataset metadata from primary (first) repo for policy config
    print("Loading dataset metadata...")
    primary_repo_id = repo_ids[0]
    ds_meta = LeRobotDatasetMetadata(
        primary_repo_id,
        root=dataset_dir(dataset_root, primary_repo_id),
        revision=dataset_revision(dataset_revisions, primary_repo_id),
    )
    print(f"  Primary dataset: {primary_repo_id}")
    print(f"  FPS: {ds_meta.fps}")
    for repo_id, info in dataset_info.items():
        print(f"  {repo_id}: {info['used_episodes']} episodes (of {info['total_episodes']} total)")
    print()

    # 3. Convert features and filter cameras
    all_features = dataset_to_policy_features(ds_meta.features)

    # Identify all camera keys
    all_camera_keys = [key for key, ft in all_features.items() if ft.type is FeatureType.VISUAL]
    # Explicit --camera-keys selects exactly those cameras; otherwise the suffix filter.
    # The dataset may STORE more cameras than the policy consumes (kept for reuse).
    # Task-default: when no explicit --camera-keys is given AND --task resolves to a spec
    # whose consumed_camera_roles is non-empty, default the consumed camera keys to those
    # ROLE names (comma-joined); the released datasets store cameras under role names.
    # Explicit --camera-keys always overrides.
    camera_keys = args.camera_keys
    if (not camera_keys or not camera_keys.strip()) and args.task:
        task_spec = _resolve_task_spec(args.task)
        if task_spec is not None and task_spec.consumed_camera_roles:
            camera_keys = ",".join(task_spec.consumed_camera_roles)
            print(
                f"--camera-keys defaulted from task {args.task!r} "
                f"(spec {task_spec.name!r}) consumed_camera_roles -> {camera_keys!r}"
            )
    selected_camera_keys = select_camera_feature_keys(
        all_camera_keys, camera_keys=camera_keys, camera_filter=args.camera_filter
    )
    excluded_camera_keys = [key for key in all_camera_keys if key not in selected_camera_keys]

    _sel = f"--camera-keys {camera_keys}" if camera_keys else f"suffix *{args.camera_filter}"
    print(f"Camera selection ({_sel}):")
    print(f"  All cameras: {all_camera_keys}")
    print(f"  Selected:    {selected_camera_keys}")
    print(f"  Excluded:    {excluded_camera_keys}")
    print()

    # Build input/output features (excluding filtered-out cameras)
    output_features = {key: ft for key, ft in all_features.items() if ft.type is FeatureType.ACTION}
    input_features = {
        key: ft
        for key, ft in all_features.items()
        if key not in output_features and key not in excluded_camera_keys
    }
    # The UMI relative arm grows the action from 7D (xyz + euler + gripper) to 10D
    # (xyz + 6D-rotation + gripper). output_features["action"] is derived from the
    # ON-DISK ds_meta (7D), but the remap that happens later mutates the in-memory
    # action column to 10D. The policy head dim is built FROM output_features here
    # (long before the remap), so without this override the DP head would be 7D
    # while the dataloader serves 10D actions — a silent shape mismatch. Patch the
    # ACTION feature shape to the post-remap dim so the head matches the data.
    if args.action_mode == "relative":
        from mulligan.real.policy.rotation6d import R6_DIM

        r6_action_dim = 3 + R6_DIM + 1  # 10
        action_ft = output_features["action"]
        if tuple(action_ft.shape) != (7,):
            raise ValueError(
                f"action_mode={args.action_mode} expects a 7D on-disk action feature to "
                f"expand to {r6_action_dim}D; got on-disk shape {action_ft.shape}. "
                "Inspect the datamix."
            )
        output_features["action"] = PolicyFeature(type=action_ft.type, shape=(r6_action_dim,))
        print(
            "Action UMI relative pose: output 'action' shape "
            f"{action_ft.shape} -> {output_features['action'].shape} (xyz + r6 + gripper)"
        )

    # Override image feature shapes if --image-height / --image-width are set
    if (args.image_height is None) != (args.image_width is None):
        raise ValueError(
            "--image-height and --image-width must both be specified (or neither). "
            f"Got height={args.image_height}, width={args.image_width}"
        )
    if args.image_height is not None:
        for key, ft in input_features.items():
            if ft.type is FeatureType.VISUAL:
                c = ft.shape[0]
                orig_shape = ft.shape
                input_features[key] = PolicyFeature(
                    type=ft.type,
                    shape=(c, args.image_height, args.image_width),
                )
                print(f"Image resize: {key} shape {orig_shape} -> {input_features[key].shape}")

    # Parse per-camera static replacement crop. The crop is
    # applied to the camera frame before resize, so the resulting feature shape
    # is unchanged (still (c, image_height, image_width)). The box rides in a
    # policy config camera_crop_boxes so eval crops identically with no extra eval
    # flags. Boxes (explicit and the STATION_CAMERA_DEFAULT_CROPS merged in below) are
    # pixels of the frames as the dataset stores them (640x480 for role-keyed datasets).
    side_crop_map = parse_side_crop(args.side_crop)
    # Apply per-camera STATION default crops to any selected camera lacking an explicit
    # --side-crop, so the same ROI is used by default at train + (via camera_crop_boxes)
    # at eval. Explicit --side-crop overrides; empty defaults map = no-op. A --task whose
    # spec declares camera_crop_overrides gets those over the station defaults (per-task
    # ROI without perturbing other lines sharing the same camera role).
    _default_crops = dict(STATION_CAMERA_DEFAULT_CROPS)
    _crop_spec = _resolve_task_spec(args.task) if args.task else None
    if _crop_spec is not None and _crop_spec.camera_crop_overrides:
        _default_crops.update(_crop_spec.camera_crop_overrides)
        print(
            f"--task {args.task!r}: per-task crop override(s) over station defaults: "
            f"{dict(_crop_spec.camera_crop_overrides)}"
        )
    _selected_cameras = {k.removeprefix("observation.images.") for k in selected_camera_keys}
    side_crop_map = merge_default_crops(side_crop_map, _selected_cameras, _default_crops)
    if side_crop_map:
        if args.image_height is None:
            raise ValueError(
                "--side-crop requires --image-height/--image-width (the crop is "
                "applied before resize, so a target resolution must be set)."
            )
        # selected_camera_keys are full feature keys (observation.images.<camera>);
        # --side-crop is keyed by the bare camera name.
        for cam in side_crop_map:
            if cam not in _selected_cameras:
                raise ValueError(
                    f"--side-crop camera {cam!r} is not in the selected cameras "
                    f"{sorted(_selected_cameras)} (check --camera-filter)."
                )
        for cam, box in side_crop_map.items():
            print(
                f"Per-camera replacement crop: {cam} box (x0,y0,x1,y1)={box} -> resize to "
                f"{args.image_height}x{args.image_width} "
                "(pixels of the stored frame)"
            )

    print("Input features:")
    for key, ft in input_features.items():
        print(f"  {key}: shape={ft.shape}, type={ft.type}")
    print("Output features:")
    for key, ft in output_features.items():
        print(f"  {key}: shape={ft.shape}, type={ft.type}")
    print()

    # 4. Create policy config
    chunk_size = args.chunk_size if args.chunk_size is not None else 16

    policy_kwargs = {
        "input_features": input_features,
        "output_features": output_features,
        "device": device,
    }

    if args.lr is not None:
        policy_kwargs["optimizer_lr"] = args.lr

    policy_kwargs["horizon"] = chunk_size
    # Resolve the execution horizon from the shared real-robot protocol
    # constant when unset, so a fresh diffusion checkpoint bakes in exec=6
    # (predict `chunk_size`/horizon, execute n_action_steps) rather than
    # silently conflating exec with chunk_size.
    from mulligan.real.policy.loader import REAL_PROTOCOL_N_ACTION_STEPS

    if args.n_action_steps is None:
        args.n_action_steps = REAL_PROTOCOL_N_ACTION_STEPS
    policy_kwargs["n_action_steps"] = args.n_action_steps
    policy_kwargs["vision_backbone"] = args.vision_backbone
    # A single observation step (current frame only), not the DP-paper default of 2.
    # The relative action pipeline relies on it: the executed window begins at the
    # relativization anchor step.
    policy_kwargs["n_obs_steps"] = 1
    # DP's built-in random crop is off (RandomAffine translate covers spatial aug).
    policy_kwargs["crop_shape"] = None
    # Use DDIM with 8 steps for fast inference (training loss is identical)
    policy_kwargs["noise_scheduler_type"] = "DDIM"
    policy_kwargs["num_inference_steps"] = 8
    if args.scheduler_name is not None:
        policy_kwargs["scheduler_name"] = args.scheduler_name
    # Mask loss for padded actions at episode boundaries
    policy_kwargs["do_mask_loss_for_padding"] = True
    if args.down_dims is not None:
        policy_kwargs["down_dims"] = tuple(int(x) for x in args.down_dims.split(","))
    if args.spatial_softmax_num_keypoints is not None:
        policy_kwargs["spatial_softmax_num_keypoints"] = args.spatial_softmax_num_keypoints
    # Separate per-camera encoders, pinned explicitly so the decision does not
    # track the LeRobot config default.
    policy_kwargs["use_separate_rgb_encoder_per_camera"] = True
    # Backbone init + norm: pin the resolved decision explicitly (resolved up front
    # next to the visual-normalization 'auto' handling so the two stay coherent).
    if resolved_pretrained_backbone_weights is not None:
        policy_kwargs["pretrained_backbone_weights"] = resolved_pretrained_backbone_weights
        # Torchvision ImageNet weights depend on BatchNorm statistics. LeRobot
        # intentionally rejects pretrained backbones if BatchNorm is replaced.
        policy_kwargs["use_group_norm"] = False
    else:
        # Random-init ResNet control: GroupNorm (small-batch friendly). Pin
        # pretrained_backbone_weights=None explicitly too: otherwise it inherits
        # the DiffusionConfig default (ResNet18_Weights.IMAGENET1K_V1) and collides
        # with use_group_norm=True ("can't replace BatchNorm in a pretrained model").
        policy_kwargs["pretrained_backbone_weights"] = None
        policy_kwargs["use_group_norm"] = True

    print("Creating diffusion policy...")
    policy_cfg = make_policy_config(POLICY_TYPE, **policy_kwargs)
    policy_cfg.camera_crop_boxes = dict(side_crop_map)
    # Policy I/O contract read by the deploy loader: the release trains base-frame
    # cartesian velocity (or the relative pose of --action-mode relative).
    policy_cfg.action_target = "cartesian_velocity"
    policy_cfg.cartesian_action_frame = "base"
    policy_cfg.action_mode = args.action_mode
    if args.visual_normalization == "identity":
        policy_cfg.normalization_mapping["VISUAL"] = NormalizationMode.IDENTITY
    if args.n_action_steps is not None:
        if policy_cfg.n_action_steps != args.n_action_steps:
            raise ValueError(
                f"Diffusion n_action_steps mismatch: requested {args.n_action_steps}, "
                f"config has {policy_cfg.n_action_steps}"
            )
        print(
            f"[protocol] Diffusion action horizon: predict {policy_cfg.horizon} (horizon), "
            f"execute {policy_cfg.n_action_steps} (n_action_steps)."
        )
        print(f"[protocol] SpatialSoftmax keypoints: {policy_cfg.spatial_softmax_num_keypoints}")

    # 5. Resolve delta timestamps and filter out excluded cameras
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    if delta_timestamps is not None:
        # Remove entries for excluded camera keys
        delta_timestamps = {
            key: ts for key, ts in delta_timestamps.items() if key not in excluded_camera_keys
        }
        if not delta_timestamps:
            delta_timestamps = None

    print("Delta timestamps:")
    if delta_timestamps:
        for key, ts in delta_timestamps.items():
            if len(ts) <= 3:
                print(f"  {key}: {ts}")
            else:
                print(f"  {key}: [{ts[0]}, ..., {ts[-1]}] ({len(ts)} steps)")
    else:
        print("  None (no temporal dependencies)")
    print()

    # 6. Image augmentation: LeRobot's standard set (up to 3 random transforms).
    image_transforms_cfg = ImageTransformsConfig(enable=True)
    image_aug_transforms = ImageTransforms(image_transforms_cfg)
    print("Image augmentation: STANDARD (up to 3 random transforms)")
    print(f"  Transforms: {list(image_transforms_cfg.tfs.keys())}")

    # Prepend the shared deterministic policy image resize if an image resolution
    # override is set. Augmentation stays separate from the resize so cropped cameras
    # can use the same crop+resize helper as live eval and then apply the train-only
    # aug; uncropped cameras get the same resize.
    image_transforms = image_aug_transforms
    resize_tf = None
    crop_resize_hw = None
    if args.image_height is not None:
        crop_resize_hw = (args.image_height, args.image_width)
        resize_tf = PolicyImagePreprocessTransform((args.image_height, args.image_width))
        image_transforms = v2.Compose([resize_tf, image_aug_transforms])
        print(f"Image resize: {args.image_height}x{args.image_width} (applied before augmentation)")

    # Validation/eval images are deployment-faithful: RESIZE ONLY, no augmentation.
    # Augmented val conflates aug robustness with the metric, and the augmentation RNG
    # is the dataloader-worker RNG, which biases val/loss and breaks cross-checkpoint
    # comparability. Resize is preserved (the policy input size must still match).
    val_image_transforms = resize_tf  # None if no resize override
    val_post_resize_transforms = None
    print("Validation images: clean (resize-only, no augmentation)")
    print()

    # ---- --uint8-native-images (must run BEFORE dataset build so wrapped transforms
    # flow into the dataset and crop-proxy installs) ----
    # The FULL worker transform stack (crop+resize+aug) runs unchanged per-sample on
    # CPU, then the float policy-res frame is quantized to uint8 IN the worker (same
    # clamp*255 formula as VisionReplayBuffer.pack_images_uint8, single-sourced in
    # image_preprocess.quantize_float01_to_uint8) so IPC ships 4x fewer bytes; the
    # trainer re-floats on GPU. Round-trip error <= 1/255 per pixel.
    uint8_native = args.uint8_native_images
    uint8_native_tier: str | None = None
    if uint8_native and args.num_workers <= 0:
        # Default-ON, so an unmet precondition downgrades LOUDLY to the
        # float path instead of crashing (perf knob — the experiment is identical either
        # way). Opt out deliberately with --no-uint8-native-images to silence the WARN.
        uint8_native = False
        args.uint8_native_images = False
        print(
            "WARNING: uint8-native images DOWNGRADED to the float path (precondition "
            "unmet): --uint8-native-images requires --num-workers>0, the whole point is "
            "shrinking IPC from the DataLoader worker processes."
        )
    if uint8_native:
        uint8_native_tier = "quantized-aug"
        # Wrap every worker-side transform slot so whatever applies LAST in the
        # worker emits quantized uint8: the stock reader loop and the crop
        # proxy's non-cropped branch run `image_transforms`/`val_image_transforms`
        # (wrapped whole), the cropped branch runs crop_post_transform =
        # `image_aug_transforms` / `val_post_resize_transforms` (wrapped, incl.
        # the None -> quantize-only case for clean val). The Compose inside the
        # wrapped image_transforms holds the UNwrapped aug object, so each
        # frame is quantized exactly once.
        image_transforms = QuantizeUint8PostTransform(image_transforms)
        image_aug_transforms = QuantizeUint8PostTransform(image_aug_transforms)
        val_image_transforms = QuantizeUint8PostTransform(val_image_transforms)
        val_post_resize_transforms = QuantizeUint8PostTransform(val_post_resize_transforms)
        print(
            "uint8-native images ENABLED: full worker crop+resize+aug stack unchanged, "
            "post-aug frames quantized to uint8 in the worker (4x less IPC); re-floated "
            "on GPU (<=1/255 round-trip)"
        )
        print()

    # 7. Create dataset (always MultiLeRobotDataset, even for single datasets)
    print("Loading dataset(s)...")
    dataset = load_multi_dataset(
        repo_ids,
        dataset_root,
        dataset_revisions,
        episodes=episodes_dict if episodes_dict else None,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        image_transforms=image_transforms,
    )
    print(f"  Loaded {dataset.num_frames} frames across {dataset.num_episodes} episodes")
    sub_datasets = require_multilerobot_subdatasets(dataset)
    if dataset.disabled_features:
        print(f"  Disabled features (not common): {dataset.disabled_features}")

    # Remove excluded camera features from sub-dataset metadata so their
    # videos are never decoded and transforms are never applied to them.
    if excluded_camera_keys:
        remove_features_from_lerobot_subdatasets(sub_datasets, excluded_camera_keys)
        print(f"  Removed {len(excluded_camera_keys)} excluded cameras from video pipeline")

    # Substitute null floats (e.g. unrecorded telemetry vectors) with NaN so the torch
    # transform does not raise on a Python None mid-batch. Done before the ROI proxies wrap
    # the sub-datasets and before stats/normalization are built; policy-critical columns
    # (action / observation.*) still fail loudly rather than being silently NaN-filled.
    null_filled = fill_null_floats_in_lerobot_subdatasets(sub_datasets)
    if null_filled:
        print(f"  NaN-filled null float columns (unrecorded aux signals): {null_filled}")

    # Column-cache fast reader: removes HF-datasets Python overhead (~14-18 ms/
    # sample, dominated by Features-deepcopy churn in delta-key queries) from
    # every worker __getitem__. Outputs are identical to the stock reader
    # (tests/unit/test_fast_lerobot_reader.py). MUST come after the null-fill
    # above — the cache snapshots the (possibly rebuilt) hf_dataset table.
    if fast_reader_enabled_by_env():
        cached_bytes = enable_fast_reader_on_subdatasets(sub_datasets)
        print(f"  Fast reader ON: {cached_bytes / 1e6:.1f} MB non-video columns cached in RAM")
    else:
        print(f"  Fast reader DISABLED via {FAST_READER_ENV}=0")

    # Per-camera static ROI crop: replace sub-datasets with key-aware proxies
    # that crop the native frame BEFORE the shared resize/aug (which is reapplied
    # as base_transform). The stock image_transforms is disabled inside each
    # proxy. Re-fetch the (possibly proxied) sub-dataset handle.
    if side_crop_map:
        crop_feature_map = build_crop_feature_map(side_crop_map)
        sub_datasets = install_per_camera_crop(
            dataset,
            crop_feature_map,
            image_transforms,
            crop_resize_hw=crop_resize_hw,
            crop_post_transform=image_aug_transforms,
        )
        print(
            f"  Installed per-camera crop on {len(side_crop_map)} camera(s): "
            f"{sorted(crop_feature_map)}"
        )
    print()

    # Decode-once shared-RAM frame cache: decode every served train frame ONCE
    # at startup into shared policy-res uint8 tensors so workers serve frames
    # from RAM instead of re-decoding video per sample. Fallbacks below are
    # perf-only (the served bytes are identical either way, modulo the accepted
    # <=1/255 pre-aug quantization) and therefore WARN + continue.
    frame_cache_active = False
    if args.decoded_frame_cache:
        if not side_crop_map or crop_resize_hw is None:
            print(
                "WARNING: --decoded-frame-cache requires per-camera crop boxes and "
                "--image-height (policy-res cache); falling back to per-sample decode."
            )
        elif not fast_reader_enabled_by_env():
            print(
                f"WARNING: --decoded-frame-cache requires the fast reader "
                f"({FAST_READER_ENV}=1); falling back to per-sample decode."
            )
        else:
            from mulligan.data.frame_cache import build_and_attach_frame_caches

            print("Building decoded frame cache (one-time sequential sweep)...")
            frame_cache_active = build_and_attach_frame_caches(
                sub_datasets,
                crop_feature_map=crop_feature_map,
                crop_reference_hw=None,
                target_hw=crop_resize_hw,
                threads=min(32, len(os.sched_getaffinity(0))),
            )
        if frame_cache_active:
            for sub in sub_datasets:
                sub.set_frames_precropped(True)
        print(f"  Decoded frame cache: {'ACTIVE' if frame_cache_active else 'OFF (fallback)'}")

    eval_dataset = None
    eval_sub_datasets = None
    if use_external_eval and args.eval_freq > 0:
        print("Loading eval-only dataset(s)...")
        eval_episodes = {repo_id: selected_episodes[repo_id] for repo_id in eval_repo_ids}
        eval_dataset = load_multi_dataset(
            eval_repo_ids,
            dataset_root,
            dataset_revisions,
            episodes=eval_episodes if any(v is not None for v in eval_episodes.values()) else None,
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
            image_transforms=val_image_transforms,
        )
        print(
            f"  Loaded {eval_dataset.num_frames} eval frames across "
            f"{eval_dataset.num_episodes} eval episodes"
        )
        eval_sub_datasets = require_multilerobot_subdatasets(eval_dataset)
        if eval_dataset.disabled_features:
            print(f"  Eval disabled features (not common): {eval_dataset.disabled_features}")
        if excluded_camera_keys:
            remove_features_from_lerobot_subdatasets(eval_sub_datasets, excluded_camera_keys)
        eval_null_filled = fill_null_floats_in_lerobot_subdatasets(eval_sub_datasets)
        if eval_null_filled:
            print(f"  Eval NaN-filled null float columns: {eval_null_filled}")
        if fast_reader_enabled_by_env():
            eval_cached = enable_fast_reader_on_subdatasets(eval_sub_datasets)
            print(f"  Fast reader ON (eval): {eval_cached / 1e6:.1f} MB cached")
        if side_crop_map:
            crop_feature_map = build_crop_feature_map(side_crop_map)
            eval_sub_datasets = install_per_camera_crop(
                eval_dataset,
                crop_feature_map,
                val_image_transforms,
                crop_resize_hw=crop_resize_hw,
                crop_post_transform=val_post_resize_transforms,
            )
        if frame_cache_active:
            # Serve external eval from the cache too when it fits the budget (the
            # served frames are the same up to the cache's <=1/255 quantization).
            from mulligan.data.frame_cache import (
                build_and_attach_frame_caches,
                frame_cache_size_bytes,
            )

            train_inners = [getattr(sub, "_inner", sub) for sub in sub_datasets]
            eval_cache_ok = build_and_attach_frame_caches(
                eval_sub_datasets,
                crop_feature_map=crop_feature_map,
                crop_reference_hw=None,
                target_hw=crop_resize_hw,
                threads=min(32, len(os.sched_getaffinity(0))),
                # Charge the resident train cache against the budget so
                # train+eval together stay under the limit.
                reserved_bytes=frame_cache_size_bytes(train_inners, crop_resize_hw),
            )
            if eval_cache_ok:
                for sub in eval_sub_datasets:
                    sub.set_frames_precropped(True)
        print()

    # 7b. UMI relative mode: remap the canonical 'action'
    # column to the absolute COMMANDED EE-pose trajectory now; the batch-time
    # relativization + per-timestep (T,D) stats happen below (after drop_n_last is
    # known, before the preprocessor is built). Must run before episode boundaries /
    # stats, which key off the 'action' column.
    #
    # TARGET SOURCE = the COMMANDED pose (action.cartesian_position), NOT the observed
    # proprio pose. The robot runs a SOFT Cartesian controller, so it does NOT reach the
    # commanded targets within a step — the proprio trajectory is the attenuated/lagged
    # ACHIEVED motion, not what the expert asked for. Training the relative targets ON
    # proprio makes the policy imitate the controller's lag (scored 0/9). We regress the
    # COMMAND trajectory instead (remap_action_to_position_r6_in_subdatasets).
    #
    # ANCHOR = the current PROPRIO pose (observation.state), applied by the
    # downstream RelativePoseActionProcessorStep / per-timestep stats — NOT the command's
    # own current pose. Relativizing command targets against measured proprio makes rel[0]
    # the command-vs-proprio lead (not identity), so the eval decode (which re-grounds on
    # live proprio each chunk) reconstructs the command trajectory with no boundary retreat
    # and no open-loop runaway. Anchoring on the command frame deleted that lead and
    # produced the replan-locked sawtooth. So: COMMAND targets, PROPRIO anchor.
    if args.action_mode == "relative":
        print(
            "Action mode: relative (UMI) — remapping canonical 'action' to the absolute "
            "COMMANDED EE-pose trajectory (10D = xyz + 6D-rotation + gripper) from "
            "action.cartesian_position (NOT proprio — soft controller); relativization "
            "+ per-timestep (T,D) normalization applied downstream."
        )
        remap_action_to_position_r6_in_subdatasets(dataset)
        if eval_dataset is not None:
            remap_action_to_position_r6_in_subdatasets(eval_dataset)
        print()
    else:
        print("Action target: cartesian_velocity (base frame)")
        print()

    # 8. Compute episode boundaries and create sampler
    # Determine how many frames to drop at end of each episode.
    drop_n_last = policy_cfg.drop_n_last_frames
    if args.drop_n_last_frames is not None:
        # The LeRobot DiffusionConfig default (7) encodes horizon-16/exec-8/obs-2
        # geometry; for predict-8/exec-6/obs-1 the same formula gives 2, so the
        # config default silently starves the final ~0.5 s of every episode of
        # anchor coverage. 'auto' applies the formula to the actual geometry.
        if args.drop_n_last_frames == "auto":
            drop_n_last = max(
                0,
                policy_cfg.horizon - policy_cfg.n_action_steps - policy_cfg.n_obs_steps + 1,
            )
        else:
            drop_n_last = int(args.drop_n_last_frames)
        print(
            f"drop_n_last_frames override: {drop_n_last} (--drop-n-last-frames={args.drop_n_last_frames})"
        )

    # The supervised action chunk extends max(action_delta_indices) frames FORWARD of
    # each anchor; LeRobot's loss masks only action_is_pad (frames beyond the RAW
    # episode end), never in-episode is_valid==0 frames, so on soft-truncated episodes
    # the anchor range must retreat far enough that no supervised timestep lands on
    # junk (see clamp_soft_truncated_anchor_ends).
    action_delta_indices = policy_cfg.action_delta_indices
    max_forward_action_offset = max(action_delta_indices) if action_delta_indices else 0

    # Real outcome-edited datasets carry a per-frame is_valid column: a leading valid
    # prefix (frames 0..outcome) then an invalid suffix (post-outcome retract/reset junk
    # + terminal pad) that soft truncation marks is_valid=0. Clamp every episode END to
    # its valid prefix so anchors never land on — and action chunks never extend into —
    # the invalid suffix. drop_n_last then counts back from the valid end (below).
    # Datasets without is_valid are returned unchanged.
    from_indices, valid_to_indices, raw_to_indices, n_suffix_excluded = (
        compute_multidataset_valid_boundaries(sub_datasets)
    )
    if n_suffix_excluded > 0:
        print(
            f"[VALID-PREFIX] Excluded {n_suffix_excluded} invalid-suffix frame(s) (is_valid==0; "
            f"terminal pad and any soft-truncation junk) across {len(from_indices)} episodes "
            "from anchor sampling — expected for outcome-edited real data."
        )
    to_indices, n_overrun_anchors_excluded = clamp_soft_truncated_anchor_ends(
        from_indices,
        valid_to_indices,
        raw_to_indices,
        drop_n_last=drop_n_last,
        max_forward_action_offset=max_forward_action_offset,
    )
    if n_overrun_anchors_excluded > 0:
        print(
            f"[VALID-PREFIX] Soft-truncated episodes: excluded {n_overrun_anchors_excluded} "
            f"additional tail anchor(s) so no supervised action timestep (max forward offset "
            f"{max_forward_action_offset}) lands on an is_valid==0 frame."
        )

    # 8a-rel. UMI per-timestep (T,D) relative-pose action stats. The relative
    # representation couples the anchor with each future pose, so its normalization
    # stats are a windowed quantity that must be fit per-timestep (t+1 != t+2). This
    # runs after the proprio-pose remap and after drop_n_last is known, and OVERWRITES
    # dataset.stats["action"] with the (horizon, 10) relative stats the preprocessor
    # then bakes into the per-timestep normalizer.
    if args.action_mode == "relative":
        compute_relative_pose_pertimestep_stats(
            dataset,
            horizon=policy_cfg.horizon,
            n_obs_steps=policy_cfg.n_obs_steps,
            drop_n_last_frames=drop_n_last,
        )

    # 8b. Episode-level train/val split
    num_episodes = len(from_indices)
    val_dataloader = None
    val_enabled = args.eval_freq > 0 and (
        (use_external_eval and eval_dataset is not None and eval_dataset.num_episodes > 0)
        or (not use_external_eval and num_episodes > 1)
    )

    if val_enabled and use_external_eval:
        train_episodes = list(range(num_episodes))
        val_episodes = []
        print(
            f"Train/Val split: {len(train_episodes)} train episodes from training repos, "
            f"{eval_dataset.num_episodes} eval-only episodes from external repos"
        )
        print("  Using all training episodes; --val-pct ignored because --eval-repo-ids is set")
    elif val_enabled:
        train_episodes, val_episodes = make_episode_val_split(
            num_episodes, args.val_pct, seed=args.seed
        )
        print(
            f"Train/Val split: {len(train_episodes)} train episodes, {len(val_episodes)} val episodes"
        )
        print(f"  Val episodes: {val_episodes}")
        # Internal-split val shares the augmented training dataset object (a val
        # SAMPLER over the same frames), so it CANNOT serve clean images. Be loud
        # rather than silently augmenting: deployment-faithful val/loss needs an
        # external held-out repo (--eval-repo-ids).
        print(
            "  WARNING: internal-split validation reuses the AUGMENTED training "
            "dataset; val/loss is measured on augmented "
            "images. Use --eval-repo-ids for clean (resize-only) validation."
        )
    else:
        if args.eval_freq > 0 and num_episodes <= 1:
            print(f"WARNING: Only {num_episodes} episode(s), disabling validation")
        train_episodes = list(range(num_episodes))
        val_episodes = []

    if args.filter_dagger:
        # Frame-level filtering: keep only human frames from DAgger datasets, all from demos.
        #
        # Episode-level filtering (above) already discards failed episodes. This means:
        # - Human corrections from failed episodes are excluded entirely. This is the right
        #   call for BC (don't imitate trajectories that didn't succeed), but note that some
        #   failed episodes contain high-quality human interventions where the *policy* failed
        #   later — that data is lost here. For RL, failed human data has value as negative
        #   examples, but we'd need a finer-grained label than episode-level success to
        #   distinguish "human made a mistake" from "human was fine, policy failed after
        #   handoff." Our current columns (source + episode-level success) can't tell these
        #   apart, so episode-level filtering is the best we can do for now.
        #
        # NOTE: This filters which frames are used as *anchor points* (sampled indices).
        # The action chunk extends forward from the anchor and may include policy-generated
        # actions near human→policy transitions. This is intentional:
        # - The last human frames before handoff are the most valuable recovery actions
        # - Dropping them to get "pure human" chunks would lose critical training signal
        # - The problematic case (policy obs → human action) is already prevented since
        #   policy frames are never sampled as anchors
        # - In some cases the policy actions in the tail may be bad (human handed back
        #   control too early and the policy went OOD), but this is a small fraction of
        #   frames and most chunks will be predominantly human actions anyway.
        # - Masking those tail policy actions would require deeper surgery into
        #   LeRobotDataset.__getitem__ / _get_query_indices to produce per-chunk
        #   source-aware padding masks — not worth the complexity for now.
        print("Building frame-level filter for DAgger data...")

        # Build a frame→episode lookup from episode boundaries
        frame_to_episode = {}
        for ep_i, (fi, ti) in enumerate(zip(from_indices, to_indices)):
            for frame_idx in range(fi, ti):
                frame_to_episode[frame_idx] = ep_i

        val_episode_set = set(val_episodes)

        train_filtered_indices = []
        val_filtered_indices = []
        global_offset = 0
        global_ep_i = 0  # aligned with compute_multidataset_valid_boundaries episode order
        total_human_frames = 0
        total_policy_frames = 0

        for sub_idx, sub_ds in enumerate(sub_datasets):
            # SERVED length, not len(hf_dataset): split real datasets can carry stale
            # extra rows in the loaded HF table, and the shared boundary iterator
            # (_iter_multidataset_episode_ranges) caps to len(sub_dataset) — global
            # offsets and episode order here must match it or the alignment check
            # below refuses a condition the shared path explicitly supports.
            n_frames = len(sub_ds)
            has_source = "source" in sub_ds.features
            source_col = sub_ds.hf_dataset["source"] if has_source else None
            sub_repo_id = repo_ids[sub_idx]

            # Build episode boundary lookup for this sub-dataset
            ep_indices = sub_ds.hf_dataset["episode_index"]
            if hasattr(ep_indices, "tolist"):
                ep_indices = ep_indices.tolist()
            if len(ep_indices) < n_frames:
                raise RuntimeError(
                    f"sub-dataset {sub_repo_id!r} has fewer episode_index rows "
                    f"({len(ep_indices)}) than its served length ({n_frames})"
                )
            ep_indices = ep_indices[:n_frames]
            ep_boundaries = {}
            if n_frames > 0:
                cur_ep = ep_indices[0]
                cur_start = 0
                for i in range(1, n_frames):
                    if ep_indices[i] != cur_ep:
                        ep_boundaries[int(cur_ep)] = (cur_start, i)
                        cur_ep = ep_indices[i]
                        cur_start = i
                ep_boundaries[int(cur_ep)] = (cur_start, n_frames)

            for ep_key, (ep_from, ep_to) in ep_boundaries.items():
                # Anchor end comes from the SHARED clamped boundaries (valid-prefix +
                # done-terminal-tail + soft-truncation chunk-overrun clamps) — the same
                # arrays frame_to_episode is built from. Do NOT re-derive the prefix
                # locally: a re-derivation silently diverges whenever a new clamp is
                # added upstream. to_indices is the sampler
                # 'to' (anchor range = [from, to - drop_n_last)).
                if from_indices[global_ep_i] != global_offset + ep_from:
                    raise ValueError(
                        f"episode alignment broken: global episode {global_ep_i} starts at "
                        f"{from_indices[global_ep_i]} per compute_multidataset_valid_boundaries, "
                        f"but sub-dataset {sub_idx} episode {ep_key} starts at "
                        f"{global_offset + ep_from}"
                    )
                ep_valid_to = to_indices[global_ep_i] - drop_n_last - global_offset
                global_ep_i += 1

                for local_idx in range(ep_from, ep_valid_to):
                    if has_source:
                        source_val = source_col[local_idx]
                        if hasattr(source_val, "item"):
                            source_val = source_val.item()
                        if int(source_val) != DataSource.HUMAN:
                            total_policy_frames += 1
                            continue
                    total_human_frames += 1

                    append_split_index(
                        global_offset + local_idx,
                        frame_to_episode,
                        val_episode_set,
                        train_filtered_indices,
                        val_filtered_indices,
                    )

            global_offset += n_frames

        if global_ep_i != len(from_indices):
            raise ValueError(
                f"episode count mismatch: DAgger filter walked {global_ep_i} episodes but "
                f"compute_multidataset_valid_boundaries returned {len(from_indices)}"
            )

        # Infinite sampler: back-to-back permutation passes so cycle(dataloader)
        # never exhausts the iterator and the DataLoader never drains + rebuilds
        # its prefetch pipeline at epoch boundaries (~3-4 s stall per epoch).
        # Train split only; val keeps the finite sampler.
        sampler = InfiniteSubsetRandomSampler(train_filtered_indices, base_seed=args.seed)
        print(
            f"  Human frames: {total_human_frames}, Policy frames excluded: {total_policy_frames}"
        )
        print(
            f"  Training on {len(train_filtered_indices)} frames (after drop_n_last={drop_n_last})"
        )

        if use_external_eval:
            val_sampler = None
        elif val_enabled and len(val_filtered_indices) >= args.eval_batch_size:
            val_sampler = SubsetRandomSampler(val_filtered_indices)
            print(f"  Validation: {len(val_filtered_indices)} frames")
        elif val_enabled:
            print(
                f"  WARNING: Only {len(val_filtered_indices)} val frames < eval_batch_size "
                f"{args.eval_batch_size}, disabling val"
            )
            val_sampler = None
        else:
            val_sampler = None
    else:
        # Standard sampler (no frame-level filtering)
        sampler = EpisodeAwareSampler(
            dataset_from_indices=from_indices,
            dataset_to_indices=to_indices,
            episode_indices_to_use=train_episodes if val_enabled else None,
            drop_n_last_frames=drop_n_last,
            shuffle=True,
        )

        if val_enabled and not use_external_eval:
            val_sampler = EpisodeAwareSampler(
                dataset_from_indices=from_indices,
                dataset_to_indices=to_indices,
                episode_indices_to_use=val_episodes,
                drop_n_last_frames=drop_n_last,
                shuffle=True,
            )
            if len(val_sampler) < args.eval_batch_size:
                print(
                    f"  WARNING: Only {len(val_sampler)} val frames < eval_batch_size "
                    f"{args.eval_batch_size}, disabling val"
                )
                val_sampler = None
            else:
                print(f"  Validation: {len(val_sampler)} frames")
        else:
            val_sampler = None

    val_dataset_for_loader = dataset
    if val_enabled and use_external_eval:
        if eval_dataset is None or eval_sub_datasets is None:
            raise RuntimeError("External eval was enabled but eval dataset was not loaded")
        eval_from_indices, eval_valid_to_indices, eval_raw_to_indices, _eval_excluded = (
            compute_multidataset_valid_boundaries(eval_sub_datasets)
        )
        eval_to_indices, _eval_overrun_excluded = clamp_soft_truncated_anchor_ends(
            eval_from_indices,
            eval_valid_to_indices,
            eval_raw_to_indices,
            drop_n_last=drop_n_last,
            max_forward_action_offset=max_forward_action_offset,
        )
        val_sampler = EpisodeAwareSampler(
            dataset_from_indices=eval_from_indices,
            dataset_to_indices=eval_to_indices,
            episode_indices_to_use=None,
            drop_n_last_frames=drop_n_last,
            shuffle=True,
        )
        val_dataset_for_loader = eval_dataset
        if len(val_sampler) < args.eval_batch_size:
            print(
                f"  WARNING: Only {len(val_sampler)} external eval frames < eval_batch_size "
                f"{args.eval_batch_size}, disabling val"
            )
            val_sampler = None
        else:
            print(
                f"  External validation: {len(val_sampler)} frames from "
                f"{eval_dataset.num_episodes} episodes"
            )

    train_frames = len(sampler)
    val_frames = len(val_sampler) if val_sampler is not None else 0
    total_frames = dataset.num_frames
    print(
        f"Sampler: {train_frames} train frames, {val_frames} val frames "
        f"({total_frames - train_frames - val_frames} dropped, drop_n_last={drop_n_last})"
    )
    print()

    # 9. Create DataLoaders
    use_workers = args.num_workers > 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
        prefetch_factor=2 if use_workers else None,
        persistent_workers=use_workers,
    )

    if val_sampler is not None:
        val_dataloader = DataLoader(
            val_dataset_for_loader,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device == "cuda",
            drop_last=True,
            prefetch_factor=2 if use_workers else None,
            persistent_workers=use_workers,
        )

    # 10. Create policy
    #
    # Resume precedence (see --no-resume): an auto-detected checkpoint in output_dir
    # ALWAYS wins over both a fresh start and an external --pretrained-path warm-start,
    # so an interrupted job relaunched with the same --output-dir continues identically.
    # The output_dir / checkpoint_root layout mirrors the save block below.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_short = repo_ids[0].split("/")[-1]
    checkpoint_root = output_dir / f"{POLICY_TYPE}_{dataset_short}"

    resume_info = find_latest_resumable_checkpoint(checkpoint_root) if args.resume else None
    resume_state: dict | None = None
    start_step = 1
    resume_wandb_run_id: str | None = None
    frame_cache_regime_mixed = False

    policy_class = get_policy_class(POLICY_TYPE)
    if resume_info is not None:
        resume_ckpt_dir, resume_step = resume_info
        print("=" * 60)
        print(f"RESUMING from step {resume_step} (checkpoint: {resume_ckpt_dir})")
        if args.model_init_seed is not None:
            print(f"  Note: --model-init-seed={args.model_init_seed} ignored while resuming.")
        if args.pretrained_path:
            print(
                f"  Note: a resumable checkpoint exists, so it takes precedence over "
                f"--pretrained-path={args.pretrained_path!r} (external warm-start ignored)."
            )
        print("=" * 60)
        policy = policy_class.from_pretrained(resume_ckpt_dir, config=policy_cfg)
        resume_state = torch.load(
            resume_ckpt_dir / TRAINING_STATE_FILENAME, map_location="cpu", weights_only=False
        )
        if int(resume_state["step"]) != resume_step:
            raise ValueError(
                f"training_state.pt step ({resume_state['step']}) disagrees with checkpoint "
                f"dir step ({resume_step}) in {resume_ckpt_dir}"
            )
        check_resume_precision_regime(resume_state, args)
        frame_cache_regime_mixed = check_resume_frame_cache_state(resume_state, frame_cache_active)
        start_step = resume_step + 1
        resume_wandb_run_id = resume_state.get("wandb_run_id") or None
    elif args.pretrained_path:
        print(f"Loading pretrained model from: {args.pretrained_path}")
        if args.model_init_seed is not None:
            print(f"  Note: --model-init-seed={args.model_init_seed} ignored for pretrained load.")
        policy = policy_class.from_pretrained(args.pretrained_path, config=policy_cfg)
    else:
        if args.model_init_seed is not None:
            print(
                f"Fresh policy model-init seed: {args.model_init_seed} "
                "(training RNG state restored after construction)"
            )
        with _temporary_seed(args.model_init_seed):
            policy = policy_class(config=policy_cfg)
    policy.to(device)
    if args.channels_last:
        # Layout-only: all 4D conv weights (the ResNet image encoders) -> NHWC;
        # activations convert at the first conv and stay NHWC through the backbone.
        # Runs AFTER any resume/pretrained load so restored weights get converted too.
        policy.to(memory_format=torch.channels_last)
        print("Policy conv weights memory format: channels_last (--channels-last)")
    policy.train()

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Diffusion policy created: {num_params:,} parameters")
    if hasattr(policy_cfg, "chunk_size"):
        print(f"  Chunk size: {policy_cfg.chunk_size}")
    elif hasattr(policy_cfg, "horizon"):
        print(f"  Horizon: {policy_cfg.horizon}")
    print(f"  Learning rate: {policy_cfg.optimizer_lr}")
    print()

    # 11. Create preprocessor/postprocessor (uses aggregated stats across all datasets)
    dataset_stats = dataset.stats
    if args.visual_normalization == "imagenet":
        dataset_stats, _imagenet_keys = apply_imagenet_visual_stats(dataset_stats)
        print(
            f"  [visual-norm=imagenet] set canonical ImageNet MEAN_STD on "
            f"{len(_imagenet_keys)} image feature(s): {sorted(_imagenet_keys)}"
        )
    if args.visual_normalization in ("dataset", "imagenet"):
        # Last-line guard: the FINAL image std about to be baked into the normalizer
        # must be physical. Catches the uint8 compute_stats overflow (~0.016 std) -- in
        # 'dataset' mode it inspects the (possibly corrupted) per-dataset std, in 'imagenet'
        # mode it confirms the override took. 'identity' is skipped (images are not std-normalized there).
        _checked_image_keys = assert_image_std_physical(dataset_stats, require_image_keys=False)
        if _checked_image_keys:
            print(
                f"  [image-std-guard] OK: {len(_checked_image_keys)} image feature(s) "
                f"have physical std ({args.visual_normalization} mode)"
            )
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_stats)

    # UMI relative-pose: splice the batch-time relativization IMMEDIATELY BEFORE the
    # NormalizerProcessorStep so the normalizer (with the per-timestep (T,D) stats)
    # sees the relative trajectory. The step fires only when an action chunk is
    # present (training/val), so it is a no-op for eval/deploy batches. It serializes
    # into the saved preprocessor (by class path) so offline recon / eval reconstruct
    # the exact relativization.
    if args.action_mode == "relative":
        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        from mulligan.real.policy.relative_pose import RelativePoseActionProcessorStep

        norm_idx = next(
            (i for i, s in enumerate(preprocessor.steps) if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if norm_idx is None:
            raise RuntimeError(
                "action_mode=relative: no NormalizerProcessorStep found in the diffusion "
                "preprocessor; cannot splice the relativization step. Inspect "
                "make_diffusion_pre_post_processors."
            )
        rel_step = RelativePoseActionProcessorStep(
            n_obs_steps=policy_cfg.n_obs_steps,
            action_dim=tuple(output_features["action"].shape)[0],
        )
        preprocessor.steps.insert(norm_idx, rel_step)
        print(
            f"  [action-mode=relative] spliced RelativePoseActionProcessorStep at index "
            f"{norm_idx} (before NormalizerProcessorStep); proprio anchor=current PROPRIO "
            f"pose (observation.state cartesian_position), targets=command horizon, "
            f"action_dim={rel_step.action_dim}"
        )

    # 12. Create optimizer and LR scheduler from policy presets
    optimizer_bundle = dp_build_optimizer(policy, policy_cfg, args)
    optimizer = optimizer_bundle.optimizer
    lr_scheduler = optimizer_bundle.lr_scheduler
    grad_clip_norm = optimizer_bundle.grad_clip_norm
    print()

    # 12b. Restore full optimizer/scheduler/EMA/RNG state if resuming. Weights were
    # already loaded from the checkpoint in step 10 (for an EMA checkpoint those are the
    # shadow, replaced here by the saved live weights); this puts the optimizer momentum,
    # the cosine LR schedule position, the EMA shadow, and the RNG streams back where
    # the interrupted run left off so the resumed curve matches an uninterrupted one.
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        if lr_scheduler is not None and resume_state["lr_scheduler"] is not None:
            lr_scheduler.load_state_dict(resume_state["lr_scheduler"])
        elif (lr_scheduler is None) != (resume_state["lr_scheduler"] is None):
            raise ValueError(
                "LR scheduler presence mismatch between checkpoint and current config: "
                f"current lr_scheduler={lr_scheduler is not None}, "
                f"checkpoint lr_scheduler={resume_state['lr_scheduler'] is not None}"
            )
        if resume_state.get("ema_shadow") is not None:
            raise ValueError(
                "The resumed checkpoint was trained with an EMA shadow, which this trainer "
                "does not support. Restart with --no-resume."
            )
        restore_rng_state(resume_state["rng"])
        print(
            f"  Restored optimizer/scheduler/RNG state; resuming at step {start_step} "
            f"(lr={optimizer.param_groups[0]['lr']:.2e})"
        )
        print()

    # 13. Run config (logged to W&B with --use-wandb, always saved with every checkpoint)
    all_repo_ids = [*repo_ids, *eval_repo_ids]
    config = {
        "policy": {
            "type": POLICY_TYPE,
            **dataclasses.asdict(policy_cfg),
        },
        "dataset": {
            "repo_ids": repo_ids,
            "eval_repo_ids": eval_repo_ids,
            "dataset_revisions": {
                repo_id: dataset_revision(dataset_revisions, repo_id) for repo_id in all_repo_ids
            },
            # Hub commit each repo was read at (null for local copies read with --no-dataset-sync).
            "dataset_commits": dataset_commits(
                all_repo_ids, dataset_revisions, synced=not args.no_dataset_sync
            ),
            "dataset_episodes": {
                repo_id: selector.to_json() for repo_id, selector in dataset_selectors.items()
            },
            "validation_source": "external_eval_repos" if use_external_eval else "train_val_split",
            "root": str(dataset_root) if dataset_root is not None else None,
            "fps": ds_meta.fps,
            "filter_dagger": args.filter_dagger,
            # Options of the source trainer, fixed in the release (format unchanged).
            "include_autonomous_success": False,
            "autonomous_success_repos": [],
            "autonomous_success_max_length": None,
            "mask_loss_padding": True,
            "training_frames": train_frames,
            "val_frames": val_frames,
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "selected_cameras": selected_camera_keys,
            "excluded_cameras": excluded_camera_keys,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "side_crop_boxes": {cam: list(box) for cam, box in side_crop_map.items()},
        },
        "training": {
            "training_steps": args.training_steps,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "action_target": "cartesian_velocity",
            "cartesian_action_frame": "base",
            "action_mode": args.action_mode,
            "action_norm": "minmax",
            "drop_n_last_frames_flag": args.drop_n_last_frames,
            "crop_shape": None,
            "n_obs_steps": 1,
            "spatial_softmax_num_keypoints": getattr(
                policy_cfg, "spatial_softmax_num_keypoints", None
            ),
            "include_failed_human_segments": False,
            "human_correction_weight": None,
            "save_freq": args.save_freq,
            "eval_freq": args.eval_freq,
            "num_eval_batches": args.num_eval_batches,
            "eval_frame_anchors_per_log": args.eval_batch_size * args.num_eval_batches,
            "val_pct": args.val_pct,
            "grad_clip_norm": grad_clip_norm,
            "seed": args.seed,
            "model_init_seed": args.model_init_seed,
            "pretrained_path": args.pretrained_path,
            # The RESOLVED backbone weights / encoder decision (what was actually built);
            # the raw request is kept under *_requested. visual_normalization is already
            # the resolved value ('auto' is mutated in place upstream).
            "pretrained_backbone_weights": resolved_pretrained_backbone_weights,
            "pretrained_backbone_weights_requested": args.pretrained_backbone_weights,
            "vision_backbone": args.vision_backbone,
            "vision_backbone_lr": None,
            "drop_state": False,
            "visual_normalization": args.visual_normalization,
            "separate_encoders": True,
            "proprio_dropout_prob": 0.0,
        },
        "augmentation": dataclasses.asdict(image_transforms_cfg),
        "system": {
            "device": device,
            "num_workers": args.num_workers,
            "sync_timing": args.sync_timing,
            "uint8_native_images": args.uint8_native_images,
            "uint8_native_tier": uint8_native_tier,
            "channels_last": args.channels_last,
            "decoded_frame_cache": frame_cache_active,
            "decoded_frame_cache_mixed": frame_cache_regime_mixed,
            "fast_reader": fast_reader_enabled_by_env(),
        },
    }

    # Add per-dataset metadata
    for repo_id, info in dataset_info.items():
        safe_name = repo_id.replace("/", "_")
        config["dataset"][safe_name] = {
            "total_episodes": info["total_episodes"],
            "used_episodes": info["used_episodes"],
            "total_frames": info["total_frames"],
        }

    run_name = args.wandb_run_name or f"{POLICY_TYPE}-{repo_ids[0].split('/')[-1]}"
    run_logger = RunLogger(args.use_wandb)
    if resume_wandb_run_id and args.use_wandb:
        # Continue the SAME W&B run so metrics append to one curve across the
        # interruption instead of forking a new run.
        print(f"W&B: resuming run id {resume_wandb_run_id} (resume=allow)")
    run_logger.init(
        project=args.wandb_project,
        name=run_name,
        notes=args.wandb_notes,
        config=config,
        resume_id=resume_wandb_run_id,
    )
    print(
        f"W&B: {'enabled' if args.use_wandb else 'disabled'}"
        + (f" (project: {args.wandb_project})" if args.use_wandb else "")
    )
    print()

    # 14. Training loop
    print("Starting training...")
    print("=" * 60)

    dl_iter = cycle(dataloader)
    val_dl_iter = iter(val_dataloader) if val_dataloader is not None else None
    first_val_done = False

    timer = TrainingTimer(cuda_sync=args.sync_timing)
    cumulative_timings = defaultdict(float)
    training_start_time = time.perf_counter()

    step_ctx = DPStepContext(
        policy=policy,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        preprocessor=preprocessor,
        policy_cfg=policy_cfg,
        device=device,
        selected_camera_keys=selected_camera_keys,
        uint8_native=uint8_native,
        grad_clip_norm=grad_clip_norm,
        args=args,
        timer=timer,
    )

    # NOTE (dataloader order caveat): cycle(dataloader) iterator order cannot be
    # perfectly restored across a resume — on resume we restart the shuffle stream
    # rather than fast-forwarding it. This is acceptable: weights, optimizer momentum,
    # LR schedule position, EMA shadow, and RNG are all restored, so the only
    # difference is which minibatches arrive in the steps right after resume.
    for step in range(start_step, args.training_steps + 1):
        with timer("data_loading"):
            batch = next(dl_iter)

        step_result = dp_train_step(batch, step_ctx, step=step)
        loss = step_result.loss
        loss_dict = step_result.loss_dict
        grad_norm = step_result.grad_norm

        # Logging
        log_metrics: dict = {}
        if step % args.log_freq == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - training_start_time
            avg_step_time = elapsed / step
            remaining_steps = args.training_steps - step
            eta_seconds = avg_step_time * remaining_steps
            eta_h, eta_rem = divmod(int(eta_seconds), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m}m{eta_s:02d}s"
            loss_str = f"step {step}/{args.training_steps}  loss={loss.item():.4f}  lr={lr_current:.2e}  grad_norm={grad_norm:.2f}  ETA={eta_str}"
            if loss_dict is not None:
                if "l1_loss" in loss_dict:
                    loss_str += f"  l1={loss_dict['l1_loss']:.4f}"
                if "kl_loss" in loss_dict:
                    loss_str += f"  kl={loss_dict['kl_loss']:.4f}"
            print(loss_str)
            timer.print_stats(prefix="  ")

            # Capture timing stats before reset
            for k, v in timer.timings.items():
                cumulative_timings[k] += v
            total_cumulative_time = sum(cumulative_timings.values())
            timer.reset()

            log_metrics = {
                "train/loss": loss.item(),
                "train/lr": lr_current,
                "train/grad_norm": grad_norm.item()
                if isinstance(grad_norm, torch.Tensor)
                else grad_norm,
            }
            if loss_dict is not None:
                for k, v in flatten_loss_dict(loss_dict).items():
                    log_metrics[f"train/{k}"] = v

            # Add timing metrics
            log_metrics["timing/eta_hours"] = eta_seconds / 3600
            log_metrics["timing/elapsed_hours"] = elapsed / 3600
            log_metrics["timing/step_time_ms"] = avg_step_time * 1000
            log_metrics["timing_cumulative/total_hours"] = total_cumulative_time / 3600
            for key, total_time in cumulative_timings.items():
                log_metrics[f"timing_cumulative/hours_{key}"] = total_time / 3600
                if total_cumulative_time > 0:
                    log_metrics[f"timing_cumulative/share_{key}"] = (
                        total_time / total_cumulative_time
                    )

        # Validation (independent of --log-freq; logged in the same call on shared steps)
        if args.eval_freq > 0 and step % args.eval_freq == 0 and val_dataloader is not None:
            with timer("validation"):
                # First eval only: save augmented image grid (before preprocessing)
                if not first_val_done:
                    grid, val_dl_iter = save_first_val_image_grid(
                        val_dataloader,
                        selected_camera_keys,
                        output_dir / "visualizations",
                        step,
                        uint8_native=uint8_native,
                    )
                    if grid is not None:
                        pil_img, save_path = grid
                        print(f"  Saved augmented image grid to {save_path}")
                        if args.use_wandb:
                            log_metrics["val/augmented_images"] = run_logger.image(pil_img)
                    first_val_done = True

                # Compute and log val loss
                val_metrics, val_dl_iter = compute_val_loss(
                    val_dataloader,
                    val_dl_iter,
                    policy,
                    preprocessor,
                    policy_cfg,
                    device,
                    args.num_eval_batches,
                    uint8_native=uint8_native,
                    camera_keys=selected_camera_keys,
                )
                val_loss_str = f"  [VAL] step {step}  val_loss={val_metrics['val/loss']:.4f}"
                print(val_loss_str)
                log_metrics.update(val_metrics)

        if log_metrics:
            run_logger.log(log_metrics, step=step)

        # Save checkpoints
        is_save_step = (step % args.save_freq == 0) or step == args.training_steps
        if is_save_step:
            with timer("checkpoint"):
                checkpoint_name = f"checkpoint_{step}" if step < args.training_steps else "final"
                checkpoint_path = checkpoint_root / checkpoint_name

                save_dp_deployment_dir(policy, preprocessor, postprocessor, checkpoint_path)
                metadata = {
                    "policy_type": policy.__class__.__name__,
                    "step": step,
                    "repo_ids": repo_ids,
                    "eval_repo_ids": eval_repo_ids,
                    "dataset_revisions": config["dataset"]["dataset_revisions"],
                    "dataset_commits": config["dataset"]["dataset_commits"],
                    "dataset_episodes": json.loads(selectors_to_json(dataset_selectors)),
                    "dataset_size": dataset.num_frames,
                    "cameras": selected_camera_keys,
                    "chunk_size": getattr(
                        policy_cfg, "chunk_size", getattr(policy_cfg, "horizon", None)
                    ),
                    "n_obs_steps": getattr(policy_cfg, "n_obs_steps", None),
                    "action_target": policy_cfg.action_target,
                    "cartesian_action_frame": policy_cfg.cartesian_action_frame,
                    "action_mode": args.action_mode,
                    "camera_crop_boxes": {
                        cam: list(box) for cam, box in policy_cfg.camera_crop_boxes.items()
                    },
                    # Provenance of the image path (training pixels differ by <=1/255
                    # quantization/device rounding) and the training-forward layout.
                    "uint8_native_images": args.uint8_native_images,
                    "uint8_native_tier": uint8_native_tier,
                    "channels_last": args.channels_last,
                    "decoded_frame_cache": frame_cache_active,
                    "decoded_frame_cache_mixed": frame_cache_regime_mixed,
                    "fast_reader": fast_reader_enabled_by_env(),
                    "run_config": config,
                }
                (checkpoint_path / TRAIN_METADATA_FILENAME).write_text(
                    json.dumps(metadata, indent=2, default=str) + "\n"
                )

                # Persist full resumable training state alongside the weights-only
                # save_pretrained. Only for intermediate checkpoints — the 'final' dir
                # marks a completed run and is intentionally never auto-resumed.
                if step < args.training_steps:
                    save_training_state(
                        checkpoint_path,
                        step=step,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        wandb_run_id=run_logger.run_id or resume_wandb_run_id,
                        precision_regime={
                            "channels_last": args.channels_last,
                        },
                        decoded_frame_cache=frame_cache_active,
                        decoded_frame_cache_mixed=frame_cache_regime_mixed,
                    )
                print(f"\nSaved checkpoint to {checkpoint_path}")

                if args.use_wandb:
                    artifact_name = f"{run_name}-{run_logger.run_id}-{checkpoint_name}"
                    run_logger.log_artifact_dir(
                        checkpoint_path,
                        name=artifact_name,
                        artifact_type="policy",
                        metadata={k: v for k, v in metadata.items() if k != "run_config"},
                        description=f"Diffusion policy checkpoint at step {step} "
                        f"trained on {', '.join(repo_ids)}",
                    )
                    print(f"Uploaded artifact: {artifact_name}")
                print()

    run_logger.finish()
    print("Done!")


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
