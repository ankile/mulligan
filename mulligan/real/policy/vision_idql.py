#!/usr/bin/env python3
"""
Vision IDQL policy for real-world robot rollout.

Combines a trained DiffusionPolicy (encoder + diffusion head) with IQL Q/V
networks for action selection via Q-max over diffusion samples.

Architecture:
    1. Encode images with the DP encoder → diffusion conditioning
    2. Build global conditioning (normalized state + visual features)
    3. Sample N action candidates via diffusion denoising
    4. Encode images with the IQL encoder (separate, possibly fine-tuned) → Q/V state
    5. Score each candidate with Q-networks (z-score normalized state+actions)
    6. Select the best action chunk (argmax Q-min)

The policy loads from a critic checkpoint directory containing:
    - iql_checkpoint.pt: IQL Q1, Q2, V state dicts + config
    - metadata.json: policy_type, dp_artifact reference, network config
plus the DP actor that ``dp_artifact`` names (resolved by :mod:`mulligan.real.policy.dp`).
"""

import json
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from lerobot.configs.types import FeatureType

from mulligan.networks.distributional_v import DistributionalVNetwork
from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.networks.action_film import apply_action_film
from mulligan.real.policy.image_preprocess import preprocess_hwc_rgb_uint8_for_policy

logger = logging.getLogger(__name__)


def load_action_film_head(
    metadata: dict, iql_ckpt: dict, state_dim: int, device: str
) -> tuple[nn.Module | None, int, int]:
    """Action FiLM head: rebuild + load the observation-conditioned
    learned action-scaling head from checkpoint metadata.

    GATED on the metadata flag: non-FiLM checkpoints (flag absent => False)
    return ``(None, 0, 0)`` and the caller's Q-scoring path is unchanged.
    Shared by ``VisionIDQLRealWorldPolicy.from_artifact`` (deploy) and the
    parity tests so the deploy loader has ONE definition.

    Returns:
        (film_head | None, arm_dims, step_dim)
    """
    if not bool(metadata.get("action_film_head", False)):
        return None, 0, 0
    arm_dims = int(metadata["action_film_arm_dims"])
    hidden = int(metadata["action_film_hidden"])
    step_dim = int(metadata["action_film_step_dim"])
    film_head = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 2 * arm_dims),
    ).to(device)
    if "film_head_state_dict" not in iql_ckpt:
        raise ValueError(
            "metadata says action_film_head=True but the checkpoint has no film_head_state_dict"
        )
    film_head.load_state_dict(iql_ckpt["film_head_state_dict"])
    film_head.eval()
    return film_head, arm_dims, step_dim


def _normalize_compiled_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    root_prefix = "_orig_mod."
    nested_marker = "._orig_mod."
    keys = list(state_dict)
    compiled = [key.startswith(root_prefix) or nested_marker in key for key in keys]
    if not any(compiled):
        return state_dict
    if not all(compiled):
        raise ValueError(
            "State dict mixes torch.compile-prefixed and unprefixed keys; refusing "
            "to guess how to load it."
        )
    return {
        key.removeprefix(root_prefix).replace(nested_marker, "."): value
        for key, value in state_dict.items()
    }


def assert_vision_idql_supported_dp_policy_contract(
    dp_config, *, critic_action_mode: str = "absolute"
) -> None:
    """Vision-IDQL supports velocity/base single-stream DP actors, in two pairings:
    an absolute critic with an absolute (velocity) DP, or a relative critic with an
    ``action_mode=relative`` (UMI proprio-anchored) DP.

    The DP actor's camera preprocessing and action contract live in its policy
    config. This wrapper has its own image path, so refuse unsupported config
    fields rather than silently feeding uncropped/base-frame observations. The
    action-mode pairing is checked HERE (not left to the action-dim mismatch) so
    the failure names the actual contract: a relative critic scores 10D relative
    pose steps and cannot rank velocity chunks, and vice versa.
    """
    from mulligan.real.policy.side_crop import reject_dual_side_crop_boxes

    action_target = getattr(dp_config, "action_target", None) or "cartesian_velocity"
    cartesian_action_frame = getattr(dp_config, "cartesian_action_frame", None) or "base"
    if action_target != "cartesian_velocity" or cartesian_action_frame != "base":
        raise NotImplementedError(
            "Vision-IDQL rollout only supports DP actors with "
            "action_target=cartesian_velocity and cartesian_action_frame=base; "
            f"got action_target={action_target!r}, "
            f"cartesian_action_frame={cartesian_action_frame!r}."
        )
    dp_action_mode = getattr(dp_config, "action_mode", None) or "absolute"
    if critic_action_mode not in ("absolute", "relative"):
        raise ValueError(f"Unknown critic action_mode {critic_action_mode!r}")
    if dp_action_mode != critic_action_mode:
        raise NotImplementedError(
            "Vision-IDQL DP/critic action-representation mismatch: DP action_mode="
            f"{dp_action_mode!r} but the critic was trained with action_mode="
            f"{critic_action_mode!r}. A relative (UMI proprio-anchored) critic pairs only "
            "with an --action-mode relative DP, an absolute critic only with a "
            "velocity DP."
        )
    reject_dual_side_crop_boxes(
        getattr(dp_config, "dual_side_crop_boxes", None), context="DP config"
    )


class VisionIDQLRealWorldPolicy:
    """Vision IDQL policy for real-world deployment.

    Implements the same interface as LeRobotRealWorldPolicy used by
    mulligan.real.collect.rollout (predict via select_action, reset, etc.)

    The DP encoder and diffusion head generate action candidates. When the IQL
    was trained with encoder fine-tuning, a separate encoder copy is used for
    Q/V scoring (loaded from the IQL checkpoint). Otherwise the DP encoder is shared.
    """

    action_space = "cartesian_velocity"
    gripper_action_space: str | None = None

    @property
    def config(self):
        """Delegate to DP config for compatibility with resolution auto-detection."""
        return self.dp.config

    @property
    def diffusion(self):
        """Delegate to DP diffusion for compatibility with scheduler overrides."""
        return self.dp.diffusion

    @property
    def camera_crops(self) -> dict[str, tuple[int, int, int, int]]:
        """The DP's per-camera replacement crop boxes (same contract as
        ``RealWorldPolicy.camera_crops``): role-keyed, in stored-frame pixels. Read-only
        copy for operator monitors."""
        return dict(self._camera_crops)

    def __init__(
        self,
        dp: nn.Module,
        dp_preprocessor,
        dp_postprocessor,
        q1: QNetwork,
        q2: QNetwork,
        v_net: VNetwork | DistributionalVNetwork,
        num_action_samples: int = 32,
        camera_keys: list[str] | None = None,
        iql_camera_keys: list[str] | None = None,
        image_height: int = 224,
        image_width: int = 224,
        device: str = "cuda",
        # IQL input normalization (z-score)
        proprio_mean: torch.Tensor | None = None,
        proprio_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        iql_encoder: nn.Module | None = None,
        iql_image_norm_mean: torch.Tensor | None = None,
        iql_image_norm_std: torch.Tensor | None = None,
        iql_encoder_autocast_bf16: bool = False,
        iql_channels_last: bool = False,
        film_head: nn.Module | None = None,
        film_arm_dims: int = 0,
        film_step_dim: int = 0,
        action_mode: str = "absolute",
    ):
        self.dp = dp
        self.dp_preprocessor = dp_preprocessor
        self.dp_postprocessor = dp_postprocessor
        self.q1 = q1
        self.q2 = q2
        self.v_net = v_net
        self.num_action_samples = num_action_samples
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        # Offline analysis can opt in to retaining the most recently sampled
        # raw candidate chunks. Keep this disabled in production so robot
        # inference does not add a GPU->CPU copy on every replan.
        self._capture_candidate_actions = False
        self._last_candidate_action_chunks_raw: np.ndarray | None = None
        # Offline diagnostics may replace diffusion sampling with an exact raw
        # action bank. All observation preprocessing and Q/V scoring below
        # stays on the deployed path.
        self._candidate_action_override_raw: np.ndarray | None = None
        self._capture_value_distribution = False
        self._last_value_distribution: tuple[np.ndarray, np.ndarray] | None = None

        # Separate IQL encoder (fine-tuned independently from DP encoder).
        # If None, falls back to the DP encoder (frozen encoder case).
        self.iql_encoder = iql_encoder
        self._has_separate_iql_encoder = iql_encoder is not None
        # Reproduce the TRAINING-time encoder numerics/layout (checkpoint
        # metadata) so the deployed critic scores the same feature distribution
        # its Q/V heads were trained on.
        self._iql_encoder_autocast_bf16 = iql_encoder_autocast_bf16
        self._iql_channels_last = iql_channels_last
        if self._iql_channels_last and self._has_separate_iql_encoder:
            self.iql_encoder.to(memory_format=torch.channels_last)

        # Mirrored DP image normalization for the separate IQL encoder path
        # (newer checkpoints store it; absent = identity, which
        # matches how older checkpoints were trained). The shared-encoder path
        # reuses DP features that are already normalized upstream.
        if (iql_image_norm_mean is None) != (iql_image_norm_std is None):
            raise ValueError("iql_image_norm_mean and iql_image_norm_std must be provided together")
        self._iql_image_norm_enabled = iql_image_norm_mean is not None
        if self._iql_image_norm_enabled:
            self._iql_image_norm_mean = iql_image_norm_mean.reshape(1, 3, 1, 1).float().to(device)
            self._iql_image_norm_std = iql_image_norm_std.reshape(1, 3, 1, 1).float().to(device)
            if not self._has_separate_iql_encoder:
                raise ValueError(
                    "iql_image_norm_* provided but no separate IQL encoder exists; "
                    "the shared-DP-encoder path is already normalized upstream and "
                    "must not be normalized twice."
                )

        # Action FiLM head: observation-conditioned learned action-scaling
        # applied to the Q-SCORING COPY of the z-scored candidate actions only —
        # NEVER the executed action (the executed chunk is action_chunks_raw
        # [best_idx], untouched). None (every non-FiLM checkpoint) => the
        # Q-scoring path is unchanged.
        self._film_head = film_head.to(device) if film_head is not None else None
        self._film_arm_dims = film_arm_dims
        self._film_step_dim = film_step_dim

        # IQL z-score normalization stats
        self._iql_proprio_mean = proprio_mean.to(device) if proprio_mean is not None else None
        self._iql_proprio_std = proprio_std.to(device) if proprio_std is not None else None
        self._iql_action_mean = action_mean.to(device) if action_mean is not None else None
        self._iql_action_std = action_std.to(device) if action_std is not None else None
        self._normalize_iql_inputs = proprio_mean is not None

        # DP config
        self.n_obs_steps = dp.config.n_obs_steps
        self.n_action_steps = dp.config.n_action_steps
        self.horizon = dp.config.horizon
        self.action_dim = dp.config.action_feature.shape[0]
        assert_vision_idql_supported_dp_policy_contract(dp.config, critic_action_mode=action_mode)

        # UMI-relative (proprio-anchored) BoN: the DP samples NORMALIZED relative-pose chunks;
        # the critic scores the physical relative chunk (per-timestep MIN_MAX
        # un-normalized — the SAME representation the relative trainer's
        # command_pose_chunk_to_relative_torch produced), and the EXECUTED best chunk
        # is composed to absolute 7D euler poses against the generation-time proprio
        # anchor (shared decode_normalized_relative_chunk; held anchor, one decode
        # per chunk) and commanded through DROID's cartesian_position space.
        self.action_mode = action_mode
        if action_mode == "relative":
            from mulligan.real.policy.loader import _extract_action_stat_bounds
            from mulligan.real.policy.relative_pose import POSE_DIM

            if self.n_obs_steps != 1:
                raise NotImplementedError(
                    "action_mode=relative Vision-IDQL assumes n_obs_steps=1 (the "
                    f"held-anchor decode uses the current proprio); got {self.n_obs_steps}."
                )
            if self.action_dim != POSE_DIM:
                raise ValueError(
                    f"relative DP action feature must be {POSE_DIM}D; got {self.action_dim}"
                )
            a_min, a_max = _extract_action_stat_bounds(dp_postprocessor)
            if a_min is None or np.asarray(a_min).ndim != 2:
                raise ValueError(
                    "action_mode=relative requires per-timestep (T,10) MIN_MAX action "
                    "stats in the DP postprocessor; got "
                    f"{None if a_min is None else np.asarray(a_min).shape}."
                )
            self._relative_action_min = np.asarray(a_min, dtype=np.float64)
            self._relative_action_max = np.asarray(a_max, dtype=np.float64)
            # Command absolute poses; instance attrs shadow the velocity class attrs
            # (rollout_episode reads them per-policy via getattr).
            self.action_space = "cartesian_position"
            self.gripper_action_space = "position"
        else:
            self._relative_action_min = None
            self._relative_action_max = None

        from mulligan.real.robot.cameras import (
            STATION_STORED_FRAME_HW,
            require_station_role_image_features,
        )
        from mulligan.real.policy.side_crop import normalize_crop_map

        require_station_role_image_features(
            dp.config.image_features, context="Vision-IDQL DP actor config"
        )

        self._camera_crops = normalize_crop_map(
            getattr(dp.config, "camera_crop_boxes", {}),
            context="DP config camera_crop_boxes",
        )
        self._crop_reference_hw = STATION_STORED_FRAME_HW

        # Encoder info — convert ModuleList → ModuleDict for name-based access
        # (same conversion as load_frozen_encoder_from_dp does at training time)
        dm = dp.diffusion
        dp_camera_key_order = list(dp.config.image_features.keys())

        if isinstance(dm.rgb_encoder, nn.ModuleList):
            # nn.ModuleDict forbids "." in keys, so strip "observation.images." prefix
            dm.rgb_encoder = nn.ModuleDict(
                {
                    key.removeprefix("observation.images."): enc
                    for key, enc in zip(dp_camera_key_order, dm.rgb_encoder, strict=True)
                }
            )
            logger.info(
                f"  Converted encoder ModuleList → ModuleDict: {list(dm.rgb_encoder.keys())}"
            )

        if isinstance(dm.rgb_encoder, nn.ModuleDict):
            first_key = next(iter(dm.rgb_encoder.keys()))
            self.feature_dim_per_camera = dm.rgb_encoder[first_key].feature_dim
            self.separate_encoders = True
        else:
            self.feature_dim_per_camera = dm.rgb_encoder.feature_dim
            self.separate_encoders = False

        # Camera key management — store policy feature keys plus live-read keys.
        self._dp_camera_key_order = (
            dp_camera_key_order  # all DP cameras (observation.images.* format)
        )
        self._dp_raw_camera_key_order = [
            k.removeprefix("observation.images.") for k in dp_camera_key_order
        ]
        iql_camera_key_order = (
            list(iql_camera_keys) if iql_camera_keys else list(dp_camera_key_order)
        )
        self._iql_camera_key_order = iql_camera_key_order
        self._iql_raw_camera_key_order = [
            k.removeprefix("observation.images.") for k in iql_camera_key_order
        ]
        if set(self._iql_raw_camera_key_order) != set(self._dp_raw_camera_key_order):
            raise ValueError(
                "Vision-IDQL IQL camera order must contain the same cameras as the DP "
                f"actor; iql={self._iql_raw_camera_key_order}, "
                f"dp={self._dp_raw_camera_key_order}."
            )
        if len(self._iql_raw_camera_key_order) != len(self._dp_raw_camera_key_order):
            raise ValueError(
                "Vision-IDQL IQL camera order has duplicate or missing cameras; "
                f"iql={self._iql_raw_camera_key_order}, dp={self._dp_raw_camera_key_order}."
            )
        self._configure_camera_keys(camera_keys or dp_camera_key_order)

        # Live camera keys as passed (the rollout loop reads them).
        self.camera_keys = list(camera_keys) if camera_keys else list(dp_camera_key_order)
        self.n_cameras = len(self._obs_camera_keys)
        self.total_visual_dim = self.feature_dim_per_camera * self.n_cameras
        self.state_dim = dp.config.robot_state_feature.shape[0]

        # IQL state dim = visual_features + proprio
        self.iql_state_dim = self.total_visual_dim + self.state_dim
        self.iql_action_dim = self.n_action_steps * self.action_dim
        if self.iql_state_dim != self.q1.state_dim or self.iql_state_dim != self.q2.state_dim:
            raise ValueError(
                "Vision-IDQL Q network state_dim does not match reconstructed camera "
                f"features: policy_iql_state_dim={self.iql_state_dim}, "
                f"q1.state_dim={self.q1.state_dim}, q2.state_dim={self.q2.state_dim}, "
                f"iql_camera_order={self._iql_raw_camera_key_order}."
            )
        if self.iql_action_dim != self.q1.action_dim or self.iql_action_dim != self.q2.action_dim:
            raise ValueError(
                "Vision-IDQL Q network action_dim does not match DP action horizon: "
                f"policy_iql_action_dim={self.iql_action_dim}, "
                f"q1.action_dim={self.q1.action_dim}, q2.action_dim={self.q2.action_dim}."
            )

        # Action queue for chunked execution
        self._action_queue: deque[np.ndarray] = deque()

        # Observation history for n_obs_steps
        self._obs_state_history: deque[torch.Tensor] = deque(maxlen=self.n_obs_steps)
        self._obs_images_history: deque[torch.Tensor] = deque(maxlen=self.n_obs_steps)

        # Debug: save encoder input images only once per episode

        # Get references to the DP's normalizer/unnormalizer steps so we can
        # apply the exact same normalization transforms the DP uses, without
        # reimplementing the formulas manually.
        self._dp_normalizer = self._find_normalizer_step(dp_preprocessor, "preprocessor")
        self._dp_unnormalizer = self._find_normalizer_step(dp_postprocessor, "postprocessor")

        # Monitoring info from last chunk generation
        self.last_chunk_info: dict = {}
        # Per-episode accumulation of every chunk's monitoring info. reset()
        # stashes it to last_episode_chunk_infos (rollout_episode resets the
        # policy before returning); the rollout loop persists the stash as a
        # JSONL sidecar so blind evals produce candidate-level diagnostics.
        self.episode_chunk_infos: list[dict] = []
        self.last_episode_chunk_infos: list[dict] = []

        logger.info(
            f"VisionIDQLRealWorldPolicy initialized: "
            f"n_obs_steps={self.n_obs_steps}, n_action_steps={self.n_action_steps}, "
            f"horizon={self.horizon}, action_dim={self.action_dim}, "
            f"visual_dim={self.total_visual_dim}, state_dim={self.state_dim}, "
            f"iql_state_dim={self.iql_state_dim}, iql_action_dim={self.iql_action_dim}, "
            f"num_samples={self.num_action_samples}"
        )
        logger.info(
            f"  Scheduler: {dm.noise_scheduler.__class__.__name__} ({dm.num_inference_steps} steps)"
        )
        logger.info(f"  DP cameras: {self._dp_camera_key_order}")
        logger.info(f"  IQL camera order: {self._iql_camera_key_order}")
        logger.info(f"  Active cameras: {self._obs_camera_keys}")
        logger.info(f"  Live camera keys: {self._live_camera_keys}")
        logger.info(f"  Camera crops: {self._camera_crops}")
        logger.info(f"  Separate encoders: {self.separate_encoders}")
        logger.info(
            f"  IQL encoder: {'separate (fine-tuned)' if self._has_separate_iql_encoder else 'shared with DP'}"
        )
        logger.info(f"  IQL input normalization: {self._normalize_iql_inputs}")
        if self._normalize_iql_inputs:
            logger.info(f"    proprio_mean={self._iql_proprio_mean.cpu().tolist()}")
            logger.info(f"    proprio_std={self._iql_proprio_std.cpu().tolist()}")
            logger.info(f"    action_mean={self._iql_action_mean.cpu().tolist()}")
            logger.info(f"    action_std={self._iql_action_std.cpu().tolist()}")
        if self._film_head is not None:
            logger.info(
                "  Action FiLM head ACTIVE: observation-conditioned "
                f"action-scaling on the Q-scoring copy only (arm_dims={self._film_arm_dims}, "
                f"step_dim={self._film_step_dim}); executed action unchanged."
            )

    def _critic_action_input(self, action_chunks_raw: torch.Tensor) -> torch.Tensor:
        """Q-scoring copy of the raw candidate chunks (N, n_action_steps, action_dim),
        z-scored with the critic's action stats. The executed chunk is never transformed.
        """
        actions = action_chunks_raw
        if self._normalize_iql_inputs:
            actions = (actions - self._iql_action_mean) / self._iql_action_std
        return actions

    def _film_q_actions(self, actions_flat: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """FiLM-transform the Q-SCORING copy of z-scored flat candidate
        actions, conditioned on the same fused IQL state the Q heads consume.

        Returns ``actions_flat`` unchanged (same object) when the head is absent,
        so every non-FiLM checkpoint's Q inputs are unchanged. NEVER applied
        to the executed action.
        """
        if self._film_head is None:
            return actions_flat
        return apply_action_film(
            actions_flat,
            self._film_head(states),
            arm_dims=self._film_arm_dims,
            step_dim=self._film_step_dim,
        )

    @staticmethod
    def _aggregate_q(q1_vals: torch.Tensor, q2_vals: torch.Tensor) -> torch.Tensor:
        """The (N,) selection score: the pessimistic ``min(Q1, Q2)`` of the twin heads."""
        return torch.min(q1_vals, q2_vals).squeeze(-1)

    def _configure_camera_keys(self, camera_keys: list[str]) -> None:
        """Map live camera keys to DP policy feature keys while preserving DP order.

        A key that already names a DP camera role is read under that name; any other
        key is a live serial key and maps to its station role (``serial_key_to_role``).
        """
        from mulligan.real.robot.cameras import serial_key_to_role

        dp_keys = set(self._dp_raw_camera_key_order)
        mapped_live_by_policy_key: dict[str, str] = {}
        for raw in camera_keys:
            live_key = raw.removeprefix("observation.images.")
            policy_key = live_key if live_key in dp_keys else serial_key_to_role(live_key)
            if policy_key not in dp_keys:
                raise ValueError(
                    f"Camera key {raw!r} maps to policy key {policy_key!r}, which is not "
                    f"in DP camera keys {self._dp_raw_camera_key_order}"
                )
            if policy_key in mapped_live_by_policy_key:
                raise ValueError(
                    f"Multiple live cameras map to policy key {policy_key!r}: "
                    f"{mapped_live_by_policy_key[policy_key]!r} and {live_key!r}"
                )
            mapped_live_by_policy_key[policy_key] = live_key

        missing = [
            key for key in self._dp_raw_camera_key_order if key not in mapped_live_by_policy_key
        ]
        extra = sorted(set(mapped_live_by_policy_key) - dp_keys)
        if missing or extra:
            raise ValueError(
                "Vision-IDQL requires the active cameras to match the DP camera set exactly; "
                f"missing={missing}, extra={extra}, dp={self._dp_raw_camera_key_order}, "
                f"given={camera_keys}"
            )

        self._raw_camera_keys = list(self._dp_raw_camera_key_order)
        self._obs_camera_keys = [f"observation.images.{key}" for key in self._raw_camera_keys]
        self._live_camera_keys = [mapped_live_by_policy_key[key] for key in self._raw_camera_keys]
        self._dp_cam_index_by_raw_key = {key: i for i, key in enumerate(self._raw_camera_keys)}

    @staticmethod
    def _find_normalizer_step(pipeline, name: str):
        """Find the (un)normalizer step in a DP preprocessor/postprocessor pipeline."""
        for step in pipeline.steps:
            if hasattr(step, "_apply_transform") and hasattr(step, "_tensor_stats"):
                return step
        raise ValueError(
            f"Could not find normalizer step in DP {name}. "
            f"Steps: {[type(s).__name__ for s in pipeline.steps]}"
        )

    @torch.no_grad()
    def _generate_and_score_actions(self) -> np.ndarray:
        """Generate action candidates via diffusion, score with Q-networks, return best chunk.

        Returns:
            (n_action_steps, action_dim) numpy array of raw actions for the best chunk
        """
        t0 = time.perf_counter()
        dm = self.dp.diffusion
        N = (
            len(self._candidate_action_override_raw)
            if self._candidate_action_override_raw is not None
            else self.num_action_samples
        )

        # ---- 1. Stack observation history ----
        # State: (1, n_obs_steps, state_dim)
        state_history = (
            torch.stack(list(self._obs_state_history), dim=0).unsqueeze(0).to(self.device)
        )
        # Images: (1, n_obs_steps, n_cameras, C, H, W)
        images_history = (
            torch.stack(list(self._obs_images_history), dim=0).unsqueeze(0).to(self.device)
        )

        batch_size = 1
        n_obs_steps = state_history.shape[1]

        # ---- 2. Encode images (once!) ----
        t_enc = time.perf_counter()
        features_per_cam = []
        features_by_raw_key: dict[str, torch.Tensor] = {}
        for cam_i, raw_key in enumerate(self._raw_camera_keys):
            cam_views = images_history[:, :, cam_i]  # (B, S, C, H, W)
            cam_views_flat = cam_views.reshape(-1, *cam_views.shape[2:])  # (B*S, C, H, W)

            # Apply DP's image normalization before encoding (same transform
            # the DP preprocessor applies during training/inference).
            obs_key = f"observation.images.{raw_key}"
            dp_cam_views = self._dp_normalizer._apply_transform(
                cam_views_flat,
                obs_key,
                FeatureType.VISUAL,
                inverse=False,
            )

            if self.separate_encoders:
                feat = dm.rgb_encoder[raw_key](dp_cam_views)
            else:
                feat = dm.rgb_encoder(dp_cam_views)

            feat_seq = feat.reshape(batch_size, n_obs_steps, -1)
            features_per_cam.append(feat_seq)
            features_by_raw_key[raw_key] = feat_seq

        img_features = torch.cat(features_per_cam, dim=-1)  # (B, S, total_visual_dim)
        t_enc_done = time.perf_counter()

        # img_features: (1, n_obs_steps, total_visual_dim)

        # ---- 3. Build global conditioning for diffusion ----
        # DP convention: [normalized_state, visual_features]
        raw_state_flat = state_history.reshape(batch_size * n_obs_steps, -1)
        normalized_state = self._dp_normalizer._apply_transform(
            raw_state_flat,
            "observation.state",
            FeatureType.STATE,
            inverse=False,
        ).reshape(batch_size, n_obs_steps, -1)

        global_cond_feats = [normalized_state, img_features]
        global_cond = torch.cat(
            global_cond_feats, dim=-1
        )  # (1, n_obs_steps, state_dim + visual_dim)
        global_cond_flat = global_cond.flatten(
            start_dim=1
        )  # (1, n_obs_steps * (state_dim + visual_dim))

        # ---- 4. Sample N trajectories, or read an explicitly locked raw bank. ----
        t_diff = time.perf_counter()
        if self._candidate_action_override_raw is None:
            global_cond_N = global_cond_flat.repeat(N, 1)  # (N, global_cond_dim)
            # conditional_sample returns (N, horizon, action_dim) in NORMALIZED space
            all_actions_norm = dm.conditional_sample(N, global_cond=global_cond_N)

            # Slice to action steps (same as generate_actions)
            start = n_obs_steps - 1
            end = start + self.n_action_steps
            action_chunks_norm = all_actions_norm[:, start:end]  # (N, n_action_steps, action_dim)
            if self.action_mode == "relative":
                # Safety bound (parity with loader._generate_relative_chunk_
                # normalized): clamp the NORMALIZED chunks to [-1,1] BEFORE un-
                # normalization so every candidate — including the executed one —
                # is bounded to the trained per-timestep MIN_MAX range. Then
                # per-timestep un-normalize with the leading window rows of the
                # trained (T,10) bounds (n_obs_steps=1 ⇒ window rows == stat rows
                # 0..n_action_steps-1). The result is the PHYSICAL relative chunk —
                # the same representation the relative critic trained on.
                if not torch.isfinite(action_chunks_norm).all():
                    # clamp() passes NaN through; a non-finite chunk would reach
                    # DROID as a NaN pose target. Fail loud (mirrors the loader).
                    raise RuntimeError(
                        "relative DP emitted a non-finite candidate chunk; refusing "
                        "to decode it into a robot command"
                    )
                action_chunks_norm = action_chunks_norm.clamp(-1.0, 1.0)
                mn = torch.as_tensor(
                    self._relative_action_min[start:end],
                    dtype=action_chunks_norm.dtype,
                    device=action_chunks_norm.device,
                )
                mx = torch.as_tensor(
                    self._relative_action_max[start:end],
                    dtype=action_chunks_norm.dtype,
                    device=action_chunks_norm.device,
                )
                action_chunks_raw = (action_chunks_norm + 1.0) / 2.0 * (mx - mn) + mn
            else:
                action_chunks_raw = self._dp_unnormalizer._apply_transform(
                    action_chunks_norm,
                    "action",
                    FeatureType.ACTION,
                    inverse=True,
                )
        else:
            action_chunks_raw = torch.from_numpy(self._candidate_action_override_raw).to(
                self.device
            )
            if self.action_mode == "relative":
                # Invert the per-timestep MIN_MAX un-normalization above; the normalized
                # chunk only feeds the executed-pose decode, Q scoring reads the raw bank.
                start = n_obs_steps - 1
                end = start + self.n_action_steps
                mn = torch.as_tensor(
                    self._relative_action_min[start:end],
                    dtype=action_chunks_raw.dtype,
                    device=action_chunks_raw.device,
                )
                span = (
                    torch.as_tensor(
                        self._relative_action_max[start:end],
                        dtype=action_chunks_raw.dtype,
                        device=action_chunks_raw.device,
                    )
                    - mn
                )
                if not bool((span > 0).all()):
                    raise ValueError(
                        "relative action stats have a zero-span dim; cannot normalize an "
                        "override candidate bank"
                    )
                action_chunks_norm = (action_chunks_raw - mn) / span * 2.0 - 1.0
            else:
                action_chunks_norm = self._dp_normalizer._apply_transform(
                    action_chunks_raw,
                    "action",
                    FeatureType.ACTION,
                    inverse=False,
                )
        t_diff_done = time.perf_counter()

        # ---- 6. Build IQL state and score with Q-networks ----
        t_q = time.perf_counter()
        # IQL convention: [visual_features, proprio]
        # Use the LAST observation step's features (current state)
        if self._has_separate_iql_encoder:
            # IQL has its own fine-tuned encoder — encode the current frame separately
            iql_features_per_cam = []
            for raw_key in self._iql_raw_camera_key_order:
                cam_i = self._dp_cam_index_by_raw_key[raw_key]
                cam_view = images_history[:, -1, cam_i]  # (B, C, H, W) — last obs step only
                if self._iql_image_norm_enabled:
                    # Mirrored DP normalization (checkpoint-recorded); the IQL
                    # encoder was trained on frames transformed exactly this way.
                    cam_view = (cam_view - self._iql_image_norm_mean) / self._iql_image_norm_std
                if self._iql_channels_last:
                    cam_view = cam_view.contiguous(memory_format=torch.channels_last)
                cam_encoder = (
                    self.iql_encoder[raw_key] if self.separate_encoders else self.iql_encoder
                )
                if self._iql_encoder_autocast_bf16 and cam_view.is_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        feat = cam_encoder(cam_view)
                    feat = feat.float()
                else:
                    feat = cam_encoder(cam_view)
                iql_features_per_cam.append(feat)
            current_visual = torch.cat(iql_features_per_cam, dim=-1)  # (1, total_visual_dim)

        else:
            # Frozen encoder shared with DP — reuse features from step 2, but
            # concatenate in the camera order used when the IQL Q/V heads were trained.
            current_visual = torch.cat(
                [features_by_raw_key[key][:, -1, :] for key in self._iql_raw_camera_key_order],
                dim=-1,
            )

        current_state = state_history[:, -1, :]  # (1, state_dim) — raw

        # Z-score normalize proprio for IQL (if training used normalization)
        if self._normalize_iql_inputs:
            iql_proprio = (current_state - self._iql_proprio_mean) / self._iql_proprio_std
        else:
            iql_proprio = current_state

        iql_state = torch.cat([current_visual, iql_proprio], dim=-1)  # (1, iql_state_dim)
        iql_state_N = iql_state.repeat(N, 1)  # (N, iql_state_dim)

        iql_actions = self._critic_action_input(action_chunks_raw)

        # Flatten action chunks for Q-network input
        actions_flat = iql_actions.reshape(N, -1)  # (N, n_action_steps * action_dim)

        # Action FiLM head: transform the Q-SCORING copy only, conditioned
        # on the same fused state the Q heads consume. Identity (same object) for
        # every non-FiLM checkpoint. Candidate-set diagnostics below deliberately
        # stay on the PRE-film z-scored actions_flat so diversity/centroid metrics
        # keep one meaning across arms. The executed action (action_chunks_raw
        # [best_idx]) is NEVER transformed.
        q_scoring_actions = self._film_q_actions(actions_flat, iql_state_N)

        # Score with both Q-networks and take the pessimistic min.
        q1_vals = self.q1(iql_state_N, q_scoring_actions)  # (N, 1)
        q2_vals = self.q2(iql_state_N, q_scoring_actions)  # (N, 1)
        q_min = self._aggregate_q(q1_vals, q2_vals)  # (N,)

        # Also compute V for monitoring (selection is on Q, not V). A distributional
        # DIVL critic's forward() returns logits, so read its scalar expected value.
        if isinstance(self.v_net, DistributionalVNetwork):
            v_val = self.v_net.expected_value(iql_state)  # (1, 1)
            if self._capture_value_distribution:
                self._last_value_distribution = (
                    self.v_net.atoms.detach().cpu().numpy().copy(),
                    self.v_net.probs(iql_state)[0].detach().cpu().numpy().copy(),
                )
            else:
                self._last_value_distribution = None
        else:
            v_val = self.v_net(iql_state)  # (1, 1)
            self._last_value_distribution = None

        # Select best action
        best_idx = q_min.argmax().item()
        t_q_done = time.perf_counter()

        # ---- 7. Get best chunk (unnormalized) and finalize per action mode ----
        if self.action_mode == "relative":
            # Compose the best NORMALIZED (clamped) relative chunk to ABSOLUTE 7D
            # euler poses against the generation-time proprio anchor (proprio-anchored;
            # anchor held for the whole chunk — same decode as the plain relative
            # DP arm). No [-1,1] clip: that would crush an absolute pose in m/rad;
            # the trained-range bound was already applied on the normalized chunk.
            from mulligan.real.policy.relative_pose import decode_normalized_relative_chunk

            anchor6 = current_state[0, :6].detach().cpu().numpy()
            best_chunk_np = decode_normalized_relative_chunk(
                action_chunks_norm[best_idx].detach().cpu().numpy(),
                anchor6,
                self._relative_action_min,
                self._relative_action_max,
            )  # (n_action_steps, 7) absolute [xyz, rpy, grip]
        else:
            best_chunk_raw = action_chunks_raw[best_idx]  # (n_action_steps, action_dim)
            best_chunk_np = best_chunk_raw.cpu().numpy()
            best_chunk_np = np.clip(best_chunk_np, -1.0, 1.0)
        if self._capture_candidate_actions:
            self._last_candidate_action_chunks_raw = action_chunks_raw.detach().cpu().numpy().copy()
        else:
            self._last_candidate_action_chunks_raw = None

        # Candidate-set diagnostics (all in z-scored action space): diversity of
        # the N sampled chunks, chosen-vs-median Q gap, and how far the argmax
        # pick sits from the candidate centroid (OOD-exploitation read).
        centroid = actions_flat.mean(dim=0, keepdim=True)  # (1, D)
        dists_to_centroid = torch.linalg.vector_norm(actions_flat - centroid, dim=-1)  # (N,)
        mean_centroid_dist = dists_to_centroid.mean().clamp_min(1e-8)
        q_min_median = q_min.median()
        # Candidate motion diagnostics, per action mode. Velocity: raw actions are
        # cartesian velocities, so speed = per-step ||v_xyz|| and net displacement =
        # signed cumsum. Relative: raw actions are displacements-FROM-ANCHOR, so the
        # per-step motion is the first difference (step 0's motion is its
        # displacement from the anchor itself) and the net displacement is simply
        # the LAST step's rel_trans. Same downstream field meanings either way.
        if self.action_mode == "relative":
            rel_trans = action_chunks_raw[:, :, :3]  # (N, T, 3) displacement-from-anchor
            step_disp = torch.cat(
                (rel_trans[:, :1], rel_trans[:, 1:] - rel_trans[:, :-1]), dim=1
            )  # (N, T, 3) per-step motion
            cand_trans_speed = torch.linalg.vector_norm(step_disp, dim=-1).mean(dim=-1)  # (N,)
            cand_cum_xyz_disp = rel_trans[:, -1]  # (N, 3) net displacement from anchor
        else:
            # Mean per-step translational speed of each candidate's executed chunk
            # (raw actions are cartesian velocities; first 3 dims = xyz).
            cand_trans_speed = torch.linalg.vector_norm(action_chunks_raw[:, :, :3], dim=-1).mean(
                dim=-1
            )  # (N,)
            # Per-candidate NET xyz displacement of the executed chunk (signed cumsum
            # of the xyz velocity dims over the n_action_steps executed steps) — a
            # DIRECTION vector, unlike cand_trans_speed which is magnitude-only, so a
            # candidate that commits toward the goal is distinguishable from one that
            # only moves fast.
            cand_cum_xyz_disp = action_chunks_raw[:, :, :3].sum(dim=1)  # (N, 3)
        # Final-step gripper command (last action dim, absolute in both modes) =
        # open/close intent; current EEF position (first 3 proprio dims) from the
        # obs used for scoring, so the displacement can be read relative to goal.
        cand_final_gripper = action_chunks_raw[:, -1, -1]  # (N,)
        current_eef_pos = current_state[0, :3]  # (3,) raw cartesian xyz

        # Store monitoring info
        self.last_chunk_info = {
            "q_min_best": q_min[best_idx].item(),
            "q_min_mean": q_min.mean().item(),
            "q_min_std": q_min.std().item(),
            "q_min_median": q_min_median.item(),
            "chosen_vs_median_q_gap": (q_min[best_idx] - q_min_median).item(),
            "q1_best": q1_vals[best_idx].item(),
            "q2_best": q2_vals[best_idx].item(),
            "v_val": v_val.item(),
            "advantage": q_min[best_idx].item() - v_val.item(),
            "best_idx": best_idx,
            "q_min_all": q_min.detach().cpu().tolist(),
            "q1_all": q1_vals.squeeze(-1).detach().cpu().tolist(),
            "q2_all": q2_vals.squeeze(-1).detach().cpu().tolist(),
            "cand_trans_speed_all": cand_trans_speed.detach().cpu().tolist(),
            "cand_cum_xyz_disp_all": cand_cum_xyz_disp.detach().cpu().tolist(),  # (N, 3)
            "cand_final_gripper_all": cand_final_gripper.detach().cpu().tolist(),  # (N,)
            "eef_pos": current_eef_pos.detach().cpu().tolist(),  # (3,) cartesian xyz
            "cand_action_std": actions_flat.std(dim=0).mean().item(),
            "cand_pairwise_centroid_dist_mean": mean_centroid_dist.item(),
            "argmax_act_outlier": (dists_to_centroid[best_idx] / mean_centroid_dist).item(),
            "time_encode_ms": (t_enc_done - t_enc) * 1000,
            "time_diffusion_ms": (t_diff_done - t_diff) * 1000,
            "time_q_score_ms": (t_q_done - t_q) * 1000,
            "time_total_ms": (t_q_done - t0) * 1000,
            "num_action_samples_effective": N,
        }
        self.episode_chunk_infos.append(
            {"chunk_ordinal": len(self.episode_chunk_infos), **self.last_chunk_info}
        )

        return best_chunk_np

    def set_candidate_action_capture(self, enabled: bool) -> None:
        """Enable exact raw-candidate capture for offline visualization.

        The captured array comes from the deployed sampling/scoring path and has
        shape ``(num_action_samples, n_action_steps, action_dim)``. Capturing is
        intentionally opt-in because it adds a device-to-host copy per replan.
        """
        self._capture_candidate_actions = bool(enabled)
        self._last_candidate_action_chunks_raw = None

    def captured_candidate_action_chunks(self) -> np.ndarray:
        """Return a defensive copy of the last opt-in candidate capture."""
        if not self._capture_candidate_actions:
            raise RuntimeError(
                "candidate action capture is disabled; call "
                "set_candidate_action_capture(True) before inference"
            )
        if self._last_candidate_action_chunks_raw is None:
            raise RuntimeError("no candidate action chunk has been generated since capture enabled")
        return self._last_candidate_action_chunks_raw.copy()

    def set_candidate_action_override(self, chunks_raw: np.ndarray | None) -> None:
        """Use exact raw chunks instead of diffusion samples for offline scoring.

        Passing ``None`` restores normal deployed sampling. The override changes
        candidate generation only; observation preprocessing, encoders, action
        normalization, Q/V evaluation, aggregation, and clipping are unchanged.
        """
        if chunks_raw is None:
            self._candidate_action_override_raw = None
            return
        chunks = np.asarray(chunks_raw, dtype=np.float32)
        expected_tail = (self.n_action_steps, self.action_dim)
        if chunks.ndim != 3 or chunks.shape[1:] != expected_tail:
            raise ValueError(
                "candidate action override must have shape "
                f"(N, {expected_tail[0]}, {expected_tail[1]}), got {chunks.shape}"
            )
        if len(chunks) < 1:
            raise ValueError("candidate action override must contain at least one chunk")
        if not np.all(np.isfinite(chunks)):
            raise ValueError("candidate action override contains non-finite values")
        self._candidate_action_override_raw = np.ascontiguousarray(chunks).copy()

    def set_value_distribution_capture(self, enabled: bool) -> None:
        """Enable exact categorical V-distribution capture for offline analysis.

        Scalar V networks never produce a categorical capture. Consumers that
        require distributional semantics should call
        :meth:`captured_value_distribution`, which fails loudly if the loaded
        checkpoint does not use ``DistributionalVNetwork``.
        """
        self._capture_value_distribution = bool(enabled)
        self._last_value_distribution = None

    def captured_value_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """Return defensive copies of the last captured ``(atoms, probs)``."""
        if not self._capture_value_distribution:
            raise RuntimeError(
                "value distribution capture is disabled; call "
                "set_value_distribution_capture(True) before inference"
            )
        if self._last_value_distribution is None:
            raise RuntimeError(
                "no categorical value distribution has been generated since capture enabled; "
                "the loaded checkpoint may use a scalar V network"
            )
        atoms, probs = self._last_value_distribution
        return atoms.copy(), probs.copy()

    def select_action(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action given preprocessed observation dict.

        This method is called by mulligan.real.collect.rollout. The obs_dict has
        already been through the DP preprocessor (normalized state, images
        on device). However, for IDQL we need raw state for Q-scoring, so
        we handle preprocessing ourselves.

        IMPORTANT: This policy does NOT use the DP preprocessor/postprocessor
        externally. It handles all normalization internally. The caller should
        pass raw observations (unnormalized state, float images in [0,1]).

        Args:
            obs_dict: Dict with keys:
                - "observation.state": (state_dim,) raw state tensor
                - "observation.images.<cam>": (C, H, W) float32 [0,1] tensors

        Returns:
            (action_dim,) action tensor
        """
        # Build state tensor
        state = obs_dict["observation.state"]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)

        # Build stacked images tensor (n_cameras, C, H, W)
        image_tensors = []
        for cam_key in self._obs_camera_keys:
            img = obs_dict[cam_key]
            image_tensors.append(img)
        images = torch.stack(image_tensors, dim=0)  # (n_cameras, C, H, W)

        # Update observation history
        self._obs_state_history.append(state.squeeze(0).cpu())
        self._obs_images_history.append(images.cpu())

        # Pad history if not full yet (repeat first observation)
        while len(self._obs_state_history) < self.n_obs_steps:
            self._obs_state_history.appendleft(self._obs_state_history[0])
            self._obs_images_history.appendleft(self._obs_images_history[0])

        # If action queue is empty, generate new chunk
        if len(self._action_queue) == 0:
            best_chunk = self._generate_and_score_actions()
            # Queue all actions in the chunk
            for i in range(best_chunk.shape[0]):
                self._action_queue.append(best_chunk[i])

        # Pop and return next action
        action = self._action_queue.popleft()
        return torch.from_numpy(action)

    def predict(self, raw_obs: dict) -> np.ndarray:
        """Build obs from raw DROID observation and return clipped numpy action.

        This matches the LeRobotRealWorldPolicy.predict() interface used by
        rollout_episode() in the real-world-droid-teleop branch.

        Args:
            raw_obs: Raw DROID observation dict with keys:
                - "robot_state": {"cartesian_position": [6], "gripper_position": float}
                - "image": {camera_key: BGR(A) uint8 array, ...}

        Returns:
            (action_dim,) numpy array clipped to [-1, 1]
        """
        # Build state vector: [cartesian_position(6), gripper_position(1)] = 7D
        state = np.concatenate(
            [
                np.array(raw_obs["robot_state"]["cartesian_position"], dtype=np.float32),
                np.array([raw_obs["robot_state"]["gripper_position"]], dtype=np.float32),
            ]
        )

        obs_dict = {
            "observation.state": torch.from_numpy(state),
        }

        # Process images with the shared deterministic policy image path: read from the
        # live serial key, write the policy's role feature key.
        for live_key, raw_key, obs_key in zip(
            self._live_camera_keys, self._raw_camera_keys, self._obs_camera_keys, strict=True
        ):
            img = raw_obs["image"][live_key]
            if img.shape[2] == 4:
                img = img[:, :, :3]  # BGRA → BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_chw = preprocess_hwc_rgb_uint8_for_policy(
                img,
                target_hw=(self.image_height, self.image_width),
                crop_box=self._camera_crops.get(raw_key),
                crop_reference_hw=self._crop_reference_hw,
            )
            obs_dict[obs_key] = torch.from_numpy(img_chw)

        action_tensor = self.select_action(obs_dict)
        action = action_tensor.cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        if self.action_mode == "relative":
            # The queued actions are decoded ABSOLUTE 7D euler poses; a [-1,1]
            # clip would crush them (m/rad). Only the absolute gripper command is
            # clipped to its [0,1] range (mirrors loader._predict_relative).
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (7,):
                raise RuntimeError(
                    f"relative decoded pose must be 7D [xyz,rpy,grip]; got {action.shape}"
                )
            action[6] = float(np.clip(action[6], 0.0, 1.0))
            return action
        return np.clip(action, -1.0, 1.0)

    def set_camera_keys(self, camera_keys: list[str]) -> None:
        """Update live camera keys (called by rollout_episode after camera discovery)."""
        self._configure_camera_keys(camera_keys)
        self.camera_keys = list(camera_keys)
        self.reset()

    def reset(self):
        """Reset internal state (action queue + observation history).

        The accumulated per-chunk diagnostics are STASHED to
        ``last_episode_chunk_infos`` rather than dropped: rollout_episode()
        resets the policy immediately after the episode ends, so consumers
        (manifest_eval / rollout sidecar writers) read the stash
        after the reset. The stash is overwritten on every reset, so a
        discarded retry/restart attempt is naturally superseded.
        """
        self.last_episode_chunk_infos = self.episode_chunk_infos
        self._action_queue.clear()
        self._obs_state_history.clear()
        self._obs_images_history.clear()
        self.last_chunk_info = {}
        self.episode_chunk_infos = []
        self._last_candidate_action_chunks_raw = None
        self._candidate_action_override_raw = None
        self._last_value_distribution = None

    def eval(self):
        """Set all components to eval mode."""
        self.dp.eval()
        self.q1.eval()
        self.q2.eval()
        self.v_net.eval()
        if self.iql_encoder is not None:
            self.iql_encoder.eval()
        if self._film_head is not None:
            self._film_head.eval()
        return self

    def to(self, device):
        """Move all components to device."""
        self.device = str(device)
        self.dp = self.dp.to(device)
        self.q1 = self.q1.to(device)
        self.q2 = self.q2.to(device)
        self.v_net = self.v_net.to(device)
        if self.iql_encoder is not None:
            self.iql_encoder = self.iql_encoder.to(device)
        if self._film_head is not None:
            self._film_head = self._film_head.to(device)
        # Move normalizer/unnormalizer stats to new device
        self._dp_normalizer.to(device=device)
        self._dp_unnormalizer.to(device=device)
        # Move IQL normalization stats
        if self._iql_proprio_mean is not None:
            self._iql_proprio_mean = self._iql_proprio_mean.to(device)
            self._iql_proprio_std = self._iql_proprio_std.to(device)
            self._iql_action_mean = self._iql_action_mean.to(device)
            self._iql_action_std = self._iql_action_std.to(device)
        if self._iql_image_norm_enabled:
            self._iql_image_norm_mean = self._iql_image_norm_mean.to(device)
            self._iql_image_norm_std = self._iql_image_norm_std.to(device)
        return self

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        device: str = "cuda",
        camera_keys: list[str] | None = None,
        dp_artifact_override: str | None = None,
        dp_local_dir: str | Path | None = None,
    ) -> "VisionIDQLRealWorldPolicy":
        """Load a VisionIDQLRealWorldPolicy from a downloaded artifact directory.

        The artifact directory must contain:
        - metadata.json: with policy_type, dp_artifact, network config
        - iql_checkpoint.pt: Q1, Q2, V state dicts + config

        Args:
            artifact_dir: Path to the critic checkpoint directory
            device: Device to load to
            camera_keys: Override camera keys from metadata (used by the loader
                when camera discovery happens at the caller level)
            dp_artifact_override: Model id of a DP actor to re-rank instead of the
                critic's trained ``metadata['dp_artifact']``.
            dp_local_dir: Load the DP actor from this local LeRobot checkpoint
                directory. When ``None``, ``metadata['dp_artifact']`` (or the
                override) is resolved to a released checkpoint by
                :func:`mulligan.real.policy.dp.critic_dp_model_id` and downloaded.

        Returns:
            Constructed VisionIDQLRealWorldPolicy ready for inference
        """
        artifact_dir = Path(artifact_dir)

        # 1. Read metadata
        metadata_path = artifact_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        policy_type = metadata["policy_type"]
        if policy_type != "vision_idql":
            raise ValueError(f"Expected policy_type='vision_idql', got '{policy_type}'")

        from mulligan.real.policy.side_crop import reject_dual_side_crop_boxes

        reject_dual_side_crop_boxes(
            metadata.get("dual_side_crop_boxes"), context=f"critic metadata {metadata_path}"
        )
        dp_artifact = metadata["dp_artifact"]
        if dp_artifact_override is not None:
            if dp_local_dir is not None:
                raise ValueError("pass either dp_local_dir or dp_artifact_override, not both")
            # Re-rank a DIFFERENT DP actor than the one this critic was trained on.
            # Sound because the IQL critic uses its OWN fine-tuned encoder for Q-state
            # (loaded from iql_checkpoint below), so only the candidate-action SAMPLER
            # is swapped — the Q sees in-distribution state, just ranks the override
            # actor's samples. The action-dim check below guards format compatibility.
            logger.info(
                f"  DP artifact OVERRIDE: re-ranking {dp_artifact_override!r} "
                f"instead of the critic's trained dp_artifact {dp_artifact!r}"
            )
        num_action_samples = metadata.get("num_action_samples", 32)
        iql_state_dim = metadata["state_dim"]
        iql_action_dim = metadata["action_dim"]
        hidden_dims = metadata["hidden_dims"]
        use_layer_norm = metadata.get("use_layer_norm", True)
        # Use caller-provided camera_keys if given, otherwise fall back to metadata
        if camera_keys is None:
            camera_keys = metadata.get("camera_keys", [])
        image_height = metadata.get("image_height", 224)
        image_width = metadata.get("image_width", 224)
        iql_camera_keys = metadata.get("iql_camera_key_order") or metadata.get("camera_keys", [])

        # IQL input normalization stats (z-score)
        normalize_input = metadata.get("normalize_input", False)
        proprio_mean = None
        proprio_std = None
        action_mean = None
        action_std = None
        if normalize_input:
            proprio_mean = torch.tensor(metadata["proprio_mean"], dtype=torch.float32)
            proprio_std = torch.tensor(metadata["proprio_std"], dtype=torch.float32)
            action_mean = torch.tensor(metadata["action_mean"], dtype=torch.float32)
            action_std = torch.tensor(metadata["action_std"], dtype=torch.float32)

        # Critic-input variants of the source code that no released critic uses: a
        # vision-only critic (inference proprio dropout 1.0, proprio zeroed at scoring),
        # the raw-action transform and the visual-feature LayerNorm. The keys stay in the
        # metadata the trainer writes (at these defaults); a critic that sets one is
        # refused rather than scored with inputs it was not trained on. Inference dropout
        # below 1.0 never changed the deployed scores.
        training_section = metadata.get("training_config", {}).get("training", {})
        proprio_dropout = metadata.get(
            "inference_proprio_dropout",
            training_section.get(
                "inference_proprio_dropout",
                training_section.get("proprio_dropout", 0.0),
            ),
        )
        critic_input_variants = {
            "inference_proprio_dropout": float(proprio_dropout) >= 1.0,
            "action_input_transform": metadata.get("action_input_transform", "none") != "none",
            "visual_input_layernorm": bool(metadata.get("visual_input_layernorm", False)),
        }
        unsupported = sorted(k for k, on in critic_input_variants.items() if on)
        if unsupported:
            raise NotImplementedError(
                f"Critic {artifact_dir} uses critic-input option(s) {unsupported} that are "
                "not part of this release (vision-only proprio masking, raw-action transform, "
                "visual LayerNorm)."
            )

        logger.info(f"Loading Vision IDQL from artifact: {artifact_dir}")
        if dp_artifact_override is not None:
            logger.info(
                f"  DP artifact: {dp_artifact_override} (override; trained with {dp_artifact})"
            )
        else:
            logger.info(f"  DP artifact: {dp_artifact}")
        logger.info(f"  IQL state_dim={iql_state_dim}, action_dim={iql_action_dim}")
        logger.info(f"  hidden_dims={hidden_dims}, layer_norm={use_layer_norm}")
        logger.info(f"  num_action_samples={num_action_samples}")
        logger.info(f"  metadata camera_keys={metadata.get('camera_keys', [])}")
        logger.info(f"  metadata iql_camera_key_order={iql_camera_keys}")
        logger.info(f"  normalize_input={normalize_input}")

        # 2. Load the DP actor (encoder + diffusion head + preprocessor/postprocessor)
        from mulligan.real.policy.dp import checkpoint_dir, critic_dp_model_id, load_dp

        if dp_local_dir is None:
            dp_model_id = critic_dp_model_id(dp_artifact, dp_override=dp_artifact_override)
            logger.info(f"  DP actor: {dp_model_id}")
            dp_local_dir = checkpoint_dir(dp_model_id)
        else:
            logger.info(f"  DP actor from local checkpoint dir: {dp_local_dir}")
        # strict=True: an incomplete DP dir (a torn download, a truncated safetensors)
        # must fail here rather than leave missing tensors randomly initialized.
        dp, dp_preprocessor, dp_postprocessor = load_dp(dp_local_dir, device=device, strict=True)
        dp.eval()
        dp_action_dim = int(dp.config.action_feature.shape[0])
        metadata_n_action_steps = metadata.get("n_action_steps")
        dp_n_action_steps = int(dp.config.n_action_steps)
        if metadata_n_action_steps is not None:
            dp_n_action_steps = int(metadata_n_action_steps)
            # Candidate chunks are sliced from index n_obs_steps - 1 of the predicted
            # horizon, so at most horizon - n_obs_steps + 1 steps are available (the
            # bound apply_diffusion_overrides enforces for the DP arm).
            max_n_action_steps = int(dp.config.horizon) - int(dp.config.n_obs_steps) + 1
            if dp_n_action_steps <= 0 or dp_n_action_steps > max_n_action_steps:
                raise ValueError(
                    "Vision-IQL artifact n_action_steps is incompatible with the DP "
                    f"prediction horizon ({metadata_n_action_steps=}; the DP has "
                    f"horizon={dp.config.horizon}, n_obs_steps={dp.config.n_obs_steps}, "
                    f"so n_action_steps must be in [1, {max_n_action_steps}])."
                )
            dp.config.n_action_steps = dp_n_action_steps
        expected_iql_action_dim = dp_n_action_steps * dp_action_dim
        if iql_action_dim != expected_iql_action_dim:
            raise ValueError(
                "Vision-IQL Q action dimension is incompatible with the DP policy "
                f"execution horizon (metadata action_dim={iql_action_dim}, expected "
                f"{dp_n_action_steps} * {dp_action_dim} = {expected_iql_action_dim})."
            )
        logger.info(
            "  horizons: "
            f"prediction={metadata.get('prediction_horizon', dp.config.horizon)}, "
            f"n_action_steps={dp_n_action_steps}"
        )

        # 3. Load IQL checkpoint
        iql_ckpt_path = artifact_dir / "iql_checkpoint.pt"
        iql_ckpt = torch.load(iql_ckpt_path, map_location=device, weights_only=False)

        # 4. Reconstruct Q/V networks
        q1 = QNetwork(
            state_dim=iql_state_dim,
            action_dim=iql_action_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        ).to(device)

        q2 = QNetwork(
            state_dim=iql_state_dim,
            action_dim=iql_action_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        ).to(device)

        # DIVL critics learn a CATEGORICAL V (num_atoms logits + an `atoms` buffer),
        # so a scalar VNetwork can't load their v_state_dict (shape [num_atoms] vs [1] +
        # unexpected `atoms` key). Rebuild the matching network from the persisted
        # distributional metadata (the critic trainer writes distributional/num_atoms/v_min/
        # v_max for exactly this). V is used only for monitoring here (the action
        # re-rank selects on Q), so the scalar readout is its expected value.
        if bool(metadata.get("distributional", False)):
            v_net = DistributionalVNetwork(
                state_dim=iql_state_dim,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
                num_atoms=int(metadata["num_atoms"]),
                v_min=float(metadata["v_min"]),
                v_max=float(metadata["v_max"]),
            ).to(device)
        else:
            v_net = VNetwork(
                state_dim=iql_state_dim,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
            ).to(device)

        q1.load_state_dict(iql_ckpt["q1_state_dict"])
        q2.load_state_dict(iql_ckpt["q2_state_dict"])
        v_net.load_state_dict(iql_ckpt["v_state_dict"])

        q1.eval()
        q2.eval()
        v_net.eval()

        logger.info(f"  Loaded IQL networks (step {iql_ckpt.get('step', '?')})")

        # Action FiLM head: gated on the metadata flag; non-FiLM
        # checkpoints get (None, 0, 0) and an unchanged Q-scoring path.
        film_head, film_arm_dims, film_step_dim = load_action_film_head(
            metadata, iql_ckpt, int(iql_state_dim), device
        )
        if film_head is not None:
            logger.info(
                "  Action FiLM head loaded from checkpoint "
                f"(arm_dims={film_arm_dims}, step_dim={film_step_dim})"
            )

        # 5. If the checkpoint has a fine-tuned encoder, create a SEPARATE copy
        #    for IQL scoring. The DP encoder must remain untouched for diffusion.
        iql_encoder = None
        if "encoder_state_dict" in iql_ckpt:
            import copy

            logger.info("  Creating separate IQL encoder from fine-tuned checkpoint")
            # Deep-copy the DP encoder as a starting point (gets architecture right)
            dm = dp.diffusion
            if isinstance(dm.rgb_encoder, nn.ModuleList):
                # Convert to ModuleDict first so the copy has named keys
                dp_camera_key_order = list(dp.config.image_features.keys())
                dm.rgb_encoder = nn.ModuleDict(
                    {
                        key.removeprefix("observation.images."): enc
                        for key, enc in zip(dp_camera_key_order, dm.rgb_encoder, strict=True)
                    }
                )
                logger.info(
                    f"  Converted DP encoder ModuleList → ModuleDict: {list(dm.rgb_encoder.keys())}"
                )
            iql_encoder = copy.deepcopy(dm.rgb_encoder)
            encoder_state_dict = _normalize_compiled_state_dict_keys(iql_ckpt["encoder_state_dict"])
            iql_encoder.load_state_dict(encoder_state_dict)
            iql_encoder.eval()
            iql_encoder.to(device)
            n_params = sum(p.numel() for p in iql_encoder.parameters())
            logger.info(f"  IQL encoder loaded ({n_params:,} params, separate from DP encoder)")

        # Image normalization for the IQL encoder path: recorded by newer
        # training runs; absent in older checkpoints (trained on raw [0,1]
        # frames), where identity is the CORRECT match to their training.
        iql_image_norm = iql_ckpt.get("image_normalization")
        iql_image_norm_mean = iql_image_norm_std = None
        if iql_image_norm is not None:
            iql_image_norm_mean = torch.as_tensor(iql_image_norm["mean"], dtype=torch.float32)
            iql_image_norm_std = torch.as_tensor(iql_image_norm["std"], dtype=torch.float32)
            logger.info(
                "  IQL image normalization active (mirrored from DP): "
                f"mean={iql_image_norm_mean.flatten().tolist()}, "
                f"std={iql_image_norm_std.flatten().tolist()}"
            )
        else:
            logger.info(
                "  IQL image normalization: none recorded in checkpoint "
                "(older run) — raw [0,1] frames, matching its training"
            )

        # Training-time encoder numerics/layout (throughput-flag
        # checkpoints); absent in older artifacts = fp32/NCHW, which matches
        # how those critics were trained.
        iql_encoder_autocast_bf16 = bool(metadata.get("encoder_autocast_bf16", False))
        iql_channels_last = bool(metadata.get("channels_last", False))
        if iql_encoder_autocast_bf16 or iql_channels_last:
            logger.info(
                "  IQL encoder numerics from metadata: "
                f"bf16-autocast={iql_encoder_autocast_bf16}, "
                f"channels_last={iql_channels_last}"
            )

        return cls(
            dp=dp,
            dp_preprocessor=dp_preprocessor,
            dp_postprocessor=dp_postprocessor,
            q1=q1,
            q2=q2,
            v_net=v_net,
            num_action_samples=num_action_samples,
            camera_keys=camera_keys,
            iql_camera_keys=iql_camera_keys,
            image_height=image_height,
            image_width=image_width,
            device=device,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            action_mean=action_mean,
            action_std=action_std,
            iql_encoder=iql_encoder,
            iql_image_norm_mean=iql_image_norm_mean,
            iql_image_norm_std=iql_image_norm_std,
            iql_encoder_autocast_bf16=iql_encoder_autocast_bf16,
            iql_channels_last=iql_channels_last,
            film_head=film_head,
            film_arm_dims=film_arm_dims,
            film_step_dim=film_step_dim,
            # Critic action representation (absent in older checkpoints =
            # absolute, matching how they were trained). "relative" selects the
            # UMI proprio-anchored candidate representation + absolute-pose execution and
            # is contract-checked against the DP's own action_mode in __init__.
            action_mode=metadata.get("action_mode", "absolute"),
        )
