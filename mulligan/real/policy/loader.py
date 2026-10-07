"""Load real-robot policies (LeRobot diffusion policies and Vision-IDQL critics) for rollout.

Model ids are resolved by :mod:`mulligan.real.policy.dp` (``hf://``, local directories,
optional ``wandb://``). The eval-loop helpers live in :mod:`mulligan.real.eval.common`.
"""

import json
import logging

import cv2
import numpy as np
import torch

import mulligan.real.policy.lerobot_patches  # noqa: F401  (h264 video codec)
from mulligan.real.eval.common import PolicyEntry
from mulligan.real.policy.dp import (
    REAL_PROTOCOL_N_ACTION_STEPS,
    checkpoint_dir,
    load_dp,
    resolve_model_id,
)
from mulligan.real.robot.cameras import (
    STATION_CAMERA_DEFAULT_CROPS,
    STATION_STORED_FRAME_HW,
    require_station_role_image_features,
    serial_key_to_role,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RealWorldPolicy protocol & wrappers
# ---------------------------------------------------------------------------


def _extract_action_stat_bounds(postprocessor):
    """Return ``(min, max)`` float32 numpy vectors for the ``action`` MIN_MAX stats.

    The LeRobot postprocessor (an Unnormalize step built from the checkpoint's
    ``stats["action"]``) holds the per-dim min/max used to un-normalize the policy
    head. The relative-pose decode un-normalizes its chunk with them. Returns
    ``(None, None)`` when no action min/max is present (any non-MIN_MAX stat); the
    relative arm refuses such a checkpoint at construction.

    Defensive but loud: a malformed stats container raises rather than guessing.
    """
    # The postprocessor is a PolicyProcessorPipeline; the action Unnormalize step
    # carries the MIN_MAX stats. Search the pipeline's steps for a step exposing
    # stats["action"]; also handle a bare step passed directly.
    candidates = []
    direct = getattr(postprocessor, "stats", None)
    if direct is not None:
        candidates.append(direct)
    for step in getattr(postprocessor, "steps", []) or []:
        step_stats = getattr(step, "stats", None)
        if step_stats is not None:
            candidates.append(step_stats)

    for stats in candidates:
        if not stats or "action" not in stats:
            continue
        action_stats = stats["action"]
        if not action_stats or "min" not in action_stats or "max" not in action_stats:
            continue
        a_min = np.asarray(action_stats["min"], dtype=np.float32)
        a_max = np.asarray(action_stats["max"], dtype=np.float32)
        # 1-D ``(D,)`` for the per-frame velocity arm; 2-D ``(T, D)`` per-timestep for
        # the UMI relative-pose arm (``action_mode="relative"``). Both are returned
        # verbatim. A higher-rank stat is a corrupt normalizer.
        if a_min.ndim == 1:
            a_min = a_min.reshape(-1)
            a_max = a_max.reshape(-1)
        elif a_min.ndim != 2:
            raise ValueError(
                f"action stat has unexpected ndim {a_min.ndim} (shape {a_min.shape}); "
                "expected 1-D (D,) per-frame or 2-D (T,D) per-timestep stats."
            )
        if a_min.shape != a_max.shape:
            raise ValueError(f"action stat min/max shapes differ: {a_min.shape} vs {a_max.shape}")
        if np.any(a_max < a_min):
            raise ValueError("action stat max < min on some dim; corrupt normalizer.")
        return a_min, a_max
    return None, None


class LeRobotRealWorldPolicy:
    """Wraps a LeRobot policy + preprocessor + postprocessor into the unified
    RealWorldPolicy interface expected by rollout_episode().

    Handles:
    - Building the obs dict from raw DROID observations (state + images)
    - Running preprocessor → policy.select_action → postprocessor
    - Converting output to clipped numpy array

    The policy's image features are station ROLE names (``observation.images.side_1``);
    each live camera is read by its serial key and fed under its role
    (:func:`serial_key_to_role`). A policy with any other image feature name is refused.
    """

    action_space: str = "cartesian_velocity"
    gripper_action_space: str | None = None  # None = DROID default (matches arm space)

    def __init__(
        self,
        policy: torch.nn.Module,
        preprocessor,
        postprocessor,
        camera_keys: list[str],
        camera_height: int = 480,
        camera_width: int = 640,
        camera_crops: dict[str, tuple[int, int, int, int]] | None = None,
        action_target: str = "cartesian_velocity",
        cartesian_action_frame: str = "base",
        action_mode: str = "absolute",
    ):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        # LIVE serial camera keys; build_processed_obs maps each to the policy's role
        # feature/crop-lookup key (e.g. "<serial>_left" -> "side_1").
        self._camera_keys = camera_keys
        require_station_role_image_features(
            getattr(getattr(policy, "config", None), "image_features", {}) or {},
            context="LeRobot policy config",
        )
        # Datasets store frames at 640x480 (downscaled from the 1280x720 native ZED feed),
        # so the role-keyed crop boxes live in this space; the live native frame is
        # downscaled into it before cropping.
        self._crop_reference_hw = STATION_STORED_FRAME_HW
        self._camera_height = camera_height
        self._camera_width = camera_width
        # Per-camera static ROI crop: {role: (x0, y0, x1, y1)} in stored-frame pixels,
        # applied BEFORE resize so eval reproduces the cropped-then-resized tensor the
        # policy trained on. Empty = no crop.
        self._camera_crops = camera_crops or {}

        # Action contract stored in the checkpoint config. Every released DP actor is a
        # base-frame cartesian_velocity policy (``action_mode`` absolute or relative); the
        # absolute-pose targets (cartesian_position / _r6) and the eef-frame velocity of
        # the source code are not in this release, so a checkpoint that declares one is
        # refused here rather than executed as base-frame velocity.
        if action_target != "cartesian_velocity" or cartesian_action_frame != "base":
            raise NotImplementedError(
                "This release deploys DP actors with action_target='cartesian_velocity' and "
                f"cartesian_action_frame='base'; the checkpoint declares "
                f"action_target={action_target!r}, cartesian_action_frame="
                f"{cartesian_action_frame!r}."
            )
        self._action_min, self._action_max = _extract_action_stat_bounds(postprocessor)

        # UMI relative-pose deploy. The policy emits a RELATIVE pose chunk (per-timestep
        # (T,10) normalized); the deploy path composes it back to ABSOLUTE poses against
        # the GENERATION-TIME anchor pose, held across all n_action_steps pops, then
        # commands DROID's cartesian_position space.
        if action_mode not in ("absolute", "relative"):
            raise ValueError(f"Unknown action_mode {action_mode!r}")
        self.action_mode = action_mode
        if action_mode == "relative":
            if self._action_min is None or np.asarray(self._action_min).ndim != 2:
                raise ValueError(
                    "action_mode=relative requires per-timestep (T,10) MIN_MAX action stats "
                    "in the postprocessor; got "
                    f"{None if self._action_min is None else np.asarray(self._action_min).shape}."
                )
            # Command absolute poses through DROID's cartesian_position space.
            self.action_space = "cartesian_position"
            self.gripper_action_space = "position"
            # The LIVE relative-pose rollout is wired in predict(): per chunk it captures
            # the generation-time anchor (current proprio pose), generates the NORMALIZED
            # relative chunk via the policy's deployment chunk API (predict_action_chunk),
            # composes it to ABSOLUTE poses against that single held anchor
            # (decode_relative_chunk_to_absolute), and pops one absolute pose per step
            # through DROID's cartesian_position space. The per-pop postprocessor is
            # BYPASSED (the per-timestep (T,10) normalizer can't un-normalize a single
            # popped action; the decode does the per-timestep un-normalization itself).
            # The chunk path requires the deployment chunk API and the single-obs-step
            # window the relative arm trains with (the held-anchor decode assumes the
            # anchor is the current proprio pose; n_obs_steps>1 would need obs history
            # the live path does not assemble here).
            if not hasattr(policy, "predict_action_chunk"):
                raise RuntimeError(
                    "action_mode=relative live rollout needs policy.predict_action_chunk "
                    f"(the lerobot deployment chunk API); {type(policy).__name__} lacks it."
                )
            n_obs = int(getattr(policy.config, "n_obs_steps", 1))
            if n_obs != 1:
                raise NotImplementedError(
                    f"action_mode=relative live rollout assumes n_obs_steps=1; got {n_obs}. "
                    "The held-anchor chunk path does not assemble multi-step obs history."
                )
            logger.info(
                "action_mode=relative: live relative-pose rollout enabled "
                "(held-anchor chunk decode -> DROID cartesian_position)."
            )
        # Per-chunk decoded absolute-pose queue for the relative arm (filled at chunk
        # generation, drained one pose per step); empty for all other action modes.
        self._relative_pose_queue: list[np.ndarray] = []
        # No Mulligan-side pose clamps for the relative arm: the decoded absolute pose is
        # commanded to DROID as-is. Hardware safety (reachability, joint/velocity/torque
        # limits, collision reflexes) is DROID's + the Franka's responsibility — the same
        # trust the velocity arm places in DROID. The only bound kept is the [-1,1] clip on
        # the NORMALIZED chunk (in _generate_relative_chunk_normalized), which mirrors the
        # velocity path clipping its normalized action to [-1,1] — a training-range bound,
        # not a hardware guard.

        # Check if this policy needs joint positions
        self._use_joint_positions = (
            hasattr(policy.config, "input_features")
            and "observation.state.joint_position" in policy.config.input_features
        )

        # The deploy state is the 7D [cartesian_position(6), gripper(1)] every released DP
        # actor trained on. The source code's 13D EE-velocity state is not in this release.
        _state_ft = (getattr(policy.config, "input_features", {}) or {}).get("observation.state")
        _state_dim = int(_state_ft.shape[0]) if _state_ft is not None else 7
        if _state_dim != 7:
            raise NotImplementedError(
                f"observation.state width {_state_dim} unsupported at deploy; this release "
                "deploys the 7D state (cart_pos + gripper)."
            )

    @property
    def config(self):
        """Expose underlying policy config for introspection (e.g., image_features)."""
        return self._policy.config

    @property
    def camera_crops(self) -> dict[str, tuple[int, int, int, int]]:
        """The per-camera replacement crop boxes this policy applies at inference.

        Keyed by camera ROLE, in stored 640x480 pixels. Read-only copy for operator
        monitors (``render_camera_monitor``) so on-screen crops match the policy's actual
        view.
        """
        return dict(self._camera_crops)

    @property
    def env_action_space(self) -> str:
        """The DROID RobotEnv action_space string this policy commands.

        ``cartesian_velocity``, or ``cartesian_position`` for the relative-pose arm (its
        chunk is decoded to absolute 7D euler poses in :meth:`predict`).
        """
        return self.action_space

    def set_camera_keys(self, camera_keys: list[str]) -> None:
        """Update the camera keys used for inference (called by rollout_episode
        after camera discovery on first step)."""
        self._camera_keys = list(camera_keys)

    def reset(self) -> None:
        self._policy.reset()
        self._relative_pose_queue = []

    def _crop_and_resize_native_rgb(self, cam_key: str, img: np.ndarray) -> np.ndarray:
        """Apply this arm's crop (if any) then resize a single NATIVE-resolution
        HWC-RGB-uint8 frame to the policy resolution, returning CHW float32 [0,1].

        This is the per-camera image pipeline the policy was trained on: crop the
        raw frame BEFORE resize (mirroring the training data path), then resize to
        the auto-detected checkpoint resolution. Shared by ``predict`` (live robot)
        and the offline encoder probes so both route every arm through its OWN
        crop/resolution — never a silent 224/no-crop fallback.
        """
        from mulligan.real.policy.image_preprocess import preprocess_hwc_rgb_uint8_for_policy

        return preprocess_hwc_rgb_uint8_for_policy(
            img,
            target_hw=(self._camera_height, self._camera_width),
            crop_box=self._camera_crops.get(cam_key),
            crop_reference_hw=self._crop_reference_hw,
        )

    def build_processed_obs(
        self,
        state: np.ndarray,
        native_rgb_images: dict[str, np.ndarray],
        joint_positions: np.ndarray | None = None,
    ) -> dict:
        """Build the normalized obs dict the policy conditions on, from NATIVE-res
        HWC-RGB-uint8 frames (one per camera in ``self._camera_keys``).

        Applies each camera's crop+resize (``_crop_and_resize_native_rgb``) then
        the LeRobot ``_preprocessor`` (VISUAL MEAN_STD normalization), exactly as
        the live ``predict`` path — so every consumer (live robot, offline encoder
        probe) sees the identical arm-specific transform. Does NOT run inference.
        """
        obs_dict = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))}
        if self._use_joint_positions:
            if joint_positions is None:
                raise KeyError(
                    "policy requires observation.state.joint_position but no "
                    "joint_positions were provided to build_processed_obs"
                )
            obs_dict["observation.state.joint_position"] = torch.from_numpy(
                np.asarray(joint_positions, dtype=np.float32)
            )
        declared_physical = set(getattr(self._policy.config, "image_features", {}) or {})
        built_physical: set[str] = set()
        for cam_key in self._camera_keys:
            if cam_key not in native_rgb_images:
                raise KeyError(
                    f"build_processed_obs missing native frame for camera {cam_key!r}; "
                    f"got {sorted(native_rgb_images.keys())}"
                )
            # The LIVE frame is read by its serial cam_key; the FEATURE key and the
            # crop-lookup key are the camera's ROLE, so the policy is fed its
            # observation.images.<role> input cropped with the role's box.
            feat_key = serial_key_to_role(cam_key)
            full_key = f"observation.images.{feat_key}"
            # A SHARED multi-arm capture may provide MORE cameras than this policy
            # consumes (e.g. a stereo arm's wrist_right captured for the session, but
            # this is a {wrist_left, side_1} arm). Build ONLY the cameras this policy
            # declares — feeding an undeclared camera would inject a key its normalizer
            # and encoder never saw. When the policy declares no image features, nothing
            # is skipped.
            if declared_physical and full_key not in declared_physical:
                continue
            img_chw = self._crop_and_resize_native_rgb(feat_key, native_rgb_images[cam_key])
            obs_dict[full_key] = torch.from_numpy(img_chw)
            built_physical.add(full_key)
        # A declared physical camera that the capture never provided is a hard
        # misconfiguration (the encoder would be missing a required input) — fail loud
        # rather than silently run inference on a partial observation.
        if declared_physical and built_physical != declared_physical:
            raise KeyError(
                "build_processed_obs did not build every declared physical camera: missing "
                f"{sorted(declared_physical - built_physical)} (captured cameras: "
                f"{sorted(self._camera_keys)})"
            )
        return self._preprocessor(obs_dict)

    def _build_state_vector(self, raw_obs: dict) -> np.ndarray:
        """The 7D ``[cartesian_position(6 euler), gripper(1)]`` observation.state."""
        return np.concatenate(
            [
                np.array(raw_obs["robot_state"]["cartesian_position"], dtype=np.float32),
                np.array([raw_obs["robot_state"]["gripper_position"]], dtype=np.float32),
            ]
        )

    def predict(self, raw_obs: dict) -> np.ndarray:
        """Build obs dict, run inference, return clipped numpy action."""
        state = self._build_state_vector(raw_obs)

        joint_positions = None
        if self._use_joint_positions:
            joint_positions = np.array(
                raw_obs["robot_state"]["joint_positions"],
                dtype=np.float32,
            )

        # Convert each camera's native frame from BGR(A) uint8 -> RGB uint8, then
        # route through the shared crop/resize/normalize pipeline.
        native_rgb_images = {}
        for cam_key in self._camera_keys:
            if "image" not in raw_obs:
                raise KeyError(
                    "LeRobotRealWorldPolicy observation is missing raw_obs['image']; "
                    f"available top-level keys: {sorted(raw_obs.keys())}"
                )
            if cam_key not in raw_obs["image"]:
                raise KeyError(
                    f"LeRobotRealWorldPolicy observation is missing camera {cam_key!r}; "
                    f"available cameras={sorted(raw_obs['image'].keys())}"
                )
            img = raw_obs["image"][cam_key]
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            native_rgb_images[cam_key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        processed_obs = self.build_processed_obs(state, native_rgb_images, joint_positions)

        # UMI relative-pose arm: chunk-level path (capture anchor -> generate chunk ->
        # decode to absolute against the HELD anchor -> queue -> pop). Must run BEFORE
        # the per-action select_action/postprocessor path, which the per-timestep (T,10)
        # normalizer cannot drive one popped action at a time. anchor = the current
        # proprio pose (state[:6]), captured once per chunk refill and held across pops.
        if getattr(self, "action_mode", "absolute") == "relative":
            return self._predict_relative(processed_obs, anchor_pose6=state[:6])

        with torch.no_grad():
            action_tensor = self._policy.select_action(processed_obs)

        # Un-normalize: MIN_MAX maps the policy head's [-1,1] output back to the
        # DROID-normalized velocity command.
        action_tensor = self._postprocessor(action_tensor)

        action = action_tensor.cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)

        return self._finalize_action(action)

    def _generate_relative_chunk_normalized(self, processed_obs: dict) -> np.ndarray:
        """Generate the NORMALIZED relative-pose chunk via the deployment chunk API.

        Mirrors the underlying policy's own ``select_action`` online preamble (stack the
        per-camera images into ``OBS_IMAGES`` then ``populate_queues`` so the obs carry
        the ``n_obs_steps`` time axis ``generate_actions`` expects) and calls the public
        ``predict_action_chunk`` — the SAME chunk ``select_action`` pops from internally —
        so the obs handling matches deployment. Returns the executed window
        ``(n_action_steps, 10)`` in the policy's own (per-timestep) NORMALIZED space; the
        per-action postprocessor is intentionally NOT applied (the per-timestep (T,10)
        normalizer cannot un-normalize a single action — ``decode_relative_chunk_to_
        absolute`` does the per-timestep un-normalization on the whole chunk).
        """
        from lerobot.policies.utils import populate_queues
        from lerobot.utils.constants import ACTION, OBS_IMAGES

        pol = self._policy
        cfg = pol.config
        batch = dict(processed_obs)
        # The preprocessor's transition->batch converter (transition_to_batch) ALWAYS
        # injects ACTION=None for an obs-only input (there is no action at obs time).
        # populate_queues would push that None into the policy's ACTION deque, and
        # predict_action_chunk's online branch then torch.stack()s the queues -> the
        # None action crashes the stack. DiffusionPolicy.select_action pops ACTION
        # before populate_queues for exactly this reason; mirror it here (the obs-only
        # reward/done/truncated/info keys are harmless — they are not in _queues, so
        # populate_queues + the online-branch comprehension both ignore them).
        batch.pop(ACTION, None)
        if cfg.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in cfg.image_features], dim=-4)
        # populate_queues adds the current obs to the policy's n_obs_steps queues so
        # predict_action_chunk's online branch stacks the time axis (n_obs_steps=1 here,
        # asserted at construction). We drive ONLY the obs queues + the chunk generator;
        # the policy's internal ACTION queue is never popped (we keep our own pose queue).
        pol._queues = populate_queues(pol._queues, batch)
        with torch.no_grad():
            chunk = pol.predict_action_chunk(batch)  # (B, n_action_steps, 10) normalized
        chunk_np = chunk.detach().cpu().numpy()
        if chunk_np.ndim != 3 or chunk_np.shape[0] != 1:
            raise RuntimeError(
                f"relative chunk must be (1, n_action_steps, dim); got {chunk_np.shape}"
            )
        # A non-finite prediction would defeat the [-1,1] clip below (np.clip passes
        # NaN through) and reach DROID as a NaN cartesian_position target — fail loudly
        # instead (a NaN chunk means the policy/normalizer is broken; there is no valid
        # command to salvage from it).
        if not np.isfinite(chunk_np).all():
            raise RuntimeError(
                "relative policy emitted a non-finite action chunk "
                f"(nan={int(np.isnan(chunk_np).sum())}, inf={int(np.isinf(chunk_np).sum())} "
                f"of {chunk_np.size} values); refusing to decode it into a robot command"
            )
        # Safety bound (the relative arm's analogue of the velocity arm's [-1,1] command
        # clip): the policy head is generative and can exceed the trained normalized
        # range, so clamp the NORMALIZED chunk to [-1,1] before decode. This bounds every
        # per-timestep relative displacement (and the gripper) to the trained per-timestep
        # MIN_MAX range, so the composed absolute pose can drift from the anchor by at
        # most the trained displacement — no unbounded reach. (No absolute-xyz workspace
        # clip is applied: the relative arm has no absolute-pose action stat, and the
        # real anchor + bounded relative displacement keep the target near the workspace.
        # Robot-bench item: confirm the bounded reach is acceptable.)
        return np.clip(chunk_np[0], -1.0, 1.0)

    def _predict_relative(self, processed_obs: dict, *, anchor_pose6: np.ndarray) -> np.ndarray:
        """Pop the next executed absolute pose for the relative arm, refilling the chunk.

        When the per-chunk queue is empty: capture the generation-time anchor = the
        current PROPRIO pose (``anchor_pose6`` = ``state[:6]``), generate the normalized
        relative chunk, compose it to absolute 7D euler poses against that single held
        anchor, and queue the executed window. Then pop one pose and clip the absolute
        gripper to [0,1]. The anchor is held across all pops of one chunk (re-anchoring
        per pop would double-count motion already executed). No Mulligan-side pose clamp: the
        absolute pose is commanded to DROID as-is — hardware safety is DROID's + Franka's.

        Proprio-anchored: the training targets are relativized against the current PROPRIO pose
        (dataset_utils.compute_relative_pose_pertimestep_stats /
        RelativePoseActionProcessorStep), so the decode anchors on the current measured
        proprio here — identical frames train↔deploy. ``rel[0]`` carries the command-vs-
        proprio lead, so re-grounding on live proprio each chunk reconstructs the command
        trajectory with no boundary retreat and no open-loop runaway.
        """
        if not self._relative_pose_queue:
            chunk_norm = self._generate_relative_chunk_normalized(processed_obs)
            poses7 = self.decode_relative_chunk_to_absolute(chunk_norm, anchor_pose6)
            if poses7.shape[0] == 0:
                raise RuntimeError("relative decode produced an empty chunk")
            self._relative_pose_queue = [poses7[i] for i in range(poses7.shape[0])]
        pose7 = np.asarray(self._relative_pose_queue.pop(0), dtype=np.float32).reshape(-1)
        if pose7.shape != (7,):
            raise RuntimeError(
                f"relative decoded pose must be 7D [xyz,rpy,grip]; got {pose7.shape}"
            )
        # The absolute gripper command space is [0,1] (open/closed), so clip to it — a
        # command-range transform, not a hardware guard.
        pose7[6] = float(np.clip(pose7[6], 0.0, 1.0))
        return pose7

    def _finalize_action(self, action: np.ndarray) -> np.ndarray:
        """Clip an un-normalized velocity command to DROID's [-1, 1] range.

        predict() calls this; tests drive it directly so no parallel copy can diverge.
        """
        if getattr(self, "action_mode", "absolute") == "relative":
            # A relative-pose chunk must be composed against the generation-time anchor
            # (decode_relative_chunk_to_absolute), never finalized one action at a time.
            raise NotImplementedError(
                "action_mode=relative: per-action _finalize_action is not valid (the relative "
                "chunk must be composed to absolute against the generation-time anchor). Use "
                "decode_relative_chunk_to_absolute()."
            )
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)

    def decode_relative_chunk_to_absolute(
        self, chunk_norm_rel: np.ndarray, base_pose6: np.ndarray
    ) -> np.ndarray:
        """Compose a NORMALIZED relative-pose chunk into ABSOLUTE 7D euler poses.

        The UMI relative arm emits ``[rel_trans(3), rel_r6(6), grip(1)]`` per timestep,
        each normalized with its own ``(T, 10)`` MIN_MAX stats. This (1) per-timestep
        un-normalizes the chunk to physical relative pose, (2) composes it back to the
        absolute EE pose against the GENERATION-TIME anchor ``base_pose6``, and (3) decodes
        r6 → euler. The anchor is captured ONCE per chunk and held across all popped steps
        (composing each pop against the live per-step pose would double-count motion).

        ``base_pose6`` MUST be in the SAME frame the dataset relativized against — the
        current PROPRIO pose (the command-pose targets are relativized against
        ``observation.state`` cartesian_position; see
        ``dataset_utils.compute_relative_pose_pertimestep_stats`` and
        ``RelativePoseActionProcessorStep``). The caller (``_predict_relative``) supplies
        the current measured proprio pose ``state[:6]``. ``rel[0]`` carries the command
        lead, so composing against live proprio reconstructs the command trajectory.

        Args:
            chunk_norm_rel: ``(T', 10)`` normalized relative chunk (the executed window).
            base_pose6: ``(6,)`` anchor pose ``[x, y, z, roll, pitch, yaw]`` (euler rad).

        Returns:
            ``(T', 7)`` absolute poses ``[x, y, z, roll, pitch, yaw, gripper]`` (gripper
            absolute 0/1, passed through from the relative chunk).
        """
        from mulligan.real.policy.relative_pose import decode_normalized_relative_chunk

        return decode_normalized_relative_chunk(
            chunk_norm_rel, base_pose6, self._action_min, self._action_max
        )


# ---------------------------------------------------------------------------
# Diffusion policy overrides
# ---------------------------------------------------------------------------


def apply_diffusion_overrides(
    policy, noise_scheduler_name, num_inference_steps, n_action_steps=None
):
    """Apply scheduler, inference-step, and action-step overrides to a diffusion policy."""
    if not hasattr(policy, "diffusion"):
        if noise_scheduler_name or num_inference_steps or n_action_steps:
            logger.warning(
                f"Policy {policy.__class__.__name__} has no 'diffusion' attribute; "
                f"skipping diffusion overrides."
            )
        return

    from diffusers import DDIMScheduler, DDPMScheduler

    dm = policy.diffusion
    if noise_scheduler_name:
        supported_schedulers = {"DDPM": DDPMScheduler, "DDIM": DDIMScheduler}
        if noise_scheduler_name not in supported_schedulers:
            raise ValueError(
                f"Unsupported noise scheduler {noise_scheduler_name!r}; expected one of "
                f"{sorted(supported_schedulers)}"
            )
        scheduler_cls = supported_schedulers[noise_scheduler_name]
        old_cfg = dm.noise_scheduler.config
        dm.noise_scheduler = scheduler_cls(
            num_train_timesteps=old_cfg.num_train_timesteps,
            beta_start=old_cfg.beta_start,
            beta_end=old_cfg.beta_end,
            beta_schedule=old_cfg.beta_schedule,
            clip_sample=old_cfg.clip_sample,
            clip_sample_range=old_cfg.clip_sample_range,
            prediction_type=old_cfg.prediction_type,
        )
    if num_inference_steps:
        dm.num_inference_steps = num_inference_steps
    if n_action_steps is not None:
        if n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
        # LeRobot DP requires n_action_steps <= horizon - n_obs_steps + 1 (the
        # number of future actions the predicted chunk can supply), not just
        # <= horizon. Use the true bound.
        n_obs = int(getattr(policy.config, "n_obs_steps", 1))
        max_exec = policy.config.horizon - n_obs + 1
        if n_action_steps > max_exec:
            raise ValueError(
                f"n_action_steps ({n_action_steps}) must be <= horizon - n_obs_steps + 1 "
                f"({max_exec}; horizon={policy.config.horizon}, n_obs_steps={n_obs})"
            )
        policy.config.n_action_steps = n_action_steps
        if hasattr(policy, "reset"):
            policy.reset()
        horizon_msg = (
            f"Diffusion action horizon: predict {policy.config.horizon} (horizon), "
            f"execute {n_action_steps} (n_action_steps)."
        )
        logger.info(horizon_msg)
        print(f"  {horizon_msg}")


def resolve_policy_action_contract(policy_config) -> tuple[str, str]:
    """Return action target/frame encoded in policy config.

    Older checkpoints have no explicit action fields; those are interpreted as
    cartesian_velocity/base policies.
    """
    action_target = getattr(policy_config, "action_target", None)
    if action_target is None:
        return "cartesian_velocity", "base"
    return action_target, getattr(policy_config, "cartesian_action_frame", None) or "base"


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def policy_image_hw(policy_config, default_hw: tuple[int, int]) -> tuple[int, int]:
    """(H, W) of the checkpoint's first image feature, else ``default_hw``."""
    image_features = getattr(policy_config, "image_features", None)
    if not image_features:
        return default_hw
    first_img_feat = next(iter(image_features.values()))
    return first_img_feat.shape[1], first_img_feat.shape[2]


def load_policy_by_model_id(
    model_id: str,
    policy_id: int,
    device: str,
    noise_scheduler: str | None = None,
    num_inference_steps: int | None = None,
    default_camera_height: int = 480,
    default_camera_width: int = 640,
    n_action_steps: int | None = REAL_PROTOCOL_N_ACTION_STEPS,
    dp_artifact_override: str | None = None,
) -> PolicyEntry:
    """Load a DP actor or a Vision-IDQL critic (with its DP actor) and return a PolicyEntry.

    ``model_id`` is any id :mod:`mulligan.real.policy.dp` accepts (``hf://``, a local
    directory, or ``wandb://`` with the optional ``wandb`` package). A directory whose
    ``metadata.json`` has ``policy_type == "vision_idql"`` is a critic; anything else must
    be a LeRobot DP checkpoint. ``dp_artifact_override`` (a model id) replaces the critic's
    trained DP; it is an error for a DP actor. The returned entry keeps ``model_id`` as
    given.
    """
    resolved = resolve_model_id(model_id)
    if resolved != model_id:
        print(f"  Resolved {model_id} -> {resolved}")
    artifact_dir = checkpoint_dir(resolved)
    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("policy_type") != "vision_idql":
            raise ValueError(
                f"{model_id}: metadata.json policy_type={metadata.get('policy_type')!r} is not "
                "a real-robot policy (expected a LeRobot DP dir or a vision_idql critic)."
            )
        from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

        logger.info(f"Detected vision-idql artifact: {model_id}")
        idql_policy = VisionIDQLRealWorldPolicy.from_artifact(
            artifact_dir,
            device=device,
            dp_artifact_override=dp_artifact_override,
        )
        idql_policy.eval()
        # Vision-IDQL exec is fixed by the checkpoint (the Q-head action dim is
        # tied to n_action_steps); the --n-action-steps override does NOT apply.
        # Warn loudly on a genuine mismatch rather than silently ignoring it.
        baked_n = idql_policy.dp.config.n_action_steps
        if n_action_steps is not None and n_action_steps != baked_n:
            logger.warning(
                "vision_idql exec horizon is fixed at %s by the checkpoint; "
                "ignoring requested n_action_steps=%s (re-train to change it).",
                baked_n,
                n_action_steps,
            )
        dm = idql_policy.dp.diffusion
        print(
            f"  Scheduler: {dm.noise_scheduler.__class__.__name__} ({dm.num_inference_steps} steps)"
        )
        return PolicyEntry(
            model_id=model_id,
            policy_id=policy_id,
            policy=idql_policy,
            camera_height=metadata.get("image_height", default_camera_height),
            camera_width=metadata.get("image_width", default_camera_width),
        )

    if dp_artifact_override is not None:
        raise ValueError(
            f"{model_id} is a DP actor, not a Vision-IDQL critic; a DP override "
            f"({dp_artifact_override}) only applies to critics."
        )
    policy, preprocessor, postprocessor = load_dp(artifact_dir, device=device)
    apply_diffusion_overrides(policy, noise_scheduler, num_inference_steps, n_action_steps)
    if hasattr(policy, "diffusion"):
        dm = policy.diffusion
        print(
            f"  Scheduler: {dm.noise_scheduler.__class__.__name__} "
            f"({dm.num_inference_steps} inference steps)"
        )
    pol_h, pol_w = policy_image_hw(policy.config, (default_camera_height, default_camera_width))
    if (pol_h, pol_w) != (default_camera_height, default_camera_width):
        print(f"  Image resolution: {pol_h}x{pol_w} (auto-detected from checkpoint)")
    wrapped = build_real_world_policy(
        policy,
        preprocessor,
        postprocessor,
        camera_keys=[],
        default_camera_height=default_camera_height,
        default_camera_width=default_camera_width,
    )
    if wrapped.camera_crops:
        print(f"  Per-camera replacement crop: {wrapped.camera_crops} (from policy config)")
    return PolicyEntry(
        model_id=model_id,
        policy_id=policy_id,
        policy=wrapped,
        camera_height=pol_h,
        camera_width=pol_w,
    )


def validate_station_role_policy_crops(
    policy_config,
    camera_crops: dict[str, tuple[int, int, int, int]],
) -> None:
    """Refuse to run a station policy whose role cameras lack their crop contract."""
    station_roles = sorted(
        require_station_role_image_features(
            getattr(policy_config, "image_features", {}) or {},
            context="policy config",
        )
    )
    if not station_roles:
        return

    missing = [
        role
        for role in station_roles
        if role in STATION_CAMERA_DEFAULT_CROPS and role not in camera_crops
    ]
    if missing:
        raise ValueError(
            "policy has station role-named image feature(s) "
            f"{station_roles}, but camera_crop_boxes is missing crop(s) for {missing}. "
            "Refusing to run full-frame eval for a station policy that trained with "
            "role-keyed default crops; check the checkpoint config/loader contract."
        )


def build_real_world_policy(
    raw_policy,
    preprocessor,
    postprocessor,
    *,
    camera_keys,
    default_camera_height: int,
    default_camera_width: int,
):
    """Single construction path for ``LeRobotRealWorldPolicy`` from a loaded checkpoint:
    resolves image resolution, the action contract and the per-camera replacement crops
    FROM ``raw_policy.config`` -- identically for every eval/rollout entrypoint (rollout,
    dagger, load_policy_by_model_id) so crop handling can't be silently omitted or drift
    between them. A policy whose image features are not station roles is refused."""
    from mulligan.real.policy.side_crop import normalize_crop_map, reject_dual_side_crop_boxes

    pol_h, pol_w = policy_image_hw(raw_policy.config, (default_camera_height, default_camera_width))
    action_target, cartesian_action_frame = resolve_policy_action_contract(raw_policy.config)
    camera_crops = normalize_crop_map(
        getattr(raw_policy.config, "camera_crop_boxes", {}),
        context="policy config camera_crop_boxes",
    )
    validate_station_role_policy_crops(raw_policy.config, camera_crops)
    reject_dual_side_crop_boxes(
        getattr(raw_policy.config, "dual_side_crop_boxes", None), context="policy config"
    )
    return LeRobotRealWorldPolicy(
        policy=raw_policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        camera_keys=camera_keys,
        camera_height=pol_h,
        camera_width=pol_w,
        camera_crops=camera_crops,
        action_target=action_target,
        cartesian_action_frame=cartesian_action_frame,
        action_mode=getattr(raw_policy.config, "action_mode", None) or "absolute",
    )
