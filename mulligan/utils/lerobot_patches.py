# Portions of this file are adapted from LeRobot (https://github.com/huggingface/lerobot,
# commit 0530dd9b): DiffusionConfig.__post_init__ in
# src/lerobot/policies/diffusion/configuration_diffusion.py, and DiffusionPolicy.reset and
# DiffusionModel.__init__ / _prepare_global_conditioning / generate_actions / compute_loss in
# src/lerobot/policies/diffusion/modeling_diffusion.py.
# Modified by the Mulligan authors: the U-Net downsampling-factor check uses
# 2 ** (len(down_dims) - 1); the robot state is optional (queues, conditioning size, batch
# shape); unsupported vision frontends are rejected at model construction; the padding-masked
# loss is written as a mean over valid elements. The rest of the file is by the Mulligan
# authors (MIT).
#
# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Runtime patches for lerobot (pinned in pyproject.toml).

Applied by :func:`mulligan.apply_runtime_patches`:

* the policy I/O contract fields on ``PreTrainedConfig`` (camera crops, action target),
  so ``config.json`` round-trips the preprocessing a real policy was trained with;
* the ``DiffusionConfig`` validation fix (downsampling factor);
* the diffusion model patch: optional robot state, per-camera ResNet encoders and the
  padding-masked loss (lerobot PR #3442);
* per-timestep ``(T, D)`` MIN_MAX action statistics (relative-pose Cable policies).

Every released real diffusion policy uses a ResNet-18 spatial-softmax encoder
(``vision_pooling="mean"``); other vision frontends are not supported and are rejected at
model construction.
"""

import logging

logger = logging.getLogger(__name__)


def apply_policy_io_contract_patch():
    """Register the policy I/O contract fields on lerobot's ``PreTrainedConfig``.

    These fields are part of the train/eval semantics, not launch-time knobs:
    eval loads ``config.json`` + weights and reconstructs the exact camera
    preprocessing and action target the policy was trained with.

      - ``camera_crop_boxes``     per-camera replacement crop (``side_crop.py``)
      - ``dual_side_crop_boxes``  additive side-ROI box (empty in every released policy)
      - ``action_target``         ``cartesian_velocity`` | ``cartesian_position`` | ...
      - ``cartesian_action_frame``  ``base`` | ``eef``
      - ``action_mode``           ``absolute`` | ``relative``
      - vision-frontend fields (``vision_pooling``, ``attn_*``, ``proprio_*``,
        ``stereo_*``): stored in released ``config.json`` files, so they must parse;
        only their defaults are supported.

    lerobot is an upstream pin, so the fields are injected at import time. This is functionally
    identical to declaring the fields in ``policies.py``: because ``_save_pretrained``
    serializes via ``draccus.dump`` (declared dataclass fields only) and
    ``from_pretrained`` reconstructs via ``draccus.parse``, the fields must be real
    dataclass fields on the concrete config classes to round-trip. Adding them to
    the base class makes every concrete policy config (``DiffusionConfig``, ...)
    inherit them: dataclass reads ``base.__dataclass_fields__`` at subclass
    decoration time, so subclasses imported after this patch pick the fields up
    automatically; subclasses imported before it are re-processed here.

    Fails loudly if injection does not take: a config.json silently missing the
    crop/action contract means eval reconstructs the wrong preprocessing and the
    experiment is invalid. ``ImportError`` (lerobot genuinely absent) is the one
    tolerated no-op — the real write/eval paths require lerobot and fail there.
    """
    import dataclasses

    try:
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError:
        logger.warning("Could not import PreTrainedConfig; policy I/O contract patch not applied")
        return

    # (name, annotation, class-level default). The crop maps use a default_factory
    # (a fresh empty dict per instance); the action fields default to None
    # ("config without an explicit target" — eval reads that as the
    # cartesian_velocity/base default). Built fresh each call so the
    # field() objects are never shared across invocations.
    contract_specs = [
        (
            "camera_crop_boxes",
            dict[str, tuple[int, int, int, int]],
            dataclasses.field(default_factory=dict),
        ),
        (
            "dual_side_crop_boxes",
            dict[str, tuple[int, int, int, int]],
            dataclasses.field(default_factory=dict),
        ),
        ("action_target", str | None, None),
        ("cartesian_action_frame", str | None, None),
        # Relative-pose action representation: "relative" re-expresses the absolute
        # EE-pose horizon relative to the current pose, with per-timestep (T,D)
        # normalization. None/"absolute" = the per-frame target; configs without the
        # field load as absolute.
        ("action_mode", str | None, None),
        # Vision-frontend fields of the research code. Released configs carry them with
        # these defaults; the model patch rejects any other value.
        ("vision_pooling", str, "mean"),
        ("attn_num_queries", int, 8),
        ("attn_num_layers", int, 2),
        ("attn_num_heads", int, 8),
        ("attn_mlp_ratio", float, 2.0),
        ("attn_feature_dim", int, 256),
        ("proprio_mode", str, "current"),
        ("proprio_history_len", int, 0),
        ("stereo_pairs", str, ""),
        ("stereo_passthrough_cameras", str, ""),
        ("stereo_feature_dim", int, 128),
        ("stereo_num_layers", int, 2),
        ("stereo_num_heads", int, 8),
    ]
    field_names = [name for name, _, _ in contract_specs]

    def _clone_field(f):
        """Rebuild a fresh, unprocessed ``field()`` spec from a processed Field.

        The first ``@dataclass`` run consumes ``default_factory`` class attributes
        (they leave no class-level value), so re-running ``@dataclass`` would treat
        those fields as required and raise "non-default argument follows default".
        Cloning the Field back into a class-body ``field()`` reproduces the original
        declaration exactly (default / default_factory / init / repr / metadata / ...).
        """
        kwargs = {
            "init": f.init,
            "repr": f.repr,
            "hash": f.hash,
            "compare": f.compare,
            "metadata": dict(f.metadata),
        }
        if f.default is not dataclasses.MISSING:
            kwargs["default"] = f.default
        if f.default_factory is not dataclasses.MISSING:
            kwargs["default_factory"] = f.default_factory
        kw_only = getattr(f, "kw_only", dataclasses.MISSING)
        if kw_only is not dataclasses.MISSING:
            kwargs["kw_only"] = kw_only
        return dataclasses.field(**kwargs)

    def _redecorate(cls):
        """Re-run @dataclass on ``cls`` preserving its fields and dataclass params.

        Mutates ``cls`` in place and returns it (same object identity), so draccus's
        ChoiceRegistry entry, the MRO, and isinstance all stay valid. ``__post_init__``
        and other methods are untouched (dataclass only regenerates __init__/__repr__/
        __eq__), so an already-applied ``apply_diffusion_config_patch`` survives.
        """
        # Re-materialize class-body field() specs for cls's OWN fields (inherited
        # fields come from the base __dataclass_fields__ and need no restoration).
        existing = getattr(cls, "__dataclass_fields__", {})
        for name in cls.__dict__.get("__annotations__", {}):
            f = existing.get(name)
            if f is not None and f._field_type is dataclasses._FIELD:
                setattr(cls, name, _clone_field(f))
        # @dataclass uses _set_new_attribute, which REFUSES to overwrite a dunder
        # already present in cls.__dict__. For an already-decorated class that means
        # __dataclass_fields__ would update but the generated __init__ (and __repr__/
        # __eq__) would stay stale — i.e. the new kwargs wouldn't be constructable.
        # Drop the generated dunders so the re-decoration regenerates them. These are
        # always dataclass-generated on lerobot configs (plain @dataclass, no manual
        # __init__/__repr__/__eq__); __post_init__ is a real method and left intact.
        params = getattr(cls, "__dataclass_params__", None)
        regen = ["__init__", "__repr__", "__eq__", "__hash__"]
        if params is not None and params.frozen:
            regen += ["__setattr__", "__delattr__"]
        for dunder in regen:
            if dunder in cls.__dict__:
                delattr(cls, dunder)
        if params is None:
            return dataclasses.dataclass(cls)
        return dataclasses.dataclass(
            cls,
            init=params.init,
            repr=params.repr,
            eq=params.eq,
            order=params.order,
            unsafe_hash=params.unsafe_hash,
            frozen=params.frozen,
        )

    if not all(name in PreTrainedConfig.__dataclass_fields__ for name in field_names):
        for name, annotation, default in contract_specs:
            PreTrainedConfig.__annotations__[name] = annotation
            setattr(PreTrainedConfig, name, default)
        _redecorate(PreTrainedConfig)

    # Re-process any concrete subclass already imported before this patch ran so it
    # inherits the new base fields. (Subclasses imported later inherit them at their
    # own @dataclass decoration.)
    def _all_subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _all_subclasses(sub)

    for sub in list(_all_subclasses(PreTrainedConfig)):
        if not all(name in sub.__dataclass_fields__ for name in field_names):
            _redecorate(sub)

    # Fail loud: the canonical real-policy config must now carry the fields, else
    # training saves a config.json missing the contract and eval silently
    # reconstructs the wrong camera preprocessing / action target.
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    missing = [n for n in field_names if n not in DiffusionConfig.__dataclass_fields__]
    if missing:
        raise RuntimeError(
            f"Policy I/O contract injection failed: DiffusionConfig is missing "
            f"{missing}. Camera-crop / action-target preprocessing would not round-trip "
            f"through config.json. Refusing to continue (a saved policy would silently "
            f"lose its train/eval preprocessing contract)."
        )
    logger.info(
        "Registered policy I/O contract fields on PreTrainedConfig: %s",
        ", ".join(field_names),
    )


def apply_diffusion_config_patch():
    """
    Patch DiffusionConfig to correct downsampling factor calculation.

    The original implementation calculated downsampling_factor = 2 ** len(self.down_dims),
    which was too strict. The correct factor should be 2 ** (len(self.down_dims) - 1).
    """
    try:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

        # Define the patched method.
        # Body = upstream DiffusionConfig.__post_init__ (lerobot 0530dd9b,
        # configuration_diffusion.py:163-210) except the downsampling factor, which is
        # max(1, 2 ** (len(down_dims) - 1)) instead of upstream's 2 ** len(down_dims).
        def patched_post_init(self):
            # Call parent's __post_init__ (PreTrainedConfig). Use the explicit
            # super(DiffusionConfig, self) form: this function is assigned onto the class,
            # so it has no zero-arg-super __class__ cell.
            super(DiffusionConfig, self).__post_init__()

            # Input validation (not exhaustive).
            if not self.vision_backbone.startswith("resnet"):
                raise ValueError(
                    f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
                )

            supported_prediction_types = ["epsilon", "sample"]
            if self.prediction_type not in supported_prediction_types:
                raise ValueError(
                    f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
                )
            supported_noise_schedulers = ["DDPM", "DDIM"]
            if self.noise_scheduler_type not in supported_noise_schedulers:
                raise ValueError(
                    f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                    f"Got {self.noise_scheduler_type}."
                )

            if self.resize_shape is not None and (
                len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
            ):
                raise ValueError(
                    f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}."
                )
            if not (0 < self.crop_ratio <= 1.0):
                raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

            if self.resize_shape is not None:
                if self.crop_ratio < 1.0:
                    self.crop_shape = (
                        int(self.resize_shape[0] * self.crop_ratio),
                        int(self.resize_shape[1] * self.crop_ratio),
                    )
                else:
                    # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                    self.crop_shape = None
            if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
                raise ValueError(
                    f"`crop_shape` must have positive dimensions. Got {self.crop_shape}."
                )

            # Check that the horizon size and U-Net downsampling is compatible.
            # U-Net downsamples by 2 with each stage.
            # (len(down_dims) - 1) instead of upstream's len(down_dims), floored at 1 so an
            # empty/short down_dims still yields a valid factor.
            downsampling_factor = max(1, 2 ** (len(self.down_dims) - 1))
            if self.horizon % downsampling_factor != 0:
                raise ValueError(
                    "The horizon should be an integer multiple of the downsampling factor (which is determined "
                    f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=} (factor={downsampling_factor})"
                )

        # Apply the patch
        DiffusionConfig.__post_init__ = patched_post_init
        assert DiffusionConfig.__post_init__ is patched_post_init
        logger.info("Patched DiffusionConfig.__post_init__ to fix downsampling factor validation")

    except ImportError:
        logger.warning("Could not import DiffusionConfig, patch not applied")


def _check_supported_frontend(config) -> None:
    """Reject vision-frontend settings that no released policy uses (fail loud)."""
    unsupported = {
        "vision_pooling": (getattr(config, "vision_pooling", "mean"), "mean"),
        "proprio_mode": (getattr(config, "proprio_mode", "current"), "current"),
        "proprio_history_len": (getattr(config, "proprio_history_len", 0), 0),
        "stereo_pairs": (getattr(config, "stereo_pairs", ""), ""),
    }
    bad = {k: v for k, (v, default) in unsupported.items() if v != default}
    if bad or not config.vision_backbone.startswith("resnet"):
        raise NotImplementedError(
            f"Unsupported diffusion vision frontend {bad or config.vision_backbone!r}: the "
            "release supports ResNet spatial-softmax encoders only (vision_pooling='mean')."
        )


def apply_diffusion_model_patch():
    """Patch LeRobot DiffusionPolicy: optional robot state and the padding-masked loss."""
    try:
        from collections import deque

        import einops
        import torch
        from torch import nn

        import lerobot.policies.diffusion.modeling_diffusion as md
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

        def patched_reset(self):
            self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
            if self.config.robot_state_feature:
                self._queues[OBS_STATE] = deque(maxlen=self.config.n_obs_steps)
            if self.config.image_features:
                self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
            if self.config.env_state_feature:
                self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

        def patched_model_init(self, config):
            super(md.DiffusionModel, self).__init__()
            self.config = config
            if config.image_features:
                _check_supported_frontend(config)

            global_cond_dim = 0
            if self.config.robot_state_feature:
                global_cond_dim += self.config.robot_state_feature.shape[0]
            if self.config.image_features:
                num_images = len(self.config.image_features)
                if self.config.use_separate_rgb_encoder_per_camera:
                    encoders = [md.DiffusionRgbEncoder(config) for _ in range(num_images)]
                    self.rgb_encoder = nn.ModuleList(encoders)
                    global_cond_dim += encoders[0].feature_dim * num_images
                else:
                    self.rgb_encoder = md.DiffusionRgbEncoder(config)
                    global_cond_dim += self.rgb_encoder.feature_dim * num_images
            if self.config.env_state_feature:
                global_cond_dim += self.config.env_state_feature.shape[0]

            self.unet = md.DiffusionConditionalUnet1d(
                config, global_cond_dim=global_cond_dim * config.n_obs_steps
            )
            # lerobot PR #2486: optional torch.compile of the U-Net (off by default).
            if config.compile_model:
                self.unet = torch.compile(self.unet, mode=config.compile_mode)
            self.noise_scheduler = md._make_noise_scheduler(
                config.noise_scheduler_type,
                num_train_timesteps=config.num_train_timesteps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
                clip_sample=config.clip_sample,
                clip_sample_range=config.clip_sample_range,
                prediction_type=config.prediction_type,
            )
            if config.num_inference_steps is None:
                self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
            else:
                self.num_inference_steps = config.num_inference_steps

        def batch_shape(self, batch):
            if self.config.robot_state_feature:
                return batch[OBS_STATE].shape[:2]
            if self.config.image_features:
                return batch[OBS_IMAGES].shape[:2]
            if self.config.env_state_feature:
                return batch[OBS_ENV_STATE].shape[:2]
            raise ValueError("Diffusion policy needs at least one observation feature.")

        def prepare_global_conditioning(self, batch):
            global_cond_feats = []
            batch_size = None
            n_obs_steps = None
            if self.config.robot_state_feature:
                batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
                global_cond_feats.append(batch[OBS_STATE])
            if self.config.image_features:
                batch_size, n_obs_steps = batch[OBS_IMAGES].shape[:2]
                if self.config.use_separate_rgb_encoder_per_camera:
                    images_per_camera = einops.rearrange(
                        batch[OBS_IMAGES], "b s n ... -> n (b s) ..."
                    )
                    per_camera_feats = [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                    img_features = einops.rearrange(
                        torch.cat(per_camera_feats),
                        "(n b s) ... -> b s (n ...)",
                        b=batch_size,
                        s=n_obs_steps,
                    )
                else:
                    img_features = self.rgb_encoder(
                        einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                    )
                    img_features = einops.rearrange(
                        img_features,
                        "(b s n) ... -> b s (n ...)",
                        b=batch_size,
                        s=n_obs_steps,
                    )
                global_cond_feats.append(img_features)
            if self.config.env_state_feature:
                batch_size, n_obs_steps = batch[OBS_ENV_STATE].shape[:2]
                global_cond_feats.append(batch[OBS_ENV_STATE])
            if batch_size is None or n_obs_steps is None:
                raise ValueError("Diffusion policy needs at least one observation feature.")
            return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

        def patched_generate_actions(self, batch, noise=None):
            batch_size, n_obs_steps = self._batch_shape(batch)
            assert n_obs_steps == self.config.n_obs_steps
            global_cond = self._prepare_global_conditioning(batch)
            actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
            start = n_obs_steps - 1
            end = start + self.config.n_action_steps
            return actions[:, start:end]

        def patched_compute_loss(self, batch):
            import torch.nn.functional as F

            required_keys = {ACTION, "action_is_pad"}
            if self.config.robot_state_feature:
                required_keys.add(OBS_STATE)
            assert set(batch).issuperset(required_keys)
            assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
            _, n_obs_steps = self._batch_shape(batch)
            horizon = batch[ACTION].shape[1]
            assert horizon == self.config.horizon
            assert n_obs_steps == self.config.n_obs_steps

            global_cond = self._prepare_global_conditioning(batch)
            trajectory = batch[ACTION]
            eps = torch.randn(trajectory.shape, device=trajectory.device)
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(trajectory.shape[0],),
                device=trajectory.device,
            ).long()
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
            pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

            if self.config.prediction_type == "epsilon":
                target = eps
            elif self.config.prediction_type == "sample":
                target = batch[ACTION]
            else:
                raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

            loss = F.mse_loss(pred, target, reduction="none")
            if not self.config.do_mask_loss_for_padding:
                return loss.sum() / loss.numel()
            # Per-element validity mask: 1 where the action is a real (non-padded)
            # frame, 0 at episode-edge padding copies ("action_is_pad" is asserted
            # present above).
            valid = (~batch["action_is_pad"]).unsqueeze(-1).to(loss.dtype).expand_as(loss)
            # Mass-preserving masked mean, matching upstream LeRobot after PR #3442:
            # `valid.sum() == mask.sum() * action_dim`, so this equals upstream
            # `(loss*mask).sum() / (mask.sum()*action_dim)`. `.mean()` would
            # re-introduce the padding underestimate.
            return (loss * valid).sum() / valid.sum().clamp_min(1e-8)

        md.DiffusionPolicy.reset = patched_reset
        md.DiffusionModel.__init__ = patched_model_init
        md.DiffusionModel._batch_shape = batch_shape
        md.DiffusionModel._prepare_global_conditioning = prepare_global_conditioning
        md.DiffusionModel.generate_actions = patched_generate_actions
        md.DiffusionModel.compute_loss = patched_compute_loss
        logger.info("Patched DiffusionPolicy for optional state and the masked padding loss")

    except ImportError:
        logger.warning("Could not import DiffusionPolicy, diffusion model patch not applied")
    # No broad `except Exception` on purpose: a half-applied patch would silently
    # drop the padded-action loss fix (PR #3442) and train on an unpatched policy.


def apply_normalizer_per_timestep_patch():
    """Let the lerobot MIN_MAX normalizer accept per-timestep ``(T, D)`` ACTION stats.

    Relative-pose training (``action_mode="relative"``) normalizes each future
    timestep with its OWN statistics (``temporally_independent_normalization=True``),
    so ``stats["action"]["min"|"max"]`` are ``(T, D)`` instead of ``(D,)``. The MIN_MAX
    formula is elementwise, so ``(B, T, D)`` broadcasts against ``(T, D)`` natively —
    the full-horizon train/normalize path needs no change. The ONE break is the
    EXECUTED-window unnormalize at deploy: a ``(B, n_action_steps, D)`` chunk with
    ``n_action_steps < T`` cannot broadcast against ``(T, D)``. This patch slices the
    stat rows to the chunk's leading timesteps (``start=0`` under the pinned
    ``n_obs_steps=1``, where the executed window begins at the anchor) and otherwise
    falls through to the original transform byte-for-byte.

    Surgically scoped: only the ``action`` key with 2-D stats and a real 3-D
    ``(B, T', D)`` action tensor is intercepted. OBS keys, every 1-D-stat (non-relative)
    policy, and the full-horizon (``T' == T``) train path are byte-identical. A 2-D
    ``(B, D)`` action (a single deploy pop) against per-timestep stats is ambiguous —
    there is no time axis to align rows to — so it RAISES rather than silently
    mis-slicing the batch dim as time.
    """
    try:
        from lerobot.processor.normalize_processor import _NormalizationMixin
        from lerobot.utils.constants import ACTION
    except ImportError:
        logger.warning(
            "Could not import _NormalizationMixin; per-timestep action-norm patch not applied"
        )
        return

    if getattr(_NormalizationMixin, "_mulligan_per_timestep_patched", False):
        return

    import torch

    from lerobot.configs.types import NormalizationMode

    _orig_apply_transform = _NormalizationMixin._apply_transform

    def _patched_apply_transform(self, tensor, key, feature_type, *, inverse=False):
        tensor_stats = getattr(self, "_tensor_stats", None)
        if key == ACTION and tensor_stats and key in tensor_stats:
            stats = tensor_stats[key]
            mn = stats.get("min")
            if mn is not None and getattr(mn, "ndim", 1) == 2:
                norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
                if norm_mode != NormalizationMode.MIN_MAX:
                    raise ValueError(
                        f"Per-timestep (T,D) action stats are only supported for MIN_MAX "
                        f"normalization; got {norm_mode}. (Relative-pose forbids quantile.)"
                    )
                T = mn.shape[0]
                if tensor.ndim == 3:
                    tprime = tensor.shape[1]
                    if tprime > T:
                        raise ValueError(
                            f"action tensor has more timesteps ({tprime}) than the per-timestep "
                            f"(T={T}, D) normalization stats; cannot align rows."
                        )
                    if tprime < T:
                        # Inline the MIN_MAX transform with stat rows [0:tprime] (executed
                        # window begins at the anchor under the pinned n_obs_steps=1). We do
                        # not delegate to the original for the sliced case: on a device/dtype
                        # mismatch (e.g. a GPU action vs CPU-resident stats at deploy) the
                        # original calls self.to(...), which REBUILDS self._tensor_stats from
                        # the full (T,D) self.stats and would discard a temporary sliced
                        # override → broadcast failure. Computing it here against
                        # device/dtype-aligned slices is robust to that.
                        min_val = stats["min"][:tprime].to(device=tensor.device, dtype=tensor.dtype)
                        max_val = stats["max"][:tprime].to(device=tensor.device, dtype=tensor.dtype)
                        denom = max_val - min_val
                        denom = torch.where(
                            denom == 0,
                            torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype),
                            denom,
                        )
                        if inverse:
                            return (tensor + 1) / 2 * denom + min_val
                        return 2 * (tensor - min_val) / denom - 1
                    # tprime == T: full horizon broadcasts natively → delegate to the
                    # original (byte-identical training path).
                elif tensor.ndim == 2:
                    raise ValueError(
                        "Per-timestep (T, D) action stats cannot normalize a single (B, D) "
                        "action (no time axis to align stat rows). The relative-pose deploy "
                        "path must compose/unnormalize the full action chunk (B, T', D), not "
                        "one popped action at a time."
                    )
        return _orig_apply_transform(self, tensor, key, feature_type, inverse=inverse)

    _NormalizationMixin._apply_transform = _patched_apply_transform
    _NormalizationMixin._mulligan_per_timestep_patched = True
    logger.info(
        "Patched _NormalizationMixin._apply_transform for per-timestep (T,D) ACTION stats "
        "(relative-pose; non-relative + obs paths byte-identical)."
    )


def apply_all_patches():
    """
    Apply all runtime patches to lerobot.
    """
    apply_policy_io_contract_patch()
    apply_diffusion_config_patch()
    apply_diffusion_model_patch()
    apply_normalizer_per_timestep_patch()
