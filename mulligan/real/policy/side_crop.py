"""Per-camera static replacement crop for real-robot policies.

The side camera squashes a 480x640 frame to the training resolution with an
anisotropic x-scale of ~0.35, rendering the insertion-target marker barrel at
~2.5px wide at 224x224. The holder is a *fixed table fixture* (verified across
the recorded episodes), so a *static* crop box concentrating pixels on the
holder region needs no localization and takes the barrel to ~30px.

The crop is carried as **per-camera policy state** that both the training data
path and eval wrapper route through ``mulligan.real.policy.image_preprocess``. That shared
helper applies crop+resize identically, so train and eval produce the identical
cropped-then-resized tensor before train-only augmentation. The box is serialized
into the saved policy ``config.json`` as ``camera_crop_boxes`` so eval
reconstructs it with zero extra eval flags.

Box convention: ``(x0, y0, x1, y1)`` half-open, i.e. the slice is
``img[..., y0:y1, x0:x1]`` (HWC numpy at eval, CHW tensor at train — both index
height first, then width). Boxes are keyed by camera role and are pixels of the
640x480 frame the dataset stores (as are the station default crops in
``cameras.STATION_CAMERA_DEFAULT_CROPS``); eval downscales the live native frame to
640x480 first, then crops.
"""

from __future__ import annotations

from mulligan.real.policy.image_preprocess import (
    coerce_crop_box,
    preprocess_chw_tensor_for_policy,
)

_coerce_crop_box = coerce_crop_box


def feature_key_for_camera(camera: str) -> str:
    """LeRobot observation feature key for a bare camera name (e.g. ``side_1``)."""
    return f"observation.images.{camera}"


def parse_side_crop(specs: list[str] | None) -> dict[str, tuple[int, int, int, int]]:
    """Parse ``--side-crop/--camera-crop CAM=x0,y0,x1,y1`` specs into a crop map.

    Repeatable per camera. Returns ``{cam_key: (x0, y0, x1, y1)}`` in stored-frame px.
    Fails loudly on malformed specs or non-positive boxes — a silently-wrong
    crop would corrupt the whole arm.
    """
    crop_map: dict[str, tuple[int, int, int, int]] = {}
    if not specs:
        return crop_map
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--side-crop spec {spec!r} must be CAM=x0,y0,x1,y1 (missing '=')")
        cam_key, box_str = spec.split("=", 1)
        cam_key = cam_key.strip()
        if cam_key in crop_map:
            raise ValueError(
                f"--side-crop camera {cam_key!r} specified twice; a later box would "
                f"silently overwrite the earlier one. Give each camera once."
            )
        crop_map[cam_key] = _coerce_crop_box(
            cam_key,
            box_str.split(","),
            context=f"--side-crop spec {spec!r}",
        )
    return crop_map


def merge_default_crops(
    explicit: dict[str, tuple[int, int, int, int]],
    selected_cameras: set[str] | frozenset[str] | list[str],
    defaults: dict[str, tuple[int, int, int, int]],
) -> dict[str, tuple[int, int, int, int]]:
    """Merge per-camera default crop boxes into the explicit ``--side-crop`` map.

    A default applies only to a SELECTED (consumed) camera that has no explicit box;
    an explicit ``--side-crop`` always wins. Returns a new dict (inputs untouched). An
    empty ``defaults`` is a no-op, so this is safe to call unconditionally.

    ``defaults`` (STATION_CAMERA_DEFAULT_CROPS) is keyed by camera ROLE in STORED-640x480
    space; ``selected_cameras`` must therefore be the consumed cameras' role names for the
    defaults to bind. If ``defaults`` is non-empty but NONE of its keys are in ``selected``
    (e.g. ``selected`` names cameras that are not station roles) and ``explicit`` was also
    empty, that is almost certainly a camera-name mismatch that would silently train the
    policy UNCROPPED. We do not raise (an all-explicit or genuinely-no-default run is
    legitimate), but we WARN loudly so the mismatch cannot pass in silence.
    """
    selected = set(selected_cameras)
    merged = dict(explicit)
    n_merged_defaults = 0
    for cam, box in defaults.items():
        if cam in selected and cam not in merged:
            merged[cam] = box
            n_merged_defaults += 1
    if defaults and n_merged_defaults == 0 and not explicit:
        print(
            "WARNING: merge_default_crops considered default crop keys "
            f"{sorted(defaults)} but 0 applied to selected cameras {sorted(selected)} "
            "(and no explicit --side-crop was given). The policy will train UNCROPPED. "
            "This is usually a camera-name mismatch: STATION_CAMERA_DEFAULT_CROPS is keyed by "
            "ROLE (e.g. 'side_1'), so the selected cameras must be passed as their ROLE names "
            "for the defaults to bind."
        )
    return merged


def normalize_crop_map(
    raw_map,
    *,
    context: str,
) -> dict[str, tuple[int, int, int, int]]:
    """Validate a policy-contract crop map loaded from policy config."""
    if raw_map is None:
        return {}
    if not isinstance(raw_map, dict):
        raise ValueError(f"{context} must be a dict of camera role -> x0,y0,x1,y1")
    return {
        str(cam): _coerce_crop_box(str(cam), box, context=f"{context} entry")
        for cam, box in raw_map.items()
    }


def reject_dual_side_crop_boxes(raw_map, *, context: str) -> None:
    """Refuse a non-empty ``dual_side_crop_boxes`` map.

    Released DP configs and critic metadata carry this field, always empty. A non-empty
    map configures the additive dual-stream side-ROI feature, which the paper did not use
    and this release does not implement.
    """
    if raw_map is None or raw_map == {}:
        return
    if not isinstance(raw_map, dict):
        raise ValueError(f"{context} dual_side_crop_boxes must be a dict, got {raw_map!r}")
    raise NotImplementedError(
        f"{context} has dual_side_crop_boxes={raw_map!r}: the dual-stream side-ROI "
        "feature is not supported in this release."
    )


def crop_box_for_chw_tensor(img, box: tuple[int, int, int, int]):
    """Slice a (..., H, W) CHW tensor to ``box`` = (x0, y0, x1, y1), failing loudly
    if the box exceeds the frame."""
    x0, y0, x1, y1 = box
    h, w = img.shape[-2], img.shape[-1]
    if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0:
        raise ValueError(
            f"side-crop box (x0={x0}, y0={y0}, x1={x1}, y1={y1}) is invalid; "
            f"require nonnegative origin and x1>x0, y1>y0"
        )
    if x1 > w or y1 > h:
        raise ValueError(
            f"side-crop box (x0={x0}, y0={y0}, x1={x1}, y1={y1}) exceeds frame "
            f"of size HxW={h}x{w}; box must lie inside the native frame"
        )
    return img[..., y0:y1, x0:x1]


def build_crop_feature_map(
    crop_map: dict[str, tuple[int, int, int, int]],
) -> dict[str, tuple[int, int, int, int]]:
    """Map crop boxes from bare camera names onto full LeRobot feature keys.

    The dataset ``__getitem__`` keys are ``observation.images.<role>``; the
    eval wrapper keys on the bare role. Both forms are accepted by callers, so
    we materialize the feature-key form here for the training data path.
    """
    return {feature_key_for_camera(cam): box for cam, box in crop_map.items()}


class _PerCameraCropSubset:
    """Proxy around one LeRobot sub-dataset that applies a key-aware
    deterministic crop+resize per configured camera in ``__getitem__``.

    LeRobot's per-camera transform loop (``item[cam] = self.image_transforms(
    item[cam])``) is key-blind: it passes a single tensor with no key, so a
    per-camera crop cannot ride inside ``image_transforms`` directly. And
    ``MultiLeRobotDataset.__getitem__`` indexes its sub-datasets via subscript
    (``self._datasets[i][j]``), so a dunder must live on the *class* — an
    instance-level ``__getitem__`` override is ignored. This proxy provides a
    class-level ``__getitem__`` and transparently delegates everything else
    (``hf_dataset``, ``meta``, ``__len__``, ...) to the wrapped sub-dataset, so
    downstream samplers/boundary code see an unchanged object.
    """

    def __init__(
        self,
        inner,
        crop_feature_map,
        base_transform,
        *,
        crop_resize_hw: tuple[int, int] | None = None,
        crop_post_transform=None,
        crop_reference_hw: tuple[int, int] | None = None,
    ):
        # Disable the inner key-blind loop; we apply transforms key-aware here.
        # In the pinned lerobot (0530dd9b) the per-camera transform is applied inside
        # DatasetReader.get_item from reader._image_transforms
        # (lerobot/datasets/dataset_reader.py), NOT from LeRobotDataset.image_transforms.
        # Setting `inner.image_transforms = None` alone leaves the reader's copy
        # live -> the inner loop would still run and double-transform. Use the
        # API that clears BOTH the reader copy and the plain attr.
        inner.clear_image_transforms()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_crop_feature_map", crop_feature_map)
        object.__setattr__(self, "_base_transform", base_transform)
        object.__setattr__(self, "_crop_resize_hw", crop_resize_hw)
        object.__setattr__(self, "_crop_post_transform", crop_post_transform)
        object.__setattr__(self, "_crop_reference_hw", crop_reference_hw)
        object.__setattr__(self, "_image_keys", list(inner.meta.camera_keys))
        # Raw uint8 native passthrough (--uint8-native-images throughput path).
        # When True, __getitem__ skips ALL per-camera transforms and the inner
        # reader returns RAW uint8 native frames; the crop+resize+float machinery
        # then runs batched on GPU via image_preprocess.GpuImagePreprocessor.
        object.__setattr__(self, "_raw_uint8_mode", False)
        # Decode-once frame cache (--decoded-frame-cache): frames arrive from the
        # reader ALREADY crop+resized to policy resolution, so the crop/resize
        # here must be skipped (a crop box applied to a policy-res frame would
        # cut a wrong sub-window) while aug/post transforms still run per-step.
        object.__setattr__(self, "_frames_precropped", False)

    def set_raw_uint8_mode(self, enabled: bool) -> None:
        """Toggle raw uint8 native passthrough on THIS proxy (and its inner reader).

        Flips the inner LeRobotDataset reader's ``return_uint8`` so video frames
        come back as native-resolution uint8 instead of float32, and short-circuits
        the per-camera crop/resize loop in ``__getitem__``. Fails loudly if the
        inner dataset does not expose the expected reader/return_uint8 plumbing —
        a silently-ignored flip would train on the wrong image format.

        NOTE (pickling-isolation invariant): callers that must NOT mutate other
        loaders' view of the dataset flip this on a WORKER-LOCAL pickled copy (see
        ``_RawUint8SubsetView`` in the critic trainer), never on the shared main-process
        proxy. The main-process seed loop flips the shared proxy but restores it in
        a try/finally; persistent holdout workers hold pre-flip pickled copies.
        """
        inner = object.__getattribute__(self, "_inner")
        reader = getattr(inner, "reader", None)
        if reader is None or not hasattr(reader, "_return_uint8"):
            raise AttributeError(
                "set_raw_uint8_mode requires the inner LeRobotDataset to expose a "
                "reader with a `_return_uint8` attribute (lerobot return_uint8 "
                f"plumbing); got inner={type(inner).__name__} reader={type(reader).__name__}. "
                "Refusing to silently ignore the raw-uint8 flip."
            )
        reader._return_uint8 = bool(enabled)
        # Keep the facade attr consistent so any later reader re-creation inherits it.
        if hasattr(inner, "_return_uint8"):
            inner._return_uint8 = bool(enabled)
        object.__setattr__(self, "_raw_uint8_mode", bool(enabled))

    def set_frames_precropped(self, enabled: bool) -> None:
        """Mark reader-served frames as already crop+resized (decoded frame cache)."""
        object.__setattr__(self, "_frames_precropped", bool(enabled))

    def __getitem__(self, idx):
        item = self._inner[idx]
        if self._raw_uint8_mode:
            # Raw path: inner reader already returned native uint8 frames; every
            # camera transform (crop/resize/float/aug) is deferred to the GPU.
            return item
        if self._frames_precropped:
            # Frame-cache path: frames are policy-res float [0,1] already; run
            # only the per-step post transforms (aug + quantize) per camera.
            for cam in self._image_keys:
                post = (
                    self._crop_post_transform
                    if cam in self._crop_feature_map and self._crop_resize_hw is not None
                    else self._base_transform
                )
                if post is not None:
                    item[cam] = post(item[cam])
            return item
        for cam in self._image_keys:
            img = item[cam]
            if cam in self._crop_feature_map:
                if self._crop_resize_hw is not None:
                    img = preprocess_chw_tensor_for_policy(
                        img,
                        target_hw=self._crop_resize_hw,
                        crop_box=self._crop_feature_map[cam],
                        crop_reference_hw=self._crop_reference_hw,
                    )
                    if self._crop_post_transform is not None:
                        img = self._crop_post_transform(img)
                else:
                    img = crop_box_for_chw_tensor(img, self._crop_feature_map[cam])
                    if self._base_transform is not None:
                        img = self._base_transform(img)
            elif self._base_transform is not None:
                img = self._base_transform(img)
            item[cam] = img
        return item

    def __len__(self):
        return len(self._inner)

    def __getattr__(self, name):
        # Delegate any attribute not found on the proxy to the wrapped dataset.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


def install_per_camera_crop(
    dataset,
    crop_feature_map: dict[str, tuple[int, int, int, int]],
    base_transform,
    *,
    crop_resize_hw: tuple[int, int] | None = None,
    crop_post_transform=None,
    crop_reference_hw: tuple[int, int] | None = None,
):
    """Replace each sub-dataset in a ``MultiLeRobotDataset`` with a proxy that
    applies a key-aware deterministic crop+resize path before train-only
    augmentation.

    Cropped cameras with ``crop_resize_hw``: shared
    ``mulligan.real.policy.image_preprocess`` crop+resize, then ``crop_post_transform``.
    This is the train/eval parity path.

    Cropped cameras without ``crop_resize_hw`` are cropped directly,
    ``frame[..., y0:y1, x0:x1]``, then ``base_transform``
    (the resize to the training resolution + any aug).

    Non-cropped cameras: ``base_transform`` only (identical to the stock path).

    ``crop_feature_map`` is keyed by full feature key
    (``observation.images.<role>``). ``base_transform`` is the resize(+aug)
    callable the stock path would have installed as ``image_transforms`` (may be
    ``None``). Returns the list of (possibly proxied) sub-datasets so callers can
    keep their ``sub_datasets`` handle pointing at the live objects.
    """
    sub_datasets = dataset._datasets
    if not crop_feature_map:
        return sub_datasets
    wrapped = [
        _PerCameraCropSubset(
            ds,
            crop_feature_map,
            base_transform,
            crop_resize_hw=crop_resize_hw,
            crop_post_transform=crop_post_transform,
            crop_reference_hw=crop_reference_hw,
        )
        for ds in sub_datasets
    ]
    dataset._datasets = wrapped
    return wrapped


def set_raw_uint8_mode_on_subdatasets(sub_datasets, enabled: bool) -> None:
    """Flip raw uint8 native passthrough on every crop proxy in ``sub_datasets``.

    Every sub-dataset MUST be a :class:`_PerCameraCropSubset` — the
    ``--uint8-native-images`` path defers crop+resize to the GPU and therefore
    requires the per-camera crop proxy to be installed (a plain LeRobotDataset
    would still resize in its reader). Fails loudly on any non-proxy sub-dataset
    rather than silently leaving that camera's frames float32-resized in-worker.
    """
    for i, ds in enumerate(sub_datasets):
        if not isinstance(ds, _PerCameraCropSubset):
            raise TypeError(
                "set_raw_uint8_mode_on_subdatasets requires every sub-dataset to be a "
                f"_PerCameraCropSubset (per-camera crop proxy); sub-dataset[{i}] is "
                f"{type(ds).__name__}. --uint8-native-images needs the crop proxy "
                "installed so crop+resize can be deferred to the GPU."
            )
        ds.set_raw_uint8_mode(enabled)
