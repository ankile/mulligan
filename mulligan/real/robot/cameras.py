"""Camera observation helpers for real-world robot code.

The station camera layout (role -> ZED serial-eye key, never-stored cameras, default
crop boxes) is read from the station config: ``$MULLIGAN_STATION_CONFIG`` if set, else the
packaged ``mulligan/real/robot/station.example.yaml`` (the paper station's roles and crops
with placeholder serials; the same file as ``configs/real/station.example.yaml``). See
``docs/station.md``.
"""

import json
import os
import shutil
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

import yaml

STATION_CONFIG_ENV = "MULLIGAN_STATION_CONFIG"
DEFAULT_STATION_CONFIG_PATH = files("mulligan.real.robot") / "station.example.yaml"
_STATION_ROLES = ("wrist_left", "wrist_right", "side_1", "side_2")


def station_config_path():
    """The station config in effect (``$MULLIGAN_STATION_CONFIG`` or the packaged example)."""
    raw = os.environ.get(STATION_CONFIG_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_STATION_CONFIG_PATH


def load_station_camera_config(path) -> dict:
    """Parse and validate the ``cameras`` block of a station config; fails loud.

    ``path`` is a ``Path`` or an ``importlib.resources`` traversable."""
    if not path.is_file():
        raise FileNotFoundError(
            f"station config {path} not found; set {STATION_CONFIG_ENV} or restore "
            f"{DEFAULT_STATION_CONFIG_PATH}"
        )
    cameras = (yaml.safe_load(path.read_text()) or {}).get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError(f"{path}: missing 'cameras' mapping")
    roles = cameras.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(_STATION_ROLES):
        raise ValueError(f"{path}: cameras.roles must map exactly {list(_STATION_ROLES)}")
    if len(set(roles.values())) != len(roles):
        raise ValueError(f"{path}: cameras.roles maps two roles to the same camera: {roles}")
    excluded = cameras.get("excluded", [])
    if not isinstance(excluded, list) or set(excluded) & set(roles.values()):
        raise ValueError(f"{path}: cameras.excluded must be a list disjoint from cameras.roles")
    crops = cameras.get("default_crops")
    if not isinstance(crops, dict) or not set(crops) <= set(_STATION_ROLES):
        raise ValueError(f"{path}: cameras.default_crops must be keyed by station roles")
    parsed_crops = {}
    for role, box in crops.items():
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)):
            raise ValueError(f"{path}: crop for {role!r} must be [x0, y0, x1, y1] ints, got {box}")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480):
            raise ValueError(f"{path}: crop for {role!r} is not inside the 640x480 stored frame")
        parsed_crops[role] = (x0, y0, x1, y1)
    return {
        "roles": {role: str(roles[role]) for role in _STATION_ROLES},
        "excluded": frozenset(str(key) for key in excluded),
        "default_crops": parsed_crops,
    }


_STATION = load_station_camera_config(station_config_path())

# --- Robot-station camera layout (single source of truth) --------------------
# Role -> DROID camera key "<serial>_<eye>". All collection + eval camera defaults derive
# from this, so serials are never repeated per-script.
STATION_CAMERA_KEYS_BY_ROLE: dict[str, str] = _STATION["roles"]

# Default record/eval camera set for this station: all roles, in order, as the
# comma-joined --camera-keys argument string.
DEFAULT_CAMERA_KEYS = ",".join(STATION_CAMERA_KEYS_BY_ROLE.values())

# Cameras that physically exist at the station but must NEVER be stored (the side
# cameras' unused right eyes).
DEFAULT_EXCLUDED_CAMERA_KEYS = _STATION["excluded"]

# Default per-camera REPLACEMENT crop boxes in STORED-frame pixels (x0, y0, x1, y1),
# half-open: frame[..., y0:y1, x0:x1]. Full frames are always STORED (never pre-cropped)
# so the data stays reusable + supports crop ablation/aug; the box is applied ON THE FLY
# before resize at BOTH train and eval (eval downscales the live native frame to 640x480
# FIRST, then crops in this stored space) and is carried in the trained policy's
# config.json (`camera_crop_boxes`) so eval crops identically. Keyed by camera ROLE
# (e.g. "side_1"), the name datasets store cameras under. The train crop-merge WARNS
# LOUDLY (side_crop.merge_default_crops) when the defaults bind to zero selected cameras,
# so a camera-name mismatch can't silently train an uncropped policy.
STATION_CAMERA_DEFAULT_CROPS: dict[str, tuple[int, int, int, int]] = _STATION["default_crops"]

# Per-station capture/training-input constants. The square the cropped camera view is
# resized to before the policy encoder, and the video decode backend. These are properties
# of the station (camera resolution + the decode stack available on this robot/cluster), not
# of any one task, so they live here next to the camera layout. the DP trainer resolves
# --image-height/--image-width/--video-backend from these when --task is given (explicit CLI
# flags still win); the per-task DP learning recipe (chunk/exec/batch/steps/drop_n_last) lives
# in mulligan.real.lifecycle.tasks.DPTrainingRecipe.
STATION_IMAGE_HW: tuple[int, int] = (224, 224)
STATION_VIDEO_BACKEND: str = "torchcodec"

# Stored-frame resolution (H, W): full station frames are saved at 640x480 (downscaled
# from the 1280x720 native ZED feed). This is the reference space the role-keyed crop
# boxes above (STATION_CAMERA_DEFAULT_CROPS + per-task overrides) live in, so eval must
# downscale the live native frame into it BEFORE cropping.
STATION_STORED_FRAME_HW: tuple[int, int] = (480, 640)

# Inverse of STATION_CAMERA_KEYS_BY_ROLE: raw ZED serial-eye key -> station role.
_ROLE_BY_SERIAL_KEY: dict[str, str] = {
    serial_key: role for role, serial_key in STATION_CAMERA_KEYS_BY_ROLE.items()
}


def serial_key_to_role(serial_key: str) -> str:
    """Map a raw ZED serial-eye key (``"<serial>_left"``) to its station ROLE name
    (``"side_1"``). The single mapping function used at BOTH the collection-save and
    eval-load boundaries so stored + consumed camera names agree. Raises on an unmapped
    serial so a re-cabled/unknown camera can never be silently stored under a serial (or
    a wrong) name (never-fail-silently)."""
    try:
        return _ROLE_BY_SERIAL_KEY[serial_key]
    except KeyError:
        raise KeyError(
            f"camera serial key {serial_key!r} is not in STATION_CAMERA_KEYS_BY_ROLE "
            f"({sorted(_ROLE_BY_SERIAL_KEY)}); add it before storing/consuming this camera."
        ) from None


def role_to_serial_key(role: str) -> str:
    """Map a station ROLE name (``"side_1"``) to its raw ZED serial-eye key
    (``"<serial>_left"``) -- used at eval to read a policy's role-named input from the
    serial-keyed live camera. Raises on an unknown role."""
    try:
        return STATION_CAMERA_KEYS_BY_ROLE[role]
    except KeyError:
        raise KeyError(
            f"camera role {role!r} is not in STATION_CAMERA_KEYS_BY_ROLE "
            f"({sorted(STATION_CAMERA_KEYS_BY_ROLE)})"
        ) from None


def camera_role_serials(serial_keys: list[str] | None = None) -> dict[str, str]:
    """``{role: serial_key}`` provenance map to record in dataset metadata. With no
    argument returns the full station map; given the serial keys actually recorded in a
    dataset, returns only those roles (so the meta reflects what was stored)."""
    if serial_keys is None:
        return dict(STATION_CAMERA_KEYS_BY_ROLE)
    return {serial_key_to_role(k): k for k in serial_keys}


def require_station_role_image_features(image_feature_keys, *, context: str) -> list[str]:
    """Station roles named by a policy's image features; refuses any other camera name.

    Datasets store cameras under station ROLE names and policies consume
    ``observation.images.<role>`` features; the live camera for each role is read by
    serial through :data:`STATION_CAMERA_KEYS_BY_ROLE`. A feature that is not a station
    role has no live camera, so the policy is refused with the offending names."""
    roles: list[str] = []
    unknown: list[str] = []
    for feature_key in image_feature_keys or ():
        cam_name = str(feature_key).removeprefix("observation.images.")
        (roles if cam_name in STATION_CAMERA_KEYS_BY_ROLE else unknown).append(cam_name)
    if unknown:
        raise ValueError(
            f"{context}: image feature(s) {unknown} are not station camera roles "
            f"{sorted(STATION_CAMERA_KEYS_BY_ROLE)}. Policies name cameras by role "
            "(observation.images.<role>); the station config maps each role to its live "
            "camera."
        )
    return roles


def policy_live_camera_keys(policy: object, recording_camera_keys: list[str]) -> list[str]:
    """Physical live camera keys this policy consumes, a subset of the recorded set.

    Datasets always store the FULL station camera set (all 4 streams), but a policy
    usually conditions on only the image features named in its config (e.g. marker_d2
    DP uses ``side_1`` + ``wrist_left``). Both collection (blind DAgger) and eval feed
    only these serials to the policy wrapper so inference never depends on extra
    recorded cameras being tolerated downstream — while recording keeps the full set so
    the dataset is reusable as a datasource for any camera configuration.

    Each role-named feature (``observation.images.side_1``) resolves to its serial via
    :func:`role_to_serial_key`; a feature that is not a station role is refused
    (:func:`require_station_role_image_features`). A policy that declares its live
    serials directly via a ``live_camera_keys`` attribute (e.g. the remote-inference
    client) is honored verbatim. Falls back to the full recorded set when the policy
    declares no image features. Raises loudly if the policy needs a camera the recording
    set does not contain (so a mis-scoped ``--camera-keys`` can't silently starve
    inference)."""
    # Explicit declaration wins: wrappers that expose the exact serial-eye keys they
    # consume (the remote-inference client). Validate against the recording set so a
    # mis-scoped --camera-keys still fails loud.
    explicit = getattr(policy, "live_camera_keys", None)
    if explicit is not None:
        required = list(explicit)
        missing = [key for key in required if key not in recording_camera_keys]
        if missing:
            raise RuntimeError(
                f"Policy requires live camera(s) {missing}, but the recording camera set "
                f"is {recording_camera_keys}. Adjust --camera-keys or the policy config."
            )
        return required
    config = getattr(policy, "config", None)
    image_features = getattr(config, "image_features", {}) or {}
    required: list[str] = []
    for role in require_station_role_image_features(image_features, context="policy config"):
        cam_key = role_to_serial_key(role)
        if cam_key not in required:
            required.append(cam_key)
    if not required:
        return list(recording_camera_keys)
    missing = [key for key in required if key not in recording_camera_keys]
    if missing:
        raise RuntimeError(
            f"Policy requires live camera(s) {missing}, but the recording camera set is "
            f"{recording_camera_keys}. Adjust --camera-keys or the policy config."
        )
    return required


def crop_frame_to_role_view(
    native_frame_bgr,
    cam_key: str,
    *,
    store_hw: tuple[int, int] = STATION_STORED_FRAME_HW,
    crop_boxes: dict[str, tuple[int, int, int, int]] | None = None,
):
    """The crop view a policy receives for a station camera (for previews/monitors).

    Replicates the collection+train+eval pipeline: downscale the NATIVE frame to the stored
    resolution (default 480x640, ``INTER_AREA`` -- matching ``process_image`` so the preview
    is pixel-faithful to the stored/trained frame), then apply the crop box (defined in that
    640x480 stored space). ``crop_boxes`` overrides ``STATION_CAMERA_DEFAULT_CROPS`` per
    ROLE: pass the loaded policy's role-keyed ``camera_crop_boxes`` so the preview shows the
    ACTUAL trained crop (per-task overrides differ from the station defaults). Boxes are
    role-keyed and in stored-frame pixels, as policies carry them. Roles absent from
    ``crop_boxes`` fall back to the station default.
    Returns ``(role, cropped_frame)``, or ``None`` if ``cam_key`` is not a known station
    camera or has no crop box. cv2 only; no hardware. Shared by ``view_cameras`` and the
    live teleop/DAgger camera monitors so the operator's on-screen crop matches what the
    policy sees.
    """
    if cam_key not in _ROLE_BY_SERIAL_KEY:
        return None
    role = _ROLE_BY_SERIAL_KEY[cam_key]
    box = None
    if crop_boxes is not None:
        box = crop_boxes.get(role)
    if box is None:
        box = STATION_CAMERA_DEFAULT_CROPS.get(role)
    if box is None:
        return None
    import cv2

    store_h, store_w = store_hw
    stored = cv2.resize(native_frame_bgr, (store_w, store_h), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = box
    return role, stored[y0:y1, x0:x1]


def select_camera_feature_keys(
    all_camera_keys: list[str],
    *,
    camera_keys: str | None = None,
    camera_filter: str = "_left",
) -> list[str]:
    """Pick which stored cameras a policy CONSUMES from the dataset's full set.

    ``all_camera_keys`` are full feature keys (``observation.images.<role>``). When
    ``camera_keys`` (comma-separated bare camera names or full keys) is given it selects
    EXACTLY those, in order, overriding the suffix ``camera_filter`` -- so a 4-camera
    dataset can be consumed as just ``{side_1, wrist_left}``. A camera repeated in the
    requested list raises (mirrors ``parse_side_crop``'s "specified twice" guard) so a
    duplicate cannot silently feed the same camera twice. Otherwise every camera matching
    the suffix is selected. Raises if a requested camera is absent or nothing matches.

    When falling back to the suffix filter, this MIRRORS ``select_image_camera_keys``'s
    guard: if any station camera (a serial in ``STATION_CAMERA_KEYS_BY_ROLE``, OR a role
    NAME key of it) is present as a feature but would be DROPPED by the suffix filter, it
    raises and tells the caller to pass explicit --camera-keys. For role-named marker_d2
    data the default ``_left`` filter would select only ``wrist_left`` and silently drop
    ``side_1``/``side_2``; the guard forces an explicit selection instead.
    """
    by_serial = {key.removeprefix("observation.images."): key for key in all_camera_keys}
    if camera_keys and camera_keys.strip():
        requested = [
            k.strip().removeprefix("observation.images.")
            for k in camera_keys.split(",")
            if k.strip()
        ]
        seen: set[str] = set()
        for k in requested:
            if k in seen:
                raise ValueError(
                    f"--camera-keys camera {k!r} specified twice; a duplicate would feed "
                    f"the same camera twice. Give each camera once. Got: {requested}"
                )
            seen.add(k)
        missing = [k for k in requested if k not in by_serial]
        if missing:
            raise ValueError(f"--camera-keys {missing} not in dataset cameras {sorted(by_serial)}")
        selected = [by_serial[k] for k in requested]
    else:
        selected = [key for key in all_camera_keys if key.endswith(camera_filter)]
        # Fail loud if the suffix filter would DROP a wanted station camera that is present
        # as a feature -- either keyed by its raw serial (e.g. the wrist's _right eye) or by
        # its ROLE name (role-named marker_d2 features, where a '_left' filter selects only
        # wrist_left and drops side_1/side_2). Mirrors select_image_camera_keys' guard so
        # role-named data can never silently train on a subset of the station cameras.
        station_names = set(STATION_CAMERA_KEYS_BY_ROLE.values()) | set(STATION_CAMERA_KEYS_BY_ROLE)
        selected_serials = {key.removeprefix("observation.images.") for key in selected}
        dropped_station = sorted(
            serial
            for serial in by_serial
            if serial in station_names and serial not in selected_serials
        )
        if dropped_station:
            raise ValueError(
                f"Camera filter '*{camera_filter}' would drop wanted station camera(s) "
                f"{dropped_station} present in the dataset (e.g. role-named marker_d2 features "
                "or a stereo right eye the suffix filter cannot select). Pass explicit "
                f"--camera-keys (station default: {DEFAULT_CAMERA_KEYS}) instead of filter "
                "discovery."
            )
    if not selected:
        raise ValueError(
            f"No cameras selected (camera_keys={camera_keys!r}, filter='*{camera_filter}'). "
            f"Available: {sorted(by_serial)}"
        )
    return selected


def image_camera_keys_from_obs(obs: dict, *, context: str = "camera discovery") -> list[str]:
    """Return sorted camera keys from a robot observation, failing on missing images."""
    if "image" not in obs:
        raise RuntimeError(
            f"Robot observation is missing image data during {context}. "
            f"Available observation keys: {sorted(obs.keys())}"
        )
    image_obs = obs["image"]
    if not isinstance(image_obs, Mapping):
        raise RuntimeError(
            f"Robot observation image data must be a mapping during {context}, "
            f"got {type(image_obs).__name__}."
        )
    return sorted(image_obs.keys())


def select_image_camera_keys(
    obs: dict,
    camera_filter: str,
    *,
    context: str = "camera discovery",
    excluded_camera_keys: set[str] | frozenset[str] = DEFAULT_EXCLUDED_CAMERA_KEYS,
) -> tuple[list[str], list[str]]:
    """Return (selected, available) camera keys matching a suffix filter."""
    all_cams = image_camera_keys_from_obs(obs, context=context)
    selected = [k for k in all_cams if k.endswith(camera_filter) and k not in excluded_camera_keys]
    if not selected:
        raise RuntimeError(
            f"No non-excluded cameras match filter '*{camera_filter}'. "
            f"Available: {all_cams}; excluded: {sorted(excluded_camera_keys)}"
        )
    # Fail loud if the suffix filter would DROP a wanted station camera that is
    # physically present (e.g. the wrist's _right eye under a '_left' filter). The
    # single-suffix filter cannot select both stereo eyes, so the station must pass
    # explicit --camera-keys; silently recording a subset would mismatch training.
    dropped_station = [
        key
        for key in STATION_CAMERA_KEYS_BY_ROLE.values()
        if key in all_cams and key not in selected
    ]
    if dropped_station:
        raise RuntimeError(
            f"Camera filter '*{camera_filter}' would drop wanted station camera(s) "
            f"{dropped_station} that are physically present (the suffix filter cannot select both "
            f"stereo eyes). Pass explicit --camera-keys (station default: {DEFAULT_CAMERA_KEYS}) "
            "instead of filter discovery."
        )
    return selected, all_cams


def remove_excluded_camera_features_from_lerobot_dataset(
    dataset_path: Path,
    *,
    excluded_camera_keys: set[str] | frozenset[str] = DEFAULT_EXCLUDED_CAMERA_KEYS,
) -> list[str]:
    """Remove excluded camera video features from an existing local LeRobot dataset.

    Every file is replaced atomically, and meta/info.json is rewritten last: an
    interrupted run leaves the feature listed in info.json, so the next run redoes the
    remaining steps.
    """
    removed_features: list[str] = []
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return removed_features

    info = json.loads(info_path.read_text())
    features = info.get("features")
    if not isinstance(features, dict):
        raise RuntimeError(f"{info_path} is missing a features object")

    excluded_feature_keys = [
        f"observation.images.{camera_key}" for camera_key in sorted(excluded_camera_keys)
    ]
    for feature_key in excluded_feature_keys:
        if feature_key in features:
            del features[feature_key]
            removed_features.append(feature_key)

    if not removed_features:
        return removed_features

    _drop_excluded_camera_columns_from_parquet(dataset_path, removed_features)

    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for feature_key in removed_features:
            stats.pop(feature_key, None)
        _atomic_write_text(stats_path, json.dumps(stats, indent=4) + "\n")

    for feature_key in removed_features:
        for media_root in ("videos", "images"):
            media_path = dataset_path / media_root / feature_key
            if media_path.exists():
                shutil.rmtree(media_path)

    _atomic_write_text(info_path, json.dumps(info, indent=4) + "\n")
    return removed_features


def _atomic_tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = _atomic_tmp_path(path)
    with open(tmp_path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _drop_excluded_camera_columns_from_parquet(
    dataset_path: Path,
    removed_features: list[str],
) -> None:
    parquet_paths = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        return

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to verify/remove excluded camera columns from "
            f"{dataset_path / 'data'}"
        ) from exc

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        columns_to_drop = [name for name in removed_features if name in table.column_names]
        if not columns_to_drop:
            continue
        keep_columns = [name for name in table.column_names if name not in columns_to_drop]
        tmp_path = _atomic_tmp_path(parquet_path)
        with open(tmp_path, "wb") as f:
            pq.write_table(table.select(keep_columns), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, parquet_path)
