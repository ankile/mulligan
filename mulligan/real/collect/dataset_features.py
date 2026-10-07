"""
Shared supplementary feature definitions for DROID real-robot datasets.

DROID's env.step(action) returns an `action_info` dict containing the
**commanded** action converted to ALL four action spaces via IK. Recording
these alongside the primary cartesian_velocity action lets us train policies
in any action space (joint_velocity, joint_position, cartesian_position)
without approximation from state differences.

See also: observation.state.* features that record the **realized** robot
state in decomposed fields (cartesian_position, joint_position, etc.).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mulligan.data.constants import DataSource, EpisodeOutcome


def _camera_feature_name(cam_data_key: str, camera_name_fn: Callable[[str], str] | None) -> str:
    """Stored LeRobot feature name for a live camera buffer key (``image_<key>``).

    ``camera_name_fn`` renames the camera AFTER stripping the ``image_`` prefix; the
    station collectors and eval pass ``serial_key_to_role`` so datasets store cameras
    under ROLE names. Without it the key is stored unchanged. Used by BOTH the schema
    build and the per-frame writer through this ONE function, so the declared features and
    the written frame keys can never disagree (the save-side consistency invariant)."""
    cam_name = cam_data_key.removeprefix("image_")
    if camera_name_fn is not None:
        cam_name = camera_name_fn(cam_name)
    return f"observation.images.{cam_name}"


def camera_feature_name(cam_data_key: str, camera_name_fn: Callable[[str], str] | None) -> str:
    """Public wrapper for the stored feature name of an ``image_<camera>`` buffer key."""
    return _camera_feature_name(cam_data_key, camera_name_fn)


def _ensure_lerobot_real_patches_loaded() -> None:
    """Load Mulligan's LeRobot monkey-patches for write-mode resume helpers."""
    import importlib

    importlib.import_module("mulligan.real.policy.lerobot_patches")


CAMERA_ROLE_SERIALS_SIDECAR = "camera_role_serials.json"


def _load_camera_role_serials(root: Path) -> dict | None:
    """Return the stored role->serial provenance map, or ``None`` if absent.

    Reads the Mulligan-owned sidecar ``meta/camera_role_serials.json`` first; without it,
    reads a ``camera_role_serials`` key inside ``meta/info.json``. That read is a RAW JSON
    read on purpose: lerobot 0.5.2 parses ``info.json`` into a typed ``DatasetInfo`` that
    silently drops unknown keys, so the value is only visible by reading the file directly.
    """
    import json

    sidecar = root / "meta" / CAMERA_ROLE_SERIALS_SIDECAR
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info_map = json.loads(info_path.read_text()).get("camera_role_serials")
        if info_map is not None:
            return info_map
    return None


def record_or_verify_camera_role_serials(dataset, camera_data_keys: list[str]) -> None:
    """Persist (on a fresh dataset) or verify (on resume/append) the serial<->role
    camera provenance map in the ``meta/camera_role_serials.json`` sidecar.

    Datasets store cameras under ROLE names; the map (e.g.
    ``{"side_1": "<serial>_left", ...}``) records which physical ZED serial filled each
    role, so role-named data can be mixed across identically-set-up stations while the
    raw provenance is never lost.

    The map lives in a Mulligan-owned sidecar, NOT in ``meta/info.json``: lerobot 0.5.2
    parses ``info.json`` into a typed ``DatasetInfo`` that drops unknown keys on load
    and re-serializes only its declared fields on every ``save_episode``, so a custom
    key written into ``info.json`` would be silently clobbered each episode. A map found
    only in ``info.json`` (:func:`_load_camera_role_serials`) is copied to the sidecar on
    first touch so the provenance survives lerobot's rewrites.

    On resume it FAILS LOUD if the live serial<->role map no longer matches the one
    stored at creation (a re-cabled camera / wrong station), which would otherwise
    silently mislabel which physical camera produced each role's frames.
    """
    import json

    from mulligan.real.robot.cameras import camera_role_serials

    serial_keys = [k.removeprefix("image_") for k in camera_data_keys]
    live_map = camera_role_serials(serial_keys)  # role -> serial; raises on unknown serial

    root = Path(dataset.root)
    sidecar = root / "meta" / CAMERA_ROLE_SERIALS_SIDECAR
    stored_map = _load_camera_role_serials(root)
    if stored_map is None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(live_map, indent=2, sort_keys=True))
    elif stored_map != live_map:
        raise RuntimeError(
            f"camera_role_serials in existing dataset {stored_map} != live cabling "
            f"{live_map}; refusing to append role-named frames under a re-cabled / "
            "wrong-station serial->role mapping"
        )
    elif not sidecar.exists():
        # The map was read from info.json: copy it to the durable sidecar so lerobot's
        # per-episode info.json rewrite cannot drop it.
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(stored_map, indent=2, sort_keys=True))

    # Independent of the provenance map, every role the live cameras map to MUST already
    # be a stored feature, or save_episode_to_dataset frames would fail validate_frame.
    features = dataset.meta.info.features
    expected = {f"observation.images.{role}" for role in live_map}
    stored = {k for k in features if k.startswith("observation.images.")}
    missing = expected - stored
    if missing:
        raise RuntimeError(
            f"role camera feature(s) {sorted(missing)} not in existing dataset features "
            f"{sorted(stored)}; the dataset was created with a different camera set"
        )


# ---------------------------------------------------------------------------
# Supplementary feature schemas (merged into the LeRobot features dict)
# ---------------------------------------------------------------------------

SUPPLEMENTARY_ACTION_FEATURES = {
    "action.cartesian_velocity": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "action.cartesian_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "action.joint_velocity": {
        "dtype": "float32",
        "shape": (7,),
        "names": [f"joint_{i}" for i in range(7)],
    },
    "action.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": [f"joint_{i}" for i in range(7)],
    },
    "action.gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper"],
    },
    "action.gripper_velocity": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper"],
    },
}

SUPPLEMENTARY_OBS_FEATURES = {
    "observation.state.cartesian_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "observation.state.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": [f"joint_{i}" for i in range(7)],
    },
    "observation.state.joint_velocity": {
        "dtype": "float32",
        "shape": (7,),
        "names": [f"joint_{i}" for i in range(7)],
    },
    "observation.state.cartesian_velocity": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "observation.state.gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper"],
    },
}


@dataclass(frozen=True)
class FrankaTelemetrySpec:
    """One optional Franka telemetry signal from DROID ``obs["robot_state"]``.

    ``episode_key`` is the per-episode buffer key; ``feature_name`` is the stored
    LeRobot column; ``obs_state_key`` is the ``robot_state`` field it reads.
    Adding a signal is a single entry in :data:`FRANKA_TELEMETRY_SPECS` — schema,
    init, NaN-safe per-frame write, terminal padding, length validation and the
    existing-dataset NaN backfill all iterate the registry.
    """

    episode_key: str
    feature_name: str
    obs_state_key: str
    shape: tuple[int, ...]
    names: tuple[str, ...]


# Every Franka telemetry sensor we record. ALL are written on EVERY real-robot save
# path (collection + eval) and are ALWAYS declared as columns; when a given
# Polymetis/DROID server does not expose one, its frames are filled with NaN (not
# omitted) so that every real dataset shares one schema and is freely mixable as a
# datasource regardless of which server produced it. NaN — never 0 — marks "no
# measurement" (0 is a valid force/torque reading).
_JOINT_NAMES = tuple(f"joint_{i}" for i in range(7))
FRANKA_TELEMETRY_SPECS: tuple[FrankaTelemetrySpec, ...] = (
    # libfranka joint-space external torque estimate (model-based), robot_state
    # ["motor_torques_external"].
    FrankaTelemetrySpec(
        "franka_motor_torques_external",
        "telemetry.franka.motor_torques_external",
        "motor_torques_external",
        (7,),
        _JOINT_NAMES,
    ),
    # libfranka's O_F_ext_hat_K: the robot's model-based external wrench estimate at
    # the EE/stiffness frame in base coordinates ([Fx,Fy,Fz,Mx,My,Mz]); surfaced
    # through the polymetis proto + DROID get_robot_state as robot_state["ee_wrench"].
    FrankaTelemetrySpec(
        "franka_ee_wrench",
        "telemetry.franka.ee_wrench",
        "ee_wrench",
        (6,),
        ("fx", "fy", "fz", "mx", "my", "mz"),
    ),
    # Measured joint torques (the sensed motor torques), robot_state
    # ["motor_torques_measured"].
    FrankaTelemetrySpec(
        "franka_motor_torques_measured",
        "telemetry.franka.motor_torques_measured",
        "motor_torques_measured",
        (7,),
        _JOINT_NAMES,
    ),
    # Controller's model-computed joint torques (gravity/coriolis/desired), robot_state
    # ["joint_torques_computed"].
    FrankaTelemetrySpec(
        "franka_joint_torques_computed",
        "telemetry.franka.joint_torques_computed",
        "joint_torques_computed",
        (7,),
        _JOINT_NAMES,
    ),
)

# Aliases for the two original torque signals (imported by tests / callers).
FRANKA_EXTERNAL_TORQUE_EPISODE_KEY = "franka_motor_torques_external"
FRANKA_EXTERNAL_TORQUE_FEATURE = "telemetry.franka.motor_torques_external"
FRANKA_EE_WRENCH_EPISODE_KEY = "franka_ee_wrench"
FRANKA_EE_WRENCH_FEATURE = "telemetry.franka.ee_wrench"

TELEMETRY_FEATURES = {
    spec.feature_name: {
        "dtype": "float32",
        "shape": spec.shape,
        "names": list(spec.names),
    }
    for spec in FRANKA_TELEMETRY_SPECS
}

# episode_data key -> dataset feature name for every Franka telemetry column. Drives
# schema build, NaN backfill, per-frame writing, and length validation uniformly.
TELEMETRY_EPISODE_KEY_TO_FEATURE = {
    spec.episode_key: spec.feature_name for spec in FRANKA_TELEMETRY_SPECS
}

ALL_SUPPLEMENTARY_FEATURES = {
    **SUPPLEMENTARY_ACTION_FEATURES,
    **SUPPLEMENTARY_OBS_FEATURES,
}


def build_real_lerobot_features(
    episode_data: dict,
    cam_data_keys: list[str],
    extra_features: dict | None = None,
    *,
    camera_name_fn: Callable[[str], str] | None = None,
) -> dict:
    """Build the shared real-robot LeRobot feature schema.

    ``camera_name_fn`` renames each camera for its stored feature; the station
    collectors and eval pass ``serial_key_to_role`` so the schema is role-named. Must be
    passed identically to ``save_episode_to_dataset`` so schema == frame keys."""
    obs_shape = episode_data["observations"][0].shape[0]
    action_shape = episode_data["actions"][0].shape[0]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs_shape,),
            "names": [
                "cart_pos_x",
                "cart_pos_y",
                "cart_pos_z",
                "cart_rot_x",
                "cart_rot_y",
                "cart_rot_z",
                "gripper_position",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_shape,),
            "names": [
                "vel_x",
                "vel_y",
                "vel_z",
                "vel_roll",
                "vel_pitch",
                "vel_yaw",
                "gripper_action",
            ],
        },
        "steps_to_go": {"dtype": "int64", "shape": (1,), "names": ["steps_to_go"]},
        "source": {"dtype": "int64", "shape": (1,), "names": ["source_id"]},
        "intervention": {"dtype": "int64", "shape": (1,), "names": ["intervention_flag"]},
        "success": {"dtype": "int64", "shape": (1,), "names": ["success_flag"]},
        "is_valid": {"dtype": "int64", "shape": (1,), "names": ["is_valid_flag"]},
        "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "done": {"dtype": "int64", "shape": (1,), "names": ["done_flag"]},
    }
    if extra_features is not None:
        features.update(extra_features)
    features.update(ALL_SUPPLEMENTARY_FEATURES)
    # Every Franka telemetry column is ALWAYS declared (NaN-filled when the server
    # does not expose it), so all real datasets share one schema and are mixable.
    for spec in FRANKA_TELEMETRY_SPECS:
        features[spec.feature_name] = TELEMETRY_FEATURES[spec.feature_name]

    for cam_data_key in cam_data_keys:
        img_shape = episode_data[cam_data_key][0].shape
        features[_camera_feature_name(cam_data_key, camera_name_fn)] = {
            "dtype": "video",
            "shape": img_shape,
            "names": ["height", "width", "channels"],
        }
    return features


def _copy_feature_info(feature_info: dict) -> dict:
    """Return a mutable copy preserving tuple-valued shapes."""
    copied = dict(feature_info)
    if "shape" in copied:
        copied["shape"] = tuple(copied["shape"])
    if "names" in copied and copied["names"] is not None:
        copied["names"] = list(copied["names"])
    return copied


def _assert_telemetry_feature_matches(feature_name: str, feature_info: dict) -> None:
    expected = TELEMETRY_FEATURES[feature_name]
    got = _copy_feature_info(feature_info)
    want = _copy_feature_info(expected)
    if got != want:
        raise RuntimeError(
            f"Existing dataset feature {feature_name!r} has schema "
            f"{got}, expected {want}; refusing to append telemetry."
        )


def ensure_dataset_can_store_episode_telemetry(
    dataset,
    episode_data: dict,
    *,
    dataset_path: str | Path,
    dataset_name: str,
    image_writer_threads: int = 4,
    streaming_encoding: bool = False,
):
    """Upgrade an existing LeRobotDataset and return an appendable handle.

    If the current episode carries any optional Franka telemetry column (e.g.
    ``franka_motor_torques_external`` or ``franka_ee_wrench``) that the existing
    dataset was created before, backfill every old frame of each such column with
    a NaN vector of the column's shape and reload the dataset with the upgraded
    schema. NaN is used deliberately: zero is a valid force/torque measurement,
    while old frames have no measurement.

    FF lerobot's bare ``LeRobotDataset(...)`` constructor is read-only. Existing
    collection/eval save paths intentionally load first for schema inspection, so
    this helper also reopens with ``resume()`` whenever the passed dataset has no
    writer, even if no telemetry schema upgrade was needed.

    Mulligan's older real-robot save paths buffer a whole episode in memory and only
    call ``add_frame`` after the trajectory ends. LeRobot streaming video
    encoding is designed for live, frame-rate-paced ``add_frame`` calls; burst
    feeding a buffered episode can overflow its encoder queue and drop frames.
    Keep ``streaming_encoding=False`` for those buffered callers. Collectors
    that write frames during the control loop should pass ``streaming_encoding=True``.
    """
    root = Path(dataset_path)

    def ensure_appendable(current_dataset):
        if getattr(current_dataset, "writer", None) is not None:
            return current_dataset
        _ensure_lerobot_real_patches_loaded()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset.resume(
            repo_id=dataset_name,
            root=str(root),
            image_writer_threads=image_writer_threads,
            streaming_encoding=streaming_encoding,
        )

    present_telemetry = [
        feat_name
        for ep_key, feat_name in TELEMETRY_EPISODE_KEY_TO_FEATURE.items()
        if ep_key in episode_data
    ]
    if not present_telemetry:
        return ensure_appendable(dataset)

    missing_features = []
    for feat_name in present_telemetry:
        feature_info = dataset.meta.features.get(feat_name)
        if feature_info is not None:
            _assert_telemetry_feature_matches(feat_name, feature_info)
        else:
            missing_features.append(feat_name)

    if not missing_features:
        return ensure_appendable(dataset)

    data_dir = root / "data"
    parquet_paths = sorted(data_dir.glob("*/*.parquet"))
    total_frames = int(dataset.meta.total_frames)
    if total_frames > 0 and not parquet_paths:
        raise RuntimeError(
            f"Dataset {root} reports {total_frames} frames but has no data parquet files; "
            f"cannot backfill {missing_features}."
        )

    print(f"Upgrading existing dataset at {root} with {missing_features} (old frames = NaN).")

    import datasets
    import pandas as pd
    import pyarrow.parquet as pq

    _ensure_lerobot_real_patches_loaded()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.io_utils import embed_images, write_info
    from lerobot.datasets.feature_utils import get_hf_features_from_features

    upgraded_features = {
        key: _copy_feature_info(value) for key, value in dataset.meta.features.items()
    }
    for feat_name in missing_features:
        upgraded_features[feat_name] = _copy_feature_info(TELEMETRY_FEATURES[feat_name])
    hf_features = get_hf_features_from_features(upgraded_features)
    has_image_features = any(ft["dtype"] == "image" for ft in upgraded_features.values())

    rows_seen = 0
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path).reset_index(drop=True)
        rows_seen += len(df)
        for feat_name in missing_features:
            if feat_name not in df.columns:
                shape = tuple(TELEMETRY_FEATURES[feat_name]["shape"])
                df[feat_name] = [np.full(shape, np.nan, dtype=np.float32) for _ in range(len(df))]

        hf_dataset = datasets.Dataset.from_dict(
            df.to_dict(orient="list"),
            features=hf_features,
            split="train",
        )
        if has_image_features:
            hf_dataset = embed_images(hf_dataset)
        table = hf_dataset.with_format("arrow")[:]
        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        writer = pq.ParquetWriter(
            tmp_path,
            schema=table.schema,
            compression="snappy",
            use_dictionary=True,
        )
        writer.write_table(table)
        writer.close()
        tmp_path.replace(parquet_path)

    if rows_seen != total_frames:
        raise RuntimeError(
            f"Backfilled {rows_seen} parquet rows for {root}, but metadata reports "
            f"{total_frames} total_frames; refusing to update dataset metadata."
        )

    dataset.meta.info.features = upgraded_features
    write_info(dataset.meta.info, root)

    # Reload in WRITE mode: lerobot 0.5.2's DatasetReader/DatasetWriter split
    # makes a plain ``LeRobotDataset(...)`` read-only, so the caller's
    # subsequent ``add_frame`` for the new episode would raise. ``resume()``
    # rebuilds the DatasetWriter for appending against the just-upgraded schema.
    return LeRobotDataset.resume(
        repo_id=dataset_name,
        root=str(root),
        image_writer_threads=image_writer_threads,
        streaming_encoding=streaming_encoding,
    )


# ---------------------------------------------------------------------------
# Helpers for extracting / building supplementary fields
# ---------------------------------------------------------------------------


def extract_action_info(action_info: dict) -> dict[str, np.ndarray]:
    """Extract commanded action fields from DROID action_info dict.

    Returns a flat dict of numpy arrays keyed by action_info_* names,
    suitable for appending to episode_data lists.

    All keys are required — missing keys raise KeyError (no silent zero-fill).
    """
    return {
        "action_info_cartesian_velocity": np.array(
            action_info["cartesian_velocity"], dtype=np.float32
        ),
        "action_info_cartesian_position": np.array(
            action_info["cartesian_position"], dtype=np.float32
        ),
        "action_info_joint_velocity": np.array(action_info["joint_velocity"], dtype=np.float32),
        "action_info_joint_position": np.array(action_info["joint_position"], dtype=np.float32),
        "action_info_gripper_position": np.array(
            [action_info["gripper_position"]], dtype=np.float32
        ),
        "action_info_gripper_velocity": np.array(
            [action_info["gripper_velocity"]], dtype=np.float32
        ),
    }


def build_canonical_action(action_info: dict) -> np.ndarray:
    """Build 7D canonical action: cartesian_velocity(6) + gripper_velocity(1).

    This is the standard action representation regardless of which action
    space was used to control the robot.
    """
    cart_vel = np.array(action_info["cartesian_velocity"], dtype=np.float32)
    grip_vel = np.array([action_info["gripper_velocity"]], dtype=np.float32)
    return np.concatenate([cart_vel, grip_vel])


# NOTE: the canonical ``action`` column is ALWAYS cartesian_velocity + gripper
# (build_canonical_action) for EVERY arm — velocity, position, or UMI-relative — so it
# stays consistent and matches its declared feature names (vel_x..vel_yaw,
# gripper_action) even in a mixed-arm eval dataset. A position/relative arm's absolute
# pose command is NOT written to the canonical column (that would put a pose under
# velocity names); it is preserved in the typed ``action.cartesian_position`` +
# ``action.gripper_position`` columns, which is where position/relative training reads
# it from.


# Keys used in episode_data for action_info fields
ACTION_INFO_KEYS = [
    "action_info_cartesian_velocity",
    "action_info_cartesian_position",
    "action_info_joint_velocity",
    "action_info_joint_position",
    "action_info_gripper_position",
    "action_info_gripper_velocity",
]


def init_supplementary_lists(episode_data: dict) -> None:
    """Initialize empty lists in episode_data for all supplementary fields.

    ALL Franka telemetry lists are created unconditionally: every real-robot save
    path records the full sensor union, NaN-filling any signal the live server does
    not expose (see :func:`append_franka_telemetry`), so datasets share one schema.
    """
    for key in ACTION_INFO_KEYS:
        episode_data[key] = []
    episode_data["joint_velocities"] = []
    episode_data["cartesian_velocities"] = []
    for spec in FRANKA_TELEMETRY_SPECS:
        episode_data[spec.episode_key] = []


def append_action_info(episode_data: dict, action_info: dict) -> None:
    """Extract and append action_info fields to episode_data lists."""
    extracted = extract_action_info(action_info)
    for key in ACTION_INFO_KEYS:
        episode_data[key].append(extracted[key])


def append_joint_velocities(episode_data: dict, obs: dict) -> None:
    """Append joint velocities from an observation to episode_data."""
    episode_data["joint_velocities"].append(
        np.array(obs["robot_state"]["joint_velocities"], dtype=np.float32)
    )


def append_franka_telemetry(episode_data: dict, obs: dict) -> None:
    """Append EVERY Franka telemetry signal for one frame, NaN-safe.

    For each signal in :data:`FRANKA_TELEMETRY_SPECS`: if the live DROID server
    exposes it in ``obs["robot_state"]``, append the measured vector (shape-checked
    loud); otherwise append an all-NaN vector of the signal's shape. NaN — never 0 —
    marks "no measurement" so that a column is present in every dataset regardless of
    which server produced it, while still being distinguishable from a genuine zero
    reading. ``init_supplementary_lists`` must have created the per-signal lists.
    """
    robot_state = obs["robot_state"]
    for spec in FRANKA_TELEMETRY_SPECS:
        if spec.obs_state_key in robot_state:
            value = np.asarray(robot_state[spec.obs_state_key], dtype=np.float32)
            if value.shape != spec.shape:
                raise ValueError(
                    f"Expected {spec.shape} robot_state.{spec.obs_state_key}; "
                    f"got shape {value.shape}"
                )
        else:
            value = np.full(spec.shape, np.nan, dtype=np.float32)
        episode_data[spec.episode_key].append(value)


_WARNED_MISSING_TELEMETRY: set[str] = set()


def warn_missing_franka_telemetry_once(obs: dict) -> None:
    """Print a one-time warning per signal that the live server does not expose.

    The signal's column is still recorded as NaN (so the schema is uniform); this
    only surfaces to the operator that no real measurement is being captured for it.
    """
    robot_state = obs.get("robot_state", {})
    for spec in FRANKA_TELEMETRY_SPECS:
        if (
            spec.obs_state_key not in robot_state
            and spec.episode_key not in _WARNED_MISSING_TELEMETRY
        ):
            _WARNED_MISSING_TELEMETRY.add(spec.episode_key)
            print(
                f"WARNING: DROID server does not expose robot_state[{spec.obs_state_key!r}]; "
                f"recording column {spec.feature_name!r} as NaN for this session."
            )


def _robot_state_timestamp_seconds(obs: dict) -> float:
    """Extract the DROID robot-state timestamp in seconds."""
    timestamp = obs["timestamp"]["robot_state"]
    if "robot_timestamp_seconds" in timestamp and "robot_timestamp_nanos" in timestamp:
        return float(timestamp["robot_timestamp_seconds"]) + 1e-9 * float(
            timestamp["robot_timestamp_nanos"]
        )
    if "read_end" in timestamp:
        return float(timestamp["read_end"]) * 1e-3
    raise KeyError(
        "Observation is missing timestamp.robot_state robot_timestamp_seconds/"
        "robot_timestamp_nanos and read_end; cannot derive cartesian_velocity"
    )


def compute_cartesian_velocity_from_observations(previous_obs: dict, obs: dict) -> np.ndarray:
    """Compute realized Cartesian velocity from consecutive DROID observations."""
    previous_pose = np.array(
        previous_obs["robot_state"]["cartesian_position"],
        dtype=np.float64,
    )
    current_pose = np.array(obs["robot_state"]["cartesian_position"], dtype=np.float64)
    if previous_pose.shape != (6,) or current_pose.shape != (6,):
        raise ValueError(
            "Expected 6D robot_state.cartesian_position in consecutive observations; "
            f"got previous={previous_pose.shape}, current={current_pose.shape}"
        )

    dt = _robot_state_timestamp_seconds(obs) - _robot_state_timestamp_seconds(previous_obs)
    if dt <= 0.0:
        raise ValueError(f"Non-positive robot_state timestamp delta while deriving velocity: {dt}")

    delta = current_pose - previous_pose
    delta[3:] = (delta[3:] + np.pi) % (2.0 * np.pi) - np.pi
    return (delta / dt).astype(np.float32)


def append_cartesian_velocities(
    episode_data: dict,
    obs: dict,
    *,
    previous_obs: dict | None = None,
) -> None:
    """Append measured Cartesian velocity from observation or adjacent poses."""
    robot_state = obs["robot_state"]
    if "cartesian_velocity" in robot_state:
        cartesian_velocity = np.array(robot_state["cartesian_velocity"], dtype=np.float32)
    else:
        if previous_obs is None:
            raise KeyError(
                "Observation is missing robot_state.cartesian_velocity. Pass previous_obs to "
                "derive observation.state.cartesian_velocity from consecutive poses."
            )
        cartesian_velocity = compute_cartesian_velocity_from_observations(previous_obs, obs)

    episode_data["cartesian_velocities"].append(cartesian_velocity)


def pad_supplementary_terminal(episode_data: dict) -> None:
    """Pad action_info fields for the terminal frame (repeat last values)."""
    for key in ACTION_INFO_KEYS:
        episode_data[key].append(episode_data[key][-1].copy())


def build_supplementary_frame_fields(episode_data: dict, i: int) -> dict:
    """Build the supplementary action.* and observation.state.* fields for frame i.

    Returns a dict that can be merged into the frame dict before add_frame().
    """
    fields = {
        # Commanded actions (from IK solver)
        "action.cartesian_velocity": episode_data["action_info_cartesian_velocity"][i],
        "action.cartesian_position": episode_data["action_info_cartesian_position"][i],
        "action.joint_velocity": episode_data["action_info_joint_velocity"][i],
        "action.joint_position": episode_data["action_info_joint_position"][i],
        "action.gripper_position": episode_data["action_info_gripper_position"][i],
        "action.gripper_velocity": episode_data["action_info_gripper_velocity"][i],
        # Realized observations (measured state)
        "observation.state.cartesian_position": episode_data["observations"][i][:6].astype(
            np.float32
        ),
        "observation.state.cartesian_velocity": episode_data["cartesian_velocities"][i].astype(
            np.float32
        ),
        "observation.state.joint_position": episode_data["joint_positions"][i].astype(np.float32),
        "observation.state.joint_velocity": episode_data["joint_velocities"][i].astype(np.float32),
        "observation.state.gripper_position": episode_data["observations"][i][6:7].astype(
            np.float32
        ),
    }
    for ep_key, feat_name in TELEMETRY_EPISODE_KEY_TO_FEATURE.items():
        if ep_key in episode_data:
            fields[feat_name] = episode_data[ep_key][i].astype(np.float32)
    return fields


# ---------------------------------------------------------------------------
# Episode finalisation & dataset saving
# ---------------------------------------------------------------------------


def compute_intervention_flags(sources: list[int]) -> list[int]:
    """Compute intervention flags marking policy->human control switches.

    The flag is 1 at the last timestep before control switches from policy
    to human, and 0 everywhere else.

    Args:
        sources: List of source IDs (0=policy, 1=human) for each timestep

    Returns:
        List of intervention flags (0 or 1) for each timestep
    """
    intervention_flags = [0] * len(sources)

    for i in range(len(sources) - 1):
        if sources[i] == 0 and sources[i + 1] == 1:
            intervention_flags[i] = 1

    return intervention_flags


def _assert_episode_data_lengths(
    episode_data: dict,
    *,
    episode_length: int,
    camera_keys: list[str],
) -> None:
    """Fail before writing if any per-frame real-data list lost alignment."""
    required_keys = [
        "observations",
        "actions",
        "steps_to_go",
        "rewards",
        "dones",
        "joint_positions",
        "joint_velocities",
        "cartesian_velocities",
        *ACTION_INFO_KEYS,
        *camera_keys,
    ]
    for ep_key in TELEMETRY_EPISODE_KEY_TO_FEATURE:
        if ep_key in episode_data:
            required_keys.append(ep_key)

    for key in required_keys:
        values = episode_data[key]
        if len(values) != episode_length:
            raise RuntimeError(
                f"episode_data[{key!r}] has {len(values)} frame(s), but actions has "
                f"{episode_length}; refusing to save a misaligned real-robot episode"
            )


def _metadata_scalar(value):
    """Return a scalar from LeRobot metadata cells stored as scalar/list/ndarray."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected single-value metadata cell, got {value!r}")
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise RuntimeError(f"Expected single-value metadata ndarray, got shape {value.shape}")
        return value.reshape(-1)[0].item()
    if hasattr(value, "item"):
        return value.item()
    return value


def _latest_lerobot_episode_metadata(dataset):
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return None
    latest = getattr(meta, "latest_episode", None)
    if latest is not None:
        return latest
    episodes = getattr(meta, "episodes", None)
    if episodes is not None and len(episodes) > 0:
        return episodes[-1]
    return None


def _assert_latest_video_spans_match_episode_length(dataset, *, episode_length: int) -> None:
    """Ensure LeRobot video timestamps describe exactly the rows just saved."""
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return

    video_keys = list(getattr(meta, "video_keys", []))
    if not video_keys:
        return

    episode = _latest_lerobot_episode_metadata(dataset)
    if episode is None:
        raise RuntimeError(
            "LeRobot dataset has video keys but no latest episode metadata after save; "
            "cannot verify row/video alignment"
        )

    fps = int(meta.fps)
    episode_index = int(_metadata_scalar(episode["episode_index"]))
    for video_key in video_keys:
        from_key = f"videos/{video_key}/from_timestamp"
        to_key = f"videos/{video_key}/to_timestamp"
        from_ts = float(_metadata_scalar(episode[from_key]))
        to_ts = float(_metadata_scalar(episode[to_key]))
        span = round(to_ts * fps) - round(from_ts * fps)
        if span != episode_length:
            raise RuntimeError(
                f"LeRobot video metadata mismatch after saving episode {episode_index}: "
                f"{video_key} spans {span} frame(s) from timestamps, but data rows have "
                f"{episode_length}. Refusing to continue because this dataset would fail "
                "fast splitting or require padding."
            )


def finalize_episode_data(
    episode_data: dict,
    obs: dict,
    is_success: bool,
    is_terminal: bool,
    subtask_frames: Sequence[int] = (),
) -> None:
    """Set reward/done on last valid frame and append padded terminal values.

    The caller must have already appended the final observation,
    joint_positions, and camera images to episode_data before calling this.

    This function:
    1. Sets reward/done on the last VALID frame (the action that achieved
       the outcome) so that RL code looking at is_valid=1 rows sees the
       correct signal.
    2. Appends a padded terminal row (is_valid=0) that copies the last
       action, action_info, reward, and done values.
    3. Writes a ``reward=1.0`` spike (``done`` untouched) at every operator
       sub-goal mark in ``subtask_frames`` -- byte-identical to what the outcome
       review (``mulligan.tools.outcome_review``) writes for a reviewed subtask mark.
       Marks must lie strictly before the outcome frame; a mark ON the outcome
       frame is legal only for a timeout (no terminal transition to conflict
       with), mirroring ``subtask_mark_count_error``.
       The spike is written after padding so the padded row keeps the terminal
       reward, never a copied spike.
    4. Computes steps_to_go for the whole episode.

    Args:
        episode_data: dict of lists being built during the episode
        obs: final observation dict (for joint velocities)
        is_success: whether the episode was successful
        is_terminal: whether the episode ended terminally (not truncation)
        subtask_frames: frame indices of operator sub-goal marks (may be empty)
    """
    outcome_frame = len(episode_data["rewards"]) - 1
    for frame in subtask_frames:
        if frame < 0 or frame > outcome_frame:
            raise ValueError(
                f"subtask mark frame {frame} is outside the episode's valid frames "
                f"[0, {outcome_frame}]"
            )
        if is_terminal and frame == outcome_frame:
            raise ValueError(
                f"subtask mark frame {frame} coincides with the outcome frame of a "
                "terminal episode; marks must precede the outcome frame"
            )

    # Set reward/done on the last VALID frame (matches sim convention)
    episode_data["rewards"][-1] = 1.0 if is_success else 0.0
    episode_data["dones"][-1] = 1 if is_terminal else 0

    # Pad: repeat last action + action_info, copy last reward/done
    episode_data["actions"].append(episode_data["actions"][-1].copy())
    pad_supplementary_terminal(episode_data)
    append_joint_velocities(episode_data, obs)
    append_franka_telemetry(episode_data, obs)
    episode_data["cartesian_velocities"].append(episode_data["cartesian_velocities"][-1].copy())
    episode_data["rewards"].append(episode_data["rewards"][-1])
    episode_data["dones"].append(episode_data["dones"][-1])

    for frame in subtask_frames:
        episode_data["rewards"][frame] = 1.0

    # Compute steps-to-go
    episode_length = len(episode_data["actions"])
    episode_data["steps_to_go"] = [episode_length - 1 - i for i in range(episode_length)]


def save_episode_to_dataset(
    dataset,
    episode_data: dict,
    episode_success: bool,
    camera_keys: list[str],
    task_name: str,
    saved_episode_count: int,
    *,
    sources: list[int] | None = None,
    default_source: int = DataSource.HUMAN,
    extra_frame_fields: dict | None = None,
    camera_name_fn: Callable[[str], str] | None = None,
    verbose: bool = True,
) -> int:
    """Write all frames for one episode into the dataset and save.

    ``camera_name_fn`` MUST match the one passed to ``build_real_lerobot_features`` so the
    written frame keys equal the declared feature keys (LeRobot rejects a mismatch).

    Args:
        dataset: LeRobotDataset instance
        episode_data: dict of lists from the episode recording loop
        episode_success: whether the episode was a success
        camera_keys: list of camera data keys (e.g. ["image_serial_left"])
        task_name: task name string for the dataset
        saved_episode_count: current count (for logging)
        sources: Per-timestep source IDs for DAgger (policy vs human).
                 If provided, also computes intervention flags.
        default_source: Fixed source ID for all frames (used when sources=None).
        extra_frame_fields: Additional constant fields per frame (e.g.
                            policy_id, round_id).

    Returns:
        Updated saved_episode_count (incremented by 1).
    """
    episode_length = len(episode_data["actions"])
    _assert_episode_data_lengths(
        episode_data,
        episode_length=episode_length,
        camera_keys=camera_keys,
    )

    if sources is not None:
        if len(sources) != episode_length:
            raise ValueError(
                "sources length must match episode length when saving real episode "
                f"({len(sources)=}, {episode_length=})"
            )
        intervention_flags = compute_intervention_flags(sources)
    else:
        intervention_flags = [0] * episode_length

    for i in range(episode_length):
        is_last_frame = i == episode_length - 1

        frame = {
            "task": task_name,
            "observation.state": episode_data["observations"][i].astype(np.float32),
            "action": episode_data["actions"][i].astype(np.float32),
            "steps_to_go": np.array([episode_data["steps_to_go"][i]], dtype=np.int64),
            "source": np.array(
                [sources[i] if sources is not None else default_source],
                dtype=np.int64,
            ),
            "intervention": np.array([intervention_flags[i]], dtype=np.int64),
            "success": np.array(
                [EpisodeOutcome.SUCCESS if episode_success else EpisodeOutcome.FAILURE],
                dtype=np.int64,
            ),
            "is_valid": np.array([0 if is_last_frame else 1], dtype=np.int64),
            "reward": np.array([episode_data["rewards"][i]], dtype=np.float32),
            "done": np.array([episode_data["dones"][i]], dtype=np.int64),
        }

        if extra_frame_fields:
            frame.update(extra_frame_fields)

        frame.update(build_supplementary_frame_fields(episode_data, i))

        for cam_data_key in camera_keys:
            frame[_camera_feature_name(cam_data_key, camera_name_fn)] = episode_data[cam_data_key][
                i
            ]

        dataset.add_frame(frame)

    dataset.save_episode(parallel_encoding=False)
    _assert_latest_video_spans_match_episode_length(dataset, episode_length=episode_length)
    saved_episode_count += 1

    if verbose:
        status_str = "SUCCESS" if episode_success else "FAILURE"
        if sources is not None:
            policy_frames = sum(1 for s in sources if s == DataSource.AUTONOMOUS)
            human_frames = sum(1 for s in sources if s == DataSource.HUMAN)
            num_interventions = sum(intervention_flags)
            print(
                f"Episode saved as {status_str} "
                f"(policy={policy_frames}, human={human_frames}, "
                f"interventions={num_interventions}, total saved={saved_episode_count})"
            )
        else:
            print(f"Episode saved as {status_str} (Total saved: {saved_episode_count})")

    return saved_episode_count
