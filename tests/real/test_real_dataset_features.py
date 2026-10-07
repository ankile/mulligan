import numpy as np
import pytest
from lerobot.utils.feature_utils import dataset_to_policy_features

from mulligan.real.robot.cameras import serial_key_to_role
from mulligan.real.collect.dataset_features import (
    FRANKA_EE_WRENCH_FEATURE,
    FRANKA_EXTERNAL_TORQUE_EPISODE_KEY,
    FRANKA_EXTERNAL_TORQUE_FEATURE,
    FRANKA_TELEMETRY_SPECS,
    _camera_feature_name,
    append_franka_telemetry,
    build_real_lerobot_features,
    ensure_dataset_can_store_episode_telemetry,
    init_supplementary_lists,
    record_or_verify_camera_role_serials,
    save_episode_to_dataset,
)

# All telemetry feature names, in registry order, for tests that assert the full union.
_ALL_TELEMETRY_FEATURES = [spec.feature_name for spec in FRANKA_TELEMETRY_SPECS]
_MEASURED_TORQUE_FEATURE = "telemetry.franka.motor_torques_measured"
_COMPUTED_TORQUE_FEATURE = "telemetry.franka.joint_torques_computed"


def test_camera_name_fn_renames_stored_features_to_roles() -> None:
    # serial_key_to_role renames the live serial key to its station role. The schema build
    # AND the per-frame writer both go through _camera_feature_name, so they agree by
    # construction (the save-side consistency invariant).
    assert (
        _camera_feature_name("image_10000001_left", serial_key_to_role)
        == "observation.images.wrist_left"
    )
    episode_data = {
        "observations": [np.zeros(7, dtype=np.float32)],
        "actions": [np.zeros(7, dtype=np.float32)],
        "image_20000002_left": [np.zeros((4, 4, 3), dtype=np.uint8)],
    }
    feats = build_real_lerobot_features(
        episode_data, ["image_20000002_left"], camera_name_fn=serial_key_to_role
    )
    assert "observation.images.side_1" in feats
    assert "observation.images.20000002_left" not in feats


def test_record_or_verify_camera_role_serials(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from mulligan.real.collect.dataset_features import CAMERA_ROLE_SERIALS_SIDECAR

    # image_<serial> live keys for side_1 + wrist_left.
    cam_keys = ["image_20000002_left", "image_10000001_left"]
    expected_map = {"side_1": "20000002_left", "wrist_left": "10000001_left"}
    swapped_map = {"side_1": "10000001_left", "wrist_left": "20000002_left"}
    role_features = {"observation.images.side_1": {}, "observation.images.wrist_left": {}}

    def make_ds(info, root):
        (root / "meta").mkdir(parents=True, exist_ok=True)
        # Production ``dataset.meta.info`` is a typed DatasetInfo (attribute access,
        # e.g. ``.features``), not a dict — mirror that so
        # ``record_or_verify_camera_role_serials``'s ``meta.info.features`` resolves.
        return SimpleNamespace(meta=SimpleNamespace(info=SimpleNamespace(**info)), root=root)

    def sidecar(root):
        return root / "meta" / CAMERA_ROLE_SERIALS_SIDECAR

    # Fresh dataset: writes the provenance map to the SIDECAR (not info.json,
    # which lerobot 0.5.2 would clobber each save_episode).
    root = tmp_path / "fresh"
    ds = make_ds({"features": dict(role_features)}, root)
    record_or_verify_camera_role_serials(ds, cam_keys)
    assert json.loads(sidecar(root).read_text()) == expected_map
    assert not (root / "meta" / "info.json").exists()  # info.json is left untouched
    # Resume with the SAME cabling is idempotent (reads the sidecar back, no raise).
    record_or_verify_camera_role_serials(ds, cam_keys)

    # Resume after the cameras were re-cabled (same roles, swapped serials) -> fail loud.
    root = tmp_path / "recabled"
    ds = make_ds({"features": dict(role_features)}, root)
    sidecar(root).write_text(json.dumps(swapped_map))
    with pytest.raises(RuntimeError, match="re-cabled"):
        record_or_verify_camera_role_serials(ds, cam_keys)

    # A map stored in info.json (no sidecar) is read from there AND copied to the durable
    # sidecar on first touch.
    root = tmp_path / "info_json_map"
    ds = make_ds({"features": dict(role_features)}, root)
    (root / "meta" / "info.json").write_text(json.dumps({"camera_role_serials": expected_map}))
    record_or_verify_camera_role_serials(ds, cam_keys)
    assert json.loads(sidecar(root).read_text()) == expected_map  # copied to the sidecar

    # An info.json map that disagrees with live cabling still fails loud.
    root = tmp_path / "info_json_recabled"
    ds = make_ds({"features": dict(role_features)}, root)
    (root / "meta" / "info.json").write_text(json.dumps({"camera_role_serials": swapped_map}))
    with pytest.raises(RuntimeError, match="re-cabled"):
        record_or_verify_camera_role_serials(ds, cam_keys)

    # A role the live cameras map to is missing from the stored features -> fail loud.
    root = tmp_path / "missing_feature"
    ds = make_ds({"features": {"observation.images.side_1": {}}}, root)  # wrist_left absent
    with pytest.raises(RuntimeError, match="not in existing dataset features"):
        record_or_verify_camera_role_serials(ds, cam_keys)


def _two_frame_episode_data(*, include_franka_telemetry: bool = True) -> dict:
    """Minimal valid episode_data (2 frames) for the real-robot save path.

    Mirrors the same camera data keys used elsewhere in this file:
    image_20000002_left -> side_1, image_10000001_left -> wrist_left.
    Each per-frame list carries one entry per frame; supplementary fields
    (action_info_*, joint_positions/velocities, cartesian_velocities) are
    required by build_supplementary_frame_fields.

    By default ALL Franka telemetry columns are populated (matching the
    now-unconditional schema, where every real dataset carries the full sensor
    union). Pass ``include_franka_telemetry=False`` to build a legacy-shaped
    episode with NO telemetry columns (used to construct pre-upgrade datasets for
    the backfill tests).
    """
    n = 2
    cam_keys = ["image_20000002_left", "image_10000001_left"]
    episode_data = {
        "observations": [np.arange(7, dtype=np.float32) + j for j in range(n)],
        "actions": [np.zeros(7, dtype=np.float32) for _ in range(n)],
        "steps_to_go": [n - 1 - j for j in range(n)],
        "rewards": [0.0 for _ in range(n)],
        "dones": [0 for _ in range(n)],
        "joint_positions": [np.zeros(7, dtype=np.float32) for _ in range(n)],
        "joint_velocities": [np.zeros(7, dtype=np.float32) for _ in range(n)],
        "cartesian_velocities": [np.zeros(6, dtype=np.float32) for _ in range(n)],
        "action_info_cartesian_velocity": [np.zeros(6, dtype=np.float32) for _ in range(n)],
        "action_info_cartesian_position": [np.zeros(6, dtype=np.float32) for _ in range(n)],
        "action_info_joint_velocity": [np.zeros(7, dtype=np.float32) for _ in range(n)],
        "action_info_joint_position": [np.zeros(7, dtype=np.float32) for _ in range(n)],
        "action_info_gripper_position": [np.zeros(1, dtype=np.float32) for _ in range(n)],
        "action_info_gripper_velocity": [np.zeros(1, dtype=np.float32) for _ in range(n)],
    }
    for cam_key in cam_keys:
        episode_data[cam_key] = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]
    if include_franka_telemetry:
        # Distinct per-signal base values so save-write assertions can tell columns apart.
        for base, spec in enumerate(FRANKA_TELEMETRY_SPECS):
            episode_data[spec.episode_key] = [
                np.full(spec.shape, 1.0 + base * 100 + j, dtype=np.float32) for j in range(n)
            ]
    return episode_data


class _CapturingDataset:
    """Fake LeRobotDataset that records every add_frame() payload.

    save_episode_to_dataset only ever touches ``add_frame(frame)`` (task is a
    key inside the frame, NOT an add_frame kwarg) and ``save_episode(
    parallel_encoding=False)`` on its dataset argument — so those are the only
    two methods stubbed here. Stubbing exactly these keeps the fake a faithful
    mirror of the real save path.
    """

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.saved_episodes = 0

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self, parallel_encoding: bool = True) -> None:
        self.saved_episodes += 1


class _BadVideoSpanDataset(_CapturingDataset):
    def __init__(self) -> None:
        from types import SimpleNamespace

        super().__init__()
        self.meta = SimpleNamespace(
            fps=15,
            video_keys=["observation.images.side_1"],
            latest_episode=None,
        )

    def save_episode(self, parallel_encoding: bool = True) -> None:
        super().save_episode(parallel_encoding=parallel_encoding)
        self.meta.latest_episode = {
            "episode_index": [0],
            "videos/observation.images.side_1/from_timestamp": [0.0],
            "videos/observation.images.side_1/to_timestamp": [1.0 / 15.0],
        }


def test_save_frame_camera_keys_match_schema_camera_keys() -> None:
    """End-to-end: the camera feature keys WRITTEN per frame by
    save_episode_to_dataset must exactly equal the camera feature keys in the
    schema from build_real_lerobot_features, for the SAME camera_name_fn.

    Both sites route camera names through _camera_feature_name, so this holds by
    construction, but it is never asserted at runtime. If the ``image_``-prefix strip /
    role rename were inlined in only one of the two sites, the schema and the written
    frames would silently diverge and only a real LeRobotDataset save would reject the
    frame. This pins the invariant.
    """
    camera_name_fn = serial_key_to_role
    expected_camera_features = {"observation.images.side_1", "observation.images.wrist_left"}
    episode_data = _two_frame_episode_data()
    camera_keys = ["image_20000002_left", "image_10000001_left"]

    schema = build_real_lerobot_features(episode_data, camera_keys, camera_name_fn=camera_name_fn)

    dataset = _CapturingDataset()
    save_episode_to_dataset(
        dataset,
        episode_data,
        episode_success=True,
        camera_keys=camera_keys,
        task_name="dummy_task",
        saved_episode_count=0,
        camera_name_fn=camera_name_fn,
        verbose=False,
    )

    assert dataset.saved_episodes == 1
    assert len(dataset.frames) == len(episode_data["actions"])

    schema_camera_keys = {k for k in schema if k.startswith("observation.images.")}
    written_camera_keys = {
        k for frame in dataset.frames for k in frame if k.startswith("observation.images.")
    }

    # Sanity: the camera_name_fn actually produced the keys we expect.
    assert schema_camera_keys == expected_camera_features
    # The load-bearing invariant: written frame camera keys == schema camera keys.
    assert written_camera_keys == schema_camera_keys


def test_save_episode_rejects_misaligned_real_episode_lists() -> None:
    episode_data = _two_frame_episode_data()
    episode_data["image_20000002_left"].pop()

    with pytest.raises(RuntimeError, match="image_20000002_left"):
        save_episode_to_dataset(
            _CapturingDataset(),
            episode_data,
            episode_success=True,
            camera_keys=["image_20000002_left", "image_10000001_left"],
            task_name="dummy_task",
            saved_episode_count=0,
            verbose=False,
        )


def test_save_episode_rejects_bad_lerobot_video_span_metadata() -> None:
    episode_data = _two_frame_episode_data()

    with pytest.raises(RuntimeError, match="video metadata mismatch"):
        save_episode_to_dataset(
            _BadVideoSpanDataset(),
            episode_data,
            episode_success=True,
            camera_keys=["image_20000002_left", "image_10000001_left"],
            task_name="dummy_task",
            saved_episode_count=0,
            camera_name_fn=serial_key_to_role,
            verbose=False,
        )


def test_all_franka_telemetry_columns_always_declared_and_never_policy_input() -> None:
    # Every Franka telemetry signal is ALWAYS a column (so all real datasets share
    # one schema), and none is ever a policy observation input.
    episode_data = {
        "observations": [np.zeros(7, dtype=np.float32)],
        "actions": [np.zeros(7, dtype=np.float32)],
    }
    init_supplementary_lists(episode_data)
    # init creates one buffer list per telemetry signal.
    for spec in FRANKA_TELEMETRY_SPECS:
        assert episode_data[spec.episode_key] == []

    features = build_real_lerobot_features(episode_data, cam_data_keys=[])
    policy_features = dataset_to_policy_features(features)
    for spec in FRANKA_TELEMETRY_SPECS:
        assert features[spec.feature_name]["shape"] == spec.shape
        assert spec.feature_name not in policy_features
    # The two shapes/names the downstream code hard-codes are pinned here.
    assert features[FRANKA_EXTERNAL_TORQUE_FEATURE]["shape"] == (7,)
    assert features[FRANKA_EE_WRENCH_FEATURE]["shape"] == (6,)
    assert features[FRANKA_EE_WRENCH_FEATURE]["names"] == ["fx", "fy", "fz", "mx", "my", "mz"]
    # The two newly-added signals are present.
    assert features[_MEASURED_TORQUE_FEATURE]["shape"] == (7,)
    assert features[_COMPUTED_TORQUE_FEATURE]["shape"] == (7,)


def test_append_franka_telemetry_records_present_and_nan_fills_absent() -> None:
    # A server that exposes only some signals: present ones are recorded verbatim,
    # absent ones are NaN-filled (never dropped, never zero-filled).
    episode_data = {
        "observations": [np.zeros(7, dtype=np.float32)],
        "actions": [np.zeros(7, dtype=np.float32)],
    }
    init_supplementary_lists(episode_data)

    obs = {
        "robot_state": {
            "motor_torques_external": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
            "ee_wrench": [1.0, -2.0, 3.0, -0.4, 0.5, -0.6],
            # motor_torques_measured / joint_torques_computed intentionally absent.
        },
    }
    append_franka_telemetry(episode_data, obs)

    ext = episode_data["franka_motor_torques_external"][0]
    wrench = episode_data["franka_ee_wrench"][0]
    assert ext.dtype == np.float32
    assert ext.tolist() == pytest.approx([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    assert wrench.tolist() == pytest.approx([1.0, -2.0, 3.0, -0.4, 0.5, -0.6])
    # Absent signals -> all-NaN vector of the right shape.
    measured = episode_data["franka_motor_torques_measured"][0]
    computed = episode_data["franka_joint_torques_computed"][0]
    assert measured.shape == (7,) and np.isnan(measured).all()
    assert computed.shape == (7,) and np.isnan(computed).all()


def test_append_franka_telemetry_nan_fills_everything_when_server_bare() -> None:
    episode_data = {
        "observations": [np.zeros(7, dtype=np.float32)],
        "actions": [np.zeros(7, dtype=np.float32)],
    }
    init_supplementary_lists(episode_data)
    append_franka_telemetry(episode_data, {"robot_state": {}})
    for spec in FRANKA_TELEMETRY_SPECS:
        vec = episode_data[spec.episode_key][0]
        assert vec.shape == spec.shape
        assert np.isnan(vec).all()


def test_append_franka_telemetry_shape_validation() -> None:
    episode_data = {
        "observations": [np.zeros(7, dtype=np.float32)],
        "actions": [np.zeros(7, dtype=np.float32)],
    }
    init_supplementary_lists(episode_data)
    with pytest.raises(ValueError, match="Expected .*7.* robot_state.motor_torques_external"):
        append_franka_telemetry(
            episode_data, {"robot_state": {"motor_torques_external": [0.0] * 6}}
        )
    with pytest.raises(ValueError, match="Expected .*6.* robot_state.ee_wrench"):
        append_franka_telemetry(episode_data, {"robot_state": {"ee_wrench": [0.0] * 7}})


def test_all_franka_telemetry_columns_written_on_every_frame() -> None:
    episode_data = _two_frame_episode_data()
    features = build_real_lerobot_features(episode_data, cam_data_keys=[])
    for spec in FRANKA_TELEMETRY_SPECS:
        assert spec.feature_name in features

    dataset = _CapturingDataset()
    save_episode_to_dataset(
        dataset,
        episode_data,
        episode_success=True,
        camera_keys=[],
        task_name="dummy_task",
        saved_episode_count=0,
        verbose=False,
    )
    # Every telemetry column is written on every frame with its declared shape.
    for frame in dataset.frames:
        for spec in FRANKA_TELEMETRY_SPECS:
            assert frame[spec.feature_name].shape == spec.shape


def _pop_telemetry_features(features: dict, feature_names) -> dict:
    """Return a copy of ``features`` with the given telemetry columns removed.

    Simulates a LEGACY dataset created before a telemetry column existed. Since the
    schema is now unconditional, this is the only way to produce a dataset genuinely
    missing a column for the backfill path to upgrade.
    """
    return {k: v for k, v in features.items() if k not in set(feature_names)}


def test_existing_dataset_backfills_all_missing_telemetry_before_append(tmp_path) -> None:
    # A legacy dataset created with NO telemetry columns; appending a modern episode
    # (full sensor union) must add every telemetry column and NaN-backfill old frames.
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "mulligan/test-franka-telemetry-upgrade"
    root = tmp_path / "dataset"

    old_episode = _two_frame_episode_data(include_franka_telemetry=False)
    legacy_features = _pop_telemetry_features(
        build_real_lerobot_features(old_episode, cam_data_keys=[]),
        _ALL_TELEMETRY_FEATURES,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        root=str(root),
        robot_type="franka",
        features=legacy_features,
    )
    save_episode_to_dataset(
        dataset,
        old_episode,
        episode_success=True,
        camera_keys=[],
        task_name="dummy_task",
        saved_episode_count=0,
        verbose=False,
    )
    dataset.finalize()

    dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
    for feat in _ALL_TELEMETRY_FEATURES:
        assert feat not in dataset.features

    new_episode = _two_frame_episode_data()  # full telemetry union
    dataset = ensure_dataset_can_store_episode_telemetry(
        dataset,
        new_episode,
        dataset_path=root,
        dataset_name=repo_id,
    )
    for feat in _ALL_TELEMETRY_FEATURES:
        assert feat in dataset.features

    save_episode_to_dataset(
        dataset,
        new_episode,
        episode_success=True,
        camera_keys=[],
        task_name="dummy_task",
        saved_episode_count=1,
        verbose=False,
    )
    dataset.finalize()

    data = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "data").glob("*/*.parquet"))],
        ignore_index=True,
    ).sort_values("index")

    for feat in _ALL_TELEMETRY_FEATURES:
        values = [np.asarray(v, dtype=np.float32) for v in data[feat]]
        assert len(values) == 4
        # Old frames NaN-backfilled; new frames carry real (non-NaN) measurements.
        assert np.isnan(values[0]).all()
        assert np.isnan(values[1]).all()
        assert not np.isnan(values[2]).any()
        assert not np.isnan(values[3]).any()


def test_existing_partial_telemetry_dataset_backfills_only_missing_columns(tmp_path) -> None:
    # Mixed-state backfill: the dataset already has the 7D external-torque column but
    # not the other three signals; a modern episode adds all four. Only the three
    # missing columns must be added + NaN-backfilled; the pre-existing torque column
    # is preserved (asserted-as-matching, not re-NaN'd).
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "mulligan/test-franka-partial-telemetry-upgrade"
    root = tmp_path / "dataset"

    # Old episode carries ONLY the external-torque signal.
    old_episode = _two_frame_episode_data(include_franka_telemetry=False)
    old_episode[FRANKA_EXTERNAL_TORQUE_EPISODE_KEY] = [
        np.full(7, 1.0 + j, dtype=np.float32) for j in range(2)
    ]
    partial_features = _pop_telemetry_features(
        build_real_lerobot_features(old_episode, cam_data_keys=[]),
        [f for f in _ALL_TELEMETRY_FEATURES if f != FRANKA_EXTERNAL_TORQUE_FEATURE],
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        root=str(root),
        robot_type="franka",
        features=partial_features,
    )
    save_episode_to_dataset(
        dataset,
        old_episode,
        episode_success=True,
        camera_keys=[],
        task_name="dummy_task",
        saved_episode_count=0,
        verbose=False,
    )
    dataset.finalize()

    dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
    assert FRANKA_EXTERNAL_TORQUE_FEATURE in dataset.features
    assert FRANKA_EE_WRENCH_FEATURE not in dataset.features

    new_episode = _two_frame_episode_data()  # full telemetry union
    dataset = ensure_dataset_can_store_episode_telemetry(
        dataset,
        new_episode,
        dataset_path=root,
        dataset_name=repo_id,
    )
    for feat in _ALL_TELEMETRY_FEATURES:
        assert feat in dataset.features

    save_episode_to_dataset(
        dataset,
        new_episode,
        episode_success=True,
        camera_keys=[],
        task_name="dummy_task",
        saved_episode_count=1,
        verbose=False,
    )
    dataset.finalize()

    data = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "data").glob("*/*.parquet"))],
        ignore_index=True,
    ).sort_values("index")

    torque = [np.asarray(v, dtype=np.float32) for v in data[FRANKA_EXTERNAL_TORQUE_FEATURE]]
    # Pre-existing torque column preserved across the upgrade (never NaN, no re-backfill).
    np.testing.assert_allclose(torque[0], np.ones(7, dtype=np.float32))
    np.testing.assert_allclose(torque[1], np.full(7, 2.0, dtype=np.float32))
    assert not np.isnan(torque[2]).any()
    assert not np.isnan(torque[3]).any()
    # The three newly-added columns: NaN on old frames, real on new frames.
    for feat in _ALL_TELEMETRY_FEATURES:
        if feat == FRANKA_EXTERNAL_TORQUE_FEATURE:
            continue
        values = [np.asarray(v, dtype=np.float32) for v in data[feat]]
        assert np.isnan(values[0]).all()
        assert np.isnan(values[1]).all()
        assert not np.isnan(values[2]).any()
        assert not np.isnan(values[3]).any()


def test_existing_dataset_without_telemetry_upgrade_reopens_appendable(tmp_path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "mulligan/test-existing-dataset-appendable"
    root = tmp_path / "dataset"
    episode = _two_frame_episode_data()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        root=str(root),
        robot_type="franka",
        features=build_real_lerobot_features(episode, cam_data_keys=[]),
    )
    save_episode_to_dataset(
        dataset,
        episode,
        episode_success=True,
        camera_keys=[],
        task_name="dummy_task",
        saved_episode_count=0,
        verbose=False,
    )
    dataset.finalize()

    read_only = LeRobotDataset(repo_id=repo_id, root=str(root))
    assert getattr(read_only, "writer", None) is None

    appendable = ensure_dataset_can_store_episode_telemetry(
        read_only,
        episode,
        dataset_path=root,
        dataset_name=repo_id,
    )
    assert getattr(appendable, "writer", None) is not None
    appendable.start_image_writer(num_processes=0, num_threads=1)
    save_episode_to_dataset(
        appendable,
        episode,
        episode_success=True,
        camera_keys=[],
        task_name="dummy_task",
        saved_episode_count=1,
        verbose=False,
    )
    appendable.finalize()

    reloaded = LeRobotDataset(repo_id=repo_id, root=str(root))
    assert reloaded.num_episodes == 2
