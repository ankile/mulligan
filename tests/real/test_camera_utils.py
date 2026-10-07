import json
from pathlib import Path

import pytest

from mulligan.real.robot.cameras import (
    DEFAULT_CAMERA_KEYS,
    DEFAULT_EXCLUDED_CAMERA_KEYS,
    STATION_CAMERA_DEFAULT_CROPS,
    STATION_CAMERA_KEYS_BY_ROLE,
    camera_role_serials,
    crop_frame_to_role_view,
    policy_live_camera_keys,
    remove_excluded_camera_features_from_lerobot_dataset,
    require_station_role_image_features,
    role_to_serial_key,
    select_camera_feature_keys,
    select_image_camera_keys,
    serial_key_to_role,
)


class _FakePolicy:
    """Minimal policy stand-in with a config exposing image_features / dual crops."""

    def __init__(self, image_features, dual_side_crop_boxes=None):
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            image_features=image_features,
            dual_side_crop_boxes=dual_side_crop_boxes or {},
        )


# The full station record set (all 4 streams), as raw serial-eye keys.
_RECORD_SET = ["10000001_left", "10000001_right", "20000002_left", "30000003_left"]


def test_policy_live_camera_keys_narrows_role_named_policy_to_its_subset() -> None:
    # A role-named 2-camera policy (side_1 + wrist_left) is fed only those serials,
    # even though the recorded dataset carries all 4 streams.
    policy = _FakePolicy({"observation.images.side_1": {}, "observation.images.wrist_left": {}})
    assert policy_live_camera_keys(policy, _RECORD_SET) == ["20000002_left", "10000001_left"]


def test_policy_live_camera_keys_refuses_non_role_image_features() -> None:
    policy = _FakePolicy({"observation.images.20000002_left": {}})
    with pytest.raises(ValueError, match="not station camera roles"):
        policy_live_camera_keys(policy, _RECORD_SET)


def test_require_station_role_image_features_returns_roles_in_order() -> None:
    features = ["observation.images.wrist_left", "observation.images.side_1"]
    assert require_station_role_image_features(features, context="test") == [
        "wrist_left",
        "side_1",
    ]
    assert require_station_role_image_features({}, context="test") == []
    with pytest.raises(ValueError, match=r"test: image feature\(s\) \['cam_a'\]"):
        require_station_role_image_features(["observation.images.cam_a"], context="test")


def test_policy_live_camera_keys_falls_back_to_full_set_without_image_features() -> None:
    assert policy_live_camera_keys(object(), _RECORD_SET) == _RECORD_SET
    assert policy_live_camera_keys(_FakePolicy({}), _RECORD_SET) == _RECORD_SET


class _ExplicitLiveKeysPolicy:
    """Stand-in for a wrapper that declares its live serials directly (no LeRobot config)."""

    def __init__(self, live_camera_keys):
        self.live_camera_keys = live_camera_keys


def test_policy_live_camera_keys_honors_explicit_live_camera_keys() -> None:
    # A wrapper with no LeRobot config exposes live_camera_keys directly. 2-cam and
    # 3-cam sets are both honored verbatim (order preserved).
    two_cam = _ExplicitLiveKeysPolicy(["20000002_left", "10000001_left"])
    assert policy_live_camera_keys(two_cam, _RECORD_SET) == ["20000002_left", "10000001_left"]
    three_cam = _ExplicitLiveKeysPolicy(["20000002_left", "10000001_left", "30000003_left"])
    assert policy_live_camera_keys(three_cam, _RECORD_SET) == [
        "20000002_left",
        "10000001_left",
        "30000003_left",
    ]


def test_policy_live_camera_keys_explicit_missing_camera_is_loud() -> None:
    # side_2 declared but absent from the recording set -> fail loud (mis-scoped
    # --camera-keys can't silently starve inference).
    policy = _ExplicitLiveKeysPolicy(["20000002_left", "30000003_left"])
    with pytest.raises(RuntimeError, match="Policy requires live camera"):
        policy_live_camera_keys(policy, ["20000002_left", "10000001_left"])


def test_policy_live_camera_keys_raises_when_camera_not_recorded() -> None:
    # side_2 is required by the policy but absent from the record set -> fail loud.
    policy = _FakePolicy({"observation.images.side_2": {}})
    with pytest.raises(RuntimeError, match="Policy requires live camera"):
        policy_live_camera_keys(policy, ["10000001_left", "20000002_left"])


def test_multi_arm_session_each_policy_gets_its_own_subset() -> None:
    """Session contract: the recorded set is shared and stable across arms, but each
    policy selects its OWN input subset from it — an arm never inherits another arm's
    subset. This is the invariant rollout_episode must uphold when it recomputes
    policy_live_camera_keys() per rollout instead of caching one arm's subset.
    """
    record_set = _RECORD_SET  # station-defined full set, discovered once per session
    arm_a = _FakePolicy({"observation.images.side_1": {}})  # side_1 only
    arm_b = _FakePolicy(
        {"observation.images.side_1": {}, "observation.images.wrist_left": {}}
    )  # side_1 + wrist_left
    subset_a = policy_live_camera_keys(arm_a, record_set)
    subset_b = policy_live_camera_keys(arm_b, record_set)
    assert subset_a == ["20000002_left"]
    assert subset_b == ["20000002_left", "10000001_left"]
    # The narrower arm must NOT leak into the wider arm (the multi-arm caching bug).
    assert subset_a != subset_b
    # Recomputing arm A again yields its own subset regardless of evaluation order.
    assert policy_live_camera_keys(arm_a, record_set) == subset_a


def test_rollout_episode_has_no_camera_keys_cache_param() -> None:
    """Regression guard: rollout_episode must NOT take a session-lifetime camera_keys
    cache (the conflation that let a multi-arm session feed arm A's camera subset to
    arm B). Camera discovery caches only the RECORD set via all_camera_keys; the
    per-policy subset is recomputed each call.
    """
    import inspect

    from mulligan.real.collect.rollout import rollout_episode

    params = inspect.signature(rollout_episode).parameters
    assert "camera_keys" not in params, (
        "rollout_episode regained a camera_keys cache param; the per-policy subset "
        "must be recomputed each rollout, not cached across arms."
    )
    assert "all_camera_keys" in params  # the record-set cache stays


# A 4-camera marker_d2-style stored dataset (side_1, side_2, both wrist eyes).
_ALL_CAMS = [
    "observation.images.20000002_left",  # side_1
    "observation.images.30000003_left",  # side_2
    "observation.images.10000001_left",  # wrist_left
    "observation.images.10000001_right",  # wrist_right
]


def test_select_camera_feature_keys_explicit_subset_in_order():
    # DP consumes side_1 + wrist_left out of the 4 stored cameras, in the given order.
    sel = select_camera_feature_keys(_ALL_CAMS, camera_keys="20000002_left,10000001_left")
    assert sel == ["observation.images.20000002_left", "observation.images.10000001_left"]


def test_select_camera_feature_keys_accepts_full_keys():
    sel = select_camera_feature_keys(_ALL_CAMS, camera_keys="observation.images.10000001_right")
    assert sel == ["observation.images.10000001_right"]


def test_select_camera_feature_keys_filter_picks_all_matching_suffix():
    # Pure suffix-filter behavior, using NON-station cameras so the dropped-station guard
    # (which fires on station serials/roles present as features) is not involved.
    non_station = [
        "observation.images.aux_a_left",
        "observation.images.aux_a_right",
        "observation.images.aux_b_left",
    ]
    sel = select_camera_feature_keys(non_station, camera_filter="_left")
    assert sel == [c for c in non_station if c.endswith("_left")]


def test_select_camera_feature_keys_missing_camera_raises():
    with pytest.raises(ValueError, match="not in dataset cameras"):
        select_camera_feature_keys(_ALL_CAMS, camera_keys="99999999_left")


def test_select_camera_feature_keys_no_match_raises():
    # Use NON-station cameras: with station cameras present the dropped-station guard
    # would fire first; this pins the "nothing matched the filter" path in isolation.
    non_station = ["observation.images.aux_a_left", "observation.images.aux_b_left"]
    with pytest.raises(ValueError, match="No cameras selected"):
        select_camera_feature_keys(non_station, camera_filter="_nope")


# A role-named marker_d2 stored dataset: cameras stored under ROLE names.
_ROLE_CAMS = [
    "observation.images.side_1",
    "observation.images.side_2",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
]


def test_select_camera_feature_keys_left_filter_drops_role_named_station_cams_raises():
    # The footgun the rework guards: for role-named marker_d2 features the default '_left'
    # filter selects only wrist_left and silently drops side_1/side_2 (station roles present
    # as features). The fallback path must FAIL LOUD and tell the caller to pass explicit
    # --camera-keys, mirroring select_image_camera_keys' guard.
    with pytest.raises(ValueError, match="would drop wanted station camera"):
        select_camera_feature_keys(_ROLE_CAMS, camera_filter="_left")


def test_select_camera_feature_keys_explicit_role_keys_unaffected_by_guard():
    # Explicit --camera-keys is the documented escape hatch; the dropped-station guard
    # only fires on the suffix-filter fallback, never the explicit path.
    sel = select_camera_feature_keys(_ROLE_CAMS, camera_keys="side_1,wrist_left")
    assert sel == ["observation.images.side_1", "observation.images.wrist_left"]


def test_select_camera_feature_keys_serial_right_eye_dropped_raises():
    # The original stereo-eye footgun still fires: both wrist eyes present, '_left' drops
    # wrist_right (a station serial), so the suffix filter must raise.
    with pytest.raises(ValueError, match="would drop wanted station camera"):
        select_camera_feature_keys(_ALL_CAMS, camera_filter="_left")


def test_select_camera_feature_keys_rejects_duplicate_camera_keys():
    # A camera repeated in the requested list must raise (mirrors parse_side_crop's
    # "specified twice" guard) rather than silently feed the same camera twice.
    with pytest.raises(ValueError, match="specified twice"):
        select_camera_feature_keys(_ROLE_CAMS, camera_keys="side_1,side_1")


def test_serial_role_mapping_roundtrips():
    # The one mapping used at both the collection-save and eval-load boundaries.
    for role, serial in STATION_CAMERA_KEYS_BY_ROLE.items():
        assert serial_key_to_role(serial) == role
        assert role_to_serial_key(role) == serial


def test_serial_key_to_role_raises_on_unknown_serial():
    with pytest.raises(KeyError, match="not in STATION_CAMERA_KEYS_BY_ROLE"):
        serial_key_to_role("99999999_left")


def test_role_to_serial_key_raises_on_unknown_role():
    with pytest.raises(KeyError, match="not in STATION"):
        role_to_serial_key("definitely_not_a_role")


def test_camera_role_serials_full_and_subset():
    assert camera_role_serials() == STATION_CAMERA_KEYS_BY_ROLE
    # The provenance map for a dataset that stored only side_1 + wrist_left.
    assert camera_role_serials(["20000002_left", "10000001_left"]) == {
        "side_1": "20000002_left",
        "wrist_left": "10000001_left",
    }


def test_station_default_crops_are_valid_role_keyed_boxes():
    # Every default crop is keyed by a real station role and is a non-degenerate
    # (x0<x1, y0<y1) box that fits inside the 640x480 STORED frame (the ROIs were scaled
    # from the 1280x720 native feed). An out-of-bounds box would fail loud at train.
    for role, box in STATION_CAMERA_DEFAULT_CROPS.items():
        assert role in STATION_CAMERA_KEYS_BY_ROLE, role
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= 640, (role, box)
        assert 0 <= y0 < y1 <= 480, (role, box)


def test_crop_frame_to_role_view_resizes_to_stored_then_crops() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    # A native 1280x720 ZED frame: the helper must downscale to 640x480 BEFORE cropping,
    # so the returned crop has exactly the STORED-space box dimensions (this is the eval/
    # train/preview contract -- the box lives in 640x480 space, not native 1280x720).
    native = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = crop_frame_to_role_view(native, "20000002_left")  # side_1
    assert result is not None
    role, cropped = result
    assert role == "side_1"
    x0, y0, x1, y1 = STATION_CAMERA_DEFAULT_CROPS["side_1"]
    assert cropped.shape[:2] == (y1 - y0, x1 - x0)
    # A non-station camera (or one without a default crop) yields None, not a crash.
    assert crop_frame_to_role_view(native, "99999999_left") is None


def test_crop_frame_to_role_view_policy_crop_boxes_override_default() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    # crop_boxes (role-keyed, stored-space -- a policy's trained camera_crop_boxes, e.g.
    # routing_d2's task-specific side ROIs) must override the station default so operator
    # monitors show the policy's ACTUAL view; roles absent from the map keep the default.
    native = np.zeros((720, 1280, 3), dtype=np.uint8)
    override = {"side_1": (140, 120, 560, 470)}
    assert override["side_1"] != STATION_CAMERA_DEFAULT_CROPS["side_1"]
    role, cropped = crop_frame_to_role_view(
        native, role_to_serial_key("side_1"), crop_boxes=override
    )
    assert role == "side_1"
    assert cropped.shape[:2] == (470 - 120, 560 - 140)
    # wrist_left is NOT in the override map -> falls back to the station default box.
    role, cropped = crop_frame_to_role_view(
        native, role_to_serial_key("wrist_left"), crop_boxes=override
    )
    x0, y0, x1, y1 = STATION_CAMERA_DEFAULT_CROPS["wrist_left"]
    assert role == "wrist_left"
    assert cropped.shape[:2] == (y1 - y0, x1 - x0)


def test_select_image_camera_keys_excludes_explicit_set() -> None:
    # Test the exclusion MECHANISM with an explicit set, so the test is robust to
    # changes in the station default (DEFAULT_EXCLUDED_CAMERA_KEYS).
    obs = {
        "image": {
            "10000001_left": object(),
            "20000002_left": object(),
            "aux_cam_left": object(),
        }
    }

    selected, all_cams = select_image_camera_keys(
        obs, "_left", excluded_camera_keys={"aux_cam_left"}
    )

    assert selected == ["10000001_left", "20000002_left"]
    assert "aux_cam_left" in all_cams


def test_left_filter_fails_loud_when_it_would_drop_a_wanted_station_camera() -> None:
    # The footgun the rework guards: at the station both wrist eyes are physically
    # present, but the '_left' suffix filter cannot select wrist_right -> it would
    # silently record a subset. select_image_camera_keys must FAIL LOUD instead, so the
    # station is forced to pass explicit --camera-keys.
    obs = {"image": {key: object() for key in STATION_CAMERA_KEYS_BY_ROLE.values()}}
    obs["image"]["20000002_right"] = object()  # an excluded side right eye, also present
    with pytest.raises(RuntimeError, match="would drop wanted station camera"):
        select_image_camera_keys(obs, "_left")  # DEFAULT excluded set


def test_remove_excluded_camera_features_from_lerobot_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    meta_path = dataset_path / "meta"
    meta_path.mkdir(parents=True)
    (dataset_path / "videos" / "observation.images.aux_cam_left").mkdir(parents=True)
    (dataset_path / "images" / "observation.images.aux_cam_left").mkdir(parents=True)
    (meta_path / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.10000001_left": {"dtype": "video"},
                    "observation.images.aux_cam_left": {"dtype": "video"},
                }
            }
        )
    )
    (meta_path / "stats.json").write_text(
        json.dumps(
            {
                "observation.images.10000001_left": {},
                "observation.images.aux_cam_left": {},
            }
        )
    )

    removed = remove_excluded_camera_features_from_lerobot_dataset(
        dataset_path, excluded_camera_keys={"aux_cam_left"}
    )

    assert removed == ["observation.images.aux_cam_left"]
    info = json.loads((meta_path / "info.json").read_text())
    stats = json.loads((meta_path / "stats.json").read_text())
    assert "observation.images.aux_cam_left" not in info["features"]
    assert "observation.images.aux_cam_left" not in stats
    assert not (dataset_path / "videos" / "observation.images.aux_cam_left").exists()
    assert not (dataset_path / "images" / "observation.images.aux_cam_left").exists()


def test_remove_excluded_camera_features_survives_interruption(tmp_path: Path, monkeypatch) -> None:
    # Runs on the resume path: a kill mid-rewrite must leave readable files, and the next
    # run must finish the removal (info.json is the last file rewritten).
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    feature = "observation.images.aux_cam_left"
    dataset_path = tmp_path / "dataset"
    (dataset_path / "meta").mkdir(parents=True)
    info = {"features": {"observation.state": {"dtype": "float32"}, feature: {"dtype": "video"}}}
    (dataset_path / "meta" / "info.json").write_text(json.dumps(info))
    (dataset_path / "meta" / "stats.json").write_text(json.dumps({feature: {}}))
    parquet_path = dataset_path / "data" / "chunk-000" / "file-000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"observation.state": [1.0], feature: [0]}), parquet_path)

    real_fsync = os.fsync

    def killed(fd):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "fsync", killed)
    with pytest.raises(KeyboardInterrupt):
        remove_excluded_camera_features_from_lerobot_dataset(
            dataset_path, excluded_camera_keys={"aux_cam_left"}
        )
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert json.loads((dataset_path / "meta" / "info.json").read_text()) == info
    assert feature in pq.read_table(parquet_path).column_names

    removed = remove_excluded_camera_features_from_lerobot_dataset(
        dataset_path, excluded_camera_keys={"aux_cam_left"}
    )
    assert removed == [feature]
    assert pq.read_table(parquet_path).column_names == ["observation.state"]
    assert feature not in json.loads((dataset_path / "meta" / "info.json").read_text())["features"]
    assert json.loads((dataset_path / "meta" / "stats.json").read_text()) == {}


def test_station_camera_config_is_consistent() -> None:
    # The current robot station: a wrist ZED (both eyes) + two left-eye side cams.
    assert set(STATION_CAMERA_KEYS_BY_ROLE) == {"wrist_left", "wrist_right", "side_1", "side_2"}
    # DEFAULT_CAMERA_KEYS is derived from the role map (single source of truth).
    assert DEFAULT_CAMERA_KEYS == ",".join(STATION_CAMERA_KEYS_BY_ROLE.values())
    # Both wrist eyes share one ZED serial, recorded as _left and _right.
    wrist_serials = {
        STATION_CAMERA_KEYS_BY_ROLE["wrist_left"].rsplit("_", 1)[0],
        STATION_CAMERA_KEYS_BY_ROLE["wrist_right"].rsplit("_", 1)[0],
    }
    assert len(wrist_serials) == 1
    assert STATION_CAMERA_KEYS_BY_ROLE["wrist_left"].endswith("_left")
    assert STATION_CAMERA_KEYS_BY_ROLE["wrist_right"].endswith("_right")
    # The excluded set must never drop a wanted station camera.
    assert DEFAULT_EXCLUDED_CAMERA_KEYS.isdisjoint(STATION_CAMERA_KEYS_BY_ROLE.values())


def test_station_config_example_is_the_default(monkeypatch) -> None:
    from mulligan.real.robot import cameras

    monkeypatch.delenv(cameras.STATION_CONFIG_ENV, raising=False)
    assert cameras.station_config_path() == cameras.DEFAULT_STATION_CONFIG_PATH
    parsed = cameras.load_station_camera_config(cameras.DEFAULT_STATION_CONFIG_PATH)
    assert parsed["roles"] == cameras.STATION_CAMERA_KEYS_BY_ROLE
    assert parsed["default_crops"] == cameras.STATION_CAMERA_DEFAULT_CROPS
    assert parsed["excluded"] == cameras.DEFAULT_EXCLUDED_CAMERA_KEYS


def test_station_config_env_override_and_validation(tmp_path, monkeypatch) -> None:
    import yaml

    from mulligan.real.robot import cameras

    raw = yaml.safe_load(cameras.DEFAULT_STATION_CONFIG_PATH.read_text())
    raw["cameras"]["roles"]["side_2"] = "11111111_left"
    good = tmp_path / "station.yaml"
    good.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv(cameras.STATION_CONFIG_ENV, str(good))
    assert cameras.station_config_path() == good
    assert cameras.load_station_camera_config(good)["roles"]["side_2"] == "11111111_left"

    raw["cameras"]["roles"].pop("side_2")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="cameras.roles must map exactly"):
        cameras.load_station_camera_config(bad)
    with pytest.raises(FileNotFoundError, match="station config"):
        cameras.load_station_camera_config(tmp_path / "missing.yaml")


def test_station_example_ships_as_package_data() -> None:
    """The default station config loads from the installed package, not the source tree."""
    import fnmatch
    import tomllib
    from pathlib import Path

    from mulligan.real.robot import cameras

    root = Path(__file__).resolve().parents[2]
    packaged = Path(str(cameras.DEFAULT_STATION_CONFIG_PATH))
    assert packaged.parent == root / "mulligan" / "real" / "robot"
    assert packaged.read_bytes() == (root / "configs/real/station.example.yaml").read_bytes()
    patterns = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"][
        "package-data"
    ]["mulligan"]
    rel = packaged.relative_to(root / "mulligan").as_posix()
    assert any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)
