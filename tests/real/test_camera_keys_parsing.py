"""The explicit-camera-keys contract for multi-camera real collection.

marker_d2 records a 2nd side camera AND both wrist eyes (``_left`` and
``_right``). The single-suffix ``--camera-filter`` cannot express that, so
collection must pass explicit ``--camera-keys``. These tests pin the pure parse /
serial-dedup contract that the marker_d2 collection script depends on -- it was
entirely unpinned before (no test referenced these functions). Assertions are
serial-agnostic where possible so a future robot-room move does not break them.
"""

from __future__ import annotations

import pytest

from mulligan.real.robot.cameras import DEFAULT_EXCLUDED_CAMERA_KEYS
from mulligan.real.collect.rollout import camera_serials_from_keys, parse_camera_keys


def test_none_and_empty_return_none():
    assert parse_camera_keys(None, "_left") is None
    assert parse_camera_keys("", "_left") is None
    assert parse_camera_keys("   ", "_left") is None


def test_already_suffixed_keys_kept_verbatim():
    assert parse_camera_keys("10000001_left,20000002_left", "_left") == [
        "10000001_left",
        "20000002_left",
    ]


def test_both_wrist_eyes_kept_under_left_filter():
    # The defining marker_d2 case: a _right eye must survive even though the filter
    # is _left, because both wrist eyes are explicitly requested.
    keys = parse_camera_keys("10000001_left,10000001_right", "_left")
    assert keys == ["10000001_left", "10000001_right"]


def test_bare_serial_gets_filter_appended():
    assert parse_camera_keys("10000001", "_left") == ["10000001_left"]
    assert parse_camera_keys("10000001", "_right") == ["10000001_right"]


def test_bare_serial_with_empty_filter_raises():
    with pytest.raises(ValueError, match="bare serial"):
        parse_camera_keys("10000001", "")


def test_observation_images_prefix_stripped():
    assert parse_camera_keys("observation.images.10000001_left", "_left") == ["10000001_left"]


def test_duplicate_keys_raise():
    with pytest.raises(ValueError, match="duplicates"):
        parse_camera_keys("10000001_left,10000001_left", "_left")


def test_excluded_key_raises():
    # Use an actual member of the excluded set so this stays valid if it changes.
    excluded_key = next(iter(DEFAULT_EXCLUDED_CAMERA_KEYS))
    with pytest.raises(ValueError, match="excluded camera"):
        parse_camera_keys(excluded_key, "_left")


def test_serials_from_keys_dedups_wrist_eyes_preserving_order():
    serials = camera_serials_from_keys(["10000001_left", "10000001_right", "20000002_left"])
    assert serials == ["10000001", "20000002"]


def test_serials_from_keys_strips_observation_prefix():
    assert camera_serials_from_keys(["observation.images.10000001_left"]) == ["10000001"]


def test_full_marker_d2_four_camera_set_roundtrip():
    # 2 side cams (distinct serials) + both eyes of one wrist serial -> 4 keys, but
    # only 3 distinct ZED serials to open.
    keys = parse_camera_keys("11111111_left,22222222_left,10000001_left,10000001_right", "_left")
    assert keys is not None
    assert keys == ["11111111_left", "22222222_left", "10000001_left", "10000001_right"]
    assert camera_serials_from_keys(keys) == ["11111111", "22222222", "10000001"]
