"""``mulligan.real.operator_ui.monitor``: monitor key resolution, rendering, and policy crops."""

import pytest

from mulligan.real.robot.cameras import STATION_CAMERA_KEYS_BY_ROLE, role_to_serial_key
from mulligan.real.operator_ui.display import (
    MONITOR_PANEL_H,
    SCHEMATIC_WINDOW_XY,
    monitor_window_xy,
)
from mulligan.real.operator_ui.monitor import (
    DEFAULT_MONITOR_ROLES,
    resolve_monitor_camera_keys,
)


def test_default_roles_are_dp_consumed_pair():
    assert DEFAULT_MONITOR_ROLES == ("side_1", "wrist_left")


def test_none_resolves_to_default_role_serial_keys():
    expected = [role_to_serial_key(role) for role in DEFAULT_MONITOR_ROLES]
    assert resolve_monitor_camera_keys(None) == expected
    # Pin the actual station serials, not just the indirection.
    assert resolve_monitor_camera_keys(None) == ["20000002_left", "10000001_left"]


def test_empty_string_resolves_to_default_role_serial_keys():
    assert resolve_monitor_camera_keys("") == resolve_monitor_camera_keys(None)


def test_role_tokens_resolve_to_serial_keys():
    assert resolve_monitor_camera_keys("side_1") == ["20000002_left"]
    assert resolve_monitor_camera_keys("wrist_left,side_2") == [
        "10000001_left",
        "30000003_left",
    ]


def test_raw_serial_keys_pass_through():
    for serial in STATION_CAMERA_KEYS_BY_ROLE.values():
        assert resolve_monitor_camera_keys(serial) == [serial]


def test_mixed_roles_and_serials_preserve_order():
    assert resolve_monitor_camera_keys("side_1,10000001_left") == [
        "20000002_left",
        "10000001_left",
    ]


def test_tokens_are_stripped_and_empty_tokens_skipped():
    assert resolve_monitor_camera_keys(" side_1 , , wrist_left ,") == [
        "20000002_left",
        "10000001_left",
    ]


def test_unknown_token_raises_valueerror_mentioning_flag():
    with pytest.raises(ValueError, match="--monitor-camera-keys"):
        resolve_monitor_camera_keys("side_1,not_a_camera")


def test_monitor_layout_constants():
    # The teleop collector's operator-window tiling depends on these exact values.
    assert MONITOR_PANEL_H == 360
    assert SCHEMATIC_WINDOW_XY == (10, 30)
    assert monitor_window_xy(0) == (1080, 30)
    assert monitor_window_xy(1) == (1080, 30 + (MONITOR_PANEL_H + 60))


# --- rendering + policy crop boxes (HighGUI replaced by a recording stub) -------------------


class _RecordingCv2:
    WINDOW_NORMAL = 0
    COLOR_BGRA2BGR = 1
    INTER_AREA = 3
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self):
        self.shown: list[tuple[str, tuple[int, ...]]] = []

    def imshow(self, name, panel):
        self.shown.append((name, panel.shape))

    def cvtColor(self, frame, _code):
        return frame[..., :3]

    def resize(self, frame, size, interpolation=None):
        import numpy as np

        w, h = size
        return np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)

    def putText(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return lambda *args, **kwargs: 1.0 if name == "getWindowProperty" else -1


@pytest.fixture
def stub_cv2(monkeypatch):
    from mulligan.real.operator_ui import display, monitor

    stub = _RecordingCv2()
    monkeypatch.setattr(display, "cv2", stub)
    monkeypatch.setattr(monitor, "cv2", stub)
    monkeypatch.setattr(display, "_placed_windows", set())
    monkeypatch.setenv("DISPLAY", ":0")
    return stub


def test_render_camera_monitor_shows_each_known_camera_at_panel_height(stub_cv2):
    import numpy as np

    from mulligan.real.operator_ui.monitor import render_camera_monitor

    side_1 = role_to_serial_key("side_1")
    obs_image = {
        side_1: np.zeros((720, 1280, 4), dtype=np.uint8),
        "unknown_cam": np.zeros((4, 4, 3)),
    }
    render_camera_monitor(obs_image, [side_1, "missing_cam"])
    assert [name for name, _ in stub_cv2.shown] == ["cam: side_1 (crop)"]
    assert stub_cv2.shown[0][1][0] == MONITOR_PANEL_H


def test_policy_role_crop_boxes_returns_the_policy_role_crops():
    from types import SimpleNamespace

    from mulligan.real.operator_ui.monitor import policy_role_crop_boxes

    policy = SimpleNamespace(camera_crops={"side_1": (0, 0, 10, 10), "wrist_left": (1, 1, 2, 2)})
    assert policy_role_crop_boxes(policy) == {"side_1": (0, 0, 10, 10), "wrist_left": (1, 1, 2, 2)}
    assert policy_role_crop_boxes(SimpleNamespace(camera_crops={})) is None
    assert policy_role_crop_boxes(object()) is None


def test_shared_policy_crop_boxes_falls_back_to_station_default_when_policies_disagree(capsys):
    from types import SimpleNamespace

    from mulligan.real.operator_ui.monitor import shared_policy_crop_boxes

    a = SimpleNamespace(camera_crops={"side_1": (0, 0, 10, 10)})
    b = SimpleNamespace(camera_crops={"side_1": (5, 5, 20, 20)})
    assert shared_policy_crop_boxes([a, a]) == {"side_1": (0, 0, 10, 10)}
    assert shared_policy_crop_boxes([a, b]) is None  # would unblind the operator
    assert "unblind" in capsys.readouterr().out
    assert shared_policy_crop_boxes([object()]) is None
    with pytest.raises(ValueError):
        shared_policy_crop_boxes([])
