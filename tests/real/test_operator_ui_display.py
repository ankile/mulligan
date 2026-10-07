"""``mulligan.real.operator_ui.display``: display detection, window tiling, and the prewarm.

HighGUI is replaced by a recording stub so the window contract (create + size + move ONCE
per name, repaint every call, never block on a key) is pinned without an X server.
"""

from __future__ import annotations

import numpy as np
import pytest

from mulligan.real.operator_ui import display
from mulligan.real.operator_ui.display import (
    SCHEMATIC_WINDOW_XY,
    close_windows,
    has_display,
    monitor_window_xy,
    prewarm_highgui,
    require_display,
    show_image_window,
)


class RecordingCv2:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 4

    def __init__(self):
        self.calls: list[tuple] = []
        self.visible = 1.0  # what getWindowProperty(WND_PROP_VISIBLE) reports

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))
            if name == "waitKey":
                return -1
            if name == "getWindowProperty":
                return self.visible
            return None

        return record

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def cv2_stub(monkeypatch):
    stub = RecordingCv2()
    monkeypatch.setattr(display, "cv2", stub)
    monkeypatch.setattr(display, "_placed_windows", set())
    return stub


@pytest.fixture
def with_display(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(display, "_x11_probe_result", True)  # skip the xdpyinfo subprocess


@pytest.fixture
def headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


def test_has_display_reads_either_env_var(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not has_display()
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert has_display()


def test_require_display_fails_loud_headless(headless):
    with pytest.raises(RuntimeError, match="--monitor-cameras needs an X11/Wayland display"):
        require_display("--monitor-cameras")


def test_window_is_created_and_tiled_once_then_only_repainted(cv2_stub, with_display):
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    show_image_window("card", image, xy=(10, 30), scale=2.0)
    assert cv2_stub.names()[:4] == ["namedWindow", "resizeWindow", "moveWindow", "imshow"]
    assert ("resizeWindow", "card", 400, 240) in cv2_stub.calls
    assert ("moveWindow", "card", 10, 30) in cv2_stub.calls
    cv2_stub.calls.clear()
    show_image_window("card", image, xy=(10, 30))
    assert cv2_stub.names() == ["getWindowProperty", "imshow"]  # repaint only, no pump
    cv2_stub.calls.clear()
    show_image_window("card", image, xy=(10, 30), pump=True)
    assert cv2_stub.names() == ["getWindowProperty", "imshow", "waitKey", "waitKey"]  # card paints


def test_close_windows_forgets_placement_so_a_reshow_recreates(cv2_stub, with_display):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    show_image_window("card", image, xy=SCHEMATIC_WINDOW_XY)
    close_windows()
    assert "destroyAllWindows" in cv2_stub.names()
    cv2_stub.calls.clear()
    show_image_window("card", image, xy=SCHEMATIC_WINDOW_XY)
    assert cv2_stub.names()[0] == "namedWindow"


def test_close_windows_and_prewarm_never_touch_highgui_headless(cv2_stub, headless):
    close_windows()
    prewarm_highgui()
    assert cv2_stub.calls == []


def test_prewarm_creates_and_destroys_one_tiny_window(cv2_stub, with_display):
    prewarm_highgui()
    assert cv2_stub.names() == [
        "namedWindow",
        "resizeWindow",
        "waitKey",
        "destroyWindow",
        "waitKey",
    ]


def test_show_rejects_non_positive_scale(cv2_stub, with_display):
    with pytest.raises(ValueError, match="scale must be positive"):
        show_image_window("card", np.zeros((1, 1, 3), dtype=np.uint8), xy=(0, 0), scale=0)


def test_window_layout_constants():
    # The operator's screen tiling: card top-left, crops down the right.
    assert SCHEMATIC_WINDOW_XY == (10, 30)
    assert monitor_window_xy(0) == (1080, 30)
    assert monitor_window_xy(1) == (1080, 30 + (display.MONITOR_PANEL_H + 60))


def test_dead_x_server_only_warns_at_prewarm_and_raises_when_a_window_is_needed(
    cv2_stub, monkeypatch, capsys
):
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(display, "_x11_probe_result", None)
    monkeypatch.setattr(display.shutil, "which", lambda name: "/usr/bin/xdpyinfo")

    def failing_probe(cmd, **kwargs):
        raise display.subprocess.CalledProcessError(1, cmd, stderr="unable to open display")

    monkeypatch.setattr(display.subprocess, "run", failing_probe)
    prewarm_highgui()  # --help and window-free runs must survive a dead DISPLAY
    assert "WARNING: skipping HighGUI prewarm" in capsys.readouterr().out
    assert cv2_stub.calls == []
    with pytest.raises(RuntimeError, match="X11 server is not reachable"):
        require_display("The initial-state card window")


def test_a_window_the_operator_closed_is_recreated_and_retiled(cv2_stub, with_display):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    show_image_window("card", image, xy=(10, 30))
    cv2_stub.visible = 0.0  # operator hit the window's close button
    cv2_stub.calls.clear()
    show_image_window("card", image, xy=(10, 30))
    assert cv2_stub.names() == [
        "getWindowProperty",
        "namedWindow",
        "resizeWindow",
        "moveWindow",
        "imshow",
    ]
