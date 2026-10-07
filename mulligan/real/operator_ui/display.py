"""OpenCV window plumbing shared by every real-robot operator entrypoint.

One place for the three things every entrypoint needs: whether a
display is reachable, the one-time X11 probe, and how operator windows are created,
tiled, repainted, and closed.

Import order matters. OpenCV's Qt HighGUI backend deadlocks in ``cv2.namedWindow`` when
the first window is created after ``av`` (pulled in by lerobot) has loaded its bundled
``libxcb``. Every entrypoint therefore imports this module (and so ``cv2``) before
lerobot / torch, and calls :func:`prewarm_highgui` at module import when run as a script,
so the first real window is created before that stack exists. A structural test
(``tests/real/test_operator_ui_structure.py``) pins that order.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Window tiling (screen pixels). The schematic (target card) sits top-left; camera monitor
# windows stack down the right of it.
SCHEMATIC_WINDOW_XY = (10, 30)
MONITOR_PANEL_H = 360

_placed_windows: set[str] = set()
_x11_probe_result: bool | str | None = None  # None = not probed; True = ok; str = error


def monitor_window_xy(idx: int) -> tuple[int, int]:
    """Top-left screen position for the idx-th cropped camera monitor window."""
    return (1080, 30 + idx * (MONITOR_PANEL_H + 60))


def has_display() -> bool:
    """True when an X11 or Wayland display is configured for this process."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def probe_x11_display() -> None:
    """Fail loud, once per process, if ``DISPLAY`` points at an unreachable X server.

    A stale ``DISPLAY`` (forwarding dropped, XQuartz not started) otherwise surfaces as a
    hang or an abort inside ``cv2.namedWindow`` at the first target card, several
    minutes into robot setup. Wayland-only sessions are not probed.
    """
    global _x11_probe_result
    if _x11_probe_result is True:
        return
    if isinstance(_x11_probe_result, str):
        raise RuntimeError(_x11_probe_result)
    display = os.environ.get("DISPLAY")
    if not display:
        _x11_probe_result = True
        return
    x_probe = shutil.which("xdpyinfo") or shutil.which("xset")
    if x_probe is None:
        _x11_probe_result = True
        return
    cmd = [x_probe, "-display", display] if x_probe.endswith("xdpyinfo") else [x_probe, "q"]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        _x11_probe_result = (
            f"DISPLAY={display!r} is set, but the X11 server is not reachable. "
            "Start the X server (macOS: XQuartz), reconnect with `ssh -Y <host>`, and retry. "
            f"Probe failure: {detail}"
        )
        raise RuntimeError(_x11_probe_result) from exc
    _x11_probe_result = True


def require_display(purpose: str) -> None:
    """Fail loud at startup when ``purpose`` needs a display and none is reachable."""
    if not has_display():
        raise RuntimeError(
            f"{purpose} needs an X11/Wayland display but neither DISPLAY nor WAYLAND_DISPLAY "
            "is set. Reconnect with `ssh -Y <host>` (macOS: start XQuartz first), or run "
            "without it (--no-show-initial-state-window / drop --monitor-cameras)."
        )
    probe_x11_display()


def prewarm_highgui() -> None:
    """Create and destroy one tiny window so HighGUI initializes before lerobot/av load.

    No-op without a display. Runs at module import time in every entrypoint, before the
    robot / dataset stack is imported (see the module docstring). A DISPLAY that points at
    a dead X server is only WARNED about here (``--help`` and window-free runs must still
    work); :func:`require_display` raises the same probe error loudly the moment a window
    is actually requested, before the robot is touched.
    """
    if not has_display():
        return
    try:
        probe_x11_display()
    except RuntimeError as exc:
        print(f"WARNING: skipping HighGUI prewarm: {exc}", flush=True)
        return
    window = "__mulligan_highgui_prewarm__"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1, 1)
    cv2.waitKey(1)
    cv2.destroyWindow(window)
    cv2.waitKey(1)


def show_image_window(
    name: str,
    image_bgr: np.ndarray,
    *,
    xy: tuple[int, int],
    scale: float = 1.0,
    pump: bool = False,
) -> None:
    """Show ``image_bgr`` in the window ``name``, creating and tiling it on first use.

    The window is created, sized to the image (times ``scale``), and moved to ``xy`` once;
    later calls only repaint it, so an operator's manual resize survives a card refresh.

    Painting happens on the caller's next ``cv2.waitKey`` (the operator key poll in
    ``keys.read_opencv_key``), and by default this function does NOT pump the event loop
    itself: a ``waitKey`` here would swallow any key the operator presses during it, and
    the camera monitor calls this at 15 Hz. ``pump=True`` is for the target card, which
    is shown right before a blocking robot reset with no key poll in between.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if name in _placed_windows and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
        # The operator closed it with the window button; a bare imshow would silently
        # re-create it auto-sized at the default position. Re-create and re-tile instead.
        _placed_windows.discard(name)
    if name not in _placed_windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        height, width = image_bgr.shape[:2]
        cv2.resizeWindow(name, round(width * scale), round(height * scale))
        cv2.moveWindow(name, int(xy[0]), int(xy[1]))
        _placed_windows.add(name)
    cv2.imshow(name, image_bgr)
    if pump:
        cv2.waitKey(1)
        cv2.waitKey(1)


def show_image_file_window(
    name: str,
    path: Path,
    *,
    xy: tuple[int, int],
    scale: float = 1.0,
) -> None:
    """Read a PNG from disk and show it, pumping the event loop so it paints at once.

    Used for the target card, which is typically shown right before a blocking robot
    reset: without the pump the operator would stare at the previous card during the reset.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image {path}")
    show_image_window(name, image, xy=xy, scale=scale, pump=True)


def close_windows() -> None:
    """Destroy every operator window. Safe to call repeatedly and without a display."""
    if not has_display():
        _placed_windows.clear()
        return
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    _placed_windows.clear()
