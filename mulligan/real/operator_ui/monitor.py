"""Live cropped camera monitor windows for real-robot operator views."""

from __future__ import annotations

import cv2

from mulligan.real.robot.cameras import (
    STATION_CAMERA_KEYS_BY_ROLE,
    crop_frame_to_role_view,
    role_to_serial_key,
)
from mulligan.real.operator_ui.display import (
    MONITOR_PANEL_H,
    monitor_window_xy,
    show_image_window,
)

DEFAULT_MONITOR_ROLES = ("side_1", "wrist_left")


def resolve_monitor_camera_keys(spec_str: str | None) -> list[str]:
    """Resolve monitor ROLE names or raw station serial keys to live camera serial keys."""
    if not spec_str:
        return [role_to_serial_key(role) for role in DEFAULT_MONITOR_ROLES]
    station_serials = set(STATION_CAMERA_KEYS_BY_ROLE.values())
    keys = []
    for tok in (t.strip() for t in spec_str.split(",")):
        if not tok:
            continue
        if tok in STATION_CAMERA_KEYS_BY_ROLE:
            keys.append(role_to_serial_key(tok))
        elif tok in station_serials:
            keys.append(tok)
        else:
            raise ValueError(
                f"--monitor-camera-keys entry {tok!r} is not a known station role "
                f"{sorted(STATION_CAMERA_KEYS_BY_ROLE)} or serial {sorted(station_serials)}"
            )
    return keys


def render_camera_monitor(
    obs_image: dict,
    monitor_camera_keys: list[str],
    crop_boxes: dict[str, tuple[int, int, int, int]] | None = None,
) -> None:
    """Show each monitored camera's crop policy view in its own tiled window.

    ``crop_boxes`` (role-keyed, stored-space -- see ``crop_frame_to_role_view``) overrides
    the station default crops so the monitor shows the loaded policy's ACTUAL trained crop
    (per-task overrides differ from the station defaults).
    Without it the station-default crop is shown (teleop, no policy loaded).

    The caller's single OpenCV key poll captures keystrokes from these windows.
    """
    shown = 0
    for cam_key in monitor_camera_keys:
        frame = obs_image.get(cam_key)
        if frame is None:
            continue
        view = crop_frame_to_role_view(frame, cam_key, crop_boxes=crop_boxes)
        if view is None:
            continue
        role, cropped = view
        if cropped.ndim == 3 and cropped.shape[2] == 4:
            # Convert the small crop, not the full native frame (this runs every step).
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGRA2BGR)
        h, w = cropped.shape[:2]
        disp_w = max(1, int(round(MONITOR_PANEL_H * w / h)))
        panel = cv2.resize(cropped, (disp_w, MONITOR_PANEL_H), interpolation=cv2.INTER_AREA)
        show_image_window(f"cam: {role} (crop)", panel, xy=monitor_window_xy(shown))
        shown += 1


def policy_role_crop_boxes(policy: object) -> dict[str, tuple[int, int, int, int]] | None:
    """The loaded policy's ROLE-keyed crop boxes for the monitor, or None.

    Policies carry their crops keyed by station role in stored-frame pixels, the space
    ``render_camera_monitor`` draws in. Policies without ``camera_crops`` get the station
    default crops.
    """
    return dict(getattr(policy, "camera_crops", None) or {}) or None


def shared_policy_crop_boxes(policies: list[object]) -> dict[str, tuple[int, int, int, int]] | None:
    """Crop boxes for a BLINDED session: the policies' boxes only if every policy agrees.

    Showing one policy's crop would leak its identity through the blind protocol (the
    operator sees the framing change with the arm), so when the loaded policies disagree the
    monitor falls back to the station-default crops with a warning. Returns None for the
    station default.
    """
    if not policies:
        raise ValueError("shared_policy_crop_boxes needs at least one policy")
    per_policy = [policy_role_crop_boxes(policy) for policy in policies]
    if any(boxes != per_policy[0] for boxes in per_policy[1:]):
        print(
            "WARNING: the loaded policies disagree on camera_crop_boxes; the camera monitor "
            "falls back to station-default crops (showing one policy's crop would unblind "
            "the operator).",
            flush=True,
        )
        return None
    if per_policy[0]:
        print(f"Camera monitor crops = shared policy camera_crop_boxes: {per_policy[0]}")
    return per_policy[0]
