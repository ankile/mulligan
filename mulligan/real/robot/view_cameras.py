"""Live ZED camera viewer with frame diagnostics.

Opens all connected ZED cameras and displays their left streams with
real-time diagnostics: SDK frame drops, grab error codes, and frame
corruption detection.

Press 'q' or Ctrl+C to quit. Press 's' to save current frame to disk.

Usage:
    python -m mulligan.real.robot.view_cameras
    python -m mulligan.real.robot.view_cameras --left-only
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

# droid_compat patches the OpenCV aruco API that droid.misc.parameters calls at import; it must
# run before any droid import (an import-sorter would otherwise move droid first).
from mulligan.real.robot.droid_compat import _aruco  # noqa: F401

# isort: split
import droid.camera_utils.camera_readers.zed_camera as _zed_mod
from droid.camera_utils.camera_readers.zed_camera import gather_zed_cameras
from mulligan.real.robot.cameras import crop_frame_to_role_view


def crop_preview_panel(native_frame_bgr, cam_key):
    """Preview a station camera's DEFAULT crop as the policy will see it.

    Thin wrapper over ``cameras.crop_frame_to_role_view`` (the shared resize-to-stored
    -then-crop pipeline, INTER_AREA), so the preview matches collection/train and the live
    teleop monitor. Returns ``(role, cropped_bgr)`` or ``None`` (no station role / crop)."""
    return crop_frame_to_role_view(native_frame_bgr, cam_key)


# --------------------------------------------------------------------------- #
# Frame corruption detection
# --------------------------------------------------------------------------- #


def check_frame_corruption(frame, prev_frame=None):
    """Detect common ZED frame corruption patterns.

    Returns list of (reason, detail_string) tuples. Empty list = frame is OK.
    """
    issues = []

    if frame is None or frame.size == 0:
        return [("empty", "")]

    # 1. Solid-color frame (grab returned garbage)
    if frame.std() < 0.5:
        return [("solid_color", f"std={frame.std():.2f}")]

    # 2. Green-tinted frame — classic ZED USB corruption artifact.
    #    A corrupted frame often has the green channel dominate unnaturally.
    if frame.ndim == 3 and frame.shape[2] >= 3:
        means = frame[:, :, :3].mean(axis=(0, 1))  # BGR
        green_mean = means[1]
        other_mean = (means[0] + means[2]) / 2
        if green_mean > 150 and green_mean > other_mean * 1.8:
            issues.append(("green_tint", f"G={green_mean:.0f} vs avg(BR)={other_mean:.0f}"))

    # 3. Horizontal tearing: detect sharp row-to-row brightness discontinuities.
    #    Real images have gradual transitions; tearing creates a hard seam.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    row_means = gray.mean(axis=1)
    row_diffs = np.abs(np.diff(row_means))
    # A jump > 40 brightness levels between adjacent rows is suspicious
    big_jumps = np.where(row_diffs > 40)[0]
    if len(big_jumps) > 0:
        # Filter: only flag if the jump is much bigger than typical variation
        median_diff = np.median(row_diffs)
        if median_diff > 0:
            anomalous = big_jumps[row_diffs[big_jumps] > max(median_diff * 8, 30)]
            if len(anomalous) > 0:
                issues.append(
                    (
                        "row_discontinuity",
                        f"rows={anomalous.tolist()[:5]} max_jump={row_diffs[anomalous].max():.0f}",
                    )
                )

    # 4. Block artifacts: look for rectangular regions where every pixel is
    #    identical (std ≈ 0), which is the signature of USB bit errors.
    #    Real surfaces — even white walls — have camera sensor noise (std ~1-3).
    #    Corruption produces perfectly uniform blocks (std < 0.1) that stand out
    #    against any non-trivial surroundings.
    h, w = gray.shape
    block_size = 32
    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block = gray[by : by + block_size, bx : bx + block_size]
            # Truly identical pixels — no real surface produces this
            if block.std() < 0.1:
                surround = gray[
                    max(0, by - block_size) : by + 2 * block_size,
                    max(0, bx - block_size) : bx + 2 * block_size,
                ]
                if surround.std() > 30:
                    issues.append(
                        (
                            "block_artifact",
                            f"at ({bx},{by}) block_std={block.std():.2f} surround_std={surround.std():.0f}",
                        )
                    )
                    break
        if any(r == "block_artifact" for r, _ in issues):
            break

    # 5. Frozen frame (identical to previous)
    if prev_frame is not None and prev_frame.shape == frame.shape:
        if np.array_equal(frame, prev_frame):
            issues.append(("frozen", ""))

    return issues


# --------------------------------------------------------------------------- #
# ZED camera diagnostics wrapper
# --------------------------------------------------------------------------- #


class ZedDiagnostics:
    """Wraps a DROID ZedCamera with SDK-level diagnostics."""

    def __init__(self, cam):
        self.cam = cam
        self.grab_errors = defaultdict(int)  # error_code -> count

    def read_camera_with_diag(self):
        """Like cam.read_camera() but tracks grab error codes.

        Returns (data_dict, diag_dict) where diag_dict contains:
            - grab_ok: bool
            - grab_error: str or None
            - frame_timestamp_ms: capture timestamp from SDK
        """
        diag = {
            "grab_ok": False,
            "grab_error": None,
            "frame_timestamp_ms": 0,
        }

        # Grab
        err = self.cam._cam.grab(self.cam._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            err_name = str(err).split(".")[-1]
            self.grab_errors[err_name] += 1
            diag["grab_error"] = err_name
            return None, diag

        diag["grab_ok"] = True

        # Frame timestamp
        ts = self.cam._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        diag["frame_timestamp_ms"] = ts.get_milliseconds()

        # Retrieve images (reuse the cam's existing sl.Mat buffers)
        from copy import deepcopy

        data_dict = {"image": {}}

        self.cam._cam.retrieve_image(
            self.cam._left_img,
            sl.VIEW.LEFT,
            resolution=self.cam.zed_resolution,
        )
        data_dict["image"][self.cam.serial_number + "_left"] = deepcopy(
            self.cam._left_img.get_data()
        )

        self.cam._cam.retrieve_image(
            self.cam._right_img,
            sl.VIEW.RIGHT,
            resolution=self.cam.zed_resolution,
        )
        data_dict["image"][self.cam.serial_number + "_right"] = deepcopy(
            self.cam._right_img.get_data()
        )

        return data_dict, diag


def main():
    parser = argparse.ArgumentParser(description="Live ZED camera viewer with frame diagnostics")
    parser.add_argument("--left-only", action="store_true", help="Only display left camera images")
    parser.add_argument(
        "--show-crops",
        action="store_true",
        help="Also show each station camera's feed AFTER its default crop "
        "(camera_utils.STATION_CAMERA_DEFAULT_CROPS), to sanity-check the cropped view "
        "the policy receives (native -> 640x480 stored -> crop).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./camera_debug",
        help="Directory to save debug frames (press 's')",
    )
    parser.add_argument("--camera-fps", type=int, default=15, help="ZED capture FPS (default: 15)")
    parser.add_argument(
        "--camera-resolution",
        type=str,
        default="HD720",
        choices=["VGA", "HD720", "HD1080", "HD2K"],
        help="ZED capture resolution (default: HD720)",
    )
    args = parser.parse_args()

    # Patch DROID camera params before init
    _zed_mod.standard_params["camera_fps"] = args.camera_fps
    _zed_mod.standard_params["camera_resolution"] = {
        "VGA": sl.RESOLUTION.VGA,
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K,
    }[args.camera_resolution]

    print("Discovering ZED cameras...")
    cameras = gather_zed_cameras()
    if not cameras:
        print("No ZED cameras found!")
        return

    print(f"Found {len(cameras)} camera(s):")
    for cam in cameras:
        print(f"  Serial: {cam.serial_number}")

    # Configure and initialize cameras
    for cam in cameras:
        cam.set_reading_parameters(
            image=True,
            concatenate_images=False,
            resolution=(0, 0),
        )
        cam.set_trajectory_mode()

    # Wrap with diagnostics
    diag_cams = [ZedDiagnostics(cam) for cam in cameras]

    print(
        f"\nCamera capture: {args.camera_fps} FPS @ {args.camera_resolution}. Press 'q' to quit, 's' to save frame."
    )
    print("Diagnostics overlay: grab errors, SDK drops, corruption detection\n")

    # Stats
    prev_frames = {}
    stats = {
        "total_frames": 0,
        "grab_errors": defaultdict(int),
        "corruption": defaultdict(lambda: defaultdict(int)),  # cam_key -> {reason: count}
        "saved_count": 0,
    }
    session_start = time.time()
    prev_timestamps = {}  # cam_serial -> last frame timestamp

    try:
        while True:
            loop_start = time.time()
            # One row per camera (full feed + optional crop), stacked vertically below.
            camera_rows = []

            for dcam in diag_cams:
                data_dict, diag = dcam.read_camera_with_diag()

                if not diag["grab_ok"]:
                    stats["grab_errors"][diag["grab_error"]] += 1
                    continue

                images = data_dict.get("image", {})
                for key in sorted(images.keys()):
                    if args.left_only and not key.endswith("_left"):
                        continue

                    frame = images[key]
                    stats["total_frames"] += 1

                    # BGRA -> BGR for display
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Corruption check
                    issues = check_frame_corruption(frame, prev_frames.get(key))
                    prev_frames[key] = frame.copy()

                    for reason, _detail in issues:
                        stats["corruption"][key][reason] += 1

                    # Timing check: detect frame timestamp gaps
                    ts = diag["frame_timestamp_ms"]
                    serial = dcam.cam.serial_number
                    if serial in prev_timestamps:
                        dt = ts - prev_timestamps[serial]
                        if dt < 0:
                            stats["corruption"][key]["timestamp_backwards"] += 1
                    prev_timestamps[serial] = ts

                    # Resize for display
                    display_frame = cv2.resize(
                        frame,
                        (640, 360),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    # --- Overlay diagnostics ---
                    is_bad = len(issues) > 0
                    color = (0, 0, 255) if is_bad else (0, 255, 0)

                    # Line 1: camera key + corruption flag
                    label = key
                    if is_bad:
                        reasons = ", ".join(r for r, _ in issues)
                        label = f"{key} [{reasons}]"
                    cv2.putText(
                        display_frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                    )

                    # Line 2: grab errors summary
                    if dcam.grab_errors:
                        err_text = " ".join(f"{k}:{v}" for k, v in dcam.grab_errors.items())
                        cv2.putText(
                            display_frame,
                            f"Grab err: {err_text}",
                            (10, 44),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 100, 255),
                            1,
                        )

                    # Line 3: corruption counts
                    cam_corrupt = stats["corruption"].get(key, {})
                    if cam_corrupt:
                        corrupt_text = " ".join(f"{k}:{v}" for k, v in cam_corrupt.items())
                        cv2.putText(
                            display_frame,
                            corrupt_text,
                            (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                        )

                    # This camera's row: its full feed and (optionally) the cropped view
                    # side-by-side. Rows are stacked vertically below, so each camera
                    # renders at full panel size instead of being squeezed into one wide
                    # strip when there are 4 cameras (+ crops).
                    row_panels = [display_frame]

                    # Optional: the same feed AFTER the role's default crop, so the
                    # operator can sanity-check the cropped view the policy will see.
                    if args.show_crops:
                        preview = crop_preview_panel(frame, key)
                        if preview is not None:
                            role, cropped = preview
                            ch, cw = cropped.shape[:2]
                            disp_w = max(1, int(round(360 * cw / ch)))
                            crop_panel = cv2.resize(
                                cropped, (disp_w, 360), interpolation=cv2.INTER_LINEAR
                            )
                            cv2.putText(
                                crop_panel,
                                f"{role} [CROP]",
                                (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 255),
                                2,
                            )
                            row_panels.append(crop_panel)

                    camera_rows.append(np.hstack(row_panels))

            if camera_rows:
                # np.vstack needs equal-width rows; pad narrower rows (a camera with no
                # crop, or a narrower crop) on the right with black up to the widest row.
                max_w = max(row.shape[1] for row in camera_rows)
                padded = [
                    row
                    if row.shape[1] == max_w
                    else cv2.copyMakeBorder(
                        row, 0, 0, 0, max_w - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    for row in camera_rows
                ]
                display = np.vstack(padded)
                cv2.imshow("ZED Cameras", display)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
            elif keypress == ord("s"):
                # Save current frames to disk for inspection
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                for key, frame in prev_frames.items():
                    ts = int(time.time() * 1000)
                    path = save_dir / f"{key}_{ts}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"  Saved: {path}")
                stats["saved_count"] += 1

            elapsed = time.time() - loop_start
            remaining = 1.0 / args.camera_fps - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for cam in cameras:
            cam.disable_camera()
        cv2.destroyAllWindows()

        # --- Session summary ---
        duration = time.time() - session_start
        print(f"\n{'=' * 60}")
        print(f"Session summary ({duration:.1f}s)")
        print(f"{'=' * 60}")
        print(f"Total frames displayed: {stats['total_frames']}")

        if stats["grab_errors"]:
            print("\nGrab errors (camera.grab() != SUCCESS):")
            for err, count in sorted(stats["grab_errors"].items()):
                print(f"  {err}: {count}")
        else:
            print("Grab errors: 0")

        if stats["corruption"]:
            print("\nCorruption detected:")
            for key, reasons in sorted(stats["corruption"].items()):
                total_bad = sum(reasons.values())
                breakdown = ", ".join(f"{r}: {c}" for r, c in sorted(reasons.items()))
                print(f"  {key}: {total_bad} ({breakdown})")
        else:
            print("Corruption detected: 0")

        if stats["saved_count"] > 0:
            print(f"\nSaved {stats['saved_count']} frame snapshots to {args.save_dir}/")

        print("Done.")


if __name__ == "__main__":
    main()
