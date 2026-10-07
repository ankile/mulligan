"""Outcome-editor camera crop resolution and display behavior."""

import numpy as np
import pytest

from mulligan.real.robot.cameras import STATION_CAMERA_DEFAULT_CROPS
from mulligan.real.lifecycle import get_task_spec
from mulligan.tools.outcome_review import (
    build_display_frame,
    camera_role_for_video_key,
    crop_frame_for_outcome_review,
    resolve_camera_crop_boxes,
)


def test_resolve_camera_crops_uses_station_defaults_for_marker() -> None:
    crops = resolve_camera_crop_boxes({"marker_d2"})
    assert crops == STATION_CAMERA_DEFAULT_CROPS
    assert crops is not STATION_CAMERA_DEFAULT_CROPS


def test_resolve_camera_crops_applies_task_overrides() -> None:
    crops = resolve_camera_crop_boxes({"routing_d2"})
    routing_overrides = get_task_spec("routing_d2").camera_crop_overrides

    assert crops["side_1"] == routing_overrides["side_1"]
    assert crops["side_2"] == routing_overrides["side_2"]
    assert crops["wrist_left"] == STATION_CAMERA_DEFAULT_CROPS["wrist_left"]


def test_resolve_camera_crops_rejects_incompatible_mixed_tasks() -> None:
    with pytest.raises(ValueError, match="incompatible camera crop maps"):
        resolve_camera_crop_boxes({"marker_d2", "routing_d2"})


@pytest.mark.parametrize(
    ("camera", "expected_role"),
    [
        ("observation.images.side_1", "side_1"),
        ("observation.images.20000002_left", None),
        ("side_2", "side_2"),
        ("unregistered_left", None),
    ],
)
def test_camera_role_for_video_key(camera: str, expected_role: str | None) -> None:
    assert camera_role_for_video_key(camera) == expected_role


def test_crop_frame_for_outcome_review_uses_stored_space_box() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crops = resolve_camera_crop_boxes({"routing_d2"})

    role_view = crop_frame_for_outcome_review(frame, "observation.images.side_1", crops)
    x0, y0, x1, y1 = crops["side_1"]
    assert role_view.shape == (y1 - y0, x1 - x0, 3)


def test_crop_frame_for_outcome_review_leaves_nonstation_camera_unchanged() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = crop_frame_for_outcome_review(
        frame, "observation.images.overhead", dict(STATION_CAMERA_DEFAULT_CROPS)
    )
    assert result is frame


def test_crop_frame_for_outcome_review_rejects_wrong_stored_resolution() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="on stored frame 100x100"):
        crop_frame_for_outcome_review(
            frame, "observation.images.side_1", dict(STATION_CAMERA_DEFAULT_CROPS)
        )


def test_display_restores_cropped_panels_to_decoded_frame_height() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crops = resolve_camera_crop_boxes({"routing_d2"})
    camera_frames = {
        "observation.images.side_1": [frame],
        "observation.images.side_2": [frame],
    }

    display = build_display_frame(
        camera_frames,
        frame_idx=0,
        total_frames=1,
        ep_idx=0,
        current_outcome="failure",
        pending_outcome=None,
        marked_frame=None,
        soft_truncate=False,
        progress_str="Progress: 0/1",
        scale=1.0,
        camera_crop_boxes=crops,
    )

    # Both ROI panels are aspect-preserving zooms back to the source's 480px height;
    # the N=0 info bar contributes 70px.
    expected_width = sum(
        int((x1 - x0) * 480 / (y1 - y0)) for x0, y0, x1, y1 in (crops["side_1"], crops["side_2"])
    )
    assert display.shape == (550, expected_width, 3)
