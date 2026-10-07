"""Compose the operator window from a cached placement diagram and live session state."""

from __future__ import annotations

import cv2
import numpy as np

from mulligan.real.operator_ui.progress import SessionProgress, duration

PANEL_WIDTH = 1000
DIAGRAM_HEIGHT = 796
HEADER_HEIGHT = 130


def _color(name: str) -> tuple[int, int, int]:
    from mulligan.plotting.colors import OPERATOR_CARD_COLORS

    value = OPERATOR_CARD_COLORS[name].lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (4, 2, 0))


def fit_diagram(image: np.ndarray) -> np.ndarray:
    """Resize once per target, keeping the whole diagram and its aspect ratio."""
    h, w = image.shape[:2]
    scale = min(PANEL_WIDTH / w, DIAGRAM_HEIGHT / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((DIAGRAM_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    left = (PANEL_WIDTH - resized.shape[1]) // 2
    top = (DIAGRAM_HEIGHT - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def compose_panel(progress: SessionProgress, diagram: np.ndarray | None) -> np.ndarray:
    """Cheap raster-only redraw. No figure rendering, disk IO or robot access."""
    height = HEADER_HEIGHT + (DIAGRAM_HEIGHT if diagram is not None else 0)
    canvas = np.full((height, PANEL_WIDTH, 3), _color("panel_background"), dtype=np.uint8)

    def text(value: str, x: int, y: int, size: float = 0.6, color: str = "panel_text") -> None:
        width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
        if width > PANEL_WIDTH - x - 20:
            raise ValueError(f"Operator panel text does not fit: {value!r}")
        cv2.putText(
            canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, _color(color), 1, cv2.LINE_AA
        )

    text(progress.phase, 24, 24, 0.66)
    text(progress.context, 24, 46, 0.48, "panel_muted")
    for idx, (label, value) in enumerate(progress.header_metrics()):
        x = 24 + idx * 245
        text(label, x, 66, 0.36, "panel_muted")
        text(value, x, 87, 0.56)
    cv2.rectangle(canvas, (24, 96), (976, 99), _color("panel_track"), -1)
    fraction = progress.bar_fraction
    if fraction:
        end = 24 + round(952 * fraction)
        cv2.rectangle(canvas, (24, 96), (end, 99), _color("panel_accent"), -1)
    detail = progress.detail
    if progress.rollout_started is not None:
        steps = str(progress.step) + (f" / {progress.max_steps}" if progress.max_steps else "")
        detail = (
            f"Step {steps}   |   Episode {duration(progress.clock() - progress.rollout_started)}"
        )
        if progress.max_marks:
            detail += f"   |   Subgoals {progress.marks} / {progress.max_marks}"
    text(detail, 24, 119, 0.46, "panel_muted")
    if diagram is not None:
        canvas[HEADER_HEIGHT : HEADER_HEIGHT + DIAGRAM_HEIGHT] = diagram
    return canvas
