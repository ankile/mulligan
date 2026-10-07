"""Initial-state target cards: the per-task placement schematic the operator sets up from.

One renderer per task family (marker / square / routing), all behind
:func:`build_initial_state_card_figure` and :func:`write_initial_state_card`, all styled by
one :class:`CardStyle`. Collection can add quota/ETA lines; eval session information
lives in the surrounding live panel. Geometry comes from the task
registry and the target row -- never from hand-typed constants -- so the drawn scene and
the printed coordinates cannot disagree.

Pure matplotlib (Agg figures, no pyplot): importable without torch / lerobot / cv2 and
without switching the process-wide backend (card previews, offline scripts).
Showing the PNG in a window is :meth:`mulligan.real.operator_ui.session.OperatorUI.show_card`.
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.text import Text

from mulligan.real.collect.initial_states import (
    M_TO_INCH,
    ROUTING_D2_INITIAL_STATE_KEYS,
    NUT_PEG_INITIAL_STATE_KEYS,
    InitialStateTarget,
    _format_initial_state_target,
    _manifest_keys,
)
from mulligan.real.lifecycle.tasks import find_task_spec_by_task_name


class _LazyCardPalette:
    """``OPERATOR_CARD_COLORS`` resolved on first use.

    ``mulligan.plotting`` imports pyplot and forces the Agg backend at import time; deferring it
    to the first card render keeps this module (and everything that imports it: the eval
    stack, the preview CLI) from switching the process-wide backend as a side effect.
    """

    def __getitem__(self, name: str) -> str:
        from mulligan.plotting.colors import OPERATOR_CARD_COLORS

        return OPERATOR_CARD_COLORS[name]


CARD_COLORS = _LazyCardPalette()

CARD_DPI = 150


@dataclass(frozen=True)
class CardStyle:
    """How a card is titled.

    ``mode_label`` names the situation ("FIRST ROLLOUT", "Policy B", "Policy A Retry").
    ``display_label`` replaces the default "target #N" (collection uses a running start
    count). ``show_coordinates`` explicitly adds textual coordinates to the title.
    Session statistics (quota, pace, ETA) never go on the card: the card PNG is the
    per-target artifact recorded in the ledger, and the live panel header shows them.
    """

    mode_label: str | None = None
    display_label: str | None = None
    show_coordinates: bool = False


def card_filename(target: InitialStateTarget) -> str:
    """Card PNG name, pinned to the manifest row (stable, 1:1 with the manifest).

    Several episodes (retries, CF replays) can share a row, so the on-card start label
    is NOT part of the filename.
    """
    return f"target_{target.manifest_idx:04d}.png"


def _new_card_figure(width_in: float, height_in: float):
    """A standalone Agg figure: no pyplot, so importing this module never switches the
    process-wide matplotlib backend or touches pyplot's figure registry."""
    fig = Figure(figsize=(width_in, height_in), layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot()


def _card_title(task_name: str, style: CardStyle) -> str:
    title = _task_display_name(task_name)
    if style.mode_label is not None:
        title += f" - {style.mode_label}"
    return title


def _set_card_title(
    ax,
    target: InitialStateTarget,
    *,
    task_name: str,
    style: CardStyle,
    wrap_width: int = 84,
) -> None:
    """Title every card the same way; long lines wrap instead of running off the figure."""
    title = _card_title(task_name, style)
    if not style.show_coordinates:
        display_label = style.display_label or f"target #{target.manifest_idx + 1}"
        ax.set_title(f"{title}   [{display_label}]", fontsize=12, loc="left")
        return
    line = _format_initial_state_target(target, display_label=style.display_label)
    ax.set_title(title + "\n" + textwrap.fill(line, width=wrap_width), fontsize=9)


REAL_NUT_PEG_TABLE_HALF_WIDTH_M = 0.4
REAL_PEG_HALF_WIDTH_M = 0.016
REAL_NUT_GEOMS_M = (
    # Copied from deps/robosuite/.../assets/objects/square-nut.xml.
    # Each tuple is (center_x, center_y, half_size_x, half_size_y), in meters.
    (-0.03325, 0.0, 0.0105, 0.04375),
    (0.0, 0.03325, 0.03125, 0.0105),
    (0.0, -0.03325, 0.03125, 0.0105),
    (0.03325, 0.0, 0.0105, 0.04375),
    (0.054, 0.0, 0.02525, 0.015875),
)
REAL_NUT_HANDLE_SITE_M = (0.054, 0.0)


def _task_display_name(task_name: str) -> str:
    return task_name.replace("_", " ").title()


def operator_card_pen_bounds_in(
    task_name: str, manifest_meta: dict
) -> tuple[float, float, float, float]:
    """Pen (x, y) bounds in INCHES for the operator card, preferring the REGISTRY.

    Returns ``(x_min, x_max, y_min, y_max)`` in inches (+x forward, +y left).

    Resolution order (single source of truth = the registry, no hardcoded default):

    1. If the task resolves to a :class:`RealTaskSpec`, use ``bounds_arr`` rows 0/1
       (the correct per-task extent, e.g. marker_d2's +/-3 in x / +/-6 in y). When the
       manifest meta also carries ``bounds`` they are cross-checked and a mismatch
       raises loudly (the validated manifest and the drawn rectangle can't disagree).
    2. Otherwise (a task that is not in the registry), use the bounds the manifest
       itself carries -- they are authoritative for an unregistered task. Fail loud if
       they are absent rather than guessing.

    """
    spec = find_task_spec_by_task_name(task_name)
    meta_bounds = manifest_meta.get("bounds")
    meta_is_dict = isinstance(meta_bounds, dict)

    if spec is not None:
        pen_bounds = spec.bounds_arr  # (3, 2): rows pen_x, pen_y, pen_yaw (meters)
        x_min_m, x_max_m = float(pen_bounds[0, 0]), float(pen_bounds[0, 1])
        y_min_m, y_max_m = float(pen_bounds[1, 0]), float(pen_bounds[1, 1])
        if meta_is_dict:
            for key, (lo_reg, hi_reg) in (
                (spec.state_keys[0], (x_min_m, x_max_m)),
                (spec.state_keys[1], (y_min_m, y_max_m)),
            ):
                if key not in meta_bounds:
                    continue
                lo_meta, hi_meta = (float(v) for v in meta_bounds[key])
                if abs(lo_meta - lo_reg) > 1e-6 or abs(hi_meta - hi_reg) > 1e-6:
                    raise ValueError(
                        f"operator card: manifest meta bounds for {key} "
                        f"[{lo_meta:.5f}, {hi_meta:.5f}] m disagree with the registry "
                        f"[{lo_reg:.5f}, {hi_reg:.5f}] m for {task_name!r}. The registry "
                        f"is the source of truth; regenerate the manifest from it."
                    )
        return (
            x_min_m * M_TO_INCH,
            x_max_m * M_TO_INCH,
            y_min_m * M_TO_INCH,
            y_max_m * M_TO_INCH,
        )

    # Unregistered task: the manifest's own bounds are the only authoritative source. Fail loud if they are missing -- never fall back
    # to a hardcoded (potentially wrong-task) default.
    pen_x_key, pen_y_key = "pen_x", "pen_y"
    if not (meta_is_dict and pen_x_key in meta_bounds and pen_y_key in meta_bounds):
        raise ValueError(
            f"operator card: task {task_name!r} has no registered task spec and the "
            f"manifest meta carries no {pen_x_key}/{pen_y_key} bounds, so the pen "
            f"randomization rectangle has no source. Register the task or carry bounds "
            f"in the manifest."
        )
    x_min_m, x_max_m = (float(v) for v in meta_bounds[pen_x_key])
    y_min_m, y_max_m = (float(v) for v in meta_bounds[pen_y_key])
    return (
        x_min_m * M_TO_INCH,
        x_max_m * M_TO_INCH,
        y_min_m * M_TO_INCH,
        y_max_m * M_TO_INCH,
    )


def _rotated_local_rect_corners_in(
    *,
    center_x_m: float,
    center_y_m: float,
    half_x_m: float,
    half_y_m: float,
    yaw: float,
    target_forward_in: float,
    target_left_in: float,
) -> list[tuple[float, float]]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    corners: list[tuple[float, float]] = []
    for local_x_m, local_y_m in [
        (center_x_m - half_x_m, center_y_m - half_y_m),
        (center_x_m + half_x_m, center_y_m - half_y_m),
        (center_x_m + half_x_m, center_y_m + half_y_m),
        (center_x_m - half_x_m, center_y_m + half_y_m),
    ]:
        world_forward_in = target_forward_in + (local_x_m * c - local_y_m * s) * M_TO_INCH
        world_left_in = target_left_in + (local_x_m * s + local_y_m * c) * M_TO_INCH
        corners.append((world_left_in, world_forward_in))
    return corners


def _local_point_to_plot_in(
    *,
    local_x_m: float,
    local_y_m: float,
    yaw: float,
    target_forward_in: float,
    target_left_in: float,
) -> tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    world_forward_in = target_forward_in + (local_x_m * c - local_y_m * s) * M_TO_INCH
    world_left_in = target_left_in + (local_x_m * s + local_y_m * c) * M_TO_INCH
    return world_left_in, world_forward_in


def _annotate_nut_peg_workspace_coordinates(
    ax,
    *,
    left_min: float,
    left_max: float,
    forward_min: float,
    forward_max: float,
) -> None:
    label_color = CARD_COLORS["pen_rect_edge"]
    tick_color = CARD_COLORS["pen_rect_edge"]
    tick_len = 0.16
    label_pad = 0.33
    x_ticks = np.arange(math.ceil(left_min), math.floor(left_max) + 1, 1.0)
    y_ticks = np.arange(math.ceil(forward_min), math.floor(forward_max) + 1, 1.0)

    for y_in in x_ticks:
        ax.plot(
            [y_in, y_in],
            [forward_min, forward_min - tick_len],
            color=tick_color,
            linewidth=0.8,
            zorder=7,
        )
        ax.plot(
            [y_in, y_in],
            [forward_max, forward_max + tick_len],
            color=tick_color,
            linewidth=0.8,
            zorder=7,
        )
        label = f"{y_in:g}"
        ax.text(
            y_in,
            forward_min - label_pad,
            label,
            ha="center",
            va="top",
            fontsize=6,
            color=label_color,
            zorder=8,
        )
        ax.text(
            y_in,
            forward_max + label_pad,
            label,
            ha="center",
            va="bottom",
            fontsize=6,
            color=label_color,
            zorder=8,
        )

    for x_in in y_ticks:
        ax.plot(
            [left_min, left_min - tick_len],
            [x_in, x_in],
            color=tick_color,
            linewidth=0.8,
            zorder=7,
        )
        ax.plot(
            [left_max, left_max + tick_len],
            [x_in, x_in],
            color=tick_color,
            linewidth=0.8,
            zorder=7,
        )
        label = f"{x_in:g}"
        ax.text(
            left_min - label_pad,
            x_in,
            label,
            ha="left",
            va="center",
            fontsize=6,
            color=label_color,
            zorder=8,
        )
        ax.text(
            left_max + label_pad,
            x_in,
            label,
            ha="right",
            va="center",
            fontsize=6,
            color=label_color,
            zorder=8,
        )

    ax.text(
        (left_min + left_max) / 2,
        forward_min - 0.78,
        "y in",
        ha="center",
        va="top",
        fontsize=6,
        color=label_color,
        zorder=8,
    )
    ax.text(
        (left_min + left_max) / 2,
        forward_max + 0.78,
        "y in",
        ha="center",
        va="bottom",
        fontsize=6,
        color=label_color,
        zorder=8,
    )
    ax.text(
        left_min - 0.78,
        (forward_min + forward_max) / 2,
        "x in",
        ha="left",
        va="center",
        fontsize=6,
        color=label_color,
        rotation=90,
        zorder=8,
    )
    ax.text(
        left_max + 0.78,
        (forward_min + forward_max) / 2,
        "x in",
        ha="right",
        va="center",
        fontsize=6,
        color=label_color,
        rotation=90,
        zorder=8,
    )


def _build_square_card(
    target: InitialStateTarget,
    manifest_meta: dict,
    *,
    task_name: str,
    style: CardStyle,
) -> Figure:
    """Nut-on-peg (square_d2) card: the nut at its pose plus the peg on its dot grid."""
    if target.peg_x is None or target.peg_y is None:
        raise ValueError(f"Nut target #{target.manifest_idx} is missing peg_x/peg_y")

    bounds = manifest_meta.get("bounds", {})
    x_min_m, x_max_m = [float(v) for v in bounds["nut_x"]]
    y_min_m, y_max_m = [float(v) for v in bounds["nut_y"]]
    forward_min = x_min_m * M_TO_INCH
    forward_max = x_max_m * M_TO_INCH
    left_min = y_min_m * M_TO_INCH
    left_max = y_max_m * M_TO_INCH
    target_forward = target.nut_x * M_TO_INCH
    target_left = target.nut_y * M_TO_INCH
    peg_forward = target.peg_x * M_TO_INCH
    peg_left = target.peg_y * M_TO_INCH
    table_half = REAL_NUT_PEG_TABLE_HALF_WIDTH_M * M_TO_INCH

    peg_extent_in = REAL_PEG_HALF_WIDTH_M * M_TO_INCH
    view_forward_min = -6.0
    view_forward_max = 13.5
    view_left_min = -10.0
    view_left_max = 10.0

    data_width = view_left_max - view_left_min
    data_height = view_forward_max - view_forward_min
    figure_width = 8.0
    figure_height = max(5.5, figure_width * data_height / data_width + 1.3)
    fig, ax = _new_card_figure(figure_width, figure_height)
    placement_points = placement_card_points(target, task_name)
    _set_card_title(ax, target, task_name=task_name, style=style)

    ax.set_xlim(view_left_max, view_left_min)
    ax.set_ylim(view_forward_min, view_forward_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(math.floor(view_left_min), math.ceil(view_left_max) + 1, 2.0))
    ax.set_yticks(np.arange(math.floor(view_forward_min), math.ceil(view_forward_max) + 1, 2.0))
    ax.set_xticks(
        np.arange(math.floor(view_left_min), math.ceil(view_left_max) + 1, 1.0),
        minor=True,
    )
    ax.set_yticks(
        np.arange(math.floor(view_forward_min), math.ceil(view_forward_max) + 1, 1.0),
        minor=True,
    )
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(True, which="major", color=CARD_COLORS["grid_major"], linewidth=0.8)
    ax.grid(True, which="minor", color=CARD_COLORS["grid_minor"], linewidth=0.6)
    ax.axhline(0.0, color=CARD_COLORS["axis"], linewidth=1.0)
    ax.axvline(0.0, color=CARD_COLORS["axis"], linewidth=1.0)

    ax.add_patch(
        Rectangle(
            (-table_half, -table_half),
            2 * table_half,
            2 * table_half,
            facecolor=CARD_COLORS["table_face"],
            edgecolor=CARD_COLORS["table_edge"],
            linewidth=1.5,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (left_min, forward_min),
            left_max - left_min,
            forward_max - forward_min,
            facecolor=CARD_COLORS["pen_rect_face"],
            edgecolor=CARD_COLORS["pen_rect_edge"],
            linewidth=2.0,
            alpha=0.35,
            zorder=1,
        )
    )
    _annotate_nut_peg_workspace_coordinates(
        ax,
        left_min=left_min,
        left_max=left_max,
        forward_min=forward_min,
        forward_max=forward_max,
    )

    # square_d2 samples the peg over a registry grid; draw the allowed snap centers as
    # faint dots (single source of truth = the registry) so the operator can place the peg
    # center on the physical marker dots. No-op for the earlier fixed-peg Nut setup. The
    # solid peg rectangle below remains the chosen per-row (peg_x, peg_y) footprint.
    peg_grid_color = CARD_COLORS["placement"]
    for _chosen_m, grid_m in placement_points:
        for grid_forward_m, grid_left_m in grid_m:
            ax.scatter(
                [grid_left_m * M_TO_INCH],
                [grid_forward_m * M_TO_INCH],
                marker="o",
                s=35,
                facecolor=peg_grid_color,
                edgecolors=peg_grid_color,
                linewidths=1.0,
                alpha=0.35,
                zorder=2,
            )

    peg_half = peg_extent_in
    ax.add_patch(
        Rectangle(
            (peg_left - peg_half, peg_forward - peg_half),
            2 * peg_half,
            2 * peg_half,
            facecolor=CARD_COLORS["peg_face"],
            edgecolor=CARD_COLORS["ink"],
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.text(
        peg_left,
        peg_forward + peg_half + 0.7,
        "square peg",
        ha="center",
        va="bottom",
        fontsize=8,
        color=CARD_COLORS["ink"],
    )
    for chosen_m, _grid_m in placement_points:
        if chosen_m is None:
            continue
        chosen_forward, chosen_left = chosen_m
        ax.scatter(
            [chosen_left * M_TO_INCH],
            [chosen_forward * M_TO_INCH],
            marker="o",
            s=85,
            facecolor=peg_grid_color,
            edgecolors="white",
            linewidths=1.2,
            alpha=1.0,
            zorder=6,
        )

    for center_x, center_y, half_x, half_y in REAL_NUT_GEOMS_M:
        ax.add_patch(
            Polygon(
                _rotated_local_rect_corners_in(
                    center_x_m=center_x,
                    center_y_m=center_y,
                    half_x_m=half_x,
                    half_y_m=half_y,
                    yaw=target.nut_yaw,
                    target_forward_in=target_forward,
                    target_left_in=target_left,
                ),
                closed=True,
                facecolor=CARD_COLORS["nut_face"],
                edgecolor=CARD_COLORS["nut_edge"],
                linewidth=1.2,
                zorder=4,
            )
        )
    handle_left, handle_forward = _local_point_to_plot_in(
        local_x_m=REAL_NUT_HANDLE_SITE_M[0],
        local_y_m=REAL_NUT_HANDLE_SITE_M[1],
        yaw=target.nut_yaw,
        target_forward_in=target_forward,
        target_left_in=target_left,
    )
    ax.plot(
        [target_left, handle_left],
        [target_forward, handle_forward],
        color=CARD_COLORS["target_arrow"],
        linewidth=2.0,
        zorder=5,
    )
    ax.scatter([target_left], [target_forward], s=45, color=CARD_COLORS["target_center"], zorder=6)
    ax.scatter([handle_left], [handle_forward], s=35, color=CARD_COLORS["target_arrow"], zorder=6)

    ax.set_xlabel("y (in, +left)")
    ax.set_ylabel("x (in, +forward)")
    return fig


def _build_routing_card(
    target: InitialStateTarget,
    manifest_meta: dict,
    *,
    task_name: str,
    style: CardStyle,
) -> Figure:
    """Operator card for the non-pen routing_d2 line: the rope (a full-width horizontal bar
    at the sampled ``rope_x``) plus the two clips, each drawn at its grid-snapped (x, y) with
    an arrow showing the chosen discrete orientation, over the faint allowed dot grids.

    Single source of truth: the clip dot grids come from the registry placement; the chosen
    rope/clip coords + angles come from the target row (the same values the text readout and
    the loader validation use), so the drawn scene and the text can never disagree."""
    spec = find_task_spec_by_task_name(task_name)
    if spec is None or set(spec.sampled_placements) != {"clip_left", "clip_right"}:
        raise ValueError(
            f"routing_d2 operator card needs the registered routing_d2 spec with clip_left + "
            f"clip_right placements; task {task_name!r} resolves to {spec!r}"
        )
    for key in (
        "rope_x",
        "clip_left_x",
        "clip_left_y",
        "clip_left_yaw",
        "clip_right_x",
        "clip_right_y",
        "clip_right_yaw",
    ):
        if target.raw.get(key) is None:
            raise ValueError(f"routing_d2 target #{target.manifest_idx} is missing {key}")

    rope_x_in = target.rope_x * M_TO_INCH
    (rope_lo_in, rope_hi_in) = (float(v) * M_TO_INCH for v in spec.bounds_arr[0])
    clips = {
        "clip_left": (
            CARD_COLORS["target_center"],
            target.clip_left_x,
            target.clip_left_y,
            target.clip_left_yaw,
        ),
        "clip_right": (
            CARD_COLORS["clip_right"],
            target.clip_right_x,
            target.clip_right_y,
            target.clip_right_yaw,
        ),
    }

    # View extent: full clip dot grids (operator frame, inches) plus a margin.
    grid_in = {
        name: [(gx * M_TO_INCH, gy * M_TO_INCH) for gx, gy in placement.grid_points()]
        for name, placement in spec.sampled_placements.items()
    }
    all_forward = [rope_lo_in, rope_hi_in, *(f for pts in grid_in.values() for f, _ in pts)]
    all_left = [left for pts in grid_in.values() for _, left in pts]
    plot_forward_min, plot_forward_max = min(all_forward) - 1.5, max(all_forward) + 1.5
    plot_left_min, plot_left_max = min(all_left) - 1.5, max(all_left) + 1.5

    data_width = plot_left_max - plot_left_min
    data_height = plot_forward_max - plot_forward_min
    figure_width = 8.0
    figure_height = max(5.0, figure_width * data_height / data_width + 1.4)
    fig, ax = _new_card_figure(figure_width, figure_height)
    _set_card_title(
        ax,
        target,
        task_name=task_name,
        style=style,
    )

    # Operator view: +x forward is up, +y left grows leftward (x-axis inverted).
    ax.set_xlim(plot_left_max, plot_left_min)
    ax.set_ylim(plot_forward_min, plot_forward_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(math.floor(plot_left_min), math.ceil(plot_left_max) + 1, 2.0))
    ax.set_yticks(np.arange(math.floor(plot_forward_min), math.ceil(plot_forward_max) + 1, 2.0))
    ax.set_xticks(
        np.arange(math.floor(plot_left_min), math.ceil(plot_left_max) + 1, 1.0), minor=True
    )
    ax.set_yticks(
        np.arange(math.floor(plot_forward_min), math.ceil(plot_forward_max) + 1, 1.0), minor=True
    )
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(True, which="major", color=CARD_COLORS["grid_major"], linewidth=0.8)
    ax.grid(True, which="minor", color=CARD_COLORS["grid_minor"], linewidth=0.5)
    ax.axhline(0.0, color=CARD_COLORS["axis"], linewidth=1.0)
    ax.axvline(0.0, color=CARD_COLORS["axis"], linewidth=1.0)

    # Faint allowed clip dot grids (left half y>0, right half y<0).
    for name, pts in grid_in.items():
        color = clips[name][0]
        ax.scatter(
            [left for _, left in pts],
            [fwd for fwd, _ in pts],
            marker="o",
            s=10,
            facecolors="none",
            edgecolors=color,
            linewidths=0.5,
            alpha=0.30,
            zorder=1,
        )

    # Rope: a full-width horizontal bar at rope_x (it spans the whole y-axis).
    rope_color = CARD_COLORS["rope"]
    rope_half_thick = 0.18
    ax.add_patch(
        Rectangle(
            (plot_left_min, rope_x_in - rope_half_thick),
            plot_left_max - plot_left_min,
            2 * rope_half_thick,
            facecolor=rope_color,
            edgecolor=CARD_COLORS["rope_edge"],
            linewidth=1.0,
            alpha=0.85,
            zorder=2,
        )
    )
    ax.text(
        plot_left_min + 0.3,
        rope_x_in + rope_half_thick + 0.25,
        f"rope  x={rope_x_in:+.1f} in",
        ha="right",
        va="bottom",
        fontsize=8,
        color=CARD_COLORS["rope_edge"],
        fontweight="bold",
    )

    # Clips: a center dot + an orientation arrow (clip long axis), per the chosen yaw.
    arm_half = 1.4 / 2.0
    for name, (color, x_m, y_m, yaw) in clips.items():
        cx_in = x_m * M_TO_INCH  # forward (vertical)
        cy_in = y_m * M_TO_INCH  # +left (horizontal)
        dx = arm_half * math.sin(yaw)  # along +left (horizontal)
        dy = arm_half * math.cos(yaw)  # along +forward (vertical)
        ax.add_patch(
            FancyArrowPatch(
                (cy_in - dx, cx_in - dy),
                (cy_in + dx, cx_in + dy),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.2,
                color=color,
                zorder=5,
            )
        )
        ax.scatter(
            [cy_in], [cx_in], s=55, color=color, edgecolors="white", linewidths=0.8, zorder=6
        )
        ax.annotate(
            f"{'L' if name == 'clip_left' else 'R'} {math.degrees(yaw):+.0f}°",
            (cy_in, cx_in),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    ax.set_xlabel("y (in, +left)")
    ax.set_ylabel("x (in, +forward)")
    return fig


def placement_card_points(
    target: InitialStateTarget, task_name: str
) -> list[tuple[tuple[float, float] | None, list[tuple[float, float]]]]:
    """For the operator card: per sampled placement, ``(chosen (x, y) from the row or
    None, [allowed grid (x, y) points from the REGISTRY])`` in operator-frame METERS.

    Single source of truth: the allowed grid comes from the registry (never the manifest
    meta), and the chosen point comes from the target row -- the same coords the text
    readout uses -- so the drawn bold point and the text readout can never disagree."""
    out: list[tuple[tuple[float, float] | None, list[tuple[float, float]]]] = []
    spec = find_task_spec_by_task_name(task_name)
    if spec is None or not spec.sampled_placements:
        # A holder-carrying target with no registry grid would draw NO holder marker
        # while _format_initial_state_target still appends a textual "| HOLDER (...)"
        # line -- a silent operator-misleading divergence. Mirror the manifest parser:
        # fail loudly when there are holder coords but no placements to draw them from.
        if target.holder_x is not None and target.holder_y is not None:
            raise ValueError(
                f"target #{target.manifest_idx} carries holder coords "
                f"(holder_x={target.holder_x}, holder_y={target.holder_y}) but task "
                f"{task_name!r} resolves to "
                f"{'no registered task spec' if spec is None else 'a spec with no sampled placements'}"
                f" -- the operator card would show a HOLDER text line but draw no holder "
                f"marker. Register the task (with its sampled placements) before rendering."
            )
        return out
    for placement in spec.sampled_placements.values():
        grid = [(float(gx), float(gy)) for gx, gy in placement.grid_points()]
        kx, ky = placement.keys
        chosen = None
        if kx in target.raw and ky in target.raw:
            chosen = (float(target.raw[kx]), float(target.raw[ky]))
        out.append((chosen, grid))
    return out


def _build_marker_card(
    target: InitialStateTarget,
    manifest_meta: dict,
    *,
    task_name: str,
    style: CardStyle,
) -> Figure:
    """Marker card (marker_d2): pen pose + optional holder dot grid."""
    forward_min, forward_max, left_min, left_max = operator_card_pen_bounds_in(
        task_name, manifest_meta
    )
    target_forward = target.pen_x * M_TO_INCH
    target_left = target.pen_y * M_TO_INCH

    # Sampled RED-holder grid (marker_d2): draw the allowed grid points from the
    # REGISTRY (single source of truth) + the chosen point from the row coords, so the
    # card never depends on manifest-meta nesting and the text + drawn point agree. The
    # holder sits OUTSIDE the pen rectangle, so the PLOT extent (not the rect) grows to
    # show it. Each entry is (chosen (forward, left) or None, [grid (forward, left)...]).
    placement_points_in: list[tuple[tuple[float, float] | None, list[tuple[float, float]]]] = []
    for chosen_m, grid_m in placement_card_points(target, task_name):
        grid_in = [(gx * M_TO_INCH, gy * M_TO_INCH) for gx, gy in grid_m]
        chosen_in = None if chosen_m is None else (chosen_m[0] * M_TO_INCH, chosen_m[1] * M_TO_INCH)
        placement_points_in.append((chosen_in, grid_in))
    holder_grid_in = [pt for _, grid in placement_points_in for pt in grid]

    plot_forward_min = min([forward_min, *(f for f, _ in holder_grid_in)])
    plot_forward_max = max([forward_max, *(f for f, _ in holder_grid_in)])
    plot_left_min = min([left_min, *(left for _, left in holder_grid_in)])
    plot_left_max = max([left_max, *(left for _, left in holder_grid_in)])
    pad_forward = max(0.01, 0.1 * (plot_forward_max - plot_forward_min))
    pad_left = max(0.01, 0.1 * (plot_left_max - plot_left_min))

    data_width = (plot_left_max - plot_left_min) + 2 * pad_left
    data_height = (plot_forward_max - plot_forward_min) + 2 * pad_forward
    figure_width = 7.0
    figure_height = max(4.2, figure_width * data_height / data_width + 1.2)
    fig, ax = _new_card_figure(figure_width, figure_height)
    _set_card_title(ax, target, task_name=task_name, style=style)
    # Operator view: +x forward is up, +y left is left.
    ax.set_xlim(plot_left_max + pad_left, plot_left_min - pad_left)
    ax.set_ylim(plot_forward_min - pad_forward, plot_forward_max + pad_forward)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(data_height / data_width)
    ax.set_xticks(np.arange(math.floor(plot_left_min), math.ceil(plot_left_max) + 1, 1.0))
    ax.set_yticks(np.arange(math.floor(plot_forward_min), math.ceil(plot_forward_max) + 1, 1.0))
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(True, which="major", color=CARD_COLORS["grid_major"], linewidth=0.8)
    ax.axhline(0.0, color=CARD_COLORS["axis"], linewidth=1.0)
    ax.axvline(0.0, color=CARD_COLORS["axis"], linewidth=1.0)
    ax.add_patch(
        Rectangle(
            (left_min, forward_min),
            left_max - left_min,
            forward_max - forward_min,
            fill=False,
            linewidth=2.0,
            color=CARD_COLORS["ink"],
        )
    )

    guide_radius = 1.15
    for guide_yaw, linewidth, alpha in [
        (0.0, 0.95, 0.85),
        (math.pi / 4, 0.65, 0.28),
        (math.pi / 2, 0.95, 0.85),
        (3 * math.pi / 4, 0.65, 0.28),
    ]:
        guide_dx = guide_radius * math.sin(guide_yaw)
        guide_dy = guide_radius * math.cos(guide_yaw)
        ax.plot(
            [target_left - guide_dx, target_left + guide_dx],
            [target_forward - guide_dy, target_forward + guide_dy],
            color=CARD_COLORS["guide"],
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )

    marker_len = 1.6
    half_marker = marker_len / 2
    marker_dx = half_marker * math.sin(target.pen_yaw)
    marker_dy = half_marker * math.cos(target.pen_yaw)
    back = (target_left - marker_dx, target_forward - marker_dy)
    front = (target_left + marker_dx, target_forward + marker_dy)
    ax.add_patch(
        FancyArrowPatch(
            back,
            front,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=3.0,
            color=CARD_COLORS["target_arrow"],
            zorder=4,
        )
    )
    ax.scatter([target_left], [target_forward], s=55, color=CARD_COLORS["target_center"], zorder=5)

    # RED holder grid (marker_d2): all allowed grid points faint, the CHOSEN point bold
    # (drawn directly from the row's holder coords), so the operator places the red
    # holder at the chosen (holder_x, holder_y).
    for chosen_in, grid_in in placement_points_in:
        for grid_forward, grid_left in grid_in:
            ax.scatter(
                [grid_left],
                [grid_forward],
                marker="s",
                s=70,
                facecolor="none",
                edgecolors=CARD_COLORS["placement"],
                linewidths=1.0,
                alpha=0.4,
                zorder=3,
            )
        if chosen_in is not None:
            ax.scatter(
                [chosen_in[1]],
                [chosen_in[0]],
                marker="s",
                s=210,
                facecolor=CARD_COLORS["placement"],
                edgecolors=CARD_COLORS["placement"],
                linewidths=2.4,
                alpha=1.0,
                zorder=6,
            )

    ax.set_xlabel("pen_y (in, +left)")
    ax.set_ylabel("pen_x (in, +forward)")
    return fig


def cleanup_unreferenced_initial_state_cards(
    visualization_dir: Path | None,
    ledger_path: Path | None,
) -> list[Path]:
    """Delete target cards that are not referenced by saved-episode ledger rows."""
    if visualization_dir is None or ledger_path is None:
        return []
    if not visualization_dir.exists() or not ledger_path.exists():
        return []

    referenced_names: set[str] = set()
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        visualization_path = row.get("initial_state_visualization_path")
        if visualization_path:
            referenced_names.add(Path(visualization_path).name)

    removed: list[Path] = []
    candidate_paths = {
        path
        for pattern in ("target_*.png", "start_*.png")
        for path in visualization_dir.glob(pattern)
    }
    for path in sorted(candidate_paths):
        if path.name in referenced_names:
            continue
        path.unlink()
        removed.append(path)
    return removed


def build_initial_state_card_figure(
    target: InitialStateTarget,
    manifest_meta: dict,
    *,
    task_name: str,
    style: CardStyle | None = None,
) -> Figure:
    """Render the operator card for ``target`` as a matplotlib Figure (caller closes it).

    Dispatches on the manifest's validated key set, the same single source the loader
    validated the rows against.
    """
    style = style or CardStyle()
    keys = _manifest_keys(manifest_meta)
    if keys == NUT_PEG_INITIAL_STATE_KEYS:
        fig = _build_square_card(target, manifest_meta, task_name=task_name, style=style)
    elif keys == ROUTING_D2_INITIAL_STATE_KEYS:
        fig = _build_routing_card(target, manifest_meta, task_name=task_name, style=style)
    else:
        fig = _build_marker_card(target, manifest_meta, task_name=task_name, style=style)
    # The operational panel fits the full figure onto a workstation screen. Keep
    # object labels readable at that size, not only in the full-resolution PNG.
    for label in fig.findobj(Text):
        label.set_fontsize(max(11, label.get_fontsize()))
    return fig


def write_initial_state_card(
    target: InitialStateTarget,
    manifest_meta: dict,
    *,
    output_dir: Path,
    task_name: str,
    style: CardStyle | None = None,
) -> Path:
    """Render the card to ``output_dir / card_filename(target)`` and return that path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / card_filename(target)
    fig = build_initial_state_card_figure(target, manifest_meta, task_name=task_name, style=style)
    fig.savefig(path, dpi=CARD_DPI, bbox_inches="tight", pad_inches=0.06)
    return path
