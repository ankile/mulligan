"""Fig. 4: THREAD NUT reset card + simplified initial-state ranges.

Three print-sized panels sharing one panel height (``PANEL_HEIGHT_IN``):

1. ``real_world_square_d2_reset_card.pdf`` -- a vector re-render of the
   square_d2 reset-helper card at its true print size (0.30 textwidth), with the
   same geometry as the appendix card (nut sampling box, peg placement grid,
   the chosen peg, the sampled nut pose) but annotated for print: object names,
   the nut range and peg grid in cm, the operator-frame axes, and a 10 cm scale
   bar instead of 21 unreadable inch ticks.

2./3. ``real_world_{square_d2,marker_d2}_init_ranges_side1.png`` -- the dense
   reset composites (``*_clean.png``, hash-pinned in
   ``paper/data/real/initial_state_ranges_composites.json``), re-annotated with ONE
   box per object, cm-only labels sized for print, and the paper crop baked in (no
   ``trim`` at include time). Projection uses the calibrated side_1 operator frame
   (``paper/data/real/side1_operator_frame_calibration.json``).

    python -m paper.fig_reset_ranges            # card + both overlays into FIGS_DIR
    python -m paper.fig_reset_ranges --only card
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from mulligan.plotting import paper  # noqa: E402
from mulligan.real.lifecycle.tasks import INCH_TO_M, get_task_spec  # noqa: E402
from mulligan.real.operator_ui.cards import (  # noqa: E402
    REAL_NUT_GEOMS_M,
    REAL_NUT_HANDLE_SITE_M,
    REAL_PEG_HALF_WIDTH_M,
)
from paper.appendix.artifacts import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper/data/real"

M_TO_INCH = 1.0 / INCH_TO_M
CM_PER_INCH = 2.54

# One shared panel height (inches at print). The tex sets
# \resetpanelheight = 0.2735\linewidth = 1.504in for all three panels.
PANEL_HEIGHT_IN = 0.2735 * paper.DOC_TEXTWIDTH_IN["main"]
CARD_WIDTH_FRAC = 0.30
CARD_NAME = "real_world_square_d2_reset_card"

# Photo-annotation ink (light blue = the sampled mover, amber = the grid-placed
# fixture), shared by the card and the photo overlays.
MOVER = "#4fc3f7"
PLACE = "#ffb300"
INK = "0.15"

# The card's example target: square_d2 R0 manifest row 1, the same row the
# appendix card shows.
CARD_TARGET = DATA / "square_d2_reset_card_target.json"
CALIBRATION = DATA / "side1_operator_frame_calibration.json"
COMPOSITES_LOCK = DATA / "initial_state_ranges_composites.json"

# Paper crop of the 640x480 side_1 frame, in source px: [128, 640] x [116.5, 480]
# with a 15 % right trim.
CROP_X0, CROP_X1 = 128.0, 563.0
CROP_Y0, CROP_Y1 = 116.5, 480.0
OVERLAY_HEIGHT_IN = PANEL_HEIGHT_IN
OVERLAY_WIDTH_IN = OVERLAY_HEIGHT_IN * (CROP_X1 - CROP_X0) / (CROP_Y1 - CROP_Y0)
# Author the raster overlays at 2x print size so fonts/lines are specified in
# print points (x2) and the PNG carries enough pixels (dpi 300 -> ~1080 px wide).
OVERLAY_AUTHOR_SCALE = 2.0
OVERLAY_DPI = 300


def _cm(inches: float) -> str:
    """Whole centimetres: the figure is read at a glance, not measured."""
    return f"{inches * CM_PER_INCH:.0f}"


# ---------------------------------------------------------------------------
# Panel 1: the reset card (vector, print scale)
# ---------------------------------------------------------------------------


def _card_target() -> dict:
    payload = json.loads(CARD_TARGET.read_text())
    if payload["task"] != "square_d2":
        raise ValueError(f"{CARD_TARGET}: task is {payload['task']!r}, expected 'square_d2'")
    return payload["state"]


def _rotated_rect(cx_m, cy_m, hx_m, hy_m, yaw, fwd_in, left_in) -> list[tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for lx, ly in [
        (cx_m - hx_m, cy_m - hy_m),
        (cx_m + hx_m, cy_m - hy_m),
        (cx_m + hx_m, cy_m + hy_m),
        (cx_m - hx_m, cy_m + hy_m),
    ]:
        fwd = fwd_in + (lx * c - ly * s) * M_TO_INCH
        left = left_in + (lx * s + ly * c) * M_TO_INCH
        out.append((left, fwd))
    return out


def build_card_record() -> paper.FigureRecord:
    """Render the reset card to ``FIGS_DIR/real_world_square_d2_reset_card.pdf``."""
    spec = get_task_spec("square_d2")
    (fwd_min, fwd_max), (left_min, left_max) = [
        (v[0] * M_TO_INCH, v[1] * M_TO_INCH) for v in spec.bounds_arr[:2]
    ]
    placement = spec.sampled_placements["peg"]
    peg_grid = [(x * M_TO_INCH, y * M_TO_INCH) for x, y in placement.grid_points()]
    (peg_fwd_min, peg_fwd_max), (peg_left_min, peg_left_max) = [
        (v[0] * M_TO_INCH, v[1] * M_TO_INCH) for v in placement.bounds
    ]
    row = _card_target()
    nut_fwd, nut_left, nut_yaw = row["nut_x"] * M_TO_INCH, row["nut_y"] * M_TO_INCH, row["nut_yaw"]
    peg_fwd, peg_left = row["peg_x"] * M_TO_INCH, row["peg_y"] * M_TO_INCH
    peg_half = REAL_PEG_HALF_WIDTH_M * M_TO_INCH

    # View (operator frame, inches): +y left is on the plot's LEFT (xlim reversed),
    # +x forward is up -- the operator's own view of the table.
    view_left_max, view_left_min = 11.3, -11.3
    width_in, height_in = paper.fig_size(CARD_WIDTH_FRAC, height_in=PANEL_HEIGHT_IN)
    span_fwd = (view_left_max - view_left_min) * height_in / width_in
    view_fwd_min = -6.2
    view_fwd_max = view_fwd_min + span_fwd

    mover_edge, mover_text = paper.darken(MOVER, 0.62), paper.darken(MOVER, 0.55)
    place_edge, place_text = paper.darken(PLACE, 0.80), paper.darken(PLACE, 0.62)
    label_fs, name_fs = 6.0, 6.0

    with paper.paper_rc():
        fig = plt.figure(figsize=(width_in, height_in))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(view_left_max, view_left_min)
        ax.set_ylim(view_fwd_min, view_fwd_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("0.975")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("0.45")
            sp.set_linewidth(0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        # 1-inch breadboard lattice (the physical hole grid the operator aligns to).
        for v in range(math.ceil(view_left_min), math.floor(view_left_max) + 1):
            ax.axvline(v, color="0.86", linewidth=0.3, zorder=0)
        for v in range(math.ceil(view_fwd_min), math.floor(view_fwd_max) + 1):
            ax.axhline(v, color="0.86", linewidth=0.3, zorder=0)

        # Nut sampling box.
        ax.add_patch(
            Rectangle(
                (left_min, fwd_min),
                left_max - left_min,
                fwd_max - fwd_min,
                facecolor=MOVER,
                alpha=0.16,
                edgecolor="none",
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (left_min, fwd_min),
                left_max - left_min,
                fwd_max - fwd_min,
                facecolor="none",
                edgecolor=mover_edge,
                linewidth=1.0,
                zorder=2,
            )
        )
        ax.text(
            left_max - 0.5,
            fwd_min + 0.45,
            f"Nut center range\n{_cm(left_max - left_min)} × {_cm(fwd_max - fwd_min)} cm, yaw 360°",
            ha="left",
            va="bottom",
            fontsize=label_fs,
            color=mover_text,
            linespacing=1.15,
            zorder=7,
        )

        # Peg placement grid (box through the outer grid points + the 15 dots).
        ax.add_patch(
            Rectangle(
                (peg_left_min, peg_fwd_min),
                peg_left_max - peg_left_min,
                peg_fwd_max - peg_fwd_min,
                facecolor=PLACE,
                alpha=0.18,
                edgecolor=place_edge,
                linewidth=1.0,
                zorder=1,
            )
        )
        ax.scatter(
            [y for _, y in peg_grid],
            [x for x, _ in peg_grid],
            s=5,
            color=place_edge,
            linewidths=0,
            zorder=3,
        )
        ax.annotate(
            f"Peg center grid\n{_cm(peg_left_max - peg_left_min)} × {_cm(peg_fwd_max - peg_fwd_min)} cm",
            xy=(peg_left_max, peg_fwd_max - 0.4),
            xytext=(view_left_max - 0.5, view_fwd_max - 0.45),
            ha="left",
            va="top",
            fontsize=label_fs,
            color=place_text,
            linespacing=1.15,
            zorder=7,
            arrowprops=dict(
                arrowstyle="-", color=place_edge, linewidth=0.5, shrinkA=1.5, shrinkB=0
            ),
        )
        # Chosen peg.
        ax.add_patch(
            Rectangle(
                (peg_left - peg_half, peg_fwd - peg_half),
                2 * peg_half,
                2 * peg_half,
                facecolor="0.55",
                edgecolor=INK,
                linewidth=0.6,
                zorder=4,
            )
        )
        ax.scatter(
            [peg_left],
            [peg_fwd],
            s=9,
            color=place_edge,
            edgecolors="white",
            linewidths=0.4,
            zorder=6,
        )

        # Sampled nut pose (the four bars of the square nut + its handle site).
        for cx, cy, hx, hy in REAL_NUT_GEOMS_M:
            ax.add_patch(
                Polygon(
                    _rotated_rect(cx, cy, hx, hy, nut_yaw, nut_fwd, nut_left),
                    closed=True,
                    facecolor="tan",
                    edgecolor=INK,
                    linewidth=0.5,
                    zorder=5,
                )
            )
        c, s = math.cos(nut_yaw), math.sin(nut_yaw)
        hx_m, hy_m = REAL_NUT_HANDLE_SITE_M
        handle_left = nut_left + (hx_m * s + hy_m * c) * M_TO_INCH
        handle_fwd = nut_fwd + (hx_m * c - hy_m * s) * M_TO_INCH
        ax.plot([nut_left, handle_left], [nut_fwd, handle_fwd], color=INK, linewidth=0.8, zorder=6)
        ax.scatter([nut_left], [nut_fwd], s=7, color=INK, linewidths=0, zorder=6)
        ax.scatter([handle_left], [handle_fwd], s=5, color=INK, linewidths=0, zorder=6)

        # Object callouts.
        arrow_kw = dict(arrowstyle="-", color=INK, linewidth=0.5, shrinkA=0, shrinkB=1.5)
        ax.annotate(
            "sampled nut pose",
            xy=(nut_left + 1.5, nut_fwd + 1.5),
            xytext=(9.0, 7.4),
            ha="left",
            va="center",
            fontsize=name_fs,
            color=INK,
            arrowprops=arrow_kw,
            zorder=8,
        )
        ax.annotate(
            "peg",
            xy=(peg_left - peg_half, peg_fwd),
            xytext=(-7.4, peg_fwd),
            ha="left",
            va="center",
            fontsize=name_fs,
            color=INK,
            arrowprops=arrow_kw,
            zorder=8,
        )

        # Operator frame axes at the table origin (matches the photo overlays).
        for (dl, df), name, (tl, tf), ha, va in [
            ((0.0, 2.4), "+x", (0.4, 2.0), "right", "center"),
            ((2.4, 0.0), "+y", (2.7, 0.0), "right", "center"),
        ]:
            ax.add_patch(
                FancyArrowPatch(
                    (0, 0),
                    (dl, df),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    linewidth=0.8,
                    color=INK,
                    zorder=7,
                    shrinkA=0,
                    shrinkB=0,
                )
            )
            ax.text(tl, tf, name, ha=ha, va=va, fontsize=name_fs, color=INK, zorder=8)
        ax.scatter([0], [0], s=6, color=INK, linewidths=0, zorder=8)

        # 10 cm scale bar (top-right of the card, an empty region of the table).
        bar_len = 10.0 / CM_PER_INCH
        bar_left0, bar_fwd = -6.9, view_fwd_max - 1.1
        ax.plot(
            [bar_left0, bar_left0 - bar_len],
            [bar_fwd, bar_fwd],
            color=INK,
            linewidth=1.2,
            solid_capstyle="butt",
            zorder=7,
        )
        for end in (bar_left0, bar_left0 - bar_len):
            ax.plot(
                [end, end], [bar_fwd - 0.25, bar_fwd + 0.25], color=INK, linewidth=0.8, zorder=7
            )
        ax.text(
            bar_left0 - bar_len / 2,
            bar_fwd - 0.4,
            "10 cm",
            ha="center",
            va="top",
            fontsize=name_fs,
            color=INK,
            zorder=8,
        )

        return paper.save_paper_figure(
            fig,
            CARD_NAME,
            width_frac=CARD_WIDTH_FRAC,
            sources=(
                str(CARD_TARGET.relative_to(ROOT)),
                "mulligan/real/lifecycle/tasks.py",
                "mulligan/real/operator_ui/cards.py",
            ),
        )


# ---------------------------------------------------------------------------
# Panels 2-3: simplified range overlays on the dense reset composites
# ---------------------------------------------------------------------------

# Full-object ENVELOPES (x0, x1, y0, y1), operator-frame inches: the centre
# sampling box grown by each object's reach from its reference point (nut corner
# 3.2 in, pen 2.4 in, peg base plate 1.3 in, holder 2 in along y). The card
# (panel 1) shows the centre range; the photos show where object pixels can
# actually appear.
OVERLAY_SPECS = {
    "square_d2": dict(
        mover=dict(box=(-5.0 - 3.2, 5.0 + 3.2, -9.0 - 3.2, 9.0 + 3.2), name="Nut envelope"),
        place=dict(box=(9.5 - 1.3, 11.5 + 1.3, -4.5 - 1.3, -0.5 + 1.3), name="Peg envelope"),
    ),
    "marker_d2": dict(
        mover=dict(box=(-3.0 - 2.4, 3.0 + 2.4, -6.0 - 2.4, 6.0 + 2.4), name="Pen envelope"),
        # holder_x is the block's -x FACE: the block occupies [holder_x, holder_x + 1.2 in].
        place=dict(box=(6.0, 8.0 + 1.2, -4.0 - 2.0, 0.0 + 2.0), name="Holder envelope"),
    ),
}


class OpFrame:
    """Operator-frame inches -> side_1 640x480 px via the calibrated camera pose."""

    def __init__(self, line: str):
        calib = json.loads(CALIBRATION.read_text())
        intr = calib["intrinsics_640x480"]
        self.K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
        c = calib["lines"][line]
        self.pose = np.array(c["pose_rotvec"] + c["pose_translation_in"])
        self.swap, self.sx, self.sy = c["swap"], c["sx"], c["sy"]
        self.t = np.asarray(c["t_in"], dtype=float)

    def to_px(self, op) -> np.ndarray:
        op = np.atleast_2d(np.asarray(op, dtype=float))
        lx = (op[:, 0] - self.t[0]) / self.sx
        ly = (op[:, 1] - self.t[1]) / self.sy
        ij = np.stack([ly, lx], axis=-1) if self.swap else np.stack([lx, ly], axis=-1)
        Rm = Rotation.from_rotvec(self.pose[:3]).as_matrix()
        P = np.concatenate([ij, np.zeros((len(ij), 1))], axis=-1)
        uv = (P @ Rm.T + self.pose[3:]) @ self.K.T
        return uv[:, :2] / uv[:, 2:3]


def _composite(task: str) -> Path:
    """The pinned dense reset composite of ``task`` (640x480 side_1 PNG)."""
    lock = json.loads(COMPOSITES_LOCK.read_text())
    rel = f"assets/reset_ranges/{task}_clean.png"
    entry = [e for e in lock["files"] if e["path"] == rel]
    if len(entry) != 1:
        raise RuntimeError(f"{COMPOSITES_LOCK}: expected one {rel} entry, got {len(entry)}")
    return fetch("reset_ranges", entry[0])


def _stroke(artist, lw: float) -> None:
    artist.set_path_effects([patheffects.withStroke(linewidth=lw, foreground="black")])


def _overlay_text(ax, xy, text, color, *, fontsize, ha, va, dy=0.0):
    t = ax.text(
        xy[0],
        xy[1] + dy,
        text,
        color=color,
        fontsize=fontsize,
        ha=ha,
        va=va,
        fontweight="bold",
        linespacing=1.12,
        zorder=10,
    )
    _stroke(t, 0.28 * fontsize)
    return t


def _project_box(f: OpFrame, box):
    x0, x1, y0, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    pts = []
    for a, b in zip(corners[:-1], corners[1:]):
        pts.append(f.to_px(np.linspace(a, b, 40)))
    return np.concatenate(pts)


def _box_label(box, name, yaw=False) -> str:
    x0, x1, y0, y1 = box
    size = f"{_cm(y1 - y0)} × {_cm(x1 - x0)} cm"
    return f"{name}\n{size}, yaw 360°" if yaw else f"{name}\n{size}"


def render_overlay(task: str, out_path: Path) -> Path:
    src = _composite(task)
    img = np.asarray(Image.open(src).convert("RGB"))
    if img.shape[:2] != (480, 640):
        raise ValueError(f"{src}: expected 640x480, got {img.shape}")
    f = OpFrame(task)
    spec = OVERLAY_SPECS[task]
    k = OVERLAY_AUTHOR_SCALE
    label_fs, axis_fs = 7.0 * k, 6.0 * k
    lw = 1.1 * k

    fig = plt.figure(figsize=(OVERLAY_WIDTH_IN * k, OVERLAY_HEIGHT_IN * k))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(img, interpolation="lanczos")
    ax.set_xlim(CROP_X0, CROP_X1)
    ax.set_ylim(CROP_Y1, CROP_Y0)
    ax.axis("off")

    for key, color in (("mover", MOVER), ("place", PLACE)):
        box = spec[key]["box"]
        px = _project_box(f, box)
        line = ax.plot(px[:, 0], px[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=5)[0]
        _stroke(line, lw + 1.4 * k)
    # Mover label: bottom-left corner of the crop (open tabletop in both side_1
    # views, below the pile), left-aligned lines.
    _overlay_text(
        ax,
        (CROP_X0 + 0.03 * (CROP_X1 - CROP_X0), CROP_Y1 - 0.04 * (CROP_Y1 - CROP_Y0)),
        _box_label(spec["mover"]["box"], spec["mover"]["name"], yaw=True),
        MOVER,
        fontsize=label_fs,
        ha="left",
        va="bottom",
    )
    x0, x1, y0, y1 = spec["place"]["box"]
    # Fixture label: pinned to the top-left corner (far wall in both side_1
    # views) with a thin leader to the grid box's near-left corner, so it never
    # collides with the standing fixture, the mover pile, or the axes.
    corner = f.to_px([(x0, y1)])[0]
    ann = ax.annotate(
        _box_label(spec["place"]["box"], spec["place"]["name"]),
        xy=corner,
        xytext=(0.03, 0.96),
        textcoords="axes fraction",
        ha="left",
        va="top",
        color=PLACE,
        fontsize=label_fs,
        fontweight="bold",
        linespacing=1.12,
        zorder=10,
        arrowprops=dict(arrowstyle="-", color=PLACE, lw=0.8 * k, shrinkA=2, shrinkB=1),
    )
    _stroke(ann, 0.28 * label_fs)
    _stroke(ann.arrow_patch, 0.8 * k + 1.2 * k)

    # Operator-frame axes at the table origin; labels sit just beyond each tip
    # along the projected arrow direction.
    o = f.to_px([(0.0, 0.0)])[0]
    for vec, name in [((3.0, 0.0), "+x"), ((0.0, 3.0), "+y")]:
        tip = f.to_px([vec])[0]
        arr = ax.annotate(
            "",
            xy=tip,
            xytext=o,
            arrowprops=dict(
                arrowstyle="-|>",
                color="white",
                lw=0.9 * k,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=6 * k,
            ),
        )
        arr.arrow_patch.set_path_effects(
            [patheffects.withStroke(linewidth=0.9 * k + 1.2 * k, foreground="black")]
        )
        direction = (tip - o) / np.linalg.norm(tip - o)
        _overlay_text(
            ax, tip + 11.0 * direction, name, "white", fontsize=axis_fs, ha="center", va="center"
        )
    dot = ax.plot(o[0], o[1], "o", color="white", ms=2.0 * k, zorder=9)[0]
    _stroke(dot, 1.5 * k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=OVERLAY_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def build_range_overlays() -> list[Path]:
    """Render ``FIGS_DIR/real_world_{square_d2,marker_d2}_init_ranges_side1.png``."""
    return [
        render_overlay(task, paper.FIGS_DIR / f"real_world_{task}_init_ranges_side1.png")
        for task in OVERLAY_SPECS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("card", "overlays"), default=None)
    args = parser.parse_args()
    if args.only in (None, "overlays"):
        build_range_overlays()
    if args.only in (None, "card"):
        build_card_record()


if __name__ == "__main__":
    main()
