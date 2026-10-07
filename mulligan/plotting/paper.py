"""Paper-figure style contract: geometry, rc, vocabularies, and the save sink.

Single source of truth for every generated figure of the paper. The core guarantee is **print scale**:
a figure is authored at exactly ``textwidth * width_frac`` inches wide (the
same ``width_frac`` its ``\\includegraphics`` declares), so the printed type
sizes are the ones set here — never a rescale artifact. ``save_paper_figure``
asserts this and refuses to write a mis-sized figure.

Callers build under :func:`paper_rc` (a context manager — other figures
rendered in the same process keep their own style) and save through
:func:`save_paper_figure`. The driver that regenerates the full set is
``python -m paper.figures`` (see ``docs/paper_figures.md``).

Whisker policy (name the interval family once in the CAPTION, not in ylabels):

===============================  ============================================
Quantity                         Interval
===============================  ============================================
proportions (success rates,      Wilson 95% CI
episode shares of a count)
unit means (durations, stage     +/- 1 SE
scores)
throughput / ratio-of-sums       bootstrap 95% CI
5-seed sim means                 Student-t 95% CI (df=4) + individual seed dots
===============================  ============================================
"""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker

from mulligan.sim.task_names import SIM_TASK_DISPLAY_NAMES

from mulligan.plotting.colors import METHOD_COLORS

REPO_ROOT = Path(__file__).resolve().parents[2]
_PANEL_TITLE_GID = "paper-panel-title"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# \textwidth of the target document, in inches (corl_2026.sty geometry). A venue
# switch is a one-line change here, not a sweep over figure scripts.
DOC_TEXTWIDTH_IN: dict[str, float] = {"main": 5.5}


def fig_size(width_frac: float, *, height_in: float, doc: str = "main") -> tuple[float, float]:
    """Figure size for a paper figure occupying ``width_frac`` of ``doc``'s
    text width. Declare the SAME fraction in the ``\\includegraphics`` width so
    print scale is exactly 1.0."""
    return (DOC_TEXTWIDTH_IN[doc] * width_frac, height_in)


# ---------------------------------------------------------------------------
# rc contract
# ---------------------------------------------------------------------------
# Font: DejaVu Sans (matplotlib default) — deliberate, do not switch to a serif
# to "match" the body text; figures read as figures.
PAPER_RC: dict[str, object] = {
    # TrueType (Type 42) fonts: searchable/copyable text, no venue Type-3 flags.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Type sizes AT PRINT SCALE (pt). width_frac=1.0 figures print exactly these.
    "font.size": 7.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "errorbar.capsize": 2.0,
    # NEVER "tight": bbox-tight rewrites the page size and breaks the
    # exact-width guarantee that save_paper_figure asserts.
    "savefig.bbox": None,
}


@contextmanager
def paper_rc():
    """Apply :data:`PAPER_RC` for the duration of the block, then restore the
    FULL previous rcParams — including keys mutated inside the block (the
    line-headline plots set ``svg.hashsalt`` at several sites). Never mutate
    global rcParams for paper output: the figure driver renders every figure
    in one process."""
    with matplotlib.rc_context(PAPER_RC):
        yield


# ---------------------------------------------------------------------------
# Semantic color aliases — one key per ROLE, paper-wide
# ---------------------------------------------------------------------------
# Teal identifies the Mulligan actor or collection campaign; purple identifies
# HiL-IDQL+Mulligan reranking; blue is Sobol;
# gray is the naive baseline. Paper figures use these aliases rather than
# METHOD_COLORS keys, so each arm has one colour across figures.
BASELINE: str = METHOD_COLORS["uniform"]
# The sim-only HiL-IDQL arm: a darker step of the baseline gray, so hue
# still means "uniform collection" and the darker shade (plus the "P" marker,
# the paper-wide +BoN marker) means the reranked deployment.
BASELINE_RERANK: str = METHOD_COLORS["uniform_dark"]
OURS: str = METHOD_COLORS["real_mulligan_with_cf"]
OURS_RERANK: str = METHOD_COLORS["real_iql_rerank"]
SOBOL: str = METHOD_COLORS["sobol"]
INK: str = METHOD_COLORS["gray_neutral"]
ERRORBAR_INK: str = INK
NEUTRAL_FILL: str = "0.85"  # legend swatches / de-emphasized fills


def darken(color: str, factor: float = 0.72) -> str:
    """Darken ``color`` by multiplying RGB by ``factor`` (hex out)."""
    r, g, b = to_rgb(color)
    return to_hex((r * factor, g * factor, b * factor))


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
# Task titles: verb form, Title Case, everywhere (panel titles, bar groups, row
# labels). Title Case is load-bearing: the small-caps pass below keeps capitals
# at full size, so these render as the paper's \textsc{Insert Marker} does.
TASK_TITLES: dict[str, str] = {
    "marker_d2": "Insert Marker",
    "square_d2": "Thread Nut",
    "routing_d2": "Route Cable",
}

# Sim task titles use descriptive reset-distribution names. The robosuite
# environment IDs are NutAssemblySquare (Square-Narrow) and Square_D1 (Square-Broad).
SIM_TASK_TITLES: dict[str, str] = dict(SIM_TASK_DISPLAY_NAMES)

# Deployed-policy register (headline / efficiency / bars): the arm IS the
# deployed policy. Variant detail (with-CF, rerank N, critic id) lives in
# captions, not labels. Keys are the line-config arm names.
POLICY_ARM_LABELS: dict[str, str] = {
    "baseline": "HG-DAgger",
    "baseline_bon": "HiL-IDQL",
    "mulligan_with_cf": "HG-DAgger+Mulligan",
    "final_iql": "HiL-IDQL+Mulligan",
    "final_marker_iql": "HiL-IDQL+Mulligan",
}

# CF-ablation register: both arms use HG-DAgger, split by the
# counterfactual-demo treatment, so the label names the treatment.
CF_ARM_LABELS: dict[str, str] = {
    "mulligan_with_cf": POLICY_ARM_LABELS["mulligan_with_cf"],
    "mulligan_no_cf": "HG-DAgger+Mulligan (no CF)",
}

# Collection register (burden / collection-success figures describe the
# collection campaigns, including their early actor-only rounds).
COLLECTION_ARM_LABELS: dict[str, str] = {
    "baseline": POLICY_ARM_LABELS["baseline"],
    "mulligan": POLICY_ARM_LABELS["final_iql"],
}

# Sim sampler register (Square-Narrow R1 ablations).
SIM_SAMPLER_LABELS: dict[str, str] = {
    "baseline_uniform": "Uniform",
    "sobol": "Sobol",
    "mulligan": "Guided (no CF)",
    "mulligan_with_cf": "Mulligan (guided + CF)",
}

# THE round-axis label, real and sim alike: every DAgger round collects data
# (R0 = the base demonstrations), so "Collection round" is accurate on eval,
# collection-success, and burden surfaces at once.
ROUND_XLABEL = "Collection round"

# Legend key for the horizontal R0 carry-forward rules. The rules themselves
# retain each lineage's color; this neutral handle names their shared meaning.
STATIC_BC_LABEL = "Static BC (R0; no further data)"
STATIC_BC_LINESTYLE = (0, (4, 3))


# ---------------------------------------------------------------------------
# Task-name typography: faux small caps
# ---------------------------------------------------------------------------
# The paper sets every task name in small caps (main.tex routes \marker, \nut,
# \cable, \sqnarrow, \sqbroad through \textsc). DejaVu Sans ships no small-caps
# variant and matplotlib synthesizes none, so a name is composed from two runs:
# characters that are already capital keep the label's own size, the rest are
# uppercased at SMALL_CAPS_RATIO of it. (Unicode's small-capital block is not
# an option: it has no small capital Q, which the Square task names need.)
#
# The pass runs in save_paper_figure over the finished figure, matching the
# vocabularies above. That placement is deliberate: several paper figures are
# drawn by shared plotting code (paper.real_headline), so styling at the paper
# sink covers every paper figure from the vocabulary alone. It runs after
# layout and only swaps a Text for an equivalently anchored offset box, so no
# panel geometry moves.
SMALL_CAPS_RATIO = 0.80

TASK_NAMES: tuple[str, ...] = tuple(TASK_TITLES.values()) + tuple(SIM_TASK_TITLES.values())
# Longest-first so a name that prefixes another cannot win the shorter match.
_TASK_NAME_RE = re.compile(
    "|".join(re.escape(name) for name in sorted(TASK_NAMES, key=len, reverse=True))
)
_HA_ALIGN = {"left": 0.0, "center": 0.5, "right": 1.0}
_VA_ALIGN = {"bottom": 0.0, "baseline": 0.0, "center": 0.5, "top": 1.0}


def _small_caps_segments(text: str) -> list[tuple[str, bool]]:
    """Split ``text`` into ``(chunk, shrink)`` segments: characters inside a
    vocabulary task name that must shrink to small caps are uppercased and
    flagged; every other character is carried verbatim at full size."""
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for match in _TASK_NAME_RE.finditer(text):
        if match.start() > cursor:
            segments.append((text[cursor : match.start()], False))
        runs: list[list] = []
        for char in match.group():
            shrink = char.islower()
            if runs and runs[-1][1] == shrink:
                runs[-1][0] += char
            else:
                runs.append([char, shrink])
        segments.extend((chunk.upper() if shrink else chunk, shrink) for chunk, shrink in runs)
        cursor = match.end()
    if cursor < len(text):
        segments.append((text[cursor:], False))
    return segments


def _small_caps_pack(
    text: str, *, size: float, color: str, weight: str, rotation: float
) -> HPacker | VPacker:
    """Offset box drawing ``text`` with its task names in faux small caps."""
    if "\n" in text:
        raise ValueError(f"cannot set a wrapped label in small caps: {text!r}")
    boxes = [
        TextArea(
            chunk,
            textprops=dict(
                size=size * (SMALL_CAPS_RATIO if shrink else 1.0),
                color=color,
                weight=weight,
                rotation=rotation,
                rotation_mode="anchor",
            ),
        )
        for chunk, shrink in _small_caps_segments(text)
        if chunk
    ]
    # A hair of tracking between runs, as a real small-caps face carries.
    if rotation == 0:
        return HPacker(children=boxes, align="baseline", pad=0.0, sep=0.4)
    if rotation == 90:
        # Rotated text reads bottom-to-top, so the first run sits lowest. A
        # rotated TextArea reserves the glyph descent along the stacking axis,
        # which would open a gap mid-word; pull the runs back by that much
        # (measured against the label's own size) to restore even tracking.
        return VPacker(children=list(reversed(boxes)), align="right", pad=0.0, sep=-0.2 * size)
    raise ValueError(f"small-caps labels support rotation 0 or 90, got {rotation}")


def small_caps_text(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    size: float,
    color: str = "black",
    weight: str = "normal",
    rotation: float = 0.0,
    xycoords: str = "axes fraction",
    ha: str = "center",
    va: str = "center",
) -> None:
    """Place ``text`` at ``xy`` with its task names in faux small caps.

    The sink pass handles titles and tick labels on its own; call this directly
    for free-standing labels (e.g. the rotated per-task row labels of the
    burden grid), which it refuses to restyle blind.
    """
    ax.add_artist(
        AnnotationBbox(
            _small_caps_pack(text, size=size, color=color, weight=weight, rotation=rotation),
            xy,
            xycoords=xycoords,
            frameon=False,
            pad=0.0,
            box_alignment=(_HA_ALIGN[ha], _VA_ALIGN[va]),
            annotation_clip=False,
        )
    )


def _restyle_title(ax: plt.Axes) -> int:
    title = ax.title
    if not _TASK_NAME_RE.search(title.get_text()):
        return 0
    replacement = AnnotationBbox(
        _small_caps_pack(
            title.get_text(),
            size=title.get_fontsize(),
            color=title.get_color(),
            weight=title.get_fontweight(),
            rotation=0.0,
        ),
        # The title's own transform already carries the title pad.
        title.get_position(),
        xycoords=title.get_transform(),
        frameon=False,
        pad=0.0,
        box_alignment=(_HA_ALIGN[title.get_ha()], _VA_ALIGN[title.get_va()]),
        annotation_clip=False,
    )
    replacement.set_gid(_PANEL_TITLE_GID)
    ax.add_artist(replacement)
    title.set_text("")
    return 1


def _restyle_ticklabels(ax: plt.Axes) -> int:
    restyled = 0
    for axis in (ax.xaxis, ax.yaxis):
        labels = list(axis.get_ticklabels())
        texts = [label.get_text() for label in labels]
        if not any(_TASK_NAME_RE.search(text) for text in texts):
            continue
        for label, text in zip(labels, texts):
            if not _TASK_NAME_RE.search(text):
                continue
            ax.add_artist(
                AnnotationBbox(
                    _small_caps_pack(
                        text,
                        size=label.get_fontsize(),
                        color=label.get_color(),
                        weight=label.get_fontweight(),
                        rotation=label.get_rotation(),
                    ),
                    label.get_position(),
                    xycoords=label.get_transform(),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(_HA_ALIGN[label.get_ha()], _VA_ALIGN[label.get_va()]),
                    annotation_clip=False,
                )
            )
            restyled += 1
        # Tick text comes from the formatter at draw time, so blanking the Text
        # artist would not stick: reinstall the list with the matched entries
        # emptied (these axes carry an explicit FixedFormatter already).
        axis.set_ticklabels(["" if _TASK_NAME_RE.search(t) else t for t in texts])
    return restyled


def stylize_task_names(fig: plt.Figure) -> int:
    """Redraw the task names in ``fig``'s panel titles and tick labels in faux
    small caps, matching the paper's ``\\textsc`` treatment. Returns the count.

    Any task name left in some other text (annotations, legend entries) fails
    loudly: those need :func:`small_caps_text` at their own call site, since
    their anchoring is not knowable here.
    """
    restyled = 0
    for ax in fig.axes:
        restyled += _restyle_title(ax)
        restyled += _restyle_ticklabels(ax)
    stragglers = sorted(
        {
            artist.get_text()
            for artist in fig.findobj(matplotlib.text.Text)
            if _TASK_NAME_RE.search(artist.get_text())
        }
    )
    if stragglers:
        raise ValueError(
            f"task names left in plain case: {stragglers} — draw these with "
            "paper.small_caps_text() so the figure matches the paper's \\textsc names"
        )
    return restyled


def assert_panel_titles_fit(fig: plt.Figure, *, name: str) -> None:
    """Fail if a panel title is clipped by the canvas or overlaps another."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bounds = fig.bbox
    title_artists = [
        axis.title for axis in fig.axes if axis.title.get_visible() and axis.title.get_text()
    ]
    title_artists.extend(
        artist
        for artist in fig.findobj(lambda candidate: candidate.get_gid() == _PANEL_TITLE_GID)
        if artist.get_visible()
    )
    titles = [(artist, artist.get_window_extent(renderer)) for artist in title_artists]

    for artist, bounds in titles:
        if not figure_bounds.contains(bounds.x0, bounds.y0) or not figure_bounds.contains(
            bounds.x1, bounds.y1
        ):
            raise ValueError(
                f"{name}: panel title extends outside the figure canvas "
                f"(artist={artist!r}, bounds={tuple(round(v, 2) for v in bounds.bounds)})"
            )

    for index, (artist, bounds) in enumerate(titles):
        for other_artist, other_bounds in titles[index + 1 :]:
            if bounds.overlaps(other_bounds):
                raise ValueError(
                    f"{name}: panel titles overlap (artists={artist!r}, {other_artist!r})"
                )


# ---------------------------------------------------------------------------
# Axis / annotation helpers
# ---------------------------------------------------------------------------


def style_axes(ax: plt.Axes, *, ygrid: bool = True, xgrid: bool = False) -> None:
    """Quiet dotted-free y-grid in the shared ink; spines come from PAPER_RC."""
    if ygrid:
        ax.grid(axis="y", color=INK, alpha=0.18, linewidth=0.8)
    if xgrid:
        ax.grid(axis="x", color=INK, alpha=0.18, linewidth=0.8)


def detach_spines(
    ax: plt.Axes,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    offset_pt: float = 5.0,
) -> None:
    """R-style axes: the left and bottom spines sit ``offset_pt`` outside the
    data area and span only ``bounds`` (the tick range), so they never touch
    each other and a series lying on the axis floor (e.g. a flat 0% curve)
    stays visually separate from the spine. Grid lines are unaffected, so the
    y=0 grid line remains the in-panel zero reference."""
    ax.spines["left"].set_position(("outward", offset_pt))
    ax.spines["left"].set_bounds(*y_bounds)
    ax.spines["bottom"].set_position(("outward", offset_pt))
    ax.spines["bottom"].set_bounds(*x_bounds)


def r0_reference(
    ax: plt.Axes,
    y: float,
    color: str,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> None:
    """THE reference-line look: a faint dashed horizontal at ``y``.

    Used for per-arm R0 (BC-only) levels on round-progression panels and for
    the unit reference in ratio plots. Call AFTER the axes' x-limits are final;
    ``xmin``/``xmax`` are data coords and default to the current limits.
    Round-progression figures using this for static R0 levels should include
    :func:`static_bc_legend_handle` in their legend.
    """
    lo, hi = ax.get_xlim()
    ax.plot(
        [lo if xmin is None else xmin, hi if xmax is None else xmax],
        [y, y],
        linestyle=STATIC_BC_LINESTYLE,
        linewidth=0.9,
        alpha=0.45,
        color=color,
        zorder=1.5,
    )
    ax.set_xlim(lo, hi)


def static_bc_legend_handle() -> Line2D:
    """Neutral legend key for lineage-colored static-R0 reference rules."""
    return Line2D(
        [0],
        [0],
        color=INK,
        linestyle=STATIC_BC_LINESTYLE,
        linewidth=0.9,
        alpha=0.45,
        label=STATIC_BC_LABEL,
    )


def y_axis_break(ax: plt.Axes, *, y_frac: float = 0.045, stroke_width_pt: float = 4.0) -> None:
    """THE zoomed-y-axis marker: two short diagonal strokes breaking the left
    spine near its bottom, signalling that the y range does not start at 0.

    Drawn in axes coordinates with ``clip_on=False`` so panel size does not
    change; call after layout is final. Pair with a caption clause naming the
    zoom range.
    """
    color = str(plt.rcParams["axes.edgecolor"])
    lw = float(plt.rcParams["axes.linewidth"])
    gap = 0.022
    # Axes-fraction x coordinates would make the printed slash width depend on
    # the subplot width. Convert one fixed physical width to axes coordinates
    # instead, so unequal-width panels carry identical break marks.
    axis_width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    if axis_width_pt <= 0:
        raise ValueError("cannot draw a y-axis break on a zero-width axis")
    x_half_width = 0.5 * stroke_width_pt / axis_width_pt
    # Blank the spine between the strokes so the axis visibly breaks.
    ax.plot(
        (0, 0),
        (y_frac - 0.004, y_frac + gap + 0.004),
        transform=ax.transAxes,
        color="white",
        clip_on=False,
        linewidth=lw * 2.5,
        zorder=5,
    )
    for yy in (y_frac, y_frac + gap):
        ax.plot(
            (-x_half_width, x_half_width),
            (yy - 0.011, yy + 0.011),
            transform=ax.transAxes,
            color=color,
            clip_on=False,
            linewidth=lw,
            zorder=6,
        )


def x_axis_break(ax: plt.Axes, *, x_frac: float) -> None:
    """THE broken-x-axis marker: two short diagonal strokes breaking the bottom
    spine at ``x_frac`` (axes fraction), signalling omitted x positions (e.g.
    skipped DAgger rounds). Same look as :func:`y_axis_break`, transposed.

    Call after layout is final; pair with a caption clause naming the omitted
    range.
    """
    color = str(plt.rcParams["axes.edgecolor"])
    lw = float(plt.rcParams["axes.linewidth"])
    gap = 0.016
    lo, hi = x_frac - gap / 2, x_frac + gap / 2
    ax.plot(
        (lo - 0.003, hi + 0.003),
        (0, 0),
        transform=ax.transAxes,
        color="white",
        clip_on=False,
        linewidth=lw * 2.5,
        zorder=5,
    )
    for xx in (lo, hi):
        ax.plot(
            (xx - 0.008, xx + 0.008),
            (-0.014, 0.014),
            transform=ax.transAxes,
            color=color,
            clip_on=False,
            linewidth=lw,
            zorder=6,
        )


def round_xlabel(fig: plt.Figure, *, y: float, label: str = ROUND_XLABEL) -> None:
    """Centered round-axis label, optionally naming a mixed checkpoint axis.

    ``y`` is figure-fraction: above the legend strip on figures that carry one
    (reserve the room via the ``tight_layout`` rect), else near the bottom.
    """
    fig.supxlabel(label, y=y, fontsize=float(plt.rcParams["axes.labelsize"]))


def panel_legend(ax: plt.Axes, *, loc: str, **kwargs) -> None:
    """In-panel frameless legend (panel corners are cheaper at print scale;
    a figure-level strip is reserved for the main headline's shared legend)."""
    ax.legend(loc=loc, frameon=False, borderaxespad=0.4, **kwargs)


# ---------------------------------------------------------------------------
# Save sink
# ---------------------------------------------------------------------------
# Build outputs live under paper/build/ in the repository; MULLIGAN_PAPER_BUILD
# moves them (CI, read-only checkouts).
BUILD_DIR = Path(os.environ.get("MULLIGAN_PAPER_BUILD", REPO_ROOT / "paper/build"))
FIGS_DIR = BUILD_DIR / "figs"
PROOF_DIR = BUILD_DIR / "proofs"


@dataclass(frozen=True)
class FigureRecord:
    """What one saved paper figure was: identity, geometry, content hash."""

    name: str
    pdf_path: Path
    proof_png: Path
    width_frac: float
    doc: str
    width_pt: float
    height_pt: float
    sha256: str
    sources: tuple[str, ...] = field(default_factory=tuple)


def save_paper_figure(
    fig: plt.Figure,
    name: str,
    *,
    width_frac: float,
    doc: str = "main",
    sources: tuple[str, ...] = (),
) -> FigureRecord:
    """Save ``fig`` as the paper figure ``<FIGS_DIR>/<name>.pdf`` plus a proof PNG.

    1. Asserts the authored width equals ``textwidth * width_frac`` — the
       permanent anti-regression against print-scale drift, then sets every
       task name in the figure in small caps (:func:`stylize_task_names`) and
       rejects panel titles that clip or overlap after that transformation.
    2. Writes the PDF (deterministic metadata, ``bbox_inches=None``).
    3. Writes ``<PROOF_DIR>/<name>.proof.png`` (dpi=200) — the eyeball surface —
       and closes the figure.
    """
    expected_w = DOC_TEXTWIDTH_IN[doc] * width_frac
    actual_w, actual_h = (float(v) for v in fig.get_size_inches())
    if abs(actual_w - expected_w) > 1e-6:
        raise ValueError(
            f"{name}: figure authored at {actual_w:.3f}in but width_frac={width_frac} of "
            f"doc={doc!r} requires exactly {expected_w:.3f}in — author the figure with "
            "paper.fig_size(), never rescale at include time"
        )
    # After layout, before any output: the swap is anchor-for-anchor, so the
    # PDF and the proof PNG below show the same geometry with the names set as
    # the paper sets them.
    stylize_task_names(fig)
    assert_panel_titles_fit(fig, name=name)
    pdf_path = FIGS_DIR / f"{name}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None, metadata={"CreationDate": None})
    proof_png = PROOF_DIR / f"{name}.proof.png"
    proof_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(proof_png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    return FigureRecord(
        name=name,
        pdf_path=pdf_path,
        proof_png=proof_png,
        width_frac=width_frac,
        doc=doc,
        width_pt=actual_w * 72.0,
        height_pt=actual_h * 72.0,
        sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        sources=tuple(sources),
    )
