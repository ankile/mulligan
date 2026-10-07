"""Project-wide plotting style guide and colour palettes.

This module is the single source of truth for visualisation choices across the
project. New plotting scripts should import constants and helpers from here
rather than redefining hex codes locally.

==============================================================================
DESIGN PRINCIPLES
==============================================================================

1. **Within a method family, lightness encodes capacity.** When showing the
   same method at multiple sizes, use the same hue and darken with capacity.
   This keeps "method identity" visually primary and "capacity" secondary.

2. **Ordinal outcome ladders use a semantic ramp.** Stage ladders run red (early
   failure) -> amber -> green (success).

3. **Seeds get no colour encoding.** Show seed-level variability via error
   bands, faded individual lines, or small multiples.

==============================================================================
PLOT TYPE -> PALETTE MAPPING
==============================================================================

::

    Method / arm comparisons            -> METHOD_COLORS
    State-based RLPD codebases          -> STATE_RLPD_CODEBASE_COLORS
    Stage-ladder share stacked bar      -> STAGE_LADDER_COLORS (stage_color)
    Annotation / errorbar ink           -> NEUTRAL_INK, NEUTRAL_INK_MID
    Operator target cards               -> OPERATOR_CARD_COLORS
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Methods (categorical)
# ---------------------------------------------------------------------------
# Hue picks the method family; lightness within a hue distinguishes capacity.

METHOD_COLORS: dict[str, str] = {
    # Sobol: structured low-discrepancy sampling (matplotlib C0 blue family).
    "sobol": "#1f77b4",
    "sobol_light": "#7da7ca",
    # Real-robot arms: HG-DAgger+Mulligan without and with counterfactual (CF)
    # replays, and the HiL-IDQL+Mulligan best-of-N re-rank arm. Purple is reserved
    # for the re-rank recipe in the paper's figures.
    "real_mulligan_no_cf": "#1f77b4",
    "real_mulligan_with_cf": "#17becf",
    "real_iql_rerank": "#9467bd",
    # Uniform random: the "no method" baseline, visually neutral grey; the darker
    # step marks the reranked deployment of the same arm.
    "uniform": "#888888",
    "uniform_dark": "#5a5a5a",
    # Outcome classes of the held-out evaluation figures.
    "outcome_success": "#2ca02c",
    "outcome_timeout": "#f4a261",
    "outcome_failure": "#d62728",
    # Extra accent for figures with more arms than named colours.
    "accent_terracotta": "#e76f51",
    # Neutral grey for references, grids and annotations.
    "gray_neutral": "#7f8c8d",
    # Simulation autonomous baselines: one warm muted "no human in the loop" family
    # (gold / terracotta / brown). IDQL is brown, not the re-rank purple.
    "sim_auto_plain_il": "#d4a017",
    "sim_auto_filtered_bc": "#e76f51",
    "sim_auto_iql": "#8c564b",
    # HiL-SERL appendix variants: operator from step 0 (red) and operator buffer,
    # then no operator (gold).
    "hilserl_from_step0": "#d62728",
    "hilserl_buffer_then_none": "#d4a017",
}

# State-based RLPD learning curves (paper intro figure and the HiL-SERL appendix).
# RLPD is SAC-based and keeps the SAC blue; HiL-SERL (split learner/actor RLPD
# with operator interventions) is green. Ours (DP+BoN) uses ``paper.OURS_RERANK``.
STATE_RLPD_CODEBASE_COLORS: dict[str, str] = {
    "rlpd": "#4c78a8",
    "hilserl": "#54a24b",
}


# ---------------------------------------------------------------------------
# Neutral ink accents
# ---------------------------------------------------------------------------
# quiet ink accents for annotations/errorbars — not a data palette
NEUTRAL_INK = "#222222"
NEUTRAL_INK_MID = "#333333"

# ---------------------------------------------------------------------------
# The stage ladder
# ---------------------------------------------------------------------------

# Stage ladder (max stage reached, S0..S7; high = success). This is a *semantic*
# outcome ladder — early failure (S0) is bad, strict success (S7) is good — so it
# uses a red -> amber -> green ramp. These 8 anchors are frozen as explicit hex so the palette is
# stable regardless of matplotlib colormap changes; ``CMAP_STAGE`` interpolates
# them so ``stage_color(s, n_stages)`` works for any rung count and lands exactly
# on the frozen anchors at the default 8 rungs. Stage-share stacked bars all draw
# their fills from here via ``stage_color`` / ``stage_palette`` rather than
# re-deriving ``cmap(s/7)``.
STAGE_LADDER_COLORS: list[str] = [
    "#b1342f",  # S0 — failed earliest (brick red)
    "#dd6a3e",  # S1 — red-orange
    "#eb9a4e",  # S2 — orange
    "#ecc15e",  # S3 — amber / gold (partial progress)
    "#bcc862",  # S4 — yellow-green
    "#86bd66",  # S5 — light green
    "#4ea76a",  # S6 — green
    "#247a4f",  # S7 — strict success (deep green)
]
CMAP_STAGE = LinearSegmentedColormap.from_list("stage_ladder", STAGE_LADDER_COLORS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stage_color(stage: int, n_stages: int = 8):
    """Colour for stage ``stage`` of an ``n_stages``-rung ladder (S0..S{n-1}).

    Samples ``CMAP_STAGE`` (the frozen red->amber->green ladder) so the lowest
    stage is red (failed early) and the top stage (strict success) is green.
    ``n_stages`` lets callers that show only a subset of rungs (e.g. S1/S3/S7
    proxies) keep the *same* absolute hue per stage by passing the full ladder
    size and indexing real stage numbers; at the default 8 rungs the samples land
    exactly on ``STAGE_LADDER_COLORS``.
    """
    if n_stages < 2:
        raise ValueError(f"n_stages must be >= 2, got {n_stages}")
    # Exact frozen hex for the canonical ladder (the only shape callers use);
    # the 256-entry colormap LUT would otherwise drift ~1 LSB off each anchor.
    if n_stages == len(STAGE_LADDER_COLORS) and 0 <= stage < n_stages:
        return STAGE_LADDER_COLORS[stage]
    return CMAP_STAGE(stage / (n_stages - 1))


def stage_palette(n_stages: int = 8) -> list:
    """Return the full ordered stage-ladder palette (S0 -> S{n_stages-1})."""
    return [stage_color(s, n_stages) for s in range(n_stages)]


def label_text_color(
    facecolor,
    *,
    threshold: float = 0.30,
    light: str = "white",
    dark: str = "black",
):
    """Pick a readable text colour to overlay on ``facecolor``.

    Returns ``light`` when the background's relative luminance (sRGB-linearised
    Rec.709) is below ``threshold``, else ``dark``. Stacked-bar and annotated-heatmap
    callers use it for per-cell / per-segment label colours; it works for any
    palette.
    """
    from matplotlib.colors import to_rgba

    r, g, b, _ = to_rgba(facecolor)

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return light if luminance < threshold else dark


# Real-robot operator target cards (mulligan/real/operator_ui/cards.py): the scene objects
# the operator places, drawn identically on every task's card.
OPERATOR_CARD_COLORS: dict[str, str] = {
    "panel_background": "#17212b",
    "panel_text": "#f5f7fa",
    "panel_muted": "#bdc9d4",
    "panel_track": "#394958",
    "panel_accent": "#70d0b2",
    "target_center": "#1f77b4",
    "target_arrow": "#8c1d18",
    "placement": "#d62728",
    "clip_right": "#e76f51",
    "rope": "#2ca02c",
    "rope_edge": "#145a14",
    "nut_face": "#c6a35b",
    "nut_edge": "#3a2a12",
    "peg_face": "#777777",
    "ink": "#202020",
    "table_face": "#f4f4f4",
    "table_edge": "#606060",
    "pen_rect_face": "#e9f3ff",
    "pen_rect_edge": "#1f4e79",
    "guide": "#4a4a4a",
    "axis": "#808080",
    "grid_major": "#c8c8c8",
    "grid_minor": "#e4e4e4",
}
