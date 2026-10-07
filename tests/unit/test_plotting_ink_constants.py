"""Pin the neutral-ink accent constants used for annotations/errorbars.

These are quiet ink accents (not a data palette). Plotting modules import them
from ``mulligan.plotting.colors`` instead of hardcoding the hex literals.
"""

from __future__ import annotations

from mulligan.plotting.colors import NEUTRAL_INK, NEUTRAL_INK_MID


def test_neutral_ink_constants_match_exact_hex() -> None:
    assert NEUTRAL_INK == "#222222"
    assert NEUTRAL_INK_MID == "#333333"
