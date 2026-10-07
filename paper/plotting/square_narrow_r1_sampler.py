"""Square-Narrow R1 grid-eval bar panels of the main-text ablation figure.

Two single-message panels drawn from the frozen-actor grid cells, consumed by
``ablations_sim_real`` (:mod:`paper.appendix.component_panels`):

  Left  -- collection strategy compounds. A monotone progression on grid-eval
           SR: Baseline (uniform) < Ours (guided) < Ours (guided + CF), all
           human-only except the last, which adds counterfactual demos.
  Right -- data composition for the base policy. The Baseline arm trained on
           human-only data beats the same arm trained on human data augmented
           with successful autonomous DAgger rollouts (`straddled_auto_success`),
           motivating training the actor on curated data only.

Both panels are the fixed 80x100 grid-eval SR: bar = 5-seed mean with a
seed-level 95% t-CI (df=4) whisker, plus the five individual seed values as
dots (whisker family named in the caption). The y axis is zoomed (these are
all high-SR policies) with per-bar means annotated.

:func:`draw_sampler_panels` fills two caller-owned axes and shares the right
one on the left's 78--100% y scale; the caller applies ``paper.y_axis_break``
after layout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mulligan.plotting import paper
from mulligan.real.lifecycle.stats import t_interval


def _cell(by_grid, cf, dm, arm):
    vals = by_grid[(cf, dm, arm)]
    # Seed-level 95% t-CI half-width (df=n_seeds-1; n=5 seeds) rather than 1 SE.
    mu, _se, lo, hi = t_interval(np.asarray(vals, dtype=float))
    return mu, (hi - lo) / 2.0, vals


SHARED_YLIM = (78.0, 100.0)
SHARED_YTICKS = tuple(range(80, 101, 5))
BAR_LABEL_OFFSET = 0.4


def draw_sampler_panels(axL: plt.Axes, axR: plt.Axes, *, by_grid, include_sobol=True) -> None:
    """Fill the two bar panels on one shared, zoomed success-rate axis.

    Sets titles, the shared ylim/yticks, and ``axL``'s ylabel; the caller
    applies ``paper.y_axis_break`` to both axes after layout. The upper limit
    is the natural 100% endpoint rather than 95% because the largest 95% CI
    reaches 94.85% and its mean label sits above that whisker.
    """
    axR.sharey(axL)
    axR.tick_params(labelleft=False)

    # Left panel: collection-strategy progression. Same CF convention as the
    # real-world CF ablation: the no-CF arms are blues (Sobol blue, guided a
    # darker blue) and only the with-CF arm takes the Ours teal. Tick labels
    # are compressed for the narrow combined-row slot (the caption carries the
    # full SIM_SAMPLER_LABELS register: only "Uniform" is the Baseline arm).
    prog = [
        ("Uniform", ("no_cf", "human_only", "baseline_uniform"), paper.BASELINE, None),
        ("Sobol", ("no_cf", "human_only", "sobol"), paper.SOBOL, None),
        (
            "Guided",
            ("no_cf", "human_only", "mulligan"),
            paper.darken(paper.SOBOL),
            None,
        ),
        (
            "Guided\n+ CF",
            ("with_cf", "human_only", "mulligan"),
            paper.OURS,
            "///",
        ),
    ]
    if not include_sobol:
        prog = [row for row in prog if row[0] != "Sobol"]
    # Right panel: base-policy data composition (Baseline arm only).
    comp = [
        ("Human\nonly", ("no_cf", "human_only", "baseline_uniform"), paper.BASELINE, None),
        (
            "+ Auto\nsuccesses",
            ("no_cf", "straddled_auto_success", "baseline_uniform"),
            paper.BASELINE,
            "xxx",
        ),
    ]

    def _bars(ax, spec, title, base_width=0.62):
        xs = np.arange(len(spec))
        for xi, (lab, key, col, hatch) in zip(xs, spec):
            mu, ci, vals = _cell(by_grid, *key)
            ax.bar(
                xi,
                mu,
                width=base_width,
                color=col,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
                zorder=2,
            )
            ax.errorbar(
                xi, mu, yerr=ci, color=paper.ERRORBAR_INK, capsize=2.0, elinewidth=1.0, zorder=3
            )
            # Individual seed values (plotting_principles: show seed-level
            # variability, no per-seed hue).
            jitter = np.linspace(-0.14, 0.14, len(vals))
            ax.scatter(
                xi + jitter, vals, s=4.0, color=paper.INK, alpha=0.65, linewidths=0, zorder=4
            )
            ax.text(
                xi,
                max(mu + ci, max(vals)) + BAR_LABEL_OFFSET,
                f"{mu:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="black",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in spec], fontsize=6.0)
        ax.set_title(title)
        paper.style_axes(ax)
        ax.set_xlim(-0.65, len(spec) - 0.35)

    _bars(axL, prog, "State initialization (sim)")
    _bars(axR, comp, "Actor data (sim)")
    axL.set_ylabel("Success rate (%)")
    # The panels repeat the same baseline bar, so one scale makes both its
    # height and every treatment delta directly comparable. Keep the zoomed
    # lower bound, but end at the natural 100% success-rate ceiling. Seed dots,
    # confidence intervals, and value labels must all fit without clipping.
    for spec in (prog, comp):
        seed_vals = [v for _, key, _, _ in spec for v in _cell(by_grid, *key)[2]]
        upper_extents = [
            max(mu + ci, max(vals)) + BAR_LABEL_OFFSET
            for _, key, _, _ in spec
            for mu, ci, vals in [_cell(by_grid, *key)]
        ]
        if min(seed_vals) < SHARED_YLIM[0] or max(upper_extents) > SHARED_YLIM[1]:
            raise RuntimeError(
                "square_narrow_r1_sampler: data or labels fall outside the shared "
                f"{SHARED_YLIM} zoom (seed min={min(seed_vals):.1f}, "
                f"annotation max={max(upper_extents):.1f})"
            )
    axL.set_ylim(*SHARED_YLIM)
    axL.set_yticks(SHARED_YTICKS)

    for lab, key, _, _ in prog + comp:
        mu, ci, vals = _cell(by_grid, *key)
        print(f"  {lab.replace(chr(10), ' '):22s} {mu:5.1f} +/- {ci:4.1f} (n={len(vals)})")
