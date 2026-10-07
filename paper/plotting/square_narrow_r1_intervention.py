"""Square-Narrow R1 per-sampler intervention-burden panel (main-text burden figure).

One panel that makes the whole point at once:

  * Solid bars: raw share of episodes with >=1 operator intervention.
    Highest for the Baseline arm, whose predecessor policy is the weakest ->
    measured using the original collection policies.
  * Hatched bars: the same quantity normalized by the collection policy's
    failure rate (1 - predecessor grid SR). The guided sampler has the highest
    adjusted ratio. This descriptive normalization does not directly measure
    failure incidence on its sampled initial states.

Whiskers: Wilson 95% CI on the episodes-with-intervention proportion,
propagated through the same baseline-indexing (and, for the hatched series,
the failure-rate divisor) as the point values; the divisor's own uncertainty
is not propagated (caption note).

Inputs are the pinned ``dataset_stats.json`` episode counts and the grid-eval SR
of the predecessor policies (:mod:`paper.appendix.component_panels.prepare`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from mulligan.plotting import paper
from mulligan.real.lifecycle.stats import wilson_ci

ARM_ORDER = ["baseline-uniform", "sobol", "mulligan"]
ARM_LABEL_KEYS = {
    "baseline-uniform": "baseline_uniform",
    "sobol": "sobol",
    "mulligan": "mulligan",
}
# Same sampler colors as the square_narrow_r1 sampler-focused panel: the no-CF sampler
# arms are blues (Sobol blue, guided darker blue); teal stays reserved for
# with-CF arms paper-wide.
ARM_COLORS = {
    "baseline-uniform": paper.BASELINE,
    "sobol": paper.SOBOL,
    "mulligan": paper.darken(paper.SOBOL),
}


def draw_panel(ax: plt.Axes, *, stats, sr, arms=ARM_ORDER) -> None:
    """Draw the per-sampler indexed intervention-burden panel onto ``ax``.

    ``stats`` maps arm -> dataset stats (``n_eps_with_intv``, ``n_episodes``);
    ``sr`` maps arm -> (predecessor mean SR, failure divisor, seeds). Used by the
    combined real+sim burden figure
    (:func:`paper.plotting.burden_progression.build_combined_figure`); no legend
    is added here — each caller places its own.
    """
    assert arms[0] == "baseline-uniform"

    raw_pct, raw_ci, adj_ratio, adj_ci, colors = [], [], [], [], []
    for arm in arms:
        s = stats[arm]
        k, n = int(s["n_eps_with_intv"]), int(s["n_episodes"])
        pct = k / n * 100.0
        lo, hi = wilson_ci(k, n)
        _, failure_div, _ = sr[arm]  # (mean_sr, 1 - mean_sr, n_seeds)
        raw_pct.append(pct)
        raw_ci.append((100.0 * lo, 100.0 * hi))
        adj_ratio.append((pct / 100.0) / failure_div)
        adj_ci.append((lo / failure_div, hi / failure_div))
        colors.append(ARM_COLORS[arm])

    # Both series are indexed to the Baseline arm so they share one unitless
    # axis; without this the ~4x raw-vs-adjusted magnitude gap (an artifact of
    # the failure-rate divisor) would dominate and bury the cross-arm story.
    # The Wilson bounds ride through the same per-series scaling.
    raw_idx = [v / raw_pct[0] for v in raw_pct]
    adj_idx = [v / adj_ratio[0] for v in adj_ratio]
    raw_ci_idx = [(lo / raw_pct[0], hi / raw_pct[0]) for lo, hi in raw_ci]
    adj_ci_idx = [(lo / adj_ratio[0], hi / adj_ratio[0]) for lo, hi in adj_ci]
    # Bar labels state the INDEXED values (what the axis shows), so label and
    # bar height agree; the Baseline absolute anchors live in the caption.
    raw_lab = [f"{v:.2f}$\\times$" for v in raw_idx]
    adj_lab = [f"{v:.2f}$\\times$" for v in adj_idx]

    x = list(range(len(arms)))
    w = 0.36

    def _draw(offset, idx, ci_idx, labels, hatch, label_ha):
        for xi, col, h, (lo, hi), lab in zip(x, colors, idx, ci_idx, labels):
            ax.bar(
                xi + offset,
                h,
                width=w,
                color=col,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
                zorder=2,
            )
            ax.errorbar(
                xi + offset,
                h,
                yerr=[[h - lo], [hi - h]],
                color=paper.ERRORBAR_INK,
                elinewidth=0.9,
                capsize=2.0,
                zorder=3,
            )
            # Value labels lean outward from the pair's center so the raw and
            # adjusted labels never collide over near-equal whisker tops.
            ax.text(
                xi + offset,
                hi + 0.03,
                lab,
                ha=label_ha,
                va="bottom",
                fontsize=6.5,
                color="black",
            )

    _draw(-w / 2, raw_idx, raw_ci_idx, raw_lab, None, "right")
    _draw(+w / 2, adj_idx, adj_ci_idx, adj_lab, "///", "left")

    ax.set_xlim(-0.6, len(arms) - 0.4 + 0.5)
    # The unit rule = the Baseline arm both series are indexed to.
    paper.r0_reference(ax, 1.0, paper.INK)
    ax.text(x[-1] + 0.42, 1.02, "Baseline", ha="left", va="bottom", fontsize=6.5, color=paper.INK)
    ax.set_ylabel("Intervention burden\n(relative to Baseline)")
    ax.set_ylim(0, max(hi for _, hi in adj_ci_idx) * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [paper.SIM_SAMPLER_LABELS[ARM_LABEL_KEYS[a]].replace(" (", "\n(") for a in arms]
    )
    paper.style_axes(ax)

    for arm, r, a in zip(arms, raw_pct, adj_ratio):
        print(f"  {arm:22s} raw={r:5.1f}%  adjusted={a:.2f}x")
