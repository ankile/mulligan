"""Main-text ablation and collection-burden figures from pinned evidence."""

import matplotlib.pyplot as plt

from paper.appendix.artifacts import source_paths
from mulligan.plotting import paper
from paper.plotting import cf_ablation as cf
from paper.plotting import burden_progression as burden
from paper.plotting import square_narrow_r1_sampler as sampler
from .prepare import HERE, CACHE, NAMES, SHARED, prepare


def ablations(groups, cf_paths):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=paper.fig_size(1.0, height_in=1.62),
        gridspec_kw={"width_ratios": [1.55, 1.05, 1.4, 1.1]},
    )
    sampler.draw_sampler_panels(axes[0], axes[1], by_grid=groups, include_sobol=False)
    axes[0].set_title("State sampling")
    axes[1].set_title("Actor data")
    for ax, task in zip(axes[2:], ("square_d2", "marker_d2"), strict=True):
        cf.draw_cf_panel(ax, task, path=cf_paths[task])
        ax.set_ylabel("")
        ax.set_xlabel("")  # R0, R1, ... ticks name the rounds on their own
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[3].sharey(axes[2])
    axes[3].tick_params(labelleft=False)
    handles, _ = axes[2].get_legend_handles_labels()
    axes[2].legend(
        handles, ["With CF", "No CF"], loc="upper left", fontsize=6, frameon=False, handlelength=1.2
    )
    fig.tight_layout(pad=0.35, w_pad=0.5)
    for axis in axes[:2]:
        paper.y_axis_break(axis)
    return fig


def build_records(*, name):
    assert name in NAMES
    groups, cf_paths, stats, sr = prepare()
    width = 0.68 if name == "collection_burden_real_sim" else 1.0
    with paper.paper_rc():
        if name == "ablations_sim_real":
            fig = ablations(groups, cf_paths)
        else:
            fig = burden.build_combined_figure(
                data_dir=CACHE / "raw/real/collection",
                sim_stats=stats,
                sim_sr=sr,
                sim_arms=("baseline-uniform", "mulligan"),
                width_frac=width,
                compact=True,
            )
        return paper.save_paper_figure(
            fig, name, width_frac=width, sources=source_paths(HERE) + SHARED
        )
