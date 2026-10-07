"""Median round-0 demonstration duration per task (fig:task-sequences caption, tab:task-summary).

The paper's 12.7 / 10.1 / 18.6 s (Marker / Nut / Cable) and 8.3 / 8.6 s (Square-Narrow /
Square-Broad) are the median episode length over all round-0 teleoperated demonstrations of
a task, divided by the control rate and rounded half up to 0.1 s. Real tasks: 250
demonstrations each (100 uniform, 100 Sobol and 50 held-out validation starts, all
successful) at 15 Hz. Simulation: every round-0 demonstration of both arms (200 on
Square-Narrow, 400 on Square-Broad) at 20 Hz; Square-Narrow's median is 165 frames (8.25 s)
and Square-Broad's 171 frames (8.55 s). They are read from ``meta/episodes`` (``length``) of
the public ``mulligan/*-c00-teleop-mixed`` datasets at their pinned revisions. Per-variant
medians (``-baseline`` / ``-sobol`` / ``-validation``) do not reproduce the Cable value.

    python -m paper.stats.r0_demo_durations
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from mulligan.release.download import pinned_revision

# task key: (paper name, round-0 dataset, demonstrations)
TASKS = {
    "marker": ("Marker", "mulligan/real-marker-d2-c00-teleop-mixed", 250),
    "square": ("Nut", "mulligan/real-square-d2-c00-teleop-mixed", 250),
    "routing": ("Cable", "mulligan/real-routing-d2-c00-teleop-mixed", 250),
    "square_narrow": ("Square-Narrow", "mulligan/sim-square-narrow-c00-teleop-mixed", 200),
    "square_broad": ("Square-Broad", "mulligan/sim-square-broad-c00-teleop-mixed", 400),
}


def _download(repo: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo, path, repo_type="dataset", revision=pinned_revision(repo), token=False
        )
    )


def episode_lengths(repo: str, n_episodes: int) -> tuple[np.ndarray, int]:
    """Per-episode frame counts and the dataset's fps."""
    from huggingface_hub import HfApi

    info = json.loads(_download(repo, "meta/info.json").read_text())
    files = sorted(
        f
        for f in HfApi().list_repo_files(
            repo, repo_type="dataset", revision=pinned_revision(repo), token=False
        )
        if f.startswith("meta/episodes/") and f.endswith(".parquet")
    )
    episodes = pd.concat([pd.read_parquet(_download(repo, f)) for f in files])
    if sorted(episodes.episode_index) != list(range(info["total_episodes"])):
        raise ValueError(f"{repo}: meta/episodes does not cover {info['total_episodes']} episodes")
    if info["total_episodes"] != n_episodes:
        raise ValueError(f"{repo}: {info['total_episodes']} episodes, expected {n_episodes}")
    if int(episodes.length.sum()) != info["total_frames"]:
        raise ValueError(f"{repo}: episode lengths do not sum to total_frames")
    return episodes.length.to_numpy(), int(info["fps"])


def round_half_up(seconds: Decimal) -> Decimal:
    """Round to 0.1 s, halves up (8.25 -> 8.3), as the paper prints."""
    return seconds.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def median_durations() -> dict[str, Decimal]:
    """{task: median round-0 demonstration duration in seconds}, exact (frames / fps)."""
    out = {}
    for task, (_, repo, n_episodes) in TASKS.items():
        lengths, fps = episode_lengths(repo, n_episodes)
        # The median of integer lengths is a whole or half frame, exact as a decimal.
        out[task] = Decimal(str(float(np.median(lengths)))) / fps
    return out


def main() -> None:
    for task, seconds in median_durations().items():
        name = TASKS[task][0]
        print(f"{name:13s} median R0 demonstration: {round_half_up(seconds)} s ({seconds:.3f})")


if __name__ == "__main__":
    main()
