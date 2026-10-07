"""The operator-UI command-line flags, registered identically by every real entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_operator_ui_args(parser: argparse.ArgumentParser, *, cards: bool) -> None:
    """Register the live camera monitor flags and (when ``cards``) the target-card flags.

    ``cards`` is True for manifest-driven entrypoints (collection, teleop, eval) that show
    an initial-state target card per episode; DAgger has no manifest and
    get only the monitor flags.
    """
    group = parser.add_argument_group("operator UI")
    group.add_argument(
        "--no-status-window",
        action="store_true",
        help="Disable the live session panel. Target cards still follow their own window flag.",
    )
    group.add_argument(
        "--monitor-cameras",
        action="store_true",
        help=(
            "Show live OpenCV windows of the camera crops the policy actually sees "
            "(default keys: side_1,wrist_left). Needs a display."
        ),
    )
    group.add_argument(
        "--monitor-camera-keys",
        type=str,
        default=None,
        help=(
            "Comma-separated ROLE names (e.g. 'side_1,wrist_left') or raw serial keys to "
            "show with --monitor-cameras. Cameras without a crop box are skipped."
        ),
    )
    if not cards:
        return
    group.add_argument(
        "--initial-state-visualization-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-target operator card PNGs "
            "(default: <dataset>/meta/initial_state_targets or a /tmp dir for evals)."
        ),
    )
    group.add_argument(
        "--no-show-initial-state-window",
        action="store_true",
        help="Write the target card PNGs but do not open the OpenCV card window.",
    )
