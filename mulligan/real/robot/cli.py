"""Robot reset/retry and fresh-observation flags shared by the real rollout entrypoints."""

from __future__ import annotations

import argparse


def add_robot_reset_args(parser: argparse.ArgumentParser) -> None:
    """Register the verified-reset retry flags and the fresh robot-state polling flags."""
    parser.add_argument(
        "--reset-max-retries",
        type=int,
        default=6,
        help="Verified robot reset attempts before failing loudly (default: 6)",
    )
    parser.add_argument(
        "--reset-retry-delay-s",
        type=float,
        default=1.0,
        help="Initial delay between failed reset attempts in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--reset-retry-backoff",
        type=float,
        default=1.5,
        help="Multiplier for reset retry delay after each failed attempt (default: 1.5)",
    )
    parser.add_argument(
        "--reset-max-retry-delay-s",
        type=float,
        default=5.0,
        help="Maximum delay between failed reset attempts in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--robot-state-refresh-max-wait-s",
        type=float,
        default=1.5,
        help=(
            "Maximum time to poll for a fresh robot-state timestamp after env.step() "
            "returns a stale observation while saving data (default: 1.5)"
        ),
    )
    parser.add_argument(
        "--robot-state-refresh-poll-interval-s",
        type=float,
        default=0.01,
        help="Polling interval for fresh robot-state observations in seconds (default: 0.01)",
    )
