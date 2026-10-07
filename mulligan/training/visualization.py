"""Visualization utilities for training analysis."""

from pathlib import Path
from typing import Optional


def create_qv_trajectory_plots(
    q_trajectories: list[list[float]],
    v_trajectories: list[list[float]],
    successes: list[bool],
    output_path: Path,
    step: Optional[int] = None,
    a_trajectories: Optional[list[list[float]]] = None,
) -> None:
    """
    Create side-by-side Q-value, V-value, and Advantage trajectory plots for IQL.

    Plots all Q-trajectories, V-trajectories, and Advantage (Q-V) trajectories over time
    (green for success, red for failure).

    Args:
        q_trajectories: List of Q-value trajectories (one list per episode)
        v_trajectories: List of V-value trajectories (one list per episode)
        successes: List of success flags (bool per episode)
        output_path: Path to save the plot
        step: Current training step (for title)
        a_trajectories: List of Advantage (Q-V) trajectories (one list per episode)
    """
    if not q_trajectories or not v_trajectories:
        return

    import matplotlib.pyplot as plt

    # Determine number of subplots based on whether advantage trajectories are provided
    if a_trajectories:
        fig, (ax_q, ax_v, ax_a) = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, (ax_q, ax_v) = plt.subplots(1, 2, figsize=(16, 6))

    # Separate successful and failed episodes
    successful_q_trajs = [traj for i, traj in enumerate(q_trajectories) if successes[i]]
    failed_q_trajs = [traj for i, traj in enumerate(q_trajectories) if not successes[i]]
    successful_v_trajs = [traj for i, traj in enumerate(v_trajectories) if successes[i]]
    failed_v_trajs = [traj for i, traj in enumerate(v_trajectories) if not successes[i]]

    # Plot Q-value trajectories
    for i, traj in enumerate(successful_q_trajs):
        steps = list(range(len(traj)))
        label = f"Success ({len(successful_q_trajs)})" if i == 0 else ""
        ax_q.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label=label)

    for i, traj in enumerate(failed_q_trajs):
        steps = list(range(len(traj)))
        label = f"Failure ({len(failed_q_trajs)})" if i == 0 else ""
        ax_q.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label=label)

    ax_q.set_xlabel("Episode Step")
    ax_q.set_ylabel("Q-Value")
    ax_q.set_title(f"Q-Value Trajectories (Training Step {step if step is not None else 'N/A'})")
    ax_q.grid(True, alpha=0.3)
    if successful_q_trajs or failed_q_trajs:
        ax_q.legend()

    # Plot V-value trajectories
    for i, traj in enumerate(successful_v_trajs):
        steps = list(range(len(traj)))
        label = f"Success ({len(successful_v_trajs)})" if i == 0 else ""
        ax_v.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label=label)

    for i, traj in enumerate(failed_v_trajs):
        steps = list(range(len(traj)))
        label = f"Failure ({len(failed_v_trajs)})" if i == 0 else ""
        ax_v.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label=label)

    ax_v.set_xlabel("Episode Step")
    ax_v.set_ylabel("V-Value")
    ax_v.set_title(f"V-Value Trajectories (Training Step {step if step is not None else 'N/A'})")
    ax_v.grid(True, alpha=0.3)
    if successful_v_trajs or failed_v_trajs:
        ax_v.legend()

    # Plot Advantage trajectories (if provided)
    if a_trajectories:
        successful_a_trajs = [traj for i, traj in enumerate(a_trajectories) if successes[i]]
        failed_a_trajs = [traj for i, traj in enumerate(a_trajectories) if not successes[i]]

        for i, traj in enumerate(successful_a_trajs):
            steps = list(range(len(traj)))
            label = f"Success ({len(successful_a_trajs)})" if i == 0 else ""
            ax_a.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label=label)

        for i, traj in enumerate(failed_a_trajs):
            steps = list(range(len(traj)))
            label = f"Failure ({len(failed_a_trajs)})" if i == 0 else ""
            ax_a.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label=label)

        ax_a.set_xlabel("Episode Step")
        ax_a.set_ylabel("Advantage (Q-V)")
        ax_a.set_title(
            f"Advantage Trajectories (Training Step {step if step is not None else 'N/A'})"
        )
        ax_a.grid(True, alpha=0.3)
        if successful_a_trajs or failed_a_trajs:
            ax_a.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Ensure file is fully written to disk before returning
    # This prevents race conditions when wandb tries to read it immediately
    import os

    if output_path.exists():
        # Force filesystem to flush the file
        with open(output_path, "rb") as f:
            os.fsync(f.fileno())

    print(f"  ✓ Saved Q/V/A-trajectory plots to: {output_path}")
