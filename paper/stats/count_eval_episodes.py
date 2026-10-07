"""Count the held-out real-robot evaluation episodes behind the paper (2,550 / 2,850).

The count is defined by the round-dataset lock ``release/round-datasets.json`` (byte-identical to
the copy pinned in the ``real_results`` paper evidence): one entry per released evaluation
dataset, each episode with a role (``counted``, ``no-cf-ablation``; excluded episodes are not in
the release). The paper's 2,550 sums the ``counted`` episodes over the 13 headline datasets (kinds
``round`` and ``cable``); the two Marker and Nut R5 candidate-screen datasets (``*-d2-r05-screen``,
400 episodes) are left out. Adding the no-CF ablation arms gives 2,850. Each
dataset's counts are re-derived from its per-episode roles. The rule is
``mulligan.release.round_counts``; ``mulligan.release.verify_results`` applies it to the public HF
metadata.

    python -m paper.stats.count_eval_episodes
"""

from __future__ import annotations

from mulligan.release.round_counts import load_lock, lock_totals

TASK_NAMES = {"real-marker-d2": "Marker", "real-square-d2": "Nut", "real-routing-d2": "Cable"}


def episode_totals() -> dict:
    """Headline totals over the lock's headline datasets, overall and per task."""
    lock, _ = load_lock()
    return lock_totals(lock)


def main() -> None:
    t = episode_totals()
    print(
        f"{t['counted']:,} held-out evaluation episodes over {t['datasets']} lock datasets "
        f"({t['counted_plus_no_cf']:,} with the no-CF ablation)"
    )
    for task, c in t["by_task"].items():
        print(f"  {TASK_NAMES[task]:6s} {c['counted']:,} (+{c['no_cf_ablation']} no-CF)")
    print(f"  excluded candidate-screen datasets: {', '.join(t['screen'])}")


if __name__ == "__main__":
    main()
