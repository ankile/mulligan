"""Human-robot team success during real-world DAgger collection (fig:dagger-collection-success).

Sums the per-round collection tables pinned by the ``productivity`` appendix package
(fresh, policy-start episodes; counterfactual replays excluded). The paper's "98% of the
1,009 fresh collection episodes ..., with no round below 90%" is the HiL-IDQL+Mulligan
collection campaign (``mulligan_sobol``) over Marker, Nut and Cable; the Cable R1-R5 increments
sum two collection sessions of the velocity-action collectors each. The HG-DAgger
campaign (``baseline_uniform``) is printed alongside.

    python -m paper.stats.print_collection_totals
"""

from __future__ import annotations

import csv

from paper.appendix.artifacts import REAL_DIRS
from paper.stats.evidence import pinned_input

TASKS = {"marker_d2": "Marker", "square_d2": "Nut", "routing_d2": "Cable"}
CAMPAIGNS = {"mulligan_sobol": "HiL-IDQL+Mulligan", "baseline_uniform": "HG-DAgger"}


def collection_rounds() -> list[dict]:
    """One row per (task, collection round, campaign): credited successes over fresh attempts."""
    rows = []
    for task in TASKS:
        path = pinned_input("productivity", f"real/collection/{REAL_DIRS[task]}/collection.csv")
        with path.open(newline="") as stream:
            for r in csv.DictReader(stream):
                if r["arm_key"] not in CAMPAIGNS:
                    raise ValueError(f"{path}: unexpected arm {r['arm_key']!r}")
                successes, n = int(r["successes"]), int(r["n"])
                assert 0 < successes <= n, (task, r["round"], r["arm_key"])
                rows.append(
                    dict(task=task, round=r["round"], arm=r["arm_key"], successes=successes, n=n)
                )
    return rows


def totals(rows: list[dict] | None = None) -> dict[str, dict]:
    """Per campaign: total successes / episodes and the lowest-rate round."""
    rows = collection_rounds() if rows is None else rows
    out = {}
    for arm in CAMPAIGNS:
        mine = [r for r in rows if r["arm"] == arm]
        assert len(mine) == 15, (arm, len(mine))  # three tasks x R1-R5
        worst = min(mine, key=lambda r: r["successes"] / r["n"])
        out[arm] = dict(
            successes=sum(r["successes"] for r in mine),
            n=sum(r["n"] for r in mine),
            min_round=worst,
        )
    return out


def main() -> None:
    for arm, t in totals().items():
        w = t["min_round"]
        print(
            f"{CAMPAIGNS[arm]:18s} {t['successes']:,}/{t['n']:,} = {100 * t['successes'] / t['n']:.1f}%; "
            f"min round {TASKS[w['task']]} {w['round']} {w['successes']}/{w['n']} "
            f"= {100 * w['successes'] / w['n']:.1f}%"
        )


if __name__ == "__main__":
    main()
