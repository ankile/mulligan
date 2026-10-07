"""Predecessor-policy grid success rate for the burden panel's failure-rate adjustment.

The simulation burden panel divides each arm's intervention share by the failure
rate ``1 - SR`` of the policy that collected its data. The SR is the mean grid-eval
success of that predecessor policy over its seeds, read from the pinned grid table.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredecessorPolicy:
    """Where to look up the grid SR that drove collection of an arm.

    `cf_mode` / `data_mode` are optional filters used when the predecessor cell
    lives at a level of the protocol matrix that includes those dims (R1+ cells
    do; R0 doesn't).
    """

    task_label: str
    stage: str
    arm: str
    cf_mode: str | None = None
    data_mode: str | None = None


def _run_key_matches(run_key: str, policy: PredecessorPolicy) -> bool:
    """Filter grid rows by cf_mode/data_mode encoded into `run_key`.

    The grid table encodes (cf_mode, data_mode) inside `run_key` as the suffix
    `_{cf_mode}_{data_mode}` (e.g. `square_narrow_r1_baseline_uniform_no_cf_human_only`)
    when those dims are part of the protocol matrix; R0 rows have no suffix.
    """
    needed = [mode for mode in (policy.cf_mode, policy.data_mode) if mode is not None]
    return all(token in run_key for token in needed)


def compute_predecessor_grid_sr(
    predecessor_policy: dict[str, PredecessorPolicy],
    grid_csv: Path,
) -> dict[str, tuple[float, float, int]]:
    """For each arm, return its predecessor policy's mean grid SR.

    Returns `{arm: (sr_mean_0_to_1, failure_rate_divisor, n_seeds)}`; an arm whose
    predecessor has no grid rows fails loudly.
    """
    by_arm_rows: dict[str, list[dict]] = defaultdict(list)
    with grid_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or not row.get("overall_sr"):
                continue
            by_arm_rows[row["arm"]].append(row)
    out: dict[str, tuple[float, float, int]] = {}
    for cur_arm, policy in predecessor_policy.items():
        cands = [
            r
            for r in by_arm_rows.get(policy.arm, [])
            if r.get("task_label") == policy.task_label
            and r.get("stage") == policy.stage
            and _run_key_matches(r.get("run_key", ""), policy)
        ]
        if not cands:
            raise RuntimeError(
                f"no grid-eval SR for predecessor {policy} of arm {cur_arm!r}; "
                f"available arms: {sorted(by_arm_rows)}"
            )
        srs = [float(r["overall_sr"]) for r in cands]
        mean_sr = statistics.mean(srs) / 100.0
        out[cur_arm] = (mean_sr, 1.0 - mean_sr, len(srs))
    return out
