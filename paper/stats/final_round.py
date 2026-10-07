"""Final-round (R5) real-world success: HiL-IDQL+Mulligan and HG-DAgger+Mulligan vs. HG-DAgger.

Reads the pooled R5 rows of the frozen line-headline tables. The paper's headline margins
(+34 / +10 / +16 pp on Marker / Nut / Cable) are full-task success on 50 held-out starts;
for Cable that is ``routing_d2_headline_sr.csv`` (both clips routed), not the clip success
rate of the plotted Cable curve (``routing_d2_headline_task_progress.csv``), which is also
reported here.

    python -m paper.stats.final_round
"""

from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data/real"
HEADLINES = {
    "marker_d2": DATA / "marker/marker_d2_headline_sr.csv",
    "square_d2": DATA / "square/square_d2_headline_sr.csv",
    "routing_d2": DATA / "routing/routing_d2_headline_sr.csv",
}
CABLE_CLIP = DATA / "routing/routing_d2_headline_task_progress.csv"
TASK_NAMES = {"marker_d2": "Marker", "square_d2": "Nut", "routing_d2": "Cable"}
FINAL_ROUND = "R5"
# arm keys of the headline tables -> paper method names
METHODS = {
    "baseline": "HG-DAgger",
    "mulligan_with_cf": "HG-DAgger+Mulligan",
    "final_iql": "HiL-IDQL+Mulligan",
    "final_marker_iql": "HiL-IDQL+Mulligan",
}


def _final_round(path: Path) -> dict[str, tuple[int, int]]:
    """{method: (successes, n)} of the pooled final-round rows."""
    with path.open(newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["round"] == FINAL_ROUND and r["is_pooled"] == "True"
        ]
    out = {}
    for r in rows:
        method = METHODS[r["arm"]]
        if method in out:
            raise ValueError(f"{path}: two pooled {FINAL_ROUND} rows for {method}")
        out[method] = (int(r["successes"]), int(r["n"]))
    if set(out) != set(METHODS.values()):
        raise ValueError(f"{path}: {FINAL_ROUND} methods {sorted(out)}")
    return out


def final_round() -> dict[str, dict]:
    """Per task: {method: (successes, n)} and the HiL-IDQL+Mulligan - HG-DAgger margin (pp)."""
    out = {}
    for task, path in HEADLINES.items():
        counts = _final_round(path)
        (s_full, n_full), (s_base, n_base) = counts["HiL-IDQL+Mulligan"], counts["HG-DAgger"]
        out[task] = dict(counts=counts, margin_pp=100 * (s_full / n_full - s_base / n_base))
    return out


def cable_clip_success() -> dict[str, float]:
    """Final-round Cable clip success rate (%) per method: mean 0-2 clip score / 2."""
    return {m: 100 * s / n for m, (s, n) in _final_round(CABLE_CLIP).items()}


def main() -> None:
    for task, result in final_round().items():
        counts = "  ".join(f"{m} {s}/{n}" for m, (s, n) in result["counts"].items())
        print(f"{TASK_NAMES[task]:6s} R5 full success: {counts}")
        print(f"       HiL-IDQL+Mulligan - HG-DAgger: {result['margin_pp']:+.0f} pp")
    clip = "  ".join(f"{m} {v:.0f}%" for m, v in cable_clip_success().items())
    print(f"Cable  R5 clip success rate: {clip}")


if __name__ == "__main__":
    main()
