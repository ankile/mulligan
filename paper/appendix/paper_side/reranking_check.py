"""Check the appendix's reranking settings and frozen diagnostic summaries (app:rerank-n).

Reads the release headline CSVs, verifies the per-task, per-round N of the ten reranked
real-world settings (parsed from the source policy names) against the frozen
``paper/data/real/reranking/release_settings.csv``, checks the two frozen summaries in that
folder, and checks every input against the hashes recorded in ``input_hashes.json`` when the
manuscript was checked.

    python -m paper.appendix.paper_side.reranking_check
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data/real"
HERE = DATA / "reranking"
SOURCES = {
    "marker_d2": DATA / "marker/marker_d2_headline_sr.csv",
    "square_d2": DATA / "square/square_d2_headline_sr.csv",
    "routing_d2": DATA / "routing/routing_d2_headline_sr_full_success.csv",
}
SWEEP = HERE / "square_r5_offline_nsweep.csv"
LATENCY = HERE / "square_r3_latency_summary.csv"
EXPECTED = {
    "marker_d2": {"R2": 16, "R3": 16, "R4": 16, "R5": 32},
    "square_d2": {"R3": 16, "R4": 16, "R5": 32},
    "routing_d2": {"R3": 32, "R4": 32, "R5": 32},
}


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_input_hashes() -> None:
    recorded = json.loads((HERE / "input_hashes.json").read_text())
    paths = {*SOURCES.values(), SWEEP, LATENCY}
    checked = set()
    for key, digest in recorded.items():
        path = DATA / key
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"{path}: sha256 differs from input_hashes.json[{key!r}]")
        checked.add(path)
    assert checked == paths, sorted(map(str, paths ^ checked))


def release_settings() -> list[dict]:
    settings = []
    for task, path in SOURCES.items():
        rows = [
            row
            for row in read(path)
            if row["is_pooled"] == "True" and row["arm"].startswith("final_")
        ]
        assert len(rows) == len(EXPECTED[task])
        actual = {}
        for row in rows:
            n = int(re.search(r"_n(16|32)$", row["source_policy_name"])[1])
            assert row["round"] not in actual
            actual[row["round"]] = n
            settings.append(
                dict(
                    task=task,
                    round=row["round"],
                    candidates=n,
                    model_id=row["model_id"],
                    source_policy_name=row["source_policy_name"],
                )
            )
        assert actual == EXPECTED[task], (task, actual)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(settings[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(settings)
    if stream.getvalue() != (HERE / "release_settings.csv").read_text():
        raise AssertionError(f"settings differ from {HERE / 'release_settings.csv'}")
    return settings


def nsweep_deltas() -> dict[str, float]:
    """Pooled AUROC change from N = 16 to N = 32 per Nut R5 candidate-screen critic."""
    sweep = read(SWEEP)
    assert len(sweep) == len({(row["arm"], row["k"]) for row in sweep}) == 28
    deltas = {}
    for arm in sorted({row["arm"] for row in sweep}):
        part = {int(row["k"]): row for row in sweep if row["arm"] == arm}
        assert set(part) == {2, 4, 8, 12, 16, 24, 32}
        assert all(int(row["n_resamples"]) == (1 if k == 32 else 20) for k, row in part.items())
        deltas[arm] = float(part[32]["pooled_mean"]) - float(part[16]["pooled_mean"])
        assert abs(deltas[arm]) < 0.003
    return deltas


def max_latency_ms() -> float:
    latency = {row["component"]: row for row in read(LATENCY) if row["scope"] == "all_chunks"}
    assert len(latency) == 4 and all(int(row["n"]) == 797 for row in latency.values())
    total = latency["time_total_ms"]
    assert [float(total[key]) for key in ("median_ms", "p99_ms", "max_ms")] == [39.1, 45.08, 48.75]
    return float(total["max_ms"])


def main() -> None:
    check_input_hashes()
    settings = release_settings()
    for arm, delta in nsweep_deltas().items():
        print(f"{arm}: N=16 to N=32 pooled AUROC change {delta:+.6f}")
    print(
        f"{len(settings)} release critic settings verified; maximum recorded interval = "
        f"{max_latency_ms() / 400 * 100:.4f}% of 400 ms."
    )


if __name__ == "__main__":
    main()
