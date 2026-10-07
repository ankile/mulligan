"""`mulligan.release.verify_results`: screen rules, the report output and the HF run."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from mulligan.release import round_counts
from mulligan.release import verify_results as vr

RELEASE = Path(__file__).resolve().parents[2] / "release"


def parity_point(task: str, rnd: str, method: str) -> dict:
    points = json.loads((RELEASE / "paper-results.json").read_text())["points"]
    (p,) = [p for p in points if (p["task"], p["round"], p["method"]) == (task, rnd, method)]
    return p


def test_sim_rule_reproduces_the_stored_broad_r0_value():
    # Example: exact mean 0.50892, stored 0.5089197999999999 (about 2e-7 off).
    seeds = [(k, 30000) for k in (15028, 16092, 14688, 15457, 15073)]
    p = parity_point("square_broad", "R0", "human_baseline_n1")
    exact = sum(k / n for k, n in seeds) / 5
    assert abs(exact - 0.50892) < 1e-15 and abs(exact - p["rate"]) > 1e-7
    rate, lo, hi = vr.sim_rule(seeds)
    assert abs(rate - p["rate"]) <= vr.EXACT
    assert abs(lo - p["lower"]) <= vr.EXACT and abs(hi - p["upper"]) <= vr.EXACT


def test_wilson_one_se_and_95():
    p = parity_point("marker_d2", "R0", "baseline")  # 10/50
    lo, hi = vr.wilson(10, 50, 1.0)
    assert abs(lo - p["lower"]) <= vr.EXACT and abs(hi - p["upper"]) <= vr.EXACT
    lo95, hi95 = vr.wilson(10, 50, vr.Z95)  # frozen paper CSV value
    assert abs(lo95 - 0.11243750015776111) <= vr.EXACT
    assert abs(hi95 - 0.33037105932225419) <= vr.EXACT


def test_cable_rule_is_clip_progress_with_one_se():
    # R0 HG-DAgger: one full success (2 seats), nine single seats, 40 zero.
    p = parity_point("routing_d2", "R0", "baseline")
    rate, lo, hi = vr.cable_rule([2] + [1] * 9 + [0] * 40)
    assert abs(rate - p["rate"]) <= vr.EXACT
    assert abs(lo - p["lower"]) <= vr.EXACT and abs(hi - p["upper"]) <= vr.EXACT


def test_rate_check_exact_then_half_unit():
    r = vr.Report()
    r.rate("g", "exact", 0.25, 0.25, "rule")
    r.rate("g", "off", 0.25, 0.2500001, "rule")
    r.rate("g", "half-unit ok", 70.79752, 70.7975, "rule", decimals=4)
    r.rate("g", "half-unit bad", 70.7976, 70.7975, "rule", decimals=4)
    assert [c["status"] for c in r.checks] == ["ok", "FAIL", "ok", "FAIL"]
    assert [h["point"] for h in r.half_unit] == ["half-unit ok", "half-unit bad"]


def test_counts_are_exact():
    r = vr.Report()
    r.count("counts", "a", 47, 47, "lock")
    r.count("counts", "b", 47, 48, "lock")
    assert [c["status"] for c in r.checks] == ["ok", "FAIL"]


def test_frozen_real_records_missing_points_as_failures(tmp_path, monkeypatch):
    """A frozen routing row without a recomputed point counts as a failure; the verifier
    still writes its report."""
    sr_cols = "is_pooled,arm,round,successes,n,success_rate,wilson_lo,wilson_hi\n"
    files = {
        "real/results/marker/headline/marker_d2_headline_sr.csv": sr_cols,
        "real/results/square/headline/square_d2_headline_sr.csv": sr_cols,
        "real/results/routing/headline/routing_d2_headline_sr.csv": sr_cols
        + "True,mulligan_no_cf,R1,5,10,0.5,0.3,0.7\n",
        "real/results/routing/headline/routing_d2_headline_task_progress.csv": sr_cols
        + "True,mulligan_no_cf,R1,9,20,0.45,0.4,0.5\n",
    }
    for task in ("marker", "square", "routing"):
        files[f"real/collection/{task}/collection.csv"] = (
            "round,arm_key,successes,n,success_rate,wilson_lo,wilson_hi\n"
        )
    locks = {"real_results": [], "productivity": []}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
        row = dict(path=rel, sha256=hashlib.sha256(text.encode()).hexdigest(), size=len(text))
        locks["productivity" if "/collection/" in rel else "real_results"].append(row)
    for package, rows in locks.items():
        (tmp_path / "appendix" / package).mkdir(parents=True)
        (tmp_path / "appendix" / package / "inputs.json").write_text(json.dumps({"files": rows}))
    monkeypatch.setattr(vr, "APPENDIX_DIR", tmp_path / "appendix")
    report = vr.Report()
    vr.check_frozen_real(vr.Evidence(tmp_path), {}, {}, report)
    assert [c["name"] for c in report.failures] == [
        "routing_d2_headline_sr R1 mulligan_no_cf",
        "routing_d2_headline_task_progress R1 mulligan_no_cf",
    ]


def test_collection_without_mulligan_rounds_records_a_failure(monkeypatch):
    monkeypatch.setattr(vr, "collection_repos", lambda: [])
    report = vr.Report()
    assert vr.collection(None, report) == {}
    assert "lowest Mulligan collection round" in [c["name"] for c in report.failures]


def test_round_dataset_integrity_detects_drift():
    lock = json.loads((RELEASE / "round-datasets.json").read_text())
    ds = next(d for d in lock["datasets"] if d["id"] == "mulligan/real-marker-d2-r00-eval")
    prov = pd.DataFrame(
        [
            {
                "episode_index": i,
                "session_id": e["session_id"],
                "source_episode_index": e["source_episode_index"],
                "role": e["role"],
                "success": e["success"],
                "policy": e["policy"],
            }
            for i, e in enumerate(ds["episodes"])
        ]
    )
    meta = {k: ds[k] for k in ("task", "round", "kind")} | {
        "counts": ds["counts"],
        "lock_sha256": "x",
    }
    assert vr.check_round_dataset(ds, meta, prov, "x") == []
    prov.loc[0, "success"] = not prov.loc[0, "success"]
    fails = vr.check_round_dataset(ds, meta, prov, "y")
    assert any("lock_sha256" in f for f in fails)
    assert any("role/success" in f for f in fails)
    assert any("by_session_policy" in f for f in fails)


def _report(failed_groups: tuple[str, ...] = (), skipped: bool = True) -> vr.Report:
    r = vr.Report()
    for group in ("counts", "real", "parity", "round-datasets", "collection", "sim"):
        r.add(group, f"{group} check", group not in failed_groups)
    if skipped:
        r.skip("frozen-csv", "paper headline and collection CSVs", "no --paper-evidence mirror")
    r.facts = {
        "result_points": 110,
        "real_points": 46,
        "sim_points": 64,
        "episodes": 2550,
        "episodes_with_no_cf": 2850,
        "collection": (988, 1009),
        "sim_evaluations": 440,
        "sim_recount": True,
    }
    return r


def test_headline_when_everything_matches():
    assert vr.headline(_report()) == (
        "110 result points (46 real, 64 simulation): counts match, rates match; "
        "2,550 evaluation episodes (2,850 with the no-CF ablation): match; "
        "988/1,009 Mulligan collection successes: match; "
        "440 simulation evaluations of released checkpoints: match; "
        "skipped: paper headline and collection CSVs (no --paper-evidence mirror); "
        "all 6 checks passed"
    )


def test_headline_names_what_failed():
    line = vr.headline(_report(("parity", "round-datasets"), skipped=False))
    assert "counts match, rates MISMATCH (1 failed checks)" in line
    assert "2,550 evaluation episodes (2,850 with the no-CF ablation): MISMATCH" in line
    assert "988/1,009 Mulligan collection successes: match" in line
    assert "skipped" not in line
    assert line.endswith("2 of 6 checks FAILED (see the FAIL lines)")


def test_headline_marks_the_bundle_index_run():
    r = _report()
    r.facts["sim_recount"] = False
    assert "released checkpoints (bundle index, not recounted): match" in vr.headline(r)


def test_progress_lines_count_stages_and_elapsed_time():
    out = io.StringIO()
    p = vr.Progress(2, out)
    p.stage("first")
    for i in range(1, 8):
        p.every(i, 7, "files", 3)
    p.stage("second")
    lines = out.getvalue().splitlines()
    assert [re.sub(r" \[\d+ s\]$", "", x) for x in lines] == [
        "[1/2] first",
        "      files 3/7",
        "      files 6/7",
        "      files 7/7",
        "[2/2] second",
    ]
    assert all(re.search(r" \[\d+ s\]$", x) for x in lines)
    silent = vr.Progress(2)
    silent.stage("nothing printed")


@pytest.mark.parametrize("failed", [(), ("counts",)])
def test_main_prints_the_headline_last_and_sets_the_exit_code(
    monkeypatch, capsys, tmp_path, failed
):
    monkeypatch.setattr(vr, "verify", lambda *a, **k: _report(failed))
    out = tmp_path / "report.json"
    assert vr.main(["--out", str(out)]) == (1 if failed else 0)
    stdout, stderr = capsys.readouterr()
    last = stdout.splitlines()[-1]
    assert last == vr.headline(_report(failed))
    assert ("FAIL counts: counts check" in stderr) == bool(failed)
    written = json.loads(out.read_text())
    assert written["headline"] == last and written["facts"]["result_points"] == 110


def _run(hf_cache: Path, recount: bool) -> vr.Report:
    evidence = os.environ.get("MULLIGAN_PAPER_EVIDENCE")
    return vr.verify(
        cache_dir=hf_cache,
        paper_evidence=Path(evidence) if evidence else None,
        sim_recount=recount,
    )


def test_episode_totals_rule_on_the_lock():
    """The 2,550 rule (one function for the paper and verify_results)."""
    lock, sha = round_counts.load_lock()
    assert sha == "ffabdce26d911002a9be21b04cfd5df0a329131c7c69b6ce62c18d54560fcb13"
    totals = round_counts.lock_totals(lock)
    for key, want in round_counts.EXPECTED_TOTALS.items():
        if key != "by_task":
            assert totals[key] == want, key
    for task, want in round_counts.EXPECTED_TOTALS["by_task"].items():
        assert totals["by_task"][task]["counted"] == want, task
    assert totals["datasets"] == 13 and len(totals["screen"]) == 2
    rows = [("a", "real-marker-d2", "round", Counter({"counted": 3, "no-cf-ablation": 1}))]
    rows.append(("h", "real-marker-d2", "screen", Counter({"counted": 5})))
    t = round_counts.episode_totals(rows)
    assert (t["counted"], t["counted_plus_no_cf"], t["counted_incl_screen"]) == (3, 4, 8)
    assert t["screen"] == ["h"]
    with pytest.raises(ValueError, match="unexpected episode roles"):
        round_counts.episode_totals([("x", "t", "round", Counter({"unknown": 1}))])


def test_sim_model_evaluations_all_accounted_for():
    """models.json records 440 sim evaluations: 290 headline seeds + 150 others."""
    n = sum(
        len(c.get("sim_evaluations", []))
        for m in json.loads((RELEASE / "models.json").read_text())["models"]
        for c in m["checkpoints"]
    )
    assert n == sum(vr.SIM_MODEL_EVALUATIONS.values()) == 440


@pytest.mark.network
def test_verify_results_from_public_hf(hf_cache):
    """[CI, network] counts then rates for all 110 points, 2,550 and 988/1,009 (bundle index)."""
    report = _run(hf_cache, recount=False)
    assert not report.failures, report.failures[:10]
    summary = report.summary()
    assert summary["parity"]["ok"] == 110 * 3 + 2
    assert summary["round-datasets"]["ok"] == 15 + 4 + 3 + 1
    assert summary["collection"]["ok"] == 3
    # 46 one-checkpoint checks (one per real point) + the Cable results.json outcome check
    assert summary["real"]["ok"] == 46 + 1
    f = report.facts
    assert (f["result_points"], f["real_points"], f["sim_points"]) == (110, 46, 64)
    assert (f["episodes"], f["episodes_with_no_cf"]) == (2550, 2850)
    assert f["collection"] == (988, 1009) and f["sim_evaluations"] == 440
    skipped = sum(c["status"] == "skipped" for c in report.checks)
    assert vr.headline(report).endswith(f"all {len(report.checks) - skipped:,} checks passed")


@pytest.mark.network
@pytest.mark.slow
def test_verify_results_recounts_sim_parquet(hf_cache):
    """[CI, network] the same, recounting every simulation seed from its per-state Parquet (~310 MB)."""
    report = _run(hf_cache, recount=True)
    assert not report.failures, report.failures[:10]
    assert report.summary()["counts"]["ok"] >= 320 * 4


@pytest.mark.network
@pytest.mark.slow
def test_arena_sim_statistics_regenerate_byte_identically(tmp_path, hf_cache):
    """The deployed Arena sim-statistics.json follows from the public bundles (~310 MB)."""
    from mulligan.release import arena_sim_stats

    arena = RELEASE.parent / "arena" / "data"
    out = tmp_path / "sim-statistics.json"
    assert (
        arena_sim_stats.main(
            [
                "--release",
                str(arena / "release.json"),
                "--output",
                str(out),
                "--cache",
                str(hf_cache),
            ]
        )
        == 0
    )
    assert out.read_bytes() == (arena / "sim-statistics.json").read_bytes()
