"""Recompute the paper's headline numbers from the pinned public HF release.

Every input is a public ``mulligan/*`` repo read anonymously at its ``release/revisions.json``
pin, plus the committed release manifests. Nothing is read from W&B, S3 or the lab cluster.

What is recomputed, from per-episode records:

- real task points (46): per-round success counts from ``meta/episode_provenance.parquet`` of the
  round datasets (Cable: clip seats from ``results.json`` and ``.label_history.jsonl``);
- simulation points (64): per-seed success counts from the evaluation bundles' per-state
  Parquet files (``--no-sim-recount`` reads the bundle's ``meta/mainline-evaluations.json``
  index instead);
- the 2,550 / 2,850 episode totals over the 13 non-screen round datasets, with a
  per-dataset integrity check of all 15 against ``release/round-datasets.json``;
- the 988 / 1,009 Mulligan collection successes from the ``*-dagger-mixed`` ledgers.

Two separate checks follow, so presentation rounding can never hide a count mismatch:

1. counts: recomputed integer counts equal every recorded count (the round lock, ``models.json``,
   the bundle index and the frozen paper CSVs), exactly;
2. rates: each stored rate and interval (all 110 points of ``release/paper-results.json``, the
   bundles' ``meta/mainline-headline.csv`` and the frozen CSVs) equals the paper's rule
   applied to those counts, within 1e-12 where the rule reproduces the stored value, else within
   half a unit in the last stored digit (listed as such in the report).

    python -m mulligan.release.verify_results [--paper-evidence DIR] [--out report.json]

``--paper-evidence`` (default ``$MULLIGAN_PAPER_EVIDENCE``) is a local mirror of the
``mulligan/paper-evidence`` dataset; without it the frozen paper CSVs are read from the dataset at
its pin. Progress goes to stderr, one line per stage; stdout gets the per-group check summary
and a one-line headline. Exits non-zero on any failure.

The summary counts checks, not points: the ``real`` group has a single-checkpoint check for each
of the 46 real points plus one Cable ``results.json`` outcome check, so it reads 47.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from scipy import stats  # noqa: E402

from mulligan.release.download import RELEASE_DIR, pinned_revision, repo_type  # noqa: E402
from mulligan.release.round_counts import (  # noqa: E402
    EXPECTED_TOTALS,
    check_round_dataset,
    episode_totals,
    load_lock,
)

EXACT = 1e-12
Z95 = float(stats.norm.ppf(0.975))
T95_DF4 = float(stats.t.ppf(0.975, 4))
EXPECTED_COLLECTION = {"successes": 988, "episodes": 1009}
# The 440 `sim_evaluations` of the released sim checkpoints: headline seeds in the bundles'
# `meta/mainline-evaluations.json`, the rest in their full `meta/evaluations.json`.
SIM_MODEL_EVALUATIONS = {"mainline": 290, "other": 150}
REAL_TASKS = {
    "marker_d2": "real-marker-d2",
    "square_d2": "real-square-d2",
    "routing_d2": "real-routing-d2",
}
SIM_BUNDLES = {
    "square_narrow": "mulligan/sim-square-narrow-r00-r03-eval",
    "square_broad": "mulligan/sim-square-broad-r00-r03-eval",
}
# public provenance policy_key (round suffix removed) -> paper-results method
POLICY_METHOD = {
    "hg_dagger": "baseline",
    "hg_dagger_mulligan": "mulligan_with_cf",
    "hil_idql_mulligan": "final_iql",
}
MULLIGAN_ARM = "mulligan_sobol"
# The appendix packages' input locks pin every frozen paper CSV compared here.
APPENDIX_DIR = RELEASE_DIR.parent / "paper" / "appendix"
EVIDENCE_DIRS = {"marker_d2": "marker", "square_d2": "square", "routing_d2": "routing"}


# ---------------------------------------------------------------------------------------------
# statistics (the rules the paper used)


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval, clipped to [0, 1] (z=1: one standard error; Z95: 95%)."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mean_se(values: list[float]) -> tuple[float, float]:
    """Mean and standard error (sample standard deviation, ddof=1) of a list."""
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n)


def seed_percent(k: int, n: int) -> float:
    """Per-seed success rate in percent at the four decimals the bundles store."""
    return round(100 * k / n, 4)


def sim_rule(seeds: list[tuple[int, int]]) -> tuple[float, float, float]:
    """Rate and Student-t 95% interval over five seeds, from the 4-decimal per-seed percents.

    The stored simulation headline is the mean of the per-seed rates *as stored* (percent,
    four decimals), not the exact pooled mean; this reproduces every stored value to 1e-12.
    """
    rates = [seed_percent(k, n) / 100 for k, n in seeds]
    m, se = mean_se(rates)
    return m, m - T95_DF4 * se, m + T95_DF4 * se


def cable_rule(scores: list[int]) -> tuple[float, float, float]:
    """Cable task progress: clip seats / (2 x episodes) with a one-standard-error interval."""
    fractions = [s / 2 for s in scores]
    m, se = mean_se(fractions)
    return m, m - se, m + se


# ---------------------------------------------------------------------------------------------
# report


@dataclass
class Report:
    checks: list[dict] = field(default_factory=list)
    half_unit: list[dict] = field(default_factory=list)
    # recomputed headline quantities, for the headline line and the JSON report
    facts: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)

    def add(self, group: str, name: str, ok: bool, detail: str = "", **extra) -> None:
        self.checks.append(
            {"group": group, "name": name, "status": "ok" if ok else "FAIL", "detail": detail}
            | extra
        )

    def skip(self, group: str, name: str, reason: str) -> None:
        self.checks.append({"group": group, "name": name, "status": "skipped", "detail": reason})

    def count(self, group: str, name: str, got: int, want: int, source: str) -> None:
        self.add(group, name, got == want, f"recomputed {got}, {source} {want}")

    def rate(
        self,
        group: str,
        name: str,
        got: float,
        stored: float,
        rule: str,
        decimals: int | None = None,
    ) -> None:
        """Rate check: 1e-12 when the rule reproduces the stored value, else half a unit."""
        diff = abs(got - stored)
        if diff <= EXACT:
            self.add(group, name, True, f"{rule}: |diff| {diff:.1e} <= 1e-12")
            return
        if decimals is None:
            self.add(group, name, False, f"{rule}: {got!r} vs stored {stored!r} (diff {diff:.3e})")
            return
        tol = 0.5 * 10.0**-decimals
        ok = diff <= tol + EXACT
        self.add(group, name, ok, f"{rule}: |diff| {diff:.3e} vs half-unit {tol:g}")
        self.half_unit.append(
            {"point": name, "rule": rule, "stored": stored, "recomputed": got, "tolerance": tol}
        )

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "FAIL"]

    def summary(self) -> dict:
        by = defaultdict(Counter)
        for c in self.checks:
            by[c["group"]][c["status"]] += 1
        return {g: dict(v) for g, v in sorted(by.items())}


class Progress:
    """Stage lines with the elapsed wall clock; silent when ``stream`` is None."""

    def __init__(self, stages: int, stream=None):
        self.stages = stages
        self.stream = stream
        self.done = 0
        self.t0 = time.monotonic()

    def _print(self, text: str) -> None:
        if self.stream is not None:
            elapsed = time.monotonic() - self.t0
            print(f"{text} [{elapsed:.0f} s]", file=self.stream, flush=True)

    def stage(self, text: str) -> None:
        self.done += 1
        self._print(f"[{self.done}/{self.stages}] {text}")

    def note(self, text: str) -> None:
        self._print(f"      {text}")

    def every(self, i: int, n: int, text: str, step: int) -> None:
        """Note ``text i/n`` at every ``step``-th item and at the last one."""
        if i == n or i % step == 0:
            self.note(f"{text} {i}/{n}")


def headline(report: Report) -> str:
    """One plain-language line: the recomputed quantities and whether their checks passed."""
    f = report.facts
    failed = Counter(c["group"] for c in report.failures)

    def verdict(*groups: str) -> str:
        n = sum(failed[g] for g in groups)
        return "match" if not n else f"MISMATCH ({n} failed checks)"

    k, n = f["collection"]
    parts = [
        f"{f['result_points']} result points ({f['real_points']} real, {f['sim_points']} "
        f"simulation): counts {verdict('counts', 'real')}, rates {verdict('parity', 'frozen-csv')}",
        f"{f['episodes']:,} evaluation episodes ({f['episodes_with_no_cf']:,} with the no-CF "
        f"ablation): {verdict('round-datasets')}",
        f"{k:,}/{n:,} Mulligan collection successes: {verdict('collection')}",
        f"{f['sim_evaluations']:,} simulation evaluations of released checkpoints"
        f"{'' if f['sim_recount'] else ' (bundle index, not recounted)'}: {verdict('sim')}",
    ]
    skipped = [c for c in report.checks if c["status"] == "skipped"]
    parts += [f"skipped: {c['name']} ({c['detail']})" for c in skipped]
    total = len(report.checks) - len(skipped)
    if report.failures:
        parts.append(f"{len(report.failures):,} of {total:,} checks FAILED (see the FAIL lines)")
    else:
        parts.append(f"all {total:,} checks passed")
    return "; ".join(parts)


# ---------------------------------------------------------------------------------------------
# inputs


class Hub:
    """Anonymous reads of released repos at their pinned revisions."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = str(cache_dir) if cache_dir else None

    def path(self, repo: str, filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo,
                filename,
                repo_type=repo_type(repo),
                revision=pinned_revision(repo),
                token=False,
                cache_dir=self.cache_dir,
            )
        )

    def json(self, repo: str, filename: str):
        return json.loads(self.path(repo, filename).read_text())

    def parquet(self, repo: str, filename: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(repo, filename))


def load_release(name: str):
    return json.loads((RELEASE_DIR / name).read_text())


# ---------------------------------------------------------------------------------------------
# round datasets: integrity and the 2,550 rule


def round_datasets(
    hub: Hub, report: Report, progress: Progress | None = None
) -> dict[str, tuple[dict, pd.DataFrame]]:
    progress = progress or Progress(0)
    lock, lock_sha = load_lock()
    out = {}
    rows = []
    for i, ds in enumerate(lock["datasets"], 1):
        rid = ds["id"]
        meta = hub.json(rid, "meta/round_dataset.json")
        prov = hub.parquet(rid, "meta/episode_provenance.parquet")
        fails = check_round_dataset(ds, meta, prov, lock_sha)
        report.add(
            "round-datasets", f"{rid} integrity", not fails, "; ".join(fails) or "matches lock"
        )
        out[rid] = (ds, prov)
        rows.append((rid, meta.get("task"), meta.get("kind"), Counter(prov["role"])))
        progress.every(i, len(lock["datasets"]), "round datasets", 5)
    totals = episode_totals(rows)
    for key in (
        "counted",
        "counted_plus_no_cf",
        "counted_incl_screen",
        "counted_plus_no_cf_incl_screen",
    ):
        report.count("round-datasets", f"total {key}", totals[key], EXPECTED_TOTALS[key], "paper")
    for task, want in EXPECTED_TOTALS["by_task"].items():
        got = totals["by_task"].get(task, {}).get("counted", 0)
        report.count("round-datasets", f"{task} counted", got, want, "paper")
    report.add(
        "round-datasets",
        "non-screen datasets",
        totals["datasets"] == 13,
        f"{totals['datasets']} datasets with kind != screen are summed",
    )
    report.facts |= {
        "round_datasets": len(lock["datasets"]),
        "episodes": totals["counted"],
        "episodes_with_no_cf": totals["counted_plus_no_cf"],
    }
    return out


# ---------------------------------------------------------------------------------------------
# real task points


def cable_marks(hub: Hub, repo: str) -> dict[int, int]:
    """Clip marks per source episode: the latest human label, else the eval-time record."""
    results = hub.json(repo, "results.json")
    marks = {r["episode_index"]: len(r.get("subtask_frames") or []) for r in results["rollouts"]}
    history = [
        json.loads(line)
        for line in hub.path(repo, ".label_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for h in sorted(history, key=lambda h: h["ts"]):
        if h["label_kind"] == "outcome" and "subtask_frames" in h["payload"]:
            marks[h["episode_index"]] = len(h["payload"]["subtask_frames"])
    return marks


def real_points(hub: Hub, rounds: dict, report: Report) -> dict[tuple, dict]:
    """(task, round, method) -> recomputed counts (and Cable clip scores)."""
    lock = load_release("round-datasets.json")
    points = {}
    for ds in lock["datasets"]:
        if ds["kind"] == "screen":
            continue
        rid = ds["id"]
        _, prov = rounds[rid]
        task = next(t for t, v in REAL_TASKS.items() if v == ds["task"])
        counted = prov[prov["role"] == "counted"]
        marks = cable_marks(hub, rid) if ds["kind"] == "cable" else None
        if marks is not None:
            outcomes = {
                r["episode_index"]: r["outcome"] for r in hub.json(rid, "results.json")["rollouts"]
            }
            bad = [
                e.source_episode_index
                for e in prov.itertuples()
                if outcomes[e.source_episode_index] != e.outcome
            ]
            report.add(
                "real",
                f"{rid} results.json outcomes",
                not bad,
                f"{len(bad)} disagree with provenance",
            )
        for key, group in counted.groupby("policy_key"):
            if ds["kind"] == "cable":  # one dataset for all rounds: policy_key ends in _rN
                base, _, r = key.rpartition("_r")
                rnd = int(r)
            else:
                base, rnd = key, int(ds["round"][1:])
            method = POLICY_METHOD.get(base)
            if method is None:
                report.add("real", f"{rid} {key}", False, "unknown policy_key")
                continue
            if method == "final_iql" and task == "marker_d2":
                method = "final_marker_iql"
            ckpts = {
                (r.actor_model_id, str(r.critic_model_id), str(r.num_action_samples))
                for r in group.itertuples()
            }
            report.add(
                "real",
                f"{task} R{rnd} {method} one checkpoint",
                len(ckpts) == 1,
                f"{len(ckpts)} checkpoints pooled",
            )
            pt = {
                "dataset": rid,
                "sessions": sorted(group.session_id.unique()),
                "successes": int(group.success.sum()),
                "episodes": len(group),
                "policies": sorted(group.policy.unique()),
            }
            if marks is not None:
                pt["scores"] = [
                    2 if e.success else min(marks[e.source_episode_index], 1)
                    for e in group.itertuples()
                ]
                pt["clip_seats"] = sum(pt["scores"])
            points[(task, f"R{rnd}", method)] = pt
    return points


def check_real_counts(points: dict, report: Report) -> None:
    """Counts vs the lock's per-policy records and models.json deployments."""
    lock = load_release("round-datasets.json")
    per_policy = {}
    for ds in lock["datasets"]:
        for p in ds["policies"]:
            if p.get("role") == "counted":
                per_policy[(ds["id"], p["session_id"], p["name"])] = (p["successes"], p["episodes"])
    deployments = defaultdict(lambda: [0, 0])
    for m in load_release("models.json")["models"]:
        for c in m["checkpoints"]:
            for d in c.get("deployments", []):
                if c["kind"] == "dp-actor" and d["critic"] is not None:
                    continue
                deployments[(d["dataset"], d["session"], d["policy"])][0] += d["successes"]
                deployments[(d["dataset"], d["session"], d["policy"])][1] += d["episodes"]
    for (task, rnd, method), pt in sorted(points.items()):
        name = f"{task} {rnd} {method}"
        lock_k = lock_n = 0
        dep_k = dep_n = 0
        for s in pt["sessions"]:
            for pol in pt["policies"]:
                if (pt["dataset"], s, pol) in per_policy:
                    k, n = per_policy[(pt["dataset"], s, pol)]
                    lock_k, lock_n = lock_k + k, lock_n + n
                if (pt["dataset"], s, pol) in deployments:
                    k, n = deployments[(pt["dataset"], s, pol)]
                    dep_k, dep_n = dep_k + k, dep_n + n
        report.count("counts", f"{name} successes (lock)", pt["successes"], lock_k, "lock")
        report.count("counts", f"{name} episodes (lock)", pt["episodes"], lock_n, "lock")
        if dep_n:
            report.count(
                "counts",
                f"{name} successes (models.json)",
                pt["successes"],
                dep_k,
                "models.json deployments",
            )
            report.count(
                "counts",
                f"{name} episodes (models.json)",
                pt["episodes"],
                dep_n,
                "models.json deployments",
            )


# ---------------------------------------------------------------------------------------------
# simulation points


def sim_points(
    hub: Hub, recount: bool, report: Report, progress: Progress | None = None
) -> dict[tuple, list[tuple[int, int]]]:
    """(task, round, arm) -> five (successes, episodes), recounted from the per-state Parquet."""
    progress = progress or Progress(0)
    points = defaultdict(dict)
    evals = {}
    for task, repo in SIM_BUNDLES.items():
        index = hub.json(repo, "meta/mainline-evaluations.json")
        report.add(
            "sim", f"{repo} mainline results", len(index) == 160, f"{len(index)} seed results"
        )
        for i, row in enumerate(index, 1):
            if row["task"] != task:
                report.add("sim", f"{repo} task", False, f"row for {row['task']}")
                continue
            k, n = row["successes"], row["episodes"]
            key = (task, f"R{row['round']}", row["arm"])
            if recount:
                path = hub.path(repo, row["data"])
                sha_ok = hashlib.sha256(path.read_bytes()).hexdigest() == row["dataSha256"]
                frame = pd.read_parquet(path, columns=["success", "length"])
                rk, rn = int(frame["success"].sum()), len(frame)
                name = f"{task} {key[1]} {row['arm']} seed {row['seed']}"
                report.add("counts", f"{name} data sha256", sha_ok, row["data"])
                report.count(
                    "counts", f"{name} successes (index)", rk, k, "meta/mainline-evaluations.json"
                )
                report.count(
                    "counts", f"{name} episodes (index)", rn, n, "meta/mainline-evaluations.json"
                )
                report.count(
                    "counts",
                    f"{name} steps (index)",
                    int(frame["length"].sum()),
                    row["totalSteps"],
                    "meta/mainline-evaluations.json",
                )
                k, n = rk, rn
                progress.every(i, len(index), f"{repo} headline seed files", 40)
            points[key][row["seed"]] = (k, n)
            evals[(_checkpoint(row["artifact"]), row["seed"], row["numActionSamples"], repo)] = (
                k,
                n,
            )
    check_sim_models(evals, sim_other_evaluations(hub, report), hub, recount, report, progress)
    return {key: [seeds[s] for s in sorted(seeds)] for key, seeds in points.items()}


def _checkpoint(model_id: str) -> str:
    """Release id of a checkpoint as the bundles record it (``hf://`` optional)."""
    return model_id.removeprefix("hf://")


def sim_other_evaluations(hub: Hub, report: Report) -> dict[tuple, dict]:
    """(artifact, seed, N, bundle) -> row of each bundle's full ``meta/evaluations.json`` index.

    The index covers the evaluation cells outside the 64 headline points; a row is keyed by
    both its actor and its critic artifact.
    """
    rows: dict[tuple, dict] = {}
    for repo in SIM_BUNDLES.values():
        for row in hub.json(repo, "meta/evaluations.json"):
            for artifact in {row["actorArtifact"], row["criticArtifact"]}:
                key = (_checkpoint(artifact), row["seed"], row["numActionSamples"], repo)
                if key in rows:
                    report.add("sim", f"{repo} evaluations.json", False, f"duplicate key {key}")
                rows[key] = row
    return rows


def check_sim_models(
    evals: dict,
    other: dict,
    hub: Hub,
    recount: bool,
    report: Report,
    progress: Progress | None = None,
) -> None:
    """Seed counts vs the ``sim_evaluations`` recorded on the released checkpoints.

    Every recorded evaluation must be found: the headline seeds in the bundles' mainline
    index (recounted in ``sim_points``), the others in the full ``meta/evaluations.json``
    (recounted from the per-state Parquet it names when ``recount``).
    """
    progress = progress or Progress(0)
    recorded = [
        ((c["checkpoint"], e["seed"], e["num_action_samples"], e["dataset"]), m, c, e)
        for m in load_release("models.json")["models"]
        for c in m["checkpoints"]
        for e in c.get("sim_evaluations", [])
    ]
    n_other = sum(key not in evals and key in other for key, *_ in recorded)
    matched = Counter()
    for key, m, c, e in recorded:
        name = f"{m['repo']}/{c['subfolder']} N={e['num_action_samples']}"
        if key in evals:
            matched["mainline"] += 1
            k, n = evals[key]
        elif key in other:
            matched["other"] += 1
            k, n = recount_other(hub, e["dataset"], other[key], name, recount, report)
            if recount:
                progress.every(matched["other"], n_other, "other evaluation files", 50)
        else:
            matched["missing"] += 1
            report.add("sim", f"{name} seed {e['seed']}", False, "not in either index")
            continue
        report.count("counts", f"{name} successes (models.json)", k, e["successes"], "models.json")
        report.count("counts", f"{name} episodes (models.json)", n, e["episodes"], "models.json")
    for group, want in SIM_MODEL_EVALUATIONS.items():
        report.count(
            "sim",
            f"models.json sim_evaluations in the {group} index",
            matched[group],
            want,
            "expected",
        )
    report.count("sim", "models.json sim_evaluations unmatched", matched["missing"], 0, "expected")
    report.facts["sim_evaluations"] = matched["mainline"] + matched["other"]


def recount_other(
    hub: Hub, repo: str, row: dict, name: str, recount: bool, report: Report
) -> tuple[int, int]:
    """(successes, episodes) of one ``meta/evaluations.json`` row, recounted if ``recount``."""
    k, n = row["successes"], row["episodes"]
    if not recount:
        return k, n
    path = hub.path(repo, row["data"])
    sha_ok = hashlib.sha256(path.read_bytes()).hexdigest() == row["dataSha256"]
    report.add("counts", f"{name} seed {row['seed']} data sha256", sha_ok, row["data"])
    frame = pd.read_parquet(path, columns=["success"])
    rk, rn = int(frame["success"].sum()), len(frame)
    source = "meta/evaluations.json"
    report.count("counts", f"{name} seed {row['seed']} successes (index)", rk, k, source)
    report.count("counts", f"{name} seed {row['seed']} episodes (index)", rn, n, source)
    return rk, rn


def check_sim_headline_csv(hub: Hub, points: dict, report: Report) -> None:
    """The bundles' frozen `meta/mainline-headline.csv` (percent, four decimals)."""
    for task, repo in SIM_BUNDLES.items():
        csv = pd.read_csv(hub.path(repo, "meta/mainline-headline.csv"))
        csv = csv[csv.task_key == task]
        report.add(
            "frozen-csv", f"{repo} mainline-headline rows", len(csv) == 160, f"{len(csv)} rows"
        )
        for (t, rnd, arm), grp in csv.groupby(["task_key", "round", "series_key"]):
            seeds = points[(t, f"R{rnd}", arm)]
            name = f"{t} R{rnd} {arm}"
            grp = grp.sort_values("seed")
            for s, (k, n), stored in zip(grp.seed, seeds, grp.overall_sr):
                report.rate(
                    "frozen-csv",
                    f"{name} seed {s} overall_sr",
                    seed_percent(k, n),
                    float(stored),
                    "100*k/n rounded to 4 decimals",
                    decimals=4,
                )
            pct = [seed_percent(k, n) for k, n in seeds]
            m, se = mean_se(pct)
            report.rate(
                "frozen-csv",
                f"{name} mean_sr",
                round(m, 4),
                float(grp.mean_sr.iloc[0]),
                "mean of stored per-seed percents, 4 decimals",
                decimals=4,
            )
            report.rate(
                "frozen-csv",
                f"{name} se_sr",
                round(se, 4),
                float(grp.se_sr.iloc[0]),
                "ddof=1 SE of stored per-seed percents, 4 decimals",
                decimals=4,
            )


# ---------------------------------------------------------------------------------------------
# collection success (988 / 1,009)


def collection_repos() -> list[dict]:
    """The real ``*-dagger-mixed`` collection rows of ``release/datasets.json``."""
    return [
        r
        for r in load_release("datasets.json")["datasets"]
        if r["task"] in REAL_TASKS.values()
        and r["role"] == "raw-collection"
        and "-dagger-mixed" in r["repo"]
    ]


def collection(
    hub: Hub, report: Report, progress: Progress | None = None
) -> dict[tuple, tuple[int, int]]:
    """(task, round, arm) -> (credited successes, fresh attempts) from the protocol ledgers."""
    progress = progress or Progress(0)
    rows = collection_repos()
    out = defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows, 1):
        (rnd,) = r["model_rounds"]
        task = next(t for t, v in REAL_TASKS.items() if v == r["task"])
        text = hub.path(r["repo"], "meta/protocol_quota_ledger.jsonl").read_text()
        ledger = [json.loads(line) for line in text.splitlines() if line.strip()]
        for e in ledger:
            if e["is_counterfactual"]:
                continue
            credited = bool(e["success"]) and bool(e["quota_credit"])
            if bool(e["success"]) != bool(e["quota_credit"]):
                report.add(
                    "collection",
                    f"{r['repo']} ep {e['episode_index']}",
                    False,
                    "success != quota_credit",
                )
            out[(task, f"R{rnd}", e["arm_key"])][0] += credited
            out[(task, f"R{rnd}", e["arm_key"])][1] += 1
        progress.every(i, len(rows), "ledgers", 5)
    ours = {k: v for k, v in out.items() if k[2] == MULLIGAN_ARM}
    k = sum(v[0] for v in ours.values())
    n = sum(v[1] for v in ours.values())
    report.count(
        "collection", "Mulligan collection successes", k, EXPECTED_COLLECTION["successes"], "paper"
    )
    report.count(
        "collection", "Mulligan collection attempts", n, EXPECTED_COLLECTION["episodes"], "paper"
    )
    report.facts["collection"] = (k, n)
    if not ours:
        report.add("collection", "lowest Mulligan collection round", False, "no Mulligan rounds")
        return {key: tuple(v) for key, v in out.items()}
    worst = min(ours, key=lambda key: ours[key][0] / ours[key][1])
    wk, wn = ours[worst]
    report.add(
        "collection",
        "lowest Mulligan collection round",
        worst == ("routing_d2", "R4", MULLIGAN_ARM) and round(100 * wk / wn, 1) == 90.1,
        f"{worst[0]} {worst[1]}: {wk}/{wn} = {100 * wk / wn:.1f}% (paper: Cable R4 90.1%)",
    )
    return {key: tuple(v) for key, v in out.items()}


# ---------------------------------------------------------------------------------------------
# paper-results.json (110 points): rates


def check_parity(real: dict, sim: dict, report: Report) -> None:
    parity = load_release("paper-results.json")["points"]
    report.add("parity", "point count", len(parity) == 110, f"{len(parity)} points")
    report.facts["result_points"] = len(parity)
    seen = set()
    for p in parity:
        key = (p["task"], p["round"], p["method"])
        name = " ".join(key)
        seen.add(key)
        if p["task"] in REAL_TASKS:
            pt = real.get(key)
            if pt is None:
                report.add("parity", name, False, "no recomputed point")
                continue
            if p["task"] == "routing_d2":
                rate, lo, hi = cable_rule(pt["scores"])
                rule = "Cable clip seats / (2n), mean +- 1 SE (ddof=1) of per-episode score"
            else:
                rate = pt["successes"] / pt["episodes"]
                lo, hi = wilson(pt["successes"], pt["episodes"], 1.0)
                rule = "k/n, Wilson z=1"
        else:
            seeds = sim.get(key)
            if seeds is None or len(seeds) != 5:
                report.add("parity", name, False, f"{0 if seeds is None else len(seeds)} seeds")
                continue
            rate, lo, hi = sim_rule(seeds)
            rule = "mean of 4-decimal per-seed percents, Student-t 95% (df=4)"
        report.rate("parity", f"{name} rate", rate, p["rate"], rule)
        report.rate("parity", f"{name} lower", lo, p["lower"], rule)
        report.rate("parity", f"{name} upper", hi, p["upper"], rule)
    extra = sorted(set(real) - seen)
    report.add(
        "parity", "no unmatched real points", not extra, f"recomputed but not in parity: {extra}"
    )


# ---------------------------------------------------------------------------------------------
# frozen paper CSVs (paper-evidence mirror)


EVIDENCE_REPO = "mulligan/paper-evidence"


class Evidence:
    """The paper evidence: a local mirror, or ``mulligan/paper-evidence`` at its pin."""

    def __init__(self, mirror: Path | None, cache_dir: str | Path | None = None):
        self.mirror, self.cache_dir = mirror, cache_dir

    def read(self, path: str) -> bytes:
        if self.mirror is not None:
            return (self.mirror / path).read_bytes()
        from huggingface_hub import hf_hub_download

        from mulligan.release.download import pinned_revision

        local = hf_hub_download(
            EVIDENCE_REPO,
            path,
            repo_type="dataset",
            revision=pinned_revision(EVIDENCE_REPO),
            cache_dir=self.cache_dir,
        )
        return Path(local).read_bytes()


def evidence_csv(evidence: Evidence, package: str, path: str) -> pd.DataFrame:
    """A frozen paper CSV of the evidence, checked against ``package``'s ``inputs.json`` lock."""
    lock = json.loads((APPENDIX_DIR / package / "inputs.json").read_text())
    (row,) = (row for row in lock["files"] if row["path"] == path)
    data = evidence.read(path)
    if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise RuntimeError(f"{path}: differs from paper/appendix/{package}/inputs.json")
    return pd.read_csv(io.BytesIO(data))


def headline_csv(evidence: Evidence, task: str, name: str) -> pd.DataFrame:
    return evidence_csv(
        evidence, "real_results", f"real/results/{EVIDENCE_DIRS[task]}/headline/{task}_{name}.csv"
    )


def check_frozen_real(evidence: Evidence, real: dict, coll: dict, report: Report) -> None:
    for task in ("marker_d2", "square_d2"):
        df = headline_csv(evidence, task, "headline_sr")
        df = df[df.is_pooled]
        arms = {
            "baseline": "baseline",
            "mulligan_with_cf": "mulligan_with_cf",
            "final_iql": "final_iql",
            "final_marker_iql": "final_marker_iql",
        }
        for r in df.itertuples():
            method = arms.get(r.arm)
            key = (task, r.round, method)
            if key not in real:
                report.add("frozen-csv", f"{task} {r.round} {r.arm}", False, "no recomputed point")
                continue
            pt = real[key]
            name = f"{task}_headline_sr {r.round} {r.arm}"
            report.count(
                "counts", f"{name} successes", pt["successes"], int(r.successes), "frozen CSV"
            )
            report.count("counts", f"{name} n", pt["episodes"], int(r.n), "frozen CSV")
            lo, hi = wilson(pt["successes"], pt["episodes"], Z95)
            report.rate(
                "frozen-csv",
                f"{name} success_rate",
                pt["successes"] / pt["episodes"],
                r.success_rate,
                "k/n",
            )
            report.rate("frozen-csv", f"{name} wilson_lo", lo, r.wilson_lo, "Wilson 95%")
            report.rate("frozen-csv", f"{name} wilson_hi", hi, r.wilson_hi, "Wilson 95%")
    full = headline_csv(evidence, "routing_d2", "headline_sr")
    progress = headline_csv(evidence, "routing_d2", "headline_task_progress")
    for r in full[full.is_pooled].itertuples():
        pt = real.get(("routing_d2", r.round, r.arm))
        name = f"routing_d2_headline_sr {r.round} {r.arm}"
        if pt is None:
            report.add("frozen-csv", name, False, "no recomputed point")
            continue
        report.count("counts", f"{name} successes", pt["successes"], int(r.successes), "frozen CSV")
        report.count("counts", f"{name} n", pt["episodes"], int(r.n), "frozen CSV")
        lo, hi = wilson(pt["successes"], pt["episodes"], Z95)
        report.rate(
            "frozen-csv",
            f"{name} success_rate",
            pt["successes"] / pt["episodes"],
            r.success_rate,
            "k/n",
        )
        report.rate("frozen-csv", f"{name} wilson_lo", lo, r.wilson_lo, "Wilson 95%")
        report.rate("frozen-csv", f"{name} wilson_hi", hi, r.wilson_hi, "Wilson 95%")
    for r in progress[progress.is_pooled].itertuples():
        pt = real.get(("routing_d2", r.round, r.arm))
        name = f"routing_d2_headline_task_progress {r.round} {r.arm}"
        if pt is None:
            report.add("frozen-csv", name, False, "no recomputed point")
            continue
        report.count(
            "counts", f"{name} clip seats", pt["clip_seats"], int(r.successes), "frozen CSV"
        )
        report.count("counts", f"{name} seat slots", 2 * pt["episodes"], int(r.n), "frozen CSV")
        rate, lo, hi = cable_rule(pt["scores"])
        rule = "clip seats / (2n), mean +- 1 SE"
        report.rate("frozen-csv", f"{name} success_rate", rate, r.success_rate, rule)
        report.rate("frozen-csv", f"{name} lo (1 SE)", lo, r.wilson_lo, rule)
        report.rate("frozen-csv", f"{name} hi (1 SE)", hi, r.wilson_hi, rule)
    for task in REAL_TASKS:
        df = evidence_csv(
            evidence, "productivity", f"real/collection/{EVIDENCE_DIRS[task]}/collection.csv"
        )
        for r in df.itertuples():
            k, n = coll.get((task, r.round, r.arm_key), (None, None))
            name = f"{task} collection {r.round} {r.arm_key}"
            if k is None:
                report.add("frozen-csv", name, False, "no recomputed collection round")
                continue
            report.count("counts", f"{name} successes", k, int(r.successes), "frozen CSV")
            report.count("counts", f"{name} n", n, int(r.n), "frozen CSV")
            lo, hi = wilson(k, n, Z95)
            report.rate("frozen-csv", f"{name} success_rate", k / n, r.success_rate, "k/n")
            report.rate("frozen-csv", f"{name} wilson_lo", lo, r.wilson_lo, "Wilson 95%")
            report.rate("frozen-csv", f"{name} wilson_hi", hi, r.wilson_hi, "Wilson 95%")


# ---------------------------------------------------------------------------------------------


def verify(
    cache_dir: str | Path | None = None,
    paper_evidence: Path | None = None,
    sim_recount: bool = True,
    progress: Progress | None = None,
) -> Report:
    """Run every check; ``progress`` (default silent) gets one line per stage."""
    progress = progress or Progress(5)
    hub = Hub(cache_dir)
    report = Report()
    n_rounds = len(load_release("round-datasets.json")["datasets"])
    progress.stage(f"round datasets: downloading and checking {n_rounds} pinned repos")
    rounds = round_datasets(hub, report, progress)
    f = report.facts
    progress.note(
        f"{f['episodes']:,} counted episodes ({f['episodes_with_no_cf']:,} with the no-CF ablation)"
    )
    progress.stage("real points: pooling counted episodes per round and policy (Cable labels)")
    real = real_points(hub, rounds, report)
    check_real_counts(real, report)
    f["real_points"] = len(real)
    progress.note(f"{len(real)} real points")
    progress.stage(
        f"simulation: {len(SIM_BUNDLES)} evaluation bundles, "
        + ("recounting every seed from its per-state Parquet" if sim_recount else "bundle index")
    )
    sim = sim_points(hub, sim_recount, report, progress)
    f["sim_points"] = len(sim)
    f["sim_recount"] = sim_recount
    progress.note(f"{len(sim)} simulation points, {f['sim_evaluations']} checkpoint evaluations")
    progress.stage(f"collection: {len(collection_repos())} protocol ledgers")
    coll = collection(hub, report, progress)
    progress.note("{:,}/{:,} Mulligan collection successes".format(*f["collection"]))
    progress.stage("rates: paper-results.json, bundle headline CSVs, frozen paper CSVs")
    check_parity(real, sim, report)
    check_sim_headline_csv(hub, sim, report)
    check_frozen_real(Evidence(paper_evidence, cache_dir), real, coll, report)
    run = sum(c["status"] != "skipped" for c in report.checks)
    progress.note(f"{run:,} checks run, {len(report.failures):,} failed")
    report.points = {
        "real": {
            " ".join(k): {x: v for x, v in pt.items() if x != "scores"}
            for k, pt in sorted(real.items())
        },
        "sim": {" ".join(k): v for k, v in sorted(sim.items())},
        "collection": {" ".join(k): v for k, v in sorted(coll.items())},
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache-dir", default=os.environ.get("MULLIGAN_HF_CACHE"), help="huggingface_hub cache_dir"
    )
    ap.add_argument(
        "--paper-evidence",
        type=Path,
        default=os.environ.get("MULLIGAN_PAPER_EVIDENCE"),
        help="local mirror of mulligan/paper-evidence (default: the dataset at its pin)",
    )
    ap.add_argument(
        "--no-sim-recount",
        action="store_true",
        help="use the bundle index instead of the per-state Parquet",
    )
    ap.add_argument("--out", type=Path, help="write the full report as JSON")
    args = ap.parse_args(argv)
    evidence = Path(args.paper_evidence) if args.paper_evidence else None
    report = verify(
        args.cache_dir,
        evidence,
        sim_recount=not args.no_sim_recount,
        progress=Progress(5, sys.stderr),
    )
    summary = report.summary()
    line = headline(report)
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "headline": line,
                    "summary": summary,
                    "facts": report.facts,
                    "half_unit_points": report.half_unit,
                    "checks": report.checks,
                    "points": report.points,
                },
                indent=1,
            )
            + "\n"
        )
    print(json.dumps(summary))
    print(f"half-unit tolerance points: {len(report.half_unit)}")
    for c in report.checks:
        if c["status"] == "skipped":
            print(f"SKIPPED {c['group']}: {c['name']} ({c['detail']})")
    for c in report.failures:
        print(f"FAIL {c['group']}: {c['name']}: {c['detail']}", file=sys.stderr)
    sys.stderr.flush()
    print(line, flush=True)
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
