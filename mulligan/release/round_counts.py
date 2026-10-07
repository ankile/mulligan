"""The paper's held-out real-robot episode totals (2,550 / 2,850) and the round-dataset integrity rule.

The round-dataset lock ``release/round-datasets.json`` lists one entry per released real
evaluation dataset, each episode with a role (``counted``, ``no-cf-ablation``; excluded episodes
are not in the release). The paper's 2,550 sums the ``counted`` episodes over the datasets whose
kind is not ``screen`` (the two ``*-r05-screen`` datasets, superseded Marker and Nut R5
evaluations, are left out); adding the no-CF ablation arms gives 2,850.

One rule, two callers: ``paper.stats.count_eval_episodes`` (roles from the lock) and
``mulligan.release.verify_results`` (roles from the public ``meta/episode_provenance.parquet``,
kind from ``meta/round_dataset.json``).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from mulligan.release.download import RELEASE_DIR

LOCK_PATH = RELEASE_DIR / "round-datasets.json"
EXPECTED_TOTALS = {
    "counted": 2550,
    "counted_plus_no_cf": 2850,
    "by_task": {"real-marker-d2": 850, "real-square-d2": 950, "real-routing-d2": 750},
    "counted_incl_screen": 2950,
    "counted_plus_no_cf_incl_screen": 3250,
}
ROLES = ("counted", "no-cf-ablation")


def load_lock(path: Path = LOCK_PATH) -> tuple[dict, str]:
    """The round-dataset lock and its sha256 (the public round datasets cite it)."""
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def episode_totals(datasets: Iterable[tuple[str, str, str, Counter]]) -> dict:
    """The 2,550 rule over ``(repo, task, kind, role counts)`` rows.

    Returns the four headline totals, per-task ``counted`` (non-screen), the number of
    summed datasets and the ids of the screen ones.
    """
    totals = Counter()
    by_task = defaultdict(Counter)
    screen = []
    for repo, task, kind, roles in datasets:
        unexpected = set(roles) - set(ROLES)
        if unexpected:
            raise ValueError(f"{repo}: unexpected episode roles {sorted(unexpected)}")
        totals["counted_incl_screen"] += roles["counted"]
        totals["counted_plus_no_cf_incl_screen"] += roles["counted"] + roles["no-cf-ablation"]
        if kind == "screen":
            screen.append(repo)
            continue
        totals["counted"] += roles["counted"]
        totals["counted_plus_no_cf"] += roles["counted"] + roles["no-cf-ablation"]
        by_task[task]["counted"] += roles["counted"]
        by_task[task]["no_cf_ablation"] += roles["no-cf-ablation"]
        by_task[task]["datasets"] += 1
    return dict(totals) | {
        "datasets": sum(c["datasets"] for c in by_task.values()),
        "by_task": {task: dict(c) for task, c in by_task.items()},
        "screen": screen,
    }


def lock_totals(lock: dict) -> dict:
    """The rule applied to the lock's own per-episode roles, checked against its counts."""
    rows = []
    for ds in lock["datasets"]:
        roles = Counter(e["role"] for e in ds["episodes"])
        for role, key in (("counted", "counted"), ("no-cf-ablation", "no_cf_ablation")):
            if roles[role] != ds["counts"][key]:
                raise ValueError(f"{ds['id']}: {roles[role]} {role} episodes != counts.{key}")
        rows.append((ds["id"], ds["task"], ds["kind"], roles))
    return episode_totals(rows)


def check_round_dataset(lock_ds: dict, meta: dict, prov: pd.DataFrame, lock_sha: str) -> list[str]:
    """Integrity of one public round dataset against the lock."""
    fails = []
    rid = lock_ds["id"]
    for key in ("task", "round", "kind"):
        if meta.get(key) != lock_ds[key]:
            fails.append(
                f"{rid}: round_dataset.json {key}={meta.get(key)!r} != lock {lock_ds[key]!r}"
            )
    if meta.get("lock_sha256") != lock_sha:
        fails.append(f"{rid}: lock_sha256 {meta.get('lock_sha256')} != release lock {lock_sha}")
    lc, mc = lock_ds["counts"], meta.get("counts", {})
    for key in ("included", "counted", "no_cf_ablation", "excluded"):
        if mc.get(key) != lc[key]:
            fails.append(f"{rid}: meta counts.{key}={mc.get(key)} != lock {lc[key]}")
    if len(prov) != lc["included"]:
        fails.append(f"{rid}: provenance rows {len(prov)} != lock included {lc['included']}")
    if sorted(prov["episode_index"]) != list(range(len(prov))):
        fails.append(f"{rid}: provenance episode_index is not 0..{len(prov) - 1}")
    lock_eps = {
        (e["session_id"], int(e["source_episode_index"])): (e["role"], bool(e["success"]))
        for e in lock_ds["episodes"]
    }
    pub_eps = {
        (r.session_id, int(r.source_episode_index)): (r.role, bool(r.success))
        for r in prov.itertuples()
    }
    if len(pub_eps) != len(prov):
        fails.append(f"{rid}: duplicate (session_id, source_episode_index) in provenance")
    missing, extra = lock_eps.keys() - pub_eps.keys(), pub_eps.keys() - lock_eps.keys()
    if missing or extra:
        fails.append(f"{rid}: episode keys differ (missing {len(missing)}, extra {len(extra)})")
    diff = [k for k in lock_eps.keys() & pub_eps.keys() if lock_eps[k] != pub_eps[k]]
    if diff:
        fails.append(f"{rid}: {len(diff)} episodes differ in role/success, e.g. {sorted(diff)[:3]}")
    by_sp = defaultdict(lambda: [0, 0])
    for r in prov.itertuples():
        k = f"{r.session_id}|{r.policy}|{r.role}"
        by_sp[k][0] += 1
        by_sp[k][1] += int(bool(r.success))
    want = {k: [v["episodes"], v["successes"]] for k, v in lc["by_session_policy"].items()}
    if dict(by_sp) != want:
        fails.append(f"{rid}: by_session_policy differs from lock")
    return fails
