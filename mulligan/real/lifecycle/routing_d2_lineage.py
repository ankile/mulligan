"""Pinned Routing D2 lineage-eval configuration shared by thin analysis drivers.

Public rounds R0-R5 of the Cable task (``routing_d2``) are its collection rounds
R0/R2/R4/R6/R8/R9. All 15 final policies were evaluated on one shared 50-start
held-out manifest; the results live in ``mulligan/real-routing-d2-r00-r05-eval``.

``python -m mulligan.real.lifecycle.routing_d2_lineage`` rebuilds the six per-round held-out
tables and figures under ``outputs/real/routing_d2/`` from the pinned release. All 750
episodes are human-reviewed; outcomes and subtask marks come from the review record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mulligan.plotting.colors import METHOD_COLORS
from mulligan.real.lifecycle.heldout_eval import HeldoutEvalConfig
from mulligan.real.lifecycle.pinned_eval_snapshot import (
    SnapshotSpec,
    prepare_snapshot,
    run_heldout_from_prepared,
)
from mulligan.real.lifecycle.plotting import REPO_ROOT
from mulligan.release.download import pinned_revision

EVAL_REPO = "mulligan/real-routing-d2-r00-r05-eval"
EVAL_REVISION = pinned_revision(EVAL_REPO)
# Revision of the eval repo the pinned snapshot was taken from (provenance only; the
# pinned files below are identical at EVAL_REVISION).
SOURCE_EVAL_REVISION = "e86d47c0368f616b62a3c9ed9b73bc89ac3d42d5"
MANIFEST_PATH = (
    REPO_ROOT / "data/real/manifests/routing_d2/lineage/"
    "routing_d2_lineage_eval_heldout_independent_sobol.json"
)
MANIFEST_SHA256 = "41ff7dddc034352f351a602cfd2f45030df628ab9ee47032de6957cb4652f82d"
SNAPSHOT_NAME = "eval_snapshot"
DATA_ROOT = REPO_ROOT / "outputs/real/routing_d2" / SNAPSHOT_NAME / "data"
PLOT_ROOT = REPO_ROOT / "outputs/real/routing_d2" / SNAPSHOT_NAME / "plots"

# Every one of the 750 episodes is human-reviewed. The scoring rule: a rollout that
# seats only the right-most (second) clip has not achieved the first-clip subtask; the 31
# episodes noted "Second clip only" carry no subtask mark (label-history source.tool
# "rule:second-clip-only").
REVIEW_RULE = (
    "All 750 episodes human-reviewed. Seating only the right-most (second) clip does "
    "not count as the first clip: the 31 episodes noted 'Second clip only' carry no "
    "subtask mark."
)

SOURCE_SHA256 = {
    "results.json": "7ddba60ec2cb2b553e5d480d5547e1e7a027fd0b6ffa2f5def7785c811d39382",
    # Eval-time payload backed up at the first reconciliation; the baseline for the review's
    # cumulative effect (results.json's reconciliation block holds only the latest apply's
    # counts).
    "results_eval_time.json": "e062aa112ec5f4060a7be6643335d11c85ef0845ec1a4a3fdcf0b532aec50747",
    ".outcome_edit_progress.json": (
        "fb8218c54e841f61b575fe586955480ef926cde80a1b167d27855bee00c2bc76"
    ),
    ".label_history.jsonl": ("ed0bf5fc8c8e219763c890414cbf13ead4a786dbda253645faf2419a39827707"),
    "meta/initial_states_manifest.json": MANIFEST_SHA256,
}


@dataclass(frozen=True)
class RoutingD2Round:
    display_round: int
    original_round: int
    baseline: str
    mulligan: str
    iql: str | None = None

    @property
    def slug(self) -> str:
        return f"r{self.display_round}"

    @property
    def policy_names(self) -> tuple[str, ...]:
        names = [self.baseline, self.mulligan]
        if self.iql is not None:
            names.append(self.iql)
        return tuple(names)


ROUNDS = (
    RoutingD2Round(0, 0, "baseline_r00_dp", "mulligan_r00_dp"),
    RoutingD2Round(1, 2, "baseline_r02_dp", "mulligan_r02_dp"),
    RoutingD2Round(2, 4, "baseline_r04_dp", "mulligan_r04_dp"),
    RoutingD2Round(
        3,
        6,
        "baseline_r06_dp",
        "mulligan_r06_dp",
        "mulligan_r06_idql_n32",
    ),
    RoutingD2Round(
        4,
        8,
        "baseline_r08_dp",
        "mulligan_r08_dp",
        "mulligan_r08_idql_n32",
    ),
    RoutingD2Round(
        5,
        9,
        "baseline_r09_dp",
        "mulligan_r09_dp",
        "mulligan_r09_idql_n32",
    ),
)

POLICY_ROSTER = tuple(name for round_spec in ROUNDS for name in round_spec.policy_names)
if len(POLICY_ROSTER) != 15 or len(set(POLICY_ROSTER)) != 15:
    raise RuntimeError("Routing D2 roster must contain exactly 15 unique policies")

DISPLAY_LABELS = {
    "baseline": "HG-DAgger baseline",
    "mulligan": "HG-DAgger mulligan",
    "iql": "HiL-IDQL mulligan",
}
COLOR_KEYS = {
    "baseline": "uniform",
    "mulligan": "real_mulligan_with_cf",
    "iql": "real_iql_rerank",
}


def snapshot_spec() -> SnapshotSpec:
    return SnapshotSpec(
        repo_id=EVAL_REPO,
        revision=EVAL_REVISION,
        expected_file_sha256=SOURCE_SHA256,
    )


def round_by_slug(slug: str) -> RoutingD2Round:
    matches = [round_spec for round_spec in ROUNDS if round_spec.slug == slug]
    if len(matches) != 1:
        raise KeyError(f"unknown Routing D2 display round {slug!r}")
    return matches[0]


def heldout_config(round_spec: RoutingD2Round) -> HeldoutEvalConfig:
    names = round_spec.policy_names
    prefixes = {round_spec.baseline: "baseline", round_spec.mulligan: "mulligan"}
    if round_spec.iql is not None:
        prefixes[round_spec.iql] = "iql"
    labels = {name: DISPLAY_LABELS[prefixes[name]] for name in names}
    colors = {name: METHOD_COLORS[COLOR_KEYS[prefixes[name]]] for name in names}
    comparisons = [(round_spec.mulligan, round_spec.baseline)]
    if round_spec.iql is not None:
        comparisons.extend(
            ((round_spec.iql, round_spec.baseline), (round_spec.iql, round_spec.mulligan))
        )
    return HeldoutEvalConfig(
        task="routing_d2",
        eval_repo=EVAL_REPO,
        manifest_path=MANIFEST_PATH,
        expected_total_rounds=50,
        policy_names=names,
        policy_labels=labels,
        policy_colors=colors,
        policy_prefixes=prefixes,
        data_dir=DATA_ROOT / round_spec.slug,
        plot_path=PLOT_ROOT / round_spec.slug / f"routing_d2_{round_spec.slug}_heldout_eval.svg",
        plot_title=(
            f"Routing D2 {round_spec.slug.upper()} (collection round R{round_spec.original_round}) "
            "— shared 50-start held-out eval"
        ),
        plot_caption_prefix=(
            "Pinned snapshot; paired by manifest_idx; 750/750 human-reviewed; "
            "seating only the right-most clip scores 0"
        ),
        bootstrap_seed=20260915 + round_spec.display_round,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_initial_states_manifest_name=MANIFEST_PATH.name,
        comparison_pairs=tuple(comparisons),
        expect_pure_dp=round_spec.iql is None,
        outcome_overrides_filename=".outcome_edit_progress.json",
        subtask_scoring=True,
        score_threshold_labels={1: "first clip seated", 2: "full success"},
        extra_summary_fields={
            "analysis_line": "routing_d2",
            "source_physical_task": "routing_d2",
            "display_round": round_spec.slug.upper(),
            "original_round": f"R{round_spec.original_round}",
            "eval_revision": EVAL_REVISION,
            "source_revision": SOURCE_EVAL_REVISION,
            "label_source": "human_reviewed_750",
            "review_rule": REVIEW_RULE,
        },
    )


def run_rounds(*, validate_frames: bool = True) -> dict[str, dict]:
    """Per-round held-out battery of the six rounds from one pinned snapshot."""
    prepared = prepare_snapshot(
        snapshot_spec(),
        task="routing_d2",
        expected_policy_names=POLICY_ROSTER,
        expected_rollouts=750,
        expected_pairing_groups=50,
        validate_frames=validate_frames,
    )
    return {
        round_spec.slug: run_heldout_from_prepared(heldout_config(round_spec), prepared)
        for round_spec in ROUNDS
    }


def round_data_dir(round_spec: RoutingD2Round) -> Path:
    return DATA_ROOT / round_spec.slug


def round_plot_dir(round_spec: RoutingD2Round) -> Path:
    return PLOT_ROOT / round_spec.slug


__all__ = [
    "COLOR_KEYS",
    "DATA_ROOT",
    "DISPLAY_LABELS",
    "EVAL_REPO",
    "EVAL_REVISION",
    "MANIFEST_PATH",
    "MANIFEST_SHA256",
    "PLOT_ROOT",
    "SOURCE_EVAL_REVISION",
    "POLICY_ROSTER",
    "REVIEW_RULE",
    "ROUNDS",
    "RoutingD2Round",
    "SNAPSHOT_NAME",
    "heldout_config",
    "round_by_slug",
    "round_data_dir",
    "round_plot_dir",
    "run_rounds",
    "snapshot_spec",
]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--no-frame-check",
        action="store_true",
        help="skip the check of results.json against the frame labels (750 episodes' data)",
    )
    args = ap.parse_args(argv)
    for slug, summary in run_rounds(validate_frames=not args.no_frame_check).items():
        rates = ", ".join(
            f"{row['policy_name']} {row['successes']}/{row['episodes']}"
            for row in summary["policy_summary"]
        )
        print(f"{slug.upper()}: {rates}")
    print(f"wrote {DATA_ROOT} and {PLOT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
