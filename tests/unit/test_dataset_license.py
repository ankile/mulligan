"""Every dataset push writes an MIT card by default (LeRobot's own default is apache-2.0)."""

from __future__ import annotations

import argparse
import sys

import pytest

from mulligan.real.collect.hf_utils import DEFAULT_DATASET_LICENSE

REAL_MODEL = "hf://mulligan/real-marker-d2-r05-mulligan-dp"
BLIND_DAGGER_ARGV = [
    "--task-name",
    "marker_d2",
    "--arm",
    "a=hf://mulligan/real-marker-d2-r01-baseline-dp",
    "--arm",
    "b=hf://mulligan/real-marker-d2-r01-mulligan-dp",
    "--dataset-name",
    "d",
]


def _sim_teleop(argv):
    from mulligan.sim.collect import teleop

    return teleop.parse_args(argv)


def _sim_dagger(argv):
    from mulligan.sim.collect import dagger

    return dagger.parse_args(["--policy", "unused", "--dataset-name", "d", *argv])


def _sim_rollouts(argv):
    from mulligan.sim.collect import rollouts

    return rollouts.build_parser().parse_args(["--checkpoint", "c", "--dataset-name", "d", *argv])


def _real_teleop(argv):
    from mulligan.real.collect import teleop

    return teleop.parse_args(argv)


def _sys_argv_parser(module_name, base):
    def parse(argv):
        import importlib

        module = importlib.import_module(module_name)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", [module_name, *base, *argv])
            return module.parse_args()

    return parse


PARSERS = {
    "sim.teleop": _sim_teleop,
    "sim.dagger": _sim_dagger,
    "sim.rollouts": _sim_rollouts,
    "real.teleop": _real_teleop,
    "real.dagger": _sys_argv_parser(
        "mulligan.real.collect.dagger",
        ["--model", REAL_MODEL, "--task-name", "marker_d2", "--dataset-name", "d"],
    ),
    "real.rollout": _sys_argv_parser(
        "mulligan.real.collect.rollout",
        ["--model", REAL_MODEL, "--task-name", "marker_d2", "--dataset-name", "d"],
    ),
    "real.blind_dagger": _sys_argv_parser("mulligan.real.collect.blind_dagger", BLIND_DAGGER_ARGV),
    "tools.outcome_review": _sys_argv_parser("mulligan.tools.outcome_review", ["--repo-id", "o/d"]),
}


@pytest.mark.parametrize("parse", PARSERS.values(), ids=PARSERS.keys())
def test_license_flag_defaults_to_mit(parse):
    assert DEFAULT_DATASET_LICENSE == "mit"
    assert parse([]).license == "mit"
    assert parse(["--license", "cc-by-4.0"]).license == "cc-by-4.0"


def test_sim_dagger_push_passes_the_license():
    from mulligan.sim.collect import dagger

    pushed = {}

    class Dataset:
        repo_id = None

        def push_to_hub(self, **kwargs):
            pushed.update(kwargs)

    collector = object.__new__(dagger.DaggerCollector)
    collector.hub_repo_id = "org/d"
    collector.dataset = Dataset()
    collector.quota_ledger_path = None
    collector.args = argparse.Namespace(license="cc-by-4.0")
    collector._push_to_hub()
    assert pushed == {"private": False, "license": "cc-by-4.0"}


def test_outcome_review_main_passes_the_license(monkeypatch, tmp_path):
    from mulligan.tools import outcome_review

    calls = []
    monkeypatch.setattr(
        outcome_review,
        "process_single_dataset",
        lambda *args, **kwargs: calls.append(("editor", kwargs["license"])),
    )
    monkeypatch.setattr(
        outcome_review,
        "headless_apply_and_push",
        lambda repo_id, overlay, **kwargs: calls.append(("headless", kwargs["license"])) or {},
    )
    overlay = tmp_path / "overlay.json"
    overlay.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["outcome_review", "--repo-id", "org/d", "--push"])
    outcome_review.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["outcome_review", "--repo-id", "org/d", "--push", "--apply-overlay", str(overlay)]
        + ["--license", "cc-by-4.0"],
    )
    outcome_review.main()
    assert calls == [("editor", "mit"), ("headless", "cc-by-4.0")]


def test_outcome_review_apply_pushes_the_license(monkeypatch, tmp_path):
    from mulligan.tools import outcome_review

    class Pushed(Exception):
        pass

    class Dataset:
        root = tmp_path

        def push_to_hub(self, **kwargs):
            raise Pushed(kwargs)

    monkeypatch.setattr(outcome_review, "repair_results_json_file", lambda *a, **k: [])
    progress = {"changed_episodes": {}, "skipped_episodes": []}
    for kwargs, expected in (({}, "mit"), ({"license": "cc-by-4.0"}, "cc-by-4.0")):
        with pytest.raises(Pushed) as exc:
            outcome_review.apply_progress_and_push(
                "org/d",
                Dataset(),
                data_files=[],
                ledger_paths=[],
                progress=progress,
                subtask_marks=0,
                push=True,
                **kwargs,
            )
        assert exc.value.args[0] == {"license": expected}
