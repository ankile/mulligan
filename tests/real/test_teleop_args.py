"""Argument validation and resume bookkeeping of the teleop collector."""

from __future__ import annotations

import json

import pytest

from mulligan.real.collect.teleop import parse_args


def _exit_message(capsys, argv) -> str:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def test_valid_arguments_parse():
    args = parse_args(
        [
            "--save-data",
            "--dataset-name",
            "demos",
            "--push-to-hub",
            "--hf-namespace",
            "me",
            "--task-name",
            "routing_d2",
        ]
    )
    assert args.push_to_hub and args.dataset_name == "demos"
    assert args.task_name == "routing_d2"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--save-data", "--push-to-hub", "--dataset-name", "demos"], "--hf-namespace"),
        (["--save-data", "--push-to-hub"], "--hf-namespace"),
        (["--push-to-hub", "--dataset-name", "me/demos"], "--push-to-hub requires --save-data"),
        (["--target-successes", "0"], "--target-successes must be positive"),
        (["--progress-window", "0"], "--progress-window must be positive"),
        (["--initial-states-manifest", "m.json"], "--ledger-path is required"),
        (["--save-data", "--dataset-name", "demos"], "--task-name is required"),
        (["--save-data", "--task-name", "droid_teleop"], "invalid choice"),
    ],
)
def test_invalid_arguments_are_parser_errors(capsys, argv, message):
    assert message in _exit_message(capsys, argv)


def test_non_manifest_resume_continues_from_local_dataset_episode_count(tmp_path):
    # Without a manifest ledger the next save index must come from the dataset, or the
    # first save of a resumed session fails the "episode count != saved count" guard.
    from mulligan.real.collect.teleop import _resume_saved_episode_count

    dataset = tmp_path / "demos"
    assert _resume_saved_episode_count(dataset, None) == 0
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps({"total_episodes": 37}))
    assert _resume_saved_episode_count(dataset, None) == 37
    assert _resume_saved_episode_count(dataset, [{"episode_index": 0}]) == 1
