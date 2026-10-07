"""The shared real-robot CLI groups keep the flags, defaults, help and order they had inline.

``add_robot_reset_args`` replaced identical blocks in ``collect/rollout.py`` and
``eval/manifest_eval.py``; ``add_dataset_revision_args`` replaced identical blocks in the
critic (``iql_args.py``) and policy (``policy.py``) trainers. The expected tables below were
captured from the inline definitions before the extraction.
"""

from __future__ import annotations

import argparse

import pytest

from mulligan.real.collect import rollout
from mulligan.real.eval import manifest_eval
from mulligan.real.train import iql_args, policy
from mulligan.real.train.hub_data import DEFAULT_DATASET_REVISION

# (option, dest, type, default, help)
ROBOT_RESET_ARGS = [
    (
        "--reset-max-retries",
        "reset_max_retries",
        int,
        6,
        "Verified robot reset attempts before failing loudly (default: 6)",
    ),
    (
        "--reset-retry-delay-s",
        "reset_retry_delay_s",
        float,
        1.0,
        "Initial delay between failed reset attempts in seconds (default: 1.0)",
    ),
    (
        "--reset-retry-backoff",
        "reset_retry_backoff",
        float,
        1.5,
        "Multiplier for reset retry delay after each failed attempt (default: 1.5)",
    ),
    (
        "--reset-max-retry-delay-s",
        "reset_max_retry_delay_s",
        float,
        5.0,
        "Maximum delay between failed reset attempts in seconds (default: 5.0)",
    ),
    (
        "--robot-state-refresh-max-wait-s",
        "robot_state_refresh_max_wait_s",
        float,
        1.5,
        "Maximum time to poll for a fresh robot-state timestamp after env.step() "
        "returns a stale observation while saving data (default: 1.5)",
    ),
    (
        "--robot-state-refresh-poll-interval-s",
        "robot_state_refresh_poll_interval_s",
        float,
        0.01,
        "Polling interval for fresh robot-state observations in seconds (default: 0.01)",
    ),
]

DATASET_REVISION_ARGS = [
    (
        "--dataset-revisions",
        "dataset_revisions",
        "REPO=REVISION",
        "Pin the Hub revision (commit sha or tag) of training/eval repos. Unpinned repos "
        f"resolve the {DEFAULT_DATASET_REVISION!r} tag. Repeatable.",
    ),
    (
        "--dataset-episodes",
        "dataset_episodes",
        "REPO=SELECTOR",
        "Read only these episodes of a repo: 'session_id:<id>[,<id>...]' (resolved via "
        "meta/episode_provenance.parquet) or 'episode_index:<i>[,<j>-<k>...]'. The repo "
        "must also be pinned with --dataset-revisions. Repeatable; one selector per repo.",
    ),
]


class _Captured(Exception):
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser


def _capture_parser(monkeypatch, parse_args_fn) -> argparse.ArgumentParser:
    """Return the parser a ``parse_args()`` entrypoint builds, without parsing sys.argv."""

    def fake(self, *args, **kwargs):
        raise _Captured(self)

    with monkeypatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", fake)
        with pytest.raises(_Captured) as info:
            parse_args_fn()
    return info.value.parser


def _options(parser: argparse.ArgumentParser) -> list[str]:
    return [a.option_strings[0] for a in parser._actions if a.option_strings]


def _action(parser: argparse.ArgumentParser, option: str) -> argparse.Action:
    return next(a for a in parser._actions if option in a.option_strings)


def _assert_block(parser, names, before, after):
    options = _options(parser)
    start = options.index(names[0])
    assert options[start - 1 : start + len(names) + 1] == [before, *names, after]


@pytest.mark.parametrize(
    ("entrypoint", "before", "after", "required"),
    [
        (rollout.parse_args, "--randomize-reset", "--dataset-name", ["--model", "m"]),
        (
            manifest_eval.parse_args,
            "--drop-fixed-policy",
            "--inference-backend",
            ["--environment", "e", "--initial-states-manifest", "m.json"],
        ),
    ],
    ids=["rollout", "manifest_eval"],
)
def test_robot_reset_args_unchanged(monkeypatch, entrypoint, before, after, required):
    parser = _capture_parser(monkeypatch, entrypoint)
    _assert_block(parser, [row[0] for row in ROBOT_RESET_ARGS], before, after)
    for option, dest, type_, default, help_ in ROBOT_RESET_ARGS:
        action = _action(parser, option)
        assert (action.dest, action.type, action.default, action.help) == (
            dest,
            type_,
            default,
            help_,
        )
        assert action.nargs is None and not action.required

    namespace = parser.parse_args(required)
    assert {row[1]: getattr(namespace, row[1]) for row in ROBOT_RESET_ARGS} == {
        row[1]: row[3] for row in ROBOT_RESET_ARGS
    }
    namespace = parser.parse_args(
        [*required, "--reset-max-retries", "3", "--reset-retry-delay-s", "0.5"]
    )
    assert (namespace.reset_max_retries, namespace.reset_retry_delay_s) == (3, 0.5)


@pytest.mark.parametrize(
    ("build_parser", "before", "required"),
    [
        (iql_args.build_parser, "--eval-repo-ids", ["--encoder-artifact", "enc"]),
        (policy.build_parser, "--dataset-root", []),
    ],
    ids=["critic", "policy"],
)
def test_dataset_revision_args_unchanged(build_parser, before, required):
    parser = build_parser()
    _assert_block(parser, [row[0] for row in DATASET_REVISION_ARGS], before, "--no-dataset-sync")
    for option, dest, metavar, help_ in DATASET_REVISION_ARGS:
        action = _action(parser, option)
        assert isinstance(action, argparse._ExtendAction)
        assert (action.dest, action.nargs, action.default, action.metavar, action.help) == (
            dest,
            "+",
            None,
            metavar,
            help_,
        )

    namespace = parser.parse_args(["--repo-ids", "a/b,c/d", *required])
    assert namespace.dataset_revisions is None and namespace.dataset_episodes is None
    namespace = parser.parse_args(
        [
            "--repo-ids",
            "a/b,c/d",
            *required,
            "--dataset-revisions",
            "a/b=v1",
            "--dataset-revisions",
            "c/d=v2",
            "--dataset-episodes",
            "a/b=episode_index:0-3",
        ]
    )
    assert namespace.dataset_revisions == ["a/b=v1", "c/d=v2"]
    assert namespace.dataset_episodes == ["a/b=episode_index:0-3"]
