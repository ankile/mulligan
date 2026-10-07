"""Tests for blind_dagger's --fixed-policy-dp-override plumbing and push flags.

A DP+IQL composite COLLECTOR loaded via the blind collector would otherwise re-rank
the critic's baked-in ``dp_artifact`` (a non-canonical auto-resume version, e.g.
``-final:v4``). The flag forces the canonical DP actor artifact per arm and fails loud on
mis-keyed or no-op overrides.
"""

import argparse
import sys

import pytest

import mulligan.real.collect.blind_dagger as blind_dagger
from mulligan.real.collect.blind_dagger import _parse_dp_artifact_override
from mulligan.real.collect.hf_utils import resolve_push_repo_id


def test_parse_valid_entry() -> None:
    key, artifact = _parse_dp_artifact_override(
        "mulligan_sobol=hf://mulligan/real-marker-d2-r04-mulligan-dp"
    )
    assert key == "mulligan_sobol"
    assert artifact == "hf://mulligan/real-marker-d2-r04-mulligan-dp"


def test_parse_strips_whitespace() -> None:
    key, artifact = _parse_dp_artifact_override("  mulligan_sobol =  art:v0 ")
    assert key == "mulligan_sobol"
    assert artifact == "art:v0"


def test_parse_preserves_equals_in_artifact() -> None:
    # Only the FIRST '=' splits; artifact may contain '=' (defensive).
    key, artifact = _parse_dp_artifact_override("k=a=b")
    assert key == "k"
    assert artifact == "a=b"


@pytest.mark.parametrize("spec", ["no_equals_here", "=art:v0", "key=", "  =  "])
def test_parse_rejects_malformed(spec: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_dp_artifact_override(spec)


def _argv(*extra: str) -> list[str]:
    return [
        "blind_dagger",
        "--task-name",
        "marker_d2",
        "--arm",
        "a=hf://mulligan/real-marker-d2-r01-baseline-dp",
        "--arm",
        "b=hf://mulligan/real-marker-d2-r01-mulligan-dp",
        "--dataset-name",
        "test-dataset",
        *extra,
    ]


def test_parse_args_has_no_arena_or_online_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", _argv())
    args = blind_dagger.parse_args()
    for name in ("arena_url", "register_in_arena", "online", "session_id", "environment"):
        assert not hasattr(args, name), name
    assert args.push_to_hub is False and args.hf_namespace is None


def test_push_with_bare_name_requires_hf_namespace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("--push-to-hub"))
    with pytest.raises(SystemExit):
        blind_dagger.parse_args()
    assert "--hf-namespace" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ("--push-to-hub", "--hf-namespace", "my-org"),
        ("--push-to-hub", "--push-repo-id", "my-org/test-dataset"),
    ],
)
def test_push_with_namespace_or_full_repo_id_parses(
    monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(*extra))
    args = blind_dagger.parse_args()
    repo_id = resolve_push_repo_id(args.push_repo_id or args.dataset_name, args.hf_namespace)
    assert repo_id == "my-org/test-dataset"


def test_resolve_push_repo_id() -> None:
    assert resolve_push_repo_id("org/name", None) == "org/name"
    assert resolve_push_repo_id("org/name", "other") == "org/name"
    assert resolve_push_repo_id("name", "org") == "org/name"
    for namespace in (None, ""):
        with pytest.raises(ValueError, match="--hf-namespace"):
            resolve_push_repo_id("name", namespace)


def test_parse_args_requires_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _argv()
    del argv[1:3]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        blind_dagger.parse_args()


def test_task_name_choices_are_the_released_tasks() -> None:
    # Insert Marker, Route Cable, Thread Nut.
    from mulligan.real.lifecycle.tasks import task_name_choices

    assert task_name_choices() == ("marker_d2", "routing_d2", "square_d2")


@pytest.mark.parametrize(
    ("task", "stored"),
    [
        ("marker_d2", "marker_d2"),
        ("square_d2", "square_d2"),
        ("routing_d2", "routing_d2"),
    ],
)
def test_task_name_is_stored_as_the_collection_task_name(
    monkeypatch: pytest.MonkeyPatch, task: str, stored: str
) -> None:
    # The datasets and manifests carry the collection task name; storing any other spelling would
    # fail the manifest task check and drop the task's sub-goal marks.
    argv = _argv()
    argv[2] = task
    monkeypatch.setattr(sys, "argv", argv)
    assert blind_dagger.parse_args().task_name == stored


@pytest.mark.parametrize("task", ["insert_marker_d1", "Square_D1", "droid_teleop"])
def test_parse_args_rejects_unreleased_tasks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], task: str
) -> None:
    argv = _argv()
    argv[2] = task
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        blind_dagger.parse_args()
    assert "invalid choice" in capsys.readouterr().err
