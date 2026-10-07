"""`mulligan.release.download`: pinned snapshot downloads and filters."""

from __future__ import annotations

import json

import pytest

from mulligan.release import download as dl


def test_pinned_revision_reads_the_lookup():
    pins = json.loads((dl.RELEASE_DIR / "revisions.json").read_text())["repos"]
    repo = "mulligan/real-marker-d2-r02-eval"
    assert dl.pinned_revision(repo) == pins[repo]["revision"]
    assert dl.repo_type(repo) == "dataset"
    assert dl.repo_type("mulligan/real-marker-d2-r04-mulligan-dp") == "model"
    with pytest.raises(KeyError, match="not a released repo"):
        dl.pinned_revision("mulligan/does-not-exist")


def test_dataset_filters():
    rows = dl.select_datasets(task=["real-marker-d2"], role=["evaluation"], round=[5])
    assert {r["repo"] for r in rows} == {
        "mulligan/real-marker-d2-r05-eval",
        "mulligan/real-marker-d2-r05-screen",
    }
    cable = dl.select_datasets(task=["real-routing-d2"], role=["evaluation"], round=[3])
    assert "mulligan/real-routing-d2-r00-r05-eval" in {r["repo"] for r in cable}
    views = dl.select_datasets(task=["sim-square-narrow"], method=["mulligan"])
    assert views and all(r["variant"] == "mulligan" for r in views)
    assert dl.select_datasets(task=["no-such-task"]) == []
    assert len(dl.select_datasets()) == 223


def test_model_filters():
    critics = dl.select_models(task=["real-square-d2"], kind=["idql-critic"], round=[5])
    assert {r["repo"] for r in critics} == {
        "mulligan/real-square-d2-r05-mulligan-idql-critic",
        "mulligan/real-square-d2-r05-mulligan-idql-critic-screen-b02",
        "mulligan/real-square-d2-r05-mulligan-idql-critic-screen-b03",
    }
    dps = dl.select_models(task=["real-marker-d2"], kind=["dp-actor"], method=["mulligan"])
    assert dps and all(c["method"] == "mulligan-dp" for r in dps for c in r["checkpoints"])
    assert len(dl.select_models()) == 180


def test_cli_dry_run(capsys):
    assert (
        dl.main(
            [
                "datasets",
                "--task",
                "real-square-d2",
                "--role",
                "evaluation",
                "--round",
                "3",
                "--dry-run",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        f"mulligan/real-square-d2-r03-eval\tdataset\t{dl.pinned_revision('mulligan/real-square-d2-r03-eval')}"
    ]
    assert dl.main(["models", "--task", "nothing", "--dry-run"]) == 1


def test_download_pins_the_revision(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(repo, **kw):
        calls.append((repo, kw))
        return str(tmp_path)

    monkeypatch.setattr(dl, "snapshot_download", fake_snapshot_download)
    assert dl.main(["repo", "mulligan/real-marker-d2-r04-mulligan-dp", "--meta-only"]) == 0
    ((repo, kw),) = calls
    assert repo == "mulligan/real-marker-d2-r04-mulligan-dp"
    assert kw["revision"] == dl.pinned_revision(repo) and kw["repo_type"] == "model"
    assert kw["allow_patterns"] == dl.META_PATTERNS


@pytest.mark.network
def test_download_meta_anonymously(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    repo = "mulligan/real-marker-d2-r00-eval"
    path = dl.download(repo, include=["meta/info.json"], cache_dir=tmp_path)
    assert path.name == dl.pinned_revision(repo)
    info = json.loads((path / "meta" / "info.json").read_text())
    row = next(r for r in dl.select_datasets() if r["repo"] == repo)
    assert info["total_episodes"] == row["episodes"]
