from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from mulligan.tools import outcome_review as outcome_editor


def test_push_results_json_files_uses_single_hub_commit(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    results = root / "results.json"
    backup = root / "results_eval_time.json"
    results.write_text('{"canonical": true}\n')
    backup.write_text('{"raw": true}\n')

    commits = []
    advanced = []

    class FakeApi:
        def create_commit(self, **kwargs):
            commits.append(kwargs)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
    monkeypatch.setattr(outcome_editor, "advance_lerobot_version_tag", advanced.append)

    outcome_editor.push_results_json_files("org/example", root, [results, backup])

    assert advanced == ["org/example"]
    assert len(commits) == 1
    assert commits[0]["repo_id"] == "org/example"
    assert commits[0]["repo_type"] == "dataset"
    assert commits[0]["commit_message"] == "Sync results.json with outcome-edited labels"
    assert {op.path_in_repo for op in commits[0]["operations"]} == {
        "results.json",
        "results_eval_time.json",
    }


def test_push_progress_file_uploads_dotfile_and_advances_tag(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    progress = root / outcome_editor.PROGRESS_FILENAME
    progress.write_text('{"changed_episodes": {}, "skipped_episodes": []}\n')

    uploads = []
    advanced = []

    class FakeApi:
        def upload_file(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
    monkeypatch.setattr(outcome_editor, "advance_lerobot_version_tag", advanced.append)

    returned = outcome_editor.push_progress_file("org/example", root)

    assert returned == progress
    assert advanced == ["org/example"]
    assert len(uploads) == 1
    assert uploads[0]["repo_id"] == "org/example"
    assert uploads[0]["repo_type"] == "dataset"
    assert uploads[0]["path_or_fileobj"] == str(progress)
    assert uploads[0]["path_in_repo"] == outcome_editor.PROGRESS_FILENAME
    assert uploads[0]["commit_message"] == "Upload outcome editor progress record"


def test_push_progress_file_missing_dotfile_is_noop(tmp_path: Path, monkeypatch) -> None:
    uploads = []
    advanced = []

    class FakeApi:
        def upload_file(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
    monkeypatch.setattr(outcome_editor, "advance_lerobot_version_tag", advanced.append)

    assert outcome_editor.push_progress_file("org/example", tmp_path) is None
    assert uploads == []
    assert advanced == []


def test_dataset_total_episodes_accepts_dict_info(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    for idx in range(2):
        (data_dir / f"file-{idx:03d}.parquet").write_bytes(b"placeholder")

    class FakeMeta:
        info = {"total_episodes": 2, "fps": 15}

        @staticmethod
        def get_data_file_path(ep_idx: int) -> Path:
            return Path(f"data/chunk-000/file-{ep_idx:03d}.parquet")

    dataset = SimpleNamespace(root=root, meta=FakeMeta())

    assert outcome_editor.dataset_total_episodes(dataset) == 2
    assert outcome_editor.dataset_fps(dataset) == 15
    assert [p.relative_to(root) for p in outcome_editor.active_data_parquet_files(dataset)] == [
        Path("data/chunk-000/file-000.parquet"),
        Path("data/chunk-000/file-001.parquet"),
    ]


def test_dataset_total_episodes_missing_dict_key_fails_loudly() -> None:
    dataset = SimpleNamespace(meta=SimpleNamespace(info={"fps": 15}))

    with pytest.raises(KeyError):
        outcome_editor.dataset_total_episodes(dataset)
