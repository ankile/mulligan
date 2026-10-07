"""Concurrent downloads publish complete, revision-specific training snapshots."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from mulligan.real.train import hub_data

REPO = "local/dataset"
A, B = "a" * 40, "b" * 40


def write_snapshot(root, revision):
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(revision)
    (root / "data").mkdir()
    (root / "data/rows").write_text(revision)


def test_concurrent_same_revision_downloads_once(tmp_path, monkeypatch):
    entered, release, second_started = Event(), Event(), Event()
    calls = []

    def download(repo_id, *, root, revision, **kwargs):
        calls.append(revision)
        entered.set()
        assert release.wait(10)
        write_snapshot(root, revision)

    def sync(second=False):
        if second:
            second_started.set()
        return hub_data.prepare_datasets([REPO], tmp_path, {REPO: A}, sync=True)

    monkeypatch.setattr(hub_data, "LeRobotDataset", download)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(sync)
        try:
            assert entered.wait(10)
            second = pool.submit(sync, True)
            assert second_started.wait(10)
        finally:
            release.set()
        roots = first.result(), second.result()
    assert roots[0] == roots[1]
    assert calls == [A]
    assert (hub_data.dataset_dir(roots[0], REPO) / "data/rows").read_text() == A


def test_revision_change_and_failed_download_preserve_existing_readers(tmp_path, monkeypatch):
    calls = []
    fail = False

    def download(repo_id, *, root, revision, **kwargs):
        calls.append(revision)
        write_snapshot(root, revision)
        if fail:
            raise OSError("interrupted")

    monkeypatch.setattr(hub_data, "LeRobotDataset", download)
    first = hub_data.prepare_datasets([REPO], tmp_path, {REPO: A}, sync=True)
    reader = hub_data.dataset_dir(first, REPO) / "data/rows"
    fail = True
    with pytest.raises(OSError, match="interrupted"):
        hub_data.prepare_datasets([REPO], tmp_path, {REPO: B}, sync=True)
    assert reader.read_text() == A
    fail = False
    second = hub_data.prepare_datasets([REPO], tmp_path, {REPO: B}, sync=True)
    assert first != second
    assert reader.read_text() == A
    assert (hub_data.dataset_dir(second, REPO) / "data/rows").read_text() == B
    assert calls == [A, B, B]
    assert hub_data.prepare_datasets([REPO], tmp_path, {REPO: A}, sync=True) == first
    assert calls == [A, B, B]


def test_no_sync_uses_original_local_layout(tmp_path, monkeypatch):
    root = tmp_path / REPO
    write_snapshot(root, "local")
    monkeypatch.setattr(hub_data, "LeRobotDataset", lambda *a, **k: pytest.fail("download"))
    assert hub_data.prepare_datasets([REPO], tmp_path, {}, sync=False) == tmp_path
    assert (root / "data/rows").read_text() == "local"


def test_synced_layout_loads_through_training_dataset_loader(tmp_path, monkeypatch):
    from tests.unit.sim_dataset import build_sim_dataset

    resolved = []

    def resolve(repo, revision):
        resolved.append((repo, revision))
        return A

    def download(repo_id, *, root, revision, **kwargs):
        assert revision == A
        build_sim_dataset(root, episode_lengths=(2, 3))

    monkeypatch.setattr(hub_data, "resolve_dataset_commit", resolve)
    monkeypatch.setattr(hub_data, "LeRobotDataset", download)
    layout = hub_data.prepare_datasets([REPO], tmp_path, {REPO: "v3.0"}, sync=True)
    ds = hub_data.load_multi_dataset([REPO], layout, {REPO: "v3.0"})
    assert ds.num_episodes == 2 and len(ds) == 7
    assert int(ds[0]["episode_index"]) == 0
    assert int(ds[6]["episode_index"]) == 1
    assert resolved == [(REPO, "v3.0")]
