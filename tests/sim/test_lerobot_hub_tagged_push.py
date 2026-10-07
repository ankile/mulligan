"""Guard the v3.0-tag-tracks-main invariant for LeRobot dataset pushes.

Training/eval reads pin the blessed CODEBASE_VERSION (v3.0) tag, not raw main, so
every push MUST land that tag on the pushed commit. push_to_hub does this via the
vendored default tag_version=True; these tests assert our explicit wrapper enforces
it, so a future LeRobot default flip fails here in CI instead of silently stranding
v3.0-pinned training on stale data.
"""

from __future__ import annotations

import pytest

from mulligan.tools import lerobot_hub


class _FakeRepoInfo:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class _FakeApi:
    def __init__(self, shas: dict[str, str], remote_files: list[str] | None = None) -> None:
        self._shas = shas
        self.remote_files = list(remote_files or [])
        self.deleted: list[str] = []
        self.commits = 0

    def repo_info(self, *, repo_id: str, repo_type: str, revision: str) -> _FakeRepoInfo:
        return _FakeRepoInfo(self._shas[revision])

    def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
        return list(self.remote_files)

    def create_commit(self, *, repo_id: str, repo_type: str, operations, commit_message: str):
        self.commits += 1
        self.deleted.extend(op.path_in_repo for op in operations)
        self.remote_files = [f for f in self.remote_files if f not in self.deleted]


class _FakeDataset:
    def __init__(self, repo_id: str, root=None) -> None:
        self.repo_id = repo_id
        self.root = root
        self.push_kwargs: dict | None = None

    def push_to_hub(self, **kwargs) -> None:
        self.push_kwargs = kwargs


def _patch_api(
    monkeypatch, shas: dict[str, str], remote_files: list[str] | None = None
) -> _FakeApi:
    api = _FakeApi(shas, remote_files)
    monkeypatch.setattr(lerobot_hub, "HfApi", lambda: api)
    return api


def _local_tree(tmp_path, files: list[str]):
    for rel in files:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return tmp_path


def test_assert_version_tag_tracks_passes_on_match(monkeypatch):
    _patch_api(monkeypatch, {"main": "abc123", lerobot_hub.CODEBASE_VERSION: "abc123"})
    lerobot_hub.assert_lerobot_version_tag_tracks("user/x")  # must not raise


def test_assert_version_tag_tracks_raises_on_mismatch(monkeypatch):
    _patch_api(monkeypatch, {"main": "abc123", lerobot_hub.CODEBASE_VERSION: "deadbeef"})
    with pytest.raises(RuntimeError, match="STALE"):
        lerobot_hub.assert_lerobot_version_tag_tracks("user/x")


def test_assert_warn_only_does_not_raise_on_mismatch(monkeypatch, capsys):
    # Eval push path: a mismatch must warn loudly, not crash the end of a session.
    _patch_api(monkeypatch, {"main": "newsha", lerobot_hub.CODEBASE_VERSION: "oldsha"})
    lerobot_hub.assert_lerobot_version_tag_tracks("user/x", warn_only=True)
    out = capsys.readouterr().out
    assert "WARNING" in out and "STALE" in out


def test_tagged_push_sets_tag_version_true_and_asserts(monkeypatch, tmp_path):
    _patch_api(monkeypatch, {"main": "abc123", lerobot_hub.CODEBASE_VERSION: "abc123"})
    ds = _FakeDataset("user/x", root=tmp_path)
    lerobot_hub.push_lerobot_dataset_tagged_main(ds, private=False)
    assert ds.push_kwargs == {"private": False, "tag_version": True, "license": "mit"}


def test_tagged_push_keeps_an_explicit_license(monkeypatch, tmp_path):
    _patch_api(monkeypatch, {"main": "abc123", lerobot_hub.CODEBASE_VERSION: "abc123"})
    ds = _FakeDataset("user/x", root=tmp_path)
    lerobot_hub.push_lerobot_dataset_tagged_main(ds, license="cc-by-4.0")
    assert ds.push_kwargs["license"] == "cc-by-4.0"


def test_tagged_push_surfaces_untagged_push(monkeypatch, tmp_path):
    # push "succeeds" but the tag never moves -> wrapper must fail loudly.
    _patch_api(monkeypatch, {"main": "newsha", lerobot_hub.CODEBASE_VERSION: "oldsha"})
    ds = _FakeDataset("user/x", root=tmp_path)
    with pytest.raises(RuntimeError, match="STALE"):
        lerobot_hub.push_lerobot_dataset_tagged_main(ds, private=False)


def test_tagged_push_rejects_tag_version_false():
    ds = _FakeDataset("user/x")
    with pytest.raises(ValueError, match="tag_version"):
        lerobot_hub.push_lerobot_dataset_tagged_main(ds, tag_version=False)
    assert ds.push_kwargs is None  # never pushed


def test_tagged_push_prunes_stale_remote_generated_shards(monkeypatch, tmp_path):
    # Interrupted sessions push un-consolidated meta/episodes shards; the deliberate-exit
    # consolidation then deletes them locally, and push_to_hub (an upload_folder) leaves
    # the duplicates on main. The tagged push must delete remote generated shards
    # absent locally, and nothing else.
    local = _local_tree(
        tmp_path,
        [
            "meta/info.json",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-000.parquet",
        ],
    )
    api = _patch_api(
        monkeypatch,
        {"main": "abc123", lerobot_hub.CODEBASE_VERSION: "abc123"},
        remote_files=[
            "README.md",
            "meta/info.json",
            "meta/protocol_quota_ledger.jsonl",
            "meta/episodes/chunk-000/file-000.parquet",
            "meta/episodes/chunk-000/file-001.parquet",
            "meta/episodes/chunk-000/file-002.parquet",
            "data/chunk-000/file-000.parquet",
            "videos/observation.images.side_1/chunk-000/file-003.mp4",
        ],
    )
    ds = _FakeDataset("user/x", root=local)
    lerobot_hub.push_lerobot_dataset_tagged_main(ds, private=False)
    assert api.deleted == [
        "meta/episodes/chunk-000/file-001.parquet",
        "meta/episodes/chunk-000/file-002.parquet",
        "videos/observation.images.side_1/chunk-000/file-003.mp4",
    ]
    assert api.commits == 1
    assert ds.push_kwargs == {"private": False, "tag_version": True, "license": "mit"}


def test_tagged_push_no_delete_commit_when_nothing_stale(monkeypatch, tmp_path):
    local = _local_tree(tmp_path, ["meta/episodes/chunk-000/file-000.parquet"])
    api = _patch_api(
        monkeypatch,
        {"main": "abc123", lerobot_hub.CODEBASE_VERSION: "abc123"},
        remote_files=["README.md", "meta/episodes/chunk-000/file-000.parquet"],
    )
    ds = _FakeDataset("user/x", root=local)
    lerobot_hub.push_lerobot_dataset_tagged_main(ds, private=False)
    assert api.commits == 0 and api.deleted == []


def test_tagged_push_leaves_remote_alone_on_branch_push(monkeypatch, tmp_path):
    # A branch push is not a statement about main's shards.
    local = _local_tree(tmp_path, ["meta/episodes/chunk-000/file-000.parquet"])
    api = _patch_api(
        monkeypatch,
        {"main": "abc123", lerobot_hub.CODEBASE_VERSION: "abc123", "wip": "abc123"},
        remote_files=["meta/episodes/chunk-000/file-001.parquet"],
    )
    ds = _FakeDataset("user/x", root=local)
    lerobot_hub.push_lerobot_dataset_tagged_main(ds, branch="wip")
    assert api.commits == 0


class _TagApi(_FakeApi):
    """Tag store whose first create_tag call fails with a transient 502."""

    def __init__(self, shas: dict[str, str]) -> None:
        super().__init__(shas)
        self.create_failures = 1

    def delete_tag(self, *, repo_id: str, repo_type: str, tag: str) -> None:
        if tag not in self._shas:
            raise lerobot_hub.RevisionNotFoundError("no such tag")
        del self._shas[tag]

    def create_tag(self, *, repo_id, repo_type, tag, revision, exist_ok=False) -> None:
        if self.create_failures:
            self.create_failures -= 1
            raise RuntimeError("HfHubHTTPError: 502 Bad Gateway")
        if tag in self._shas and not exist_ok:
            raise RuntimeError("409 tag exists")
        self._shas.setdefault(tag, revision)


def test_tag_move_retries_a_transient_create_failure(monkeypatch):
    # Regression: delete_tag and create_tag ran un-retried, so a 502 on
    # create after a successful delete left the dataset without its v3.0 tag.
    monkeypatch.setenv("MULLIGAN_HF_RETRY_BASE_SLEEP_S", "0")
    api = _TagApi({"main": "new", lerobot_hub.CODEBASE_VERSION: "old"})
    monkeypatch.setattr(lerobot_hub, "HfApi", lambda: api)
    assert lerobot_hub.advance_lerobot_version_tag("user/x") == "new"
    assert api._shas[lerobot_hub.CODEBASE_VERSION] == "new"


def test_replacing_push_refuses_a_branch(monkeypatch, tmp_path):
    # Regression: stale-shard deletion and the tag move act on main, so a
    # branch push through this helper deleted shards main still referenced.
    api = _patch_api(
        monkeypatch, {"main": "abc123"}, remote_files=["data/chunk-000/file-001.parquet"]
    )
    ds = _FakeDataset("user/x", root=tmp_path)
    with pytest.raises(ValueError, match="replaces main"):
        lerobot_hub.push_lerobot_dataset_replacing_remote(ds, branch="experimental")
    assert api.commits == 0 and ds.push_kwargs is None
