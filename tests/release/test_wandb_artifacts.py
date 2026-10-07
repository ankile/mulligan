"""mulligan.release.hub: HF checkpoint resolver and the optional W&B artifact helpers."""

import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

import mulligan.release.hub as hub
from mulligan.release.hub import (
    HubCheckpoint,
    _normalize_wandb_artifact_identifier,
    get_artifact_cache_root,
    parse_hf_uri,
    release_revision,
    resolve_checkpoint,
)


def test_artifact_cache_root_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MULLIGAN_WANDB_ARTIFACT_CACHE", str(tmp_path / "override"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert get_artifact_cache_root() == tmp_path / "override"


def test_artifact_cache_root_uses_xdg_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MULLIGAN_WANDB_ARTIFACT_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert get_artifact_cache_root() == tmp_path / "xdg" / "mulligan" / "wandb_artifacts"


def test_artifact_cache_root_defaults_to_home_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MULLIGAN_WANDB_ARTIFACT_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert get_artifact_cache_root() == tmp_path / ".cache" / "mulligan" / "wandb_artifacts"


def test_normalize_wandb_artifact_identifier_accepts_uri():
    assert (
        _normalize_wandb_artifact_identifier("wandb://entity/project/model:v0")
        == "entity/project/model:v0"
    )


def test_normalize_wandb_artifact_identifier_accepts_bare_path():
    assert (
        _normalize_wandb_artifact_identifier("entity/project/model:v0") == "entity/project/model:v0"
    )


def test_normalize_wandb_artifact_identifier_rejects_empty_uri():
    with pytest.raises(ValueError, match="include a path"):
        _normalize_wandb_artifact_identifier("wandb://")


def test_normalize_wandb_artifact_identifier_rejects_other_uri_schemes():
    with pytest.raises(ValueError, match="Unsupported artifact URI scheme"):
        _normalize_wandb_artifact_identifier("hf://entity/model")


def test_get_checkpoint_files_excludes_stale_policy_contract_sidecars(tmp_path):
    """Policy I/O contract belongs in config.json, not sidecar files."""
    from mulligan.release.hub import get_checkpoint_files

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    (tmp_path / "action_target.json").write_text('{"action_target": "cartesian_position"}')
    (tmp_path / "side_crop.json").write_text(
        '{"side_crop_boxes": {"wrist_left": [126, 0, 640, 357]}}'
    )
    (tmp_path / "dual_side_crop.json").write_text(
        '{"dual_side_crop_boxes": {"wrist_left": [126, 0, 640, 357]}}'
    )

    names = {p.name for p in get_checkpoint_files(tmp_path)}
    assert names == {"config.json", "model.safetensors"}
    assert "action_target.json" not in names
    assert "side_crop.json" not in names
    assert "dual_side_crop.json" not in names


def test_create_artifact_metadata_includes_config_backed_crop_metadata():
    metadata = hub.create_artifact_metadata(
        repo_id="dataset",
        success_rate=0.0,
        avg_reward=0.0,
        step=50000,
        camera_crop_boxes={"wrist_left": (126, 0, 640, 357)},
        dual_side_crop_boxes={"side_1": (330, 100, 560, 360)},
    )

    assert metadata["camera_crop_boxes"] == {"wrist_left": [126, 0, 640, 357]}
    assert metadata["dual_side_crop_boxes"] == {"side_1": [330, 100, 560, 360]}


def test_download_checkpoint_clears_removed_policy_config_files_from_cache(monkeypatch, tmp_path):
    download_dir = tmp_path / "cached-artifact"
    download_dir.mkdir()
    for filename in hub._REMOVED_POLICY_CONFIG_FILES:
        (download_dir / filename).write_text("stale")
    (download_dir / "unrelated.txt").write_text("keep")

    class FakeArtifact:
        name = "cached-artifact:v0"

        def download(self, root):
            root_path = tmp_path / "cached-artifact"
            assert root == str(root_path)
            for filename in hub._REMOVED_POLICY_CONFIG_FILES:
                assert not (root_path / filename).exists()
            (root_path / "config.json").write_text("{}")
            return str(root_path)

    class FakeApi:
        def artifact(self, artifact_identifier):
            assert artifact_identifier == "entity/project/cached-artifact:v0"
            return FakeArtifact()

    monkeypatch.setattr(hub, "wandb", SimpleNamespace(run=None, Api=FakeApi))

    result = hub.download_checkpoint_from_wandb(
        "wandb://entity/project/cached-artifact:v0",
        download_dir=download_dir,
    )

    assert result == download_dir
    assert (download_dir / "config.json").exists()
    assert (download_dir / "unrelated.txt").read_text() == "keep"
    for filename in hub._REMOVED_POLICY_CONFIG_FILES:
        assert not (download_dir / filename).exists()


def test_download_checkpoint_uses_api_for_disabled_run(monkeypatch, tmp_path):
    class DisabledRun:
        settings = SimpleNamespace(mode="disabled")

        def use_artifact(self, artifact_identifier):
            raise AssertionError("disabled W&B run cannot resolve artifacts")

    class FakeArtifact:
        name = "parent:v0"

        def download(self, root):
            return root

    class FakeApi:
        def artifact(self, artifact_identifier):
            assert artifact_identifier == "entity/project/parent:v0"
            return FakeArtifact()

    monkeypatch.setattr(
        hub,
        "wandb",
        SimpleNamespace(run=DisabledRun(), Api=FakeApi),
    )

    result = hub.download_checkpoint_from_wandb(
        "entity/project/parent:v0", download_dir=tmp_path / "parent"
    )

    assert result == tmp_path / "parent"


def test_download_checkpoint_rejects_none_from_active_run(monkeypatch, tmp_path):
    class ActiveRun:
        settings = SimpleNamespace(mode="online")

        def use_artifact(self, artifact_identifier):
            return None

    monkeypatch.setattr(
        hub,
        "wandb",
        SimpleNamespace(run=ActiveRun(), Api=lambda: None),
    )

    with pytest.raises(RuntimeError, match="pinned checkpoint cannot be loaded"):
        hub.download_checkpoint_from_wandb(
            "entity/project/parent:v0", download_dir=tmp_path / "parent"
        )


def test_managed_checkpoint_cache_publishes_once_after_complete_download(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(hub, "get_artifact_cache_root", lambda: cache_root)
    download_started = threading.Event()
    allow_download_to_finish = threading.Event()
    calls = 0

    class FakeArtifact:
        name = "shared-artifact:v0"

        def download(self, root):
            nonlocal calls
            calls += 1
            root_path = Path(root)
            (root_path / "policy.pt").write_bytes(b"partial")
            download_started.set()
            assert allow_download_to_finish.wait(timeout=5)
            (root_path / "policy.pt").write_bytes(b"complete")
            return str(root_path)

    class FakeApi:
        def artifact(self, artifact_identifier):
            assert artifact_identifier == "entity/project/shared-artifact:v0"
            return FakeArtifact()

    monkeypatch.setattr(hub, "wandb", SimpleNamespace(run=None, Api=FakeApi))
    results = []

    def download():
        results.append(hub.download_checkpoint_from_wandb("entity/project/shared-artifact:v0"))

    first = threading.Thread(target=download)
    second = threading.Thread(target=download)
    first.start()
    assert download_started.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert second.is_alive(), "second reader bypassed the in-progress artifact lock"
    allow_download_to_finish.set()
    first.join(timeout=5)
    second.join(timeout=5)

    expected = cache_root / "shared-artifact:v0"
    assert results == [expected, expected]
    assert calls == 1
    assert (expected / "policy.pt").read_bytes() == b"complete"
    assert not list(cache_root.glob(".shared-artifact:v0.partial-*"))


def test_managed_checkpoint_cache_replaces_untrusted_partial_directory(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    partial_cache = cache_root / "shared-artifact:v0"
    partial_cache.mkdir(parents=True)
    (partial_cache / "policy.pt").write_bytes(b"partial-from-crashed-writer")
    monkeypatch.setattr(hub, "get_artifact_cache_root", lambda: cache_root)

    class FakeArtifact:
        name = "shared-artifact:v0"

        def download(self, root):
            root_path = Path(root)
            (root_path / "policy.pt").write_bytes(b"complete")
            return str(root_path)

    class FakeApi:
        def artifact(self, artifact_identifier):
            return FakeArtifact()

    monkeypatch.setattr(hub, "wandb", SimpleNamespace(run=None, Api=FakeApi))

    result = hub.download_checkpoint_from_wandb("entity/project/shared-artifact:v0")

    assert result == partial_cache
    assert (partial_cache / "policy.pt").read_bytes() == b"complete"
    assert not list(cache_root.glob(".shared-artifact:v0.stale-*"))


REVISION = "0123456789abcdef0123456789abcdef01234567"
REPO = "example-org/example-model"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (f"hf://{REPO}@{REVISION}/seed-1", HubCheckpoint(REPO, REVISION, "seed-1")),
        (f"hf://{REPO}/seed-1", HubCheckpoint(REPO, None, "seed-1")),
        (f"hf://{REPO}", HubCheckpoint(REPO, None, None)),
        (f"hf://{REPO}@v1.0/a/b/", HubCheckpoint(REPO, "v1.0", "a/b")),
    ],
)
def test_parse_hf_uri(uri, expected):
    ref = parse_hf_uri(uri)
    assert ref == expected
    assert parse_hf_uri(ref.uri()) == ref


@pytest.mark.parametrize("uri", ["hf://only-org", "wandb://e/p/a:v0", "mulligan/x/seed-1"])
def test_parse_hf_uri_rejects_other_strings(uri):
    with pytest.raises(ValueError, match="Not an HF checkpoint URI"):
        parse_hf_uri(uri)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        # both orders of @<revision> and /<subfolder> give the same reference
        (f"hf://{REPO}/seed-1@{REVISION}", HubCheckpoint(REPO, REVISION, "seed-1")),
        (f"hf://{REPO}/a/b@v1.0", HubCheckpoint(REPO, "v1.0", "a/b")),
        (
            f"hf://{REPO}@release-1",
            HubCheckpoint(REPO, "release-1", None),
        ),
        (f"hf://{REPO}@refs/pr/12", HubCheckpoint(REPO, "refs/pr/12", None)),
        (f"hf://{REPO}@refs/pr/12/seed-1", HubCheckpoint(REPO, "refs/pr/12", "seed-1")),
        (f"hf://{REPO}/seed-1@refs/pr/12", HubCheckpoint(REPO, "refs/pr/12", "seed-1")),
    ],
)
def test_parse_hf_uri_accepts_both_orders_and_pr_refs(uri, expected):
    ref = parse_hf_uri(uri)
    assert ref == expected
    assert parse_hf_uri(ref.uri()) == ref


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        (f"hf://{REPO}/a@{REVISION}/b", "ambiguous"),
        (f"hf://{REPO}/a@refs/pr/1/b", "ambiguous"),
        (f"hf://{REPO}@refs/heads/main", "refs/pr"),
        (f"hf://{REPO}@refs/pr/x", "refs/pr"),
        (f"hf://{REPO}@", "single non-empty"),
        (f"hf://{REPO}@a@b", "single non-empty"),
        (f"hf://{REPO}@{REVISION}//seed-1", "empty path"),
        (f"hf://{REPO}@-x", "malformed revision"),
        ("hf:///name", "Not an HF checkpoint URI"),
    ],
)
def test_parse_hf_uri_rejects_ambiguous_or_malformed(uri, match):
    with pytest.raises(ValueError, match=match):
        parse_hf_uri(uri)


def test_default_revision_pins_mulligan_repos(release_dir):
    from mulligan.release.hub import default_revision

    assert default_revision(REPO) == REVISION  # listed in revisions.json
    assert default_revision("someone/own-model") is None  # default branch
    with pytest.raises(LookupError, match="Pin the revision explicitly"):
        default_revision("mulligan/not-released")


def test_default_revision_without_release_checkout(monkeypatch, tmp_path):
    """An installed package has no release/ dir: other orgs' hf:// ids resolve without
    load_revisions()."""
    from mulligan.release import download
    from mulligan.release.hub import default_revision

    monkeypatch.setattr(download, "RELEASE_DIR", tmp_path / "absent")
    download._load.cache_clear()
    try:
        assert default_revision("someone/own-model") is None
        # A mulligan/* id still needs the pin and says where to get it.
        with pytest.raises(FileNotFoundError, match="checkout of the release repo"):
            default_revision("mulligan/real-marker-d2-models")
    finally:
        download._load.cache_clear()


def test_real_model_ids_use_the_same_parser():
    from mulligan.real.policy.dp import HFModelRef, parse_hf_model_id, resolve_model_id
    from mulligan.release.download import load_revisions

    assert parse_hf_model_id(f"hf://{REPO}@{REVISION}/seed-1") == HFModelRef(
        REPO, "seed-1", REVISION
    )
    assert parse_hf_model_id(f"hf://{REPO}/seed-1@{REVISION}") == HFModelRef(
        REPO, "seed-1", REVISION
    )
    with pytest.raises(ValueError, match="ambiguous"):
        parse_hf_model_id(f"hf://{REPO}/a@{REVISION}/b")
    repo = next(r for r, e in load_revisions().items() if e["type"] == "model")
    assert (
        resolve_model_id(f"hf://{repo}/sub")
        == f"hf://{repo}/sub@{load_revisions()[repo]['revision']}"
    )


@pytest.fixture()
def release_dir(monkeypatch, tmp_path):
    from mulligan.release import download

    root = tmp_path / "release"
    root.mkdir()
    (root / "revisions.json").write_text(
        json.dumps({"repos": {REPO: {"type": "model", "revision": REVISION, "tag": None}}})
    )
    monkeypatch.setattr(download, "RELEASE_DIR", root)
    download._load.cache_clear()
    yield root
    download._load.cache_clear()


def test_release_revision_reads_revisions_json(release_dir):
    assert release_revision(REPO) == REVISION
    with pytest.raises(LookupError, match="Pin the revision explicitly"):
        release_revision("example-org/unreleased")


def test_resolve_checkpoint_local_dir_and_errors(tmp_path):
    assert resolve_checkpoint(tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError, match="neither a local directory"):
        resolve_checkpoint(tmp_path / "missing")


def test_resolve_checkpoint_routes_wandb_artifacts(monkeypatch, tmp_path):
    seen = []

    def fake_download(identifier, download_dir=None):
        seen.append(identifier)
        return tmp_path

    monkeypatch.setattr(hub, "download_checkpoint_from_wandb", fake_download)
    assert resolve_checkpoint("entity/project/idql-final:v3") == tmp_path
    assert resolve_checkpoint("wandb://entity/project/idql-final:v3") == tmp_path
    assert seen == ["entity/project/idql-final:v3", "wandb://entity/project/idql-final:v3"]


def test_resolve_checkpoint_cache_dir_is_a_root_for_wandb(monkeypatch, tmp_path):
    """cache_dir is a root: two W&B artifacts resolved with one cache_dir land in separate
    directories."""
    dirs = []

    def fake_download(identifier, download_dir=None):
        dirs.append(download_dir)
        return download_dir

    monkeypatch.setattr(hub, "download_checkpoint_from_wandb", fake_download)
    a = resolve_checkpoint("entity/project/idql-final:v3", cache_dir=tmp_path)
    b = resolve_checkpoint("wandb://entity/project/dp-final:v1", cache_dir=tmp_path)
    assert a != b and a.parent == tmp_path and b.parent == tmp_path
    assert resolve_checkpoint("entity/project/idql-final:v3") is None  # managed default


def test_resolve_checkpoint_hf_uses_pin(monkeypatch, tmp_path, release_dir):
    calls = []

    def fake_snapshot_download(repo_id, revision=None, allow_patterns=None, cache_dir=None):
        calls.append((repo_id, revision, allow_patterns))
        (tmp_path / "seed-1").mkdir(exist_ok=True)
        (tmp_path / "seed-1" / "policy.pt").write_bytes(b"x")
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert resolve_checkpoint(f"hf://{REPO}/seed-1") == tmp_path / "seed-1"
    assert calls == [(REPO, REVISION, ["seed-1/*"])]
    with pytest.raises(FileNotFoundError, match="has no files"):
        resolve_checkpoint(f"hf://{REPO}@{REVISION}/seed-9")
