import json
from types import SimpleNamespace

from huggingface_hub.errors import EntryNotFoundError
import pytest

from mulligan.real.collect.hub_resume_guard import (
    assert_local_dataset_not_behind_hub,
    assert_local_episode_count_matches_ledger,
)


class _FakeApi:
    def __init__(self, repo_files=None):
        self.repo_files = list(repo_files or [])

    def list_repo_files(self, **_kwargs):
        return self.repo_files


def _write_info(root, total_episodes: int) -> None:
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(json.dumps({"total_episodes": total_episodes}))


def test_hub_resume_guard_rejects_shorter_local_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 4)

    with pytest.raises(RuntimeError, match="local dataset has 4 episode"):
        assert_local_dataset_not_behind_hub(
            repo_id="my-org/real-square-d2",
            dataset_path=dataset_path,
            api=_FakeApi(),
            remote_info_loader=lambda repo_id, revision: {"total_episodes": 234},
        )


def test_hub_resume_guard_allows_local_equal_to_remote(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 234)

    check = assert_local_dataset_not_behind_hub(
        repo_id="my-org/real-square-d2",
        dataset_path=dataset_path,
        api=_FakeApi(),
        remote_info_loader=lambda repo_id, revision: {"total_episodes": 234},
    )

    assert check.repo_id == "my-org/real-square-d2"
    assert check.local_total_episodes == 234
    assert check.remote_total_episodes == 234


def test_hub_resume_guard_keeps_full_repo_id(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 250)

    seen = SimpleNamespace(repo_id=None)

    def load_info(repo_id: str, revision: str) -> dict:
        seen.repo_id = repo_id
        return {"total_episodes": 234}

    check = assert_local_dataset_not_behind_hub(
        repo_id="someone/dataset",
        dataset_path=dataset_path,
        api=_FakeApi(),
        remote_info_loader=load_info,
    )

    assert seen.repo_id == "someone/dataset"
    assert check.repo_id == "someone/dataset"
    assert check.local_total_episodes == 250


def test_hub_resume_guard_allows_empty_repo_when_requested(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 75)

    check = assert_local_dataset_not_behind_hub(
        repo_id="my-org/real-routing-d2",
        dataset_path=dataset_path,
        allow_empty_repo=True,
        api=_FakeApi(repo_files=[".gitattributes"]),
        remote_info_loader=lambda repo_id, revision: (_ for _ in ()).throw(
            EntryNotFoundError("missing")
        ),
    )

    assert check.repo_id == "my-org/real-routing-d2"
    assert check.local_total_episodes == 75
    assert check.remote_total_episodes == 0


def test_hub_resume_guard_rejects_nonempty_repo_without_info(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 75)

    with pytest.raises(RuntimeError, match="non-empty-repo file"):
        assert_local_dataset_not_behind_hub(
            repo_id="my-org/real-routing-d2",
            dataset_path=dataset_path,
            allow_empty_repo=True,
            api=_FakeApi(repo_files=[".gitattributes", "README.md"]),
            remote_info_loader=lambda repo_id, revision: (_ for _ in ()).throw(
                EntryNotFoundError("missing")
            ),
        )


def test_hub_resume_guard_rejects_bare_repo_id(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 1)

    with pytest.raises(ValueError, match="NAMESPACE/NAME"):
        assert_local_dataset_not_behind_hub(
            repo_id="real-square-d2",
            dataset_path=dataset_path,
            api=_FakeApi(),
            remote_info_loader=lambda repo_id, revision: {"total_episodes": 1},
        )


def test_local_episode_count_matches_ledger_allows_equal_count(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 234)

    count = assert_local_episode_count_matches_ledger(
        dataset_path=dataset_path,
        ledger_count=234,
        ledger_path=dataset_path / "meta" / "teleop_manifest_ledger.jsonl",
    )

    assert count == 234


def test_local_episode_count_matches_ledger_rejects_mismatch(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    _write_info(dataset_path, 234)

    with pytest.raises(RuntimeError, match="reports 234 episode"):
        assert_local_episode_count_matches_ledger(
            dataset_path=dataset_path,
            ledger_count=4,
            ledger_path=dataset_path / "meta" / "teleop_manifest_ledger.jsonl",
        )


def test_local_episode_count_matches_ledger_rejects_ledger_without_dataset(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        assert_local_episode_count_matches_ledger(
            dataset_path=tmp_path / "missing",
            ledger_count=4,
            ledger_path=tmp_path / "missing" / "meta" / "teleop_manifest_ledger.jsonl",
        )
