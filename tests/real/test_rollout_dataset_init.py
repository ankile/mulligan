"""First save / resume of the rollout dataset (mulligan.real.collect.rollout)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lerobot.datasets.lerobot_dataset as lerobot_dataset
from mulligan.real.collect import rollout


class _FakeDataset:
    """Mimics the LeRobot 0.5 root contract: create() refuses an existing root and
    loading needs meta/info.json."""

    def __init__(self, repo_id: str, root: str):
        self.root = Path(root)
        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(self.root / "meta" / "info.json")
        self.num_episodes = json.loads((self.root / "meta" / "info.json").read_text())["n"]

    @classmethod
    def create(cls, *, repo_id: str, root: str, **_kwargs):
        root_path = Path(root)
        (root_path / "meta").mkdir(parents=True, exist_ok=False)
        (root_path / "meta" / "info.json").write_text(json.dumps({"n": 0}))
        return cls(repo_id, root)


@pytest.fixture
def calls(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", _FakeDataset)
    monkeypatch.setattr(rollout, "build_real_lerobot_features", lambda *a, **k: {})
    monkeypatch.setattr(
        rollout, "ensure_dataset_can_store_episode_telemetry", lambda dataset, *a, **k: dataset
    )
    monkeypatch.setattr(
        rollout,
        "record_or_verify_camera_role_serials",
        lambda dataset, keys: seen.append(str(dataset.root)),
    )
    return seen


def _open(path: Path):
    return rollout.open_or_create_rollout_dataset(
        dataset_path=path,
        dataset_name="local/rollouts",
        episode_data={},
        cam_data_keys=["image_20000002_left"],
        fps=15,
    )


def test_first_save_after_idql_chunk_info_sidecar_creates_dataset(tmp_path, calls):
    # The IDQL sidecar lands under dataset_path before the first episode is saved.
    path = tmp_path / "rollouts"
    (path / rollout.CHUNK_INFO_DIRNAME).mkdir(parents=True)
    (path / rollout.CHUNK_INFO_DIRNAME / "episode_0000.jsonl").write_text("{}\n")

    dataset = _open(path)

    assert (path / "meta" / "info.json").exists()
    assert (path / rollout.CHUNK_INFO_DIRNAME / "episode_0000.jsonl").read_text() == "{}\n"
    assert not path.with_name(f"rollouts.{rollout.CHUNK_INFO_DIRNAME}.precreate").exists()
    assert dataset.num_episodes == 0
    assert calls == [str(path)]


def test_resume_verifies_camera_role_serials(tmp_path, calls):
    path = tmp_path / "rollouts"
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text(json.dumps({"n": 3}))

    dataset = _open(path)

    assert dataset.num_episodes == 3
    assert calls == [str(path)]


def test_create_refuses_unknown_pre_dataset_content(tmp_path, calls):
    path = tmp_path / "rollouts"
    (path / "data").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="without meta/info.json"):
        _open(path)
    assert calls == []
