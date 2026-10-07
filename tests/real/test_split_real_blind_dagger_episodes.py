import argparse
import json
from pathlib import Path

import pytest

from mulligan.data.split_copy import build_frame_lookup_by_index, last_frame_success
from mulligan.real.data import split as splitter
from mulligan.real.data.split import _load_rows, _load_sidecar_rows


class _ParserCaptured(Exception):
    pass


def test_split_cli_has_no_arena_registration(monkeypatch) -> None:
    defaults = {}

    def capture_defaults(parser: argparse.ArgumentParser):
        defaults.update({action.dest: action.default for action in parser._actions})
        raise _ParserCaptured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_defaults)
    with pytest.raises(_ParserCaptured):
        splitter.main()

    assert defaults["push"] is False
    assert not [dest for dest in defaults if "arena" in dest]


class _FakeEpisodes:
    rows = [{"dataset_from_index": 0, "dataset_to_index": 2}]

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class _FakeMeta:
    episodes = _FakeEpisodes()


class _FakeDataset:
    hf_dataset = [
        {"index": 0, "success": 0},
        {"index": 1, "success": 0},
        {"index": 1, "success": 1},
    ]
    num_frames = 2
    meta = _FakeMeta()


def test_frame_lookup_uses_latest_duplicate_global_index() -> None:
    ds = _FakeDataset()
    lookup = build_frame_lookup_by_index(ds)

    assert sorted(lookup) == [0, 1]
    assert lookup[1]["success"] == 1
    assert last_frame_success(ds, 0, frame_lookup=lookup) is True


def test_ledger_rows_must_be_contiguous(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "episode_index": 1,
                "arm_key": "baseline_uniform",
                "success": True,
                "outcome": "success",
                "steps": 10,
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="contiguous"):
        _load_rows(ledger)


def test_teleop_ledger_can_split_by_manifest_source(tmp_path: Path) -> None:
    ledger = tmp_path / "teleop_manifest_ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "manifest_source": "mulligan_sobol",
                "success": True,
                "steps": 12,
            }
        )
        + "\n"
    )

    rows = _load_rows(
        ledger,
        split_key="manifest_source",
        outcome_key="outcome",
        default_outcome="success",
    )

    assert rows == [
        {
            "episode_index": 0,
            "manifest_source": "mulligan_sobol",
            "success": True,
            "steps": 12,
            "outcome": "success",
        }
    ]


def test_existing_sidecar_rows_must_be_contiguous(tmp_path: Path) -> None:
    sidecar = tmp_path / "split.jsonl"
    sidecar.write_text(json.dumps({"episode_index": 1, "source_episode_index": 3}) + "\n")

    with pytest.raises(SystemExit, match="contiguous"):
        _load_sidecar_rows(sidecar)


def test_append_to_existing_target_reopens_in_write_mode(tmp_path: Path) -> None:
    # --append-to-existing --copy-backend frame adds frames to a finalized target; the bare
    # LeRobotDataset constructor is read-only in the pinned LeRobot.
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "target"
    features = {"observation.state": {"dtype": "float32", "shape": (2,), "names": None}}

    def add_episode(ds) -> None:
        for _ in range(3):
            ds.add_frame({"observation.state": np.zeros(2, dtype=np.float32), "task": "t"})
        ds.save_episode()

    created = LeRobotDataset.create(
        repo_id="local/target", fps=15, root=str(root), features=features, use_videos=False
    )
    add_episode(created)
    created.finalize()

    reopened = splitter._load_or_create_target_dataset(
        None,
        "local/target",
        root,
        drop_visual_features=True,
        append_to_existing=True,
    )
    assert reopened.num_episodes == 1
    add_episode(reopened)
    reopened.finalize()

    assert LeRobotDataset("local/target", root=str(root)).meta.total_episodes == 2
