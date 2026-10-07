"""Per-frame episode copy of the dataset splitters on small local LeRobot datasets.

Covers ``mulligan.data.split_blind``, ``mulligan.real.data.split`` (``--copy-backend
frame``) and the copy step of ``mulligan.data.split_protocol_quota``: copied rows and
labels, task metadata, visual vs ``--drop-visual-features`` outputs, the global-index
frame lookup (duplicate and missing indices), the real splitter's ``intervention``
column and target ``robot_type`` defaults, and the real splitter's lineage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data import split_blind, split_protocol_quota
from mulligan.real.data import split as real_split
from mulligan.utils.state_to_grid import extract_sampler_state_from_env_state
from tests.unit.sim_dataset import build_sim_dataset

IMAGE = "observation.images.cam"
IMAGE_FEATURE = {"dtype": "image", "shape": (4, 5, 3), "names": ["height", "width", "channels"]}
LENGTHS = (2, 3, 2)  # + one padded frame each: global frames [0, 3), [3, 7), [7, 10)
FAILED_EPISODE = 1
ARMS = ("a", "b", "a")
TASK = "marker_d2"
AUTO = ["episode_index", "index", "task_index"]
ROBOT_TYPE = "test_arm"


def _image(ep: int, t: int) -> np.ndarray:
    grid = np.arange(4 * 5 * 3, dtype=np.int64).reshape(4, 5, 3)
    return ((grid * 7 + 40 * ep + 11 * t) % 256).astype(np.uint8)


def _labels(ep: int, t: int) -> dict:
    return {
        "task": TASK,
        "source": np.array([(ep + t) % 2], np.int64),
        "success": np.array([int(ep != FAILED_EPISODE)], np.int64),
    }


def _overrides(ep: int, t: int, length: int) -> dict:
    return {**_labels(ep, t), IMAGE: _image(ep, t)}


@pytest.fixture
def source(tmp_path) -> Path:
    return build_sim_dataset(
        tmp_path / "source",
        episode_lengths=LENGTHS,
        seed=3,
        extra_features={IMAGE: IMAGE_FEATURE},
        overrides=_overrides,
    )


def _parquet(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _write_parquet(root: Path, frame: pd.DataFrame) -> None:
    (path,) = sorted((root / "data").rglob("*.parquet"))
    frame.to_parquet(path, index=False)


def _plain(frame: pd.DataFrame) -> list[dict]:
    columns = [c for c in frame.columns if c not in AUTO and c != IMAGE]
    return [
        {c: np.asarray(row[c]).tolist() for c in columns}
        for _, row in frame.sort_values("index").iterrows()
    ]


def _source_rows(frame: pd.DataFrame, episodes: list[int]) -> list[dict]:
    frame = frame.drop_duplicates("index", keep="last")
    return _plain(frame[frame["episode_index"].isin(episodes)])


def _images(root: Path) -> list[torch.Tensor]:
    ds = LeRobotDataset(f"local/{root.name}", root=root)
    return [ds[i][IMAGE] for i in range(ds.num_frames)]


def _assert_images(root: Path, src_images: list[torch.Tensor], episodes: list[int]) -> None:
    spans = (range(0, 3), range(3, 7), range(7, 10))
    expected = [src_images[i] for ep in episodes for i in spans[ep]]
    got = _images(root)
    assert len(got) == len(expected)
    assert all(torch.equal(g, e) for g, e in zip(got, expected))


def _tasks(root: Path) -> list[str]:
    return [str(t) for t in LeRobotDataset(f"local/{root.name}", root=root).meta.tasks.index]


def _set_robot_type(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = ROBOT_TYPE
    info_path.write_text(json.dumps(info, indent=4))


def _insert_stale_row(root: Path, global_index: int) -> pd.DataFrame:
    """Put a stale copy of one global frame before the canonical row (a resumed-file layout)."""
    canonical = _parquet(root)
    pos = int(np.flatnonzero(canonical["index"].to_numpy() == global_index)[0])
    stale = canonical.iloc[[pos]].copy()
    for column in ("action", "observation.environment_state"):
        stale[column] = [np.asarray(v) + 100.0 for v in stale[column]]
    _write_parquet(root, pd.concat([canonical.iloc[:pos], stale, canonical.iloc[pos:]]))
    return canonical


# --- mulligan.real.data.split -------------------------------------------------------


def _real_ledger(tmp_path: Path) -> Path:
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        {
            "episode_index": ep,
            "arm_key": arm,
            "success": ep != FAILED_EPISODE,
            "outcome": "success" if ep != FAILED_EPISODE else "failure",
            "steps": LENGTHS[ep],
            "model_id": f"hf://org/{arm}",
            "task_name": TASK,
        }
        for ep, arm in enumerate(ARMS)
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return ledger


def _run_real(monkeypatch, source: Path, ledger: Path, out: Path, *extra: str) -> int:
    argv = [
        "split",
        "--source-repo=local/source",
        f"--source-root={source}",
        f"--ledger={ledger}",
        "--expected-success-per-arm=a=2,b=0",
        "--target=a=local/arm-a",
        "--target=b=local/arm-b",
        f"--output-root={out}",
        "--copy-backend=frame",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return real_split.main()


@pytest.mark.parametrize("drop_visual", [True, False], ids=["nonvisual", "visual"])
def test_real_frame_split_copies_rows_labels_task_and_lineage(
    source, tmp_path, monkeypatch, drop_visual
):
    _set_robot_type(source)
    src = _parquet(source)
    src_images = _images(source)
    ledger = _real_ledger(tmp_path)
    out = tmp_path / "out"
    extra = ["--drop-visual-features"] if drop_visual else []
    assert _run_real(monkeypatch, source, ledger, out, *extra) == 0

    for arm, episodes in (("a", [0, 2]), ("b", [1])):
        root = out / f"arm-{arm}"
        got = _parquet(root)
        # The source has no intervention column; the real splitter adds one, all zeros.
        assert [np.asarray(v).tolist() for v in got["intervention"]] == [0] * len(got)
        assert _plain(got.drop(columns=["intervention"])) == _source_rows(src, episodes)
        assert sorted(got["episode_index"].unique().tolist()) == list(range(len(episodes)))
        assert (IMAGE in got.columns) is not drop_visual
        if not drop_visual:
            _assert_images(root, src_images, episodes)
        info = json.loads((root / "meta" / "info.json").read_text())
        assert info["robot_type"] == ROBOT_TYPE
        assert info["features"]["intervention"]["names"] == ["intervention_flag"]
        assert _tasks(root) == [TASK]

        sidecar = [
            json.loads(line)
            for line in (root / "meta" / "real_blind_dagger_split_manifest.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["source_episode_index"] for r in sidecar] == episodes
        assert [r["source_success"] for r in sidecar] == [e != FAILED_EPISODE for e in episodes]
        lineage = json.loads((root / "meta" / "dataset_lineage.json").read_text())
        assert {k: lineage[k] for k in ("role", "task", "parent_repo_id", "split_key")} == {
            "role": "derived_split",
            "task": TASK,
            "parent_repo_id": "local/source",
            "split_key": "arm_key",
        }
        assert lineage["derivation_script"] == "mulligan.real.data.split"
        assert lineage["view_family_id"] == f"local/source::arm_key:{arm}"
        assert lineage["producer_model_ids"] == [f"hf://org/{arm}"]
    parent = json.loads((source / "meta" / "dataset_lineage.json").read_text())
    assert parent["derived_repos"] == ["local/arm-a", "local/arm-b"]


def test_real_frame_split_keeps_an_existing_intervention_column(tmp_path, monkeypatch):
    source = build_sim_dataset(
        tmp_path / "source",
        episode_lengths=LENGTHS,
        extra_features={"intervention": {"dtype": "int64", "shape": (1,), "names": None}},
        overrides=lambda ep, t, n: {
            **_labels(ep, t),
            "intervention": np.array([int(t == 1)], np.int64),
        },
    )
    assert _run_real(monkeypatch, source, _real_ledger(tmp_path), tmp_path / "out") == 0
    got = _parquet(tmp_path / "out" / "arm-a")
    assert _plain(got) == _source_rows(_parquet(source), [0, 2])
    assert [np.asarray(v).tolist() for v in got["intervention"]] == [0, 1, 0] * 2


def test_real_frame_split_reads_the_last_row_of_a_duplicate_global_index(
    source, tmp_path, monkeypatch, capsys
):
    canonical = _insert_stale_row(source, 7)
    out = tmp_path / "out"
    ledger = _real_ledger(tmp_path)
    assert _run_real(monkeypatch, source, ledger, out, "--drop-visual-features") == 0
    assert "source dataset has 1 duplicate global frame rows" in capsys.readouterr().out
    got = _parquet(out / "arm-a").drop(columns=["intervention"])
    assert _plain(got) == _source_rows(canonical, [0, 2])


def test_real_frame_split_fails_on_a_missing_global_index(source, tmp_path, monkeypatch):
    frame = _parquet(source)
    _write_parquet(source, frame[frame["index"] != 5])
    with pytest.raises(RuntimeError, match=r"missing 1 global frame indices; .*\[5\]"):
        _run_real(
            monkeypatch, source, _real_ledger(tmp_path), tmp_path / "out", "--drop-visual-features"
        )


# --- mulligan.data.split_blind ------------------------------------------------------


def _blind_manifest(tmp_path: Path, source: Path) -> Path:
    frame = _parquet(source).drop_duplicates("index", keep="last").sort_values("index")
    firsts = frame.groupby("episode_index", sort=True).first()
    keys = ["nut_x", "nut_y", "nut_yaw"]
    states = []
    for ep, arm in enumerate(ARMS):
        env_state = np.asarray(firsts.loc[ep, "observation.environment_state"])
        values = extract_sampler_state_from_env_state(env_state, task="NutAssemblySquare")
        states.append({**dict(zip(keys, map(float, values))), "source": arm})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"task": "square_narrow", "keys": keys, "states": states}))
    return manifest


def _run_blind(source: Path, manifest: Path, out: Path, *extra: str) -> int:
    return split_blind.main(
        [
            "--source-repo=local/source",
            f"--source-root={source}",
            f"--manifest={manifest}",
            "--target=a=local/blind-a",
            "--target=b=local/blind-b",
            "--expected-per-source=a=2,b=0",
            f"--output-root={out}",
            *extra,
        ]
    )


@pytest.mark.parametrize("drop_visual", [True, False], ids=["nonvisual", "visual"])
def test_blind_split_copies_successful_rows_labels_and_task(source, tmp_path, drop_visual):
    _set_robot_type(source)
    src = _parquet(source)
    src_images = _images(source)
    out = tmp_path / "out"
    extra = ["--drop-visual-features"] if drop_visual else []
    assert _run_blind(source, _blind_manifest(tmp_path, source), out, *extra) == 0

    got = _parquet(out / "blind-a")
    assert "intervention" not in got.columns
    assert _plain(got) == _source_rows(src, [0, 2])
    assert (IMAGE in got.columns) is not drop_visual
    if not drop_visual:
        _assert_images(out / "blind-a", src_images, [0, 2])
    assert (
        json.loads((out / "blind-a" / "meta" / "info.json").read_text())["robot_type"] == ROBOT_TYPE
    )
    assert _tasks(out / "blind-a") == [TASK]
    assert json.loads((out / "blind-b" / "meta" / "info.json").read_text())["total_episodes"] == 0


def test_blind_split_reads_the_last_row_of_a_duplicate_global_index(source, tmp_path, capsys):
    manifest = _blind_manifest(tmp_path, source)
    canonical = _insert_stale_row(source, 7)  # the first frame, which is matched to the manifest
    out = tmp_path / "out"
    assert _run_blind(source, manifest, out, "--drop-visual-features") == 0
    assert "source dataset has 1 duplicate global frame rows" in capsys.readouterr().out
    assert _plain(_parquet(out / "blind-a")) == _source_rows(canonical, [0, 2])


def test_blind_split_fails_on_a_missing_global_index(source, tmp_path):
    manifest = _blind_manifest(tmp_path, source)
    frame = _parquet(source)
    _write_parquet(source, frame[frame["index"] != 5])
    with pytest.raises(RuntimeError, match=r"missing 1 global frame indices; .*\[5\]"):
        _run_blind(source, manifest, tmp_path / "out", "--drop-visual-features")


# --- mulligan.data.split_protocol_quota (frame copy step) ---------------------------


@pytest.mark.parametrize("drop_visual", [True, False], ids=["nonvisual", "visual"])
def test_protocol_quota_frame_copy_preserves_rows_labels_and_task(source, tmp_path, drop_visual):
    ds = LeRobotDataset("local/source", root=source)
    src = _parquet(source)
    src_images = _images(source)
    features = dict(ds.features)
    if drop_visual:
        features.pop(IMAGE)
    target = LeRobotDataset.create(
        repo_id="local/pq",
        fps=ds.fps,
        root=str(tmp_path / "pq"),
        robot_type="panda",
        features=features,
        use_videos=not drop_visual,
    )
    for ep in (2, 0):
        split_protocol_quota._copy_episode(ds, ep, target, drop_visual_features=drop_visual)
    target.finalize()

    got = _parquet(tmp_path / "pq")
    assert _plain(got) == _source_rows(src, [2]) + _source_rows(src, [0])
    assert (IMAGE in got.columns) is not drop_visual
    if not drop_visual:
        _assert_images(tmp_path / "pq", src_images, [2, 0])
    assert _tasks(tmp_path / "pq") == [TASK]
