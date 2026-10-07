"""Replay operator: episode matching and the key/action script it plays back."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulligan.sim.collect.replay_operator import load_recorded_episodes, make_replay_operator
from mulligan.sim.collect.utils import EpisodeStart, Operator


def _env_state(x: float, y: float, yaw: float) -> list[float]:
    state = np.zeros(14)
    state[7:10] = [x, y, 0.83]
    state[10:14] = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]  # xyzw
    return state.tolist()


def _write_dataset(
    root: Path, episodes: list[tuple[tuple, list[tuple[int, int]], bool, bool]]
) -> None:
    """episodes: (start (x, y, yaw), [(source, n_frames), ...], success, done).

    Like the collector, every episode ends with one padded ``is_valid == 0`` frame;
    its source label differs from the last recorded one, so replaying it would add
    a one-step segment.
    """
    rows, index = [], 0
    for episode_index, (start, segments, success, done) in enumerate(episodes):
        frames = [src for src, n in segments for _ in range(n)]
        frames.append(1 - frames[-1])
        for frame_index, source in enumerate(frames):
            last = frame_index == len(frames) - 1
            action = [0.1 * (frame_index + 1)] * 6 + [-1.0 if frame_index % 2 else 0.7]
            rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "index": index,
                    "source": source,
                    "action": action,
                    "observation.environment_state": _env_state(*start),
                    "success": int(success),
                    "done": int(done and last),
                    "is_valid": int(not last),
                }
            )
            index += 1
    path = root / "data" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


START_A = (-0.112, 0.15, 0.3)
START_B = (-0.113, 0.20, -2.0)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _write_dataset(
        tmp_path,
        [
            (START_A, [(0, 3), (1, 2), (0, 2)], True, True),
            (START_A, [(1, 2), (0, 1)], True, True),  # counterfactual of episode 0
            (START_B, [(0, 2), (1, 1)], False, True),  # terminal failure after a takeover
        ],
    )
    return tmp_path


def _start(state, *, cf: bool = False, index: int = 0) -> EpisodeStart:
    return EpisodeStart(
        episode_index=index,
        start_state=state,
        manifest_idx=None,
        is_counterfactual=cf,
        human_first=cf,
    )


def _run(op, start: EpisodeStart, human_first: bool, max_steps: int = 50) -> list[str]:
    """Drive the operator like the collector does; returns the key trace."""
    op.begin_episode(start)
    phase = "human" if human_first else "policy"
    trace = []
    for _ in range(max_steps):
        key = op.poll_key(phase)
        if key is None:
            if phase == "human":
                action = op.human_input().action
                assert action[6] in (-1.0, 1.0)
                trace.append("H")
            else:
                trace.append("P")
            continue
        trace.append(key)
        if key == "h":
            if phase == "policy":
                op.begin_human_segment(1.0)
            phase = "human" if phase == "policy" else "policy"
            continue
        return trace
    return trace + ["<budget>"]


def test_segments_and_start_poses_are_read(dataset: Path):
    episodes = load_recorded_episodes(dataset)
    assert [ep.episode_index for ep in episodes] == [0, 1, 2]
    assert [(s.source, len(s.actions)) for s in episodes[0].segments] == [(0, 3), (1, 2), (0, 2)]
    np.testing.assert_allclose(episodes[0].nut, START_A, atol=1e-9)
    assert episodes[1].human_first and not episodes[0].human_first


def test_replay_script_follows_the_recording(dataset: Path):
    op = make_replay_operator(root=str(dataset), max_episode_steps=12)
    assert isinstance(op, Operator)
    # Episode 0 succeeded; without auto-save the replayed policy runs to the step budget.
    trace = _run(op, _start(START_A), human_first=False)
    assert trace[:9] == ["P", "P", "P", "h", "H", "H", "h", "P", "P"]
    assert trace[-1] == "0" and len(trace) == 12 + 3
    # The unused counterfactual of the same start is requested next.
    assert op.choose("ncq") == "c"
    assert op.choose("nq") == "n"
    trace = _run(op, _start(START_A, cf=True), human_first=True)
    assert trace[:4] == ["H", "H", "h", "P"]
    assert op.choose("ncq") == "n"
    # Recorded terminal failure is replayed as '9'.
    assert _run(op, _start(START_B), human_first=False) == ["P", "P", "h", "H", "9"]


def test_redrawn_start_reuses_its_policy_first_recording(dataset: Path):
    op = make_replay_operator(root=str(dataset))
    assert _run(op, _start(START_B), human_first=False) == ["P", "P", "h", "H", "9"]
    # The quota redraws a start whose fresh episode failed: the recording is replayed again.
    assert _run(op, _start(START_B), human_first=False) == ["P", "P", "h", "H", "9"]
    assert op.current.episode_index == 2
    # Counterfactual recordings stay use-once.
    _run(op, _start(START_A, cf=True), human_first=True)
    assert op.choose("ncq") == "n"
    with pytest.raises(LookupError, match="no unused human-first recorded episode"):
        op.begin_episode(_start(START_A, cf=True))


def test_padded_frame_is_not_replayed(dataset: Path):
    episodes = load_recorded_episodes(dataset)
    # The padded frame of episode 2 (source 0 after a human segment) is dropped.
    assert [(s.source, len(s.actions)) for s in episodes[2].segments] == [(0, 2), (1, 1)]
    assert [ep.success for ep in episodes] == [True, True, False]
    assert [ep.done for ep in episodes] == [True, True, True]


def test_collector_refuses_replay_without_auto_save(dataset: Path, tmp_path: Path):
    from mulligan.sim.collect import dagger

    args = dagger.parse_args(
        ["--policy=unused", f"--dataset-path={tmp_path / 'out'}", "--dataset-name=x", "--headless"]
    )
    with pytest.raises(ValueError, match="needs --auto-save-on-success"):
        dagger.DaggerCollector(
            args,
            operator=make_replay_operator(root=str(dataset)),
            actor=None,
            env_name="NutAssemblySquare",
            robot_name="Panda",
        )


def test_unknown_start_and_episode_limit(dataset: Path):
    op = make_replay_operator(root=str(dataset), max_episodes=1)
    with pytest.raises(LookupError, match="no policy-first recorded episode"):
        op.begin_episode(_start((-0.1, 0.1, 1.0)))
    op.begin_episode(_start(START_B))
    assert op.choose("ncq") == "q"
    with pytest.raises(ValueError, match="pinned revision"):
        make_replay_operator(repo_id="mulligan/sim-square-narrow-c01-dagger-mixed")


def test_collector_replays_a_redrawn_start(tmp_path: Path, capsys):
    """The real collector (headless, protocol quota) redraws a start whose fresh episode
    failed; the replay operator replays that start's policy-first recording again."""
    from mulligan.sim.collect import dagger
    from tests.sim.test_dagger_operator import (
        STATES,
        FakeActor,
        _args,
        _read_frames,
        _write_routing_manifest,
        _write_states,
    )

    # One start, so the quota's redraw after a failed fresh episode is that start.
    start = (STATES[0]["nut_x"], STATES[0]["nut_y"], STATES[0]["nut_yaw"])
    recordings = tmp_path / "recorded"
    # A policy-first episode that ends in a recoverable failure after a takeover,
    # and its counterfactual (also a failure).
    _write_dataset(
        recordings,
        [(start, [(0, 3), (1, 2)], False, False), (start, [(1, 2), (0, 1)], False, False)],
    )
    manifest = tmp_path / "manifest.json"
    _write_routing_manifest(manifest, ["ours"])
    states_file = tmp_path / "states.json"
    _write_states(states_file, STATES[:1])
    ledger = tmp_path / "ledger.jsonl"
    args = _args(
        tmp_path,
        "--sampler",
        "list",
        "--initial-states-file",
        str(states_file),
        "--no-sampler-shuffle",
        "--adaptive-protocol-quota-manifest",
        str(manifest),
        "--adaptive-protocol-quota-targets",
        "no_cf=1,with_cf=1",
        "--adaptive-protocol-quota-arms",
        "no_cf=ours;with_cf=ours",
        "--adaptive-protocol-quota-ledger",
        str(ledger),
        "--auto-save-on-success",
    )
    operator = make_replay_operator(root=str(recordings), max_episodes=3)
    collector = dagger.DaggerCollector(
        args,
        operator=operator,
        actor=FakeActor("single", []),
        env_name="NutAssemblySquare",
        robot_name="Panda",
    )
    collector.setup()
    collector.run()

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["success"] for row in rows] == [False, False, False]
    assert [row["is_counterfactual"] for row in rows] == [False, True, False]
    # Episode 2 is the quota's redraw of the failed fresh start of episode 0.
    assert [row["manifest_idx"] for row in rows] == [0, 0, 0]
    assert operator.episodes_started == 3
    assert "[replay] episode 2: recorded episode 0 (P3 H2) (reused)" in capsys.readouterr().out
    frames = _read_frames(tmp_path / "data" / "dagger-test")
    assert sorted(frames["episode_index"].unique()) == [0, 1, 2]


@pytest.mark.network
def test_released_collection_replays_through_a_redraw(tmp_path: Path, capsys):
    """Replay the released Square-Narrow C01 recording through the real collector: a
    fresh episode fails (a fake policy drives), its counterfactual is replayed, and the
    quota's redraw of the start replays the policy-first recording again."""
    from huggingface_hub import snapshot_download

    from mulligan.sim import recipes as R
    from mulligan.sim.collect import dagger
    from tests.sim.test_dagger_operator import FakeActor, _args, _write_states

    source = R.replay_source("mulligan/sim-square-narrow-c01-dagger-mixed")
    operator = make_replay_operator(**source, max_episodes=3)
    episodes = operator.episodes
    # Every episode's padded is_valid==0 frame is dropped from the replay.
    root = snapshot_download(
        repo_id=source["repo_id"],
        repo_type="dataset",
        revision=source["revision"],
        allow_patterns=["data/*/*.parquet", "meta/info.json"],
    )
    counts = pd.concat(
        [
            pd.read_parquet(f, columns=["episode_index"])
            for f in sorted(Path(root).glob("data/*/*.parquet"))
        ]
    )["episode_index"].value_counts()
    assert len(episodes) == len(counts) > 100
    for ep in episodes:
        assert sum(len(s.actions) for s in ep.segments) == counts[ep.episode_index] - 1

    # A start with a single policy-first recording and a counterfactual recording.
    by_start: dict = {}
    for ep in episodes:
        by_start.setdefault(tuple(round(v, 4) for v in ep.nut), []).append(ep)
    recorded = next(
        eps[0]
        for _, eps in sorted(by_start.items())
        if [ep.human_first for ep in eps] == [False, True]
    )
    state = dict(zip(("nut_x", "nut_y", "nut_yaw"), recorded.nut))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "square_narrow",
                "keys": ["nut_x", "nut_y", "nut_yaw"],
                "match_tolerance": 1e-3,
                "states": [{**state, "policy_source": "ours", "sources": ["ours"]}],
            }
        )
    )
    states_file = tmp_path / "states.json"
    _write_states(states_file, [state])
    ledger = tmp_path / "ledger.jsonl"
    args = _args(
        tmp_path,
        "--sampler=list",
        f"--initial-states-file={states_file}",
        "--no-sampler-shuffle",
        f"--adaptive-protocol-quota-manifest={manifest}",
        "--adaptive-protocol-quota-targets=no_cf=1,with_cf=1",
        "--adaptive-protocol-quota-arms=no_cf=ours;with_cf=ours",
        f"--adaptive-protocol-quota-ledger={ledger}",
        "--auto-save-on-success",
    )
    collector = dagger.DaggerCollector(
        args,
        operator=operator,
        actor=FakeActor("fake", []),
        env_name="NutAssemblySquare",
        robot_name="Panda",
    )
    collector.setup()
    collector.run()

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["is_counterfactual"] for row in rows] == [False, True, False]
    # The fresh episode failed (the redraw needs that); the open-loop counterfactual may succeed.
    assert not rows[0]["success"]
    assert [row["manifest_idx"] for row in rows] == [0, 0, 0]
    out = capsys.readouterr().out
    assert f"[replay] episode 0: recorded episode {recorded.episode_index} " in out
    assert f"[replay] episode 2: recorded episode {recorded.episode_index} " in out
    assert "(reused)" in out
