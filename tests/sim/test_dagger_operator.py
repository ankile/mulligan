"""The sim DAgger collector driven by a scripted Operator (no SpaceMouse, no display).

The loop tests run a headless NutAssemblySquare env on CPU with a fake policy,
so they exercise takeovers, hand-backs, counterfactual replays, discards,
blinded routing, the protocol quota and the saved dataset labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulligan.sim.collect import dagger
from mulligan.sim.collect.utils import (
    EpisodeStart,
    HumanInput,
    Operator,
    SpaceMouseKeyboardOperator,
)

STATES = [
    {"nut_x": -0.1125, "nut_y": 0.13, "nut_yaw": 0.3},
    {"nut_x": -0.1140, "nut_y": 0.20, "nut_yaw": -1.2},
    {"nut_x": -0.1110, "nut_y": 0.16, "nut_yaw": 2.0},
]
HUMAN_ACTION = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
POLICY_ACTION = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -1.0])


class FakeActor:
    """Stands in for IDQLActor: a constant action, with a log of calls."""

    def __init__(self, name: str, log: list):
        self.name = name
        self.log = log

    def reset(self) -> None:
        pass

    def act(self, robot_state, env_state):
        assert robot_state.shape == (9,)
        self.log.append(self.name)
        return POLICY_ACTION.copy()


class ScriptedOperator:
    """Plays per-episode scripts of (phase, steps, closing key) segments."""

    def __init__(self, scripts, choices):
        self.scripts = list(scripts)
        self.choices = list(choices)
        self.starts: list[EpisodeStart] = []
        self.segment_gripper: list[float] = []
        self.closed = False
        self._segments: list[list] = []

    def begin_episode(self, start: EpisodeStart) -> None:
        self.starts.append(start)
        script = self.scripts.pop(0)
        if callable(script):
            script = script(start)
        self._segments = [list(segment) for segment in script]

    def poll_key(self, phase):
        segment = self._segments[0]
        assert segment[0] == phase, f"expected {segment[0]} phase, got {phase}"
        if segment[1] == 0:
            self._segments.pop(0)
            return segment[2]
        if phase == "policy":
            segment[1] -= 1
        return None

    def begin_human_segment(self, gripper_action: float) -> None:
        self.segment_gripper.append(gripper_action)

    def human_input(self) -> HumanInput:
        self._segments[0][1] -= 1
        return HumanInput(action=HUMAN_ACTION.copy(), arm_active=True, strong=True)

    def choose(self, options: str) -> str:
        choice = self.choices.pop(0)
        if callable(choice):
            choice = choice(options)
        assert choice in options, f"{choice!r} not offered ({options!r})"
        return choice

    def close(self) -> None:
        self.closed = True


def _read_frames(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "data").rglob("*.parquet"))
    assert files, f"no parquet files under {dataset_dir}"
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frames.sort_values("index").reset_index(drop=True)


def _column(frames: pd.DataFrame, episode: int, name: str) -> list:
    values = frames.loc[frames["episode_index"] == episode, name]
    return [int(np.asarray(v).reshape(-1)[0]) for v in values]


def _args(tmp_path: Path, *extra: str):
    return dagger.parse_args(
        [
            "--policy",
            "unused",
            "--dataset-path",
            str(tmp_path / "data"),
            "--dataset-name",
            "dagger-test",
            "--headless",
            "--max-fr",
            "0",
            "--device",
            "cpu",
            *extra,
        ]
    )


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def test_intervention_flags_mark_last_policy_frame_before_takeover():
    assert dagger.compute_intervention_flags([0, 0, 1, 1, 0, 1, 1]) == [0, 1, 0, 0, 1, 0, 0]
    assert dagger.compute_intervention_flags([1, 1, 0, 0]) == [0, 0, 0, 0]


def test_takeover_before_the_first_policy_step_keeps_the_gripper_open():
    """robosuite: -1 opens, +1 closes. The policy segment's qpos fallback must
    return -1 for an open gripper, or an immediate 'h' would close it on takeover."""
    from mulligan.sim.envs import create_robosuite_env

    assert dagger._gripper_hold_action(np.array([0.04, -0.04])) == -1.0
    assert dagger._gripper_hold_action(np.array([0.001, -0.001])) == 1.0

    env = create_robosuite_env("NutAssemblySquare", use_render_wrapper=False)
    try:
        env.reset()
        operator = ScriptedOperator([[("policy", 0, "h")]], [])
        operator.begin_episode(None)
        _, outcome, _, gripper = dagger.policy_rollout_segment(
            env,
            FakeActor("a", []),
            operator,
            camera_names=[],
            max_fr=0,
            has_renderer=False,
            auto_save_on_success=False,
            initial_gripper_action=None,
        )
    finally:
        env.close()
    assert outcome == "intervention"
    assert gripper == -1.0


def test_object_motion_trigger_is_off_on_square_as_in_the_paper():
    """Pins inherited paper behaviour (docs/reproduce.md, known issues): the Square
    envs have no ``objects`` attribute, so a falling nut does not keep recording."""
    from mulligan.sim.collect.utils import check_objects_moving
    from mulligan.sim.envs import create_robosuite_env, unwrap_env

    env = create_robosuite_env("NutAssemblySquare", use_render_wrapper=False)
    try:
        env.reset()
        base = unwrap_env(env)
        # Give the nut a fixed velocity instead of dropping it: a drop depends on the
        # unseeded start placement (the nut can land on the gripper or table first).
        joint = base.sim.model.joint_name2id(base.nuts[0].joints[0])
        base.sim.data.qvel[base.sim.model.jnt_dofadr[joint] + 2] = -1.5
        base.sim.forward()
        assert np.linalg.norm(base.sim.data.get_body_xvelp(base.nuts[0].root_body)) > 0.5
        assert not hasattr(base, "objects")
        assert check_objects_moving(env) is False
    finally:
        env.close()


def test_a_slow_background_save_is_waited_for_not_failed(capsys):
    import concurrent.futures
    import time

    from mulligan.sim.collect.utils import wait_for_save

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(time.sleep, 0.3)
        dagger._wait_for_save(slow, poll_s=0.05)
        assert slow.done()
        assert "Still waiting for the background episode save" in capsys.readouterr().out

        failing = pool.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            wait_for_save(failing, poll_s=0.05)
        with pytest.raises(RuntimeError, match="Episode save failed"):
            dagger._wait_for_save(failing)


def test_accumulate_segments_keeps_order_and_labels():
    first = {"actions": [1, 2], "observations": ["a", "b"]}
    second = {"actions": [3], "observations": ["c"]}
    data, labels = dagger.accumulate_segment_data(None, first, 0)
    assert labels == [0, 0]
    data, labels = dagger.accumulate_segment_data(data, second, 1)
    assert labels == [1]
    assert data == {"actions": [1, 2, 3], "observations": ["a", "b", "c"]}
    assert first["actions"] == [1, 2]


def _write_routing_manifest(path: Path, labels: list[str]) -> None:
    """A blind manifest that serves both the router and the protocol quota."""
    states = [
        {**state, "policy_source": label, "sources": [label]}
        for state, label in zip(STATES, labels)
    ]
    payload = {
        "task": "square_narrow",
        "keys": ["nut_x", "nut_y", "nut_yaw"],
        "match_tolerance": 1e-3,
        "states": states,
    }
    path.write_text(json.dumps(payload))


def test_router_matches_start_states_to_labels(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_routing_manifest(manifest, ["a", "b", "a"])
    router = dagger.PolicyRouter(manifest, {"a": "actor-a", "b": "actor-b"})
    assert router.keys == ["nut_x", "nut_y", "nut_yaw"]
    for state, expected in zip(STATES, ["a", "b", "a"]):
        start = (state["nut_x"], state["nut_y"], state["nut_yaw"])
        assert router.route(start) == (expected, f"actor-{expected}")
    with pytest.raises(RuntimeError, match="BLINDING ROUTING ERROR"):
        router.route((0.0, 0.0, 0.0))


def test_router_rejects_unrouted_labels_and_single_policy(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_routing_manifest(manifest, ["a", "b", "c"])
    with pytest.raises(ValueError, match="without a policy"):
        dagger.PolicyRouter(manifest, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="at least two"):
        dagger.PolicyRouter(manifest, {"a": 1})


def test_router_handles_square_broad_states(tmp_path):
    manifest = tmp_path / "manifest.json"
    states = [
        {"nut_x": 0.1, "nut_y": 0.2, "nut_yaw": 0.3, "peg_x": 0.4, "peg_y": 0.5, "source": "x"},
        {"nut_x": 0.1, "nut_y": 0.2, "nut_yaw": 0.3, "peg_x": 0.0, "peg_y": 0.5, "source": "y"},
    ]
    manifest.write_text(json.dumps({"_meta": {"task": "square_broad"}, "states": states}))
    router = dagger.PolicyRouter(manifest, {"x": 1, "y": 2})
    assert router.route(((0.1, 0.2, 0.3), (0.0, 0.5))) == ("y", 2)


def test_routed_policy_flags():
    assert dagger._parse_routed_policies(["a=hf://o/r@v/seed-1", "b=/ckpt"]) == {
        "a": "hf://o/r@v/seed-1",
        "b": "/ckpt",
    }
    with pytest.raises(ValueError, match="duplicate"):
        dagger._parse_routed_policies(["a=x", "a=y"])
    with pytest.raises(ValueError, match="LABEL=REF"):
        dagger._parse_routed_policies(["a"])


def make_test_operator(choices="nq"):
    return ScriptedOperator([], list(choices))


def test_operator_flag_loads_a_factory(tmp_path):
    args = _args(
        tmp_path,
        "--operator",
        f"{__name__}:make_test_operator",
        "--operator-kwargs",
        '{"choices": "q"}',
    )
    operator = dagger.make_operator(args)
    assert isinstance(operator, Operator)
    assert operator.choices == ["q"]

    args = _args(tmp_path, "--operator", "collections:OrderedDict")
    with pytest.raises(TypeError, match="not an Operator"):
        dagger.make_operator(args)


class _FakeSpaceMouse:
    pos_sensitivity = 1.0
    rot_sensitivity = 1.5

    def __init__(self, control, gripper_closed):
        self.control = np.asarray(control, dtype=float)
        self.gripper_closed = gripper_closed
        self.started = False

    @property
    def control_gripper(self):
        return 1 if self.gripper_closed else 0

    def start_control(self):
        self.started = True


def test_spacemouse_operator_action_mapping():
    operator = SpaceMouseKeyboardOperator.__new__(SpaceMouseKeyboardOperator)
    operator.device = _FakeSpaceMouse([0.1, -0.2, 0.05, 0.01, 0.02, 0.03], gripper_closed=True)
    human = operator.human_input()
    # dpos = c * 0.005 * 1.0 * 125; drot = [pitch, roll, -yaw] * 0.005 * 1.5 * 50, clipped.
    np.testing.assert_allclose(
        human.action,
        [0.0625, -0.125, 0.03125, 0.0075, 0.00375, -0.01125, -1.0],
    )
    assert human.arm_active and human.strong

    operator.device = _FakeSpaceMouse([0.05, 0, 0, 0, 0, 0], gripper_closed=False)
    human = operator.human_input()
    assert human.action[-1] == 1.0
    assert human.arm_active and not human.strong

    # A human segment continues the current gripper command.
    for command in (-1.0, 1.0, -0.3):
        operator.begin_human_segment(command)
        assert operator.device.started
        assert operator.human_input().action[-1] == (-1.0 if command < 0 else 1.0)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _write_states(path: Path, states) -> None:
    path.write_text(json.dumps({"states": states}))


def test_collector_loop_with_takeover_counterfactual_and_discard(tmp_path):
    states_file = tmp_path / "states.json"
    _write_states(states_file, STATES[:2])
    args = _args(
        tmp_path,
        "--sampler",
        "list",
        "--initial-states-file",
        str(states_file),
        "--no-sampler-shuffle",
    )
    operator = ScriptedOperator(
        scripts=[
            # 1: policy 3 steps, take over, human 4 steps, recoverable failure.
            [("policy", 3, "h"), ("human", 4, "0")],
            # 2: counterfactual of start 1: human first, hand back, terminal failure.
            [("human", 2, "h"), ("policy", 2, "9")],
            # 3: start 2, discarded.
            [("policy", 2, "d")],
            # 4: start 2 again, takeover, success.
            [("policy", 2, "h"), ("human", 1, "1")],
        ],
        choices=["c", "n", "n", "n"],
    )
    log: list[str] = []
    collector = dagger.DaggerCollector(
        args,
        operator=operator,
        actor=FakeActor("single", log),
        env_name="square_narrow",
        robot_name="Panda",
    )
    collector.setup()
    collector.run()

    assert operator.closed
    assert not operator.scripts and not operator.choices
    starts = operator.starts
    assert [s.is_counterfactual for s in starts] == [False, True, False, False]
    assert [s.human_first for s in starts] == [False, True, False, False]
    assert [s.episode_index for s in starts] == [0, 1, 2, 2]
    first = (STATES[0]["nut_x"], STATES[0]["nut_y"], STATES[0]["nut_yaw"])
    second = (STATES[1]["nut_x"], STATES[1]["nut_y"], STATES[1]["nut_yaw"])
    np.testing.assert_allclose(starts[0].start_state, first)
    np.testing.assert_allclose(starts[1].start_state, first)
    np.testing.assert_allclose(starts[2].start_state, second)
    np.testing.assert_allclose(starts[3].start_state, second)
    # Gripper command handed from the policy's last action (-1.0) into each human segment;
    # the counterfactual's first human segment starts from the gripper qpos (open -> -1.0).
    assert operator.segment_gripper == [-1.0, -1.0, -1.0]

    frames = _read_frames(tmp_path / "data" / "dagger-test")
    assert sorted(frames["episode_index"].unique()) == [0, 1, 2]
    assert _column(frames, 0, "source") == [0, 0, 0, 1, 1, 1, 1, 1]
    assert _column(frames, 0, "intervention") == [0, 0, 1, 0, 0, 0, 0, 0]
    assert _column(frames, 0, "is_valid") == [1] * 7 + [0]
    assert set(_column(frames, 0, "success")) == {0}
    assert _column(frames, 0, "done")[-1] == 0
    assert _column(frames, 1, "source") == [1, 1, 0, 0, 0]
    assert _column(frames, 1, "intervention") == [0] * 5
    assert _column(frames, 1, "done")[-1] == 1
    assert set(_column(frames, 1, "success")) == {0}
    assert _column(frames, 2, "source") == [0, 0, 1, 1]
    assert _column(frames, 2, "intervention") == [0, 1, 0, 0]
    assert set(_column(frames, 2, "success")) == {1}
    assert _column(frames, 2, "done")[-1] == 1
    actions = frames.loc[frames["episode_index"] == 0, "action"].tolist()
    np.testing.assert_allclose(actions[0], POLICY_ACTION)
    np.testing.assert_allclose(actions[3], HUMAN_ACTION)
    assert log == ["single"] * (3 + 2 + 2 + 2)


def test_collector_blinded_routing_with_protocol_quota(tmp_path):
    labels = ["ours", "baseline", "ours"]
    manifest = tmp_path / "manifest.json"
    _write_routing_manifest(manifest, labels)
    states_file = tmp_path / "states.json"
    _write_states(states_file, STATES)
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
        "no_cf=1,with_cf=2",
        "--adaptive-protocol-quota-arms",
        "no_cf=baseline,ours;with_cf=ours",
        "--adaptive-protocol-quota-ledger",
        str(ledger),
    )

    def script(start: EpisodeStart):
        if start.human_first:
            return [("human", 2, "h"), ("policy", 1, "h"), ("human", 1, "1")]
        return [("policy", 2, "h"), ("human", 2, "1")]

    log: list[str] = []
    actors = {label: FakeActor(label, log) for label in ("ours", "baseline")}
    router = dagger.PolicyRouter(manifest, actors)
    # Ask for a counterfactual whenever the prompt offers one.
    operator = ScriptedOperator(
        scripts=[script] * 10, choices=[lambda options: "c" if "c" in options else "n"] * 10
    )
    collector = dagger.DaggerCollector(
        args,
        operator=operator,
        actor=actors["ours"],
        env_name="NutAssemblySquare",
        robot_name="Panda",
        router=router,
    )
    collector.setup()
    collector.run()

    quota = collector.protocol_quota
    assert quota.is_complete()
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    frames = _read_frames(tmp_path / "data" / "dagger-test")
    assert len(rows) == frames["episode_index"].nunique() == len(operator.starts)
    assert [row["episode_index"] for row in rows] == list(range(len(rows)))

    # Every episode was rolled out by the policy of its start's manifest label,
    # counterfactuals included.
    by_start = {(s["nut_x"], s["nut_y"], s["nut_yaw"]): label for s, label in zip(STATES, labels)}
    policy_episodes = []
    for start in operator.starts:
        label = by_start[tuple(round(v, 6) for v in start.start_state)]
        policy_episodes.append(label)
    steps_per_episode = [1 if s.human_first else 2 for s in operator.starts]
    expected_log = [
        label for label, steps in zip(policy_episodes, steps_per_episode) for _ in range(steps)
    ]
    assert log == expected_log

    counterfactuals = [s for s in operator.starts if s.is_counterfactual]
    assert len(counterfactuals) == 1
    assert policy_episodes[operator.starts.index(counterfactuals[0])] == "ours"
    assert [row["is_counterfactual"] for row in rows].count(True) == 1
    assert all(row["success"] for row in rows)


def test_collector_resumes_an_existing_dataset(tmp_path):
    states_file = tmp_path / "states.json"
    _write_states(states_file, STATES[:2])
    args = _args(
        tmp_path,
        "--sampler",
        "list",
        "--initial-states-file",
        str(states_file),
        "--no-sampler-shuffle",
    )
    success = [("policy", 1, "h"), ("human", 1, "1")]

    def run_session(operator):
        collector = dagger.DaggerCollector(
            args,
            operator=operator,
            actor=FakeActor("single", []),
            env_name="NutAssemblySquare",
            robot_name="Panda",
        )
        collector.setup()
        collector.run()
        return collector

    first = ScriptedOperator(scripts=[success], choices=["q"])
    run_session(first)
    second = ScriptedOperator(scripts=[success], choices=["n"])
    collector = run_session(second)

    assert collector.base_episode_count == 1
    assert second.starts[0].episode_index == 1
    np.testing.assert_allclose(
        second.starts[0].start_state,
        (STATES[1]["nut_x"], STATES[1]["nut_y"], STATES[1]["nut_yaw"]),
    )
    frames = _read_frames(tmp_path / "data" / "dagger-test")
    assert sorted(frames["episode_index"].unique()) == [0, 1]


def test_idql_actor_normalizes_splits_and_denormalizes():
    import torch

    class Normalizer:
        def normalize_state(self, state):
            return state * 2.0

        def denormalize_action(self, action):
            return action + 1.0

    class Policy:
        def __init__(self):
            self.batches = []

        def select_action(self, batch):
            self.batches.append(batch)
            return torch.zeros(1, 7)

    policy = Policy()
    actor = dagger.IDQLActor(policy, Normalizer(), Normalizer(), "cpu")
    action = actor.act(np.arange(9, dtype=float), np.arange(9, 23, dtype=float))
    np.testing.assert_allclose(action, np.ones(7))
    batch = policy.batches[0]
    np.testing.assert_allclose(batch["observation.state"][0].numpy(), np.arange(9) * 2.0)
    np.testing.assert_allclose(
        batch["observation.environment_state"][0].numpy(), np.arange(9, 23) * 2.0
    )
