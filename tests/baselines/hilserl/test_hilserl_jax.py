"""HiL-SERL tests that need JAX and/or robosuite.

The agent's reference numerics are pinned by tests/baselines/rlpd/test_agent_reference.py. The
demo test (network) downloads the released teleop demos.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if sys.platform == "linux":  # EGL is Linux-only; macOS uses mujoco's default backend
    os.environ.setdefault("MUJOCO_GL", "egl")

jax = pytest.importorskip("jax")

from mulligan.baselines.hilserl.config import HilSerlConfig  # noqa: E402
from mulligan.baselines.hilserl.obs import ACTION_DIM  # noqa: E402
from mulligan.baselines.hilserl.session import Session  # noqa: E402
from mulligan.baselines.hilserl.tasks import TASKS, get_task  # noqa: E402

OBS_DIM = TASKS["square_narrow"].obs_dim
REPO = Path(__file__).resolve().parents[3]


def _tiny_cfg(**kw) -> HilSerlConfig:
    base = dict(
        hidden_dims=(16, 16),
        num_qs=3,
        num_min_qs=2,
        batch_size=8,
        utd_ratio=2,
        start_training=4,
        max_steps=64,
        eval_interval=32,
        eval_episodes=2,
        eval_workers=2,
        horizon=6,
        log_interval=1,
        wandb_mode="disabled",
        resume_interval_s=1e9,
        actor_max_lag=8,
    )
    base.update(kw)
    return HilSerlConfig(**base)


def _fake_demos(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = dict(
        observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        actions=rng.uniform(-0.9, 0.9, (n, ACTION_DIM)).astype(np.float32),
        rewards=np.zeros(n, np.float32),
        masks=np.ones(n, np.float32),
        dones=np.zeros(n, bool),
        next_observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
    )
    d["rewards"][-1] = 1.0
    d["masks"][-1] = 0.0
    d["dones"][-1] = True
    return d


# ----------------------------------------------------------------- agent


def test_agent_update_is_deterministic_and_shapes():
    from mulligan.baselines.rlpd.agent import SACLearner

    cfg = _tiny_cfg()
    demos = _fake_demos()
    a1 = SACLearner.create(1, demos["observations"][0], demos["actions"][0], **cfg.agent_kwargs())
    a2 = SACLearner.create(1, demos["observations"][0], demos["actions"][0], **cfg.agent_kwargs())
    batch = {k: v[: cfg.batch_size * cfg.utd_ratio] for k, v in demos.items()}
    a1, i1 = a1.update(batch, cfg.utd_ratio)
    a2, i2 = a2.update(batch, cfg.utd_ratio)
    assert set(i1) == {
        "actor_loss",
        "entropy",
        "critic_loss",
        "q",
        "temperature",
        "temperature_loss",
    }
    assert float(i1["critic_loss"]) == float(i2["critic_loss"])
    assert a1.target_entropy == -ACTION_DIM / 2
    acts, _ = a1.sample_actions(demos["observations"][:3])
    assert acts.shape == (3, ACTION_DIM) and np.all(np.abs(acts) <= 1)


# ----------------------------------------------------------------- demos


@pytest.mark.network
@pytest.mark.parametrize("task_key", sorted(TASKS))
def test_demo_transitions(task_key):
    """The task's teleop demos as transitions: the pinned content sha and the success contract."""
    from mulligan.baselines.hilserl.demos import content_sha256, load_demo_transitions

    task = get_task(task_key)
    mine = load_demo_transitions(task)
    assert content_sha256(mine) == task.demos_sha256
    assert mine["observations"].shape == (task.num_transitions, task.obs_dim)
    assert int(mine["rewards"].sum()) == task.num_episodes
    assert (
        int((mine["masks"] == 0).sum()) == task.num_episodes
        and int(mine["dones"].sum()) == task.num_episodes
    )
    assert np.abs(mine["actions"]).max() <= 1 - 1e-5


# ------------------------------------------------------------------- env


@pytest.fixture(scope="module", params=sorted(TASKS))
def env(request):
    from mulligan.baselines.hilserl.env import HilSerlEnv

    e = HilSerlEnv(request.param, horizon=6)
    yield e
    e.close()


def test_env_contract_and_seeding(env):
    o = env.reset(seed=10000)
    assert o.shape == (env.task.obs_dim,) and o.dtype == np.float32
    o2 = env.reset(seed=10000)
    assert np.array_equal(o, o2)
    assert not np.array_equal(o, env.reset(seed=10001))
    env.reset(seed=5)
    res = None
    for _ in range(6):
        res = env.step(np.zeros(ACTION_DIM))
    assert res.done and res.truncated and res.mask == 0.0 and res.reward == 0.0 and res.t == 6
    # idle (uncounted) steps do not advance the clock
    env.reset(seed=5)
    env.step(np.zeros(ACTION_DIM), count=False)
    assert env.t == 0


def test_env_truncation_bootstrap_flag():
    from mulligan.baselines.hilserl.env import HilSerlEnv

    e = HilSerlEnv("square_narrow", horizon=3, truncation_bootstrap=True)
    e.reset(seed=1)
    for _ in range(3):
        res = e.step(np.zeros(ACTION_DIM))
    assert res.done and res.mask == 1.0
    e.close()


def test_env_snapshot_restore(env):
    env.reset(seed=3)
    env.step(np.array([0.3, 0, 0, 0, 0, 0, -1]))
    snap = env.snapshot()
    state = env.sim_state_flat()
    for _ in range(3):
        env.step(np.array([0.5, 0.5, 0, 0, 0, 0, 1]))
    assert env.t == 4
    env.restore(snap)
    assert env.t == 1 and np.allclose(env.sim_state_flat(), state)
    assert env.nut_speed() >= 0.0


# ------------------------------------------------------------------ eval


def test_evaluate_in_process_is_reproducible():
    from mulligan.baselines.hilserl.evaluate import _run_episode, _worker_init, summarize
    from mulligan.baselines.hilserl.policy import ActorPolicy

    def evaluate_in_process(params, cfg, num_episodes):
        """The evaluation protocol without worker processes."""
        _worker_init(cfg.to_dict(), params)
        episodes = [_run_episode(cfg.eval_seed_base + i) for i in range(num_episodes)]
        return summarize(episodes), episodes

    cfg = _tiny_cfg(horizon=5)
    params = ActorPolicy(cfg, rng_seed=0).params
    m1, eps1 = evaluate_in_process(params, cfg, 2)
    m2, eps2 = evaluate_in_process(params, cfg, 2)
    assert [e["seed"] for e in eps1] == [10000, 10001]
    assert (
        eps1 == eps2
        and m1["n"] == 2
        and set(m1) == {"return", "length", "success_rate", "success_length_mean", "n"}
    )


# --------------------------------------------------------------- learner


def _episode_arrays(n, intervened_from=None):
    rng = np.random.default_rng(n)
    a = dict(
        observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        actions=rng.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32),
        policy_actions=np.zeros((n, ACTION_DIM), np.float32),
        rewards=np.zeros(n, np.float32),
        masks=np.ones(n, np.float32),
        dones=np.zeros(n, bool),
        next_observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        intervened=np.zeros(n, bool),
    )
    if intervened_from is not None:
        a["intervened"][intervened_from:] = True
    return a


def _rec(ep, n, success, intv):
    return dict(
        episode_id=ep,
        length=n,
        success=success,
        fail_terminated=False,
        intervention_count=int(intv > 0),
        intervention_steps=intv,
        idle_steps_dropped=0,
        redo_count=0,
        param_version=0,
        env_steps_before=0,
        wall_start=0.0,
        wall_end=0.0,
        extra={},
    )


@pytest.mark.parametrize("gate", ["success", "none"])
def test_learner_demo_gate_routing_and_dedup(tmp_path, gate):
    from mulligan.baselines.hilserl.learner import Learner

    cfg = _tiny_cfg(demo_gate=gate)
    demos = _fake_demos()
    L = Learner(cfg, Session(tmp_path / "s"), None, demos=demos)
    n_demo = len(demos["observations"])
    assert L.ingest_episode(
        0, _episode_arrays(6, intervened_from=4), _rec(0, 6, success=False, intv=2)
    )
    assert L.ingest_episode(
        1, _episode_arrays(5, intervened_from=2), _rec(1, 5, success=True, intv=3)
    )
    assert not L.ingest_episode(1, _episode_arrays(5), _rec(1, 5, True, 0))  # duplicate
    assert len(L.online) == 11 and L.counters["env_steps"] == 11 and L.counters["episodes"] == 2
    expected_demo = n_demo + (3 if gate == "success" else 5)
    assert len(L.demo_buf) == expected_demo and L.counters["demo_added"] == expected_demo - n_demo
    m = L.hil_metrics()
    assert m["buffer/demo_demo_frac"] == pytest.approx(n_demo / expected_demo)
    assert m["hil/intervention_rate_20ep"] == pytest.approx(np.mean([2 / 6, 3 / 5]))


def test_learner_governor_update_and_resume_roundtrip(tmp_path):
    from mulligan.baselines.hilserl.learner import Learner

    cfg = _tiny_cfg(start_training=4)
    demos = _fake_demos()
    L = Learner(cfg, Session(tmp_path / "s"), None, demos=demos)
    L.ingest_episode(0, _episode_arrays(10), _rec(0, 10, False, 0))
    assert L.governor.owed(L.counters["env_steps"], L.counters["calls_done"]) == 6
    for _ in range(6):
        L.update_once()
    assert L.governor.owed(10, L.counters["calls_done"]) == 0 and L.env_step_of_last_call() == 10
    assert L.milestone_due() == 0
    L.save_checkpoint(0)
    L.next_milestone = cfg.eval_interval
    L.save_resume("test")
    M = Learner(cfg, Session(tmp_path / "s"), None, demos=demos)
    assert M.load_resume()
    assert (
        M.counters == L.counters and M.ingested_ids == {0} and M.next_milestone == cfg.eval_interval
    )
    assert np.array_equal(M.online.sample(4)["observations"], L.online.sample(4)["observations"])
    p1 = jax.tree_util.tree_leaves(L.agent.actor.params)
    p2 = jax.tree_util.tree_leaves(M.agent.actor.params)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(p1, p2))
    with pytest.raises(RuntimeError):
        Learner(_tiny_cfg(discount=0.5), Session(tmp_path / "s"), None, demos=demos).load_resume()


def test_learner_accepts_the_episode_that_crosses_max_steps(tmp_path):
    """The actor stops only after the episode that crosses max_steps, so the buffers
    must hold up to horizon-1 extra steps; one more episode past that is a real overflow."""
    from mulligan.baselines.hilserl.learner import Learner

    cfg = _tiny_cfg(max_steps=64, horizon=6)
    L = Learner(cfg, Session(tmp_path / "s"), None, demos=_fake_demos())
    assert L.online.capacity == 70
    for i in range(10):
        assert L.ingest_episode(i, _episode_arrays(6), _rec(i, 6, False, 0))
    assert L.counters["env_steps"] == 60
    assert L.ingest_episode(10, _episode_arrays(6), _rec(10, 6, True, 0))  # crosses 64
    assert L.counters["env_steps"] == 66 >= cfg.max_steps
    with pytest.raises(RuntimeError, match="buffer full"):
        L.ingest_episode(11, _episode_arrays(6), _rec(11, 6, False, 0))


# ----------------------------------------------------------------- actor


def test_actor_episode_with_scripted_takeover_redo_and_fail(tmp_path):
    from mulligan.baselines.hilserl.actor import Actor
    from mulligan.baselines.hilserl.env import HilSerlEnv
    from mulligan.baselines.hilserl.policy import ActorPolicy
    from mulligan.baselines.hilserl.takeover import ScriptedKeys, SpacemouseTakeover
    from tests.baselines.hilserl.scripted_device import ScriptedDevice

    cfg = _tiny_cfg(horizon=8)
    env = HilSerlEnv("square_narrow", horizon=8)
    z = np.zeros(6)
    push = np.array([0.6, 0, 0, 0, 0, 0])
    # decide() is called once per non-idle-or-idle loop iteration; steps 2,3 human, then release
    # gripper None = no button press: the toggle stays where the takeover synced it
    readings = [(z, None), (z, None), (push, None), (push, None)] + [(z, None)] * 20
    takeover = SpacemouseTakeover(ScriptedDevice(readings), latch_s=0.0, idle_filter=True)
    # key presses are drained once per loop iteration: 'r' at iteration 4 (after the bout), 'x' at iteration 6
    keys = ScriptedKeys({4: ["r"], 6: ["x"]})
    actor = Actor(
        cfg,
        Session(tmp_path / "s"),
        env,
        ActorPolicy(cfg, rng_seed=0),
        client=None,
        takeover=takeover,
        keys=keys,
        paced=False,
    )
    record, arrays, init_state = actor.run_episode(0, 0)
    assert record.intervention_count == 1 and record.redo_count == 1
    # the redo dropped the 2 human steps, so no intervened rows remain
    assert record.intervention_steps == 0 and not arrays["intervened"].any()
    assert record.fail_terminated and not record.success
    assert arrays["dones"][-1] and arrays["masks"][-1] == 0.0 and arrays["rewards"][-1] == 0.0
    assert record.length == len(arrays["observations"]) and record.length < 8
    assert init_state.shape == env.sim_state_flat().shape
    env.close()


def test_actor_idle_steps_are_not_recorded(tmp_path):
    from mulligan.baselines.hilserl.actor import Actor
    from mulligan.baselines.hilserl.env import HilSerlEnv
    from mulligan.baselines.hilserl.policy import ActorPolicy
    from mulligan.baselines.hilserl.takeover import ScriptedKeys, SpacemouseTakeover
    from tests.baselines.hilserl.scripted_device import ScriptedDevice

    cfg = _tiny_cfg(horizon=5)
    env = HilSerlEnv("square_narrow", horizon=5)
    z = np.zeros(6)
    readings = [(z, None), (z, None)] + [(z, None)] * 30  # None = no button press
    # object_vel_thresh high: the nut settling right after reset must not count as "object moving" here
    takeover = SpacemouseTakeover(
        ScriptedDevice(readings), latch_s=0.0, idle_filter=True, object_vel_thresh=10.0
    )
    takeover.set_hold(
        True
    )  # held takeover with no input: every step idle -> nothing recorded until we release
    actor = Actor(
        cfg,
        Session(tmp_path / "s"),
        env,
        ActorPolicy(cfg, rng_seed=0),
        client=None,
        takeover=takeover,
        keys=ScriptedKeys({3: ["h"]}),
        paced=False,
    )
    record, arrays, _ = actor.run_episode(0, 0)
    # iterations 0..2 are idle bout steps (dropped), 'h' at iteration 3 releases the hold, then 5 policy steps
    assert record.idle_steps_dropped == 3 and record.length == 5 and not arrays["intervened"].any()
    assert record.intervention_count == 1 and record.intervention_steps == 0
    env.close()


def test_actor_resync_reads_the_ledger_once(tmp_path, monkeypatch):
    """sync_pending reads (and stats) the ledger once, not once per episode it pushes."""
    from dataclasses import asdict

    from mulligan.baselines.hilserl.actor import Actor
    from mulligan.baselines.hilserl.session import EpisodeRecord

    session = Session(tmp_path / "s")
    for ep in range(5):
        rec = EpisodeRecord(**{**_rec(ep, 6, False, 0), "env_steps_before": 6 * ep})
        session.write_episode(rec, _episode_arrays(6), np.zeros(3))

    class Client:
        def __init__(self):
            self.pushed = []

        def request(self, type_, payload):
            if type_ == "status":
                return {"ingested_ids": [1], "owed_calls": 0}
            self.pushed.append(payload["record"])
            return {"success": True}

    reads = []
    read_ledger = session.read_ledger
    monkeypatch.setattr(session, "read_ledger", lambda: reads.append(1) or read_ledger())
    client = Client()
    Actor(_tiny_cfg(), session, None, None, client=client).sync_pending()
    assert len(reads) == 1
    assert client.pushed == [asdict(r) for r in read_ledger() if r.episode_id != 1]


class _InProcessClient:
    """ActorClient stand-in that calls a Learner's request handler directly."""

    def __init__(self, learner):
        self.learner = learner
        self.pushed = []

    def request(self, type_, payload):
        reply = self.learner.request_callback(type_, payload)
        if type_ == "push-episode":
            self.pushed.append(payload["episode_id"])
            self.learner.drain_incoming()
        return reply

    def stop(self):
        pass


def test_actor_with_complete_ledger_resends_what_a_resumed_learner_lacks(tmp_path, monkeypatch):
    """The actor logged max_steps, but the learner was preempted after ingesting only
    episode 0. A restarted actor must re-send episode 1 instead of exiting early."""
    from mulligan.baselines.hilserl import __main__ as cli
    from mulligan.baselines.hilserl import transport
    from mulligan.baselines.hilserl.learner import Learner
    from mulligan.baselines.hilserl.session import EpisodeRecord

    cfg = _tiny_cfg(max_steps=12, horizon=6)
    session = Session(tmp_path / "s")
    for ep in (0, 1):
        rec = EpisodeRecord(**{**_rec(ep, 6, False, 0), "env_steps_before": 6 * ep})
        session.write_episode(rec, _episode_arrays(6), np.zeros(3))
    learner = Learner(cfg, session, None, demos=_fake_demos())
    learner.ingest_episode(0, session.load_episode(0), _rec(0, 6, False, 0))

    client = _InProcessClient(learner)
    monkeypatch.setattr(transport, "ActorClient", lambda *a, **kw: client)
    assert cli.flush_complete_session(cfg, session, "127.0.0.1") == 0
    assert client.pushed == [1]
    assert learner.ingested_ids == {0, 1} and learner.counters["env_steps"] == 12

    def unreachable(*a, **kw):
        raise Exception("Failed to connect to server")

    monkeypatch.setattr(transport, "ActorClient", unreachable)
    assert cli.flush_complete_session(cfg, session, "127.0.0.1") == 1


# ---------------------------------------------------------- split E2E


def _cli(*args):
    return [sys.executable, "-m", "mulligan.baselines.hilserl", *args]


def _wait_for(path: Path, timeout=120):
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(path)
        time.sleep(0.5)


def _wait_until(condition, what: str, timeout=180):
    """Poll ``condition()`` instead of sleeping a fixed time (the learner is slow under load)."""
    t0 = time.time()
    while not condition():
        if time.time() - t0 > timeout:
            raise TimeoutError(what)
        time.sleep(0.5)


def _saved_learner_state(session_dir: Path):
    import pickle

    path = Session(session_dir).state_path
    return pickle.loads(path.read_bytes()) if path.exists() else None


@pytest.mark.slow
@pytest.mark.parametrize("task_key", sorted(TASKS))
def test_split_kill_and_resume_e2e(tmp_path, task_key):
    """learner + scripted actor over agentlace; SIGTERM the learner mid-run,
    restart it, kill the actor, restart it: no duplicates, counters agree,
    every logged episode ends up in the learner exactly once."""
    import pickle

    session = tmp_path / "e2e"
    port = 5800 + os.getpid() % 100
    common = [
        "--task",
        task_key,
        "--session",
        str(session),
        "--port",
        str(port),
        "--start_training",
        "6",
        "--utd_ratio",
        "2",
        "--batch_size",
        "8",
        "--max_steps",
        "60",
        "--eval_interval",
        "30",
        "--eval_episodes",
        "1",
        "--eval_workers",
        "1",
        "--wandb_mode",
        "disabled",
        "--horizon",
        "10",
        "--resume_interval_s",
        "1",
        "--actor_max_lag",
        "20",
        "--hidden_dims",
        "16,16",
        "--num_qs",
        "3",
        "--num_min_qs",
        "2",
    ]
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"
    logs = tmp_path / "logs"
    logs.mkdir()

    def start_learner(tag):
        return subprocess.Popen(
            _cli("--learner", *common),
            env=env,
            cwd=REPO,
            stdout=open(logs / f"learner_{tag}.log", "w"),
            stderr=subprocess.STDOUT,
        )

    def run_actor(tag, max_episodes):
        return subprocess.run(
            _cli(
                "--actor",
                *common,
                "--ip",
                "127.0.0.1",
                "--no-spacemouse",
                "--no-render",
                "--unpaced",
                "--max_episodes",
                str(max_episodes),
            ),
            env=env,
            cwd=REPO,
            stdout=open(logs / f"actor_{tag}.log", "w"),
            stderr=subprocess.STDOUT,
            timeout=300,
        )

    L1 = start_learner("1")
    try:
        _wait_for(Session(session).endpoint_path)
        assert run_actor("1", 2).returncode == 0
        Session(session)  # ledger has 2 episodes
        assert Session(session).next_episode_id() == 2
        # let the learner ingest both episodes and save its periodic resume state
        _wait_until(
            lambda: (
                (st := _saved_learner_state(session)) is not None
                and st["counters"]["env_steps"] == 20
            ),
            "learner resume state with both episodes",
        )
        L1.send_signal(signal.SIGTERM)
        assert L1.wait(timeout=60) == 75  # PREEMPTED_EXIT_CODE after saving state
    finally:
        if L1.poll() is None:
            L1.kill()
    st = pickle.loads(Session(session).state_path.read_bytes())
    assert st["counters"]["env_steps"] == Session(session).env_steps_logged() == 20
    assert st["ingested_ids"] == [0, 1]
    # restart learner; actor collects 2 more, then everything is consistent
    endpoint = Session(session).endpoint_path
    endpoint_before = endpoint.stat().st_mtime_ns
    L2 = start_learner("2")
    try:
        # the restarted learner rewrites its endpoint once it has resumed and serves
        _wait_until(
            lambda: endpoint.stat().st_mtime_ns != endpoint_before,
            "restarted learner endpoint",
        )
        assert run_actor("2", 2).returncode == 0
        assert Session(session).next_episode_id() == 4
        t0 = time.time()
        while time.time() - t0 < 120:
            st = pickle.loads(Session(session).state_path.read_bytes())
            if st["counters"]["episodes"] == 4:
                break
            time.sleep(1)
        assert (
            st["counters"]["episodes"] == 4
            and st["ingested_ids"] == [0, 1, 2, 3]
            and st["counters"]["env_steps"] == 40
        )
        # actor restart with nothing new to push: hello + sync must not duplicate anything
        assert run_actor("3", 1).returncode == 0
        assert Session(session).next_episode_id() == 5
        t0 = time.time()
        while time.time() - t0 < 120:
            st = pickle.loads(Session(session).state_path.read_bytes())
            if st["counters"]["episodes"] == 5:
                break
            time.sleep(1)
        assert st["counters"]["episodes"] == 5 and len(st["online"]["data"]["observations"]) == 50
        # the buffer content equals the concatenated episode logs, in order
        s = Session(session)
        logged = np.concatenate([s.load_episode(i)["observations"] for i in range(5)])
        assert np.array_equal(st["online"]["data"]["observations"], logged)
    finally:
        L2.send_signal(signal.SIGTERM)
        try:
            L2.wait(timeout=60)
        except subprocess.TimeoutExpired:
            L2.kill()
    actor_log = (logs / "actor_3.log").read_text()
    assert "re-pushed" not in actor_log
