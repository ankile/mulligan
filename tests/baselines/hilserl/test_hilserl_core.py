"""HiL-SERL unit tests that need only numpy (replay, governor, obs, session,
takeover, config). JAX / robosuite tests live in test_hilserl_jax.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mulligan.baselines.hilserl.config import HilSerlConfig
from mulligan.baselines.hilserl.governor import Governor
from mulligan.baselines.hilserl.obs import assemble_obs, object_state_to_robomimic_order
from mulligan.baselines.hilserl.replay import (
    TRANSITION_KEYS,
    Dataset,
    ReplayBuffer,
    combine,
    make_transition,
)
from mulligan.baselines.hilserl import session as session_mod
from mulligan.baselines.hilserl.session import EpisodeRecord, Session, hilserl_sha
from mulligan.baselines.hilserl.tasks import TASKS, get_task
from mulligan.baselines.hilserl.takeover import (
    DEFAULT_DEADZONE,
    GRIPPER_SETTLE_STEPS,
    SpacemouseTakeover,
    teleop_action,
)
from tests.baselines.hilserl.scripted_device import ScriptedDevice

OBS_DIM = TASKS["square_narrow"].obs_dim

# ----------------------------------------------------------------- replay


def _tr(i: int, obs_dim=OBS_DIM, act_dim=7):
    return make_transition(
        np.full(obs_dim, i, np.float32),
        np.full(act_dim, i / 10, np.float32),
        float(i % 2),
        1.0,
        False,
        np.full(obs_dim, i + 1, np.float32),
    )


def test_replay_insert_sample_and_ring():
    buf = ReplayBuffer(np.zeros(OBS_DIM, np.float32), np.zeros(7, np.float32), capacity=5)
    buf.seed(0)
    for i in range(7):
        buf.insert(_tr(i))
    assert len(buf) == 5
    # ring: rows 5,6 overwrote 0,1
    assert set(buf.dataset_dict["observations"][:, 0].tolist()) == {2, 3, 4, 5, 6}
    b = buf.sample(8)
    assert set(b) == set(TRANSITION_KEYS)
    assert b["observations"].shape == (8, OBS_DIM) and b["dones"].dtype == bool


def test_replay_sampling_matches_gym_seeding_generator():
    ds = Dataset({"x": np.arange(100)}, seed=1)
    idx = ds.sample(5)["x"]
    assert idx.tolist() == np.random.default_rng(1).integers(100, size=5).tolist()


def test_replay_insert_dataset_then_insert_continues_after_preload():
    buf = ReplayBuffer(np.zeros(OBS_DIM, np.float32), np.zeros(7, np.float32), capacity=10)
    data = {k: np.stack([_tr(i)[k] for i in range(3)]) for k in TRANSITION_KEYS}
    buf.insert_dataset(data)
    assert len(buf) == 3
    buf.insert(_tr(99))
    assert len(buf) == 4 and buf.dataset_dict["observations"][3, 0] == 99
    with pytest.raises(RuntimeError):
        buf.insert_dataset(data)


def test_replay_state_roundtrip():
    a = ReplayBuffer(np.zeros(OBS_DIM, np.float32), np.zeros(7, np.float32), capacity=10)
    a.seed(3)
    for i in range(4):
        a.insert(_tr(i))
    a.sample(2)
    st = a.state()
    b = ReplayBuffer(np.zeros(OBS_DIM, np.float32), np.zeros(7, np.float32), capacity=10)
    b.seed(0)
    b.load_state(st)
    assert len(b) == 4 and b._insert_index == 4
    assert np.array_equal(a.sample(3)["observations"], b.sample(3)["observations"])


def test_combine_interleaves_and_utd_slices_are_half_half():
    off = {"x": np.zeros(40)}
    on = {"x": np.ones(40)}
    c = combine(off, on)["x"]
    assert c.shape == (80,) and c[0::2].sum() == 0 and c[1::2].sum() == 40
    # SACLearner.update slices x[i*bs:(i+1)*bs] with bs = 80 // utd; each slice must be 50/50
    utd = 20
    bs = 80 // utd
    for i in range(utd):
        assert combine(off, on)["x"][i * bs : (i + 1) * bs].sum() == bs / 2


# --------------------------------------------------------------- governor


def test_governor_target_and_owed():
    g = Governor(start_training=5000)
    assert (
        g.target_calls(0) == 0
        and g.target_calls(5000) == 0
        and g.target_calls(5001) == 1
        and g.target_calls(300_000) == 295_000
    )
    assert g.owed(6000, 400) == 600 and g.owed(6000, 1000) == 0 and g.owed(6000, 2000) == 0
    with pytest.raises(ValueError):
        g.target_calls(-1)


# -------------------------------------------------------------------- obs


def test_object_state_reorder_and_assemble():
    es = np.arange(14, dtype=np.float64)
    reordered = object_state_to_robomimic_order(es)
    assert reordered.tolist() == list(range(7, 14)) + list(range(0, 7))
    obs = assemble_obs(np.arange(3), np.arange(4), np.arange(2), es)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert obs[9:16].tolist() == list(range(7, 14))
    batch = assemble_obs(np.zeros((5, 3)), np.zeros((5, 4)), np.zeros((5, 2)), np.tile(es, (5, 1)))
    assert batch.shape == (5, OBS_DIM)
    with pytest.raises(ValueError):
        object_state_to_robomimic_order(np.zeros(16))


def test_object_state_reorder_square_broad_keeps_peg_last():
    # Square_D1: nut_to_eef(7), nut pose(7), peg_pos(3) -> robomimic: nut pose, nut_to_eef, peg_pos
    es = np.arange(17, dtype=np.float64)
    assert object_state_to_robomimic_order(es).tolist() == (
        list(range(7, 14)) + list(range(0, 7)) + [14, 15, 16]
    )
    obs = assemble_obs(np.arange(3), np.arange(4), np.arange(2), es)
    assert obs.shape == (TASKS["square_broad"].obs_dim,) == (26,)
    assert obs[23:].tolist() == [14, 15, 16]


def test_task_specs():
    narrow, broad = get_task("square_narrow"), get_task("square_broad")
    assert (narrow.env_name, narrow.obs_dim, narrow.num_transitions) == (
        "NutAssemblySquare",
        23,
        16233,
    )
    assert (broad.env_name, broad.obs_dim, broad.num_transitions) == ("Square_D1", 26, 35486)
    with pytest.raises(ValueError):
        get_task("square_unknown")


# ----------------------------------------------------------------- config


def test_config_defaults_are_the_rlpd_agent():
    from mulligan.baselines.rlpd.configs import AGENT_DEFAULTS

    c = HilSerlConfig()
    assert {k: v for k, v in c.agent_kwargs().items()} == {
        k: v for k, v in AGENT_DEFAULTS.items() if k in c.agent_kwargs()
    }
    assert (
        c.hidden_dims == (256, 256, 256)
        and c.num_qs == 10
        and c.num_min_qs == 2
        and c.critic_layer_norm
    )
    assert (
        c.discount == 0.99 and c.tau == 0.005 and not c.backup_entropy and c.init_temperature == 1.0
    )
    assert (
        c.utd_ratio == 20
        and c.batch_size == 256
        and c.start_training == 5000
        and c.max_steps == 300_000
    )
    assert c.offline_batch == 2560 and c.online_batch == 2560
    assert c.eval_seed_base == 10_000 and c.eval_episodes == 50 and c.horizon == 400
    assert c.demo_gate == "success" and not c.truncation_bootstrap
    assert c.pinned_hash() == HilSerlConfig(wandb_mode="disabled", port=1).pinned_hash()
    assert c.pinned_hash() != HilSerlConfig(discount=0.97).pinned_hash()
    assert c.task == "square_narrow" and "task" in c.pinned()
    assert c.pinned_hash() != HilSerlConfig(task="square_broad").pinned_hash()
    with pytest.raises(ValueError):
        HilSerlConfig(demo_gate="maybe")
    with pytest.raises(ValueError):
        HilSerlConfig(task="square")


# ---------------------------------------------------------------- session


def _record(ep, n, **kw):
    base = dict(
        episode_id=ep,
        length=n,
        success=False,
        fail_terminated=False,
        intervention_count=0,
        intervention_steps=0,
        idle_steps_dropped=0,
        redo_count=0,
        param_version=0,
        env_steps_before=0,
        wall_start=0.0,
        wall_end=1.0,
    )
    base.update(kw)
    return EpisodeRecord(**base)


def _arrays(n):
    return dict(
        observations=np.zeros((n, OBS_DIM), np.float32),
        actions=np.zeros((n, 7), np.float32),
        policy_actions=np.zeros((n, 7), np.float32),
        rewards=np.zeros(n, np.float32),
        masks=np.ones(n, np.float32),
        dones=np.zeros(n, bool),
        next_observations=np.zeros((n, OBS_DIM), np.float32),
        intervened=np.zeros(n, bool),
    )


def test_session_episode_log_and_resume_counters(tmp_path):
    s = Session(tmp_path / "sess")
    assert s.next_episode_id() == 0 and s.env_steps_logged() == 0
    s.write_episode(_record(0, 5), _arrays(5), np.zeros(3))
    s.write_episode(_record(1, 7, env_steps_before=5, success=True), _arrays(7), np.zeros(3))
    assert s.next_episode_id() == 2 and s.env_steps_logged() == 12
    recs = s.read_ledger()
    assert [r.episode_id for r in recs] == [0, 1] and recs[1].success
    ep = s.load_episode(1)
    assert ep["observations"].shape == (7, OBS_DIM) and "initial_sim_state" in ep
    with pytest.raises(FileExistsError):
        s.write_episode(_record(1, 2), _arrays(2), np.zeros(3))
    with pytest.raises(ValueError):
        s.write_episode(_record(2, 3), _arrays(4), np.zeros(3))


def test_eval_watcher_stops_at_the_final_checkpoint_without_reading_learner_state(
    tmp_path, monkeypatch
):
    from mulligan.baselines.hilserl import evaluate

    s = Session(tmp_path / "sess")
    cfg = HilSerlConfig(max_steps=25, eval_interval=10)
    for step in (0, 10, 20):  # milestones <= max_steps; 20 is the last one
        (s.checkpoints_dir / f"step_{step:07d}").mkdir(parents=True)
    s.state_path.parent.mkdir(parents=True, exist_ok=True)
    s.state_path.write_bytes(b"large learner resume state; the watcher must not load it")
    evaluated = []
    monkeypatch.setattr(evaluate, "load_checkpoint_params", lambda ckpt: ({}, 0))

    def fake_eval(params, cfg, workers=None):
        evaluated.append(len(evaluated))
        return {"success_rate": 0.0}, []

    monkeypatch.setattr(evaluate, "evaluate_native", fake_eval)
    evaluate.watch(s, cfg, poll_s=0.01)
    assert [r["env_step"] for r in session_mod.read_jsonl(s.eval_ledger)] == [0, 10, 20]
    assert evaluate.final_checkpoint_dir(s, cfg).name == "step_0000020"


def test_session_ledger_corruption_is_loud(tmp_path):
    s = Session(tmp_path / "sess")
    s.write_episode(_record(0, 2), _arrays(2), np.zeros(3))
    s.actor_ledger.write_text(s.actor_ledger.read_text() + "{not json\n")
    with pytest.raises(ValueError):
        s.read_ledger()


def test_hilserl_sha_covers_the_contract_and_ignores_read_only_surfaces(tmp_path, monkeypatch):
    """The handshake gate must follow the experiment code, not the commit sha.

    A commit that touches only the offline tools must leave the hash alone, or
    editing one locks every running actor out of its learner; a change to a module
    the actor/learner loop runs (this package or the RLPD agent) must move it.
    """
    pkg = tmp_path / "hilserl"
    agent = tmp_path / "rlpd" / "agent"
    agent.mkdir(parents=True)
    (pkg / "tools").mkdir(parents=True)
    (pkg / "actor.py").write_text("ACTOR = 1\n")
    (agent / "sac_learner.py").write_text("SAC = 1\n")
    (pkg / "tools" / "fork_session.py").write_text("FORK = 1\n")
    monkeypatch.setattr(session_mod, "_PACKAGE_ROOT", pkg)
    monkeypatch.setattr(session_mod, "_AGENT_ROOT", agent)

    base = hilserl_sha()
    (pkg / "tools" / "fork_session.py").write_text("FORK = 2\n")
    assert hilserl_sha() == base

    (agent / "sac_learner.py").write_text("SAC = 2\n")
    assert hilserl_sha() != base
    (pkg / "actor.py").write_text("ACTOR = 2\n")
    assert hilserl_sha() != base


def test_session_meta_pins_config(tmp_path):
    s = Session(tmp_path / "sess")
    s.ensure_meta("learner", {"seed": 1, "discount": 0.99})
    s.ensure_meta("actor", {"seed": 1, "discount": 0.99, "port": 5})
    with pytest.raises(RuntimeError):
        s.ensure_meta("actor", {"seed": 2})
    meta = json.loads(s.meta_path.read_text())
    assert meta["config"]["seed"] == 1 and len(meta["repo_sha"]) == 40


def test_repo_sha_outside_a_checkout_is_unknown(tmp_path):
    assert session_mod.repo_sha(tmp_path) == "unknown"
    assert "slurm_job_id" not in Session(tmp_path / "s").write_endpoint(5555, 5556)


# ------------------------------------------------------------------ demos


def _write_demo_snapshot(root, episode_lengths, object_dim):
    """A LeRobot-layout snapshot (meta/info.json + one parquet) in the teleop format:
    each episode ends with a padding row that copies the last action/reward."""
    import pandas as pd

    rng = np.random.default_rng(0)
    rows, index = [], 0
    for ep, n in enumerate(episode_lengths):
        actions = rng.uniform(-1, 1, (n, 7))
        for t in range(n + 1):
            last = min(t, n - 1)
            rows.append(
                {
                    "index": index,
                    "episode_index": ep,
                    "observation.state": rng.standard_normal(9).astype(np.float32),
                    "observation.environment_state": rng.standard_normal(object_dim).astype(
                        np.float32
                    ),
                    "action": actions[last].astype(np.float32),
                    "reward": np.float32(last == n - 1),
                    "done": int(last == n - 1),
                    "is_valid": int(t < n),
                }
            )
            index += 1
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": len(episode_lengths), "total_frames": index})
    )


def test_demo_loader_rejects_content_that_does_not_match_the_pin(tmp_path):
    """Offline: demos read from the wrong revision (content sha != pinned) raise."""
    import dataclasses

    from mulligan.baselines.hilserl.demos import content_sha256, load_demo_transitions

    lengths = (5, 7)
    _write_demo_snapshot(tmp_path, lengths, TASKS["square_narrow"].object_dim)
    task = dataclasses.replace(
        TASKS["square_narrow"], num_episodes=2, num_frames=sum(lengths) + 2, demos_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="wrong dataset revision"):
        load_demo_transitions(task, root=tmp_path)
    got = load_demo_transitions(task, root=tmp_path, check_sha=False)
    pinned = dataclasses.replace(task, demos_sha256=content_sha256(got))
    assert len(load_demo_transitions(pinned, root=tmp_path)["observations"]) == sum(lengths)


# --------------------------------------------------------------- takeover


def test_teleop_action_mapping_matches_collector():
    a = teleop_action(np.array([1.0, 0, 0, 0, 0, 0]), False, 1.0, 1.0)
    assert a[0] == pytest.approx(min(1.0, 0.0055 * 125)) and a[6] == -1.0
    a = teleop_action(np.array([0, 0, 0, 0.2, 0.1, 0.3]), True, 1.0, 1.0)
    # [roll, pitch, yaw] -> [pitch, roll, -yaw], x 0.0055 x 50
    assert (
        a[3] == pytest.approx(0.1 * 0.0055 * 50)
        and a[4] == pytest.approx(0.2 * 0.0055 * 50)
        and a[5] == pytest.approx(-0.3 * 0.0055 * 50)
    )
    assert a[6] == 1.0  # gripper closed -> +1 (demo convention)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_takeover_rule_latch_and_idle_filter():
    z = np.zeros(6)
    push = np.array([0.5, 0, 0, 0, 0, 0])
    readings = [
        (z, False),
        (push, False),
        (push, False),
        (z, False),
        (z, False),
        (z, False),
        (z, False),
    ]
    clock = _Clock()
    to = SpacemouseTakeover(ScriptedDevice(readings), latch_s=0.5, clock=clock)
    d = to.decide(nut_speed=0.0, gripper_closed_now=False)
    assert not d.intervening  # no input
    d = to.decide(0.0, False)
    assert d.intervening and d.bout_started and not d.idle and d.action[0] > 0
    d = to.decide(0.0, False)
    assert d.intervening and not d.bout_started and not d.idle
    clock.t = 0.2  # released, inside latch: bout continues but the step is idle (nut still)
    d = to.decide(0.0, False)
    assert (
        d.intervening
        and d.idle
        and d.exec_action[:6].tolist() == [0] * 6
        and d.exec_action[6] == -1.0
    )
    d = to.decide(
        nut_speed=0.05, gripper_closed_now=False
    )  # nut moving -> recorded even without input
    assert d.intervening and not d.idle and np.all(d.action[:6] == 0)
    clock.t = 1.0  # latch expired
    d = to.decide(0.0, False)
    assert not d.intervening
    assert to.state.stats["bouts"] == 1 and to.state.stats["idle_dropped"] == 1


def test_takeover_gripper_toggle_starts_bout_and_settles():
    z = np.zeros(6)
    readings = [(z, False), (z, True)] + [(z, True)] * (GRIPPER_SETTLE_STEPS + 2)
    clock = _Clock()
    to = SpacemouseTakeover(ScriptedDevice(readings), latch_s=0.0, clock=clock)
    assert not to.decide(0.0, False).intervening
    d = to.decide(0.0, False)
    assert d.intervening and d.bout_started and d.action[6] == 1.0 and not d.idle
    n_recorded = 1
    for _ in range(GRIPPER_SETTLE_STEPS):
        clock.t += 0.05
        d = to.decide(0.0, False)
        assert d.intervening and not d.idle
        n_recorded += 1
    clock.t += 0.05
    assert not to.decide(0.0, False).intervening
    assert n_recorded == 1 + GRIPPER_SETTLE_STEPS


def test_takeover_syncs_gripper_to_policy_between_bouts():
    """A stale device toggle must not flip the gripper when a bout starts from puck motion, and a
    button press must flip the gripper the policy has, not the stale toggle."""
    z = np.zeros(6)
    push = np.array([0.5, 0, 0, 0, 0, 0])
    clock = _Clock()
    # Toggle left "closed" by an earlier bout; policy is driving with the gripper OPEN.
    dev = ScriptedDevice([(z, True), (push, None), (push, None)])
    to = SpacemouseTakeover(dev, latch_s=0.0, clock=clock)
    assert not to.decide(0.0, gripper_closed_now=False).intervening
    assert dev.gripper_closed is False  # synced to the policy, stale "closed" discarded
    d = to.decide(0.0, gripper_closed_now=False)
    assert d.intervening and d.bout_started and d.action[6] == -1.0  # stays open
    # Policy holds the nut (closed); a press between bouts means "open", whatever the toggle was.
    dev = ScriptedDevice([(z, None), (z, False)])
    to = SpacemouseTakeover(dev, latch_s=0.0, clock=clock)
    assert not to.decide(0.0, gripper_closed_now=True).intervening
    assert dev.gripper_closed is True
    d = to.decide(0.0, gripper_closed_now=True)  # reading False != synced True: a press
    assert d.intervening and d.bout_started and d.action[6] == -1.0


def test_takeover_hold_and_deadzone():
    tiny = np.full(6, DEFAULT_DEADZONE / 2)
    to = SpacemouseTakeover(ScriptedDevice([(tiny, False)] * 3), clock=_Clock())
    assert not to.decide(0.0, False).intervening
    to.set_hold(True)
    d = to.decide(0.0, False)
    assert d.intervening and d.idle  # hold with no input = idle bout step
    to.set_hold(False)
    to.end_episode()
    assert not to.decide(0.0, False).intervening


@pytest.mark.parametrize(
    "name,max_steps",
    [("square_narrow", 300_000), ("square_broad", 1_000_000)],
)
def test_recipes_pin_the_rlpd_agent_and_schedule(name, max_steps, tmp_path):
    """The HiL-SERL recipes pin the config defaults (the RLPD agent and schedule); only the task,
    the budget and non-pinned transport keys differ. Their shared flags equal the RLPD recipe's."""
    from mulligan.baselines.hilserl.__main__ import build_parser, config_from_args
    from mulligan.sim import recipes as R

    recipes = R.load()
    recipe = R.get_recipe(recipes, f"{name.replace('_', '-')}-hilserl")
    argv = R.train_argv(recipe, "hilserl_agent", 1)
    cfg = config_from_args(
        build_parser().parse_args(["--learner", "--session", str(tmp_path), *argv])
    )
    assert cfg.pinned() == HilSerlConfig(task=name, max_steps=max_steps).pinned()
    # a later flag overrides the recipe's
    args = build_parser().parse_args(
        ["--learner", "--session", str(tmp_path), *argv, "--seed", "3"]
    )
    assert config_from_args(args).seed == 3

    rlpd = R.get_recipe(recipes, f"{name.replace('_', '-')}-rlpd")["rlpd_agent"]["args"]
    shared = dict(a.split("=", 1) for a in recipe["hilserl_agent"]["args"])
    shared = {k: v for k, v in shared.items() if k in dict(a.split("=", 1) for a in rlpd)}
    assert len(shared) >= 12
    assert shared.items() <= dict(a.split("=", 1) for a in rlpd).items()
