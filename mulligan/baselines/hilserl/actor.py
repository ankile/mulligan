"""The actor: env stepping + CPU inference + human takeover, on the
operator's machine. Every completed episode is written to the session log (source
of truth) and pushed to the learner; on restart the actor resumes numbering
and counters from the ledger and re-pushes anything the learner has not
acknowledged, so the collection can be killed and resumed at any time.

Kill/resume semantics:
    q (or one SIGINT/Ctrl-C)  finish the current episode, flush, exit
    second SIGINT             abort now; the partial episode is discarded
    restart with the same --session --ip  continues at the next episode id
"""

from __future__ import annotations

import signal
import time
from dataclasses import asdict
from typing import Optional

import numpy as np

from mulligan.baselines.hilserl.config import HilSerlConfig
from mulligan.baselines.hilserl.env import HilSerlEnv
from mulligan.baselines.hilserl.policy import ActorPolicy
from mulligan.baselines.hilserl.session import (
    EpisodeRecord,
    Session,
    hilserl_sha,
    repo_sha,
)
from mulligan.baselines.hilserl.takeover import (
    ScriptedKeys,
    SpacemouseTakeover,
    TakeoverDecision,
)
from mulligan.baselines.hilserl.transport import ActorClient


def _log(msg: str) -> None:
    print(f"[actor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Actor:
    def __init__(
        self,
        cfg: HilSerlConfig,
        session: Session,
        env: Optional[HilSerlEnv],
        policy: Optional[ActorPolicy],
        client: Optional[ActorClient],
        takeover: Optional[SpacemouseTakeover] = None,
        keys=None,
        paced: bool = True,
        control_hz: float = 20.0,
        max_episodes: Optional[int] = None,
        pause_poll_s: float = 0.05,
    ):
        self.cfg = cfg
        self.session = session
        self.env = env
        self.policy = policy
        self.client = client
        self.takeover = takeover
        self.keys = keys if keys is not None else ScriptedKeys()
        self.paced = paced
        self.period = 1.0 / control_hz
        self.max_episodes = max_episodes
        self.pause_poll_s = pause_poll_s
        self.stop_after_episode = False
        self.abort_now = False
        self.paused = False
        self._sigints = 0
        self.pending_push: list[int] = []
        self.ingested_ids: set[int] = set()

    # ------------------------------------------------------------ signals
    def install_signal_handlers(self) -> None:
        def _on_sigint(signum, _frame):
            self._sigints += 1
            if self._sigints == 1:
                self.stop_after_episode = True
                _log("SIGINT: finishing this episode then stopping (press again to abort now)")
            else:
                self.abort_now = True
                _log("SIGINT x2: aborting now; the partial episode is discarded")

        signal.signal(signal.SIGINT, _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigint)

    # ------------------------------------------------------------ transport
    def on_params(self, payload: dict) -> None:
        self.policy.set_params(payload["params"], payload["param_version"])

    def hello(self) -> bool:
        if self.client is None:
            return False
        reply = self.client.request(
            "hello",
            {
                "session_id": self.session.root.name,
                "hilserl_sha": hilserl_sha(),
                "repo_sha": repo_sha(),
                "config_hash": self.cfg.pinned_hash(),
            },
        )
        if reply is None:
            _log("hello: learner unreachable")
            return False
        if not reply.get("success"):
            raise RuntimeError(f"learner refused the actor: {reply.get('message')}")
        self.ingested_ids = set(reply["ingested_ids"])
        if self.policy is not None:
            self.on_params(reply)
        _log(
            f"hello ok: learner env_steps={reply['counters']['env_steps']} calls={reply['counters']['calls_done']} ingested={len(self.ingested_ids)} params v{reply['param_version']}"
        )
        return True

    def push_episode(self, episode_id: int, rec: Optional[EpisodeRecord] = None) -> bool:
        """Send one logged episode; ``rec`` is its ledger record (read if not given)."""
        if self.client is None:
            return True
        arrays = self.session.load_episode(episode_id)
        if rec is None:
            rec = next(r for r in self.session.read_ledger() if r.episode_id == episode_id)
        reply = self.client.request(
            "push-episode",
            {
                "session_id": self.session.root.name,
                "episode_id": episode_id,
                "record": asdict(rec),
                "arrays": {k: v for k, v in arrays.items() if k != "initial_sim_state"},
            },
        )
        if reply is None or not reply.get("success"):
            _log(
                f"push episode {episode_id} failed ({None if reply is None else reply.get('message')}); will retry"
            )
            return False
        return True

    def sync_pending(self) -> None:
        """Push every logged episode the learner has not acknowledged."""
        if self.client is None:
            return
        reply = self.client.request("status", {})
        if reply is None:
            _log("status: learner unreachable; episodes stay queued locally")
            return
        self.ingested_ids = set(reply["ingested_ids"])
        # One ledger read for all of them (read_ledger stats every episode file).
        missing_records = [
            r for r in self.session.read_ledger() if r.episode_id not in self.ingested_ids
        ]
        missing = [r.episode_id for r in missing_records]
        for rec in missing_records:
            if not self.push_episode(rec.episode_id, rec):
                break
        if missing:
            _log(
                f"re-pushed {len(missing)} episode(s) not yet ingested: {missing[:8]}{'...' if len(missing) > 8 else ''}"
            )

    def flush(self) -> bool:
        """For a ledger that already holds ``max_steps``: re-push every episode the
        learner has not acknowledged. True once the learner holds all of them."""
        if not self.hello():
            raise RuntimeError("learner unreachable; check the endpoint / tunnel")
        self.sync_pending()
        reply = self.client.request("status", {})
        if reply is None:
            _log("status: learner unreachable after the re-push")
            return False
        missing = sorted(
            {r.episode_id for r in self.session.read_ledger()} - set(reply["ingested_ids"])
        )
        if missing:
            _log(f"learner still lacks {len(missing)} episode(s): {missing[:8]}")
        return not missing

    def wait_for_learner(self) -> None:
        """Actor-side UTD floor: at an episode boundary, wait while the learner
        owes more than ``actor_max_lag`` update calls (never mid-episode)."""
        if self.client is None:
            return
        while not self.abort_now:
            reply = self.client.request("status", {})
            if reply is None:
                _log("status: learner unreachable; continuing without the lag check")
                return
            owed = int(reply["owed_calls"])
            if owed <= self.cfg.actor_max_lag:
                return
            _log(
                f"learner owes {owed} update calls (> {self.cfg.actor_max_lag}); waiting at the episode boundary"
            )
            time.sleep(2.0)

    # --------------------------------------------------------------- keys
    def handle_keys(self) -> dict:
        flags = {"redo": False, "fail": False}
        for k in self.keys.drain():
            if k == "q":
                self.stop_after_episode = True
                _log("q: stopping after this episode")
            elif k == "p":
                self.paused = not self.paused
                _log("paused" if self.paused else "resumed")
            elif k == "t":
                self.paced = not self.paced
                _log(f"pacing {'on (real time)' if self.paced else 'off (as fast as possible)'}")
            elif k == "h" and self.takeover is not None:
                _log(f"hold takeover {'ON' if self.takeover.toggle_hold() else 'off'}")
            elif k == "r":
                flags["redo"] = True
            elif k == "x":
                flags["fail"] = True
        return flags

    # ------------------------------------------------------------- episode
    def run_episode(
        self, episode_id: int, env_steps_before: int
    ) -> Optional[tuple[EpisodeRecord, dict, np.ndarray]]:
        env = self.env
        obs = env.reset()
        initial_sim_state = env.sim_state_flat()
        rows = {
            k: []
            for k in (
                "observations",
                "actions",
                "policy_actions",
                "rewards",
                "masks",
                "dones",
                "next_observations",
                "intervened",
            )
        }
        bout_snapshot = None
        bout_start_len = 0
        intervention_count = 0
        intervention_steps = 0
        idle_dropped = 0
        redo_count = 0
        fail_terminated = False
        success = False
        ep_return = 0.0
        wall_start = time.time()
        next_tick = time.monotonic()
        if self.takeover is not None:
            self.takeover.end_episode()
        gripper_closed = False  # robosuite resets the gripper open; tracks the executed command
        while True:
            if self.abort_now:
                return None
            flags = self.handle_keys()
            while self.paused and not self.abort_now:
                env.render()
                time.sleep(self.pause_poll_s)
                flags = self.handle_keys() | {k: v for k, v in flags.items() if v}
            if flags["redo"]:
                if bout_snapshot is not None:
                    obs = env.restore(bout_snapshot)
                    dropped = len(rows["observations"]) - bout_start_len
                    for k in rows:
                        del rows[k][bout_start_len:]
                    intervention_steps = (
                        int(np.sum(rows["intervened"])) if rows["intervened"] else 0
                    )
                    ep_return = float(np.sum(rows["rewards"])) if rows["rewards"] else 0.0
                    redo_count += 1
                    bout_snapshot = None
                    if self.takeover is not None:
                        self.takeover.end_episode()
                    _log(
                        f"redo: restored the takeover-onset state, dropped {dropped} recorded steps (t={env.t})"
                    )
                else:
                    _log("redo: no intervention bout to redo in this episode")
            policy_action = self.policy.sample(obs)
            decision: Optional[TakeoverDecision] = (
                self.takeover.decide(env.nut_speed(), gripper_closed)
                if self.takeover is not None
                else None
            )
            if decision is not None and decision.bout_started:
                bout_snapshot = env.snapshot()
                bout_start_len = len(rows["observations"])
                intervention_count += 1
            if decision is not None and decision.idle:
                env.step(decision.exec_action, count=False)
                gripper_closed = bool(decision.exec_action[6] > 0)
                idle_dropped += 1
                env.render()
                next_tick = self._pace(next_tick)
                continue
            intervened = decision is not None and decision.intervening
            action = decision.action if intervened else policy_action
            res = env.step(action)
            gripper_closed = bool(action[6] > 0)
            reward, mask, done = res.reward, res.mask, res.done
            if flags["fail"] and not res.success:
                reward, mask, done, fail_terminated = 0.0, 0.0, True, True
                _log("x: episode ended as failure by the operator")
            rows["observations"].append(np.asarray(obs, np.float32))
            rows["actions"].append(np.asarray(action, np.float32))
            rows["policy_actions"].append(np.asarray(policy_action, np.float32))
            rows["rewards"].append(np.float32(reward))
            rows["masks"].append(np.float32(mask))
            rows["dones"].append(bool(done))
            rows["next_observations"].append(np.asarray(res.obs, np.float32))
            rows["intervened"].append(bool(intervened))
            if intervened:
                intervention_steps += 1
            ep_return += reward
            obs = res.obs
            env.render()
            next_tick = self._pace(next_tick)
            if done:
                success = bool(res.success)
                break
        n = len(rows["observations"])
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        record = EpisodeRecord(
            episode_id=episode_id,
            length=n,
            success=success,
            fail_terminated=fail_terminated,
            intervention_count=intervention_count,
            intervention_steps=intervention_steps,
            idle_steps_dropped=idle_dropped,
            redo_count=redo_count,
            param_version=self.policy.version,
            env_steps_before=env_steps_before,
            wall_start=wall_start,
            wall_end=time.time(),
            extra={
                "return": float(ep_return),
                "paced": self.paced,
                "hold": bool(self.takeover.state.hold) if self.takeover else False,
            },
        )
        return record, arrays, initial_sim_state

    def _pace(self, next_tick: float) -> float:
        if not self.paced:
            return time.monotonic()
        next_tick += self.period
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -1.0:
            next_tick = time.monotonic()  # fell far behind (pause, GC); do not burst
        return next_tick

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        self.install_signal_handlers()
        episode_id = self.session.next_episode_id()
        env_steps = self.session.env_steps_logged()
        _log(
            f"session {self.session.root}: resuming at episode {episode_id}, env_steps {env_steps}/{self.cfg.max_steps}"
        )
        if self.client is not None:
            if not self.hello():
                raise RuntimeError("learner unreachable at start; check the endpoint / tunnel")
            self.client.on_params(self.on_params)
            self.sync_pending()
        episodes_this_run = 0
        while env_steps < self.cfg.max_steps and not self.stop_after_episode and not self.abort_now:
            if self.max_episodes is not None and episodes_this_run >= self.max_episodes:
                break
            self.wait_for_learner()
            out = self.run_episode(episode_id, env_steps)
            if out is None:
                break
            record, arrays, initial_sim_state = out
            self.session.write_episode(record, arrays, initial_sim_state)
            env_steps = record.env_steps_after
            _log(
                f"episode {episode_id}: len={record.length} success={record.success} bouts={record.intervention_count} human_steps={record.intervention_steps} idle_dropped={record.idle_steps_dropped} redo={record.redo_count} fail={record.fail_terminated} v{record.param_version} env_steps={env_steps}"
            )
            if not self.push_episode(episode_id, record):
                self.pending_push.append(episode_id)
            elif self.pending_push:
                self.sync_pending()
                self.pending_push = []
            episode_id += 1
            episodes_this_run += 1
        if self.client is not None:
            self.sync_pending()
            self.client.stop()
        _log(f"stopped at episode {episode_id}, env_steps {env_steps}")
        return env_steps
