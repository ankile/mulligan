"""The learner: RLPD updates on the RLPD arm's schedule, fed by the split
actor over agentlace (``run_split``).

Per env step >= ``start_training`` exactly one ``agent.update(batch, utd_ratio)``
call on ``combine(demo.sample(2560), online.sample(2560))``; the UTD governor
never lets the split learner run ahead of that (``governor.py``). Episode ingestion
routes every transition to the online buffer and, when the episode was
intervened and (``demo_gate == success``) succeeded, the intervened
transitions to the growable demo buffer too (``docs/baselines.md``).

Resume: ``learner/state.pkl`` (agent bytes, both buffers, counters, ingested
episode ids, W&B run id) is written at every eval milestone, every
``resume_interval_s`` and on SIGTERM/SIGINT; a restart with the same session
continues, and a reconnecting actor re-pushes anything not in
``ingested_ids``. Eval is NOT run in this process: checkpoints
land in ``learner/checkpoints/step_NNNNNNN/`` for the CPU eval watcher
(``evaluate.py``) whose results this loop tails and logs.
"""

from __future__ import annotations

import os
import pickle
import queue
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from mulligan.baselines.hilserl import require_jax
from mulligan.baselines.hilserl.config import HilSerlConfig
from mulligan.baselines.hilserl.demos import load_demo_transitions
from mulligan.baselines.hilserl.governor import Governor
from mulligan.baselines.hilserl.policy import actor_params_to_numpy
from mulligan.baselines.hilserl.replay import (
    TRANSITION_KEYS,
    ReplayBuffer,
    combine,
)
from mulligan.baselines.hilserl.session import (
    Session,
    hilserl_sha,
    read_jsonl,
    repo_sha,
    write_json_atomic,
)

PREEMPTED_EXIT_CODE = 75  # saved resume state after a signal, expects relaunch (as RLPD)


def _log(msg: str) -> None:
    print(f"[learner {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Learner:
    def __init__(
        self, cfg: HilSerlConfig, session: Session, wandb_run=None, demos: Optional[dict] = None
    ):
        require_jax()
        import jax
        from flax import serialization  # noqa: F401  (import check)

        from mulligan.baselines.rlpd.agent import SACLearner

        self.cfg = cfg
        self.session = session
        self.wandb = wandb_run
        self.demos = demos if demos is not None else load_demo_transitions(cfg.task_spec)
        n_demo = len(self.demos["observations"])
        example_obs = self.demos["observations"][0]
        example_act = self.demos["actions"][0]
        self.agent = SACLearner.create(cfg.seed, example_obs, example_act, **cfg.agent_kwargs())
        # The actor stops after the first episode that crosses ``max_steps`` (never
        # mid-episode), so the last episode can end up to ``horizon - 1`` steps past it;
        # size both buffers for that or the final ingest raises "buffer full" before the
        # ``max_steps`` milestone.
        self.online = ReplayBuffer(example_obs, example_act, cfg.max_steps + cfg.horizon)
        self.online.seed(cfg.seed)
        self.demo_buf = ReplayBuffer(example_obs, example_act, n_demo + cfg.max_steps + cfg.horizon)
        self.demo_buf.insert_dataset(self.demos)
        self.demo_buf.seed(cfg.seed + 1)
        self.n_demo = n_demo
        self.governor = Governor(cfg.start_training)
        self.counters = dict(
            env_steps=0,
            calls_done=0,
            episodes=0,
            successes=0,
            intervened_steps=0,
            bouts=0,
            demo_added=0,
            fail_terminated=0,
            idle_dropped=0,
            redo_count=0,
        )
        self.ingested_ids: set[int] = set()
        self.param_version = 0
        self.next_milestone = 0
        self.recent = deque(maxlen=20)
        self.stop_requested = False
        self._last_resume_save = time.time()
        self._last_saved_counters: Optional[dict] = None
        self._incoming: "queue.Queue[tuple[int, dict, dict]]" = queue.Queue()
        self._lock = threading.Lock()
        self._eval_rows_logged = 0
        self._devices = jax.devices()
        _log(f"jax devices {self._devices}; demos {n_demo}; agent {type(self.agent).__name__}")

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        from flax import serialization

        return {
            "agent": serialization.to_bytes(self.agent),
            "online": self.online.state(),
            "demo": self.demo_buf.state(),
            "counters": dict(self.counters),
            "ingested_ids": sorted(self.ingested_ids),
            "param_version": self.param_version,
            "next_milestone": self.next_milestone,
            "recent": list(self.recent),
            "config_hash": self.cfg.pinned_hash(),
            "wandb_run_id": self.wandb.id if self.wandb is not None else None,
            "saved_at": time.time(),
        }

    def save_resume(self, reason: str) -> Path:
        path = self.session.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(self.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self._last_resume_save = time.time()
        self._last_saved_counters = dict(self.counters)
        _log(
            f"RESUME_STATE saved ({reason}) env_steps={self.counters['env_steps']} calls={self.counters['calls_done']} -> {path}"
        )
        return path

    def load_resume(self) -> bool:
        from flax import serialization

        path = self.session.state_path
        if not path.exists():
            return False
        with open(path, "rb") as f:
            st = pickle.load(f)
        if st["config_hash"] != self.cfg.pinned_hash():
            raise RuntimeError(
                f"{path} was saved with pinned-config hash {st['config_hash']}, this run has {self.cfg.pinned_hash()}"
            )
        self.agent = serialization.from_bytes(self.agent, st["agent"])
        self.online.load_state(st["online"])
        self.demo_buf.load_state(st["demo"])
        self.counters = dict(st["counters"])
        self.ingested_ids = set(st["ingested_ids"])
        self.param_version = st["param_version"]
        self.next_milestone = st["next_milestone"]
        self.recent = deque(st["recent"], maxlen=20)
        if len(self.online) != self.counters["env_steps"]:
            raise RuntimeError(
                f"online buffer has {len(self.online)} rows but counters say {self.counters['env_steps']} env steps"
            )
        _log(
            f"RESUMED from {path}: env_steps={self.counters['env_steps']} calls={self.counters['calls_done']} episodes={self.counters['episodes']} next_milestone={self.next_milestone}"
        )
        return True

    # ---------------------------------------------------------------- ingest
    def ingest_transitions(self, arrays: dict, intervened: np.ndarray, success: bool) -> int:
        """Insert one episode's transitions; returns the number of demo rows added."""
        n = len(arrays["observations"])
        intervened = np.asarray(intervened, dtype=bool)
        if intervened.shape != (n,):
            raise ValueError(f"intervened mask shape {intervened.shape} != ({n},)")
        gate_ok = self.cfg.demo_gate == "none" or bool(success)
        demo_added = 0
        for i in range(n):
            tr = {k: arrays[k][i] for k in TRANSITION_KEYS}
            self.online.insert(tr)
            if intervened[i] and gate_ok:
                self.demo_buf.insert(tr)
                demo_added += 1
        self.counters["env_steps"] += n
        self.counters["intervened_steps"] += int(intervened.sum())
        self.counters["demo_added"] += demo_added
        return demo_added

    def ingest_episode(self, episode_id: int, arrays: dict, record: dict) -> bool:
        """Insert an actor episode (idempotent on episode_id). Returns False for duplicates."""
        with self._lock:
            if episode_id in self.ingested_ids:
                return False
            n = int(record["length"])
            if n != len(arrays["observations"]):
                raise ValueError(
                    f"episode {episode_id}: record.length {n} != {len(arrays['observations'])} rows"
                )
            if self.counters["env_steps"] + n > self.online.capacity:
                raise RuntimeError(
                    f"online buffer full ({self.counters['env_steps']} + {n} > {self.online.capacity}): "
                    "max_steps reached; stop the actor"
                )
            demo_added = self.ingest_transitions(
                arrays, arrays["intervened"], bool(record["success"])
            )
            self.ingested_ids.add(episode_id)
            self.counters["episodes"] += 1
            self.counters["successes"] += int(bool(record["success"]))
            self.counters["bouts"] += int(record["intervention_count"])
            self.counters["fail_terminated"] += int(bool(record["fail_terminated"]))
            self.counters["idle_dropped"] += int(record["idle_steps_dropped"])
            self.counters["redo_count"] += int(record["redo_count"])
            self.recent.append(
                (float(bool(record["success"])), float(record["intervention_steps"]) / max(1, n))
            )
            env_step = self.counters["env_steps"]
        self.log_episode(record, n, demo_added, env_step)
        return True

    def log_episode(self, record: dict, n: int, demo_added: int, env_step: int) -> None:
        payload = {
            "train/return": float(
                np.sum(record.get("extra", {}).get("return", float(record["success"])))
            ),
            "train/length": n,
            "train/success": float(bool(record["success"])),
            "hil/episode_intervention_steps": int(record["intervention_steps"]),
            "hil/episode_intervention_frac": float(record["intervention_steps"]) / max(1, n),
            "hil/episode_intervention_count": int(record["intervention_count"]),
            "hil/episode_intervention_kept_to_demo": demo_added,
            "hil/episode_fail_terminated": float(bool(record["fail_terminated"])),
            "hil/episode_idle_dropped": int(record["idle_steps_dropped"]),
            "hil/episode_redo_count": int(record["redo_count"]),
        }
        payload.update(self.hil_metrics())
        self.wandb_log(payload, env_step)

    def hil_metrics(self) -> dict:
        c = self.counters
        n_demo_buf = len(self.demo_buf)
        rec = list(self.recent)
        return {
            "hil/env_steps": c["env_steps"],
            "hil/grad_steps": c["calls_done"] * self.cfg.utd_ratio,
            "hil/update_calls": c["calls_done"],
            "hil/utd_effective": (c["calls_done"] * self.cfg.utd_ratio)
            / max(1, c["env_steps"] - self.cfg.start_training)
            if c["env_steps"] > self.cfg.start_training
            else 0.0,
            "hil/owed_calls": self.governor.owed(c["env_steps"], c["calls_done"]),
            "hil/intervention_steps_cum": c["intervened_steps"],
            "hil/intervention_count_cum": c["bouts"],
            "hil/human_frames_cum": c["intervened_steps"],
            "hil/intervention_rate_20ep": float(np.mean([r[1] for r in rec])) if rec else 0.0,
            "hil/success_rate_20ep": float(np.mean([r[0] for r in rec])) if rec else 0.0,
            "hil/episodes": c["episodes"],
            "hil/fail_terminated_cum": c["fail_terminated"],
            "hil/idle_dropped_cum": c["idle_dropped"],
            "hil/redo_cum": c["redo_count"],
            "buffer/online_size": len(self.online),
            "buffer/demo_size": n_demo_buf,
            "buffer/online_intervention_frac": c["intervened_steps"] / max(1, c["env_steps"]),
            "buffer/demo_intervention_frac": c["demo_added"] / max(1, n_demo_buf),
            "buffer/demo_demo_frac": self.n_demo / max(1, n_demo_buf),
            "hil/param_version": self.param_version,
        }

    # ---------------------------------------------------------------- update
    def update_once(self) -> dict:
        c = self.cfg
        batch = combine(self.demo_buf.sample(c.offline_batch), self.online.sample(c.online_batch))
        self.agent, info = self.agent.update(batch, c.utd_ratio)
        self.counters["calls_done"] += 1
        return info

    def env_step_of_last_call(self) -> int:
        return self.cfg.start_training + self.counters["calls_done"]

    def params_payload(self) -> dict:
        return {
            "params": actor_params_to_numpy(self.agent.actor.params),
            "param_version": self.param_version,
        }

    def maybe_log_update(self, info: dict) -> None:
        step = self.env_step_of_last_call()
        if step % self.cfg.log_interval == 0:
            payload = {f"train/{k}": float(v) for k, v in info.items()}
            payload.update(self.hil_metrics())
            self.wandb_log(payload, step)

    # ------------------------------------------------------------ milestones
    def milestone_due(self) -> Optional[int]:
        m = self.next_milestone
        if m > self.cfg.max_steps:
            return None
        if self.counters["env_steps"] >= m and self.counters[
            "calls_done"
        ] >= self.governor.target_calls(m):
            return m
        return None

    def checkpoint_dir(self, env_step: int) -> Path:
        return self.session.checkpoints_dir / f"step_{env_step:07d}"

    def save_checkpoint(self, env_step: int) -> Path:
        from flax import serialization

        final = self.checkpoint_dir(env_step)
        if final.exists():
            _log(f"checkpoint {final} exists (resume replay); keeping it")
            return final
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=final.parent, prefix=f".step_{env_step:07d}-"))
        with open(tmp / "actor_params.pkl", "wb") as f:
            pickle.dump(self.params_payload(), f, protocol=pickle.HIGHEST_PROTOCOL)
        (tmp / "agent.msgpack").write_bytes(serialization.to_bytes(self.agent))
        write_json_atomic(
            tmp / "meta.json",
            {
                "env_step": env_step,
                "counters": dict(self.counters),
                "param_version": self.param_version,
                "config": self.cfg.to_dict(),
                "repo_sha": repo_sha(),
            },
        )
        os.replace(tmp, final)
        _log(f"checkpoint saved {final}")
        return final

    def tail_eval_ledger(self) -> None:
        rows = read_jsonl(self.session.eval_ledger)
        for row in rows[self._eval_rows_logged :]:
            payload = {f"eval/{k}": v for k, v in row["metrics"].items()}
            payload["eval/env_step"] = row["env_step"]
            self.wandb_log(payload, int(row["env_step"]))
            _log(f"eval @ {row['env_step']}: {row['metrics']}")
        self._eval_rows_logged = len(rows)

    # ---------------------------------------------------------------- wandb
    def wandb_log(self, payload: dict, env_step: int) -> None:
        if self.wandb is None:
            return
        payload = dict(payload)
        payload["env_step"] = int(env_step)
        self.wandb.log(payload)

    # ---------------------------------------------------------------- signals
    def install_signal_handlers(self) -> None:
        def _request_stop(signum, _frame):
            self.stop_requested = True
            _log(f"SIGNAL {signum}: will save resume state and exit {PREEMPTED_EXIT_CODE}")

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)

    def _stop_and_exit(self) -> None:
        self.save_resume("signal")
        sys.stdout.flush()
        os._exit(PREEMPTED_EXIT_CODE)

    def finished(self) -> bool:
        return (
            self.counters["env_steps"] >= self.cfg.max_steps
            and self.governor.owed(self.counters["env_steps"], self.counters["calls_done"]) == 0
            and self.next_milestone > self.cfg.max_steps
        )

    # ------------------------------------------------------------- transport
    def request_callback(self, type_: str, payload: dict) -> dict:
        """agentlace request handler (server thread)."""
        try:
            if type_ == "hello":
                if payload["hilserl_sha"] != hilserl_sha():
                    return {
                        "success": False,
                        "message": (
                            f"hilserl content mismatch: learner {hilserl_sha()} "
                            f"(commit {repo_sha()[:10]}) actor {payload['hilserl_sha']} "
                            f"(commit {payload['repo_sha'][:10]})"
                        ),
                    }
                if payload["config_hash"] != self.cfg.pinned_hash():
                    return {
                        "success": False,
                        "message": f"pinned-config mismatch: learner {self.cfg.pinned_hash()} actor {payload['config_hash']}",
                    }
                if payload["session_id"] != self.session.root.name:
                    return {
                        "success": False,
                        "message": f"session mismatch: learner {self.session.root.name} actor {payload['session_id']}",
                    }
                with self._lock:
                    return {
                        "success": True,
                        "ingested_ids": sorted(self.ingested_ids | self._queued_ids()),
                        "counters": dict(self.counters),
                        "owed_calls": self.governor.owed(
                            self.counters["env_steps"], self.counters["calls_done"]
                        ),
                        **self.params_payload(),
                    }
            if type_ == "push-episode":
                if payload["session_id"] != self.session.root.name:
                    return {"success": False, "message": "session mismatch"}
                ep_id = int(payload["episode_id"])
                with self._lock:
                    dup = ep_id in self.ingested_ids or ep_id in self._queued_ids()
                if not dup:
                    self._incoming.put((ep_id, payload["arrays"], payload["record"]))
                return {"success": True, "queued": not dup, "duplicate": dup}
            if type_ == "status":
                with self._lock:
                    return {
                        "success": True,
                        "counters": dict(self.counters),
                        "owed_calls": self.governor.owed(
                            self.counters["env_steps"], self.counters["calls_done"]
                        ),
                        "param_version": self.param_version,
                        "ingested_ids": sorted(self.ingested_ids | self._queued_ids()),
                    }
            return {"success": False, "message": f"unknown request {type_}"}
        except Exception as exc:  # the server thread must not die silently
            _log(f"request {type_} failed: {exc!r}")
            return {"success": False, "message": repr(exc)}

    def _queued_ids(self) -> set[int]:
        """Episodes received but not yet inserted by the main loop (acknowledged to the actor)."""
        return {e[0] for e in list(self._incoming.queue)}

    def drain_incoming(self) -> int:
        n = 0
        while True:
            try:
                ep_id, arrays, record = self._incoming.get_nowait()
            except queue.Empty:
                return n
            if self.ingest_episode(ep_id, arrays, record):
                n += 1
                _log(
                    f"ingested episode {ep_id}: len={record['length']} success={record['success']} intervened={record['intervention_steps']} env_steps={self.counters['env_steps']}"
                )

    # ------------------------------------------------------------------ loops
    def publish(self, server) -> None:
        self.param_version = self.counters["calls_done"]
        server.publish(self.params_payload())

    def run_split(self, server, heartbeat_s: float = 30.0) -> None:
        """Serve the actor; update on the governor's schedule; save/eval by milestone."""
        self.install_signal_handlers()
        self.session.write_endpoint(
            self.cfg.port,
            self.cfg.port + 1,
            {"config_hash": self.cfg.pinned_hash(), "session_id": self.session.root.name},
        )
        self.publish(server)
        last_hb = time.time()
        while not self.finished():
            if self.stop_requested:
                self._stop_and_exit()
            self.drain_incoming()
            m = self.milestone_due()
            if m is not None:
                self.save_checkpoint(m)
                self.next_milestone = m + self.cfg.eval_interval
                self.save_resume(f"milestone {m}")
            owed = self.governor.owed(self.counters["env_steps"], self.counters["calls_done"])
            if owed > 0:
                info = self.update_once()
                self.maybe_log_update(info)
                if self.counters["calls_done"] % self.cfg.steps_per_publish == 0:
                    self.publish(server)
                if self.governor.owed(self.counters["env_steps"], self.counters["calls_done"]) == 0:
                    # The debt just drained. hil/* rows are otherwise only logged while updating,
                    # so hil/owed_calls would keep its last busy value while the actor is idle.
                    self.wandb_log(self.hil_metrics(), self.counters["env_steps"])
            else:
                time.sleep(0.02)
            now = time.time()
            if now - self._last_resume_save > self.cfg.resume_interval_s:
                if self._last_saved_counters != self.counters:
                    self.save_resume("periodic")
                else:
                    self._last_resume_save = now
            if now - last_hb > heartbeat_s:
                self.tail_eval_ledger()
                c = self.counters
                _log(
                    f"heartbeat env_steps={c['env_steps']} calls={c['calls_done']} owed={self.governor.owed(c['env_steps'], c['calls_done'])} episodes={c['episodes']} demo_buf={len(self.demo_buf)} v={self.param_version}"
                )
                last_hb = now
        self.tail_eval_ledger()
        self.save_resume("final")
        _log("finished")


def init_wandb(cfg: HilSerlConfig, session: Session, role: str):
    """W&B run keyed by env_step (define_metric), resumable across restarts."""
    if cfg.wandb_mode == "disabled":
        return None
    import wandb

    run_id_path = session.learner_dir / "wandb_run_id.txt"
    run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
    name = cfg.wandb_name or f"{cfg.task_spec.run_stem}_{role}_s{cfg.seed}"
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=name,
        id=run_id,
        resume="allow",
        mode=cfg.wandb_mode,
        config={
            **cfg.to_dict(),
            "role": role,
            "session": str(session.root),
            "repo_sha": repo_sha(),
        },
    )
    run.define_metric("env_step")
    run.define_metric("*", step_metric="env_step")
    session.learner_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id_path.write_text(run.id)
    return run
