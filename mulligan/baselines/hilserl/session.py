"""On-disk session: the actor's authoritative episode log, ledgers, endpoint
file, and resume bookkeeping.

Layout (``<session>/``):

    session.json                 first writer's metadata (session_id, code hashes, config)
    actor/episodes/ep_NNNNNN.npz one file per completed episode (all fields)
    actor/ledger.jsonl           one JSON line per completed episode
    learner/state.pkl            learner resume state (agent + buffers + counters)
    learner/checkpoints/step_NNNNNNN/  flax checkpoints for the eval watcher
    learner/endpoint.json        {hostname, ip, ports, repo_sha, hilserl_sha, ...}
    eval/native_NNNNNNN.json     eval-watcher results per checkpoint
    eval/ledger.jsonl

Everything is written atomically (tmp file + ``os.replace``); the ledger is
append-only. The npz files are the source of truth: the learner's buffers can
be rebuilt from them, and a restarted actor re-pushes any episode the learner
has not acknowledged (see ``actor.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EPISODE_ARRAY_KEYS = (
    "observations",
    "actions",
    "policy_actions",
    "rewards",
    "masks",
    "dones",
    "next_observations",
    "intervened",
)


@dataclass
class EpisodeRecord:
    episode_id: int
    length: int
    success: bool
    fail_terminated: bool
    intervention_count: int
    intervention_steps: int
    idle_steps_dropped: int
    redo_count: int
    param_version: int
    env_steps_before: int
    wall_start: float
    wall_end: float
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def env_steps_after(self) -> int:
        return self.env_steps_before + self.length


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{i + 1} is not valid JSON ({exc}); the ledger is corrupt"
                ) from exc
    return rows


def repo_sha(repo_root: Optional[Path] = None) -> str:
    """Commit of the checkout this process runs from, or ``"unknown"`` for an installed
    package (no ``.git`` at the repo root). Provenance only — see ``hilserl_sha``."""
    root = repo_root or Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return "unknown"
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


_PACKAGE_ROOT = Path(__file__).resolve().parent
# The agent both roles run (the RLPD SAC agent), hashed with this package.
_AGENT_ROOT = _PACKAGE_ROOT.parent / "rlpd" / "agent"
# Offline tools do not run inside the actor/learner loop, so editing them must not
# lock a running actor out of its learner.
_NON_CONTRACT = frozenset({"tools"})


def hilserl_sha() -> str:
    """Content hash of the sources the actor and the learner must agree on.

    The handshake compares this, not ``repo_sha``: the commit sha moves with every
    unrelated commit in the repo, so an unrelated edit would lock a running actor
    out of a learner whose sources are identical. Covers this
    package (minus ``tools/``) and ``mulligan.baselines.rlpd.agent``.
    """
    h = hashlib.sha256()
    for root in (_PACKAGE_ROOT, _AGENT_ROOT):
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root)
            if root is _PACKAGE_ROOT and rel.parts[0] in _NON_CONTRACT:
                continue
            h.update(f"{root.name}/{rel}".encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


class Session:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.actor_dir = self.root / "actor"
        self.episodes_dir = self.actor_dir / "episodes"
        self.actor_ledger = self.actor_dir / "ledger.jsonl"
        self.learner_dir = self.root / "learner"
        self.state_path = self.learner_dir / "state.pkl"
        self.checkpoints_dir = self.learner_dir / "checkpoints"
        self.endpoint_path = self.learner_dir / "endpoint.json"
        self.eval_dir = self.root / "eval"
        self.eval_ledger = self.eval_dir / "ledger.jsonl"
        self.meta_path = self.root / "session.json"

    # --- metadata -----------------------------------------------------------
    def ensure_meta(self, role: str, config: dict) -> dict:
        """Create ``session.json`` on first use; on later uses verify the pinned
        config keys match (a resumed session must not silently change the
        experiment)."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            pinned = meta["config"]
            mismatch = {
                k: (pinned[k], config[k]) for k in pinned if k in config and pinned[k] != config[k]
            }
            if mismatch:
                raise RuntimeError(f"{self.meta_path}: resumed with a different config: {mismatch}")
            return meta
        meta = {
            "session_id": self.root.name,
            "created_at": time.time(),
            "created_by": role,
            "created_on": socket.gethostname(),
            "repo_sha": repo_sha(),
            "hilserl_sha": hilserl_sha(),
            "config": config,
        }
        write_json_atomic(self.meta_path, meta)
        return meta

    # --- actor episode log -------------------------------------------------
    def episode_path(self, episode_id: int) -> Path:
        return self.episodes_dir / f"ep_{episode_id:06d}.npz"

    def write_episode(
        self, record: EpisodeRecord, arrays: dict, initial_sim_state: np.ndarray
    ) -> Path:
        missing = set(EPISODE_ARRAY_KEYS) - set(arrays)
        if missing:
            raise KeyError(f"episode arrays missing {sorted(missing)}")
        n = record.length
        for k in EPISODE_ARRAY_KEYS:
            if len(arrays[k]) != n:
                raise ValueError(f"{k} has {len(arrays[k])} rows, record.length is {n}")
        path = self.episode_path(record.episode_id)
        if path.exists():
            raise FileExistsError(f"{path} already exists; episode ids must be unique")
        write_npz_atomic(
            path,
            initial_sim_state=np.asarray(initial_sim_state),
            **{k: np.asarray(arrays[k]) for k in EPISODE_ARRAY_KEYS},
        )
        append_jsonl(self.actor_ledger, asdict(record))
        return path

    def read_ledger(self) -> list[EpisodeRecord]:
        rows = read_jsonl(self.actor_ledger)
        records = [EpisodeRecord(**row) for row in rows]
        ids = [r.episode_id for r in records]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError(
                f"{self.actor_ledger}: episode ids are not strictly increasing: {ids[:10]}..."
            )
        for r in records:
            if not self.episode_path(r.episode_id).exists():
                raise FileNotFoundError(
                    f"ledger lists episode {r.episode_id} but {self.episode_path(r.episode_id)} is missing"
                )
        return records

    def load_episode(self, episode_id: int) -> dict:
        with np.load(self.episode_path(episode_id)) as z:
            return {k: z[k] for k in z.files}

    def next_episode_id(self) -> int:
        records = self.read_ledger()
        return (records[-1].episode_id + 1) if records else 0

    def env_steps_logged(self) -> int:
        return sum(r.length for r in self.read_ledger())

    # --- endpoint --------------------------------------------------------------
    def write_endpoint(self, port: int, broadcast_port: int, extra: Optional[dict] = None) -> dict:
        payload = {
            "hostname": socket.gethostname(),
            "ip": _primary_ip(),
            "port": int(port),
            "broadcast_port": int(broadcast_port),
            "repo_sha": repo_sha(),
            "hilserl_sha": hilserl_sha(),
            "pid": os.getpid(),
            "started_at": time.time(),
        }
        if extra:
            payload.update(extra)
        write_json_atomic(self.endpoint_path, payload)
        return payload

    def read_endpoint(self) -> dict:
        if not self.endpoint_path.exists():
            raise FileNotFoundError(
                f"no learner endpoint at {self.endpoint_path}; start the learner first"
            )
        return json.loads(self.endpoint_path.read_text())


def _primary_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostbyname(socket.gethostname())
