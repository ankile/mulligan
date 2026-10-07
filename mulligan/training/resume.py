"""Local resume support for preemptible training jobs.

A run is identified by its name. Re-running the same command with the same
``checkpoint_dir`` and run name picks up the rolling resume checkpoint under
``<checkpoint_dir>/_resume/<sha1(run_name)>/`` and continues from the last saved step.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mulligan.utils.atomic import atomic_write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    # Flush before the rename so a preemption cannot leave a truncated checkpoint.
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


@dataclass
class ResumeMetadata:
    run_name: str
    run_dir: str
    completed: bool = False
    last_checkpoint_step: int = 0
    created_at: str = ""
    updated_at: str = ""


class AutoResumeManager:
    """Rolling local resume state for one named run.

    Metadata is read from disk once at startup and kept in memory afterwards, so a
    network filesystem with weak read-after-write coherence cannot make a running job
    lose track of its own state.
    """

    def __init__(self, base_checkpoint_dir: Path, *, run_name: str, enabled: bool):
        self.enabled = enabled
        self.run_name = run_name
        identity_hash = hashlib.sha1(run_name.encode("utf-8")).hexdigest()
        self.resume_dir = Path(base_checkpoint_dir) / "_resume" / identity_hash
        self.meta_path = self.resume_dir / "resume_meta.json"
        self.state_path = self.resume_dir / "resume_state.pt"
        self.lock_path = self.resume_dir / "lock"
        self._lock_file = None
        self._meta: ResumeMetadata | None = None

    def acquire_lock(self) -> None:
        if not self.enabled:
            return
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.lock_path, "a+")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Resume lock already held for run '{self.run_name}'. "
                "Refusing to start a second concurrent instance."
            ) from exc

    def load_metadata(self) -> ResumeMetadata | None:
        """Read metadata from disk (None for a fresh run) and cache it."""
        if not self.enabled or not self.meta_path.exists():
            return None
        with open(self.meta_path) as f:
            payload = json.load(f)
        self._meta = ResumeMetadata(**payload)
        return self._meta

    def _get_meta_or_raise(self, where: str) -> ResumeMetadata:
        if self._meta is not None:
            return self._meta
        loaded = self.load_metadata()
        if loaded is None:
            raise RuntimeError(f"Cannot {where} before initializing resume metadata.")
        return loaded

    def _write_meta(self, meta: ResumeMetadata) -> None:
        atomic_write_json(self.meta_path, asdict(meta))
        self._meta = meta

    def initialize_fresh_run(self, run_dir: Path) -> None:
        if not self.enabled:
            return
        now = _utc_now()
        self._write_meta(
            ResumeMetadata(
                run_name=self.run_name, run_dir=str(run_dir), created_at=now, updated_at=now
            )
        )

    def save_training_state(
        self,
        *,
        policy_state_dict: dict[str, Any],
        optimizer_state_dicts: dict[str, Any],
        step: int,
        best_success_rate: float,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "policy_state_dict": policy_state_dict,
            "optimizer_state_dicts": optimizer_state_dicts,
            "last_checkpoint_step": step,
            "best_success_rate": best_success_rate,
            "saved_at": _utc_now(),
        }
        # Optional caller provenance, checked on resume by the caller.
        if extra_state:
            payload.update(extra_state)
        _atomic_torch_save(self.state_path, payload)

        meta = self._get_meta_or_raise("save training state")
        meta.last_checkpoint_step = step
        meta.updated_at = _utc_now()
        self._write_meta(meta)

    def load_training_state(self) -> dict[str, Any] | None:
        if not self.enabled or not self.state_path.exists():
            return None
        return torch.load(self.state_path, map_location="cpu", weights_only=False)

    def mark_completed(self) -> None:
        if not self.enabled:
            return
        meta = self._get_meta_or_raise("mark the run completed")
        meta.completed = True
        meta.updated_at = _utc_now()
        self._write_meta(meta)


def should_save_resume_checkpoint(*, step: int, resume_checkpoint_freq: int) -> bool:
    """Whether the rolling resume checkpoint is committed after ``step``."""
    return step > 0 and resume_checkpoint_freq > 0 and step % resume_checkpoint_freq == 0
