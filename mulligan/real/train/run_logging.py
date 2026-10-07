"""Optional W&B logging for the real trainers.

Training never needs W&B: without ``--use-wandb`` every call here is a no-op and ``wandb`` is
never imported. With ``--use-wandb`` the run logs metrics, images and checkpoint artifacts to the
project given on the command line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaFile:
    """An image or video written to disk, logged as W&B media by :meth:`RunLogger.log`."""

    path: str
    kind: str  # "image" | "video"
    fps: int = 10


class RunLogger:
    """Thin wrapper around ``wandb`` that is inert when disabled."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._wandb = None
        self._run = None

    def init(
        self,
        *,
        project: str,
        name: str,
        notes: str | None,
        config: dict,
        resume_id: str | None = None,
        resume: str = "allow",
    ) -> None:
        if not self.enabled:
            return
        import wandb

        # Honor an explicit WANDB_MODE (e.g. offline for a local smoke run).
        env_mode = os.environ.get("WANDB_MODE", "").strip().lower()
        mode = env_mode if env_mode in ("online", "offline", "disabled") else "online"
        kwargs = dict(project=project, name=name, notes=notes, config=config, mode=mode)
        if resume_id:
            kwargs["id"] = resume_id
            kwargs["resume"] = resume
        self._wandb = wandb
        self._run = wandb.init(**kwargs)

    @property
    def run_id(self) -> str | None:
        return None if self._run is None else self._run.id

    @property
    def run_name(self) -> str | None:
        return None if self._run is None else self._run.name

    def log(self, metrics: dict, *, step: int) -> None:
        if self._run is None:
            return
        self._wandb.log({k: self._to_wandb(v) for k, v in metrics.items()}, step=step)

    def _to_wandb(self, value):
        if not isinstance(value, MediaFile):
            return value
        if value.kind == "image":
            return self._wandb.Image(value.path)
        if value.kind == "video":
            return self._wandb.Video(value.path, fps=value.fps, format="mp4")
        raise ValueError(f"unknown media kind {value.kind!r} for {value.path}")

    def image(self, pil_image):
        """Wrap an image for :meth:`log`; only valid while the run is active."""
        if self._run is None:
            raise RuntimeError("RunLogger.image() requires an active W&B run")
        return self._wandb.Image(pil_image)

    def log_artifact_dir(
        self,
        directory: Path,
        *,
        name: str,
        artifact_type: str,
        metadata: dict,
        description: str | None = None,
    ) -> None:
        if self._run is None:
            return
        artifact = self._wandb.Artifact(
            name=name, type=artifact_type, description=description, metadata=metadata
        )
        artifact.add_dir(str(directory))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self._run is not None:
            self._wandb.finish()
            self._run = None
