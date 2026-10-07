"""Machine-readable progress lines for long-running jobs (stdout only).

Jobs print one line per update:

    MULLIGAN_PROGRESS {"schema":"mulligan.progress.v1","ts":"2026-05-23T03:00:00Z",...}

Required fields: ``schema``, ``ts`` (UTC ISO-8601) and ``job_type`` (e.g. ``training``,
``grid-eval``). Optional fields: ``job_id``, ``run_id``, ``phase``, ``status``
(``running``/``succeeded``/``failed``), ``progress`` (0..1), ``current``/``total``,
``eta_s``, ``exit_code``, ``message``, ``metrics`` and ``outputs`` (paths the job
writes; a trailing ``/`` marks a directory). Empty fields are omitted.

The prefix lets a log reader pick structured updates out of ordinary job output.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

PROGRESS_PREFIX = "MULLIGAN_PROGRESS "
PROGRESS_SCHEMA = "mulligan.progress.v1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ProgressEtaEstimator:
    """ETA from the average rate since ``start_current`` for monotonic current/total progress."""

    start_current: float = 0.0
    start_time: float = field(default_factory=time.monotonic)

    def eta_s(
        self,
        current: int | float | None,
        total: int | float | None,
        *,
        now: float | None = None,
    ) -> float | None:
        if current is None or total is None:
            return None
        current_f = float(current)
        total_f = float(total)
        if total_f <= 0 or current_f >= total_f:
            return 0.0
        completed = max(0.0, current_f - self.start_current)
        if completed <= 0:
            return None
        elapsed = max(0.0, (time.monotonic() if now is None else now) - self.start_time)
        if elapsed <= 0:
            return None
        return (total_f - current_f) / max(completed / elapsed, 1e-9)


@dataclass
class JobProgressEvent:
    job_type: str
    ts: str = field(default_factory=utc_now)
    schema: str = PROGRESS_SCHEMA
    job_id: str = ""
    run_id: str = ""
    phase: str = ""
    status: str = ""
    progress: float | None = None
    current: int | float | None = None
    total: int | float | None = None
    eta_s: float | None = None
    exit_code: int | None = None
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if not (value is None or value == "" or value == {} or value == [])
        }

    def to_line(self) -> str:
        return PROGRESS_PREFIX + json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def emit_progress(
    *,
    job_type: str,
    job_id: str = "",
    run_id: str = "",
    phase: str = "",
    status: str = "",
    progress: float | None = None,
    current: int | float | None = None,
    total: int | float | None = None,
    eta_s: float | None = None,
    exit_code: int | None = None,
    message: str = "",
    metrics: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
    file: TextIO | None = None,
) -> JobProgressEvent:
    """Print one progress line to ``file`` (default: current ``sys.stdout``) and return it."""
    event = JobProgressEvent(
        job_type=job_type,
        job_id=job_id,
        run_id=run_id,
        phase=phase,
        status=status,
        progress=progress,
        current=current,
        total=total,
        eta_s=eta_s,
        exit_code=exit_code,
        message=message,
        metrics=metrics or {},
        outputs=list(outputs) if outputs else [],
    )
    print(event.to_line(), file=sys.stdout if file is None else file, flush=True)
    return event


def emit_completion(
    *,
    job_type: str,
    success: bool,
    job_id: str = "",
    run_id: str = "",
    current: int | float | None = None,
    total: int | float | None = None,
    exit_code: int | None = None,
    message: str = "",
    metrics: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
    file: TextIO | None = None,
) -> JobProgressEvent:
    """Print the terminal ``complete`` / ``failed`` progress line."""
    if exit_code is None:
        exit_code = 0 if success else 1
    return emit_progress(
        job_type=job_type,
        job_id=job_id,
        run_id=run_id,
        phase="complete" if success else "failed",
        status="succeeded" if success else "failed",
        progress=1.0 if success else None,
        current=current,
        total=total,
        eta_s=0.0 if success else None,
        exit_code=exit_code,
        message=message,
        metrics=metrics,
        outputs=outputs,
        file=file,
    )


def parse_progress_line(line: str) -> JobProgressEvent | None:
    """Parse a line written by :func:`emit_progress`; None for any other line."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        data = json.loads(line[len(PROGRESS_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if data.get("schema") != PROGRESS_SCHEMA:
        return None
    return JobProgressEvent(
        schema=data["schema"],
        ts=data.get("ts", ""),
        job_type=data.get("job_type", ""),
        job_id=data.get("job_id", ""),
        run_id=data.get("run_id", ""),
        phase=data.get("phase", ""),
        status=data.get("status", ""),
        progress=data.get("progress"),
        current=data.get("current"),
        total=data.get("total"),
        eta_s=data.get("eta_s"),
        exit_code=data.get("exit_code"),
        message=data.get("message", ""),
        metrics=data.get("metrics") or {},
        outputs=list(data.get("outputs") or []),
    )
