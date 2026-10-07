"""Session timing and anonymous scene context, independent of robot and GUI libraries."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from mulligan.real.collect.initial_states import InitialStateTarget


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "Estimating..."
    minutes, secs = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@dataclass(frozen=True)
class EvalScene:
    """Only information the blinded operator may see. Slots are one-based."""

    round_num: int
    total_rounds: int | None
    label: str
    slot: int
    slots: int
    target: InitialStateTarget | None = None

    @property
    def description(self) -> str:
        round_text = str(self.round_num)
        if self.total_rounds is not None:
            round_text += f" / {self.total_rounds}"
        return (
            f"Round {round_text}   |   Policy {self.label}   |   Rollout {self.slot} / {self.slots}"
        )


@dataclass(frozen=True)
class CollectionScene:
    """The collector's context line: which start, fresh or replay, which episode.

    The line never names the arm, so the operator stays blind to it.
    """

    start_label: str
    mode: str
    episode_num: int
    target: InitialStateTarget | None = None

    @property
    def description(self) -> str:
        return (
            f"{self.start_label[:1].upper()}{self.start_label[1:]}   |   {self.mode}"
            f"   |   Episode {self.episode_num}"
        )


@dataclass(frozen=True)
class CollectionStatus:
    """Quota-driven progress for a collection session, refreshed by the collector.

    ``metrics`` are up to three ``(LABEL, value)`` header cells; the panel inserts the
    live ``ELAPSED`` cell third so collection and eval headers read the same way.
    ``fraction`` fills the progress bar (``None`` for an open-ended session) and
    ``summary`` is appended to the context line (pace, credits this session, ...).
    """

    metrics: tuple[tuple[str, str], ...]
    fraction: float | None = None
    summary: str = ""

    def __post_init__(self) -> None:
        if len(self.metrics) > 3:
            raise ValueError(f"CollectionStatus takes at most 3 metrics, got {len(self.metrics)}")
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"CollectionStatus.fraction must lie in [0, 1], got {self.fraction}")


@dataclass
class SessionProgress:
    """Finished rollouts, not durable saves. Retries contribute time, never progress.

    The clock starts at the first placement gate, after model/hardware initialization.
    ETA uses this launch's whole-session pace, including placement, reset, saving waits
    and discarded retries. Restored completions reduce the remaining work but do not
    pretend to be timed observations from this launch.
    """

    total: int | None = None
    completed: int = 0
    clock: Callable[[], float] = time.monotonic
    started: float | None = None
    session_completed: int = 0
    phase: str = "Preparing session"
    detail: str = "Loading policies and preparing the robot"
    controls: tuple[str, ...] = ()
    scene: EvalScene | CollectionScene | None = None
    status: CollectionStatus | None = None
    rollout_started: float | None = None
    step: int = 0
    max_steps: int = 0
    marks: int = 0
    max_marks: int = 0

    def __post_init__(self) -> None:
        if self.completed < 0 or (self.total is not None and self.completed > self.total):
            raise ValueError(f"Invalid rollout progress: {self.completed}/{self.total}")

    def start(self) -> None:
        if self.started is None:
            self.started = self.clock()

    @property
    def elapsed(self) -> float:
        return 0.0 if self.started is None else self.clock() - self.started

    @property
    def remaining(self) -> int | None:
        return None if self.total is None else self.total - self.completed

    @property
    def eta(self) -> float | None:
        if self.remaining == 0:
            return 0.0
        if self.remaining is None or not self.session_completed:
            return None
        return self.remaining * self.elapsed / self.session_completed

    def begin_rollout(self, max_steps: int, max_marks: int) -> None:
        self.start()
        self.rollout_started = self.clock()
        self.step = self.marks = 0
        self.max_steps, self.max_marks = max_steps, max_marks

    def end_rollout(self, outcome: str) -> None:
        """Stop the episode clock without counting: collection progress is quota-driven."""
        if not outcome:
            raise ValueError("A rollout outcome is required")
        self.rollout_started = None

    def finish_rollout(self, outcome: str) -> None:
        if outcome not in {"success", "failure", "timeout", "restart", "quit"}:
            raise ValueError(f"Unknown rollout outcome: {outcome}")
        self.end_rollout(outcome)
        if outcome in {"restart", "quit"}:
            return
        if self.total is not None and self.completed == self.total:
            raise ValueError("More rollouts finished than the session planned")
        self.completed += 1
        self.session_completed += 1

    @property
    def context(self) -> str:
        """The panel's second line: the scene, plus the collection summary when set."""
        parts = [self.scene.description if self.scene is not None else "Robot operator"]
        if self.status is not None and self.status.summary:
            parts.append(self.status.summary)
        return "   |   ".join(parts)

    def header_metrics(self) -> tuple[tuple[str, str], ...]:
        """The header cells: eval rollout accounting, or the collector's quota status."""
        elapsed = ("ELAPSED", duration(self.elapsed))
        if self.status is not None:
            cells = list(self.status.metrics)
            cells.insert(min(2, len(cells)), elapsed)
            return tuple(cells)
        finished = str(self.completed)
        if self.total is not None:
            finished += f" / {self.total}"
        return (
            ("ROLLOUTS FINISHED", finished),
            ("REMAINING", str(self.remaining) if self.remaining is not None else "Open session"),
            elapsed,
            ("EST. TIME LEFT", duration(self.eta) if self.total is not None else "Open session"),
        )

    @property
    def bar_fraction(self) -> float | None:
        if self.status is not None:
            return self.status.fraction
        if not self.total:
            return None
        return self.completed / self.total
