"""UTD governor: keep the async learner on the RLPD arm's update schedule.

In ``train_robo.py`` every env step ``i >= start_training`` triggers exactly one
``agent.update(batch, utd_ratio)`` call (= ``utd_ratio`` critic steps + one
actor/temperature step). The split learner therefore owes

    target_calls = max(0, env_steps_received - start_training)

update calls and must not run ahead of that (it sleeps instead). The actor side
uses the same arithmetic (``owed``) to decide whether the learner has fallen too
far behind and waits at an episode boundary until it catches up.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Governor:
    start_training: int
    calls_per_env_step: int = 1

    def target_calls(self, env_steps_received: int) -> int:
        if env_steps_received < 0:
            raise ValueError("env_steps_received must be >= 0")
        return max(0, env_steps_received - self.start_training) * self.calls_per_env_step

    def owed(self, env_steps_received: int, calls_done: int) -> int:
        """Update calls the learner still owes (never negative)."""
        return max(0, self.target_calls(env_steps_received) - calls_done)
