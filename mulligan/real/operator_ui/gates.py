"""Operator decision gates: "set up the scene, then press a key".

One implementation for every "place the object / reset the scene, then continue" prompt
across collection, DAgger, teleop, and eval. The gate always drains buffered keystrokes
first (so a key mashed during the previous reset can never start an episode), keeps
repainting the camera monitor while it waits, and offers the same optional escape hatches
everywhere: ``r`` re-homes the robot, ``k`` skips the start, ``q`` quits. The key legend in
the prompt is generated from the enabled options and :func:`key_label`, so the prompt and
the handling cannot drift.

:func:`operator_choice` is the sibling for the collector's between-episode decisions
("counterfactual replay, next start, sync, or quit"): the same drain-first, keep-painting
poll, with the legend generated from the offered choices.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum

from mulligan.real.operator_ui.keys import (
    TerminalKeyboardListener,
    drain_operator_keys,
    is_start_key,
    key_label,
    read_operator_key,
)


class GateOutcome(str, Enum):
    START = "start"
    SKIP = "skip"
    QUIT = "quit"


def gate_legend(
    *,
    any_key_starts: bool,
    can_reset: bool,
    can_skip: bool,
    can_quit: bool,
) -> str:
    """The parenthesized key legend appended to every gate prompt."""
    parts = []
    if can_reset:
        parts.append(f"{key_label('r')} resets robot")
    if can_skip:
        parts.append(f"{key_label('k')} skips this start")
    if can_quit:
        parts.append(f"{key_label('q')} quits")
    start = "any key" if any_key_starts else "Enter"
    legend = f"press {start} to start"
    if parts:
        legend += " (" + ", ".join(parts) + ")"
    return legend


def operator_gate(
    kbd_listener: TerminalKeyboardListener,
    *,
    prompt: str,
    on_reset: Callable[[], object] | None = None,
    can_skip: bool = False,
    can_quit: bool = True,
    any_key_starts: bool = False,
    render: Callable[[], None] | None = None,
    poll_interval_s: float = 0.05,
    render_interval_s: float = 0.5,
) -> GateOutcome:
    """Block until the operator starts, skips, or quits.

    ``prompt`` is the situation ("Place the marker at the shown target"); the key legend is
    appended automatically. ``on_reset`` (when given) enables ``r``: it runs the robot reset
    and the gate re-prompts and drains again, since keys mashed during a multi-second reset
    must not be read as the "go" that follows. ``render`` repaints the camera monitor: at
    entry, after each reset, and then every ``render_interval_s`` while waiting. Pass a
    callback that fetches a FRESH observation so the operator watches the object appear in
    the policy's crop as they place it; a few hertz is plenty for that.
    """
    legend = gate_legend(
        any_key_starts=any_key_starts,
        can_reset=on_reset is not None,
        can_skip=can_skip,
        can_quit=can_quit,
    )
    print(f"{prompt}, then {legend}.", flush=True)
    last_render = None
    if render is not None:
        render()
        last_render = time.monotonic()
    drain_operator_keys(kbd_listener)
    while True:
        if render is not None and time.monotonic() - last_render >= render_interval_s:
            render()
            last_render = time.monotonic()
        key = read_operator_key(kbd_listener)
        if key is None:
            time.sleep(poll_interval_s)
            continue
        if key == "q" and can_quit:
            return GateOutcome.QUIT
        if key == "k" and can_skip:
            return GateOutcome.SKIP
        if key == "r" and on_reset is not None:
            on_reset()
            print(
                f"Robot reset. Set up the same target again, then {legend}.",
                flush=True,
            )
            if render is not None:
                render()
                last_render = time.monotonic()
            drain_operator_keys(kbd_listener)
            continue
        if any_key_starts or is_start_key(key):
            return GateOutcome.START


def choice_legend(choices: dict[str, str], *, default: str | None) -> str:
    """The key legend for :func:`operator_choice`: "press 'c'/numpad'.' to ..., or 'q' to quit"."""
    if not choices:
        raise ValueError("operator_choice needs at least one choice")
    for key in choices:
        if len(key) != 1 or not key.isalpha() or key != key.lower():
            raise ValueError(f"Choice keys must be single lowercase letters, got {key!r}")
    if default is not None and default not in choices:
        raise ValueError(f"default {default!r} is not one of the choices {sorted(choices)}")
    parts = []
    for key, action in choices.items():
        token = key_label(key)
        if key == default:
            token += " (or any other key)"
        parts.append(f"{token} to {action}")
    if len(parts) == 1:
        return f"press {parts[0]}"
    return "press " + ", ".join(parts[:-1]) + f", or {parts[-1]}"


def operator_choice(
    kbd_listener: TerminalKeyboardListener,
    *,
    prompt: str,
    choices: dict[str, str],
    default: str | None,
    render: Callable[[], None] | None = None,
    poll_interval_s: float = 0.05,
    render_interval_s: float = 0.5,
) -> str:
    """Block until the operator picks one of ``choices``; returns the chosen key.

    ``choices`` maps an operator letter to what it does ("collect a counterfactual replay").
    ``default`` (a key in ``choices``, or ``None``) is what any OTHER key selects; with
    ``None`` unlisted keys are ignored. Buffered keys are drained first, exactly like
    :func:`operator_gate`, and ``render`` keeps the operator windows fresh while waiting.
    """
    legend = choice_legend(choices, default=default)
    print(f"{prompt}: {legend}.", flush=True)
    last_render = None
    if render is not None:
        render()
        last_render = time.monotonic()
    drain_operator_keys(kbd_listener)
    while True:
        if render is not None and time.monotonic() - last_render >= render_interval_s:
            render()
            last_render = time.monotonic()
        key = read_operator_key(kbd_listener)
        if key is None:
            time.sleep(poll_interval_s)
            continue
        if key in choices:
            return key
        if default is not None:
            return default
