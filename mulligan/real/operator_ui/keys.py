"""Operator keyboard input: terminal (cbreak) plus the focused OpenCV window.

Every real-robot entrypoint reads operator keys through :func:`read_operator_key`, which
polls the terminal listener first and then the OpenCV HighGUI key queue. The single
``cv2.waitKey`` inside that poll is also what repaints every operator window (target card
and camera monitors), so callers must keep polling while windows are up.

A plain USB numpad has no letter keys, so its symbol keys (plus digits 2/3) are aliased to
the operator-letter actions used across collection / DAgger / eval. The numpad already
types 1/9/0/Enter natively (success / failure / timeout / start, NumLock ON). None of the
aliased source keys are read literally by any entrypoint, so the aliases only ever ADD an
operator action, never shadow one.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty

import cv2

from mulligan.real.operator_ui.display import has_display

OPERATOR_KEY_ALIASES = {
    "/": "q",  # quit session
    "*": "d",  # discard episode
    "-": "r",  # retry / reset / restart
    "+": "h",  # human intervention toggle
    ".": "c",  # counterfactual replay
    "2": "n",  # next target / episode
    "3": "g",  # sub-goal reached (mid-episode subtask mark; same letter as the outcome editor)
}

OPERATOR_LETTER_TO_NUMPAD = {letter: sym for sym, letter in OPERATOR_KEY_ALIASES.items()}

START_KEYS = frozenset({"\n", "\r"})


def apply_operator_key_alias(key: str | None) -> str | None:
    """Translate a numpad symbol key to its operator-letter action; pass through otherwise."""
    if key is None:
        return None
    return OPERATOR_KEY_ALIASES.get(key, key)


def key_label(letter: str) -> str:
    """Operator-key token for prompts, showing the letter plus its numpad alias.

    E.g. "r" -> "'r'/numpad'-'". Keys already native on the numpad (1/9/0) have no
    alias and render as just "'1'". Driven by OPERATOR_KEY_ALIASES so prompts can
    never drift from the actual key handling.
    """
    sym = OPERATOR_LETTER_TO_NUMPAD.get(letter)
    return f"'{letter}'/numpad'{sym}'" if sym else f"'{letter}'"


def is_start_key(key: str | None) -> bool:
    """Enter (either line ending) is the "object placed, go" key at placement gates."""
    return key in START_KEYS


class TerminalKeyboardListener:
    """Non-blocking single-character reader using termios cbreak mode.

    Works over SSH without X11. This is the single terminal chokepoint every entrypoint
    funnels through; numpad symbol keys are translated to operator-letter actions (see
    OPERATOR_KEY_ALIASES) so a plain USB numpad can drive every entrypoint.
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        if not os.isatty(self.fd):
            raise RuntimeError(
                "operator keys are read from the terminal, but stdin is not a TTY; run the "
                "collector in an interactive terminal (or tmux/screen), not with nohup or a pipe"
            )
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def read_key(self) -> str | None:
        """Return a single character if available, otherwise None."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return apply_operator_key_alias(sys.stdin.read(1))
        return None

    def flush(self) -> None:
        """Drain all buffered characters from stdin."""
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)

    def close(self) -> None:
        """Restore original terminal settings."""
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_opencv_key() -> str | None:
    """One non-blocking poll of the OpenCV key queue (also repaints the windows).

    Returns None without touching HighGUI when no display is configured. Enter (LF or CR)
    is normalized to ``"\\n"``; printable ASCII goes through the numpad aliases.
    """
    if not has_display():
        return None
    key = cv2.waitKey(1)
    if key < 0:
        return None
    key &= 0xFF
    if key in {10, 13}:
        return "\n"
    if 0 < key < 128:
        return apply_operator_key_alias(chr(key))
    return None


def read_operator_key(kbd_listener: TerminalKeyboardListener) -> str | None:
    """Terminal first, then the focused OpenCV window. Letters are lowercased.

    Lowercasing at the source means caps-lock can never silently disable an operator
    action ('R', 'Q', 'H' ...) in any consumer.
    """
    key = kbd_listener.read_key()
    if key is None:
        key = read_opencv_key()
    if key is None:
        return None
    return key.lower()


def drain_operator_keys(kbd_listener: TerminalKeyboardListener) -> None:
    """Discard buffered keystrokes so only a FRESH keypress is read as the next decision.

    The terminal runs in cbreak mode, so every key typed at any time -- mashed
    during a stuck/failed reset, a held-key auto-repeat, foot-pedal repeats, or a
    key pressed during the previous rollout -- sits in the stdin buffer. The
    rollout loop and the operator gates consume one buffered char per poll, so
    without draining, those stale keys are read as deliberate end-of-rollout
    decisions ('1'=success, '9'=failure, '0'=timeout, 'r'=restart) and auto-advance
    through rollouts/rounds in 0-1 steps. Call this at every transition to a NEW
    operator decision (every gate, and each rollout start) so a key pressed
    BEFORE the prompt can never advance -- only one pressed after it.
    """
    kbd_listener.flush()
    if has_display():
        # cv2's HighGUI key queue can buffer too; drain it (bounded so a key held
        # down indefinitely cannot wedge this loop).
        for _ in range(100):
            if cv2.waitKey(1) < 0:
                break
