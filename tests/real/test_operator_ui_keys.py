"""``mulligan.real.operator_ui.keys``: aliases, the OpenCV key poll, the drain.

No terminal and no HighGUI: the terminal listener is a duck-typed fake and ``cv2.waitKey``
is a scripted stub, so these pin the exact semantics every entrypoint relies on: numpad
aliases, Enter normalization, lowercasing, "no display => never touch HighGUI", and the
drain that makes a buffered key unable to advance a gate.
"""

from __future__ import annotations

import os

import pytest

from mulligan.real.operator_ui import keys
from mulligan.real.operator_ui.keys import (
    OPERATOR_KEY_ALIASES,
    apply_operator_key_alias,
    drain_operator_keys,
    is_start_key,
    key_label,
    read_opencv_key,
    read_operator_key,
)


class FakeListener:
    """Stand-in for TerminalKeyboardListener: ``buffered`` keys are what a flush discards,
    ``fresh`` keys are what the operator types after the prompt."""

    def __init__(self, buffered: list[str] | None = None, fresh: list[str | None] | None = None):
        self.buffered = list(buffered or [])
        self.fresh = list(fresh or [])
        self.flushes = 0

    def read_key(self) -> str | None:
        if self.buffered:
            return apply_operator_key_alias(self.buffered.pop(0))
        if self.fresh:
            return apply_operator_key_alias(self.fresh.pop(0))
        return None

    def flush(self) -> None:
        self.buffered.clear()
        self.flushes += 1

    def close(self) -> None:
        pass


class ScriptedWaitKey:
    """``cv2.waitKey`` stub returning a scripted sequence of key codes, then -1 forever."""

    def __init__(self, codes: list[int]):
        self.codes = list(codes)
        self.calls = 0

    def __call__(self, delay_ms: int) -> int:
        self.calls += 1
        return self.codes.pop(0) if self.codes else -1


@pytest.fixture
def headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


@pytest.fixture
def with_display(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")


def test_numpad_aliases_cover_every_operator_letter_and_never_shadow_a_native_key():
    assert apply_operator_key_alias(None) is None
    assert apply_operator_key_alias("-") == "r"
    assert apply_operator_key_alias("/") == "q"
    assert apply_operator_key_alias("3") == "g"
    assert apply_operator_key_alias("1") == "1"  # native numpad digit passes through
    assert not set(OPERATOR_KEY_ALIASES) & set("1908\n\r")


def test_key_label_shows_the_numpad_alias_only_when_one_exists():
    assert key_label("r") == "'r'/numpad'-'"
    assert key_label("1") == "'1'"


def test_is_start_key_accepts_both_line_endings_only():
    assert is_start_key("\n") and is_start_key("\r")
    assert not is_start_key(" ") and not is_start_key(None) and not is_start_key("1")


def test_read_opencv_key_never_touches_highgui_without_a_display(headless, monkeypatch):
    def boom(_ms):
        raise AssertionError("cv2.waitKey must not be called headless")

    monkeypatch.setattr(keys.cv2, "waitKey", boom)
    assert read_opencv_key() is None


def test_read_opencv_key_normalizes_enter_and_applies_aliases(with_display, monkeypatch):
    stub = ScriptedWaitKey([13, 10, ord("-"), ord("R"), 0xFF00 | ord("q"), 200, -1])
    monkeypatch.setattr(keys.cv2, "waitKey", stub)
    assert read_opencv_key() == "\n"  # CR
    assert read_opencv_key() == "\n"  # LF
    assert read_opencv_key() == "r"  # numpad alias
    assert read_opencv_key() == "R"  # raw; read_operator_key lowercases
    assert read_opencv_key() == "q"  # HighGUI modifier bits masked off
    assert read_opencv_key() is None  # non-ASCII code
    assert read_opencv_key() is None  # nothing pressed


def test_read_operator_key_prefers_the_terminal_then_lowercases(with_display, monkeypatch):
    stub = ScriptedWaitKey([ord("Q")])
    monkeypatch.setattr(keys.cv2, "waitKey", stub)
    assert read_operator_key(FakeListener(fresh=["H"])) == "h"
    assert stub.calls == 0
    assert read_operator_key(FakeListener()) == "q"
    assert read_operator_key(FakeListener()) is None


def test_drain_flushes_the_terminal_and_empties_the_highgui_queue(with_display, monkeypatch):
    stub = ScriptedWaitKey([ord("1"), ord("1"), ord("9")])
    monkeypatch.setattr(keys.cv2, "waitKey", stub)
    listener = FakeListener(buffered=["1", "9"], fresh=["\n"])
    drain_operator_keys(listener)
    assert listener.flushes == 1 and listener.buffered == []
    assert stub.calls == 4  # three buffered codes, then the -1 that ends the drain
    assert read_operator_key(listener) == "\n"


def test_drain_is_bounded_when_a_key_is_held_down(with_display, monkeypatch):
    stub = ScriptedWaitKey([ord("r")] * 10_000)
    monkeypatch.setattr(keys.cv2, "waitKey", stub)
    drain_operator_keys(FakeListener())
    assert stub.calls == 100


def test_drain_skips_highgui_headless(headless, monkeypatch):
    def boom(_ms):
        raise AssertionError("cv2.waitKey must not be called headless")

    monkeypatch.setattr(keys.cv2, "waitKey", boom)
    listener = FakeListener(buffered=["q"])
    drain_operator_keys(listener)
    assert listener.flushes == 1 and listener.buffered == []


def test_terminal_listener_refuses_non_tty_stdin(monkeypatch):
    """A collector started without a terminal fails with a clear message, not a termios error."""
    import io

    from mulligan.real.operator_ui import keys

    r, w = os.pipe()
    try:
        monkeypatch.setattr(keys.sys, "stdin", io.TextIOWrapper(io.FileIO(r, closefd=False)))
        with pytest.raises(RuntimeError, match="not a TTY"):
            keys.TerminalKeyboardListener()
    finally:
        os.close(r)
        os.close(w)
