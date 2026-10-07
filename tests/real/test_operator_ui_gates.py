"""``operator_gate``: the one "set up the scene, then press a key" decision for every
real entrypoint. Table-driven over the option flags; headless (no HighGUI), fake listener."""

from __future__ import annotations

import pytest

from mulligan.real.operator_ui.gates import (
    GateOutcome,
    choice_legend,
    gate_legend,
    operator_choice,
    operator_gate,
)
from tests.real.test_operator_ui_keys import FakeListener


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


def run(listener, **kw) -> GateOutcome:
    return operator_gate(listener, prompt="Place the marker", poll_interval_s=0, **kw)


@pytest.mark.parametrize(
    ("fresh", "kw", "expected"),
    [
        (["x", "\n"], {}, GateOutcome.START),  # only Enter starts by default
        (["\r"], {}, GateOutcome.START),
        (["x"], {"any_key_starts": True}, GateOutcome.START),
        (["q"], {}, GateOutcome.QUIT),
        (["/"], {}, GateOutcome.QUIT),  # numpad alias
        (["Q"], {}, GateOutcome.QUIT),  # caps lock
        (["q", "\n"], {"can_quit": False}, GateOutcome.START),  # q inert when disabled
        (["k"], {"can_skip": True}, GateOutcome.SKIP),
        (["k", "\n"], {}, GateOutcome.START),  # k inert unless the caller allows skipping
        (["r", "\n"], {}, GateOutcome.START),  # r inert without an on_reset
    ],
)
def test_outcomes(fresh, kw, expected):
    assert run(FakeListener(fresh=fresh), **kw) is expected


def test_buffered_keys_are_drained_before_the_prompt_can_be_answered():
    # An Enter (or a quit!) mashed during the previous reset sits in the buffer; it must
    # never satisfy the gate. Only the fresh key typed after the prompt counts.
    listener = FakeListener(buffered=["\n", "q"], fresh=["k"])
    assert run(listener, can_skip=True) is GateOutcome.SKIP
    assert listener.flushes == 1


def test_reset_runs_the_callback_re_prompts_and_drains_again(capsys):
    resets = []
    listener = FakeListener(fresh=["r", "\n"])

    def on_reset():
        resets.append(1)
        listener.buffered.append("\n")  # a key mashed while the arm was moving

    # The mashed Enter is drained after the reset, so the gate needs the real one.
    listener.fresh = ["r", "\n"]
    assert run(listener, on_reset=on_reset) is GateOutcome.START
    assert resets == [1]
    assert listener.flushes == 2
    out = capsys.readouterr().out
    assert "Robot reset. Set up the same target again" in out


def test_render_paints_at_entry_after_reset_and_at_the_throttled_rate():
    frames = []
    listener = FakeListener(fresh=[None, None, "\n"])
    run(listener, render=lambda: frames.append(1), render_interval_s=60.0)
    assert frames == [1]  # entry only within the interval
    frames.clear()
    run(
        FakeListener(fresh=["r", "\n"]),
        on_reset=lambda: None,
        render=lambda: frames.append(1),
        render_interval_s=60.0,
    )
    assert frames == [1, 1]  # entry + after the reset
    frames.clear()
    run(
        FakeListener(fresh=[None, None, "\n"]), render=lambda: frames.append(1), render_interval_s=0
    )
    assert len(frames) >= 3  # live: every poll when the interval is zero


def test_prompt_carries_a_legend_that_matches_the_enabled_options(capsys):
    run(FakeListener(fresh=["\n"]), on_reset=lambda: None, can_skip=True)
    out = capsys.readouterr().out
    assert "Place the marker, then press Enter to start" in out
    assert "'r'/numpad'-' resets robot" in out
    assert "'k' skips this start" in out
    assert "'q'/numpad'/' quits" in out

    run(FakeListener(fresh=["x"]), any_key_starts=True, can_quit=False)
    out = capsys.readouterr().out
    assert "press any key to start." in out
    assert "quits" not in out and "resets" not in out and "skips" not in out


def test_gate_legend_wording():
    assert (
        gate_legend(any_key_starts=False, can_reset=False, can_skip=False, can_quit=False)
        == "press Enter to start"
    )


CHOICES = {"c": "collect a counterfactual replay", "n": "move on", "q": "quit"}


def choose(listener, **kw) -> str:
    return operator_choice(
        listener, prompt="Episode saved", choices=CHOICES, poll_interval_s=0, **kw
    )


@pytest.mark.parametrize(
    ("buffered", "fresh", "default", "expected"),
    [
        (["c"], ["n"], "n", "n"),  # the buffered key is drained, never read as the decision
        ([], ["c"], "n", "c"),
        ([], ["."], "n", "c"),  # numpad alias for c
        ([], ["Q"], "n", "q"),
        ([], ["x"], "n", "n"),  # any other key selects the default
        ([], ["x", "z", "c"], None, "c"),  # ... unless there is no default
    ],
)
def test_choice_outcomes(buffered, fresh, default, expected):
    listener = FakeListener(buffered=buffered, fresh=fresh)
    assert choose(listener, default=default) == expected
    assert listener.flushes == 1


def test_choice_legend_names_every_key_and_the_default(capsys):
    legend = choice_legend(CHOICES, default="n")
    assert legend == (
        "press 'c'/numpad'.' to collect a counterfactual replay, "
        "'n'/numpad'2' (or any other key) to move on, or 'q'/numpad'/' to quit"
    )
    assert choice_legend({"q": "quit"}, default=None) == "press 'q'/numpad'/' to quit"
    choose(FakeListener(fresh=["q"]), default="n")
    assert "Episode saved: press 'c'/numpad'.'" in capsys.readouterr().out
    with pytest.raises(ValueError, match="not one of the choices"):
        choice_legend(CHOICES, default="z")
    with pytest.raises(ValueError, match="single lowercase letters"):
        choice_legend({"\n": "start"}, default=None)
    with pytest.raises(ValueError, match="at least one choice"):
        choice_legend({}, default=None)


def test_choice_repaints_while_waiting():
    painted = []
    listener = FakeListener(fresh=[None, None, "n"])
    operator_choice(
        listener,
        prompt="Next",
        choices=CHOICES,
        default="n",
        render=lambda: painted.append(1),
        poll_interval_s=0,
        render_interval_s=0,
    )
    assert len(painted) >= 3  # entry + one per poll
