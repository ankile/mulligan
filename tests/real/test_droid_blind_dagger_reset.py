from __future__ import annotations

import threading

import pytest

from mulligan.real.collect.blind_dagger import _DeferredResetCoordinator


def test_deferred_reset_start_does_not_call_reset_until_finish() -> None:
    calls: list[tuple[object, bool]] = []

    def reset_fn(env: object, *, randomize: bool) -> dict:
        calls.append((env, randomize))
        return {"env": env, "randomize": randomize}

    coordinator = _DeferredResetCoordinator("fake-env", randomize=True, reset_fn=reset_fn)

    coordinator.start("after target card")
    assert coordinator.has_pending
    assert calls == []

    assert coordinator.finish("before rollout") == {"env": "fake-env", "randomize": True}
    assert not coordinator.has_pending
    assert calls == [("fake-env", True)]


def test_deferred_reset_runs_reset_on_calling_thread() -> None:
    caller_thread = threading.get_ident()
    reset_thread: list[int] = []

    def reset_fn(env: object, *, randomize: bool) -> dict:
        del env, randomize
        reset_thread.append(threading.get_ident())
        return {"ok": True}

    coordinator = _DeferredResetCoordinator("fake-env", randomize=False, reset_fn=reset_fn)

    coordinator.start("main-thread reset")
    assert coordinator.finish("before rollout") == {"ok": True}
    assert reset_thread == [caller_thread]


def test_deferred_reset_rejects_second_start_until_first_is_finished() -> None:
    coordinator = _DeferredResetCoordinator(
        "fake-env",
        randomize=False,
        reset_fn=lambda env, *, randomize: {"ok": True},
    )

    coordinator.start("first")
    with pytest.raises(RuntimeError, match="still pending"):
        coordinator.start("second")

    assert coordinator.finish("first") == {"ok": True}


def test_deferred_reset_finish_propagates_reset_failure() -> None:
    def reset_fn(env: object, *, randomize: bool) -> dict:
        del env, randomize
        raise RuntimeError("reset exploded")

    coordinator = _DeferredResetCoordinator("fake-env", randomize=False, reset_fn=reset_fn)

    coordinator.start("failure")
    with pytest.raises(RuntimeError, match="reset exploded"):
        coordinator.finish("failure")
    assert not coordinator.has_pending


def test_deferred_reset_announces_the_reset_before_motion() -> None:
    events: list[str] = []

    def reset_fn(env: object, *, randomize: bool) -> dict:
        del env, randomize
        events.append("motion")
        return {"ok": True}

    coordinator = _DeferredResetCoordinator(
        "fake-env",
        randomize=False,
        reset_fn=reset_fn,
        before_reset=lambda reason: events.append(f"panel:{reason}"),
    )
    coordinator.start("end of episode")
    assert events == []  # queued only; the panel still shows the next target card
    coordinator.finish("before episode start")
    assert events == ["panel:end of episode", "motion"]
