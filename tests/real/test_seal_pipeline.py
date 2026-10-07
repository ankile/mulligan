"""Unit tests for the background episode seal pipeline (mulligan.real.collect.seal_pipeline).

These pin the scheduling + durability-ordering contract manifest_eval relies on:

  (a) submitting a chain while the pipeline is idle does NOT block on the newly
      submitted chain (the rollout overlaps the seal);
  (b) submitting while a chain is still in flight blocks until it finishes
      (headroom: at most one in-flight chain, bounding episode RAM);
  (c) the results record step runs ONLY after the parquet footer step succeeded
      (results.json can never list a record whose episode is not durable);
  (d) a round-boundary/end-of-session drain completes all outstanding chains;
  (e) an exception inside a chain surfaces loudly at the next sync point
      (submit/poll/drain) -- never silently dropped.
"""

from __future__ import annotations

import threading
import time

import pytest

from mulligan.real.collect.seal_pipeline import SealPipeline, run_seal_chain


def _make_pipeline() -> SealPipeline:
    return SealPipeline()


# ---------------------------------------------------------------------------
# (a) one in-flight chain does not block submission
# ---------------------------------------------------------------------------


def test_submit_returns_while_single_chain_still_running():
    pipeline = _make_pipeline()
    started = threading.Event()
    release = threading.Event()

    def chain() -> None:
        started.set()
        assert release.wait(timeout=10)

    t0 = time.monotonic()
    pipeline.submit_chain(chain, description="first seal", timeout=5)
    submit_elapsed = time.monotonic() - t0

    assert started.wait(timeout=5)
    assert submit_elapsed < 1.0, "submit_chain must not wait on the chain it submits"
    assert pipeline.in_flight
    # Non-blocking poll while the chain runs: nothing to reap, no error.
    assert pipeline.poll_completed(description="poll") is False

    release.set()
    pipeline.drain(description="drain", timeout=5)
    assert not pipeline.in_flight
    pipeline.shutdown()


# ---------------------------------------------------------------------------
# (b) a second submission while one chain is in flight forces a wait
# ---------------------------------------------------------------------------


def test_second_submit_blocks_until_previous_chain_finishes():
    pipeline = _make_pipeline()
    order: list[str] = []
    release_first = threading.Event()

    def first() -> None:
        assert release_first.wait(timeout=10)
        order.append("first")

    def second() -> None:
        order.append("second")

    pipeline.submit_chain(first, description="first seal", timeout=5)

    releaser = threading.Timer(0.3, release_first.set)
    releaser.start()
    t0 = time.monotonic()
    pipeline.submit_chain(second, description="headroom wait", timeout=5)
    waited = time.monotonic() - t0
    releaser.join()

    # The submit had to wait for the release (~0.3s), and the first chain fully
    # finished before the second was queued (serial handoff).
    assert waited >= 0.2, "second submit must block on the in-flight chain"
    assert order[0] == "first"

    pipeline.drain(description="drain", timeout=5)
    assert order == ["first", "second"]
    pipeline.shutdown()


def test_second_submit_times_out_loudly_if_previous_chain_hangs():
    pipeline = _make_pipeline()
    release = threading.Event()

    def hanging() -> None:
        assert release.wait(timeout=10)

    pipeline.submit_chain(hanging, description="first seal", timeout=5)
    with pytest.raises(RuntimeError, match="headroom wait timed out"):
        pipeline.submit_chain(lambda: None, description="headroom wait", timeout=0.05)
    # A timed-out chain is still running: it stays in flight so a later drain /
    # shutdown still waits for it instead of orphaning the dataset write.
    assert pipeline.in_flight
    release.set()
    pipeline.drain(description="drain", timeout=5)
    pipeline.shutdown()


# ---------------------------------------------------------------------------
# (c) results-record ordering: record only after footer success
# ---------------------------------------------------------------------------


def test_run_seal_chain_orders_save_footer_record():
    calls: list[str] = []
    run_seal_chain(
        save=lambda: calls.append("save"),
        footer=lambda: calls.append("footer"),
        record=lambda: calls.append("record"),
    )
    assert calls == ["save", "footer", "record"]


def test_run_seal_chain_footer_failure_blocks_record():
    calls: list[str] = []

    def footer() -> None:
        calls.append("footer")
        raise ValueError("parquet footer failed")

    with pytest.raises(ValueError, match="parquet footer failed"):
        run_seal_chain(
            save=lambda: calls.append("save"),
            footer=footer,
            record=lambda: calls.append("record"),
        )
    assert calls == ["save", "footer"], "record must not run after a failed footer"


def test_run_seal_chain_save_failure_blocks_footer_and_record():
    calls: list[str] = []

    def save() -> None:
        calls.append("save")
        raise OSError("encoder died")

    with pytest.raises(OSError, match="encoder died"):
        run_seal_chain(
            save=save,
            footer=lambda: calls.append("footer"),
            record=lambda: calls.append("record"),
        )
    assert calls == ["save"]


def test_footer_failure_in_background_chain_never_records():
    """End-to-end through the pipeline: a footer error surfaces at the sync point
    and the record step never ran (round re-collected, no phantom results row)."""
    pipeline = _make_pipeline()
    recorded: list[str] = []

    def chain() -> None:
        run_seal_chain(
            save=lambda: None,
            footer=_raise_footer,
            record=lambda: recorded.append("record"),
        )

    def _raise_footer() -> None:
        raise ValueError("footer boom")

    pipeline.submit_chain(chain, description="seal", timeout=5)
    with pytest.raises(RuntimeError, match="round drain failed") as excinfo:
        pipeline.drain(description="round drain", timeout=5)
    assert "footer boom" in repr(excinfo.value.__cause__)
    assert recorded == []
    pipeline.shutdown()


# ---------------------------------------------------------------------------
# (d) drain completes all outstanding chains
# ---------------------------------------------------------------------------


def test_drain_completes_outstanding_chain_before_returning():
    pipeline = _make_pipeline()
    done = threading.Event()

    def chain() -> None:
        time.sleep(0.2)
        done.set()

    pipeline.submit_chain(chain, description="seal", timeout=5)
    pipeline.drain(description="round-boundary drain", timeout=5)
    assert done.is_set(), "drain returned before the chain finished"
    assert not pipeline.in_flight
    # Idempotent: draining an empty pipeline is a no-op.
    pipeline.drain(description="noop drain", timeout=5)
    pipeline.shutdown()


# ---------------------------------------------------------------------------
# (e) chain exceptions surface loudly at the next sync point
# ---------------------------------------------------------------------------


def _failing_chain() -> None:
    raise RuntimeError("episode seal exploded")


def _wait_until_done(pipeline: SealPipeline) -> None:
    # Wait for the background chain to finish (failure included) without reaping
    # it, so the next sync point is what surfaces the error.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        future = pipeline._in_flight
        assert future is not None
        if future.done():
            return
        time.sleep(0.01)
    raise AssertionError("chain never finished")


def test_chain_error_surfaces_at_poll():
    pipeline = _make_pipeline()
    pipeline.submit_chain(_failing_chain, description="seal", timeout=5)
    _wait_until_done(pipeline)
    with pytest.raises(RuntimeError, match="poll sync failed"):
        pipeline.poll_completed(description="poll sync")
    # The dead chain was reaped; the pipeline is usable/idle again.
    assert not pipeline.in_flight
    pipeline.shutdown()


def test_chain_error_surfaces_at_next_submit_and_blocks_it():
    pipeline = _make_pipeline()
    pipeline.submit_chain(_failing_chain, description="seal", timeout=5)
    _wait_until_done(pipeline)
    second_ran = threading.Event()
    with pytest.raises(RuntimeError, match="next submit failed"):
        pipeline.submit_chain(second_ran.set, description="next submit", timeout=5)
    # Fail-loud contract: after a failed chain the next episode is NOT quietly
    # queued on top of a broken dataset writer.
    assert not second_ran.wait(timeout=0.2)
    assert not pipeline.in_flight
    pipeline.shutdown()


def test_chain_error_surfaces_at_drain():
    pipeline = _make_pipeline()
    pipeline.submit_chain(_failing_chain, description="seal", timeout=5)
    with pytest.raises(RuntimeError, match="final drain failed") as excinfo:
        pipeline.drain(description="final drain", timeout=5)
    assert "episode seal exploded" in repr(excinfo.value.__cause__)
    assert not pipeline.in_flight
    pipeline.shutdown()
