"""Serial background "seal pipeline" for real-robot eval episode saves.

A *seal chain* is the full per-episode durability sequence that would otherwise run
on the operator's critical path between rollouts:

    save_episode (streaming-encoder drain)
    -> parquet footer (checkpoint_dataset finalize+reopen)
    -> rollout-record bookkeeping
    -> results.json rewrite

:class:`SealPipeline` schedules those chains on ONE single-thread executor so
the dataset-object handoff between consecutive chains (``checkpoint_dataset``
returns a NEW dataset object) is sequentially consistent -- chain N+1 only
starts after chain N returned -- while the main thread overlaps the next
rollout with the previous episode's seal.

Memory contract: each buffered episode holds all of its camera frames in RAM
(up to ~3 GB at 800 frames x 4 cams x 480x640x3), so at most ONE chain may be
in flight. :meth:`submit_chain` blocks on the previous chain first (the
headroom wait), bounding RAM to one completed-but-unsubmitted episode plus one
being sealed.

Failure contract (never fail silently): an exception inside a chain
is captured by its future and re-raised -- wrapped with the sync point's
description by :func:`mulligan.real.collect.save_utils.wait_for_background_save` -- at the
next sync point: the next :meth:`submit_chain`, :meth:`poll_completed`, or
:meth:`drain`. Callers place a blocking :meth:`drain` at every point that
requires the queue to be empty (round-boundary hub checkpoint, end-of-session
shutdown). No chain error can be silently dropped as long as the caller ends
with a drain, which ``manifest_eval`` does in its ``finally`` block.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable

from mulligan.real.collect.save_utils import wait_for_background_save


def run_seal_chain(
    *,
    save: Callable[[], None],
    footer: Callable[[], None],
    record: Callable[[], None],
) -> None:
    """Run one episode's seal chain in durability order.

    ``record`` (rollout-record bookkeeping + results.json rewrite) runs ONLY
    after ``footer`` succeeded. This is the durability-ordering contract: a
    crash or error can leave a durably-footered episode WITHOUT a results.json
    record (its round shows incomplete and is re-collected; the orphan episode
    is ignored by record-keyed analysis), but never a results.json record whose
    episode is not durably footered on disk.
    """
    save()
    footer()
    record()


class SealPipeline:
    """At-most-one-in-flight serial scheduler for background seal chains."""

    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="seal-chain"
        )
        self._in_flight: concurrent.futures.Future | None = None

    @property
    def in_flight(self) -> bool:
        """True while a submitted chain has not been reaped by a sync point."""
        return self._in_flight is not None

    def submit_chain(
        self,
        chain: Callable[[], None],
        *,
        description: str,
        timeout: float | None,
    ) -> None:
        """Submit the next seal chain, blocking on the previous one first.

        This is the headroom wait: exactly one chain may be in flight, so if
        the previous chain is still running this blocks (with a loud message)
        until it finishes, re-raising any error it hit. Submitting while idle
        returns immediately -- the chain runs in the background.
        """
        if self._in_flight is not None and not self._in_flight.done():
            print(
                "  Previous episode's background seal is still running; waiting for it "
                f"before queueing the next ({description})..."
            )
        self.drain(description=description, timeout=timeout)
        self._in_flight = self._executor.submit(chain)

    def poll_completed(self, *, description: str) -> bool:
        """Reap an already-finished chain without blocking.

        Returns True if a finished chain was reaped. Re-raises (wrapped with
        *description*) any error the chain hit, so failures surface at the
        earliest sync point instead of episodes silently going unsealed.
        """
        if self._in_flight is None or not self._in_flight.done():
            return False
        self.drain(description=description, timeout=0)
        return True

    def drain(self, *, description: str, timeout: float | None) -> None:
        """Block until the in-flight chain (if any) finishes.

        Re-raises chain errors wrapped with *description*. After a chain
        ERROR the pipeline is cleared (that chain is dead). After a TIMEOUT
        the chain is still running and stays in flight, so a later drain or
        ``shutdown(wait=True)`` still waits for it rather than orphaning an
        in-progress dataset write.
        """
        future = self._in_flight
        if future is None:
            return
        try:
            wait_for_background_save(future, description=description, timeout=timeout)
        except RuntimeError:
            if future.done():
                self._in_flight = None
            raise
        self._in_flight = None

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut the executor down; with ``wait=True`` blocks on a running chain."""
        self._executor.shutdown(wait=wait)
