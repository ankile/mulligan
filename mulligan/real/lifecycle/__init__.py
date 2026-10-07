"""Shared per-round lifecycle primitives for real-world task lines.

Every real task line (marker_d2, square_d2, routing_d2) runs the same round
lifecycle: collection manifest -> blind collection -> split -> training ->
held-out eval ingest -> start design for the next round. The math and the
ingest/plot structure are task-invariant; only the
:class:`~mulligan.real.lifecycle.tasks.RealTaskSpec` (state keys, bounds,
placements, grid dims) differs. Per-round drivers import from this package.

Modules:
- ``tasks``: the task registry (``get_task_spec``, ``registered_task_specs``, ``find_task_spec_by_task_name``).
- ``geometry``: normalized periodic geometry, coverage metrics, samplers, FPS.
- ``stats``: paired-eval statistics (Wilson, exact McNemar, sign-flip permutation, bootstrap, t intervals).
- ``heldout_eval``: N-arm paired held-out eval ingest -> CSVs + plot.
- ``pinned_eval_snapshot``: held-out eval analysis bound to one Hub commit.
- ``episode_lengths``: per-round episode-length figure.
- ``label_history``: the append-only ``.label_history.jsonl`` label-provenance ledger.
"""

from mulligan.real.lifecycle.tasks import (
    RealTaskSpec,
    find_task_spec_by_task_name,
    get_task_spec,
    registered_task_specs,
)

__all__ = [
    "RealTaskSpec",
    "find_task_spec_by_task_name",
    "get_task_spec",
    "registered_task_specs",
]
