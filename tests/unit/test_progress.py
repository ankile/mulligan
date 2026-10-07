"""Log-only progress lines (mulligan.utils.progress)."""

from __future__ import annotations

import io

from mulligan.utils.progress import (
    PROGRESS_PREFIX,
    ProgressEtaEstimator,
    emit_completion,
    emit_progress,
    parse_progress_line,
)


def test_progress_lines_round_trip():
    stream = io.StringIO()
    emit_progress(
        job_type="training",
        run_id="run-1",
        phase="train",
        progress=0.5,
        current=5,
        total=10,
        metrics={"losses/critic": 0.25},
        file=stream,
    )
    emit_completion(job_type="training", success=False, message="boom", file=stream)
    lines = stream.getvalue().splitlines()
    assert all(line.startswith(PROGRESS_PREFIX) for line in lines)
    running, failed = (parse_progress_line(line) for line in lines)
    assert running.phase == "train" and running.current == 5
    assert running.metrics == {"losses/critic": 0.25}
    assert running.schema == "mulligan.progress.v1"
    assert failed.status == "failed" and failed.exit_code == 1 and failed.message == "boom"
    assert "job_id" not in running.to_dict()  # empty fields are omitted


def test_parse_ignores_other_lines():
    assert parse_progress_line("step: 1000/150000 critic: 0.1") is None
    assert parse_progress_line(PROGRESS_PREFIX + "{not json") is None


def test_eta_estimator():
    eta = ProgressEtaEstimator(start_current=0, start_time=0.0)
    assert eta.eta_s(0, 100, now=10.0) is None
    assert eta.eta_s(50, 100, now=10.0) == 10.0
    assert eta.eta_s(100, 100, now=10.0) == 0.0
