"""Timing, retries, resume and pre-reset scene selection without a robot."""

from __future__ import annotations

import pytest

from mulligan.real.operator_ui.progress import (
    CollectionScene,
    CollectionStatus,
    EvalScene,
    SessionProgress,
)
from mulligan.real.operator_ui.session import OperatorUI


def test_eta_uses_full_cycles_and_excludes_restored_completions_from_pace():
    now = [500.0]
    p = SessionProgress(total=100, completed=24, clock=lambda: now[0])
    assert p.eta is None and p.elapsed == 0
    now[0] += 1000  # loading models is not timed work
    p.start()
    now[0] += 20  # initial placement
    p.begin_rollout(850, 1)
    now[0] += 40
    p.finish_rollout("success")
    assert p.completed == 25 and p.eta == 75 * 60
    now[0] += 15  # reset + save wait + placement
    p.begin_rollout(850, 1)
    now[0] += 20
    p.finish_rollout("restart")
    assert p.completed == 25 and p.session_completed == 1
    now[0] += 25  # retry reset and placement
    p.begin_rollout(850, 1)
    now[0] += 30
    p.finish_rollout("timeout")
    assert p.completed == 26 and p.eta == 74 * 75


def test_unknown_total_quit_and_completion():
    p = SessionProgress()
    p.finish_rollout("quit")
    assert p.completed == 0 and p.eta is None and p.remaining is None
    p.finish_rollout("failure")
    assert p.completed == 1 and p.eta is None
    p = SessionProgress(total=1)
    p.finish_rollout("success")
    assert p.eta == 0
    with pytest.raises(ValueError, match="More rollouts"):
        p.finish_rollout("timeout")


@pytest.mark.parametrize(
    "outcome,use_next",
    [("success", True), ("failure", True), ("timeout", True), ("restart", False)],
)
def test_preview_chooses_next_or_retry_scene(outcome, use_next, monkeypatch):
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    current = EvalScene(8, 50, "B", 2, 2)
    upcoming = EvalScene(11, 50, "A", 1, 2)  # resume may skip already-complete rounds
    shown = []
    monkeypatch.setattr(ui, "show_eval_scene", lambda scene, *_args, **_kw: shown.append(scene))
    ui.preview_after_rollout(
        outcome, current=current, upcoming=upcoming, manifest_meta={}, task_name="routing_d2"
    )
    assert shown == [upcoming if use_next else current]


def test_final_rollout_and_quit_never_show_a_nonexistent_target(monkeypatch):
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    current = EvalScene(50, 50, "B", 2, 2)

    def unexpected(*_args, **_kw):
        pytest.fail("There is no next scene")

    monkeypatch.setattr(ui, "show_eval_scene", unexpected)
    ui.preview_after_rollout(
        "success", current=current, upcoming=None, manifest_meta={}, task_name="routing_d2"
    )
    assert "All planned rollouts finished" in ui.progress.detail
    ui.preview_after_rollout(
        "quit", current=current, upcoming=None, manifest_meta={}, task_name="routing_d2"
    )
    assert ui.progress.phase == "Stopping session"


def test_eval_gate_polls_fresh_camera_observations(monkeypatch):
    from mulligan.real.operator_ui import session
    from mulligan.real.operator_ui.gates import GateOutcome
    from tests.real.test_operator_ui_keys import FakeListener

    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui.monitor_camera_keys = ["camera"]
    ui._keyboard = FakeListener()
    observations = iter([{"image": {"frame": 1}}, {"image": {"frame": 2}}])
    ui.observe = lambda: next(observations)
    painted = []
    monkeypatch.setattr(ui, "render_monitor", painted.append)

    def gate(_keyboard, **kwargs):
        kwargs["render"]()
        kwargs["render"]()
        return GateOutcome.START

    monkeypatch.setattr(session, "operator_gate", gate)
    ui.gate(prompt="Set up the target")
    assert painted == [{"frame": 1}, {"frame": 2}]


def test_end_rollout_stops_the_clock_without_counting():
    now = [0.0]
    p = SessionProgress(clock=lambda: now[0])
    p.begin_rollout(0, 1)
    now[0] += 30
    p.end_rollout("discard")  # collection outcomes are not eval outcomes; never counted
    assert p.rollout_started is None
    assert p.completed == 0 and p.session_completed == 0
    with pytest.raises(ValueError, match="outcome is required"):
        p.end_rollout("")


def test_collection_status_validates_its_cells():
    with pytest.raises(ValueError, match="at most 3 metrics"):
        CollectionStatus(metrics=(("A", "1"), ("B", "2"), ("C", "3"), ("D", "4")))
    with pytest.raises(ValueError, match="fraction"):
        CollectionStatus(metrics=(), fraction=1.2)
    assert CollectionStatus(metrics=(), fraction=None).summary == ""


def test_collection_status_drives_the_header_and_bar():
    now = [100.0]
    p = SessionProgress(clock=lambda: now[0])
    assert p.header_metrics()[0] == ("ROLLOUTS FINISHED", "0") and p.bar_fraction is None
    p.status = CollectionStatus(
        metrics=(
            ("WITH-CF QUOTA", "12 / 50"),
            ("NO-CF QUOTA", "20 / 50"),
            ("EST. TIME LEFT", "1h 02m"),
        ),
        fraction=0.32,
        summary="Credited 7   |   4m 12s / ep   |   ~27 left",
    )
    p.start()
    now[0] += 125
    assert [label for label, _ in p.header_metrics()] == [
        "WITH-CF QUOTA",
        "NO-CF QUOTA",
        "ELAPSED",
        "EST. TIME LEFT",
    ]
    assert p.header_metrics()[2] == ("ELAPSED", "2m 05s")
    assert p.bar_fraction == 0.32
    p.scene = CollectionScene("start 004", "CF replay", 13)
    assert p.context == (
        "Start 004   |   CF replay   |   Episode 13   |   "
        "Credited 7   |   4m 12s / ep   |   ~27 left"
    )
    p.status = CollectionStatus(metrics=(("EPISODES SAVED", "3"),), fraction=None)
    assert p.header_metrics() == (("EPISODES SAVED", "3"), ("ELAPSED", "2m 05s"))
    assert p.context == "Start 004   |   CF replay   |   Episode 13"


def test_collection_scene_never_names_the_arm():
    assert "Arm" not in CollectionScene("target 013", "First rollout", 13).description


def test_widest_collection_context_fits_the_panel():
    # compose_panel raises on overflow by design; pin the widest realistic line.
    from mulligan.real.operator_ui.panel import compose_panel

    p = SessionProgress(
        clock=lambda: 10 * 3600.0,
        started=0.0,
        scene=CollectionScene("start 100", "CF replay", 113),
        status=CollectionStatus(
            metrics=(
                ("WITH-CF QUOTA", "100 / 100"),
                ("NO-CF QUOTA", "100 / 100"),
                ("EST. TIME LEFT", "12h 59m"),
            ),
            fraction=1.0,
            summary="Credited 117   |   14m 12s / ep   |   ~127 left",
        ),
        phase="Choose what happens next",
        detail="Episode ended (no counterfactual eligible)",
    )
    assert compose_panel(p, None).shape == (130, 1000, 3)
