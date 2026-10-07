"""``OperatorUI``: the one object an entrypoint holds for its operator surface.

Bundles the keyboard listener, the target-card window, the live camera monitor, and the
operator gates, resolved once from the shared command-line flags
(:func:`mulligan.real.operator_ui.cli.add_operator_ui_args`). Display requirements are checked
loudly at construction, before the robot is touched.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import cv2

from mulligan.real.collect.initial_states import InitialStateTarget
from mulligan.real.operator_ui.cards import CardStyle, write_initial_state_card
from mulligan.real.operator_ui.display import (
    SCHEMATIC_WINDOW_XY,
    close_windows,
    has_display,
    require_display,
    show_image_window,
    show_image_file_window,
)
from mulligan.real.operator_ui.gates import GateOutcome, operator_choice, operator_gate
from mulligan.real.operator_ui.keys import (
    TerminalKeyboardListener,
    drain_operator_keys,
    key_label,
    read_operator_key,
)
from mulligan.real.operator_ui.monitor import (
    render_camera_monitor,
    resolve_monitor_camera_keys,
)
from mulligan.real.operator_ui.panel import compose_panel, fit_diagram
from mulligan.real.operator_ui.progress import (
    CollectionScene,
    CollectionStatus,
    EvalScene,
    SessionProgress,
)


def card_window_name(task_name: str) -> str:
    return f"{task_name} initial state"


class OperatorUI:
    def __init__(
        self,
        *,
        show_card_window: bool,
        card_dir: Path | None,
        monitor_camera_keys: list[str] | None,
        show_status_window: bool = False,
    ) -> None:
        if show_card_window:
            require_display("The initial-state card window")
        if monitor_camera_keys:
            require_display("--monitor-cameras")
        if show_status_window:
            require_display("The operator status window")
        self.show_card_window = show_card_window
        self.card_dir = card_dir
        self.monitor_camera_keys = list(monitor_camera_keys) if monitor_camera_keys else None
        # Role-keyed, stored-space crop boxes of the LOADED policy (``policy.camera_crops``)
        # so the monitor shows the policy's actual trained crop rather than the station
        # default. Set by the entrypoint once the policy is loaded.
        self.monitor_crop_boxes: dict[str, tuple[int, int, int, int]] | None = None
        self._keyboard: TerminalKeyboardListener | None = None
        self.show_status_window = show_status_window or show_card_window
        self.progress = SessionProgress()
        self._progress_enabled = False
        self._diagram = None
        self._window_name = "Robot operator"
        self._last_panel_render = -float("inf")
        self.observe: Callable[[], dict] | None = None
        self._card_request = None
        self._card_target: InitialStateTarget | None = None
        self._card_meta: dict | None = None
        self._card_path: Path | None = None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        cards: bool,
        default_card_dir: Path | None = None,
    ) -> OperatorUI:
        """Resolve the flags registered by :func:`add_operator_ui_args`.

        ``cards`` must match what was passed to ``add_operator_ui_args``. When the operator
        did not pass ``--initial-state-visualization-dir``, ``default_card_dir`` is used.
        """
        monitor_keys = (
            resolve_monitor_camera_keys(args.monitor_camera_keys) if args.monitor_cameras else None
        )
        status_window = has_display() and not args.no_status_window
        if not cards:
            return cls(
                show_card_window=False,
                card_dir=None,
                monitor_camera_keys=monitor_keys,
                show_status_window=status_window,
            )
        card_dir = args.initial_state_visualization_dir or default_card_dir
        ui = cls(
            show_card_window=not args.no_show_initial_state_window,
            card_dir=None if card_dir is None else Path(card_dir),
            monitor_camera_keys=monitor_keys,
        )
        ui.show_status_window = status_window and not args.no_show_initial_state_window
        return ui

    # ---- keyboard ---------------------------------------------------------------
    @property
    def keyboard(self) -> TerminalKeyboardListener:
        """The terminal listener (cbreak mode), created on first use."""
        if self._keyboard is None:
            self._keyboard = TerminalKeyboardListener()
        return self._keyboard

    def read_key(self) -> str | None:
        self.render_status()
        return read_operator_key(self.keyboard)

    def drain_keys(self) -> None:
        drain_operator_keys(self.keyboard)

    # ---- target card ------------------------------------------------------------
    def show_card(
        self,
        target: InitialStateTarget,
        manifest_meta: dict,
        *,
        task_name: str,
        style: CardStyle | None = None,
        message: str | None = None,
    ) -> Path:
        """Write the target card PNG and (unless disabled) show it in the card window.

        Returns the PNG path, which callers record in their ledgers. ``message`` is
        printed before the card path so the operator knows why a new card appeared.
        """
        if self.card_dir is None:
            raise RuntimeError("OperatorUI.show_card needs a card_dir (no manifest card dir set)")
        style = style or CardStyle()
        request = (task_name, style)
        if (
            target is self._card_target
            and manifest_meta is self._card_meta
            and request == self._card_request
        ):
            assert self._card_path is not None
            if self.show_card_window and not (self.show_status_window and self._progress_enabled):
                show_image_file_window(self._window_name, self._card_path, xy=SCHEMATIC_WINDOW_XY)
            else:
                self.render_status(force=True, pump=True)
            return self._card_path
        path = write_initial_state_card(
            target,
            manifest_meta,
            output_dir=self.card_dir,
            task_name=task_name,
            style=style,
        )
        if message:
            print()
            print(message)
        print(f"  target card: {path}", flush=True)
        if self.show_card_window:
            self._window_name = card_window_name(task_name)
            if self.show_status_window and self._progress_enabled:
                card = cv2.imread(str(path))
                if card is None:
                    raise RuntimeError(f"Cannot read rendered target card: {path}")
                self._diagram = fit_diagram(card)
                self.render_status(force=True, pump=True)
            else:
                show_image_file_window(self._window_name, path, xy=SCHEMATIC_WINDOW_XY)
        self._card_request, self._card_path = request, path
        self._card_target, self._card_meta = target, manifest_meta
        return path

    def configure_progress(self, *, total: int | None, completed: int) -> None:
        self.progress = SessionProgress(total=total, completed=completed)
        self._progress_enabled = True

    def set_phase(self, phase: str, detail: str = "", *, controls: tuple[str, ...] = ()) -> None:
        self.progress.phase, self.progress.detail = phase, detail
        self.progress.controls = controls
        self.render_status(force=True, pump=True)

    def render_status(self, *, force: bool = False, pump: bool = False) -> None:
        if not self.show_status_window or not self._progress_enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_panel_render < 0.25:
            return
        self._last_panel_render = now
        show_image_window(
            self._window_name,
            compose_panel(self.progress, self._diagram),
            xy=SCHEMATIC_WINDOW_XY,
            pump=pump,
        )

    def show_eval_scene(self, scene: EvalScene, manifest_meta: dict, *, task_name: str) -> None:
        self.progress.scene = scene
        if scene.target is not None:
            self.show_card(
                scene.target,
                manifest_meta,
                task_name=task_name,
                style=CardStyle(show_coordinates=False),
            )
        else:
            self.render_status(force=True, pump=True)

    def show_collection_scene(
        self,
        scene: CollectionScene,
        manifest_meta: dict,
        *,
        task_name: str,
        style: CardStyle,
    ) -> Path | None:
        """The collector's counterpart of :meth:`show_eval_scene`; returns the card path."""
        self.progress.scene = scene
        if scene.target is None:
            self.render_status(force=True, pump=True)
            return None
        return self.show_card(scene.target, manifest_meta, task_name=task_name, style=style)

    def update_status(self, status: CollectionStatus) -> None:
        """Refresh the collector's quota / pace cells; repainted on the next poll."""
        self.progress.status = status
        self.render_status(force=True)

    def preview_after_rollout(
        self,
        outcome: str,
        *,
        current: EvalScene,
        upcoming: EvalScene | None,
        manifest_meta: dict,
        task_name: str,
    ) -> None:
        """Present the actual next scene before reset; retries keep the current scene."""
        if outcome == "quit":
            self.set_phase("Stopping session", "Robot returning home")
            return
        scene = current if outcome == "restart" else upcoming
        self.progress.phase = "Resetting robot"
        if scene is None:
            self.progress.detail = "All planned rollouts finished. Saving and uploading results."
        elif outcome == "restart":
            self.progress.detail = "Retry the same target. Keep clear of the arm during reset."
        elif scene.round_num == current.round_num:
            self.progress.detail = "Same target, next policy. Keep clear of the arm during reset."
        elif scene.target is None:
            self.progress.detail = "Prepare a new scene. Keep clear of the arm during reset."
        else:
            self.progress.detail = "Next target shown. Keep clear of the arm during reset."
        self.progress.controls = ()
        if scene is not None:
            self.show_eval_scene(scene, manifest_meta, task_name=task_name)
        else:
            self.render_status(force=True, pump=True)

    def begin_rollout(
        self,
        max_steps: int,
        subtask_marks: int,
        *,
        phase: str = "Policy running",
        controls: tuple[str, ...] | None = None,
    ) -> None:
        """Start the episode clock and step counter; ``phase`` names who is in control."""
        self.progress.begin_rollout(max_steps, subtask_marks)
        if controls is None:
            second = f"{key_label('r')} retry   |   {key_label('q')} quit"
            if subtask_marks:
                second += f"   |   {key_label('g')} subgoal"
            controls = ("1 success   |   9 failure   |   0 timeout", second)
        self.set_phase(phase, controls=controls)

    def finish_rollout(self, outcome: str) -> None:
        self.progress.finish_rollout(outcome)
        self.set_phase("Resetting robot", f"Last rollout: {outcome}. Robot returning home.")

    def end_rollout(self, outcome: str, *, phase: str, detail: str = "") -> None:
        """Stop the episode clock without counting it (collection: quotas count credits)."""
        self.progress.end_rollout(outcome)
        self.set_phase(phase, detail)

    # ---- live camera monitor ----------------------------------------------------
    def render_monitor(self, obs_image: dict) -> None:
        """Repaint the cropped camera monitor windows; no-op when the monitor is off."""
        if self.monitor_camera_keys:
            render_camera_monitor(
                obs_image, self.monitor_camera_keys, crop_boxes=self.monitor_crop_boxes
            )

    # ---- gates ------------------------------------------------------------------
    def gate(
        self,
        *,
        prompt: str,
        on_reset: Callable[[], object] | None = None,
        can_skip: bool = False,
        can_quit: bool = True,
        any_key_starts: bool = False,
        render: Callable[[], None] | None = None,
        render_interval_s: float = 0.5,
    ) -> GateOutcome:
        """Block on an operator decision; see :func:`operator_gate`."""
        self.progress.start()
        controls = ("Any key starts" if any_key_starts else "Enter starts",)
        actions = []
        if on_reset is not None:
            actions.append(f"{key_label('r')} re-home")
        if can_skip:
            actions.append(f"{key_label('k')} skip")
        if can_quit:
            actions.append(f"{key_label('q')} quit")
        if actions:
            controls += ("   |   ".join(actions),)
        self.set_phase("Set up the target", prompt, controls=controls)

        def refresh() -> None:
            if render is not None:
                render()
            elif self.monitor_camera_keys and self.observe is not None:
                self.render_monitor(self.observe()["image"])
            self.render_status()

        def reset() -> None:
            self.set_phase("Resetting robot", "Keep clear of the arm during reset.")
            on_reset()
            self.set_phase("Set up the target", prompt, controls=controls)

        return operator_gate(
            self.keyboard,
            prompt=prompt,
            on_reset=reset if on_reset is not None else None,
            can_skip=can_skip,
            can_quit=can_quit,
            any_key_starts=any_key_starts,
            render=refresh,
            render_interval_s=render_interval_s,
        )

    def choose(
        self,
        *,
        prompt: str,
        choices: dict[str, str],
        default: str | None,
        phase: str = "Choose what happens next",
    ) -> str:
        """Block on a between-episode decision; see :func:`operator_choice`.

        The panel shows ``phase`` with the prompt as its detail and the offered keys as
        its controls while the operator decides.
        """
        self.progress.start()
        controls = tuple(f"{key_label(key)} {action}" for key, action in choices.items())
        self.set_phase(phase, prompt, controls=controls)

        def refresh() -> None:
            if self.monitor_camera_keys and self.observe is not None:
                self.render_monitor(self.observe()["image"])
            self.render_status()

        return operator_choice(
            self.keyboard,
            prompt=prompt,
            choices=choices,
            default=default,
            render=refresh,
        )

    # ---- lifecycle --------------------------------------------------------------
    def close(self) -> None:
        """Restore the terminal and close every operator window. Idempotent."""
        if self._keyboard is not None:
            self._keyboard.close()
            self._keyboard = None
        close_windows()

    def __enter__(self) -> OperatorUI:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
