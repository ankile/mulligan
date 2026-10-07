"""Operator-facing UI for the real robot: target cards, OpenCV windows, keys, gates.

Modules:

- ``display``: display detection, the HighGUI prewarm, window creation / tiling / close.
  Import it (and call ``prewarm_highgui()``) before lerobot / torch in every entrypoint.
- ``keys``: the terminal listener, numpad aliases, the OpenCV key poll, the drain.
- ``cards``: the per-task initial-state target card renderers (matplotlib only).
- ``monitor``: live cropped camera monitor windows.
- ``gates``: the "set up the scene, then press a key" decision gate.
- ``progress`` / ``panel``: session timing, anonymous eval context and live raster display.
- ``session``: :class:`OperatorUI`, the object an entrypoint holds for all of the above.
- ``cli``: the shared argparse flags.
- ``preview``: ``python -m mulligan.real.operator_ui.preview`` to render / show a manifest's cards.
"""
