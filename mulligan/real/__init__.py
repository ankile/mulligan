"""Real-robot (DROID / LeRobot) code.

No import-time side effects: this package is also imported by analysis and
training code in the main venv (no ``droid``). Robot entrypoints reach DROID through
:mod:`mulligan.real.robot.droid_compat`, which validates the per-machine station file
(:mod:`mulligan.real.robot.station`) before importing ``droid.robot_env``.
"""
