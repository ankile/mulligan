"""Robot-station identity: validation of the per-machine DROID station file.

The values themselves (NUC/robot/laptop IPs, robot type, Franka serial, and on
the NUC its sudo password) are NOT in this repo. DROID reads them from
``~/.config/droid/station.env`` (or ``DROID_*`` env vars) via
``droid.misc.station_env`` in the DROID fork pinned by ``robot/pyproject.toml``
(``docs/hardware/droid_fork.md``; keys in ``configs/real/station.env.example``).
This module only checks, before any robot motion, that the workstation has a
complete file, and reports the non-secret part of it.

Called from :mod:`mulligan.real.robot.droid_compat`, i.e. only in robot mode
(the ``robot/`` project); the main venv has no ``droid`` and never needs the file.
"""

from __future__ import annotations

import sys

#: Keys a WORKSTATION needs to reach the robot. The NUC additionally needs
#: DROID_SUDO_PASSWORD, which must never be set on the workstation.
REQUIRED_WORKSTATION_KEYS = (
    "DROID_NUC_IP",
    "DROID_ROBOT_IP",
    "DROID_ROBOT_TYPE",
    "DROID_ROBOT_SERIAL_NUMBER",
)
FORBIDDEN_WORKSTATION_KEYS = ("DROID_SUDO_PASSWORD", "DROID_UBUNTU_PRO_TOKEN")

_PARAMETERS_MODULE = "droid.misc.parameters"


def require_station_env() -> dict[str, str]:
    """Load + validate the station file for the workstation role; return non-secret values.

    Raises RuntimeError with the fix when the file is missing/incomplete, when a
    NUC-only secret is present on the workstation, or when ``droid.misc.parameters``
    was already imported with different values (it binds them at import time).
    """
    try:
        from droid.misc.station_env import load_station_env, station_env_path
    except ImportError as exc:  # ML venv
        raise RuntimeError(
            "droid is not installed in this environment; real-robot entrypoints run in robot "
            "mode: `uv run --project robot --frozen python -m mulligan.real...` "
            "(bash scripts/sync_robot_env.sh to build it; see docs/station.md)."
        ) from exc

    path = station_env_path()
    values = load_station_env(path)
    missing = [k for k in REQUIRED_WORKSTATION_KEYS if not values[k]]
    if missing:
        raise RuntimeError(
            f"station file {path} is missing {missing}. Create it (chmod 600) with one KEY=VALUE "
            "per line -- keys in configs/real/station.env.example; the values are this "
            "station's network/robot identity and deliberately live outside git."
        )
    present_secrets = [k for k in FORBIDDEN_WORKSTATION_KEYS if values[k]]
    if present_secrets:
        raise RuntimeError(
            f"{present_secrets} set in {path}: those are NUC-only; the workstation never runs "
            "`sudo` for the controller. Remove them here."
        )
    params = sys.modules.get(_PARAMETERS_MODULE)
    if params is not None:
        bound = {
            "DROID_NUC_IP": params.nuc_ip,
            "DROID_ROBOT_IP": params.robot_ip,
            "DROID_ROBOT_TYPE": params.robot_type,
            "DROID_ROBOT_SERIAL_NUMBER": params.robot_serial_number,
        }
        expected = {k: values[k] for k in bound}
        if bound != expected:
            raise RuntimeError(
                f"{_PARAMETERS_MODULE} bound {bound} but the station file now says {expected}; "
                "the file changed after droid was imported -- restart the process."
            )
    return {k: v for k, v in values.items() if k not in FORBIDDEN_WORKSTATION_KEYS and v}


def station_summary() -> str:
    """One-line, secret-free description for logs."""
    v = require_station_env()
    return " ".join(f"{k.removeprefix('DROID_').lower()}={v[k]}" for k in REQUIRED_WORKSTATION_KEYS)
