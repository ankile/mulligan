"""mulligan.real.robot.station + droid.misc.station_env: per-machine station file handling.

Runs only in robot mode (needs the DROID fork, docs/hardware/droid_fork.md); skipped in the main venv.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

station_env = pytest.importorskip("droid.misc.station_env")

from mulligan.real.robot import station  # noqa: E402

WORKSTATION_FILE = """\
# comment
DROID_NUC_IP=192.0.2.3
DROID_ROBOT_IP="192.0.2.2"
DROID_ROBOT_TYPE='fr3'
DROID_ROBOT_SERIAL_NUMBER=000000-0000000
"""


@pytest.fixture
def station_file(tmp_path, monkeypatch):
    path = tmp_path / "station.env"
    path.write_text(WORKSTATION_FILE)
    path.chmod(0o600)
    monkeypatch.setenv("DROID_STATION_ENV_FILE", str(path))
    for key in station_env.STATION_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delitem(sys.modules, station._PARAMETERS_MODULE, raising=False)
    return path


def test_loader_parses_quotes_comments_and_blanks(station_file):
    values = station_env.load_station_env()
    assert values["DROID_NUC_IP"] == "192.0.2.3"
    assert values["DROID_ROBOT_IP"] == "192.0.2.2"
    assert values["DROID_ROBOT_TYPE"] == "fr3"
    assert values["DROID_LAPTOP_IP"] == ""
    assert values["DROID_SUDO_PASSWORD"] == ""


def test_explicit_env_var_wins_over_file(station_file, monkeypatch):
    monkeypatch.setenv("DROID_NUC_IP", "192.0.2.9")
    assert station_env.load_station_env()["DROID_NUC_IP"] == "192.0.2.9"


def test_unknown_key_fails_loud(station_file):
    station_file.write_text(WORKSTATION_FILE + "DROID_NUC_IPP=typo\n")
    with pytest.raises(ValueError, match="unknown key 'DROID_NUC_IPP'"):
        station_env.load_station_env()


def test_secret_in_world_readable_file_refused(station_file):
    station_file.write_text(WORKSTATION_FILE + "DROID_SUDO_PASSWORD=hunter2\n")
    station_file.chmod(0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        station_env.load_station_env()


def test_missing_file_means_blank_values(station_file):
    os.remove(station_file)
    assert all(v == "" for v in station_env.load_station_env().values())


def test_require_station_env_returns_non_secret_values(station_file):
    values = station.require_station_env()
    assert values == {
        "DROID_NUC_IP": "192.0.2.3",
        "DROID_ROBOT_IP": "192.0.2.2",
        "DROID_ROBOT_TYPE": "fr3",
        "DROID_ROBOT_SERIAL_NUMBER": "000000-0000000",
    }
    assert "nuc_ip=192.0.2.3" in station.station_summary()


def test_require_station_env_missing_key(station_file):
    station_file.write_text("DROID_NUC_IP=192.0.2.3\n")
    with pytest.raises(RuntimeError, match="missing \\['DROID_ROBOT_IP'"):
        station.require_station_env()


def test_require_station_env_rejects_nuc_secret_on_workstation(station_file):
    station_file.write_text(WORKSTATION_FILE + "DROID_SUDO_PASSWORD=hunter2\n")
    with pytest.raises(RuntimeError, match="NUC-only"):
        station.require_station_env()


def test_require_station_env_detects_stale_parameters_module(station_file, monkeypatch):
    stale = types.ModuleType(station._PARAMETERS_MODULE)
    stale.nuc_ip = "192.0.2.3"
    stale.robot_ip = "192.0.2.2"
    stale.robot_type = "panda"  # file says fr3
    stale.robot_serial_number = "000000-0000000"
    monkeypatch.setitem(sys.modules, station._PARAMETERS_MODULE, stale)
    with pytest.raises(RuntimeError, match="restart the process"):
        station.require_station_env()
