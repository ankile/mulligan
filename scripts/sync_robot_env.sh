#!/bin/bash
# Build / repair the robot-workstation environment ("robot mode") from robot/uv.lock.
#
# Two uv projects live in this repo (see robot/pyproject.toml for the why):
#   ML mode    -- repo root:  uv sync --frozen [--extra ...]   -> ./.venv       (numpy 2.x)
#   robot mode -- robot/:     bash scripts/sync_robot_env.sh   -> robot/.venv   (numpy 1.26.4,
#                                                                droid, gym, zerorpc, pyzed)
# They have separate lockfiles and venvs, so a plain `uv sync` / `uv run` at the repo root
# never touches the robot venv. Robot launches use
#     uv run --project robot --frozen python -m mulligan.real.<entrypoint> ...
# and a forgotten --project fails loudly (no `droid` in the ML venv) instead of silently
# re-pinning numpy.
#
# Everything in robot mode is in robot/uv.lock except the pyzed wheel bytes: the wheel must
# match the ZED SDK installed at /usr/local/zed, so the station owner supplies it at
# robot/wheels/ (gitignored). If it is missing, this script downloads it from Stereolabs;
# the lock pins its sha256, so a different wheel fails `uv sync --frozen`. Idempotent; safe
# to run before every launch.
set -euo pipefail

cd "$(dirname "$0")/.."
ZED_HEADER=/usr/local/zed/include/sl/Camera.hpp
WHEEL=robot/wheels/pyzed-4.2-cp312-cp312-linux_x86_64.whl
WHEEL_URL=https://download.stereolabs.com/zedsdk/4.2/whl/linux_x86_64/pyzed-4.2-cp312-cp312-linux_x86_64.whl

echo "### [1/3] pyzed wheel (must match the installed ZED SDK)"
[ -f "$ZED_HEADER" ] || { echo "ERROR: ZED SDK not installed ($ZED_HEADER missing); robot mode needs it." >&2; exit 1; }
SDK_VER="$(grep -E '#define ZED_SDK_MAJOR_VERSION' "$ZED_HEADER" | awk '{print $3}').$(grep -E '#define ZED_SDK_MINOR_VERSION' "$ZED_HEADER" | awk '{print $3}')"
[ "$SDK_VER" = "4.2" ] || { echo "ERROR: installed ZED SDK is $SDK_VER but robot/pyproject.toml + uv.lock pin pyzed 4.2. Update the wheel path/URL there and re-lock (uv lock --project robot)." >&2; exit 1; }
if [ -f "$WHEEL" ]; then
  echo "    $WHEEL present"
else
  mkdir -p "$(dirname "$WHEEL")"
  echo "    downloading $WHEEL_URL"
  curl -sSfL -o "$WHEEL" "$WHEEL_URL"
fi

echo "### [2/3] uv sync --frozen --project robot (exact: prunes anything not in robot/uv.lock)"
uv sync --frozen --project robot
# opencv-python and opencv-python-headless unpack into the same cv2/ directory. The lock
# excludes headless (robot/pyproject.toml), but a venv that previously had it loses the
# shared cv2 files when headless is pruned, leaving opencv-python registered yet gutted.
# Detect that and reinstall the GUI build; the verify step below asserts it.
if ! robot/.venv/bin/python -c "import cv2; assert 'GUI:' in cv2.getBuildInformation() and 'GUI:                           NONE' not in cv2.getBuildInformation()" 2>/dev/null; then
  echo "    cv2 has no highgui (headless leftovers) -- reinstalling opencv-python"
  uv sync --frozen --project robot --reinstall-package opencv-python
fi

echo "### [3/3] verify the eval/collection import chain + station identity"
robot/.venv/bin/python - <<'PYEOF'
import warnings
warnings.filterwarnings("ignore")
import cv2
_gui = [l.strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith("GUI:")]
assert _gui and "NONE" not in _gui[0], f"cv2 {cv2.__version__} has no highgui ({_gui}); operator windows need Qt/GTK"
assert hasattr(cv2, "aruco"), "cv2 lacks aruco (droid.misc.parameters needs it)"
import numpy, torch, lerobot, pyzed.sl, zerorpc  # noqa: F401
# droid_compat validates the per-machine station file (~/.config/droid/station.env,
# outside git) before importing droid.robot_env; a missing/incomplete file fails here.
from mulligan.real.robot.droid_compat import RobotEnv  # noqa: F401
from mulligan.real.robot.station import station_summary
import droid
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} breaks pyzed's dtype ABI"
assert "/robot/.venv/" in droid.__file__, f"droid resolved outside robot/.venv: {droid.__file__}"
print(f"numpy {numpy.__version__} | torch {torch.__version__} | lerobot {lerobot.__version__}")
print(f"droid {droid.__file__}")
print(f"station {station_summary()}")
print("pyzed / zerorpc / droid.RobotEnv all import OK")
PYEOF
echo
echo "DONE. Launch robot work with:  uv run --project robot --frozen python -m mulligan.real.<entrypoint> ..."
