"""Packaging invariants for the two uv projects (root and robot/).

The robot project installs the root package editable and replaces its numpy bound
with a lock-global numpy==1.26.4 override for pyzed. That only works while the root
package leaves numpy unpinned in [project.dependencies] and bounds it solely through
[tool.uv].override-dependencies.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO = Path(__file__).resolve().parents[2]
ROOT_PYPROJECT = REPO / "pyproject.toml"
ROBOT_PYPROJECT = REPO / "robot" / "pyproject.toml"
ROOT_LOCK = REPO / "uv.lock"
ROBOT_LOCK = REPO / "robot" / "uv.lock"

OPENCV_DISTS = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}
HEADLESS_OVERRIDE = "opencv-python-headless ; sys_platform == 'never'"
GIT_PINNED = ("lerobot", "robosuite", "mimicgen")


def _load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _name(req: Requirement) -> str:
    return req.name.lower().replace("_", "-")


def _project_requirements(pyproject: dict) -> list[Requirement]:
    project = pyproject["project"]
    reqs = [Requirement(r) for r in project.get("dependencies", [])]
    for extra in project.get("optional-dependencies", {}).values():
        reqs.extend(Requirement(r) for r in extra)
    return reqs


def _overrides(pyproject: dict) -> list[Requirement]:
    return [Requirement(r) for r in pyproject["tool"]["uv"]["override-dependencies"]]


def _lock_packages(lock: dict, name: str) -> list[dict]:
    return [p for p in lock["package"] if p["name"] == name]


@pytest.fixture(scope="module")
def root() -> dict:
    return _load(ROOT_PYPROJECT)


@pytest.fixture(scope="module")
def robot() -> dict:
    return _load(ROBOT_PYPROJECT)


@pytest.fixture(scope="module")
def root_lock() -> dict:
    return _load(ROOT_LOCK)


@pytest.fixture(scope="module")
def robot_lock() -> dict:
    return _load(ROBOT_LOCK)


def test_root_numpy_unpinned_in_dependencies(root):
    numpy_deps = [r for r in _project_requirements(root) if _name(r) == "numpy"]
    assert len(numpy_deps) == 1, [str(r) for r in numpy_deps]
    (numpy,) = numpy_deps
    assert str(numpy.specifier) == "", f"numpy must be unpinned in the root project: {numpy}"
    assert numpy.marker is None
    assert str(numpy) in root["project"]["dependencies"]


def test_root_numpy_bounded_only_by_override(root):
    overrides = [r for r in _overrides(root) if _name(r) == "numpy"]
    assert [str(r) for r in overrides] == ["numpy<2.3,>=2.0"]
    constraints = root["tool"]["uv"].get("constraint-dependencies", [])
    assert not [c for c in constraints if _name(Requirement(c)) == "numpy"]


def test_robot_numpy_override(robot):
    overrides = [r for r in _overrides(robot) if _name(r) == "numpy"]
    assert [str(r) for r in overrides] == ["numpy==1.26.4"]


def test_robot_installs_root_package_editable(robot):
    names = {_name(Requirement(r)) for r in robot["project"]["dependencies"]}
    assert "mulligan" in names
    source = robot["tool"]["uv"]["sources"]["mulligan"]
    assert source == {"path": "..", "editable": True}


@pytest.mark.parametrize("which", ["root", "robot"])
def test_headless_opencv_disabled_by_override(which, root, robot):
    pyproject = root if which == "root" else robot
    assert HEADLESS_OVERRIDE in pyproject["tool"]["uv"]["override-dependencies"]


def test_root_declares_exactly_one_opencv(root):
    declared = {_name(r) for r in _project_requirements(root)} & OPENCV_DISTS
    assert declared == {"opencv-python"}


@pytest.mark.parametrize("which", ["root", "robot"])
def test_lock_has_exactly_one_opencv(which, root_lock, robot_lock):
    lock = root_lock if which == "root" else robot_lock
    present = sorted({p["name"] for p in lock["package"]} & OPENCV_DISTS)
    assert present == ["opencv-python"]


def test_git_pins_match_between_projects(root, robot):
    root_sources = root["tool"]["uv"]["sources"]
    robot_sources = robot["tool"]["uv"]["sources"]
    for name in (*GIT_PINNED, "torch", "torchvision"):
        assert robot_sources[name] == root_sources[name], name
    for name in GIT_PINNED:
        assert len(root_sources[name]["rev"]) == 40, f"{name} must be pinned to a full commit"
    assert root["tool"]["uv"]["index"] == robot["tool"]["uv"]["index"]


def test_root_lock_numpy_and_single_sim_stack(root_lock):
    assert [p["version"] for p in _lock_packages(root_lock, "numpy")] == ["2.2.6"]
    robosuite = _lock_packages(root_lock, "robosuite")
    assert len(robosuite) == 1 and robosuite[0]["version"].startswith("1.5.")
    assert "85abee228d1c43ab1939bce33028099945d453b4" in robosuite[0]["source"]["git"]
    assert [p["version"] for p in _lock_packages(root_lock, "mujoco")] == ["3.3.7"]
    assert len(_lock_packages(root_lock, "nvidia-cudnn-cu12")) == 1


def test_root_lock_has_no_paper_rlpd_stack(root_lock):
    names = {p["name"] for p in root_lock["package"]}
    assert not names & {"robomimic", "d4rl", "policy-arena"}
    for gym in _lock_packages(root_lock, "gym"):
        assert Version(gym["version"]) >= Version("0.26"), gym["version"]


def test_root_lock_torch_linux_from_cu128_index(root_lock):
    sources = {p["version"]: p["source"] for p in _lock_packages(root_lock, "torch")}
    assert sources["2.11.0+cu128"] == {"registry": "https://download.pytorch.org/whl/cu128"}
    assert sources["2.11.0"] == {"registry": "https://pypi.org/simple"}


def test_robot_lock_numpy_1x(robot_lock):
    assert [p["version"] for p in _lock_packages(robot_lock, "numpy")] == ["1.26.4"]
    (droid,) = _lock_packages(robot_lock, "droid")
    assert "c0c8b29e1fcf76e424177529d5440620234918be" in droid["source"]["git"]


def test_robot_python_version_matches_pyzed_wheel(robot):
    # robot/ is its own uv project, so uv does not read the root .python-version for it. Without
    # robot/.python-version uv picks the newest Python >= 3.12, where the cp312 pyzed wheel does not
    # install and numpy 1.26.4 has no wheel.
    pinned = (REPO / "robot" / ".python-version").read_text().strip()
    assert pinned == (REPO / ".python-version").read_text().strip()
    wheel = robot["tool"]["uv"]["sources"]["pyzed"]["path"]
    assert f"-cp{pinned.replace('.', '')}-" in wheel, (pinned, wheel)
