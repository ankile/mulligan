"""The HiL-SERL operator scripts and the no-human runner: usage, --help, and the recipe flags
every role gets.

A fake interpreter stands in for the roles (it answers ``mulligan.sim.recipes`` with the real
one), so these run without a GPU or a learner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mulligan.sim import recipes as R

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "scripts/hilserl/learner.sh": 2,
    "scripts/hilserl/eval_watcher.sh": 2,
    "scripts/hilserl/actor.sh": 3,
    "scripts/hilserl/fork.sh": 3,
}


def _fake_python(path: Path, body: str) -> Path:
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ " $* " == *" mulligan.sim.recipes "* ]]; then exec {sys.executable} "$@"; fi\n'
        + body
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_help_and_usage(script, tmp_path):
    path = ROOT / script
    subprocess.run(["bash", "-n", str(path)], check=True)
    for flag in ("-h", "--help"):
        result = subprocess.run(["bash", str(path), flag], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert f"Usage: {script} <" in result.stdout
    # too few arguments: usage on stderr, exit 2, nothing run
    few = ["x"] * (SCRIPTS[script] - 1)
    result = subprocess.run(["bash", str(path), *few], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 2 and "usage:" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("script", ["learner.sh", "eval_watcher.sh", "actor.sh"])
def test_roles_get_the_recipe_flags(script, tmp_path):
    fake = _fake_python(tmp_path / "python", 'printf "%s\\n" "$@"\n')
    args = ["square-broad-hilserl", str(tmp_path / "s")] + (
        ["host"] if script == "actor.sh" else []
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/hilserl" / script), *args, "--max_steps", "6000"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON": f"{fake} -u"},  # PYTHON may hold several words
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.splitlines()
    recipe = R.get_recipe(R.load(), "square-broad-hilserl")
    flags = R.train_argv(recipe, "hilserl_agent", 1)
    assert out[:3] == ["-u", "-u", "-m"]
    assert out[-len(flags) - 2 :] == [*flags, "--max_steps", "6000"]
    assert out[out.index("--session") + 1] == str(tmp_path / "s")

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/hilserl" / script), "square-narrow-rlpd", *args[1:]],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON": str(fake)},
        cwd=ROOT,
    )
    assert result.returncode == 1 and "is not a HiL-SERL recipe" in result.stderr


def test_nohuman_runs_every_role_with_the_same_flags(tmp_path, monkeypatch):
    """Watcher, learner, a 13-episode warm-up actor, then the autonomous actor once the learner
    is past its JIT compiles; every role gets the same config flags."""
    from mulligan.baselines.hilserl import nohuman

    log = tmp_path / "argv.log"
    fake = _fake_python(
        tmp_path / "python",
        f'echo "$*" >> {log}\n'
        'session=""; prev=""\n'
        'for a in "$@"; do [[ "${prev}" == --session ]] && session="${a}"; prev="${a}"; done\n'
        'if [[ " $* " == *" --learner "* ]]; then\n'
        '  mkdir -p "${session}/learner"\n'
        '  echo "heartbeat env_steps=6000 calls=10"\n'
        '  echo "{}" > "${session}/learner/endpoint.json"\n'
        "fi\n",
    )
    monkeypatch.setattr(sys, "executable", str(fake))
    monkeypatch.setattr(nohuman, "POLL_S", 0.05)
    flags = ["--task=square_narrow", "--max_steps", "6000", "--port", "5688"]
    assert nohuman.main(["--session", str(tmp_path / "session"), *flags]) == 0
    calls = log.read_text().splitlines()
    roles = sorted(c.split()[3] for c in calls)
    assert roles == ["--actor", "--actor", "--eval-watcher", "--learner"]
    for call in calls:
        assert " ".join(flags) in call
    actors = [c for c in calls if " --actor " in c]
    assert sum(c.endswith(" --max_episodes 13") for c in actors) == 1
    assert all("--no-spacemouse --no-render --unpaced" in c for c in actors)
