"""Acceptance checks for the Arena: the read-only snapshot and the self-deploy app.

Snapshot: install, closure, lint, unit tests, package build, data, leak and browser checks.
Self-deploy app: type-check (app, Convex backend, release, build configs), the captured release API
types against the backend, the app build against a placeholder Convex URL with its leak check, and
the Python client tests.

Each check shells out to the same commands CI runs from `arena/`. Every check needs `bun install`
(or `uv` for the Python client), which downloads packages, so the whole module is marked `network`
and the offline selection (`-m "not network"`) neither downloads nor writes into `arena/`. The
browser check serves `arena/dist-release` on a free local port and also needs network access for the
Hugging Face videos.
"""

from __future__ import annotations

import http.server
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ARENA = Path(__file__).resolve().parents[2] / "arena"
BUN = shutil.which("bun")
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
PLAYWRIGHT = "playwright==1.63.0"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(BUN is None, reason="bun is not on PATH (https://bun.sh)"),
]


def run(*args: str, timeout: int = 900, cwd: Path = ARENA, env: dict | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=None if env is None else {**os.environ, **env},
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{' '.join(args)} exited {result.returncode}:\n{output[-4000:]}"
    return output


@pytest.fixture(scope="module")
def installed() -> None:
    run(BUN, "install", "--frozen-lockfile")


@pytest.fixture(scope="module")
def packaged(installed: None) -> str:
    return run(BUN, "run", "package:release", "data/release.json")


def test_release_closure_matches_committed_set(installed: None) -> None:
    assert "Release closure OK" in run(BUN, "run", "check:closure")


def test_lint(installed: None) -> None:
    run(BUN, "run", "lint")


def test_unit_tests(installed: None) -> None:
    assert " 0 fail" in run(BUN, "test")


def test_package_release_builds(packaged: str) -> None:
    assert "Validated 5 tasks" in packaged
    staged = {p.name for p in (ARENA / "dist-release" / "data").iterdir()}
    expected = {
        line.split("  ", 1)[1] for line in (ARENA / "data" / "SHA256SUMS").read_text().splitlines()
    }
    assert staged == expected
    for name in expected:
        assert (ARENA / "dist-release" / "data" / name).read_bytes() == (
            ARENA / "data" / name
        ).read_bytes()


def test_verify_release_data(installed: None) -> None:
    assert "Verified 110 policy points" in run(BUN, "run", "verify:data")


def test_leak_check(packaged: str) -> None:
    assert "Leak check passed" in run(BUN, "run", "leak-check")


# Self-deploy app (docs/arena.md, "Deploy your own Arena"). No Convex account or deployment is
# involved: the build only embeds the configured URL.
SELF_DEPLOY_URL = "https://example.convex.cloud"


def test_typecheck_app_backend_release(installed: None) -> None:
    run(BUN, "run", "typecheck")


def test_release_api_types_match_backend(installed: None) -> None:
    assert "matches convex/" in run(BUN, "run", "check:api")


def test_self_deploy_build(installed: None) -> None:
    # Process env beats a developer's arena/.env.local, so the result does not depend on it.
    env = {"VITE_CONVEX_URL": SELF_DEPLOY_URL, "VITE_CONVEX_SITE_URL": ""}
    output = run(BUN, "run", "build", env=env)
    assert "Leak check passed for dist (self-deploy build)" in output
    html = (ARENA / "dist" / "index.html").read_text()
    assert 'http-equiv="Content-Security-Policy"' in html
    assert f"{SELF_DEPLOY_URL} wss://example.convex.cloud https://example.convex.site" in html
    assert (ARENA / "dist" / "_headers").is_file()
    bundle = "".join(p.read_text() for p in (ARENA / "dist" / "assets").glob("*.js"))
    assert SELF_DEPLOY_URL in bundle


def test_python_client() -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    output = run(
        uv, "run", "--no-project", "--isolated", "--with", "convex", "--with", "pytest",
        "env", "PYTHONPATH=.", "python", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
        cwd=ARENA / "python",
    )  # fmt: skip
    assert " passed" in output and "failed" not in output


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ARENA / "dist-release"), **kwargs)

    def log_message(self, *args) -> None:
        pass


def _chrome() -> str | None:
    configured = os.environ.get("MULLIGAN_CHROME")
    if configured:
        return shutil.which(configured)
    return next((path for name in CHROME_NAMES if (path := shutil.which(name))), None)


def _playwright_python() -> list[str]:
    if importlib.util.find_spec("playwright") is not None:
        return [sys.executable]
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("verify_release_browser.py needs playwright importable or uv on PATH")
    return [uv, "tool", "run", "--from", PLAYWRIGHT, "python"]


@pytest.mark.skipif(
    _chrome() is None, reason="no Chrome: set MULLIGAN_CHROME or put google-chrome/chromium on PATH"
)
def test_browser_against_local_serve(packaged: str, tmp_path: Path) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        output = run(
            *_playwright_python(),
            "scripts/verify_release_browser.py",
            url,
            str(tmp_path),
            timeout=1200,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert "No live backend traffic or write requests" in output
