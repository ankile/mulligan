"""Repository-wide pytest options and the shared setup of the network tests.

Network tests (``-m network``) get a wall-clock timeout (``NETWORK_TIMEOUT_S``, longer
for ``slow``), share one Hugging Face cache per session (``hf_cache``: ``$MULLIGAN_HF_CACHE``
or a session temp dir), and an anonymous HF rate limit (HTTP 429) fails with a message
that says so instead of a bare HTTP error.
"""

from __future__ import annotations

import os
import signal
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

# JAX tests run on the CPU backend: deterministic, and xdist workers never compete for a GPU.
# Set before any test module initializes a JAX backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HF_CACHE_ENV = "MULLIGAN_HF_CACHE"
NETWORK_TIMEOUT_S = {"network": 900, "slow": 3600}


@pytest.fixture(scope="session")
def hf_cache(tmp_path_factory) -> Path:
    """One huggingface_hub cache for the session: ``$MULLIGAN_HF_CACHE`` or a session temp dir."""
    configured = os.environ.get(HF_CACHE_ENV)
    return Path(configured) if configured else tmp_path_factory.mktemp("hf-cache")


class NetworkTimeout(Exception):
    pass


@contextmanager
def alarm_after(seconds: int, what: str):
    """Raise ``NetworkTimeout`` in the main thread once ``seconds`` of wall clock have passed.

    SIGALRM-based (pytest-timeout is not a dependency); interrupts blocking socket reads.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("alarm_after needs the main thread (SIGALRM)")

    def expire(signum, frame):
        raise NetworkTimeout(f"{what} exceeded its {seconds} s timeout")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture(autouse=True)
def network_timeout(request):
    """Wall-clock limit for every network test."""
    if request.node.get_closest_marker("network") is None:
        yield
        return
    seconds = NETWORK_TIMEOUT_S["slow" if request.node.get_closest_marker("slow") else "network"]
    with alarm_after(seconds, request.node.nodeid):
        yield


def http_status(error: BaseException | None) -> int | None:
    """HTTP status of ``error`` or of the first exception in its cause/context chain."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status is not None:
            return status
        error = error.__cause__ or error.__context__
    return None


RATE_LIMIT_MESSAGE = (
    "Hugging Face rate limit (HTTP 429) for anonymous requests. Wait a few minutes and rerun; "
    f"set ${HF_CACHE_ENV} to a persistent directory so files already downloaded are reused."
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except Exception as error:
        if item.get_closest_marker("network") is not None and http_status(error) == 429:
            raise pytest.fail.Exception(
                f"{RATE_LIMIT_MESSAGE} ({type(error).__name__}: {(str(error).splitlines() or [''])[0]})"
            ) from error
        raise
