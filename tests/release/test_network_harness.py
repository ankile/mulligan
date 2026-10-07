"""The network-test harness in tests/conftest.py: timeouts and the HTTP 429 message."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tests.conftest import NetworkTimeout, alarm_after, http_status


def test_alarm_interrupts_a_blocking_call():
    start = time.monotonic()
    with pytest.raises(NetworkTimeout, match="exceeded its 1 s timeout"):
        with alarm_after(1, "sleeper"):
            time.sleep(30)
    assert time.monotonic() - start < 5


def test_alarm_is_cleared_after_the_block():
    with alarm_after(1, "quick"):
        pass
    time.sleep(1.5)  # a leftover alarm would raise here


def test_http_status_follows_the_exception_chain():
    class HTTPError(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.response = SimpleNamespace(status_code=status)

    try:
        try:
            raise HTTPError(429)
        except HTTPError as inner:
            raise RuntimeError("download failed") from inner
    except RuntimeError as outer:
        assert http_status(outer) == 429
    assert http_status(ValueError("no response")) is None
