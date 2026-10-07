"""Helpers for fail-loud background episode saves."""

from __future__ import annotations

import concurrent.futures


def wait_for_background_save(
    save_future: concurrent.futures.Future,
    *,
    description: str,
    timeout: float | None = None,
) -> None:
    """Wait for a background save and raise with context on failure."""
    try:
        save_future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        timeout_msg = f" after {timeout:g}s" if timeout is not None else ""
        raise RuntimeError(f"{description} timed out{timeout_msg}") from exc
    except Exception as exc:
        raise RuntimeError(f"{description} failed") from exc
