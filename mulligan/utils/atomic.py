"""Crash-safe file writes: tmp file + fsync + rename.

A reader of the target path only ever sees the complete old content or the complete
new content. Used for resume metadata and for collector session files.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "fsync_dir",
]


def _tmp_path_for(path: Path) -> Path:
    """A unique sibling temp path (same directory => same filesystem => rename is atomic)."""
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def fsync_dir(directory: Path | str) -> None:
    """Flush a DIRECTORY entry to stable storage.

    ``fsync`` on the file only durably stores its CONTENT; the rename that publishes it
    lives in the parent directory's own metadata, and after a power loss a directory
    whose entry was never flushed can come back pointing at the old name (or at neither
    name).  Every commit point here is a rename, so every rename is followed by this.
    Best-effort by design on filesystems that refuse to open a directory for fsync.
    """
    path = Path(directory).expanduser()
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        # Some networked filesystems reject fsync on a directory fd; the rename is
        # still ordered by the server, so this is genuinely best-effort.
        pass
    finally:
        os.close(dir_fd)


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (tmp file + fsync + :func:`os.replace`).

    On any failure the temp file is removed and the exception propagates, so ``path``
    keeps its previous content.  A crash between the fsync and the rename likewise
    leaves the old content in place — the rename is the single commit point.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(target)
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        # The rename itself is only durable once the directory entry is flushed.
        fsync_dir(target.parent)
    except BaseException:
        # Never leave a half-written temp file behind.
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(
    path: Path | str, obj: Any, *, indent: int = 2, sort_keys: bool = True
) -> Path:
    """Atomically write ``obj`` as JSON (trailing newline, deterministic key order)."""
    payload = json.dumps(obj, indent=indent, sort_keys=sort_keys) + "\n"
    return atomic_write_bytes(path, payload.encode("utf-8"))
