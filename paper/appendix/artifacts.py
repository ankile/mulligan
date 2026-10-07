"""Hash-locked paper evidence shared by the appendix packages.

Each package pins the evidence files it reads in ``inputs.json`` (path, sha256, size). The
bytes live at ``<path>`` of the paper evidence, read from a local mirror directory
(``$MULLIGAN_PAPER_EVIDENCE``) or from the HF dataset ``mulligan/paper-evidence``. Every file
is checked against its lock and copied into the package's build cache
(``paper/build/cache/<package>/raw/<path>``); changed bytes are rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mulligan.plotting import paper

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mulligan.paper.appendix.inputs.v3"
EVIDENCE_ENV = "MULLIGAN_PAPER_EVIDENCE"
HF_REPO = "mulligan/paper-evidence"
# The release-1 commit of mulligan/paper-evidence (release/revisions.json).
HF_REVISION: str = "4bf9994af8a4299d0a5810a10347d85215b400f2"
# Evidence directory of each real-world task key (``real/{results,collection}/<dir>/``).
REAL_DIRS = {"marker_d2": "marker", "square_d2": "square", "routing_d2": "routing"}
# Labels of every evidence file requested in this process, cache hit or not; paper.figures
# records from it which registry entries read the paper evidence.
REQUESTED: list[str] = []


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cache_dir(package: str) -> Path:
    """Build cache of one package: verified inputs under ``raw/``, derived tables under ``data/``."""
    return paper.BUILD_DIR / "cache" / package


TABLES_DIR = paper.BUILD_DIR / "tables"
# The frozen manuscript's generated table bodies (appendix/generated/*.tex).
REFERENCE_TABLES = ROOT / "paper/reference/tables"


def write_table(name: str, content: str, *, check: bool = False) -> Path:
    """Write one generated LaTeX table body to ``paper/build/tables``.

    With ``check``, the body must equal the frozen manuscript's copy byte for byte.
    """
    if check and (REFERENCE_TABLES / name).read_text() != content:
        raise AssertionError(f"generated table differs from the manuscript: {name}")
    path = TABLES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _check(path: Path, *, digest: str, size: int, label: str) -> None:
    if path.stat().st_size != size:
        raise RuntimeError(f"paper evidence size mismatch for {label}: {path}")
    if sha256(path) != digest:
        raise RuntimeError(f"paper evidence hash mismatch for {label}: {path}")


def _source(relative: str) -> Path:
    mirror = os.environ.get(EVIDENCE_ENV)
    if mirror:
        path = Path(mirror).expanduser() / relative
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing from the ${EVIDENCE_ENV} mirror")
        return path
    import huggingface_hub

    try:
        path = huggingface_hub.hf_hub_download(
            HF_REPO, relative, repo_type="dataset", revision=HF_REVISION
        )
    except Exception as error:
        raise RuntimeError(
            f"cannot download {relative} from {HF_REPO}@{HF_REVISION} ({error}); set "
            f"${EVIDENCE_ENV} to a local paper-evidence mirror (see docs/paper_figures.md)"
        ) from error
    return Path(path)


def evidence_file(path: str, *, digest: str, size: int, destination: Path) -> Path:
    """Materialize the pinned evidence file ``path`` at ``destination`` and return it.

    A warm cache is re-hashed; a corrupt cache fails without being overwritten.
    """
    REQUESTED.append(path)
    if destination.is_file():
        _check(destination, digest=digest, size=size, label=path)
        return destination
    source = _source(path)
    _check(source, digest=digest, size=size, label=path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, partial)
    partial.replace(destination)
    return destination


def fetch(package: str, row: dict) -> Path:
    """Materialize one lock row in ``package``'s build cache."""
    return evidence_file(
        row["path"],
        digest=row["sha256"],
        size=row["size"],
        destination=cache_dir(package) / "raw" / row["path"],
    )


def load_inputs(package: Path) -> dict[str, Path]:
    """Fetch the exact bytes of ``package``'s ``inputs.json``: {evidence path: local path}."""
    lock = json.loads((package / "inputs.json").read_text())
    if lock["schema"] != SCHEMA:
        raise ValueError(f"{package / 'inputs.json'}: unexpected schema {lock['schema']!r}")
    files = lock["files"]
    if len({row["path"] for row in files}) != len(files):
        raise ValueError(f"{package / 'inputs.json'}: duplicate input paths")

    def fetch_row(row: dict) -> tuple[str, Path]:
        return row["path"], fetch(package.name, row)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(fetch_row, files))


def source_paths(package: Path) -> tuple[str, ...]:
    """Record the input lock and rendering code, not disposable cache paths, in figure pins."""
    paths = [
        Path(__file__),
        ROOT / "mulligan/plotting/paper.py",
        ROOT / "mulligan/plotting/colors.py",
        *package.glob("*.py"),
        *package.glob("*.json"),
    ]
    return tuple(sorted({str(path.relative_to(ROOT)) for path in paths}))
