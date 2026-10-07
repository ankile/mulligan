"""Every pin of a public `mulligan/*` repo equals `release/revisions.json`.

`release/revisions.json` is the one repo -> {revision, tag} lookup. This test scans every file
under `release/`, `configs/`, `mulligan/`, `paper/` and `tests/` and fails if any of them pins
a `mulligan/*` repo at another revision, or pins a repo the lookup does not know. It also
catches stale pins that still resolve on HF, which the network resolution check cannot.

What counts as a pin (a revision is a 40-hex commit, a >=7-hex prefix, or a `release-*` tag):

- JSON / YAML / TOML / CSV: a mapping with a repo key and a revision key of the same prefix
  (`repo`/`revision`, `destination_repo`/`destination_revision`, `demo_repo_id`/`demo_revision`,
  ...); a mapping `{"mulligan/x": "<revision>"}`; a mapping `{"mulligan/x": {"revision": ...}}`.
- Python: the same key pairing over dict literals, call keywords, and assignments in one scope
  (`EVAL_REPO = ...` / `EVAL_REVISION = ...`).
- Any text file: `mulligan/x@<rev>`, HF URLs `.../mulligan/x/(resolve|tree|blob)/<rev>/...`, and a
  repo and `revision=<rev>` on the same line.

Files exempt from the scan are listed with a reason (EXEMPT below).
"""

from __future__ import annotations

import ast
import csv
import fnmatch
import io
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
REVISIONS = REPO / "release" / "revisions.json"
SCAN_DIRS = ("release", "configs", "mulligan", "paper", "tests")
# Not scanned: the frozen deployed Arena data pins the copy revisions the live Arena
# reads (docs/arena.md); test_arena_pins_are_released_repos and the network test
# test_arena_pins_resolve_anonymously check them instead.
ARENA_DATA = "arena/data"
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".csv", ".sh", ".md", ".txt", ".cfg"}
MAX_BYTES = 20 * 1024 * 1024

REPO_ID = r"mulligan/[A-Za-z0-9][A-Za-z0-9._-]*"
REPO_RE = re.compile(rf"^{REPO_ID}$")
REV_VALUE = re.compile(r"^(?:[0-9a-f]{7,40}|release-[A-Za-z0-9._-]+)$")
REPO_SUFFIXES = ("repo_id", "repo", "dataset_id", "dataset")
TEXT_PATTERNS = (
    re.compile(rf"(?P<repo>{REPO_ID})@(?P<rev>[0-9a-f]{{7,40}}|release-[A-Za-z0-9._-]+)"),
    re.compile(
        rf"huggingface\.co/(?:datasets/)?(?P<repo>{REPO_ID})/(?:resolve|tree|blob)/"
        r"(?P<rev>[0-9a-f]{7,40}|release-[A-Za-z0-9._-]+)"
    ),
    re.compile(
        rf"(?P<repo>{REPO_ID})\b[^\n]*?\brevision\s*[=:]\s*[\"']?"
        r"(?P<rev>[0-9a-f]{7,40}|release-[A-Za-z0-9._-]+)"
    ),
)

# (path glob, key that is exempt or None for the whole file, reason)
EXEMPT = (
    (
        "release/revisions.json",
        None,
        "the lookup itself",
    ),
    (
        "tests/release/test_revision_consistency.py",
        None,
        "fixtures of this test",
    ),
)


@dataclass(frozen=True)
class Pin:
    path: str
    where: str
    repo: str
    revision: str


def load_canonical() -> dict[str, dict]:
    return json.loads(REVISIONS.read_text())["repos"]


def repo_prefix(key: str) -> str | None:
    k = key.lower()
    for suffix in REPO_SUFFIXES:
        if k == suffix or k == "id":
            return ""
        if k.endswith("_" + suffix):
            return k[: -len(suffix) - 1]
    return None


def rev_prefix(key: str) -> str | None:
    k = key.lower()
    if k == "revision":
        return ""
    if k.endswith("_revision"):
        return k[: -len("_revision")]
    return None


def is_repo(value) -> bool:
    return isinstance(value, str) and bool(REPO_RE.match(value))


def is_rev(value) -> bool:
    return isinstance(value, str) and bool(REV_VALUE.match(value))


def pairs_from_keys(items: list[tuple[str, object]], path: str, where: str) -> list[Pin]:
    """Pair repo keys with revision keys of the same prefix among one scope's items."""
    repos = {}
    revs = {}
    for key, value in items:
        if not isinstance(key, str):
            continue
        p = repo_prefix(key)
        if p is not None and is_repo(value):
            repos[p] = value
        p = rev_prefix(key)
        if p is not None and is_rev(value):
            revs[p] = (key, value)
    pins = [Pin(path, f"{where}.{revs[p][0]}", repos[p], revs[p][1]) for p in repos if p in revs]
    # A scope that names one repo of its own (besides `parent_*` repos) pins it with every other revision key
    # (e.g. `release_revision`, `v3_tag_revision`).
    own = {v for p, v in repos.items() if not p.startswith("parent")}
    if len(own) == 1:
        (repo,) = own
        pins += [
            Pin(path, f"{where}.{key}", repo, rev)
            for p, (key, rev) in revs.items()
            if p not in repos
        ]
    return pins


def walk_structured(obj, path: str, where: str = "$") -> list[Pin]:
    pins = []
    if isinstance(obj, dict):
        pins += pairs_from_keys(list(obj.items()), path, where)
        for key, value in obj.items():
            if is_repo(key) and is_rev(value):
                pins.append(Pin(path, f"{where}[{key}]", key, value))
            elif is_repo(key) and isinstance(value, dict) and is_rev(value.get("revision")):
                pins.append(Pin(path, f"{where}[{key}].revision", key, value["revision"]))
            pins += walk_structured(value, path, f"{where}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            pins += walk_structured(value, path, f"{where}[{i}]")
    return pins


def literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def assigned_name(target) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def walk_python(source: str, path: str) -> list[Pin]:
    tree = ast.parse(source)
    pins = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            items = [(literal(k), literal(v)) for k, v in zip(node.keys, node.values) if k]
            pins += pairs_from_keys(items, path, f"line {node.lineno}")
            for k, v in items:
                if is_repo(k) and is_rev(v):
                    pins.append(Pin(path, f"line {node.lineno}", k, v))
        elif isinstance(node, ast.Call):
            items = [(kw.arg, literal(kw.value)) for kw in node.keywords if kw.arg]
            if node.args and is_repo(literal(node.args[0])):
                items.append(("repo", literal(node.args[0])))
            pins += pairs_from_keys(items, path, f"line {node.lineno}")
        body = getattr(node, "body", None)
        if isinstance(body, list):
            items = []
            for stmt in body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    name = assigned_name(stmt.targets[0])
                    items.append((name, literal(stmt.value)))
                elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    items.append((assigned_name(stmt.target), literal(stmt.value)))
            pins += pairs_from_keys(items, path, f"scope line {getattr(node, 'lineno', 1)}")
    return pins


def walk_text(text: str, path: str) -> list[Pin]:
    pins = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern in TEXT_PATTERNS:
            for m in pattern.finditer(line):
                pins.append(Pin(path, f"line {lineno}", m["repo"], m["rev"]))
    return pins


def parse_structured(path: Path, text: str):
    suffix = path.suffix
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        return list(yaml.safe_load_all(text))
    if suffix == ".toml":
        return tomllib.loads(text)
    if suffix == ".csv":
        return list(csv.DictReader(io.StringIO(text)))
    return None


def exemption(rel: str, where: str) -> str | None:
    for glob, key, reason in EXEMPT:
        if not fnmatch.fnmatch(rel, glob):
            continue
        if key is None or where.rsplit(".", 1)[-1] == key:
            return reason
    return None


def pins_in_file(path: Path, root: Path) -> list[Pin]:
    rel = path.relative_to(root).as_posix()
    text = path.read_text(errors="replace")
    if "mulligan/" not in text:
        return []
    pins = walk_text(text, rel)
    if path.suffix == ".py":
        pins += walk_python(text, rel)
    else:
        data = parse_structured(path, text)
        if data is not None:
            pins += walk_structured(data, rel)
    # One pin per (repo, revision) and file: repeats and overlapping matchers collapse.
    return list({(pin.repo, pin.revision): pin for pin in reversed(pins)}.values())


def check_pin(pin: Pin, canonical: dict[str, dict]) -> str | None:
    entry = canonical.get(pin.repo)
    if entry is None:
        return f"{pin.path} {pin.where}: {pin.repo}@{pin.revision} is not in release/revisions.json"
    rev = pin.revision
    if rev == entry["revision"] or (entry["tag"] and rev == entry["tag"]):
        return None
    if re.fullmatch(r"[0-9a-f]{7,39}", rev) and entry["revision"].startswith(rev):
        return None
    return f"{pin.path} {pin.where}: {pin.repo} pinned at {rev}, canonical {entry['revision']}" + (
        f" (tag {entry['tag']})" if entry["tag"] else ""
    )


def _repo_files(root: Path, top: str) -> list[Path]:
    """Tracked and untracked-but-not-ignored files under ``top`` (build caches are gitignored).

    Outside a git checkout (the scanner's own unit tests use a tmp dir) every file is scanned.
    """
    if not (root / ".git").exists():
        return sorted((root / top).rglob("*"))
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", top],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(root / rel for rel in out.split("\0") if rel)


def scan(root: Path, canonical: dict[str, dict]) -> tuple[list[str], int]:
    failures, checked = [], 0
    for top in SCAN_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for path in _repo_files(root, top):
            if (
                not path.is_file()
                or path.suffix not in TEXT_SUFFIXES
                or path.stat().st_size > MAX_BYTES
                or "__pycache__" in path.parts
            ):
                continue
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, g) and k is None for g, k, _ in EXEMPT):
                continue
            for pin in pins_in_file(path, root):
                if exemption(rel, pin.where):
                    continue
                checked += 1
                problem = check_pin(pin, canonical)
                if problem:
                    failures.append(problem)
    return failures, checked


def test_revisions_json_shape():
    data = json.loads(REVISIONS.read_text())
    repos = data["repos"]
    assert data["counts"] == {"dataset": 223, "model": 180, "evidence": 1}
    for repo, entry in repos.items():
        assert REPO_RE.match(repo), repo
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), repo
        if entry["type"] in ("dataset", "evidence"):
            assert entry["tag"] == data["dataset_tag"], repo
        else:
            assert entry["type"] == "model", repo
            assert entry["tag"] == data["model_tag"], repo
            assert ("pending" in entry) == (entry["tag"] is None), repo


def test_no_stale_pins():
    failures, checked = scan(REPO, load_canonical())
    assert checked > 0
    assert not failures, f"{len(failures)} non-canonical pins:\n" + "\n".join(failures)


def arena_pins() -> list[Pin]:
    pins = []
    for path in _repo_files(REPO, ARENA_DATA):
        if path.suffix in (".json", ".csv"):
            pins += pins_in_file(path, REPO)
    return pins


def test_arena_pins_are_released_repos():
    """The Arena's frozen pins name released dataset repos (at earlier revisions, by design)."""
    pins = arena_pins()
    canonical = load_canonical()
    assert len({(p.repo, p.revision) for p in pins}) == 223  # one pin per repo: its single commit
    for pin in pins:
        assert canonical.get(pin.repo, {}).get("type") == "dataset", pin
        assert REV_VALUE.match(pin.revision) and len(pin.revision) == 40, pin


CANON = {
    "mulligan/a": {"revision": "a" * 40, "tag": "release-x"},
    "mulligan/m": {"revision": "b" * 40, "tag": None},
}


@pytest.mark.parametrize(
    ("name", "content", "bad"),
    [
        ("ok.json", {"repo": "mulligan/a", "revision": "a" * 40}, 0),
        ("tag.json", {"repo": "mulligan/a", "revision": "release-x"}, 0),
        ("stale.json", {"repo": "mulligan/a", "revision": "c" * 40}, 1),
        ("model_tag.json", {"repo": "mulligan/m", "revision": "release-x"}, 1),
        ("dest.json", {"destination_repo": "mulligan/a", "destination_revision": "c" * 40}, 1),
        ("map.json", {"dataset_revisions": {"mulligan/a": "c" * 40}}, 1),
        ("nested.json", {"repos": {"mulligan/a": {"revision": "c" * 40}}}, 1),
        ("unknown.json", {"repo": "mulligan/z", "revision": "a" * 40}, 1),
        ("short.json", {"repo": "mulligan/a", "revision": "aaaaaaa"}, 0),
        ("unpinned.json", {"repo": "mulligan/a", "revision": "main"}, 0),
    ],
)
def test_scanner_structured(tmp_path, name, content, bad):
    path = tmp_path / "release" / name
    path.parent.mkdir()
    path.write_text(json.dumps(content, indent=1))
    failures, _ = scan(tmp_path, CANON)
    assert len(failures) == bad, failures


@pytest.mark.parametrize(
    ("source", "bad"),
    [
        ('X = dict(repo="mulligan/a", revision="' + "c" * 40 + '")\n', 1),
        ('spec = Spec(demo_repo_id="mulligan/a", demo_revision="' + "c" * 40 + '")\n', 1),
        ('EVAL_REPO = "mulligan/a"\nEVAL_REVISION = "' + "c" * 40 + '"\n', 1),
        ('EVAL_REPO = "mulligan/a"\nEVAL_REVISION = "' + "a" * 40 + '"\n', 0),
        ('hf_hub_download("mulligan/a", "x", revision="' + "c" * 40 + '")\n', 1),
        ('URL = "https://huggingface.co/datasets/mulligan/a/resolve/' + "c" * 40 + '/x"\n', 1),
        ("# see mulligan/a@" + "c" * 12 + "\n", 1),
        ('REPO = "mulligan/a"  # no pin\n', 0),
    ],
)
def test_scanner_python(tmp_path, source, bad):
    path = tmp_path / "mulligan" / "mod.py"
    path.parent.mkdir()
    path.write_text(source)
    failures, _ = scan(tmp_path, CANON)
    assert len(failures) == bad, failures


def test_scanner_yaml(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "x.yaml").write_text(
        "datasets:\n  - repo: mulligan/a\n    revision: " + "c" * 40 + "\n"
    )
    failures, _ = scan(tmp_path, CANON)
    assert len(failures) == 1 and failures[0].startswith("configs/x.yaml"), failures


def resolve_commit(repo: str, repo_type: str, revision: str) -> str:
    """Commit a revision (sha or tag) resolves to, read anonymously through the file resolver.

    Uses a HEAD request on README.md (every release repo has one): the resolver endpoint has a
    much larger anonymous rate limit than the /api endpoints. Retries on HTTP 429.
    """
    import time

    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import HfHubHTTPError

    url = hf_hub_url(repo, "README.md", repo_type=repo_type, revision=revision)
    for attempt in range(4):
        try:
            return get_hf_file_metadata(url, token=False).commit_hash
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status != 429 or attempt == 3:
                raise
            time.sleep(int(e.response.headers.get("Retry-After", "60")) + 1)
    raise AssertionError("unreachable")


@pytest.mark.network
def test_every_revision_resolves_anonymously():
    """[CI, network] Each pinned revision exists on HF for an anonymous client; dataset tags point at it."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    from tests.conftest import RATE_LIMIT_MESSAGE, http_status

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    canonical = load_canonical()

    def check(item: tuple[str, dict]) -> str | None:
        repo, entry = item
        # the paper evidence is a dataset repo with its own manifest type
        hub_type = "dataset" if entry["type"] == "evidence" else entry["type"]
        try:
            got = resolve_commit(repo, hub_type, entry["revision"])
            if got != entry["revision"]:
                return f"{repo}: {entry['revision']} resolves to {got}"
            if entry["tag"]:
                got = resolve_commit(repo, hub_type, entry["tag"])
                if got != entry["revision"]:
                    return f"{repo}: tag {entry['tag']} -> {got}, pinned {entry['revision']}"
        except Exception as e:  # noqa: BLE001 - every failure is collected and reported
            if http_status(e) == 429:
                return f"{repo}@{entry['revision']}: {RATE_LIMIT_MESSAGE}"
            return f"{repo}@{entry['revision']}: {type(e).__name__}: {str(e).splitlines()[0]}"
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        problems = [p for p in pool.map(check, sorted(canonical.items())) if p]
    assert not problems, f"{len(problems)} of {len(canonical)} pins fail:\n" + "\n".join(problems)


@pytest.mark.network
def test_arena_pins_resolve_anonymously():
    """[network] Every revision the frozen Arena data pins still exists on HF (the Arena reads them)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    from tests.conftest import RATE_LIMIT_MESSAGE, http_status

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    def check(pin: tuple[str, str]) -> str | None:
        repo, revision = pin
        try:
            got = resolve_commit(repo, "dataset", revision)
        except Exception as e:  # noqa: BLE001 - every failure is collected and reported
            if http_status(e) == 429:
                return f"{repo}@{revision}: {RATE_LIMIT_MESSAGE}"
            return f"{repo}@{revision}: {type(e).__name__}: {str(e).splitlines()[0]}"
        return None if got == revision else f"{repo}: {revision} resolves to {got}"

    pins = sorted({(p.repo, p.revision) for p in arena_pins()})
    with ThreadPoolExecutor(max_workers=8) as pool:
        problems = [p for p in pool.map(check, pins) if p]
    assert not problems, f"{len(problems)} of {len(pins)} Arena pins fail:\n" + "\n".join(problems)
