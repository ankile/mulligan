"""The Arena task-spec exporter writes exactly what the Convex mutations and the review UI accept.

The row shapes are checked against the argument validators of ``taskSpecs:upsert`` and
``stageTaskSpecs:upsert`` (parsed from ``arena/convex``), the ``taskSpecs`` table in
``arena/convex/schema.ts`` and the server-side checks those mutations run. The stage-spec
documents are compared with the specs in ``arena/tests/fixtures/stage-consistency-fixtures.json``,
the vectors the Arena's TypeScript consistency checker is tested against.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from mulligan.real.lifecycle.tasks import registered_task_specs
from mulligan.real.stage_specs import registered_label_specs
from mulligan.tools import export_arena_task_specs as exporter

CONVEX = Path(__file__).resolve().parents[2] / "arena" / "convex"
FIXTURES = CONVEX.parent / "tests" / "fixtures" / "stage-consistency-fixtures.json"


def _split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    return [*parts, tail] if tail else parts


def _parse_validator(expr: str) -> tuple:
    """``v.record(v.string(), v.array(v.int64()))`` -> ``("record", ("string",), ("array", ("int64",)))``."""
    match = re.fullmatch(r"v\.(\w+)\((.*)\)", expr.strip(), flags=re.S)
    assert match, f"unsupported validator {expr!r}"
    kind, inner = match.groups()
    return (kind, *(_parse_validator(arg) for arg in _split_top_level(inner)))


def _block(text: str, opener: str) -> str:
    """The balanced ``{...}`` that follows ``opener``."""
    start = text.index(opener) + len(opener) - 1
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1 : i]
    raise AssertionError(f"unbalanced block after {opener!r}")


def _fields(block: str) -> dict[str, tuple]:
    block = re.sub(r"//[^\n]*", "", block)
    fields = {}
    for entry in _split_top_level(block):
        name, expr = entry.split(":", 1)
        fields[name.strip()] = _parse_validator(expr)
    return fields


def _mutation_args(module: str) -> dict[str, tuple]:
    text = (CONVEX / f"{module}.ts").read_text()
    upsert = text[text.index("export const upsert = mutation({") :]
    args = _fields(_block(upsert, "args: {"))
    assert args.pop("serviceToken") == ("optional", ("string",))
    return args


def _conforms(value, validator: tuple) -> bool:
    kind, *inner = validator
    if kind == "optional":
        return value is None or _conforms(value, inner[0])
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind in ("int64", "float64", "number"):
        integral = isinstance(value, int) and not isinstance(value, bool)
        return integral if kind == "int64" else integral or isinstance(value, float)
    if kind == "any":
        return True
    if kind == "array":
        return isinstance(value, list) and all(_conforms(v, inner[0]) for v in value)
    if kind == "record":
        return isinstance(value, dict) and all(
            _conforms(k, inner[0]) and _conforms(v, inner[1]) for k, v in value.items()
        )
    raise AssertionError(f"validator kind {kind!r} is not handled by this test")


def _assert_matches_args(row: dict, args: dict[str, tuple]) -> None:
    required = {name for name, validator in args.items() if validator[0] != "optional"}
    assert required <= set(row) <= set(args)
    for name, value in row.items():
        assert _conforms(value, args[name]), (name, value, args[name])
    # The exported file is plain JSON.
    assert json.loads(json.dumps(row)) == row


@pytest.fixture(scope="module")
def export() -> dict:
    return exporter.export_task_specs()


def test_exports_every_registered_task(export):
    assert [row["task"] for row in export["task_specs"]] == [
        exporter.arena_task_name(spec.name) for spec in registered_task_specs()
    ]
    assert [row["task"] for row in export["stage_task_specs"]] == [
        exporter.arena_task_name(spec.name) for spec in registered_label_specs()
    ]
    assert exporter.arena_task_name("routing_d2") == "routing_d2"


def test_task_spec_rows_match_the_upsert_validators_and_checks(export):
    args = _mutation_args("taskSpecs")
    schema = (CONVEX / "schema.ts").read_text()
    table = _fields(_block(schema, "taskSpecs: defineTable({"))
    # The table stores the mutation arguments plus the server's export time.
    assert set(table) == set(args) | {"exported_at"}
    for row in export["task_specs"]:
        _assert_matches_args(row, args)
        # The checks taskSpecs:upsert runs before writing.
        height, width = row["stored_frame_hw"]
        for role, (x0, y0, x1, y1) in row["crop_boxes"].items():
            assert role in row["camera_keys_by_role"]
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        assert set(row["review_camera_roles"]) <= set(row["camera_keys_by_role"])


def test_stage_task_spec_rows_match_the_upsert_validators_and_checks(export):
    args = _mutation_args("stageTaskSpecs")
    text = (CONVEX / "stageTaskSpecs.ts").read_text()
    listing = re.search(r"REQUIRED_SPEC_KEYS = \[(.*?)\]", text, flags=re.S)
    assert listing is not None
    required_keys = re.findall(r'"(\w+)"', listing.group(1))
    assert "taxonomy_hash" in required_keys
    for row in export["stage_task_specs"]:
        _assert_matches_args(row, args)
        spec = row["spec"]
        assert set(required_keys) <= set(spec)
        assert (spec["task"], spec["taxonomy_version"], spec["taxonomy_hash"]) == (
            row["task"],
            row["taxonomy_version"],
            row["taxonomy_hash"],
        )
        assert row["live"] is True


def test_stage_specs_equal_the_ui_consistency_fixtures(export):
    fixtures = json.loads(FIXTURES.read_text())
    for spec, row in zip(registered_label_specs(), export["stage_task_specs"], strict=True):
        expected = fixtures[f"{spec.name}@{spec.taxonomy_version}"]["spec"]
        assert {**row["spec"], "task": expected["task"]} == expected


def test_out_writes_the_rows_without_a_deployment(tmp_path, capsys):
    out = tmp_path / "specs.json"
    assert exporter.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text()) == exporter.export_task_specs()
    assert "Wrote 3 task specs and 3 stage task specs" in capsys.readouterr().out


def test_requires_an_output():
    with pytest.raises(SystemExit):
        exporter.main([])


def test_upload_sends_every_row_through_the_client(export):
    class Client:
        def __init__(self):
            self.calls = []

        def upsert_task_spec(self, **row):
            self.calls.append(("taskSpecs:upsert", row))
            return f"task-{len(self.calls)}"

        def upsert_stage_task_spec(self, **row):
            self.calls.append(("stageTaskSpecs:upsert", row))
            return f"stage-{len(self.calls)}"

    client = Client()
    ids = exporter.upload(export, client)
    assert len(ids) == len(export["task_specs"]) + len(export["stage_task_specs"])
    assert [row for _, row in client.calls] == export["task_specs"] + export["stage_task_specs"]


def test_rows_match_the_python_client_keywords(export):
    # Parsed, not imported: the policy-arena client needs the `convex` package.
    source = (CONVEX.parent / "python" / "policy_arena" / "client.py").read_text()
    methods = {
        node.name: {arg.arg for arg in node.args.kwonlyargs}
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }
    for row in export["task_specs"]:
        assert set(row) == methods["upsert_task_spec"]
    for row in export["stage_task_specs"]:
        assert set(row) == methods["upsert_stage_task_spec"]
