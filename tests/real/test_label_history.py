"""Contract tests for the append-only label-provenance ledger."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mulligan.real.lifecycle.label_history import (
    SKIP_PAYLOAD,
    UNLABEL_PAYLOAD,
    append_events,
    diff_progress_events,
    make_event,
)

SOURCE = {"kind": "human", "agent": "reviewer", "tool": "web-review"}
EVIDENCE = {"apply_job": "job1", "pre_sha": "abc"}
TS = "2026-08-19T20:00:00+00:00"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _event(**kwargs):
    defaults = dict(
        label_kind="outcome",
        episode_index=1,
        payload={"new_outcome": "failure", "outcome_frame": 10, "soft_truncate": False},
        source=SOURCE,
        evidence=EVIDENCE,
        ts=TS,
    )
    defaults.update(kwargs)
    return make_event(**defaults)


class TestMakeEvent:
    def test_unknown_source_kind_raises(self):
        with pytest.raises(ValueError, match="unknown source kind"):
            _event(source={"kind": "gremlin", "tool": "x"})

    def test_missing_tool_raises(self):
        with pytest.raises(ValueError, match="source.tool"):
            _event(source={"kind": "human", "agent": "a"})

    def test_prev_and_taxonomy_only_present_when_given(self):
        bare = _event()
        assert "prev" not in bare and "taxonomy" not in bare
        full = _event(prev=dict(SKIP_PAYLOAD), taxonomy="stage-ladder-v2")
        assert full["prev"] == SKIP_PAYLOAD
        assert full["taxonomy"] == "stage-ladder-v2"


class TestLedgerFile:
    def test_append_roundtrip(self, tmp_path: Path):
        e1 = _event(
            episode_index=7,
            payload={"new_outcome": "failure", "outcome_frame": 600, "soft_truncate": False},
        )
        e2 = _event(
            episode_index=7,
            payload={"new_outcome": "failure", "outcome_frame": 599, "soft_truncate": False},
        )
        e3 = _event(
            episode_index=8, label_kind="stage", payload={"stages": ["S1"]}, taxonomy="ladder-v1"
        )
        append_events(tmp_path, [e1])
        path = append_events(tmp_path, [e2, e3])
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert events == [e1, e2, e3]

    def test_append_refuses_symlinked_ledger_under_python_O(self, tmp_path: Path):
        # The guard must survive `python -O` (asserts are stripped there): appending
        # through a hub-cache snapshot symlink would corrupt the shared blob.
        blob = tmp_path / "blob.jsonl"
        blob.write_text("")
        root = tmp_path / "snapshot"
        root.mkdir()
        (root / ".label_history.jsonl").symlink_to(blob)
        code = (
            "import sys; from pathlib import Path; "
            "from mulligan.real.lifecycle.label_history import append_events; "
            f"append_events(Path({str(root)!r}), [{{'x': 1}}])"
        )
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.run(
            [sys.executable, "-O", "-c", code], capture_output=True, text=True, env=env
        )
        assert proc.returncode != 0
        assert "hub-cache symlink" in proc.stderr
        assert blob.read_text() == ""


class TestDiffProgressEvents:
    def _diff(self, prev, curr):
        return diff_progress_events(prev, curr, source=SOURCE, evidence=EVIDENCE, ts=TS)

    def test_unchanged_records_emit_nothing(self):
        state = {
            "changed_episodes": {
                "1": {"new_outcome": "failure", "outcome_frame": 5, "soft_truncate": False}
            },
            "skipped_episodes": [2],
        }
        assert self._diff(state, state) == []

    def test_change_skip_upgrade_and_new_skip(self):
        prev = {
            "changed_episodes": {
                "7": {"new_outcome": "failure", "outcome_frame": 600, "soft_truncate": False}
            },
            "skipped_episodes": [50],
        }
        curr = {
            "changed_episodes": {
                "7": {"new_outcome": "failure", "outcome_frame": 599, "soft_truncate": False},
                "50": {"new_outcome": "failure", "outcome_frame": 374, "soft_truncate": False},
            },
            "skipped_episodes": [60],
        }
        events = self._diff(prev, curr)
        by_ep = {e["episode_index"]: e for e in events}
        assert set(by_ep) == {7, 50, 60}
        assert by_ep[7]["prev"]["outcome_frame"] == 600
        assert by_ep[50]["prev"] == SKIP_PAYLOAD
        assert by_ep[60]["payload"] == SKIP_PAYLOAD and "prev" not in by_ep[60]

    def test_removed_entries_become_unlabel_events(self):
        prev = {
            "changed_episodes": {
                "7": {"new_outcome": "failure", "outcome_frame": 1, "soft_truncate": False}
            },
            "skipped_episodes": [9],
        }
        curr = {"changed_episodes": {}, "skipped_episodes": []}
        events = self._diff(prev, curr)
        by_ep = {e["episode_index"]: e for e in events}
        assert set(by_ep) == {7, 9}
        assert by_ep[7]["payload"] == UNLABEL_PAYLOAD
        assert by_ep[7]["prev"]["outcome_frame"] == 1
        assert by_ep[9]["payload"] == UNLABEL_PAYLOAD
        assert by_ep[9]["prev"] == SKIP_PAYLOAD
