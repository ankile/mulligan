"""Append-only label-provenance ledger (``.label_history.jsonl``) for HF datasets.

Every label mutation on a dataset (outcome edits, stage labels) is recorded
as one JSON event per line, appended in the same HF commit as the label change
it describes. Latest-state files
(``.outcome_edit_progress.json``, stage gold CSVs) stay pure decisions; this
ledger owns WHO/WHEN/HOW. Reconstructing provenance from HF commit
archaeology is possible but unusable in practice (tracing one episode took a
25-commit walk); the ledger makes it one file read.

Design rules:

- **Append-only.** Events are never rewritten or deleted. Corrections are new
  events; a reader resolves current state by last-event-wins.
- **Taxonomy-blind.** ``payload`` is opaque to this module and validated only
  by consumers at read time. Each event may carry a free-form ``taxonomy``
  ref (e.g. a stage-ladder version). When the stage vocabulary evolves
  (splitting/removing stages), old events remain interpretable via their
  taxonomy ref, and a remap is appended as new ``source.kind ==
  "migration"`` events — never a rewrite.
- **Sources are structured**: ``{"kind": auto|human|vlm|heuristic|migration,
  "agent": who/which model, "tool": which pipeline}``. ``evidence`` links the
  event to its apply job / battery run / repo shas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILENAME = ".label_history.jsonl"
EVENT_VERSION = 1
SOURCE_KINDS = ("auto", "human", "vlm", "heuristic", "migration")
# Payload used when a reviewer explicitly keeps existing labels as-is.
SKIP_PAYLOAD = {"action": "skip"}
# Payload used when a decision is retracted (entry removed from the record,
# e.g. an operator removing an episode from the outcome edit progress).
UNLABEL_PAYLOAD = {"action": "unlabel"}


def make_event(
    *,
    label_kind: str,
    episode_index: int,
    payload: dict,
    source: dict,
    evidence: dict,
    ts: str,
    prev: dict | None = None,
    taxonomy: str | None = None,
) -> dict:
    """Build one validated ledger event (payload/prev stay opaque)."""
    if not label_kind:
        raise ValueError("label_kind is required")
    if source["kind"] not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind {source['kind']!r}")
    if "tool" not in source:
        raise ValueError("source.tool is required")
    event = {
        "v": EVENT_VERSION,
        "label_kind": label_kind,
        "episode_index": int(episode_index),
        "payload": payload,
        "source": source,
        "evidence": evidence,
        "ts": ts,
    }
    if prev is not None:
        event["prev"] = prev
    if taxonomy is not None:
        event["taxonomy"] = taxonomy
    return event


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def diff_progress_events(
    prev: dict, curr: dict, *, source: dict, evidence: dict, ts: str
) -> list[dict]:
    """Events for one progress-record transition. Unchanged records emit
    nothing; entries removed outright become ``unlabel`` events (a decision
    retraction)."""
    events = []
    prev_changed = prev.get("changed_episodes", {})
    curr_changed = curr.get("changed_episodes", {})
    prev_skipped = set(prev.get("skipped_episodes", []))
    curr_skipped = set(curr.get("skipped_episodes", []))

    for ep_str, record in curr_changed.items():
        if prev_changed.get(ep_str) == record:
            continue
        ep_idx = int(ep_str)
        prev_state = prev_changed.get(ep_str)
        if prev_state is None and ep_idx in prev_skipped:
            prev_state = dict(SKIP_PAYLOAD)
        events.append(
            make_event(
                label_kind="outcome",
                episode_index=ep_idx,
                payload=dict(record),
                prev=prev_state,
                source=source,
                evidence=evidence,
                ts=ts,
            )
        )
    for ep_idx in sorted(curr_skipped - prev_skipped):
        events.append(
            make_event(
                label_kind="outcome",
                episode_index=int(ep_idx),
                payload=dict(SKIP_PAYLOAD),
                prev=prev_changed.get(str(ep_idx)),
                source=source,
                evidence=evidence,
                ts=ts,
            )
        )
    for ep_str in sorted(set(prev_changed) - set(curr_changed), key=int):
        if int(ep_str) in curr_skipped:
            continue  # changed -> skipped transition, already emitted above
        events.append(
            make_event(
                label_kind="outcome",
                episode_index=int(ep_str),
                payload=dict(UNLABEL_PAYLOAD),
                prev=dict(prev_changed[ep_str]),
                source=source,
                evidence=evidence,
                ts=ts,
            )
        )
    for ep_idx in sorted(prev_skipped - curr_skipped):
        if str(ep_idx) in curr_changed:
            continue  # skipped -> changed transition, already emitted above
        events.append(
            make_event(
                label_kind="outcome",
                episode_index=int(ep_idx),
                payload=dict(UNLABEL_PAYLOAD),
                prev=dict(SKIP_PAYLOAD),
                source=source,
                evidence=evidence,
                ts=ts,
            )
        )
    return sorted(events, key=lambda e: e["episode_index"])


def append_events(dataset_root: Path, events: list[dict]) -> Path:
    """Append events to the dataset-root ledger (created on first use)."""
    path = Path(dataset_root) / HISTORY_FILENAME
    # Refuse to write through a hub-cache snapshot symlink: the target blob is
    # shared by every revision with identical content, so an append would
    # corrupt history for all of them.
    if path.is_symlink():
        raise RuntimeError(f"{path} is a hub-cache symlink; refuse to append")
    with open(path, "a") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    return path
