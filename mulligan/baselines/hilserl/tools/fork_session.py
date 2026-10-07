"""Fork a finished/running session at one of its eval checkpoints into a new session.

The new session starts exactly where the source stood when it wrote
``learner/checkpoints/step_<step>/``: the checkpoint's agent (actor, critics,
temperature, optimizer states), the online buffer truncated to the checkpoint's
``env_steps`` rows, the demo buffer truncated to the base demos plus the
checkpoint's ``demo_added`` rows, the checkpoint's counters, and the actor's
episode log up to the same env step. A learner and an actor started on the new
session resume from there (``Learner.load_resume`` / ``Session.next_episode_id``)
under a new W&B run.

The source session root must be staged locally with the layout of ``session.py``::

    <src>/actor/{ledger.jsonl,episodes/}           the source actor's episode log
    <src>/learner/state.pkl                         the source learner's LATEST resume state
    <src>/learner/checkpoints/step_<step>/          agent.msgpack + meta.json
    <src>/eval/ledger.jsonl                         the source eval ledger

The online rows in ``state.pkl`` are checked byte-for-byte against the copied
episodes, and every counter is recomputed from the copied ledger, so a fork whose
parts disagree fails here rather than in the learner.

    uv run python -m mulligan.baselines.hilserl.tools.fork_session --src SRC --dst DST --step 150000
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from mulligan.baselines.hilserl.replay import TRANSITION_KEYS
from mulligan.baselines.hilserl.session import (
    Session,
    append_jsonl,
    read_jsonl,
    write_json_atomic,
)

LEDGER_COUNTERS = (
    "episodes",
    "successes",
    "bouts",
    "fail_terminated",
    "idle_dropped",
    "redo_count",
    "intervened_steps",
    "env_steps",
)


def fork(src_root: Path, dst_root: Path, step: int) -> dict:
    src, dst = Session(src_root), Session(dst_root)
    if dst.root.exists():
        raise FileExistsError(f"{dst.root} exists; a fork must create a new session")
    ckpt = src.checkpoints_dir / f"step_{step:07d}"
    meta = json.loads((ckpt / "meta.json").read_text())
    counters = meta["counters"]
    n_rows = counters["env_steps"]

    with open(src.state_path, "rb") as f:
        st = pickle.load(f)

    # --- actor episode log: the ledger prefix that fills exactly n_rows env steps ---------
    records = src.read_ledger()
    prefix, total = [], 0
    for r in records:
        if total >= n_rows:
            break
        prefix.append(r)
        total += r.length
    if total != n_rows:
        raise RuntimeError(f"no ledger prefix sums to {n_rows} env steps (closest {total})")

    online = st["online"]
    for k in TRANSITION_KEYS:
        rows = np.concatenate([src.load_episode(r.episode_id)[k] for r in prefix]).astype(
            online["data"][k].dtype
        )
        if not np.array_equal(rows, online["data"][k][:n_rows]):
            raise RuntimeError(
                f"online buffer rows [:{n_rows}] of {k!r} differ from the episode log"
            )

    recomputed = {
        "episodes": len(prefix),
        "successes": sum(int(r.success) for r in prefix),
        "bouts": sum(r.intervention_count for r in prefix),
        "fail_terminated": sum(int(r.fail_terminated) for r in prefix),
        "idle_dropped": sum(r.idle_steps_dropped for r in prefix),
        "redo_count": sum(r.redo_count for r in prefix),
        "intervened_steps": sum(r.intervention_steps for r in prefix),
        "env_steps": total,
    }
    mismatch = {
        k: (recomputed[k], counters[k]) for k in LEDGER_COUNTERS if recomputed[k] != counters[k]
    }
    if mismatch:
        raise RuntimeError(f"ledger prefix vs checkpoint counters (ledger, ckpt): {mismatch}")
    ids = [r.episode_id for r in prefix]
    if not set(ids) <= set(st["ingested_ids"]):
        raise RuntimeError("the source learner never ingested some of the prefix episodes")

    # --- buffers: rows are appended in ingest order and never wrap below max_steps + horizon ----
    demo = st["demo"]
    for name, buf in (("online", online), ("demo", demo)):
        if buf["insert_index"] != buf["size"]:
            raise RuntimeError(
                f"source {name} buffer wrapped (insert_index {buf['insert_index']} != size {buf['size']})"
            )
    n_base_demo = demo["size"] - st["counters"]["demo_added"]
    n_demo = n_base_demo + counters["demo_added"]

    def truncated(buf: dict, n: int) -> dict:
        return {
            "data": {k: buf["data"][k][:n].copy() for k in TRANSITION_KEYS},
            "size": n,
            "insert_index": n,
            "rng_state": buf["rng_state"],
        }

    rec = [(float(r.success), r.intervention_steps / max(1, r.length)) for r in prefix[-20:]]
    state = {
        "agent": (ckpt / "agent.msgpack").read_bytes(),
        "online": truncated(online, n_rows),
        "demo": truncated(demo, n_demo),
        "counters": dict(counters),
        "ingested_ids": ids,
        "param_version": meta["param_version"],
        "next_milestone": step + meta["config"]["eval_interval"],
        "recent": rec,
        "config_hash": st["config_hash"],
        "wandb_run_id": None,
        "saved_at": time.time(),
    }

    provenance = {
        "forked_from": src.root.name,
        "step": step,
        "checkpoint_repo_sha": meta["repo_sha"],
        "counters": counters,
        "param_version": meta["param_version"],
        "demo_rows": n_demo,
        "base_demo_rows": n_base_demo,
        "episodes": [ids[0], ids[-1]],
        "created_at": time.time(),
    }

    # --- write the new session --------------------------------------------------------------
    dst.episodes_dir.mkdir(parents=True)
    dst.learner_dir.mkdir(parents=True)
    dst.eval_dir.mkdir(parents=True)
    for r in prefix:
        shutil.copy2(src.episode_path(r.episode_id), dst.episode_path(r.episode_id))
        row = json.loads(json.dumps(r.__dict__))
        row["extra"]["forked_from"] = src.root.name
        append_jsonl(dst.actor_ledger, row)
    with open(dst.state_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    for row in read_jsonl(src.eval_ledger):
        if row["env_step"] <= step:
            append_jsonl(dst.eval_ledger, row | {"forked_from": src.root.name})
    write_json_atomic(dst.learner_dir / "forked_from.json", provenance)
    if dst.next_episode_id() != ids[-1] + 1 or dst.env_steps_logged() != n_rows:
        raise RuntimeError("the written actor log does not resume at the fork point")
    return provenance


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, required=True, help="locally staged source session root")
    p.add_argument("--dst", type=Path, required=True, help="new session root (must not exist)")
    p.add_argument("--step", type=int, required=True, help="eval checkpoint env step to fork at")
    args = p.parse_args()
    print(json.dumps(fork(args.src, args.dst, args.step), indent=2))


if __name__ == "__main__":
    main()
