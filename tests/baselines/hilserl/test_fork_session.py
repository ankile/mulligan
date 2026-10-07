"""Fork an older checkpoint from a newer learner state, then resume the fork."""

import json
import pickle

import numpy as np
import pytest

from mulligan.baselines.hilserl.session import EpisodeRecord, Session, append_jsonl
from mulligan.baselines.hilserl.tools.fork_session import fork
from tests.baselines.hilserl.test_hilserl_jax import _episode_arrays, _fake_demos, _rec, _tiny_cfg


@pytest.fixture
def session(tmp_path):
    from mulligan.baselines.hilserl.learner import Learner

    cfg, demos = _tiny_cfg(eval_interval=6), _fake_demos()
    src = Session(tmp_path / "src")
    learner = Learner(cfg, src, demos=demos)
    for ep in range(2):
        arrays = _episode_arrays(6, intervened_from=4)
        arrays["observations"] += ep * 10
        arrays["next_observations"] += ep * 10
        record = _rec(ep, 6, True, 2)
        record["env_steps_before"] = ep * 6
        src.write_episode(EpisodeRecord(**record), arrays, np.zeros(1))
        learner.ingest_episode(ep, arrays, record)
        learner.update_once()
        if ep == 0:
            learner.save_checkpoint(6)
        append_jsonl(src.eval_ledger, {"env_step": (ep + 1) * 6, "success_rate": ep / 2})
    learner.save_resume("fixture")
    return src, cfg, demos


def test_fork_restores_checkpoint_prefix_and_resumes(session, tmp_path):
    import jax
    from flax import serialization

    from mulligan.baselines.hilserl.learner import Learner

    src, cfg, demos = session
    dst = Session(tmp_path / "fork")
    original = src.state_path.read_bytes()
    fork(src.root, dst.root, 6)
    meta = json.loads((src.checkpoints_dir / "step_0000006/meta.json").read_text())
    resumed = Learner(cfg, dst, demos=demos)
    assert resumed.load_resume()
    expected = serialization.from_bytes(
        resumed.agent, (src.checkpoints_dir / "step_0000006/agent.msgpack").read_bytes()
    )
    for actual, wanted in zip(
        jax.tree_util.tree_leaves(resumed.agent), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual, wanted)
    assert resumed.counters == meta["counters"]
    assert resumed.ingested_ids == {0} and dst.next_episode_id() == 1
    assert resumed.next_milestone == 12
    assert len(resumed.online) == 6 and len(resumed.demo_buf) == len(demos["actions"]) + 2
    for key in demos:
        np.testing.assert_array_equal(
            resumed.online.dataset_dict[key][:6], src.load_episode(0)[key]
        )
        np.testing.assert_array_equal(
            resumed.demo_buf.dataset_dict[key][: len(demos[key])], demos[key]
        )
        np.testing.assert_array_equal(
            resumed.demo_buf.dataset_dict[key][len(demos[key]) : len(demos[key]) + 2],
            src.load_episode(0)[key][4:],
        )
    assert [json.loads(line)["env_step"] for line in dst.eval_ledger.read_text().splitlines()] == [
        6
    ]
    assert pickle.loads(dst.state_path.read_bytes())["wandb_run_id"] is None
    # Continue from the fork's next ID, with neither replay duplication nor source mutation.
    arrays = _episode_arrays(3)
    record = _rec(1, 3, False, 0)
    record["env_steps_before"] = 6
    dst.write_episode(EpisodeRecord(**record), arrays, np.zeros(1))
    assert resumed.ingest_episode(1, arrays, record)
    assert not resumed.ingest_episode(1, arrays, record)
    assert len(resumed.online) == 9 and dst.next_episode_id() == 2
    assert src.state_path.read_bytes() == original and src.next_episode_id() == 2


@pytest.mark.parametrize("problem", ["existing", "ledger", "buffer"])
def test_fork_rejects_inconsistent_source_or_existing_destination(session, tmp_path, problem):
    src, _, _ = session
    dst = tmp_path / "fork"
    if problem == "existing":
        dst.mkdir()
        (dst / "keep").write_text("untouched")
        error, match = FileExistsError, "exists"
    elif problem == "ledger":
        rows = [json.loads(line) for line in src.actor_ledger.read_text().splitlines()]
        rows[0]["success"] = False
        src.actor_ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
        error, match = RuntimeError, "counters"
    else:
        state = pickle.loads(src.state_path.read_bytes())
        state["online"]["data"]["rewards"][0] = 99
        src.state_path.write_bytes(pickle.dumps(state))
        error, match = RuntimeError, "differ"
    with pytest.raises(error, match=match):
        fork(src.root, dst, 6)
    if problem == "existing":
        assert (dst / "keep").read_text() == "untouched"
    else:
        assert not dst.exists()
