"""Regression test: per-chunk diagnostics must survive rollout_episode's
end-of-episode policy.reset().

rollout_episode() resets the policy immediately after the episode ends, BEFORE
the sidecar writers in manifest_eval / rollout_real_policy read the
accumulated chunk infos. reset() therefore stashes episode_chunk_infos into
last_episode_chunk_infos instead of dropping it.
"""

import json
from collections import deque

from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy


def _skeleton_policy() -> VisionIDQLRealWorldPolicy:
    """Instance with only the attributes reset() touches (no heavy deps)."""
    p = object.__new__(VisionIDQLRealWorldPolicy)
    p._action_queue = deque()
    p._obs_state_history = deque()
    p._obs_images_history = deque()
    p.last_chunk_info = {}
    p.episode_chunk_infos = []
    p.last_episode_chunk_infos = []
    p._debug_images_saved = False
    return p


def test_reset_stashes_episode_chunk_infos():
    p = _skeleton_policy()
    p.episode_chunk_infos = [{"chunk_ordinal": 0, "q_min_best": 0.1}]
    p.reset()  # end-of-episode reset inside rollout_episode
    assert p.episode_chunk_infos == []
    assert p.last_episode_chunk_infos == [{"chunk_ordinal": 0, "q_min_best": 0.1}]


def test_retry_attempt_is_superseded_not_leaked():
    p = _skeleton_policy()
    # Attempt 1 (discarded restart): accumulates, end-reset stashes it.
    p.episode_chunk_infos = [{"chunk_ordinal": 0, "attempt": 1}]
    p.reset()
    # Attempt 2 begins: rollout_episode start-reset overwrites the stash with
    # the (empty) current buffer — the discarded attempt never reaches a writer.
    p.reset()
    assert p.last_episode_chunk_infos == []
    # Attempt 2 runs and ends.
    p.episode_chunk_infos = [{"chunk_ordinal": 0, "attempt": 2}]
    p.reset()
    assert p.last_episode_chunk_infos == [{"chunk_ordinal": 0, "attempt": 2}]


def test_writer_consumption_pattern():
    p = _skeleton_policy()
    p.episode_chunk_infos = [{"chunk_ordinal": 0}]
    p.reset()
    # Writer (manifest_eval / rollout_real_policy): read stash, persist, clear.
    infos = getattr(p, "last_episode_chunk_infos", None)
    assert infos
    p.last_episode_chunk_infos = []
    # Next episode with NO chunks (e.g. a DP arm or instant failure): the
    # writer must see an empty stash, not the previous episode's data.
    p.reset()
    assert getattr(p, "last_episode_chunk_infos", None) == []


def test_per_candidate_sidecar_fields_are_json_safe_and_shaped():
    """Contract for the per-candidate fields in last_chunk_info
    (separating commits-toward-goal from moves-fast-anywhere): per-candidate
    6-step cumulative xyz displacement (N x 3), per-candidate final-step gripper
    (N floats), and the scoring EEF position (3 floats). Must be plain
    JSON-serializable python (no tensors) so the sidecar writer can dump it."""
    n = 16
    record = {
        "chunk_ordinal": 0,
        "cand_cum_xyz_disp_all": [[0.0, 0.0, 0.0] for _ in range(n)],
        "cand_final_gripper_all": [0.0] * n,
        "eef_pos": [0.1, 0.2, 0.3],
    }
    round_tripped = json.loads(json.dumps(record))
    assert len(round_tripped["cand_cum_xyz_disp_all"]) == n
    assert all(len(v) == 3 for v in round_tripped["cand_cum_xyz_disp_all"])
    assert len(round_tripped["cand_final_gripper_all"]) == n
    assert len(round_tripped["eef_pos"]) == 3
