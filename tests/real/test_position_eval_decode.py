"""Unit tests for the DP actor's deploy action path in ``LeRobotRealWorldPolicy``.

- velocity finalization (the unconditional [-1, 1] clip);
- the action contract the constructor accepts (base-frame cartesian_velocity, 7D state)
  and the clear errors for the source code's other targets, frames and state widths;
- action-stat extraction;
- the UMI relative-pose chunk path (held-anchor decode, clip, gripper clamp);
- which cameras ``build_processed_obs`` consumes.

The parts that REQUIRE the robot host (DROID cartesian_position units, gripper
convention) are NOT tested here.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mulligan.real.policy.loader import (
    LeRobotRealWorldPolicy,
    _extract_action_stat_bounds,
    resolve_policy_action_contract,
)


# --------------------------------------------------------------------------- #
# Lightweight stubs so we can exercise predict()'s finalization without a real
# LeRobot policy / robot. We bypass __init__ (which needs a real policy) and set
# only the attributes the finalization path reads.
# --------------------------------------------------------------------------- #


def _make_policy():
    pol = LeRobotRealWorldPolicy.__new__(LeRobotRealWorldPolicy)
    pol.action_space = "cartesian_velocity"
    return pol


def _finalize(pol, action):
    """Drive the REAL finalization (predict()'s post-postprocessor tail)."""
    return pol._finalize_action(np.asarray(action, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Velocity branch: the unconditional clip.
# --------------------------------------------------------------------------- #


def test_velocity_branch_byte_identical_to_plain_clip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-3, 3, size=7).astype(np.float32)
        pol = _make_policy()
        out = _finalize(pol, a)
        np.testing.assert_array_equal(out, np.clip(a, -1.0, 1.0))


def test_real_predict_routes_through_finalize_action():
    # End-to-end: the REAL predict() must use _finalize_action (no parallel tail).
    # Stub the heavy internals (obs build, inference, postprocessor) to identity so
    # we exercise the dispatch + finalization on a known action vector.
    pol = _make_policy()
    raw7 = np.array([0.3, -2.0, 0.5, 1.5, 0.0, -0.1, 0.7], dtype=np.float32)

    pol._camera_keys = []
    pol._use_joint_positions = False
    pol.build_processed_obs = lambda *a, **k: {}  # type: ignore
    pol._policy = type("P", (), {"select_action": lambda self, o: torch.from_numpy(raw7)})()
    pol._postprocessor = lambda t: t  # identity un-normalize

    raw_obs = {
        "robot_state": {"cartesian_position": [0, 0, 0, 0, 0, 0], "gripper_position": 0.0},
        "image": {},
    }
    out = pol.predict(raw_obs)
    np.testing.assert_array_equal(out, np.clip(raw7, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Action-stat extraction helper.
# --------------------------------------------------------------------------- #


class _StubStep:
    def __init__(self, stats):
        self.stats = stats


class _StubPipeline:
    def __init__(self, steps):
        self.steps = steps


def test_extract_action_stat_bounds_from_pipeline_step():
    stats = {"action": {"min": [0.0, 1.0, 2.0], "max": [1.0, 2.0, 3.0]}}
    pipe = _StubPipeline([_StubStep({}), _StubStep(stats)])
    a_min, a_max = _extract_action_stat_bounds(pipe)
    np.testing.assert_allclose(a_min, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(a_max, [1.0, 2.0, 3.0])


def test_extract_action_stat_bounds_absent_returns_none():
    pipe = _StubPipeline([_StubStep({"observation.state": {"mean": [0.0]}})])
    assert _extract_action_stat_bounds(pipe) == (None, None)


def test_extract_action_stat_bounds_corrupt_max_lt_min_raises():
    stats = {"action": {"min": [1.0], "max": [0.0]}}
    pipe = _StubPipeline([_StubStep(stats)])
    with pytest.raises(ValueError, match="max < min"):
        _extract_action_stat_bounds(pipe)


# --------------------------------------------------------------------------- #
# Action contract: the constructor accepts base-frame cartesian_velocity on a 7D
# state and refuses everything else with a clear error. (Full ctor needs a real
# policy; stubs carry only what __init__ reads.)
# --------------------------------------------------------------------------- #


class _StubConfig:
    input_features = {}  # no joint_position -> _use_joint_positions False


class _Feature:
    def __init__(self, dim):
        self.shape = (dim,)


class _NoContractActionConfig:
    pass


class _EefVelocityActionConfig:
    action_target = "cartesian_velocity"
    cartesian_action_frame = "eef"


class _StubPolicy:
    def __init__(self, state_dim=None):
        self.config = _StubConfig()
        if state_dim is not None:
            self.config.input_features = {"observation.state": _Feature(state_dim)}


def _ctor(action_target="cartesian_velocity", cartesian_action_frame="base", state_dim=None):
    return LeRobotRealWorldPolicy(
        policy=_StubPolicy(state_dim),
        preprocessor=None,
        postprocessor=_StubPipeline(
            [_StubStep({"action": {"min": [0.0] * 10, "max": [1.0] * 10}})]
        ),
        camera_keys=[],
        action_target=action_target,
        cartesian_action_frame=cartesian_action_frame,
    )


def test_ctor_velocity_default_constructs():
    pol = _ctor()
    assert pol.action_space == "cartesian_velocity"
    assert pol.env_action_space == "cartesian_velocity"
    assert pol.gripper_action_space is None
    assert _ctor(state_dim=7).action_space == "cartesian_velocity"


def test_resolve_policy_action_contract_reads_config_or_default():
    assert resolve_policy_action_contract(_NoContractActionConfig()) == (
        "cartesian_velocity",
        "base",
    )
    assert resolve_policy_action_contract(_EefVelocityActionConfig()) == (
        "cartesian_velocity",
        "eef",
    )


@pytest.mark.parametrize(
    ("action_target", "frame"),
    [
        ("cartesian_position", "base"),
        ("cartesian_position_r6", "base"),
        ("cartesian_velocity", "eef"),
        ("nonsense", "base"),
    ],
)
def test_ctor_refuses_action_contracts_outside_the_release(action_target, frame):
    with pytest.raises(NotImplementedError, match="cartesian_velocity"):
        _ctor(action_target, frame)


@pytest.mark.parametrize("state_dim", [13, 10])
def test_ctor_refuses_state_widths_other_than_7(state_dim):
    # 13D was the source code's EE-velocity-augmented state; no released DP uses it.
    with pytest.raises(NotImplementedError, match=f"observation.state width {state_dim}"):
        _ctor(state_dim=state_dim)


def test_state_vector_is_7d_pose_and_gripper():
    obs = {
        "robot_state": {
            "cartesian_position": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            "gripper_position": 0.5,
        }
    }
    s = _make_policy()._build_state_vector(obs)
    assert s.dtype == np.float32
    np.testing.assert_allclose(s, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5])


# --------------------------------------------------------------------------- #
# UMI relative-pose LIVE rollout: chunk-level predict() path (held-anchor decode
# -> queue -> pop). The DROID cartesian_position step contract is NOT tested here
# (robot host); these cover the off-robot logic: chunk generation cadence, the
# [-1,1] safety clip, the held anchor, the gripper clamp, and that the per-action
# _finalize_action is never the relative path.
# --------------------------------------------------------------------------- #

_REL_T = 12  # horizon (per-timestep stat rows)
_REL_EXEC = 6  # n_action_steps (executed window the chunk API returns)
_REL_DIM = 10  # [rel_trans(3), rel_r6(6), grip(1)]


class _StubChunkPolicy:
    """Minimal lerobot-policy stand-in exposing the deployment chunk API."""

    def __init__(self, chunk: torch.Tensor):
        self._chunk = chunk  # (1, n_action_steps, 10) NORMALIZED
        self._queues: dict = {}
        self.predict_calls = 0
        self.config = SimpleNamespace(
            image_features={}, n_action_steps=chunk.shape[1], n_obs_steps=1
        )

    def predict_action_chunk(self, batch):  # noqa: ARG002 - batch unused by the stub
        self.predict_calls += 1
        return self._chunk

    def reset(self) -> None:
        self._queues = {}


def _make_relative_policy(chunk: torch.Tensor):
    """Relative-arm wrapper via __new__ wired to a stub chunk policy + (T,10) stats."""
    pol = LeRobotRealWorldPolicy.__new__(LeRobotRealWorldPolicy)
    pol.action_mode = "relative"
    pol.action_space = "cartesian_position"
    pol.gripper_action_space = "position"
    # Per-timestep (T,10) MIN_MAX stats: min=0, max=1 so un-norm = (x+1)/2 (in [0,1]
    # for a [-1,1] input). Shapes match the real relative checkpoint (12,10).
    pol._action_min = np.zeros((_REL_T, _REL_DIM), dtype=np.float32)
    pol._action_max = np.ones((_REL_T, _REL_DIM), dtype=np.float32)
    pol._relative_pose_queue = []
    pol._policy = _StubChunkPolicy(chunk)
    return pol


def _valid_chunk() -> torch.Tensor:
    """A (1, n_action_steps, 10) NORMALIZED chunk that decodes to a VALID rotation.

    Stats are min=0/max=1, so normalized = 2*physical - 1. We encode the identity r6
    [1,0,0,0,1,0] (two orthonormal columns -> Gram-Schmidt succeeds) as normalized
    [1,-1,-1,-1,1,-1]; trans/grip = physical 0 -> normalized -1. An all-equal r6
    (e.g. the all-zeros chunk -> physical 0.5 everywhere) is a DEGENERATE rotation and
    correctly raises in the decode, so the chunk content must be a real rotation.
    """
    row = [-1.0, -1.0, -1.0] + [1.0, -1.0, -1.0, -1.0, 1.0, -1.0] + [-1.0]
    return torch.tensor([[row] * _REL_EXEC], dtype=torch.float32)


def _oob_chunk(value: float = 5.0) -> torch.Tensor:
    return torch.full((1, _REL_EXEC, _REL_DIM), float(value), dtype=torch.float32)


def test_relative_predict_refills_chunk_every_n_action_steps():
    pol = _make_relative_policy(_valid_chunk())
    anchor = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    poses = [pol._predict_relative({}, anchor_pose6=anchor) for _ in range(2 * _REL_EXEC)]
    # Each pop is a 7D euler pose; the chunk generator runs exactly once per
    # n_action_steps pops (2 chunks over 2*n_action_steps calls) — the re-plan cadence.
    assert all(p.shape == (7,) and np.all(np.isfinite(p)) for p in poses)
    assert pol._policy.predict_calls == 2
    assert pol._relative_pose_queue == []  # fully drained after the last pop


def test_relative_chunk_clipped_to_unit_range_before_decode():
    # Policy head over-shoots the trained normalized range; the safety clip must bound
    # it to [-1,1] so the decoded displacement can't exceed the trained per-timestep max.
    pol = _make_relative_policy(_oob_chunk(5.0))
    clipped = pol._generate_relative_chunk_normalized({})
    assert clipped.shape == (_REL_EXEC, _REL_DIM)
    assert clipped.max() <= 1.0 + 1e-6 and clipped.min() >= -1.0 - 1e-6
    # +5 clamps to +1 -> un-norm to the per-timestep max (1.0 here) -> bounded.
    assert np.allclose(clipped, 1.0)


def test_relative_nonfinite_chunk_raises_before_decode():
    # np.clip passes NaN/inf through, so the [-1,1] safety clip alone would let a
    # non-finite prediction decode into a NaN cartesian_position robot command. The
    # chunk generator must refuse it loudly instead.
    nan_chunk = _valid_chunk().clone()
    nan_chunk[0, 2, 1] = float("nan")
    pol = _make_relative_policy(nan_chunk)
    with pytest.raises(RuntimeError, match="non-finite"):
        pol._generate_relative_chunk_normalized({})

    inf_chunk = _valid_chunk().clone()
    inf_chunk[0, 0, 0] = float("inf")
    pol = _make_relative_policy(inf_chunk)
    with pytest.raises(RuntimeError, match="non-finite"):
        pol._generate_relative_chunk_normalized({})


def test_relative_anchor_held_across_pops_not_reanchored():
    pol = _make_relative_policy(_valid_chunk())
    a0 = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    first = pol._predict_relative({}, anchor_pose6=a0)
    # A later pop with a DIFFERENT anchor must reuse the chunk decoded at refill time
    # (anchor captured once); re-anchoring per pop would double-count executed motion.
    moved = a0 + np.array([0.05, -0.05, 0.02, 0.0, 0.0, 0.0])
    second = pol._predict_relative({}, anchor_pose6=moved)
    assert pol._policy.predict_calls == 1  # no regen within the chunk
    # Both poses came from the SAME (a0-anchored) decode; pose[0] != pose[1] only via
    # the chunk's per-timestep content, not via the changed anchor.
    assert first.shape == (7,) and second.shape == (7,)


def test_relative_decode_anchors_on_live_proprio_each_chunk():
    # Proprio-anchored: the eval decode anchors each chunk on the current MEASURED proprio pose
    # (state[:6]) passed in — the training targets are relativized against proprio, so
    # deploy must too. With the identity/zero _valid_chunk (rel_trans=0, rel_rot=identity)
    # every decoded pose equals its anchor, so the anchor is directly observable.
    pol = _make_relative_policy(_valid_chunk())
    proprio_a = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    chunk1 = [pol._predict_relative({}, anchor_pose6=proprio_a) for _ in range(_REL_EXEC)]
    assert np.allclose(chunk1[0][:6], proprio_a)  # rel=0 -> pose == the proprio anchor
    # Refill chunk 2 at a NEW live proprio: the decode re-grounds on it (re-reads proprio
    # each replan), so the fresh chunk anchors on the new pose, not a chained command.
    proprio_b = np.array([0.6, -0.1, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
    chunk2_first = pol._predict_relative({}, anchor_pose6=proprio_b)
    assert pol._policy.predict_calls == 2  # a genuine refill happened
    assert np.allclose(chunk2_first[:6], proprio_b)  # re-grounded on the live proprio


def test_relative_gripper_clamped_to_unit_interval():
    # Gripper dim (index 9) huge -> un-norm clips at +1 -> decode passes grip through ->
    # output gripper (index 6) must be clamped to [0,1].
    chunk = _valid_chunk()
    chunk[..., 9] = 50.0
    pol = _make_relative_policy(chunk)
    pose = pol._predict_relative({}, anchor_pose6=np.zeros(6))
    assert 0.0 <= pose[6] <= 1.0


def test_relative_finalize_action_is_never_the_relative_path():
    # Defensive: per-action _finalize_action must refuse a relative policy (the chunk
    # path bypasses it). A regression that routed relative through here would raise.
    pol = _make_relative_policy(_valid_chunk())
    with pytest.raises(NotImplementedError, match="action_mode=relative"):
        pol._finalize_action(np.zeros(10, dtype=np.float32))


def test_relative_chunk_gen_drops_injected_none_action_before_queueing():
    """Regression: the preprocessor's transition->batch converter ALWAYS injects
    ACTION=None for an obs-only input, so ``processed_obs`` carries ``action=None``.
    ``_generate_relative_chunk_normalized`` must pop it before ``populate_queues``
    (mirroring ``DiffusionPolicy.select_action``); otherwise the None poisons the
    policy's ACTION deque and ``predict_action_chunk``'s online branch ``torch.stack``s
    ``[None]`` -> "expected Tensor ... got NoneType".

    Uses a queue-faithful stub that runs the SAME online-branch stack as the real
    DiffusionPolicy — the plain ``_StubChunkPolicy`` ignores the batch and keeps empty
    queues, so it cannot exercise (or catch) the queue poisoning.
    """
    from collections import deque

    from lerobot.utils.constants import ACTION, OBS_STATE

    chunk = _valid_chunk()

    class _QueueFaithfulPolicy:
        def __init__(self):
            self.config = SimpleNamespace(
                image_features={}, n_action_steps=chunk.shape[1], n_obs_steps=1
            )
            self._queues = {
                OBS_STATE: deque(maxlen=1),
                ACTION: deque(maxlen=self.config.n_action_steps),
            }
            self.predict_calls = 0

        def predict_action_chunk(self, batch):
            # Byte-for-byte the DiffusionPolicy online branch that crashes on a
            # None-poisoned ACTION queue (modeling_diffusion.predict_action_chunk).
            if any(len(q) > 0 for q in self._queues.values()):
                _ = {
                    k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues
                }
            self.predict_calls += 1
            return chunk

    pol = _make_relative_policy(chunk)
    pol._policy = _QueueFaithfulPolicy()
    processed_obs = {OBS_STATE: torch.zeros(1, 7), ACTION: None}
    out = pol._generate_relative_chunk_normalized(processed_obs)
    assert out.shape == (_REL_EXEC, _REL_DIM)
    assert pol._policy.predict_calls == 1


# --------------------------------------------------------------------------- #
# build_processed_obs consumes ONLY the policy's declared cameras, so a shared
# multi-arm capture (e.g. a stereo arm's wrist_right) does not leak an undeclared
# camera into a {wrist_left, side_1} arm. Required-but-uncaptured cameras fail loud.
# --------------------------------------------------------------------------- #


def _make_camera_policy(declared_image_features):
    pol = LeRobotRealWorldPolicy.__new__(LeRobotRealWorldPolicy)
    pol._policy = SimpleNamespace(
        config=SimpleNamespace(image_features=dict.fromkeys(declared_image_features))
    )
    pol._use_joint_positions = False
    # Live serial-eye keys map to roles through the placeholder station config
    # (10000001_left/_right -> wrist_left/_right, 20000002_left -> side_1).
    # Stub the crop/resize + preprocessor to identity so we test ONLY the selection.
    pol._crop_and_resize_native_rgb = lambda feat_key, img: np.zeros((3, 2, 2), dtype=np.float32)
    pol._preprocessor = lambda d: d
    return pol


def test_build_obs_consumes_only_declared_cameras_from_shared_capture():
    # Policy declares {wrist_left, side_1}; the SESSION captured 3 cameras (a stereo
    # arm shared the env). wrist_right must be IGNORED, not injected.
    pol = _make_camera_policy(["observation.images.wrist_left", "observation.images.side_1"])
    pol._camera_keys = ["10000001_left", "10000001_right", "20000002_left"]
    frames = {k: np.zeros((4, 4, 3), dtype=np.uint8) for k in pol._camera_keys}
    obs = pol.build_processed_obs(np.zeros(7, dtype=np.float32), frames)
    img_keys = {k for k in obs if k.startswith("observation.images.")}
    assert img_keys == {"observation.images.wrist_left", "observation.images.side_1"}


def test_build_obs_stereo_arm_consumes_all_three_cameras():
    pol = _make_camera_policy(
        [
            "observation.images.wrist_left",
            "observation.images.wrist_right",
            "observation.images.side_1",
        ]
    )
    pol._camera_keys = ["10000001_left", "10000001_right", "20000002_left"]
    frames = {k: np.zeros((4, 4, 3), dtype=np.uint8) for k in pol._camera_keys}
    obs = pol.build_processed_obs(np.zeros(7, dtype=np.float32), frames)
    img_keys = {k for k in obs if k.startswith("observation.images.")}
    assert img_keys == {
        "observation.images.wrist_left",
        "observation.images.wrist_right",
        "observation.images.side_1",
    }


def test_build_obs_missing_declared_camera_raises_loud():
    # Policy declares wrist_right but the capture never provided it -> hard error
    # (the encoder would be missing a required input).
    pol = _make_camera_policy(
        [
            "observation.images.wrist_left",
            "observation.images.wrist_right",
            "observation.images.side_1",
        ]
    )
    pol._camera_keys = ["10000001_left", "20000002_left"]  # wrist_right NOT captured
    frames = {k: np.zeros((4, 4, 3), dtype=np.uint8) for k in pol._camera_keys}
    with pytest.raises(KeyError, match="did not build every declared physical camera"):
        pol.build_processed_obs(np.zeros(7, dtype=np.float32), frames)


def test_canonical_action_is_velocity_even_for_a_position_command():
    # Data-integrity guard: the canonical 'action' column (named vel_x..gripper_action)
    # must ALWAYS be the cartesian_velocity+gripper representation, so a mixed-arm eval
    # dataset never stores an absolute pose under velocity names. DROID's action_info
    # carries every space; build_canonical_action must pick the velocity one regardless
    # of the deployed action space.
    from mulligan.real.collect.dataset_features import build_canonical_action

    action_info = {
        "cartesian_velocity": [0.1, -0.2, 0.3, 0.01, -0.02, 0.03],
        "gripper_velocity": 0.4,
        # a position command is ALSO present (the relative/position arm) — must be ignored
        # by the canonical builder and only recorded in the typed action.cartesian_position.
        "cartesian_position": [0.45, 0.0, 0.30, 0.1, -0.1, 0.2],
        "gripper_position": 0.9,
    }
    canonical = build_canonical_action(action_info)
    np.testing.assert_allclose(canonical, [0.1, -0.2, 0.3, 0.01, -0.02, 0.03, 0.4], atol=1e-6)
    # It is the velocity, NOT the pose (the two are distinct here).
    assert not np.allclose(canonical[:6], action_info["cartesian_position"])


def test_build_canonical_position_action_removed():
    # The pose-into-canonical builder is gone (it wrote a pose under velocity names).
    import mulligan.real.collect.dataset_features as df

    assert not hasattr(df, "build_canonical_position_action")
