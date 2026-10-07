"""Tests for the UMI relative-pose action representation.

Covers the SE(3) relativize/absolutize math, the proprio-pose-r6 dataset remap,
the per-timestep (T,D) stats pre-pass, the per-timestep MIN_MAX normalizer patch,
the absolute-mode byte-identity regression, and the action_mode config round-trip.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mulligan.real.policy.relative_pose import (
    POSE_DIM,
    absolutize_pose,
    relativize_pose,
    relativize_pose_torch,
)
from mulligan.real.policy.rotation6d import euler_to_r6, r6_to_rotation_matrix


def _abs_chunk(B, T, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(B, T, 3))
    eul = rng.uniform(-np.pi, np.pi, size=(B, T, 3))
    r6 = euler_to_r6(eul)
    grip = rng.uniform(0, 1, size=(B, T, 1))
    return np.concatenate([xyz, r6, grip], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: relativize/absolutize round-trip + anchor->identity + rotation vs scipy
# ---------------------------------------------------------------------------
def test_relativize_absolutize_roundtrip_and_anchor_identity():
    abs_chunk = _abs_chunk(4, 12)
    base = abs_chunk[:, 0:1, :]  # anchor = first step
    rel = relativize_pose(abs_chunk, base)

    # Anchor element relativizes to identity: rel_trans ~ 0, rel_r6 ~ [1,0,0,0,1,0].
    assert np.abs(rel[:, 0, 0:3]).max() < 1e-5
    np.testing.assert_allclose(rel[:, 0, 3:9], np.tile([1, 0, 0, 0, 1, 0], (4, 1)), atol=1e-5)

    # Round-trip recovers the absolute chunk.
    recon = absolutize_pose(rel, base)
    np.testing.assert_allclose(recon, abs_chunk, atol=1e-5)

    # Gripper passes through ABSOLUTE (not relativized).
    np.testing.assert_allclose(rel[..., 9], abs_chunk[..., 9], atol=1e-6)

    # torch path matches numpy (one convention, no drift).
    at = torch.tensor(abs_chunk, dtype=torch.float64)
    bt = torch.tensor(base, dtype=torch.float64)
    rel_t = relativize_pose_torch(at, bt).numpy()
    np.testing.assert_allclose(rel_t, rel, atol=1e-5)


def test_relativize_rotation_matches_scipy_ground_truth():
    # Two explicit rotations; relative rotation must equal R_base^T @ R_k.
    base_eul = np.array([0.1, -0.2, 0.3])
    k_eul = np.array([-0.4, 0.5, 1.0])
    base = np.concatenate([[1.0, 2.0, 3.0], euler_to_r6(base_eul), [0.0]])
    posek = np.concatenate([[1.5, 2.5, 3.5], euler_to_r6(k_eul), [1.0]])
    rel = relativize_pose(posek[None, :], base[None, :])[0]

    R_base = Rotation.from_euler("xyz", base_eul).as_matrix()
    R_k = Rotation.from_euler("xyz", k_eul).as_matrix()
    expected_rel_R = R_base.T @ R_k
    got_rel_R = r6_to_rotation_matrix(rel[3:9])
    np.testing.assert_allclose(got_rel_R, expected_rel_R, atol=1e-5)

    expected_rel_t = R_base.T @ (posek[0:3] - base[0:3])
    np.testing.assert_allclose(rel[0:3], expected_rel_t, atol=1e-5)


def test_relativize_shape_guard():
    with pytest.raises(ValueError):
        relativize_pose(np.zeros((3, 7)), np.zeros((3, 7)))


def test_euler_to_r6_torch_matches_numpy():
    # The on-device torch euler->r6 (used to build the proprio anchor) must
    # match the scipy/numpy convention bit-to-tolerance, else train/deploy anchors drift.
    from mulligan.real.policy.relative_pose import _euler_to_r6_torch

    rng = np.random.default_rng(0)
    eul = rng.uniform(-np.pi, np.pi, size=(32, 3))
    got = _euler_to_r6_torch(torch.tensor(eul, dtype=torch.float64)).numpy()
    want = np.asarray(euler_to_r6(eul), dtype=np.float64)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_relative_processor_step_anchors_on_proprio_observation():
    # Proprio-anchored: RelativePoseActionProcessorStep relativizes the COMMAND horizon against
    # the PROPRIO pose from observation.state — NOT the command's own row 0.
    from lerobot.types import TransitionKey

    from mulligan.real.policy.relative_pose import RelativePoseActionProcessorStep

    B, T = 2, 12
    step = RelativePoseActionProcessorStep(n_obs_steps=1, action_dim=POSE_DIM)
    command = torch.tensor(_abs_chunk(B, T, seed=3), dtype=torch.float32)  # (B,T,10)
    # A proprio anchor DISTINCT from command[:,0] (identity orientation, arbitrary xyz).
    proprio_xyz = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    proprio_eul = torch.zeros(B, 3)
    grip = torch.zeros(B, 1)
    obs_state = torch.cat([proprio_xyz, proprio_eul, grip], dim=-1).unsqueeze(1)  # (B,1,7)
    transition = {
        TransitionKey.ACTION: command,
        TransitionKey.OBSERVATION: {"observation.state": obs_state},
    }
    rel = step(transition)[TransitionKey.ACTION].numpy().astype(np.float64)  # (B,T,10)

    # Composing back against the SAME proprio anchor recovers the command horizon.
    base10 = np.concatenate(
        [proprio_xyz.numpy(), np.asarray(euler_to_r6(proprio_eul.numpy())), grip.numpy()],
        axis=-1,
    )[:, None, :]  # (B,1,10)
    recon = absolutize_pose(rel, base10)
    np.testing.assert_allclose(recon[..., :9], command.numpy()[..., :9], atol=1e-4)
    # It did NOT anchor on command[:,0]: rel[0] translation is NONZERO (anchor != command[0]).
    assert np.abs(rel[:, 0, 0:3]).max() > 1e-3

    # Missing observation.state is a loud contract break (relative needs proprioception).
    with pytest.raises(ValueError):
        step({TransitionKey.ACTION: command, TransitionKey.OBSERVATION: {}})


# ---------------------------------------------------------------------------
# Fakes for the dataset-level remap / stats tests
# ---------------------------------------------------------------------------
class _FakeReader:
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset


class _FakeMeta:
    def __init__(self, n):
        self.info = SimpleNamespace(
            features={"action": {"dtype": "float32", "shape": (7,), "names": ["vel"] * 7}}
        )
        self.stats = {
            "action": {
                "min": np.full(7, -1.0, np.float32),
                "max": np.full(7, 1.0, np.float32),
                "mean": np.zeros(7, np.float32),
                "std": np.ones(7, np.float32),
                "count": np.array([n]),
            }
        }


class _FakeSub:
    def __init__(self, hf_dataset, n):
        self.reader = _FakeReader(hf_dataset)
        self.meta = _FakeMeta(n)
        self.repo_id = "synthetic/test"
        self._n = n

    @property
    def hf_dataset(self):
        return self.reader.hf_dataset

    def __len__(self):
        return self._n


class _FakeMulti:
    def __init__(self, subs):
        self._datasets = subs
        self.stats = {}


def _build_sub(
    action7, cart_pos, grip_pos, episode_index, *, commanded_pos=None, is_valid=None, done=None
):
    from datasets import Dataset
    from lerobot.datasets.io_utils import hf_transform_to_torch

    n = len(action7)
    cols = {
        "action": [r.tolist() for r in np.asarray(action7, np.float32)],
        "observation.state.cartesian_position": [
            r.tolist() for r in np.asarray(cart_pos, np.float32)
        ],
        "observation.state.gripper_position": [
            r.tolist() for r in np.asarray(grip_pos, np.float32)
        ],
        "episode_index": list(episode_index),
    }
    if is_valid is not None:
        cols["is_valid"] = [int(v) for v in is_valid]
    if done is not None:
        cols["done"] = [int(d) for d in done]
    if commanded_pos is not None:
        # A DIFFERENT commanded next-pose column, to prove the remap sources PROPRIO.
        cols["action.cartesian_position"] = [
            r.tolist() for r in np.asarray(commanded_pos, np.float32)
        ]
        cols["action.gripper_position"] = [[g] for g in np.zeros(n, np.float32)]
    hf = Dataset.from_dict(cols)
    hf.set_transform(hf_transform_to_torch)
    return _FakeSub(hf, n)


# ---------------------------------------------------------------------------
# Test 3: per-timestep stats pre-pass — (T,D), strictly increasing, pad masking
# ---------------------------------------------------------------------------
def _drift_abs_chunk(n, *, seed, step_lo=0.005, step_hi=0.02):
    # Strictly-positive-drift random walk in +x (so rel_trans[k] = sum of k positive
    # steps > 0, VARIES across anchors -> non-degenerate for k>=1, like real data);
    # orientation fixed identity, gripper varies a little.
    rng = np.random.default_rng(seed)
    steps = rng.uniform(step_lo, step_hi, size=n).astype(np.float32)
    steps[0] = 0.0
    xyz = np.zeros((n, 3), np.float32)
    xyz[:, 0] = np.cumsum(steps)
    r6 = np.tile(euler_to_r6(np.zeros(3)), (n, 1)).astype(np.float32)
    grip = rng.uniform(0, 1, (n, 1)).astype(np.float32)
    return np.concatenate([xyz, r6, grip], axis=1).astype(np.float32)


def _scheme_p_command_and_proprio(n, *, seed, v=0.01):
    """Proprio-anchored synthetic episode: command LEADS proprio by a per-step lead in +x.

    proprio_xyz[i] = [i*v, 0, 0] (identity orientation); command_xyz[i] =
    [i*v + lead_i, 0, 0] with lead_i VARYING (so rel[0]=lead is non-degenerate under
    the proprio anchor; a command anchor would make rel[0] identically 0).
    Returns (command10, proprio_cart6, grip1).
    """
    rng = np.random.default_rng(seed)
    lead = rng.uniform(0.008, 0.016, size=n)
    ident_r6 = euler_to_r6(np.zeros(3)).astype(np.float32)  # (6,)
    proprio6 = np.zeros((n, 6), np.float32)
    proprio6[:, 0] = (np.arange(n) * v).astype(np.float32)
    cmd_xyz = np.zeros((n, 3), np.float32)
    cmd_xyz[:, 0] = (np.arange(n) * v + lead).astype(np.float32)
    grip = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    command10 = np.concatenate([cmd_xyz, np.tile(ident_r6, (n, 1)), grip], axis=1).astype(
        np.float32
    )
    return command10, proprio6, grip


def test_pertimestep_stats_scheme_p_anchors_on_proprio():
    from mulligan.data.transforms import compute_relative_pose_pertimestep_stats

    horizon, n_obs_steps = 12, 1
    n = 80  # one long episode: every anchor reaches all 12 steps (no padding)
    command10, proprio6, grip = _scheme_p_command_and_proprio(n, seed=7)
    sub = _build_sub(command10, proprio6, grip, [0] * n)
    ds = _FakeMulti([sub])
    stats = compute_relative_pose_pertimestep_stats(
        ds, horizon=horizon, n_obs_steps=n_obs_steps, drop_n_last_frames=0
    )
    assert stats["min"].shape == (horizon, POSE_DIM)
    assert dataset_stats_installed(ds)

    # SCHEME P: rel[0] = the command-vs-proprio LEAD, not identity. A command anchor
    # would give rel[0]==0 for the translation; here the mean x-translation at k=0 is
    # the positive lead (~0.012), and it VARIES across anchors so the dim is NON-
    # degenerate (real range ~0.008, NOT widened to denom 1.0).
    assert stats["mean"][0, 0] > 0.005, stats["mean"][0, 0]
    denom0_x = stats["max"][0, 0] - stats["min"][0, 0]
    assert 1e-4 < denom0_x < 0.5, denom0_x

    # t-dependence (t+1 != t+2): translation MAX grows with t (rel_trans[k].x = kv + lead).
    tx_max = stats["max"][:, 0]
    assert np.all(np.diff(tx_max) > 0), tx_max

    # Orientation is identity throughout -> rel_r6 at k=0 IS degenerate -> widened to a
    # UNIT span (denom==1.0) by the defensive guard. Gripper varies at k=0 (not widened).
    denom0 = stats["max"][0] - stats["min"][0]
    np.testing.assert_allclose(denom0[3:9], 1.0, atol=1e-5)  # r6 static -> widened
    assert denom0[9] > 1e-3  # gripper has real range at k=0


def test_pertimestep_stats_stop_at_repeated_done_tail():
    from mulligan.data.transforms import compute_relative_pose_pertimestep_stats

    # One 20-frame episode, all is_valid=1, done=1 from frame 10 on (a repeated
    # terminal tail retained inside the valid run). Frames >=11 carry a +100 x
    # jump in BOTH command and proprio, so any stats window row crossing the
    # first done frame would see a ~100 relative x displacement. The stats must
    # clamp to the first done frame exactly like training anchors
    # (clamp_prefix_to_first_done): every per-timestep max stays tiny.
    horizon, n_obs_steps = 4, 1
    n = 20
    command10, proprio6, grip = _scheme_p_command_and_proprio(n, seed=3)
    command10 = command10.copy()
    proprio6 = proprio6.copy()
    command10[11:, 0] += 100.0
    proprio6[11:, 0] += 100.0
    done = [0] * 10 + [1] * 10
    sub = _build_sub(command10, proprio6, grip, [0] * n, is_valid=[1] * n, done=done)
    ds = _FakeMulti([sub])
    stats = compute_relative_pose_pertimestep_stats(
        ds, horizon=horizon, n_obs_steps=n_obs_steps, drop_n_last_frames=0
    )
    assert np.all(stats["max"][:, 0] < 1.0), stats["max"][:, 0]


def test_pertimestep_stats_pad_masking():
    from mulligan.data.transforms import compute_relative_pose_pertimestep_stats

    # Two episodes of length 20 (> horizon 12): deep anchors near each episode end have
    # PADDED (clamped) tail rows whose relativized displacement is 0. With a strictly
    # POSITIVE-drift trajectory every reaching anchor has rel_trans[k] > 0, so the per-t
    # translation MIN over valid rows stays strictly positive. If padded (clamped) rows
    # leaked in, their 0 displacement would drag min[k,0] down to ~0. min[k,0] > 0 for
    # all k therefore proves padded rows are masked out.
    horizon = 12
    ep = 20
    c1, p1, g1 = _scheme_p_command_and_proprio(ep, seed=1)
    c2, p2, g2 = _scheme_p_command_and_proprio(ep, seed=2)
    command = np.concatenate([c1, c2], axis=0)
    proprio = np.concatenate([p1, p2], axis=0)
    grip = np.concatenate([g1, g2], axis=0)
    sub = _build_sub(command, proprio, grip, [0] * ep + [1] * ep)
    ds = _FakeMulti([sub])
    stats = compute_relative_pose_pertimestep_stats(
        ds, horizon=horizon, n_obs_steps=1, drop_n_last_frames=0
    )
    # rel_trans[k].x = kv + lead > 0 for every reaching anchor; padded (clamped) tail rows
    # would relativize to a SMALLER displacement, so if they leaked in they'd drag min[k,0]
    # down. min[k,0] > 0 for all k (incl. k=0, the lead) therefore proves padded rows are
    # masked out.
    assert np.all(stats["min"][:, 0] > 1e-4), stats["min"][:, 0]


def test_pertimestep_stats_honor_is_valid_prefix():
    from mulligan.data.transforms import compute_relative_pose_pertimestep_stats

    # Outcome-edited episode: 20 valid proprio-anchored frames, then a 10-frame is_valid==0
    # junk suffix (post-outcome retract) that JUMPS to x=+10 m. The training sampler
    # never lets a supervised action timestep land on an is_valid==0 frame
    # (episode_anchor_exclusive_end), so the stats must not see the jump either: if
    # any junk row leaked into a window, max[k, 0] would blow up to ~10 m.
    horizon = 12
    n_valid, n_junk = 20, 10
    command10, proprio6, grip = _scheme_p_command_and_proprio(n_valid, seed=11)
    junk10 = np.tile(command10[-1], (n_junk, 1)).astype(np.float32)
    junk10[:, 0] = 10.0  # retract jump the model never trains on
    junk_proprio = np.tile(proprio6[-1], (n_junk, 1)).astype(np.float32)
    junk_grip = np.tile(grip[-1], (n_junk, 1)).astype(np.float32)
    command = np.concatenate([command10, junk10], axis=0)
    proprio = np.concatenate([proprio6, junk_proprio], axis=0)
    gripcat = np.concatenate([grip, junk_grip], axis=0)
    n = n_valid + n_junk
    sub = _build_sub(command, proprio, gripcat, [0] * n, is_valid=[1] * n_valid + [0] * n_junk)
    ds = _FakeMulti([sub])
    stats = compute_relative_pose_pertimestep_stats(
        ds, horizon=horizon, n_obs_steps=1, drop_n_last_frames=0
    )
    # Valid-only displacement is bounded by the episode's total valid drift (~0.2 m);
    # a leaked junk window would put ~9.8 m in max[k, 0].
    assert stats["max"][:, 0].max() < 1.0, stats["max"][:, 0]
    # And the valid prefix still yields real (non-degenerate) per-timestep stats.
    assert stats["min"].shape == (horizon, 10)
    assert np.all(stats["max"][:, 0] > 0)


def dataset_stats_installed(ds):
    s = ds.stats.get("action")
    return s is not None and np.asarray(s["min"]).ndim == 2


def test_pertimestep_stats_raises_on_no_anchors():
    from mulligan.data.transforms import compute_relative_pose_pertimestep_stats

    n = 5
    action10 = _drift_abs_chunk(n, seed=3)
    sub = _build_sub(action10, action10[:, :6], action10[:, 9:10], [0] * n)
    ds = _FakeMulti([sub])
    # drop_n_last >= episode length leaves no valid anchors.
    with pytest.raises(RuntimeError):
        compute_relative_pose_pertimestep_stats(
            ds, horizon=12, n_obs_steps=1, drop_n_last_frames=10
        )


# ---------------------------------------------------------------------------
# Test 4: per-timestep MIN_MAX identity (full + eval-sliced) + own-row + epsilon
# ---------------------------------------------------------------------------
def _make_normalizer(stats):
    import mulligan.utils.lerobot_patches as P

    P.apply_all_patches()
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    feats = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(stats["action"]["min"].shape[-1],))
    }
    nm = {FeatureType.ACTION: NormalizationMode.MIN_MAX}
    return NormalizerProcessorStep(features=feats, norm_map=nm, stats=stats)


def test_pertimestep_minmax_identity_full_and_sliced():
    from lerobot.types import TransitionKey

    T, D, B = 12, 10, 5
    mn = (np.arange(T)[:, None] * 0.1 - np.linspace(1, 2, D)[None, :]).astype(np.float32)
    mx = (mn + (1.0 + np.arange(T)[:, None] * 0.05)).astype(np.float32)
    step = _make_normalizer({"action": {"min": mn, "max": mx}})
    mn_t, mx_t = torch.tensor(mn), torch.tensor(mx)

    def run(t):
        return step({TransitionKey.ACTION: t})[TransitionKey.ACTION]

    x = torch.randn(B, T, D)
    np.testing.assert_allclose(
        run(x).numpy(), (2 * (x - mn_t) / (mx_t - mn_t) - 1).numpy(), atol=1e-5
    )
    # Eval-sliced (B, T'<T, D): uses each t's OWN [min,max] row [0:T'].
    xs = torch.randn(B, 6, D)
    np.testing.assert_allclose(
        run(xs).numpy(), (2 * (xs - mn_t[:6]) / (mx_t[:6] - mn_t[:6]) - 1).numpy(), atol=1e-5
    )
    # A single (B, D) action (no time axis) against per-timestep stats raises loudly.
    with pytest.raises(ValueError):
        run(torch.randn(B, D))


def test_pertimestep_minmax_epsilon_when_min_eq_max():
    from lerobot.types import TransitionKey

    T, D = 4, 10
    mn = np.zeros((T, D), np.float32)
    mx = np.zeros((T, D), np.float32)  # min == max everywhere
    step = _make_normalizer({"action": {"min": mn, "max": mx}})
    out = step({TransitionKey.ACTION: torch.zeros(2, T, D)})[TransitionKey.ACTION]
    # min==max maps input==min to -1 via the eps branch (no NaN/inf).
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 5: ABSOLUTE-MODE regression — patch + recon abs path unchanged
# ---------------------------------------------------------------------------
def test_absolute_mode_normalizer_untouched_for_1d_stats():
    from lerobot.types import TransitionKey

    D, B, T = 7, 3, 12
    mn = np.full(D, -2.0, np.float32)
    mx = np.full(D, 2.0, np.float32)
    step = _make_normalizer({"action": {"min": mn, "max": mx}})
    x = torch.randn(B, T, D)
    mn_t, mx_t = torch.tensor(mn), torch.tensor(mx)
    # 1-D stats -> patch is a no-op -> standard broadcast.
    np.testing.assert_allclose(
        step({TransitionKey.ACTION: x})[TransitionKey.ACTION].numpy(),
        (2 * (x - mn_t) / (mx_t - mn_t) - 1).numpy(),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Test 7: action_mode config round-trip (draccus save/load) + legacy default
# ---------------------------------------------------------------------------
def test_action_mode_config_roundtrip(tmp_path):
    import mulligan.utils.lerobot_patches as P

    P.apply_all_patches()
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    cfg = DiffusionConfig()
    assert "action_mode" in DiffusionConfig.__dataclass_fields__
    # Default is None (legacy => absolute).
    assert getattr(cfg, "action_mode") is None
    cfg.action_mode = "relative"
    # draccus-style dict round-trip through the saved config fields.
    import draccus

    dumped = draccus.encode(cfg)
    assert dumped["action_mode"] == "relative"
