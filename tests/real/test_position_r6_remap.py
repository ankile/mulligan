"""Unit test for the cartesian_position_r6 action remap.

Pins the in-memory contract of remap_action_to_position_r6_in_subdatasets on a
lightweight fake MultiLeRobotDataset (no network / no real datamix): the
canonical 'action' column grows 7D -> 10D, names/shape are updated, xyz +
gripper pass through untouched, the r6 orientation decodes back to the source
euler rotation, and per-sub-dataset + aggregated stats are 10D.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from scipy.spatial.transform import Rotation

from mulligan.real.policy.rotation6d import EULER_CONVENTION, r6_to_rotation_matrix
from mulligan.data.transforms import remap_action_to_position_r6_in_subdatasets


class _FakeMeta:
    def __init__(self, n):
        self.info = SimpleNamespace(
            features={"action": {"dtype": "float32", "shape": (7,), "names": list("abcdefg")}}
        )
        self.stats = {"action": {"min": np.zeros(7), "max": np.ones(7), "count": np.array([n])}}


class _FakeReader:
    """Minimal stand-in for LeRobotDataset.reader: holds a settable hf_dataset."""

    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset


class _FakeSub:
    def __init__(self, repo_id, n, seed):
        rng = np.random.default_rng(seed)
        # cartesian_position euler covering the +/-pi seam; gripper 0/1.
        cart_pos = np.empty((n, 6), dtype=np.float32)
        cart_pos[:, :3] = rng.uniform(-0.5, 0.5, size=(n, 3))
        cart_pos[:, 3:6] = rng.uniform(-np.pi, np.pi, size=(n, 3))
        grip = rng.integers(0, 2, size=(n, 1)).astype(np.float32)
        self.repo_id = repo_id
        self._cart_pos = cart_pos
        self._grip = grip
        self.reader = _FakeReader(
            Dataset.from_dict(
                {
                    "action": [
                        [0.0] * 7 for _ in range(n)
                    ],  # canonical 7D (velocity-ish placeholder)
                    "action.cartesian_position": cart_pos.tolist(),
                    "action.gripper_position": grip.tolist(),
                }
            )
        )
        self.meta = _FakeMeta(n)

    @property
    def hf_dataset(self):
        """Read-only property mirroring LeRobotDataset.hf_dataset (delegates to reader)."""
        return self.reader.hf_dataset


class _FakeMulti:
    def __init__(self, subs):
        self._datasets = subs
        self.stats = None


def test_remap_position_r6_contract():
    n = 300
    subs = [_FakeSub("repo/a", n, 0), _FakeSub("repo/b", n, 1)]
    ds = _FakeMulti(subs)

    remap_action_to_position_r6_in_subdatasets(ds)

    for sub in subs:
        action = np.asarray(sub.hf_dataset.with_format(None)["action"], dtype=np.float64)
        assert action.shape == (n, 10), action.shape

        # Feature schema updated.
        feat = sub.meta.info.features["action"]
        assert list(feat["shape"]) == [10]
        assert feat["names"] == [
            "x",
            "y",
            "z",
            "r6_0",
            "r6_1",
            "r6_2",
            "r6_3",
            "r6_4",
            "r6_5",
            "gripper",
        ]

        # xyz + gripper pass through untouched.
        assert np.allclose(action[:, :3], sub._cart_pos[:, :3], atol=1e-5)
        assert np.allclose(action[:, 9:10], sub._grip, atol=1e-5)

        # r6 orientation decodes to the source euler rotation.
        mat_src = Rotation.from_euler(EULER_CONVENTION, sub._cart_pos[:, 3:6]).as_matrix()
        mat_r6 = r6_to_rotation_matrix(action[:, 3:9])
        rel = np.swapaxes(mat_src, -1, -2) @ mat_r6
        angle = Rotation.from_matrix(rel).magnitude()
        assert angle.max() < 1e-5, angle.max()

        # r6 dims within [-1,1].
        assert action[:, 3:9].min() >= -1.0 - 1e-5
        assert action[:, 3:9].max() <= 1.0 + 1e-5

        # Per-sub-dataset action stats are 10D.
        assert np.asarray(sub.meta.stats["action"]["min"]).shape == (10,)
        assert np.asarray(sub.meta.stats["action"]["max"]).shape == (10,)

    # Aggregated multi-dataset stats are 10D.
    assert np.asarray(ds.stats["action"]["min"]).shape == (10,)


def test_remap_position_r6_missing_column_raises():
    sub = _FakeSub("repo/a", 10, 0)
    sub.reader.hf_dataset = sub.reader.hf_dataset.remove_columns("action.cartesian_position")
    ds = _FakeMulti([sub])
    with pytest.raises(KeyError, match="action.cartesian_position"):
        remap_action_to_position_r6_in_subdatasets(ds)


def test_remap_position_r6_wrong_shape_raises():
    sub = _FakeSub("repo/a", 10, 0)
    # Corrupt the canonical action feature shape so the 7D guard trips.
    sub.meta.info.features["action"]["shape"] = (6,)
    ds = _FakeMulti([sub])
    with pytest.raises(ValueError, match=r"canonical 'action'"):
        remap_action_to_position_r6_in_subdatasets(ds)
