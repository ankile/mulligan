"""End-to-end smoke of the RLPD loop on CPU (tiny networks, a synthetic 23-D
robomimic file with the PH env_args), including resume, plus the network-marked
checks of the named offline-data sources."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from mulligan.baselines.rlpd.datasets import OBS_KEYS, PH_ENV_META

REPO = Path(__file__).resolve().parents[3]
DIMS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2, "object": 14}


def _synthetic(root: Path) -> Path:
    """A three-demo robomimic low_dim file at ``root/toy.hdf5``."""
    rng = np.random.default_rng(0)
    path = root / "toy.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(PH_ENV_META)
        for i in range(3):
            n = 12
            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = n
            obs = {k: rng.standard_normal((n + 1, d)) for k, d in DIMS.items()}
            for grp, sl in (("obs", slice(0, n)), ("next_obs", slice(1, n + 1))):
                og = g.create_group(grp)
                for k in OBS_KEYS:
                    og.create_dataset(k, data=obs[k][sl])
            g.create_dataset("actions", data=rng.uniform(-1, 1, (n, 7)))
            rewards = np.zeros(n)
            rewards[-1] = 1
            g.create_dataset("rewards", data=rewards)
            g.create_dataset("dones", data=rewards.astype(np.int64))
    return path


def _train(data: Path, run: Path, max_steps: int) -> subprocess.CompletedProcess:
    argv = [
        sys.executable, "-m", "mulligan.baselines.rlpd.train",
        "--task=square_narrow", f"--offline_data={data}", "--seed=1",
        "--utd_ratio=2", "--batch_size=8", "--start_training=10", f"--max_steps={max_steps}",
        "--eval_episodes=1", "--eval_workers=1", "--eval_interval=20", "--log_interval=5",
        "--tqdm=False", "--hidden_dims=16,16", "--num_qs=2",
        f"--output_dir={run}", "--checkpoint_model=True",
    ]  # fmt: skip
    env = dict(os.environ, JAX_PLATFORMS="cpu", MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    return subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)


@pytest.mark.slow
def test_train_smoke_and_resume(tmp_path):
    pytest.importorskip("jax")
    data = _synthetic(tmp_path / "data")
    run = tmp_path / "run"
    first = _train(data, run, 30)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    evals = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [0, 20]
    assert {"return", "length", "success_rate", "success_length_mean"} <= set(evals[0])
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == [
        "agent_0000000.msgpack",
        "agent_0000020.msgpack",
    ]
    # a second process with the same resume dir continues after the completed step 30
    second = _train(data, run, 40)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "RESUMED" in second.stdout and "at step 31" in second.stdout
    evals = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [0, 20, 40]


@pytest.mark.network
@pytest.mark.parametrize(
    "task,offline_data,n,obs_dim",
    [
        ("square_narrow", "teleop", 16233, 23),
        ("square_broad", "teleop", 35486, 26),
        ("square_narrow", "robomimic_ph", 30154, 23),
    ],
)
def test_offline_sources_match_their_pins(task, offline_data, n, obs_dim):
    """The named sources load (download on first use) and match their pinned content sha256
    (load_offline raises otherwise). mimicgen_core (1.7 GB) is not downloaded here."""
    from mulligan.baselines.rlpd.datasets import load_offline

    ds, env_meta = load_offline(task, offline_data)
    assert ds.dataset_dict["observations"].shape == (n, obs_dim)
    assert (
        env_meta["env_kwargs"]["controller_configs"]
        == PH_ENV_META["env_kwargs"]["controller_configs"]
    )


def test_offline_data_must_name_a_source_of_the_task(tmp_path):
    from mulligan.baselines.rlpd.datasets import load_offline

    with pytest.raises(ValueError, match="is square_narrow data"):
        load_offline("square_broad", "robomimic_ph")
    with pytest.raises(ValueError, match="robomimic .hdf5 file"):
        load_offline("square_narrow", "ph")
    with pytest.raises(ValueError, match="holds NutAssemblySquare data"):
        load_offline("square_broad", str(_synthetic(tmp_path / "data")))


@pytest.mark.network
def test_mimicgen_source_is_the_pinned_file():
    """The MimicGen core file at its pinned revision has the pinned sha256 (Hub metadata; the
    1.7 GB download itself is not repeated here)."""
    from huggingface_hub import HfApi

    from mulligan.baselines.rlpd import datasets as D

    (info,) = HfApi().get_paths_info(
        D.MIMICGEN_REPO, [D.MIMICGEN_FILE], repo_type="dataset", revision=D.MIMICGEN_REVISION
    )
    assert info.lfs.sha256 == D.MIMICGEN_SHA256


def test_reader_fills_missing_next_obs_with_the_shifted_obs(tmp_path):
    """MimicGen's file has no next_obs: next_obs[t] = obs[t + 1], the last row repeated."""
    from mulligan.baselines.rlpd.datasets import load_robomimic_low_dim

    path = tmp_path / "no_next_obs.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("data").create_group("demo_0")
        g.attrs["num_samples"] = 4
        for j, k in enumerate(OBS_KEYS):
            g.create_dataset(
                f"obs/{k}", data=np.arange(4)[:, None] + 10 * j + np.zeros((4, DIMS[k]))
            )
        g.create_dataset("actions", data=np.zeros((4, 7)))
        g.create_dataset("rewards", data=np.array([0, 0, 0, 1.0]))
        g.create_dataset("dones", data=np.array([0, 0, 0, 1]))
    d = load_robomimic_low_dim(path)
    np.testing.assert_array_equal(d["next_observations"][:3], d["observations"][1:])
    np.testing.assert_array_equal(d["next_observations"][3], d["observations"][3])
