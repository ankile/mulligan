# Based on the EXPO source code (https://github.com/pd-perry/EXPO) and the RLPD source code
# (https://github.com/ikostrikov/rlpd), on which EXPO builds.
# Vendored from EXPO's expo/data/robomimic_datasets.py. Both projects are MIT-licensed; their
# notices follow.
#
# ---- EXPO ----
# MIT License
#
# Copyright (c) 2025 pd-perry
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---- RLPD (https://github.com/ikostrikov/rlpd, LICENCE) ----
# MIT License
#
# Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Offline data for RLPD.

Every source gives the RLPD transition arrays (``observations``, ``actions``, ``rewards``,
``masks``, ``dones``, ``next_observations``; float32, ``dones`` = episode boundaries) and the
robomimic ``env_args`` the env is checked against:

- ``teleop``: the task's round-0 teleop demos (``mulligan/sim-<task>-c00-teleop-baseline`` at
  the pin in :mod:`mulligan.baselines.hilserl.tasks`), read from the LeRobot dataset;
- ``robomimic_ph``: robomimic's 200 Square PH demos (``square_narrow``), downloaded once into
  ``$XDG_CACHE_HOME/mulligan/rlpd`` (``~/.cache`` when unset);
- ``mimicgen_core``: MimicGen's 1,000 core ``Square_D1`` demos (``square_broad``), from the Hub;
- a path to a robomimic low_dim hdf5 with the same observation keys.

The named sources are checked against a pinned content sha256 of the arrays.

Robomimic files are read with h5py: demos in ``int(name[5:])`` order, each contributing
``attrs["num_samples"]`` transitions, every array cast to float32, observations the four
``OBS_KEYS`` concatenated (eef_pos 3, eef_quat 4, gripper_qpos 2, object 14 or 17 -> 23-D /
26-D). A file without ``next_obs`` (MimicGen's) uses ``obs[t + 1]`` with the last row repeated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from mulligan.baselines.rlpd.data.dataset import Dataset

OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")

# robomimic env_args of the Square PH file (robosuite 1.4.1 kwargs). The teleop demos carry the
# same controller; RLPDEnv checks the live env against them.
PH_ENV_META = {
    "env_name": "NutAssemblySquare",
    "env_version": "1.4.1",
    "type": 1,
    "env_kwargs": {
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": False,
        "control_freq": 20,
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
        "robots": ["Panda"],
        "camera_depths": False,
        "camera_heights": 84,
        "camera_widths": 84,
        "reward_shaping": False,
    },
}

PH_URL = "http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/low_dim_v141.hdf5"
PH_SHA256 = "afc1fc1bd72193d1fd89c689b738cb9eb1a17352217aea2a2e029ccc49eefcbd"
MIMICGEN_REPO = "amandlek/mimicgen_datasets"
MIMICGEN_FILE = "core/square_d1.hdf5"
MIMICGEN_REVISION = "33016f8a62c02334f929f2913af8fdd2a8a129e1"
MIMICGEN_SHA256 = "bdb88ebb36f791645ed4b42512b5cf92e337c1673a71ddf4a7a6295c446fabdf"


@dataclass(frozen=True)
class Source:
    task: str
    num_transitions: int
    content_sha256: str


# Named robomimic-file sources; ``teleop`` pins live in hilserl.tasks.
SOURCES = {
    "robomimic_ph": Source(
        "square_narrow", 30154, "755593a3993027cc958bb41949a4a9b2da14c44294992f7d44e3ad672bf78662"
    ),
    "mimicgen_core": Source(
        "square_broad", 152400, "26251cdc5022e2a245a72f2320a610d05c3e1e74e4ee6ef5e9f948a047ece952"
    ),
}
OFFLINE_DATA = ("teleop", *SOURCES)


def env_meta_for(env_name: str) -> dict:
    """robomimic env_args with the PH controller for ``env_name`` (the teleop demos' env)."""
    meta = copy.deepcopy(PH_ENV_META)
    meta["env_name"] = env_name
    return meta


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mulligan" / "rlpd"


def robomimic_ph_path() -> Path:
    """The robomimic Square PH low_dim file, downloaded once and sha256-checked."""
    path = cache_dir() / "robomimic_square_ph_low_dim_v141.hdf5"
    if path.exists() and sha256(path) == PH_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {PH_URL} -> {path}", flush=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(PH_URL, tmp)
    got = sha256(tmp)
    if got != PH_SHA256:
        tmp.unlink()
        raise ValueError(f"{PH_URL}: sha256 {got} != {PH_SHA256}")
    tmp.replace(path)
    return path


def mimicgen_core_path() -> Path:
    """MimicGen's core/square_d1.hdf5 (1.7 GB) from the Hub cache, sha256-checked."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            MIMICGEN_REPO, MIMICGEN_FILE, repo_type="dataset", revision=MIMICGEN_REVISION
        )
    )
    got = sha256(path)
    if got != MIMICGEN_SHA256:
        raise ValueError(f"{path}: sha256 {got} != {MIMICGEN_SHA256}")
    return path


def read_env_meta(path: str | Path) -> dict:
    """The ``data.attrs["env_args"]`` JSON of a robomimic file."""
    with h5py.File(path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def load_robomimic_low_dim(path: str | Path) -> dict[str, np.ndarray]:
    """observations, next_observations, actions, rewards, terminals of a robomimic file (float32)."""
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("observations", "next_observations", "actions", "rewards", "terminals")
    }
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k[5:]))
        for name in demos:
            g = f["data"][name]
            n = int(g.attrs["num_samples"])
            obs = np.concatenate([g["obs"][k][()].astype("float32")[:n] for k in OBS_KEYS], axis=1)
            if "next_obs" in g:
                next_obs = np.concatenate(
                    [g["next_obs"][k][()].astype("float32")[:n] for k in OBS_KEYS], axis=1
                )
            else:
                next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
            parts["observations"].append(obs)
            parts["next_observations"].append(next_obs)
            parts["actions"].append(g["actions"][()].astype("float32")[:n])
            parts["rewards"].append(g["rewards"][()].astype("float32")[:n])
            parts["terminals"].append(g["dones"][()].astype("float32")[:n])
    return {k: np.concatenate(v) for k, v in parts.items()}


class RoboD4RLDataset(Dataset):
    def __init__(
        self,
        dataset: dict[str, np.ndarray],
        clip_to_eps: bool = True,
        eps: float = 1e-5,
        ignore_done: bool = False,
    ):
        dataset = dict(dataset)
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
        dones_float = np.zeros_like(dataset["rewards"])
        for i in range(len(dones_float) - 1):
            boundary = (
                np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i])
                > 1e-6
            )
            if ignore_done:
                dones_float[i] = 1 if boundary else 0
            else:
                dones_float[i] = 1 if boundary or dataset["terminals"][i] == 1.0 else 0
        dones_float[-1] = 1
        super().__init__(
            {
                "observations": dataset["observations"].astype(np.float32),
                "actions": dataset["actions"].astype(np.float32),
                "rewards": dataset["rewards"].astype(np.float32),
                "masks": 1.0 - dataset["terminals"].astype(np.float32),
                "dones": dones_float.astype(np.float32),
                "next_observations": dataset["next_observations"].astype(np.float32),
            }
        )


def content_sha256(dataset_dict: dict[str, np.ndarray]) -> str:
    from mulligan.baselines.hilserl.demos import content_sha256 as sha

    return sha({**dataset_dict, "dones": dataset_dict["dones"].astype(bool)})


def load_offline(task: str, offline_data: str) -> tuple[Dataset, dict]:
    """(dataset, env_args) of ``offline_data`` (a source name or a robomimic hdf5 path) for ``task``."""
    from mulligan.baselines.hilserl.tasks import get_task

    spec = get_task(task)
    if offline_data == "teleop":
        from mulligan.baselines.hilserl.demos import load_demo_transitions

        transitions = load_demo_transitions(spec)  # checks the pinned content sha256
        transitions["dones"] = transitions["dones"].astype(np.float32)
        return Dataset(transitions), env_meta_for(spec.env_name)
    if offline_data in SOURCES:
        source = SOURCES[offline_data]
        if source.task != task:
            raise ValueError(f"--offline_data={offline_data} is {source.task} data, not {task}")
        path = robomimic_ph_path() if offline_data == "robomimic_ph" else mimicgen_core_path()
    else:
        path = Path(offline_data)
        if path.suffix != ".hdf5" or not path.is_file():
            raise ValueError(
                f"--offline_data must be one of {OFFLINE_DATA} or a robomimic .hdf5 file, "
                f"got {offline_data!r}"
            )
        source = None
    print(f"Loading offline data from {path}", flush=True)
    ds = RoboD4RLDataset(load_robomimic_low_dim(path))
    if source is not None:
        n, got = len(ds.dataset_dict["observations"]), content_sha256(ds.dataset_dict)
        if (n, got) != (source.num_transitions, source.content_sha256):
            raise ValueError(
                f"{path}: {n} transitions, content sha256 {got}; expected "
                f"{source.num_transitions} / {source.content_sha256}"
            )
    env_meta = read_env_meta(path)
    if env_meta["env_name"] != spec.env_name:
        raise ValueError(f"{path} holds {env_meta['env_name']} data, not {spec.env_name} ({task})")
    return ds, env_meta
