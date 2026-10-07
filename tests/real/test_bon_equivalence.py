"""Golden regression of the real-robot Best-of-N deploy path.

The fixture (tests/real/fixtures/bon/, provenance in bon_r05.json) holds 3 recorded
observations per task and the recorded outputs on CPU: the N candidate chunks the
R5 Mulligan DP sampled, the R5 critic's twin-Q scores and the argmax. N = 32, the deployed
R5 eval override of the critic metadata default (16), applied like manifest_eval does for
a local critic (``policy.num_action_samples = N``).

1. The critic scores the RECORDED candidates and picks the same argmax (the
   ranking check; independent of diffusion sampling): bitwise on the recording setup (see
   EXACT), else within atol 1e-5.
2. With the same seed on CPU, the DP samples the same candidates: bitwise on the
   recording setup, else within 1e-4 (other CPU kernels, e.g. AVX-512 nodes, move the
   candidates by up to ~2e-5 after the denoising loop).

The bitwise comparisons run with ``MULLIGAN_BITWISE_PARITY=1`` on the recording setup (the
recorded torch version, x86 with ATen dispatching AVX2 kernels); by default they use the
tolerances, because other CPUs move the last float bits. The tests run on CPU only: CUDA is reported unavailable to the policies,
also on GPU nodes.

Checkpoints are downloaded anonymously from Hugging Face at the pinned release revisions.
"""

import json
import shutil
from pathlib import Path

import os

import numpy as np
import pytest
import torch

from mulligan.real.policy.dp import checkpoint_dir, critic_dp_model_id

pytestmark = pytest.mark.network

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "bon"
SIDECAR = json.loads((FIXTURE_DIR / "bon_r05.json").read_text())
TASKS = list(SIDECAR["tasks"])
CASES = [(task, i) for task in TASKS for i in range(3)]
# Compare without the local-version suffix (+cu128, +cpu): it names the build, not the CPU kernels.
RECORDED_TORCH = SIDECAR["setup"]["torch"].split("+", 1)[0]
TORCH = torch.__version__.split("+", 1)[0]
EXACT = os.environ.get("MULLIGAN_BITWISE_PARITY") == "1" and TORCH == RECORDED_TORCH


@pytest.fixture(scope="module")
def fixture_arrays():
    data = np.load(FIXTURE_DIR / "bon_r05.npz")
    return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def deterministic_cpu():
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(SIDECAR["setup"]["num_threads"])
    yield
    torch.set_num_threads(prev_threads)
    torch.use_deterministic_algorithms(prev_det)


@pytest.fixture(scope="module")
def policies(tmp_path_factory, deterministic_cpu, monkeypatch_module):
    from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

    monkeypatch_module.delenv("WANDB_API_KEY", raising=False)
    monkeypatch_module.setattr(torch.cuda, "is_available", lambda: False)
    cache = tmp_path_factory.mktemp("hf-bon")
    loaded = {}
    try:
        for task, spec in SIDECAR["tasks"].items():
            critic_id = f"hf://{spec['critic']['repo']}@{spec['critic']['revision']}"
            critic_dir = checkpoint_dir(critic_id, cache_dir=cache, token=False)
            metadata = json.loads((critic_dir / "metadata.json").read_text())
            dp_id = critic_dp_model_id(metadata["dp_artifact"])
            assert dp_id == f"hf://{spec['dp']['repo']}@{spec['dp']['revision']}"
            dp_dir = checkpoint_dir(dp_id, cache_dir=cache, token=False)
            policy = VisionIDQLRealWorldPolicy.from_artifact(
                critic_dir, device="cpu", dp_local_dir=dp_dir
            ).eval()
            assert policy.num_action_samples == spec["metadata_num_action_samples"]
            policy.num_action_samples = spec["num_action_samples"]
            assert policy.action_mode == spec["action_mode"]
            loaded[task] = policy
        yield loaded
    finally:
        shutil.rmtree(cache, ignore_errors=True)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _raw_obs(arrays, task, i):
    cams = SIDECAR["tasks"][task]["cameras"]
    return {
        "robot_state": {
            "cartesian_position": arrays[f"{task}/{i}/cartesian_position"],
            "gripper_position": float(arrays[f"{task}/{i}/gripper_position"]),
        },
        "image": {cam: arrays[f"{task}/{i}/image/{cam}"] for cam in cams},
    }


@pytest.mark.parametrize(("task", "i"), CASES, ids=[f"{t}-{i}" for t, i in CASES])
def test_release_critic_scores_recorded_candidates(policies, fixture_arrays, task, i):
    policy = policies[task]
    candidates = fixture_arrays[f"{task}/{i}/candidates"]
    policy.reset()
    policy.set_candidate_action_override(candidates)
    policy.predict(_raw_obs(fixture_arrays, task, i))
    info = policy.last_chunk_info
    assert info["num_action_samples_effective"] == len(candidates)
    for key, recorded in (("q1_all", "q1"), ("q2_all", "q2"), ("q_min_all", "q_min")):
        got = np.asarray(info[key], dtype=np.float32)
        want = fixture_arrays[f"{task}/{i}/{recorded}"]
        if EXACT:
            np.testing.assert_array_equal(got, want, err_msg=f"{task}/{i} {key}")
        else:
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-5, err_msg=f"{task}/{i} {key}")
    assert info["best_idx"] == int(fixture_arrays[f"{task}/{i}/best_idx"])


@pytest.mark.parametrize(("task", "i"), CASES, ids=[f"{t}-{i}" for t, i in CASES])
def test_release_dp_samples_recorded_candidates(policies, fixture_arrays, task, i):
    policy = policies[task]
    policy.reset()
    policy.set_candidate_action_capture(True)
    torch.manual_seed(SIDECAR["tasks"][task]["seeds"][i])
    action = policy.predict(_raw_obs(fixture_arrays, task, i))
    candidates = policy.captured_candidate_action_chunks()
    policy.set_candidate_action_capture(False)
    recorded = fixture_arrays[f"{task}/{i}/candidates"]
    assert next(policy.dp.parameters()).device.type == "cpu"
    if EXACT:
        np.testing.assert_array_equal(candidates, recorded)
        np.testing.assert_array_equal(
            np.asarray(action, dtype=np.float32), fixture_arrays[f"{task}/{i}/action"]
        )
    else:
        np.testing.assert_allclose(candidates, recorded, rtol=1e-4, atol=1e-4)
    assert policy.last_chunk_info["best_idx"] == int(fixture_arrays[f"{task}/{i}/best_idx"])
