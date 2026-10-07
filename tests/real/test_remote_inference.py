import os
import shlex
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mulligan.real.eval import inference_server
from mulligan.real.robot.cameras import policy_live_camera_keys
from mulligan.real.eval.common import PolicyEntry


class FakePolicy:
    action_space = "cartesian_velocity"
    gripper_action_space = None
    live_camera_keys = ["cam_a"]

    def __init__(self):
        self.config = SimpleNamespace(
            image_features={"observation.images.cam_a": None},
        )
        self.camera_keys = []
        self.reset_count = 0
        self.num_action_samples = 4
        self.last_chunk_info = {"v_val": 1.25}

    @property
    def env_action_space(self):
        return self.action_space

    def set_camera_keys(self, camera_keys):
        self.camera_keys = list(camera_keys)

    def reset(self):
        self.reset_count += 1

    def predict(self, raw_obs):
        assert sorted(raw_obs["image"]) == ["cam_a"]
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def _post(client, path, payload, *, token="tok"):
    return inference_server._dispatch_request(
        client,
        method="POST",
        path=path,
        body=inference_server._pack(payload),
        headers={"X-Mulligan-Inference-Token": token},
    )


def test_remote_inference_server_load_configure_predict(monkeypatch):
    fake_policy = FakePolicy()

    def fake_load_policy_by_model_id(**kwargs):
        return PolicyEntry(
            model_id=f"wandb://{kwargs['model_id'].removeprefix('wandb://')}",
            policy_id=kwargs["policy_id"],
            policy=fake_policy,
            camera_height=224,
            camera_width=224,
        )

    monkeypatch.setattr(
        inference_server,
        "load_policy_by_model_id",
        fake_load_policy_by_model_id,
    )
    state = inference_server._ServerState(device="cuda", token="tok")

    status, data = _post(
        state,
        "/load",
        {
            "model_id": "entity/project/policy:v0",
            "policy_id": 7,
            "device": "cuda",
            "default_camera_height": 480,
            "default_camera_width": 640,
            "n_action_steps": 6,
        },
    )
    assert status == 200
    assert data["ok"] is True
    assert data["metadata"]["policy_id"] == 7
    assert data["metadata"]["camera_height"] == 224
    assert data["metadata"]["live_camera_keys"] == ["cam_a"]

    status, data = _post(state, "/configure", {"policy_id": 7, "num_action_samples": 16})
    assert status == 200
    assert data["changed"]["num_action_samples"] is True
    assert fake_policy.num_action_samples == 16

    status, data = _post(state, "/set_camera_keys", {"policy_id": 7, "camera_keys": ["cam_a"]})
    assert status == 200
    assert fake_policy.camera_keys == ["cam_a"]

    status, data = _post(state, "/reset", {"policy_id": 7})
    assert status == 200
    assert fake_policy.reset_count == 1

    status, data = _post(
        state,
        "/predict",
        {
            "policy_id": 7,
            "raw_obs": {
                "robot_state": {},
                "image": {"cam_a": np.zeros((2, 2, 3), dtype=np.uint8)},
            },
        },
    )
    assert status == 200
    np.testing.assert_allclose(data["action"], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert data["last_chunk_info"] == {"v_val": 1.25}
    assert fake_policy.last_chunk_info == {}


class FakeRemoteSession:
    control_timeout_s = 30.0
    predict_timeout_s = 30.0

    def __init__(self):
        self.calls = []

    def request(self, method, path, payload=None, *, timeout_s):
        self.calls.append((method, path, payload, timeout_s))
        if path == "/configure":
            return {"changed": {"num_action_samples": True}}
        if path == "/predict":
            assert sorted(payload["raw_obs"]["image"]) == ["cam_a"]
            return {
                "action": np.array([1.0, 2.0], dtype=np.float32),
                "last_chunk_info": {"q_min_best": 0.5},
            }
        return {}


def test_remote_policy_client_matches_rollout_policy_protocol():
    session = FakeRemoteSession()
    policy = inference_server.RemotePolicyClient(
        session,
        policy_id=3,
        metadata={
            "action_space": "cartesian_velocity",
            "env_action_space": "cartesian_velocity",
            "gripper_action_space": None,
            "live_camera_keys": ["cam_a"],
            "config": {"image_features": ["observation.images.cam_a"]},
        },
    )

    assert policy_live_camera_keys(policy, ["cam_a", "cam_b"]) == ["cam_a"]

    policy.set_camera_keys(["cam_a"])
    policy.reset()
    assert policy.set_num_action_samples(32) is True
    action = policy.predict(
        {
            "robot_state": {"cartesian_position": np.zeros(6), "gripper_position": 0.0},
            "image": {
                "cam_a": np.zeros((2, 2, 3), dtype=np.uint8),
                "cam_b": np.ones((2, 2, 3), dtype=np.uint8),
            },
        }
    )

    np.testing.assert_allclose(action, np.array([1.0, 2.0], dtype=np.float32))
    assert policy.last_chunk_info == {"q_min_best": 0.5}
    assert [call[1] for call in session.calls] == [
        "/set_camera_keys",
        "/reset",
        "/configure",
        "/predict",
    ]


class CaptureLoadSession(inference_server.RemoteInferenceSession):
    def __init__(self):
        self.load_timeout_s = 900.0
        self.captured_payload = None

    def request(self, method, path, payload=None, *, timeout_s):
        self.captured_payload = dict(payload)
        return {
            "metadata": {
                "model_id": "wandb://entity/project/policy:v0",
                "policy_id": 0,
                "camera_height": 480,
                "camera_width": 640,
                "action_space": "cartesian_velocity",
                "env_action_space": "cartesian_velocity",
                "gripper_action_space": None,
                "live_camera_keys": None,
                "config": {"image_features": []},
            }
        }


def test_remote_policy_load_uses_server_device_not_robot_local_device():
    session = CaptureLoadSession()

    entry = session.load_policy_entry(
        model_id="wandb://entity/project/policy:v0",
        policy_id=0,
        device="cpu",
    )

    assert entry.model_id == "wandb://entity/project/policy:v0"
    assert "device" not in session.captured_payload


class ChunkInfoPolicy(FakePolicy):
    camera_crops = {"side_1": (10, 20, 330, 260)}

    def __init__(self):
        super().__init__()
        self.episode_chunk_infos = []
        self.last_episode_chunk_infos = []

    def reset(self):
        super().reset()
        self.last_episode_chunk_infos = self.episode_chunk_infos
        self.episode_chunk_infos = []


def test_remote_path_carries_episode_chunk_infos_and_camera_crops(monkeypatch):
    # The local IDQL policy stashes per-chunk diagnostics on reset() and exposes its crops;
    # manifest_eval reads both from the policy object, so the remote client must too.
    fake_policy = ChunkInfoPolicy()
    monkeypatch.setattr(
        inference_server,
        "load_policy_by_model_id",
        lambda **kw: PolicyEntry(
            model_id=kw["model_id"], policy_id=kw["policy_id"], policy=fake_policy
        ),
    )
    state = inference_server._ServerState(device="cpu", token="tok")
    status, data = _post(state, "/load", {"model_id": "hf://mulligan/critic", "policy_id": 2})
    assert status == 200

    class DispatchSession:
        control_timeout_s = predict_timeout_s = 30.0

        def request(self, method, path, payload=None, *, timeout_s):
            status, result = _post(state, path, payload)
            assert status == 200, result
            return result

    client = inference_server.RemotePolicyClient(
        DispatchSession(), policy_id=2, metadata=data["metadata"]
    )
    assert client.camera_crops == {"side_1": (10, 20, 330, 260)}
    assert client.num_action_samples == 4
    assert client.set_num_action_samples(32) is True
    assert client.num_action_samples == 32

    fake_policy.episode_chunk_infos = [{"chunk_ordinal": 0, "q_min_best": 0.5}]
    client.reset()  # rollout_episode's end-of-episode reset
    assert client.last_episode_chunk_infos == [{"chunk_ordinal": 0, "q_min_best": 0.5}]
    assert fake_policy.last_episode_chunk_infos == []

    client.last_episode_chunk_infos = []  # the sidecar writer consumed them
    client.reset()
    assert client.last_episode_chunk_infos == []


REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _serving(argv):
    """Run the server CLI on a free loopback port; yield its base URL."""
    server = inference_server.build_server(inference_server.parse_args(argv))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_refuses_to_start_without_token(capsys, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        inference_server.parse_args(["--port", "0"])
    assert exc.value.code == 2
    assert "--insecure-no-token" in capsys.readouterr().err

    monkeypatch.delenv("UNSET_TOKEN_VAR", raising=False)
    with pytest.raises(SystemExit):
        inference_server.parse_args(["--port", "0", "--token-env", "UNSET_TOKEN_VAR"])
    with pytest.raises(SystemExit):
        inference_server.parse_args(["--port", "0", "--token", "tok", "--insecure-no-token"])


def test_server_defaults_to_loopback_and_reads_token_from_env(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN_VAR", "from-env")
    args = inference_server.parse_args(["--port", "0", "--token-env", "TEST_TOKEN_VAR"])
    assert args.host == "127.0.0.1"
    assert args.token == "from-env"


def test_server_starts_with_insecure_no_token(capsys):
    with _serving(["--port", "0", "--device", "cpu", "--insecure-no-token"]) as url:
        session = inference_server.RemoteInferenceSession(url)
        assert session.health()["ok"] is True
        session.close()
    err = capsys.readouterr().err
    assert "WARNING: --insecure-no-token" in err
    assert "arbitrary code" in err


def test_server_rejects_wrong_or_missing_token():
    # The server handles one keep-alive connection at a time, so each session closes its
    # connection before the next one connects.
    with _serving(["--port", "0", "--device", "cpu", "--token", "tok"]) as url:
        good = inference_server.RemoteInferenceSession(url, token="tok")
        assert good.health()["ok"] is True
        good.close()
        for token in ("wrong", None):
            bad = inference_server.RemoteInferenceSession(url, token=token)
            with pytest.raises(RuntimeError, match="HTTP 401"):
                bad.health()
            with pytest.raises(RuntimeError, match="HTTP 401"):
                bad.request("POST", "/load", {"model_id": "hf://x", "policy_id": 0}, timeout_s=3)
            bad.close()


def test_token_is_checked_before_the_body_is_unpickled():
    state = inference_server._ServerState(device="cpu", token="tok")
    status, data = inference_server._dispatch_request(
        state,
        method="POST",
        path="/load",
        body=b"not a pickle",
        headers={"X-Mulligan-Inference-Token": "wrong"},
    )
    assert status == 401, data


def test_managed_session_passes_token_on_stdin(tmp_path, monkeypatch):
    # A stand-in `ssh` that runs the remote command locally (no port forward, so the
    # local and remote ports are the same free port).
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        '#!/bin/sh\nif [ "$1" = "-G" ]; then exit 0; fi\nfor last; do :; done\nexec sh -c "$last"\n'
    )
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    port = inference_server._free_local_port()
    python_cmd = f"env PYTHONPATH={shlex.quote(str(REPO_ROOT))} {shlex.quote(sys.executable)}"

    session = inference_server.start_managed_remote_inference(
        ssh_host="gpu-host",
        gpu="0",
        workdir=str(REPO_ROOT),
        python_cmd=python_cmd,
        remote_device="cpu",
        local_port=port,
        remote_port=port,
        startup_timeout_s=60.0,
        token="managed-secret",
    )
    try:
        assert not any("managed-secret" in arg for arg in session._process.args)
        # The server requires a token, so a healthy response means it read the right one.
        assert session.health()["ok"] is True
    finally:
        session.close()
    assert session._process.poll() is not None
