"""Remote real-robot policy inference over an SSH-forwarded HTTP tunnel.

The server wraps the same RealWorldPolicy objects used by local eval. The
client exposes the same small protocol that ``rollout_episode`` already uses:
``predict(raw_obs)``, ``reset()``, ``set_camera_keys()``, action-space attrs,
and camera metadata. This keeps robot control, saving, and result bookkeeping
local while moving policy inference to a stronger GPU host.

Trust model: requests and responses are pickles over plain HTTP. The server
unpickles a request body only after the request's ``X-Mulligan-Inference-Token``
header matches its token, but the token is a shared secret sent in the clear:
anyone who holds it, or can read the traffic, can run arbitrary code as the
server's user. The server therefore refuses to start without a token unless
``--insecure-no-token`` is given, binds to 127.0.0.1 by default, and is meant to
be reached through an SSH tunnel or a trusted network. The client unpickles the
server's responses, so point it only at a server you trust.
"""

from __future__ import annotations

import argparse
import atexit
import hmac
import http.server
import os
import pickle
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import requests

if TYPE_CHECKING:
    from mulligan.real.eval.common import PolicyEntry

_WIRE_MIMETYPE = "application/vnd.mulligan.remote-inference.pickle"
_DEFAULT_REMOTE_PORT = 48881
_DEFAULT_LOAD_TIMEOUT_S = 900.0
_DEFAULT_PREDICT_TIMEOUT_S = 30.0
_TOKEN_HEADER = "X-Mulligan-Inference-Token"
_TOKEN_ENV = "MULLIGAN_REMOTE_INFERENCE_TOKEN"


def load_policy_by_model_id(**kwargs: Any) -> PolicyEntry:
    """:func:`mulligan.real.policy.loader.load_policy_by_model_id`, imported on first use.

    The loader pulls in torch and lerobot; deferring it keeps this module, and with it
    :func:`add_remote_inference_args`, importable by the entrypoints' argument parsers
    before the robot stack loads.
    """
    from mulligan.real.policy.loader import load_policy_by_model_id as load

    return load(**kwargs)


def _pack(payload: Any) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _unpack(data: bytes) -> Any:
    return pickle.loads(data)


def _metadata_for_policy(entry: PolicyEntry) -> dict[str, Any]:
    policy = entry.policy
    config = getattr(policy, "config", None)
    image_features = getattr(config, "image_features", {}) or {}
    action_space = getattr(policy, "action_space", None)
    env_action_space = getattr(policy, "env_action_space", action_space)
    if action_space is None:
        raise AttributeError(
            f"Policy {entry.model_id} has no action_space attribute; cannot expose remotely."
        )
    return {
        "model_id": entry.model_id,
        "policy_id": entry.policy_id,
        "camera_height": entry.camera_height,
        "camera_width": entry.camera_width,
        "action_space": action_space,
        "env_action_space": env_action_space,
        "gripper_action_space": getattr(policy, "gripper_action_space", None),
        "live_camera_keys": getattr(policy, "live_camera_keys", None),
        "camera_crops": dict(getattr(policy, "camera_crops", None) or {}),
        "num_action_samples": getattr(policy, "num_action_samples", None),
        "config": {
            "image_features": list(image_features.keys()),
        },
    }


def _config_from_metadata(metadata: dict[str, Any]) -> SimpleNamespace:
    config = dict(metadata.get("config") or {})
    image_feature_keys = config.get("image_features") or []
    return SimpleNamespace(image_features={str(key): None for key in image_feature_keys})


def _filter_raw_obs_for_cameras(raw_obs: dict, camera_keys: Sequence[str]) -> dict:
    if not camera_keys or "image" not in raw_obs:
        return raw_obs
    images = raw_obs["image"]
    missing = [key for key in camera_keys if key not in images]
    if missing:
        raise KeyError(
            f"RemotePolicyClient missing camera(s) {missing}; available={sorted(images.keys())}"
        )
    filtered = dict(raw_obs)
    filtered["image"] = {key: images[key] for key in camera_keys}
    return filtered


class _HTTPError(RuntimeError):
    def __init__(self, status: int, error: str):
        super().__init__(error)
        self.status = status
        self.error = error


@dataclass
class _ServerState:
    device: str
    token: str | None = None
    policies: dict[int, PolicyEntry] | None = None

    def __post_init__(self) -> None:
        if self.policies is None:
            self.policies = {}


def _require_token(state: _ServerState, headers) -> None:
    """Reject the request unless it carries the server token (no-op without a token)."""
    if state.token is None:
        return
    got = headers.get(_TOKEN_HEADER) or ""
    if not hmac.compare_digest(got.encode(), state.token.encode()):
        raise _HTTPError(
            401,
            f"bad or missing {_TOKEN_HEADER} (pass --remote-inference-token or set "
            f"{_TOKEN_ENV} on the client)",
        )


def _payload_from_body(body: bytes) -> dict[str, Any]:
    try:
        data = _unpack(body)
    except Exception as exc:  # noqa: BLE001 - malformed wire payload
        raise _HTTPError(400, f"could not decode request payload: {exc}") from exc
    if not isinstance(data, dict):
        raise _HTTPError(400, f"request payload must be a dict, got {type(data).__name__}")
    return data


def _policy_entry(state: _ServerState, policy_id: int) -> PolicyEntry:
    assert state.policies is not None
    try:
        return state.policies[int(policy_id)]
    except KeyError as exc:
        raise _HTTPError(404, f"policy_id={policy_id} is not loaded") from exc


def _dispatch_request(
    state: _ServerState,
    *,
    method: str,
    path: str,
    body: bytes = b"",
    headers=None,
    server=None,
) -> tuple[int, dict[str, Any]]:
    try:
        _require_token(state, headers or {})
        assert state.policies is not None

        if method == "GET" and path == "/health":
            return 200, {
                "ok": True,
                "pid": os.getpid(),
                "device": state.device,
                "policy_count": len(state.policies),
            }

        if method != "POST":
            raise _HTTPError(404, f"unknown endpoint {method} {path}")
        req = _payload_from_body(body)

        if path == "/load":
            entry = load_policy_by_model_id(
                model_id=str(req["model_id"]),
                policy_id=int(req["policy_id"]),
                device=str(req.get("device") or state.device),
                noise_scheduler=req.get("noise_scheduler"),
                num_inference_steps=req.get("num_inference_steps"),
                default_camera_height=int(req.get("default_camera_height", 480)),
                default_camera_width=int(req.get("default_camera_width", 640)),
                n_action_steps=int(req.get("n_action_steps", 6)),
                dp_artifact_override=req.get("dp_artifact_override"),
            )
            state.policies[entry.policy_id] = entry
            return 200, {"ok": True, "metadata": _metadata_for_policy(entry)}

        if path == "/reset":
            policy = _policy_entry(state, int(req["policy_id"])).policy
            if not hasattr(policy, "reset"):
                raise AttributeError(f"policy_id={req['policy_id']} has no reset()")
            policy.reset()
            # reset() stashes the finished episode's per-chunk diagnostics; hand them to the
            # client, which the rollout loop reads after its end-of-episode reset.
            chunk_infos = list(getattr(policy, "last_episode_chunk_infos", None) or [])
            if chunk_infos:
                policy.last_episode_chunk_infos = []
            return 200, {"ok": True, "last_episode_chunk_infos": chunk_infos}

        if path == "/set_camera_keys":
            policy = _policy_entry(state, int(req["policy_id"])).policy
            camera_keys = list(req["camera_keys"])
            if hasattr(policy, "set_camera_keys"):
                policy.set_camera_keys(camera_keys)
            return 200, {"ok": True}

        if path == "/configure":
            policy = _policy_entry(state, int(req["policy_id"])).policy
            changed: dict[str, bool] = {}
            if "num_action_samples" in req and req["num_action_samples"] is not None:
                if hasattr(policy, "num_action_samples"):
                    policy.num_action_samples = int(req["num_action_samples"])
                    changed["num_action_samples"] = True
                else:
                    changed["num_action_samples"] = False
            return 200, {"ok": True, "changed": changed}

        if path == "/predict":
            entry = _policy_entry(state, int(req["policy_id"]))
            policy = entry.policy
            action = policy.predict(req["raw_obs"])
            action_np = np.asarray(action)
            last_chunk_info = getattr(policy, "last_chunk_info", None) or {}
            if last_chunk_info and hasattr(policy, "last_chunk_info"):
                policy.last_chunk_info = {}
            return 200, {
                "ok": True,
                "action": action_np,
                "last_chunk_info": last_chunk_info,
            }

        if path == "/shutdown":
            if server is not None:
                threading.Thread(target=server.shutdown, daemon=True).start()
            return 200, {"ok": True}

        raise _HTTPError(404, f"unknown endpoint {method} {path}")
    except _HTTPError as exc:
        return exc.status, {"ok": False, "type": type(exc).__name__, "error": exc.error}
    except Exception as exc:  # noqa: BLE001 - return a structured remote failure
        traceback.print_exc()
        return 500, {"ok": False, "type": type(exc).__name__, "error": str(exc)}


class _InferenceRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[remote-inference] {self.address_string()} - {fmt % args}", flush=True)

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        status, payload = _dispatch_request(
            self.server.state,
            method=self.command,
            path=self.path,
            body=body,
            headers=self.headers,
            server=self.server,
        )
        raw = _pack(payload)
        self.send_response(status)
        self.send_header("Content-Type", _WIRE_MIMETYPE)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _InferenceHTTPServer(http.server.HTTPServer):
    def __init__(self, server_address, state: _ServerState):
        self.state = state
        super().__init__(server_address, _InferenceRequestHandler)


class RemotePolicyClient:
    """Client-side RealWorldPolicy facade for one remote server policy."""

    def __init__(
        self,
        session: "RemoteInferenceSession",
        *,
        policy_id: int,
        metadata: dict[str, Any],
    ):
        self._session = session
        self._policy_id = int(policy_id)
        self.action_space = str(metadata["action_space"])
        self.env_action_space = str(metadata.get("env_action_space") or self.action_space)
        self.gripper_action_space = metadata.get("gripper_action_space")
        self.live_camera_keys = metadata.get("live_camera_keys")
        self.camera_crops = {
            key: tuple(box) for key, box in (metadata.get("camera_crops") or {}).items()
        }
        self.num_action_samples = metadata.get("num_action_samples")
        self.config = _config_from_metadata(metadata)
        self._camera_keys: list[str] = []
        self.last_chunk_info: dict[str, Any] = {}
        self.last_episode_chunk_infos: list[dict[str, Any]] = []
        self.debug = bool(metadata.get("debug", False))

    def set_camera_keys(self, camera_keys: list[str]) -> None:
        self._camera_keys = list(camera_keys)
        self._session.request(
            "POST",
            "/set_camera_keys",
            {"policy_id": self._policy_id, "camera_keys": self._camera_keys},
            timeout_s=self._session.control_timeout_s,
        )

    def reset(self) -> None:
        self.last_chunk_info = {}
        result = self._session.request(
            "POST",
            "/reset",
            {"policy_id": self._policy_id},
            timeout_s=self._session.control_timeout_s,
        )
        self.last_episode_chunk_infos = list(result.get("last_episode_chunk_infos") or [])

    def set_num_action_samples(self, num_action_samples: int) -> bool:
        result = self._session.request(
            "POST",
            "/configure",
            {"policy_id": self._policy_id, "num_action_samples": int(num_action_samples)},
            timeout_s=self._session.control_timeout_s,
        )
        changed = bool((result.get("changed") or {}).get("num_action_samples"))
        if changed:
            self.num_action_samples = int(num_action_samples)
        return changed

    def predict(self, raw_obs: dict) -> np.ndarray:
        filtered_obs = _filter_raw_obs_for_cameras(raw_obs, self._camera_keys)
        result = self._session.request(
            "POST",
            "/predict",
            {"policy_id": self._policy_id, "raw_obs": filtered_obs},
            timeout_s=self._session.predict_timeout_s,
        )
        self.last_chunk_info = dict(result.get("last_chunk_info") or {})
        return np.asarray(result["action"])


class RemoteInferenceSession:
    """Connection to a running remote inference server."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        diagnostics_fn: Callable[[], str] | None = None,
        load_timeout_s: float = _DEFAULT_LOAD_TIMEOUT_S,
        predict_timeout_s: float = _DEFAULT_PREDICT_TIMEOUT_S,
        control_timeout_s: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._http = requests.Session()
        self._diagnostics_fn = diagnostics_fn
        self.load_timeout_s = load_timeout_s
        self.predict_timeout_s = predict_timeout_s
        self.control_timeout_s = control_timeout_s

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": _WIRE_MIMETYPE, "Accept": _WIRE_MIMETYPE}
        if self.token is not None:
            headers[_TOKEN_HEADER] = self.token
        return headers

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            if method == "GET":
                response = self._http.get(url, headers=self.headers(), timeout=timeout_s)
            elif method == "POST":
                response = self._http.post(
                    url,
                    data=_pack(payload or {}),
                    headers=self.headers(),
                    timeout=timeout_s,
                )
            else:
                raise ValueError(f"unsupported HTTP method {method!r}")
        except requests.RequestException as exc:
            detail = self._diagnostics()
            raise RuntimeError(
                f"remote inference request {method} {path} failed: {exc}{detail}"
            ) from exc

        try:
            data = _unpack(response.content)
        except Exception as exc:  # noqa: BLE001 - include response preview for diagnostics
            preview = response.text[:2000]
            detail = self._diagnostics()
            raise RuntimeError(
                f"remote inference response {method} {path} was not a valid remote-inference payload "
                f"(HTTP {response.status_code}): {preview}{detail}"
            ) from exc

        if response.status_code >= 400 or not data.get("ok", False):
            detail = self._diagnostics()
            raise RuntimeError(
                f"remote inference {method} {path} failed "
                f"(HTTP {response.status_code}, {data.get('type', 'Error')}): "
                f"{data.get('error', data)}{detail}"
            )
        return data

    def _diagnostics(self) -> str:
        if self._diagnostics_fn is None:
            return ""
        diag = self._diagnostics_fn()
        return f"\nRemote server log tail:\n{diag}" if diag else ""

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health", None, timeout_s=3.0)

    def load_policy_entry(
        self,
        *,
        model_id: str,
        policy_id: int,
        device: str,
        noise_scheduler: str | None = None,
        num_inference_steps: int | None = None,
        default_camera_height: int = 480,
        default_camera_width: int = 640,
        n_action_steps: int = 6,
        dp_artifact_override: str | None = None,
    ) -> PolicyEntry:
        result = self.request(
            "POST",
            "/load",
            {
                "model_id": model_id,
                "policy_id": int(policy_id),
                "noise_scheduler": noise_scheduler,
                "num_inference_steps": num_inference_steps,
                "default_camera_height": int(default_camera_height),
                "default_camera_width": int(default_camera_width),
                "n_action_steps": int(n_action_steps),
                "dp_artifact_override": dp_artifact_override,
            },
            timeout_s=self.load_timeout_s,
        )
        metadata = dict(result["metadata"])
        from mulligan.real.eval.common import PolicyEntry

        return PolicyEntry(
            model_id=str(metadata["model_id"]),
            policy_id=int(metadata["policy_id"]),
            policy=RemotePolicyClient(
                self, policy_id=int(metadata["policy_id"]), metadata=metadata
            ),
            camera_height=int(metadata["camera_height"]),
            camera_width=int(metadata["camera_width"]),
        )

    def close(self) -> None:
        self._http.close()


class _PipeTail:
    def __init__(self, stream, *, max_lines: int = 120):
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self._thread.start()

    def _read(self, stream) -> None:
        for line in iter(stream.readline, ""):
            self._lines.append(line.rstrip("\n"))

    def text(self) -> str:
        return "\n".join(self._lines)


class ManagedRemoteInferenceSession(RemoteInferenceSession):
    """Remote inference session whose server process is owned by this process."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        process: subprocess.Popen[str],
        stdout_tail: _PipeTail,
        stderr_tail: _PipeTail,
    ):
        self._process = process
        self._stdout_tail = stdout_tail
        self._stderr_tail = stderr_tail
        self._closed = False
        super().__init__(base_url, token=token, diagnostics_fn=self.log_tail)

    def log_tail(self) -> str:
        parts = []
        stdout = self._stdout_tail.text()
        stderr = self._stderr_tail.text()
        if stdout:
            parts.append("[stdout]\n" + stdout)
        if stderr:
            parts.append("[stderr]\n" + stderr)
        return "\n".join(parts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self.request("POST", "/shutdown", {}, timeout_s=2.0)
            except Exception:
                pass
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        finally:
            super().close()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(session: ManagedRemoteInferenceSession, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if session._process.poll() is not None:
            raise RuntimeError(
                "remote inference SSH/server process exited before becoming healthy "
                f"(code={session._process.returncode}).\nRemote server log tail:\n"
                f"{session.log_tail()}"
            )
        try:
            session.health()
            return
        except Exception as exc:  # noqa: BLE001 - health retries until timeout
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(
        f"remote inference server did not become healthy within {timeout_s:.1f}s. "
        f"Last error: {last_error}\nRemote server log tail:\n{session.log_tail()}"
    )


def start_managed_remote_inference(
    *,
    ssh_host: str,
    gpu: str,
    workdir: str,
    python_cmd: str,
    remote_device: str,
    local_port: int = 0,
    remote_port: int = _DEFAULT_REMOTE_PORT,
    startup_timeout_s: float = 180.0,
    token: str | None = None,
) -> ManagedRemoteInferenceSession:
    if local_port <= 0:
        local_port = _free_local_port()
    if not token:
        token = secrets.token_urlsafe(24)

    ssh_probe = subprocess.run(
        ["ssh", "-G", ssh_host],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if ssh_probe.returncode != 0:
        raise RuntimeError(
            f"ssh config resolution failed for {ssh_host!r}: {ssh_probe.stderr.strip()}"
        )

    # The token reaches the remote server on ssh's stdin, never on a command line (where
    # `ps` on either host would show it). `sh -c` makes `read` independent of the remote
    # user's login shell.
    server_cmd = (
        f"cd {shlex.quote(workdir)} && "
        f"IFS= read -r {_TOKEN_ENV} && export {_TOKEN_ENV} && "
        "exec env "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))} "
        "PYTHONUNBUFFERED=1 "
        f"{python_cmd} -m mulligan.real.eval.inference_server "
        "--host 127.0.0.1 "
        f"--port {int(remote_port)} "
        f"--device {shlex.quote(remote_device)} "
        f"--token-env {_TOKEN_ENV}"
    )
    remote_cmd = f"sh -c {shlex.quote(server_cmd)}"
    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "-L",
        f"{int(local_port)}:127.0.0.1:{int(remote_port)}",
        ssh_host,
        remote_cmd,
    ]
    process = subprocess.Popen(
        ssh_cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        process.stdin.write(token + "\n")
        process.stdin.close()
    except BrokenPipeError:
        pass  # ssh already exited; _wait_for_health reports its exit code and log tail
    session = ManagedRemoteInferenceSession(
        f"http://127.0.0.1:{int(local_port)}",
        token=token,
        process=process,
        stdout_tail=_PipeTail(process.stdout),
        stderr_tail=_PipeTail(process.stderr),
    )
    atexit.register(session.close)
    _wait_for_health(session, timeout_s=startup_timeout_s)
    print(
        "Remote inference server ready: "
        f"ssh_host={ssh_host}, gpu={gpu}, local_url={session.base_url}, "
        f"remote_port={remote_port}, device={remote_device}"
    )
    return session


def add_remote_inference_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("remote inference")
    group.add_argument(
        "--inference-backend",
        choices=["local", "remote"],
        default="local",
        help="Run policy inference locally or through a remote inference server.",
    )
    group.add_argument(
        "--remote-inference-host",
        default=None,
        help="SSH host (as in ~/.ssh/config) of the GPU machine for auto-managed remote inference.",
    )
    group.add_argument("--remote-inference-gpu", default="1")
    group.add_argument("--remote-inference-url", default=None)
    group.add_argument(
        "--remote-inference-port",
        type=int,
        default=0,
        help="Local forwarded port for auto-managed remote inference (0 = choose a free port).",
    )
    group.add_argument(
        "--remote-inference-remote-port",
        type=int,
        default=_DEFAULT_REMOTE_PORT,
        help=f"Port bound on the remote host loopback interface (default: {_DEFAULT_REMOTE_PORT}).",
    )
    group.add_argument(
        "--remote-inference-workdir",
        default=None,
        help="Repo checkout path on the remote host (default: current local working directory).",
    )
    group.add_argument(
        "--remote-inference-python-cmd",
        default="uv run --no-sync python",
        help="Shell command used on the remote host before '-m mulligan.real.eval.inference_server'.",
    )
    group.add_argument("--remote-inference-device", default="cuda")
    group.add_argument(
        "--remote-inference-token",
        default=None,
        help=(
            f"Shared-secret token the server checks on every request (default: ${_TOKEN_ENV}; "
            "auto-managed mode generates one when neither is set). Prefer the environment "
            "variable: a command-line value is visible in `ps`."
        ),
    )
    group.add_argument("--remote-inference-start-timeout-s", type=float, default=180.0)


def remote_inference_session_from_args(
    args: argparse.Namespace,
    *,
    default_workdir: Path,
) -> RemoteInferenceSession | None:
    if args.inference_backend == "local":
        if args.remote_inference_url:
            raise ValueError(
                "--remote-inference-url was provided but --inference-backend is local; "
                "set --inference-backend remote."
            )
        return None
    token = args.remote_inference_token or os.environ.get(_TOKEN_ENV) or None
    if args.remote_inference_url:
        session = RemoteInferenceSession(args.remote_inference_url, token=token)
        session.health()
        print(f"Remote inference server ready: url={session.base_url} (manual mode)")
        return session
    if not args.remote_inference_host:
        raise ValueError(
            "--inference-backend remote needs --remote-inference-url (manual server) or "
            "--remote-inference-host (auto-managed over SSH)."
        )
    workdir = args.remote_inference_workdir or str(default_workdir)
    return start_managed_remote_inference(
        ssh_host=args.remote_inference_host,
        gpu=str(args.remote_inference_gpu),
        workdir=workdir,
        python_cmd=str(args.remote_inference_python_cmd),
        remote_device=str(args.remote_inference_device),
        local_port=int(args.remote_inference_port),
        remote_port=int(args.remote_inference_remote_port),
        startup_timeout_s=float(args.remote_inference_start_timeout_s),
        token=token,
    )


def _is_loopback(host: str) -> bool:
    return host in {"localhost", "::1"} or host.startswith("127.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the server CLI and resolve ``args.token`` (None only with --insecure-no-token)."""
    parser = argparse.ArgumentParser(
        description=(
            "Mulligan real-robot remote inference server. Requests are pickles over plain "
            "HTTP: anyone who can send an accepted request can run arbitrary code as this "
            "user. Bind to 127.0.0.1 and connect through an SSH tunnel, or use a trusted "
            "network."
        )
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1, reachable only from this host or a tunnel).",
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--token",
        default=None,
        help="Shared-secret request token. Visible in `ps`; prefer --token-env.",
    )
    parser.add_argument(
        "--token-env",
        default=None,
        help=f"Read the request token from this environment variable (e.g. {_TOKEN_ENV}).",
    )
    parser.add_argument(
        "--insecure-no-token",
        action="store_true",
        help=(
            "Accept requests without a token. Every request body is unpickled, so anyone "
            "who can reach the port can run arbitrary code as this user."
        ),
    )
    args = parser.parse_args(argv)
    token = args.token
    if args.token_env:
        token = os.environ.get(args.token_env)
        if not token:
            parser.error(f"--token-env {args.token_env!r} is unset or empty")
    if token and args.insecure_no_token:
        parser.error("--insecure-no-token conflicts with --token/--token-env")
    if not token and not args.insecure_no_token:
        parser.error(
            "a request token is required: pass --token-env VAR (or --token), or "
            "--insecure-no-token to accept unauthenticated requests"
        )
    args.token = token or None
    return args


def build_server(args: argparse.Namespace) -> _InferenceHTTPServer:
    """Bind the server for parsed ``args`` and print the trust-model warnings."""
    where = f"{args.host}:{args.port}"
    if args.token is None:
        print(
            f"WARNING: --insecure-no-token: {where} accepts requests without a token and "
            "unpickles every request body. Anyone who can reach this port can run "
            "arbitrary code as this user.",
            file=sys.stderr,
            flush=True,
        )
    elif not _is_loopback(args.host):
        print(
            f"WARNING: {where} is not a loopback address. Requests are pickles over plain "
            "HTTP and the token is a shared secret sent in the clear: anyone who holds it "
            "or can read the traffic can run arbitrary code as this user. Prefer "
            "--host 127.0.0.1 with an SSH tunnel, or use a trusted network.",
            file=sys.stderr,
            flush=True,
        )
    state = _ServerState(device=args.device, token=args.token)
    return _InferenceHTTPServer((args.host, args.port), state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    server = build_server(args)
    host, port = server.server_address[:2]
    print(
        "MULLIGAN_REMOTE_INFERENCE_READY "
        f"host={host} port={port} pid={os.getpid()} device={args.device}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
