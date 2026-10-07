# ruff: noqa: E402
"""Sim DAgger collector: policy rollouts with human takeover and counterfactual replays.

Each episode starts from a placed start state (``--sampler``), the policy drives
and the operator takes over (``h``) when it is about to fail, hands control back
(``h``) or ends the episode. The whole episode, policy and human segments, is
saved as one LeRobot episode. This is the collector behind the simulated
Mulligan rounds (Square-Narrow = ``NutAssemblySquare``, Square-Broad =
``Square_D1``); the policies are IDQL / DIVL agents, run with best-of-N action
selection (``--num-action-samples``, N=32 in the paper).

Features used by the paper protocol:

* blinded routing (``--routed-policy LABEL=REF`` twice or more, plus
  ``--policy-routing-manifest``): each start state is matched to its source
  label in the manifest and rolled out by that label's policy; the operator is
  not told which policy is driving;
* counterfactual replays (``c`` between episodes): the same start state again,
  with the human driving first, saved as a separate episode;
* ``--adaptive-protocol-quota-*``: simultaneous no-CF / with-CF quotas per arm
  (fresh successes credit both protocols, counterfactual successes with-CF
  only; see :class:`mulligan.sim.collect.quota.ProtocolQuotaLedger`).

Policies are loaded from a local checkpoint directory or a pinned
``hf://<org>/<repo>@<revision>/<subdir>`` reference
(:func:`mulligan.release.hub.resolve_checkpoint`).

Human input goes through an :class:`mulligan.sim.collect.utils.Operator`. The
default is the SpaceMouse + keyboard; ``--operator module:factory`` (or the
``operator`` argument of :func:`main`) plugs in a scripted operator, e.g. a
replay of recorded human segments, so the loop runs without a person.

Dataset fields (per frame): ``observation.state`` (eef pos, eef quat, gripper
qpos), ``observation.environment_state`` (``object-state``), ``action``,
``source`` (0 = policy, 1 = human), ``success`` (episode outcome),
``intervention`` (1 on the last policy frame before a takeover), ``is_valid``
(0 on the padded final frame), ``reward``, ``done``, and
``observation.images.<camera>`` for each ``--cameras`` entry.

Controls (keyboard; SpaceMouse moves the end effector, left button toggles
the gripper):

    policy driving:  h take over | 0 save as recoverable failure (done=False)
                     9 save as terminal failure (done=True) | d discard
    human driving:   h hand back to the policy | 1 save as success | 0 / 9 / d
    between:         n next start | c counterfactual (same start) | q or Ctrl+C quit

With ``--auto-save-on-success`` an episode ends as a success when the env
reports success.

Example (a blinded Square-Broad round with the protocol quota, as in the paper):

    mjpython -m mulligan.sim.collect.dagger \\
        --env Square_D1 --robot Panda \\
        --routed-policy mulligan_policy=hf://mulligan/<mulligan-repo>@<revision>/seed-1 \\
        --routed-policy baseline_policy=hf://mulligan/<baseline-repo>@<revision>/seed-1 \\
        --policy-routing-manifest <manifest.json> \\
        --cameras agentview,robot0_eye_in_hand \\
        --dataset-name square-broad-r01-dagger \\
        --auto-save-on-success \\
        --sampler list --initial-states-file <blind_inputs.json> --no-sampler-shuffle \\
        --adaptive-protocol-quota-manifest <manifest.json> \\
        --adaptive-protocol-quota-targets no_cf=100,with_cf=100 \\
        --adaptive-protocol-quota-arms 'no_cf=baseline_uniform,sobol,mulligan;with_cf=sobol,mulligan' \\
        --adaptive-protocol-quota-ledger ledgers/square_broad_r01.jsonl \\
        --adaptive-protocol-quota-balance-slack 2 \\
        --num-action-samples 32

On Linux use ``python`` instead of ``mjpython``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

if platform.system() == "Darwin" and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "cgl"

from mulligan import apply_runtime_patches

apply_runtime_patches()

import robosuite.macros as macros
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.env_state_names import get_environment_state_names
from mulligan.real.collect.hf_utils import add_license_arg
from mulligan.release.hub import resolve_checkpoint
from mulligan.sim.collect.utils import (
    EpisodeStart,
    Operator,
    SpaceMouseKeyboardOperator,
    OBJECT_VEL_THRESHOLD,
    check_objects_moving,
    robot_state,
    wait_for_save,
)
from mulligan.sim.envs import configure_viewer_shadows, create_robosuite_env, resolve_env_name

warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)

# Camera observations in image convention (origin top-left), as stored in the datasets.
macros.IMAGE_CONVENTION = "opencv"

# Shuffles --sampler list; also seeds the protocol-quota tie-breaks.
SAMPLER_SHUFFLE_SEED = 42

# The quota ledger travels with the dataset on the Hub, so a collection can be
# resumed on another machine.
HUB_LEDGER_PATH = "meta/collection_ledger.jsonl"

POLICY_KEYS = {
    "h": "intervention",
    "0": "failure",
    "9": "terminal_failure",
    "d": "discard",
}
HUMAN_KEYS = {
    "h": "continue",
    "1": "success",
    "0": "failure",
    "9": "terminal_failure",
    "d": "discard",
}


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sim DAgger collection with policy rollouts and human takeovers"
    )

    # Policies
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Checkpoint (local dir or hf://<org>/<repo>@<revision>/<subdir>) for "
        "single-policy mode. Mutually exclusive with --routed-policy.",
    )
    parser.add_argument(
        "--routed-policy",
        action="append",
        default=[],
        metavar="LABEL=REF",
        help="Blinded routing: a manifest source label and the checkpoint that rolls out "
        "the starts with that label. Repeat for every label (at least two). Requires "
        "--policy-routing-manifest and --sampler.",
    )
    parser.add_argument(
        "--policy-routing-manifest",
        type=str,
        default=None,
        help="Blind manifest JSON with 'states' (and 'keys' or '_meta.task'). Each sampled "
        "start is matched against it (L2, tolerance 'match_tolerance', default 1e-3) and "
        "rolled out by the policy of its 'policy_source' / 'source' / single 'sources' "
        "label. A counterfactual replay keeps the policy of the episode it replays.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Robosuite env (NutAssemblySquare or Square_D1, or square_narrow / "
        "square_broad); default: the checkpoint's metadata",
    )
    parser.add_argument(
        "--robot", type=str, default=None, help="Robot (default: the checkpoint's metadata)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        help="Device for policy inference (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=32,
        help="Best-of-N: diffusion action samples ranked by the critic (paper: 32)",
    )

    # Cameras
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated cameras stored as videos (e.g. 'agentview,robot0_eye_in_hand'). "
        "Default: none (state-only dataset). The policies are state-based.",
    )

    # Operator
    parser.add_argument(
        "--operator",
        type=str,
        default="spacemouse",
        help="'spacemouse' (SpaceMouse + keyboard) or 'module.path:factory' returning an "
        "Operator (see mulligan.sim.collect.utils); the factory is called with the "
        "keyword arguments in --operator-kwargs.",
    )
    parser.add_argument(
        "--operator-kwargs",
        type=str,
        default="{}",
        help="JSON object of keyword arguments for a --operator factory",
    )
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.5)

    # Dataset and Hub
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Output dataset name (a directory under --dataset-path; the Hub repo name)",
    )
    parser.add_argument(
        "--hub-namespace",
        type=str,
        default=None,
        help="HF user or org of the dataset repo <namespace>/<dataset-name>. When set, an "
        "existing repo (and its bundled quota ledger) is downloaded to resume a collection "
        "that is not on this machine. Required with --push-to-hub.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset (and the quota ledger) to <hub-namespace>/<dataset-name> at the end",
    )
    add_license_arg(parser)

    # W&B (off unless a project is given)
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Log per-episode collection metrics to this W&B project (default: no W&B)",
    )

    # Runtime
    parser.add_argument("--headless", action="store_true", help="No onscreen viewer")
    parser.add_argument(
        "--max-fr", type=int, default=20, help="Control-rate limit in Hz (0: no limit)"
    )
    parser.add_argument(
        "--auto-save-on-success",
        action="store_true",
        help="End and save an episode as a success when the env reports success",
    )

    # Start states
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["sobol", "list"],
        default=None,
        help="Start-state sampler: sobol (quasi-random) or list (--initial-states-file). "
        "Default: none (robosuite's own reset distribution).",
    )
    parser.add_argument(
        "--initial-states-file",
        type=str,
        default=None,
        help="JSON with a top-level 'states' list (nut_x, nut_y, nut_yaw[, peg_x, peg_y])",
    )
    parser.add_argument(
        "--no-sampler-shuffle",
        action="store_false",
        dest="sampler_shuffle",
        help="Keep the --initial-states-file order",
    )

    # Quotas
    parser.add_argument(
        "--adaptive-protocol-quota-manifest",
        type=str,
        default=None,
        help="Multi-source manifest for simultaneous no-CF / with-CF quotas: fresh successes "
        "credit no_cf and with_cf, counterfactual successes with_cf only, within the "
        "balance constraint.",
    )
    parser.add_argument(
        "--adaptive-protocol-quota-targets",
        type=str,
        default="no_cf=100,with_cf=100",
        help="Per-arm targets, e.g. 'no_cf=100,with_cf=100'",
    )
    parser.add_argument(
        "--adaptive-protocol-quota-arms",
        type=str,
        default=None,
        help="Arms per protocol, e.g. 'no_cf=baseline_uniform,sobol;with_cf=sobol'. "
        "Protocols not listed target every manifest arm.",
    )
    parser.add_argument(
        "--adaptive-protocol-quota-ledger",
        type=str,
        default=None,
        help="JSONL ledger for the protocol quota (required with its manifest)",
    )
    parser.add_argument(
        "--adaptive-protocol-quota-balance-slack",
        type=int,
        default=1,
        help="Maximum per-protocol spread of arm counts after a credit",
    )
    parser.add_argument(
        "--adaptive-protocol-quota-progress-window",
        type=int,
        default=20,
        help="Recent successes used for the pace / ETA display",
    )

    return parser.parse_args(argv)


def _parse_protocol_quota_targets(spec: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--adaptive-protocol-quota-targets entries must be NAME=INT; got {item!r}"
            )
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("protocol target name must not be empty")
        if name in targets:
            raise ValueError(f"duplicate protocol target {name!r}")
        targets[name] = int(value)
    missing = {"no_cf", "with_cf"} - set(targets)
    if missing:
        raise ValueError(
            "--adaptive-protocol-quota-targets must include no_cf and with_cf; "
            f"missing {sorted(missing)}"
        )
    return targets


def _parse_protocol_quota_arms(spec: str | None) -> dict[str, list[str]] | None:
    if spec is None:
        return None
    arms_by_protocol: dict[str, list[str]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--adaptive-protocol-quota-arms entries must be NAME=ARM[,ARM...]; got {item!r}"
            )
        protocol, raw_arms = item.split("=", 1)
        protocol = protocol.strip()
        if not protocol:
            raise ValueError("protocol arm allowlist name must not be empty")
        if protocol in arms_by_protocol:
            raise ValueError(f"duplicate protocol arm allowlist for {protocol!r}")
        arms = [arm.strip() for arm in raw_arms.split(",") if arm.strip()]
        if not arms:
            raise ValueError(f"protocol {protocol!r} must list at least one arm")
        if len(set(arms)) != len(arms):
            raise ValueError(f"duplicate arm in protocol {protocol!r}: {arms}")
        arms_by_protocol[protocol] = arms
    if not arms_by_protocol:
        raise ValueError("--adaptive-protocol-quota-arms must not be empty if set")
    return arms_by_protocol


def _parse_routed_policies(specs: list[str]) -> dict[str, str]:
    routed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--routed-policy must be LABEL=REF; got {spec!r}")
        label, ref = (part.strip() for part in spec.split("=", 1))
        if not label or not ref:
            raise ValueError(f"--routed-policy has an empty label or reference: {spec!r}")
        if label in routed:
            raise ValueError(f"duplicate --routed-policy label {label!r}")
        routed[label] = ref
    return routed


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class IDQLActor:
    """State-based IDQL / DIVL policy with z-score state and action normalization."""

    def __init__(self, policy, preprocessor, postprocessor, device: str):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, robot_state: np.ndarray, env_state: np.ndarray) -> np.ndarray:
        state = np.concatenate([robot_state, env_state])
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state_normalized = self.preprocessor.normalize_state(state_tensor)
        robot_dim = len(robot_state)
        batch = {
            "observation.state": state_normalized[:, :robot_dim],
            "observation.environment_state": state_normalized[:, robot_dim:],
        }
        action = self.postprocessor.denormalize_action(self.policy.select_action(batch))
        if action.dim() == 2:
            action = action[0]
        return action.cpu().numpy()


def load_idql_actor(
    ref: str,
    *,
    device: str,
    num_action_samples: Optional[int],
) -> tuple[IDQLActor, dict]:
    """Load a sim IDQL / DIVL checkpoint; return the actor and its ``metadata.json``."""
    from mulligan.agents.idql import IDQLPolicy
    from mulligan.training.checkpoint_utils import load_checkpoint_metadata
    from mulligan.utils.load_pretrained import load_policy_from_checkpoint

    checkpoint_dir = resolve_checkpoint(ref)
    print(f"Loading policy {ref} from {checkpoint_dir}")
    policy, preprocessor, postprocessor = load_policy_from_checkpoint(
        checkpoint_path=checkpoint_dir, device=device
    )
    if not isinstance(policy, IDQLPolicy):
        raise ValueError(
            f"{ref}: the sim DAgger collector runs IDQL / DIVL checkpoints; "
            f"got {type(policy).__name__}"
        )
    policy.eval()
    if num_action_samples is not None:
        policy.config.num_action_samples = num_action_samples
    print(
        f"  {type(policy).__name__}: state_dim={policy.state_dim}, "
        f"action_dim={policy.action_dim}, "
        f"num_action_samples={policy.config.num_action_samples}"
    )
    metadata = load_checkpoint_metadata(checkpoint_dir)
    return IDQLActor(policy, preprocessor, postprocessor, device), metadata


def start_state_vector(state) -> np.ndarray:
    """Flatten a sampler start state to (x, y, yaw) or (x, y, yaw, peg_x, peg_y)."""
    if len(state) == 2:
        (nut_x, nut_y, nut_yaw), (peg_x, peg_y) = state
        return np.array([nut_x, nut_y, nut_yaw, peg_x, peg_y], dtype=np.float64)
    x, y, yaw = state
    return np.array([x, y, yaw], dtype=np.float64)


class PolicyRouter:
    """Blinded routing of start states to policies through a blind manifest."""

    def __init__(self, manifest_path: Path, actors_by_source: dict[str, Any]):
        from scipy.spatial import cKDTree

        if len(actors_by_source) < 2:
            raise ValueError("Blinded routing needs at least two --routed-policy labels")
        payload = json.loads(Path(manifest_path).read_text())
        keys = payload.get("keys")
        if keys is None:
            meta = payload.get("_meta")
            if not isinstance(meta, dict):
                raise KeyError(
                    f"{manifest_path}: routing manifest must define top-level 'keys' or a "
                    "dict '_meta' with the task"
                )
            task = str(meta["task"])
            if task == "square_narrow":
                keys = ["nut_x", "nut_y", "nut_yaw"]
            elif task == "square_broad":
                keys = ["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"]
            else:
                raise KeyError(
                    f"{manifest_path}: cannot infer routing keys for task={task!r}; "
                    "write top-level 'keys' into the manifest"
                )
        if len(keys) not in (3, 5):
            raise ValueError(f"{manifest_path}: routing supports 3 or 5 keys, got {keys}")
        states = payload["states"]
        sources = []
        for row_idx, state in enumerate(states):
            if "policy_source" in state:
                sources.append(str(state["policy_source"]))
            elif "source" in state:
                sources.append(str(state["source"]))
            else:
                labels = [str(s) for s in state.get("sources", [])]
                if len(labels) != 1:
                    raise ValueError(
                        "Policy-routing manifest entries must have exactly one source "
                        f"label unless policy_source is set; entry {row_idx} has {labels!r}"
                    )
                sources.append(labels[0])
        unknown = set(sources) - set(actors_by_source)
        if unknown:
            raise ValueError(
                f"Routing manifest has source labels without a policy: {sorted(unknown)}; "
                f"--routed-policy labels are {sorted(actors_by_source)}"
            )
        self.keys = list(keys)
        self.sources = sources
        self.tolerance = float(payload.get("match_tolerance", 1e-3))
        self.actors_by_source = actors_by_source
        self._tree = cKDTree(np.array([[s[k] for k in self.keys] for s in states], dtype=float))
        print(
            f"Routing manifest {manifest_path}: {len(states)} states, "
            f"sources={sorted(set(sources))}, match_tolerance={self.tolerance}"
        )

    def route(self, start_state) -> tuple[str, Any]:
        vec = start_state_vector(start_state)
        if len(vec) != len(self.keys):
            raise ValueError(f"start state {start_state} does not match routing keys {self.keys}")
        dist, idx = self._tree.query(vec, k=1)
        if dist > self.tolerance:
            raise RuntimeError(
                "BLINDING ROUTING ERROR: sampled state did not match any manifest entry "
                f"within tolerance (min L2 dist {float(dist):.4g} > {self.tolerance}). "
                "Refusing to route to prevent mis-attribution. Check that "
                "--initial-states-file and --policy-routing-manifest belong to the same "
                "blinded collection."
            )
        source = self.sources[int(idx)]
        return source, self.actors_by_source[source]


# ---------------------------------------------------------------------------
# Start states
# ---------------------------------------------------------------------------


def build_sampler(args: argparse.Namespace, env_name: str):
    """Square start-state sampler for ``--sampler`` (None without one)."""
    from mulligan.sampling.sobol import (
        SobolSampler,
        SquareD1ListSampler,
        SquareD1SobolSampler,
        SquareListSampler,
    )

    if args.sampler is None:
        return None
    if env_name not in ("NutAssemblySquare", "Square_D1"):
        raise ValueError(f"--sampler supports NutAssemblySquare and Square_D1, not {env_name}")
    if args.sampler == "sobol":
        return SquareD1SobolSampler() if env_name == "Square_D1" else SobolSampler()
    if not args.initial_states_file:
        raise ValueError("--sampler list requires --initial-states-file")
    list_cls = SquareD1ListSampler if env_name == "Square_D1" else SquareListSampler
    return list_cls(
        states_file=Path(args.initial_states_file),
        seed=SAMPLER_SHUFFLE_SEED,
        shuffle=args.sampler_shuffle,
    )


def place_start_state(env, sampler, state) -> str:
    """Place the nut (and the Square_D1 peg) at ``state``; return a description."""
    from mulligan.sampling.sobol import SquareD1SobolSampler
    from mulligan.sim.placement import place_nut, place_square_broad_state

    if isinstance(sampler, SquareD1SobolSampler):
        nut_state, peg_state = state
        place_square_broad_state(env, sampler, nut_state, peg_state)
        (x, y, yaw), (px, py) = nut_state, peg_state
        return (
            f"nut at x={x:.4f}, y={y:.4f}, yaw={np.degrees(yaw):.1f}deg; "
            f"peg at x={px:.4f}, y={py:.4f}"
        )
    x, y, yaw = state
    place_nut(env, sampler.to_qpos(x, y, yaw))
    return f"nut at x={x:.4f}, y={y:.4f}, yaw={np.degrees(yaw):.1f}deg"


def record_sampler_start(sampler, env_state_0: np.ndarray) -> None:
    """Record the observed start of a saved episode; advances the sampler."""
    from mulligan.sampling.sobol import SquareD1SobolSampler
    from mulligan.utils.state_to_grid import (
        extract_nut_pose_from_env_state,
        extract_peg_pos_from_env_state,
    )

    if isinstance(sampler, SquareD1SobolSampler):
        x, y, yaw = extract_nut_pose_from_env_state(env_state_0, task="Square_D1")
        peg_x, peg_y, _ = extract_peg_pos_from_env_state(env_state_0)
        sampler.record_state((x, y, yaw), (peg_x, peg_y))
    else:
        x, y, yaw = extract_nut_pose_from_env_state(env_state_0)
        sampler.record_state(x, y, yaw)


# ---------------------------------------------------------------------------
# Episode segments
# ---------------------------------------------------------------------------


def _empty_segment(obs: dict, camera_names: list[str]) -> dict:
    data = {"observations": [], "environment_state": [], "actions": [], "rewards": [], "dones": []}
    for cam in camera_names:
        if f"{cam}_image" in obs:
            data[f"{cam}_image"] = []
    return data


def _append_step(data: dict, obs: dict, action: np.ndarray, camera_names: list[str]) -> None:
    """Store obs[t] with the action chosen from it (before stepping)."""
    data["observations"].append(robot_state(obs))
    data["environment_state"].append(obs["object-state"])
    data["actions"].append(action.copy())
    for cam in camera_names:
        key = f"{cam}_image"
        if key in data:
            data[key].append(obs[key].copy())


def _limit_rate(start: float, max_fr: int) -> None:
    if max_fr:
        remaining = 1 / max_fr - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)


def _announce_success(detail: str) -> None:
    print("\n" + "*" * 60)
    print(f"TASK SUCCESSFUL! {detail}")
    print("*" * 60 + "\n")


def _gripper_hold_action(gripper_qpos) -> float:
    """robosuite gripper command that keeps the gripper as it is (-1 open, +1 closed).

    Parallel-jaw fingers sit near +-0.04 when open and near 0 when closed.
    """
    return -1.0 if np.mean(np.abs(gripper_qpos)) >= 0.02 else 1.0


def policy_rollout_segment(
    env,
    actor,
    operator: Operator,
    *,
    camera_names: list[str],
    max_fr: int,
    has_renderer: bool,
    auto_save_on_success: bool,
    initial_gripper_action: Optional[float],
) -> tuple[Optional[dict], str, bool, float]:
    """Let the policy drive from the current state until an operator key or auto-save.

    Returns (segment data or None if discarded, outcome, task_success,
    last gripper action). Outcomes: ``intervention``, ``complete``
    (auto-saved success), ``failure``, ``terminal_failure``, ``discard``.
    """
    obs = env._get_observations()
    if has_renderer:
        env.render()
    data = _empty_segment(obs, camera_names)

    print("\n" + "=" * 60)
    print("POLICY ROLLOUT (no timeout)")
    if auto_save_on_success:
        print("  Auto-save on success is on")
    print("  'h' take over | '0' recoverable failure | '9' terminal failure | 'd' discard")
    print("=" * 60 + "\n")

    if initial_gripper_action is not None:
        last_gripper_action = initial_gripper_action
    elif "robot0_gripper_qpos" in obs:
        last_gripper_action = _gripper_hold_action(obs["robot0_gripper_qpos"])
    else:
        last_gripper_action = -1.0  # open

    step_count = 0
    task_success = False
    while True:
        start = time.time()
        outcome = POLICY_KEYS.get(operator.poll_key("policy"))
        if outcome is not None:
            print(f"\n>>> {outcome.upper()} after {step_count} policy steps")
            if outcome == "discard":
                return None, outcome, False, last_gripper_action
            if outcome == "intervention":
                return data, outcome, task_success, last_gripper_action
            return data, outcome, False, last_gripper_action

        action = actor.act(robot_state(obs), obs["object-state"])
        last_gripper_action = action[-1]
        _append_step(data, obs, action, camera_names)

        obs, reward, done, _ = env.step(action)
        if has_renderer:
            env.render()
        data["rewards"].append(reward)
        # Sparse success reward ends the MDP.
        data["dones"].append(True if reward == 1.0 else done)

        if not task_success and env._check_success():
            task_success = True
            _announce_success("The policy completed the task.")
            if auto_save_on_success:
                print(f"Auto-save: policy success in {step_count + 1} steps")
                return data, "complete", True, last_gripper_action

        _limit_rate(start, max_fr)
        step_count += 1


def human_correction_segment(
    env,
    operator: Operator,
    *,
    camera_names: list[str],
    max_fr: int,
    has_renderer: bool,
    auto_save_on_success: bool,
    initial_gripper_action: Optional[float],
    record_gripper_motion: bool = True,
    gripper_vel_threshold: float = 0.01,
) -> tuple[Optional[dict], str, bool, float]:
    """Let the operator drive from the current state.

    The env steps every control step; a step is recorded once a clear input
    started recording and then while there is input, the gripper moves or an
    object moves. Unrecorded steps execute a zero arm action with the current
    gripper command, so no unrecorded arm motion happens.

    Returns (segment data or None if discarded, outcome, task_success,
    last gripper action). Outcomes: ``continue`` (hand back to the policy),
    ``success``, ``failure``, ``terminal_failure``, ``discard``.
    """
    obs = env._get_observations()
    if has_renderer:
        env.render()

    # Continue the current gripper command across the switch (the gripper
    # takes time to move, so the last action is more reliable than qpos).
    if initial_gripper_action is not None:
        gripper_action = -1.0 if initial_gripper_action < 0.0 else 1.0
        print(f"Gripper command continues from action {initial_gripper_action:.2f}")
    elif "robot0_gripper_qpos" in obs:
        gripper_action = _gripper_hold_action(obs["robot0_gripper_qpos"])
        print(f"Gripper command from qpos: physically {'open' if gripper_action < 0 else 'closed'}")
    else:
        gripper_action = -1.0
        print("Warning: no gripper qpos in observations; gripper command -1.0")
    operator.begin_human_segment(gripper_action)
    data = _empty_segment(obs, camera_names)

    print("\n" + "=" * 60)
    print("HUMAN CORRECTION")
    if auto_save_on_success:
        print("  Auto-save on success is on")
    print("  'h' back to policy | '1' success | '0' recoverable failure")
    print("  '9' terminal failure | 'd' discard")
    print(f"  Recording on input, gripper motion or object motion > {OBJECT_VEL_THRESHOLD} m/s")
    print("=" * 60 + "\n")

    step_count = 0
    prev_gripper = gripper_action
    last_gripper_action = gripper_action
    gripper_is_moving = False
    task_success = False
    recording_started = False

    while True:
        start = time.time()
        outcome = HUMAN_KEYS.get(operator.poll_key("human"))
        if outcome is not None:
            print(f"\n>>> {outcome.upper()} after {step_count} recorded human steps")
            if outcome == "discard":
                return None, outcome, False, last_gripper_action
            if outcome == "continue":
                return data, outcome, task_success, last_gripper_action
            return data, outcome, outcome == "success", last_gripper_action

        human = operator.human_input()
        action = np.asarray(human.action, dtype=np.float64)
        gripper = float(action[-1])
        gripper_toggled = gripper != prev_gripper
        if gripper_toggled and record_gripper_motion:
            gripper_is_moving = True
        last_gripper_action = gripper

        has_input = human.arm_active or gripper_toggled or gripper_is_moving
        if not recording_started and (human.strong or gripper_toggled):
            recording_started = True
            print("Recording started (clear operator input)")
        should_record = recording_started and (has_input or check_objects_moving(env))

        if should_record:
            _append_step(data, obs, action, camera_names)
            executed = action
        else:
            executed = np.concatenate([np.zeros(6), [gripper]])

        obs, reward, done, _ = env.step(executed)
        if has_renderer:
            env.render()

        if should_record:
            data["rewards"].append(reward)
            data["dones"].append(True if reward == 1.0 else done)
            if not task_success and env._check_success():
                task_success = True
                _announce_success("The operator completed the task.")
                if auto_save_on_success:
                    print(f"Auto-save: success after {step_count} recorded human steps")
                    return data, "success", True, last_gripper_action
            if gripper_is_moving and record_gripper_motion:
                max_qvel = np.max(np.abs(obs["robot0_gripper_qvel"]))
                if max_qvel < gripper_vel_threshold:
                    gripper_is_moving = False
                    print(f"Gripper motion complete (max qvel: {max_qvel:.4f})")
            step_count += 1

        prev_gripper = gripper
        _limit_rate(start, max_fr)


# ---------------------------------------------------------------------------
# Episode data and saving
# ---------------------------------------------------------------------------


def accumulate_segment_data(
    accumulated_data: Optional[dict], segment_data: dict, source_id: int
) -> tuple[dict, list[int]]:
    """Append a segment to the episode; return the episode and the segment's source labels."""
    if accumulated_data is None:
        accumulated_data = {key: list(values) for key, values in segment_data.items()}
    else:
        for key, values in segment_data.items():
            accumulated_data[key].extend(values)
    return accumulated_data, [source_id] * len(segment_data["actions"])


def compute_intervention_flags(sources: list[int]) -> list[int]:
    """1 on the last policy frame before a policy -> human switch, else 0."""
    flags = [0] * len(sources)
    for i in range(len(sources) - 1):
        if sources[i] == 0 and sources[i + 1] == 1:
            flags[i] = 1
    return flags


def dataset_features(
    env_name: str, episode_data: dict, camera_names: list[str]
) -> dict[str, dict[str, Any]]:
    """LeRobot features of a DAgger dataset (names as in the released datasets)."""
    env_state_dim = episode_data["environment_state"][0].shape[0]
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (episode_data["observations"][0].shape[0],),
            "names": [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "eef_quat_x",
                "eef_quat_y",
                "eef_quat_z",
                "eef_quat_w",
                "gripper_qpos_left",
                "gripper_qpos_right",
            ],
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (env_state_dim,),
            "names": get_environment_state_names(env_name, env_state_dim),
        },
        "action": {
            "dtype": "float32",
            "shape": (episode_data["actions"][0].shape[0],),
            "names": [
                "delta_eef_pos_x",
                "delta_eef_pos_y",
                "delta_eef_pos_z",
                "delta_eef_rot_x",
                "delta_eef_rot_y",
                "delta_eef_rot_z",
                "gripper_action",
            ],
        },
        "source": {"dtype": "int64", "shape": (1,), "names": ["source_id"]},
        "success": {"dtype": "int64", "shape": (1,), "names": ["success_flag"]},
        "intervention": {"dtype": "int64", "shape": (1,), "names": ["intervention_flag"]},
        "is_valid": {"dtype": "int64", "shape": (1,), "names": ["is_valid_flag"]},
        "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "done": {"dtype": "int64", "shape": (1,), "names": ["done_flag"]},
    }
    for cam in camera_names:
        key = f"{cam}_image"
        if key in episode_data:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": episode_data[key][0].shape,
                "names": ["height", "width", "channels"],
            }
    return features


def save_episode_to_dataset(
    dataset: LeRobotDataset,
    episode_data: dict,
    task_name: str,
    camera_names: list[str],
    sources: list[int],
    success: bool,
) -> None:
    """Write one episode: per-frame source labels, episode-level success."""
    num_frames = len(episode_data["actions"])
    if len(sources) != num_frames:
        raise ValueError(f"{len(sources)} source labels for {num_frames} frames")
    intervention_flags = compute_intervention_flags(sources)
    success_flag = 1 if success else 0
    for i in range(num_frames):
        frame = {
            "task": task_name,
            "observation.state": episode_data["observations"][i].astype(np.float32),
            "observation.environment_state": episode_data["environment_state"][i].astype(
                np.float32
            ),
            "action": episode_data["actions"][i].astype(np.float32),
            "source": np.array([sources[i]], dtype=np.int64),
            "success": np.array([success_flag], dtype=np.int64),
            "intervention": np.array([intervention_flags[i]], dtype=np.int64),
            # The final frame carries a padded action.
            "is_valid": np.array([0 if i == num_frames - 1 else 1], dtype=np.int64),
            "reward": np.array([episode_data["rewards"][i]], dtype=np.float32),
            "done": np.array([episode_data["dones"][i]], dtype=np.int64),
        }
        for cam in camera_names:
            key = f"{cam}_image"
            if key in episode_data:
                frame[f"observation.images.{cam}"] = episode_data[key][i]
        dataset.add_frame(frame)
    dataset.save_episode()
    print(
        f"  [bg] Saved episode: success={success}, frames={num_frames} "
        f"(policy={sources.count(0)}, human={sources.count(1)}, "
        f"interventions={sum(intervention_flags)})"
    )


def save_episode_to_dataset_and_append_ledger(
    *,
    dataset: LeRobotDataset,
    episode_data: dict,
    task_name: str,
    camera_names: list[str],
    sources: list[int],
    success: bool,
    quota_ledger=None,
    quota_row: dict | None = None,
) -> None:
    """Save an episode, then append its already-reserved quota ledger row."""
    save_episode_to_dataset(
        dataset=dataset,
        episode_data=episode_data,
        task_name=task_name,
        camera_names=camera_names,
        sources=sources,
        success=success,
    )
    if quota_ledger is not None and quota_row is not None:
        quota_ledger.append_reserved_row(quota_row)


def _wait_for_save(save_future: concurrent.futures.Future, poll_s: float = 60.0) -> None:
    try:
        wait_for_save(save_future, poll_s=poll_s)
    except Exception as err:
        raise RuntimeError(f"Episode save failed: {err}") from err


def _check_completed_saves(save_futures: list[concurrent.futures.Future]) -> None:
    """Raise on failed background saves without blocking on running ones."""
    pending = []
    for future in save_futures:
        if future.done():
            _wait_for_save(future)
        else:
            pending.append(future)
    save_futures[:] = pending


def _finalize_episode_data(
    env,
    accumulated_episode_data: Optional[dict],
    accumulated_sources: list[int],
    episode_task_success: bool,
    is_terminal_failure: bool,
    camera_names: list[str],
) -> bool:
    """Append the final observation (T+1 observations for T actions) and pad the rest.

    Needs the env, so it runs before the episode is handed to the background
    saver. Returns False if there is nothing to save.
    """
    if accumulated_episode_data is None or len(accumulated_sources) == 0:
        return False
    final_obs = env._get_observations()
    data = accumulated_episode_data
    data["observations"].append(robot_state(final_obs))
    data["environment_state"].append(final_obs["object-state"])
    data["actions"].append(data["actions"][-1].copy())
    data["rewards"].append(data["rewards"][-1])
    data["dones"].append(bool(is_terminal_failure or episode_task_success))
    for cam in camera_names:
        key = f"{cam}_image"
        if key in final_obs and key in data:
            data[key].append(final_obs[key].copy())
    accumulated_sources.append(accumulated_sources[-1])
    outcome = (
        "terminal" if is_terminal_failure else "success" if episode_task_success else "recoverable"
    )
    print(
        f"Episode prepared: {len(accumulated_sources)} frames "
        f"(policy={accumulated_sources.count(0)}, human={accumulated_sources.count(1)}), "
        f"done={outcome}"
    )
    return True


# ---------------------------------------------------------------------------
# Hugging Face Hub (optional)
# ---------------------------------------------------------------------------


def restore_quota_ledger_from_hub(repo_id: str, ledger_path: Path) -> None:
    """Download the ledger bundled in ``repo_id`` if it is not present locally.

    A repo or ledger that does not exist yet is the normal state of a fresh
    collection.
    """
    from huggingface_hub import HfApi, hf_hub_download

    if ledger_path.exists():
        return
    api = HfApi()
    if not api.repo_exists(repo_id, repo_type="dataset"):
        return
    if not api.file_exists(repo_id, HUB_LEDGER_PATH, repo_type="dataset"):
        print(f"{repo_id} has no bundled quota ledger; starting a new ledger")
        return
    cached = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=HUB_LEDGER_PATH)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, ledger_path)
    print(f"Restored quota ledger from {repo_id}:{HUB_LEDGER_PATH} -> {ledger_path}")


def bundle_quota_ledger_to_hub(repo_id: str, ledger_path: Path) -> None:
    """Upload the ledger into the dataset repo (after ``push_to_hub``, so they match)."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(ledger_path),
        path_in_repo=HUB_LEDGER_PATH,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update collection quota ledger",
    )
    print(f"Bundled quota ledger into {repo_id}:{HUB_LEDGER_PATH}")


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------


class DaggerCollector:
    """One collection session: env, sampler, quotas, dataset and the episode loop."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        operator: Operator,
        actor,
        env_name: str,
        robot_name: str,
        router: Optional[PolicyRouter] = None,
        wandb_run=None,
    ):
        _check_operator(operator, args)
        self.args = args
        self.operator = operator
        self.actor = actor
        self.router = router
        self.env_name = resolve_env_name(env_name)
        self.robot_name = robot_name
        self.task_name = f"{self.env_name}_{robot_name}"
        self.wandb_run = wandb_run
        self.has_renderer = not args.headless
        self.camera_names = (
            [cam.strip() for cam in args.cameras.split(",") if cam.strip()] if args.cameras else []
        )
        self.dataset_path = Path(args.dataset_path) / args.dataset_name
        self.hub_repo_id = (
            f"{args.hub_namespace}/{args.dataset_name}" if args.hub_namespace else None
        )
        self.quota_ledger_path = args.adaptive_protocol_quota_ledger

        self.dataset: Optional[LeRobotDataset] = None
        self.base_episode_count = 0
        self.saved_episode_count = 0
        self.env = None
        self.sampler = None
        self.sampler_label = (args.sampler or "").upper()
        self.protocol_quota = None
        self.protocol_rng = np.random.default_rng(SAMPLER_SHUFFLE_SEED)

        self.current_state = None
        self.current_manifest_idx: Optional[int] = None
        self.current_match_dist: Optional[float] = None
        self.current_source: Optional[str] = None
        self.active_actor = actor
        self.save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.save_futures: list[concurrent.futures.Future] = []

    # -- setup ------------------------------------------------------------

    def setup(self) -> None:
        args = self.args
        if "/" in args.dataset_name:
            raise ValueError("--dataset-name is a plain name; set the owner with --hub-namespace")
        if args.push_to_hub and not args.hub_namespace:
            raise ValueError("--push-to-hub requires --hub-namespace")
        if self.router is not None and args.sampler is None:
            raise ValueError("Blinded routing needs --sampler (routing is by start state)")

        self.sampler = build_sampler(args, self.env_name)
        if self.hub_repo_id and self.quota_ledger_path:
            restore_quota_ledger_from_hub(self.hub_repo_id, Path(self.quota_ledger_path))
        if args.adaptive_protocol_quota_manifest:
            self._setup_protocol_quota()
        self._open_existing_dataset()
        if self.sampler is not None:
            self._seed_sampler()

    def _setup_protocol_quota(self) -> None:
        from mulligan.sim.collect.quota import ProtocolQuotaLedger

        args = self.args
        if args.sampler != "list":
            raise ValueError("--adaptive-protocol-quota-manifest requires --sampler list")
        if not args.adaptive_protocol_quota_ledger:
            raise ValueError(
                "--adaptive-protocol-quota-ledger is required with "
                "--adaptive-protocol-quota-manifest"
            )
        quota = ProtocolQuotaLedger(
            manifest_path=Path(args.adaptive_protocol_quota_manifest),
            targets_by_protocol=_parse_protocol_quota_targets(args.adaptive_protocol_quota_targets),
            ledger_path=Path(args.adaptive_protocol_quota_ledger),
            arms_by_protocol=_parse_protocol_quota_arms(args.adaptive_protocol_quota_arms),
            balance_slack=args.adaptive_protocol_quota_balance_slack,
            progress_window=args.adaptive_protocol_quota_progress_window,
        )
        if len(quota.keys) not in (3, 5):
            raise ValueError(f"protocol quota supports 3 or 5 manifest keys, got {quota.keys}")
        self.protocol_quota = quota
        self.sampler._next_idx = len(quota.fresh_consumed_manifest_idxs)
        print("\nAdaptive protocol-quota collection: ENABLED")
        print(f"  manifest: {args.adaptive_protocol_quota_manifest}")
        print(f"  ledger:   {args.adaptive_protocol_quota_ledger}")
        print(
            "  targets:  " + ", ".join(f"{p}={t}/arm" for p, t in quota.targets_by_protocol.items())
        )
        print(
            "  arm sets: "
            + "; ".join(f"{p}={','.join(arms)}" for p, arms in quota.arms_by_protocol.items())
        )
        print(f"  balance:  max arm-count spread <= {quota.balance_slack}")
        for line in quota.progress_lines(saved_episode_count=0):
            print(f"  {line}" if line == "Progress" else line)

    def _open_existing_dataset(self) -> None:
        """Resume an existing output dataset (downloading it from the Hub if configured)."""
        fresh_protocol_quota = (
            self.protocol_quota is not None and self.protocol_quota.n_saved_rows == 0
        )
        if not self.dataset_path.exists() and self.hub_repo_id and not fresh_protocol_quota:
            from huggingface_hub import HfApi, snapshot_download

            if HfApi().repo_exists(self.hub_repo_id, repo_type="dataset"):
                print(f"Downloading dataset from the Hub: {self.hub_repo_id}")
                snapshot_download(
                    repo_id=self.hub_repo_id,
                    repo_type="dataset",
                    local_dir=str(self.dataset_path),
                )
        if not self.dataset_path.exists():
            return

        dataset = LeRobotDataset.resume(repo_id=self.args.dataset_name, root=self.dataset_path)
        has_video = any(key.startswith("observation.images.") for key in dataset.features)
        if has_video != bool(self.camera_names):
            raise RuntimeError(
                f"{self.dataset_path} {'has' if has_video else 'has no'} video features but "
                f"--cameras is {self.args.cameras!r}; resume with the same cameras or start "
                "a fresh dataset and ledger"
            )
        if (
            self.protocol_quota is not None
            and dataset.num_episodes != self.protocol_quota.n_saved_rows
        ):
            raise RuntimeError(
                "Protocol quota dataset/ledger mismatch: "
                f"{self.dataset_path} has {dataset.num_episodes} episode(s), but "
                f"{self.protocol_quota.ledger_path} has {self.protocol_quota.n_saved_rows} "
                "row(s). Purge both together or restore the matching pair."
            )
        self.dataset = dataset
        self.base_episode_count = dataset.num_episodes
        print(f"Resuming {self.dataset_path} ({self.base_episode_count} episodes)")

    def _seed_sampler(self) -> None:
        """Skip start states that already have episodes (not in protocol-quota mode)."""
        sampler = self.sampler
        if self.protocol_quota is None:
            if self.dataset_path.exists():
                sampler.load_from_dataset(self.dataset_path)
        print(
            f"\n{self.sampler_label} sampler: {len(sampler.collected_points)} seed points, "
            f"next index {sampler._next_idx}"
        )

    # -- per-episode helpers ---------------------------------------------

    def _quota_vector(self, quota, state) -> np.ndarray:
        from mulligan.sim.collect.quota import vector_from_broad_state, vector_from_narrow_state

        return (
            vector_from_narrow_state(state)
            if len(quota.keys) == 3
            else vector_from_broad_state(state)
        )

    def _state_from_manifest_idx(self, manifest_idx: int):
        quota = self.protocol_quota
        state = quota.states[int(manifest_idx)]
        if len(quota.keys) == 3:
            return tuple(float(state[k]) for k in quota.keys)
        nut = (float(state["nut_x"]), float(state["nut_y"]), float(state["nut_yaw"]))
        return nut, (float(state["peg_x"]), float(state["peg_y"]))

    def _advance_to_protocol_eligible(self) -> None:
        """Move a highest-score quota-eligible start to the sampler's next index."""
        sampler = self.sampler
        quota = self.protocol_quota
        if sampler._next_idx >= len(sampler.planned_points):
            raise RuntimeError(
                "Adaptive protocol quota sampler reached the end of the planned list before "
                f"all quotas were filled. Remaining: {quota.remaining()}"
            )
        best_indices: list[int] = []
        best_score = -1
        for offset, candidate in enumerate(sampler.planned_points):
            manifest_idx, _ = quota.match(self._quota_vector(quota, candidate))
            if not quota.is_fresh_state_eligible(manifest_idx):
                continue
            score = quota.state_score(manifest_idx)
            if score > best_score:
                best_score = score
                best_indices = [offset]
            elif score == best_score:
                best_indices.append(offset)
        if not best_indices:
            raise RuntimeError(
                "Adaptive protocol quota sampler exhausted before all quotas were filled. "
                f"Remaining: {quota.remaining()}"
            )
        current = sampler._next_idx
        selected = int(self.protocol_rng.choice(best_indices))
        if selected != current:
            points = sampler.planned_points
            points[current], points[selected] = points[selected], points[current]
            print(
                "  Protocol quota: selected next quota-eligible start "
                f"from {len(sampler.planned_points)} hidden candidates"
            )

    def _protocol_tail_progress(self) -> tuple[int, int]:
        quota = self.protocol_quota
        planned = self.sampler.planned_points
        total = max(0, len(planned) - len(quota.fresh_consumed_manifest_idxs))
        eligible = sum(
            1
            for candidate in planned
            if quota.is_fresh_state_eligible(quota.match(self._quota_vector(quota, candidate))[0])
        )
        return total, eligible

    def _route(self, is_counterfactual: bool) -> None:
        if self.router is None:
            return
        # A counterfactual replays with the policy of the episode it replays.
        if is_counterfactual and self.current_source is not None:
            return
        self.current_source, self.active_actor = self.router.route(self.current_state)

    def _place_start(self, is_counterfactual: bool) -> None:
        sampler = self.sampler
        if self.protocol_quota is not None and not is_counterfactual:
            self._advance_to_protocol_eligible()
        if not is_counterfactual:
            self.current_state = sampler.sample_next()
        where = place_start_state(self.env, sampler, self.current_state)
        print(f"  {self.sampler_label}: {where}{' (reused)' if is_counterfactual else ''}")
        self._route(is_counterfactual)

        if self.protocol_quota is not None:
            self.current_manifest_idx, self.current_match_dist = self.protocol_quota.match(
                self._quota_vector(self.protocol_quota, self.current_state)
            )
            can_cf = self.protocol_quota.can_accept_counterfactual(self.current_manifest_idx)
            print(
                "  Protocol quota: current start matched; "
                f"CF {'available' if can_cf else 'not available'}"
            )

    def _sampler_exhausted(self) -> bool:
        sampler = self.sampler
        return (
            sampler is not None
            and self.protocol_quota is None
            and sampler._next_idx >= len(sampler.planned_points)
        )

    def _run_segments(self, start_with_human: bool) -> dict:
        """Alternate policy and human segments until the episode ends."""
        args = self.args
        segment_kwargs = dict(
            camera_names=self.camera_names,
            max_fr=args.max_fr,
            has_renderer=self.has_renderer,
            auto_save_on_success=args.auto_save_on_success,
        )
        episode = {
            "data": None,
            "sources": [],
            "success": False,
            "terminal": False,
            "interventions": 0,
            "policy_steps": [],
        }

        def add(segment: Optional[dict], source_id: int) -> None:
            if segment is not None and len(segment["actions"]) > 0:
                episode["data"], labels = accumulate_segment_data(
                    episode["data"], segment, source_id
                )
                episode["sources"].extend(labels)

        def discard() -> None:
            episode["data"] = None
            episode["sources"] = []

        gripper_action = None
        if start_with_human:
            human_data, outcome, success, gripper_action = human_correction_segment(
                self.env, self.operator, initial_gripper_action=None, **segment_kwargs
            )
            if outcome != "discard":
                add(human_data, 1)
            episode["success"] |= success
            if outcome == "discard":
                discard()
                return episode
            if outcome in ("failure", "terminal_failure"):
                episode["success"] = False
                episode["terminal"] = outcome == "terminal_failure"
                return episode
            if outcome == "success":
                return episode
            self.active_actor.reset()
            print("\n>>> Switching to POLICY control")

        while True:
            policy_data, outcome, success, policy_gripper = policy_rollout_segment(
                self.env,
                self.active_actor,
                self.operator,
                initial_gripper_action=gripper_action,
                **segment_kwargs,
            )
            if policy_data is not None and len(policy_data["actions"]) > 0:
                add(policy_data, 0)
                episode["policy_steps"].append(len(policy_data["actions"]))
            episode["success"] |= success

            if outcome == "discard":
                discard()
                return episode
            if outcome in ("failure", "terminal_failure"):
                episode["success"] = False
                episode["terminal"] = outcome == "terminal_failure"
                return episode
            if outcome == "complete":
                return episode

            episode["interventions"] += 1
            human_data, outcome, success, gripper_action = human_correction_segment(
                self.env, self.operator, initial_gripper_action=policy_gripper, **segment_kwargs
            )
            if outcome != "discard":
                add(human_data, 1)
            episode["success"] |= success
            if outcome == "continue":
                # Drop any action queue computed before the takeover.
                self.active_actor.reset()
                print("\n>>> Resuming POLICY control")
                continue
            if outcome == "discard":
                discard()
            elif outcome in ("failure", "terminal_failure"):
                episode["success"] = False
                episode["terminal"] = outcome == "terminal_failure"
            return episode

    def _ensure_dataset(self, episode_data: dict) -> None:
        if self.dataset is not None:
            return
        print(f"Creating new dataset at {self.dataset_path}")
        self.dataset = LeRobotDataset.create(
            repo_id=self.args.dataset_name,
            fps=20,
            root=str(self.dataset_path),
            robot_type=self.robot_name.lower(),
            features=dataset_features(self.env_name, episode_data, self.camera_names),
        )

    def _save(self, episode: dict, is_counterfactual: bool) -> None:
        args = self.args
        if self.protocol_quota is not None and self.save_futures:
            print("Waiting for the previous save before reserving the next quota row...")
            for future in self.save_futures:
                _wait_for_save(future)
            self.save_futures.clear()

        self._ensure_dataset(episode["data"])
        self.saved_episode_count += 1
        episode_index = self.base_episode_count + self.saved_episode_count - 1
        extra = {
            "dataset_name": args.dataset_name,
            "intervention_count": int(episode["interventions"]),
            "policy_steps_per_segment": [int(x) for x in episode["policy_steps"]],
        }
        quota_ledger = self.protocol_quota
        quota_row = None
        if quota_ledger is not None:
            quota_row = quota_ledger.credit_episode(
                manifest_idx=self.current_manifest_idx,
                episode_index=episode_index,
                success=bool(episode["success"]),
                is_counterfactual=bool(is_counterfactual),
                matched_distance=self.current_match_dist,
                extra=extra,
                write_ledger=False,
            )
        if self.protocol_quota is not None:
            units = sum(len(arms) for arms in quota_row["credited_protocol_arms"].values())
            print(f"  Protocol quota ledger: credited {units} protocol-arm unit(s)")

        print("Saving episode in background...")
        self.save_futures.append(
            self.save_executor.submit(
                save_episode_to_dataset_and_append_ledger,
                dataset=self.dataset,
                episode_data=episode["data"],
                task_name=self.task_name,
                camera_names=self.camera_names,
                sources=episode["sources"],
                success=episode["success"],
                quota_ledger=quota_ledger,
                quota_row=quota_row,
            )
        )
        _check_completed_saves(self.save_futures)

    def _next_choice(self) -> str:
        """Ask for the next start; returns 'n', 'c' (counterfactual granted) or 'q'."""
        if self.sampler is None or self.current_state is None:
            print("Press 'n' for the NEXT episode, or 'q' / Ctrl+C to quit...")
            return self.operator.choose("nq")

        allow_cf = True
        if self.protocol_quota is not None:
            allow_cf = self.protocol_quota.can_accept_counterfactual(self.current_manifest_idx)
            if self.protocol_quota.is_protocol_complete("with_cf"):
                print(
                    "With-CF quotas complete; press 'n' for the NEXT normal rollout "
                    "('c' is skipped), or 'q' / Ctrl+C to quit..."
                )
            else:
                print(
                    "Press 'n' for the NEXT eligible start, 'c' for a quota-safe "
                    "COUNTERFACTUAL (skipped if closed), or 'q' / Ctrl+C to quit..."
                )
            key = self.operator.choose("ncq")
            if key == "c" and not allow_cf:
                print("CF request skipped by the quota controller; advancing to the next start.")
                return "n"
            return key
        if allow_cf:
            print(
                "Press 'n' for the NEXT start, 'c' for a COUNTERFACTUAL (same start), "
                "or 'q' / Ctrl+C to quit..."
            )
            return self.operator.choose("ncq")
        print("Press 'n' for the NEXT eligible start, or 'q' / Ctrl+C to quit...")
        return self.operator.choose("nq")

    def _resume_counterfactual(self) -> bool:
        """At startup, offer the pending with-CF replay of the last successful start."""
        quota = self.protocol_quota
        if quota is None:
            return False
        resume_idx = quota.resume_counterfactual_manifest_idx()
        if resume_idx is None:
            if quota.last_successful_manifest_idx is not None:
                print(
                    "\nThe previous successful start cannot take another With-CF replay; "
                    "starting from the next eligible fresh start."
                )
            return False
        print(
            "\nThe previous successful start can still take a With-CF replay. Press 'c' to "
            "collect it now, 'n' to skip to the next eligible fresh start, or 'q' to quit..."
        )
        key = self.operator.choose("cnq")
        if key == "q":
            raise KeyboardInterrupt
        if key == "n":
            return False
        self.current_state = self._state_from_manifest_idx(resume_idx)
        self.current_manifest_idx = int(resume_idx)
        self.current_match_dist = 0.0
        return True

    def _log_episode(self, episode: dict, episode_num: int, is_counterfactual: bool) -> None:
        steps = episode["policy_steps"]
        print(f"  Task success: {episode['success']}")
        print(f"  Interventions: {episode['interventions']}")
        if steps:
            print(f"  Policy steps per segment: {steps} (mean {np.mean(steps):.1f})")
        print(
            f"  Dataset total: ~{episode_num} episodes "
            f"({self.saved_episode_count} saved this session)\n"
        )
        if self.wandb_run is None:
            return
        metrics = {
            "episode": episode_num,
            "performance/interventions": episode["interventions"],
            "performance/task_success": int(episode["success"]),
            "performance/total_segments": len(steps),
            "performance/is_counterfactual": int(is_counterfactual),
        }
        if steps:
            metrics["performance/avg_policy_steps"] = float(np.mean(steps))
            metrics["performance/total_policy_steps"] = sum(steps)
            metrics["performance/max_policy_steps"] = max(steps)
            metrics["performance/min_policy_steps"] = min(steps)
        self.wandb_run.log(metrics, step=episode_num)

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        print("\nSetup complete. Starting DAgger collection.")
        print("=" * 70)
        try:
            reuse_state = self._resume_counterfactual()
            while True:
                if self.protocol_quota is not None and self.protocol_quota.is_complete():
                    print("\nProtocol-quota targets reached for all protocols.")
                    break
                is_counterfactual = reuse_state
                reuse_state = False
                if not is_counterfactual and self._sampler_exhausted():
                    print(f"\n{self.sampler_label} sampler exhausted; ending collection.")
                    break

                episode_num = self.base_episode_count + self.saved_episode_count + 1
                label = "COUNTERFACTUAL" if is_counterfactual else "Episode"
                print(f"\n{'=' * 70}\n{label} {episode_num}\n{'=' * 70}")

                if self.env is None:
                    self.env = create_robosuite_env(
                        self.env_name,
                        robot_name=self.robot_name,
                        camera_names=self.camera_names or None,
                        render_camera=self.camera_names[0] if self.camera_names else "agentview",
                        has_renderer=self.has_renderer,
                        # The frame-render wrapper is only used with the viewer.
                        use_render_wrapper=self.has_renderer,
                    )
                    if self.has_renderer:
                        configure_viewer_shadows(self.env)
                self.env.reset()
                if self.sampler is not None:
                    self._place_start(is_counterfactual)
                if self.has_renderer:
                    self.env.render()

                self.operator.begin_episode(
                    EpisodeStart(
                        episode_index=episode_num - 1,
                        start_state=self.current_state if self.sampler is not None else None,
                        manifest_idx=self.current_manifest_idx,
                        is_counterfactual=is_counterfactual,
                        human_first=is_counterfactual,
                    )
                )
                self.active_actor.reset()
                episode = self._run_segments(is_counterfactual)

                has_data = _finalize_episode_data(
                    self.env,
                    episode["data"],
                    episode["sources"],
                    episode["success"],
                    episode["terminal"],
                    self.camera_names,
                )
                # Counterfactuals reuse a recorded start; the protocol quota
                # consumes a fresh start only on success.
                if (
                    self.sampler is not None
                    and has_data
                    and not is_counterfactual
                    and (self.protocol_quota is None or episode["success"])
                ):
                    record_sampler_start(self.sampler, episode["data"]["environment_state"][0])
                    print(f"  Sampler: recorded start ({len(self.sampler.collected_points)} total)")
                if has_data:
                    self._save(episode, is_counterfactual)

                episode_num = self.base_episode_count + self.saved_episode_count
                print(f"\n{label} {episode_num} summary:")
                self._log_episode(episode, episode_num, is_counterfactual)

                if self.protocol_quota is not None:
                    total_remaining, eligible_remaining = self._protocol_tail_progress()
                    for line in self.protocol_quota.progress_lines(
                        saved_episode_count=self.saved_episode_count,
                        total_manifest_remaining=total_remaining,
                        eligible_manifest_remaining=eligible_remaining,
                    ):
                        print(line)
                    if self.protocol_quota.is_complete():
                        print("Protocol-quota targets reached; ending collection.")
                        break

                choice = self._next_choice()
                if choice == "q":
                    break
                reuse_state = choice == "c"
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        print("\nCleaning up...")
        save_errors = []
        if self.save_futures:
            print(f"Waiting for {len(self.save_futures)} background save(s)...")
            for future in self.save_futures:
                try:
                    _wait_for_save(future)
                except RuntimeError as err:
                    save_errors.append(err)
                    print(f"ERROR: {err}")
        self.save_executor.shutdown(wait=True)

        if self.dataset is not None:
            print(f"Finalizing dataset ({self.dataset.num_episodes} episodes)...")
            self.dataset.finalize()
            if self.args.push_to_hub and not save_errors:
                self._push_to_hub()
        self.operator.close()
        if self.env is not None:
            self.env.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if save_errors:
            raise RuntimeError(
                f"{len(save_errors)} episode save(s) failed; the dataset was not pushed"
            ) from save_errors[0]

    def _push_to_hub(self) -> None:
        repo_id = self.hub_repo_id
        print(f"\nPushing dataset to https://huggingface.co/datasets/{repo_id} ...")
        self.dataset.repo_id = repo_id
        self.dataset.push_to_hub(private=False, license=self.args.license)
        if self.quota_ledger_path and Path(self.quota_ledger_path).exists():
            bundle_quota_ledger_to_hub(repo_id, Path(self.quota_ledger_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def make_operator(args: argparse.Namespace) -> Operator:
    """The operator named by ``--operator``."""
    if args.operator == "spacemouse":
        return SpaceMouseKeyboardOperator(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
    module_name, sep, attr = args.operator.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"--operator must be 'spacemouse' or 'module.path:factory', got {args.operator!r}"
        )
    kwargs = json.loads(args.operator_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--operator-kwargs must be a JSON object")
    operator = getattr(importlib.import_module(module_name), attr)(**kwargs)
    if not isinstance(operator, Operator):
        raise TypeError(f"{args.operator} returned {type(operator).__name__}, not an Operator")
    return operator


def _check_operator(operator: Operator, args: argparse.Namespace) -> None:
    """Refuse flag combinations a scripted operator cannot run under."""
    if getattr(operator, "requires_auto_save_on_success", False) and not (
        args.auto_save_on_success
    ):
        raise ValueError(
            f"{type(operator).__name__} needs --auto-save-on-success: it does not press "
            "'1' for a recorded success and relies on the collector to end the episode "
            "when the env reports success"
        )


def _check_mjpython(args: argparse.Namespace) -> None:
    if platform.system() != "Darwin" or args.headless:
        return
    import mujoco.viewer

    if not hasattr(mujoco.viewer, "_MJPYTHON"):
        print("ERROR: the MuJoCo viewer on macOS needs mjpython:")
        print(f"  mjpython -m mulligan.sim.collect.dagger {' '.join(sys.argv[1:])}")
        sys.exit(1)


def main(argv: Optional[list[str]] = None, operator: Optional[Operator] = None) -> None:
    """Run a collection session; ``operator`` overrides ``--operator``."""
    args = parse_args(argv)
    routed = _parse_routed_policies(args.routed_policy)
    if routed and args.policy:
        raise ValueError("--policy and --routed-policy are mutually exclusive")
    if not routed and not args.policy:
        raise ValueError("Pass --policy, or --routed-policy LABEL=REF (at least two)")
    if routed and not args.policy_routing_manifest:
        raise ValueError("--routed-policy requires --policy-routing-manifest")
    if args.policy_routing_manifest and not routed:
        raise ValueError("--policy-routing-manifest requires --routed-policy")
    _check_mjpython(args)
    if operator is None:
        operator = make_operator(args)
    _check_operator(operator, args)

    load_kwargs = dict(
        device=args.device,
        num_action_samples=args.num_action_samples,
    )
    router = None
    if routed:
        actors = {}
        env_names = {}
        for label, ref in routed.items():
            actors[label], metadata = load_idql_actor(ref, **load_kwargs)
            env_names[label] = metadata["env_name"]
            robot_name = args.robot or metadata["robot_name"]
        if len(set(env_names.values())) != 1:
            raise ValueError(f"Routed policies were trained on different envs: {env_names}")
        env_name = args.env or next(iter(env_names.values()))
        router = PolicyRouter(Path(args.policy_routing_manifest), actors)
        actor = next(iter(actors.values()))
    else:
        actor, metadata = load_idql_actor(args.policy, **load_kwargs)
        env_name = args.env or metadata["env_name"]
        robot_name = args.robot or metadata["robot_name"]
    print(f"Env: {env_name}, robot: {robot_name}")

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            job_type="dagger-collection",
            config={**vars(args), "env_name": env_name, "robot_name": robot_name},
        )

    collector = DaggerCollector(
        args,
        operator=operator,
        actor=actor,
        env_name=env_name,
        robot_name=robot_name,
        router=router,
        wandb_run=wandb_run,
    )
    collector.setup()
    collector.run()


if __name__ == "__main__":
    main()
