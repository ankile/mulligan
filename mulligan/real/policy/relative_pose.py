"""UMI-faithful relative end-effector-pose action representation.

UMI (Universal Manipulation Interface, Chi et al. 2024) re-expresses an absolute
end-effector pose *horizon* relative to the EE pose at the current step: timestep
``t+k`` carries the cumulative SE(3) displacement-from-now, so its magnitude grows
with ``k`` and the anchor element (``k`` = the current step) relativizes to
identity (zero translation, identity rotation). This module implements that
``convert_pose_mat_rep(pose_rep='relative')`` transform mapped onto our 10D action
layout::

    [rel_trans(3), rel_rot6d(6), gripper(1)]

The orientation uses the continuous 6D rotation of Zhou et al. 2019
(:mod:`mulligan.real.policy.rotation6d`) — seam-free, the SAME convention as the
absolute-pose remap the relative trainer runs first. The gripper passes through ABSOLUTE (UMI keeps
the gripper command non-relative).

SE(3) relativization (per chunk element ``k`` against the anchor base pose)::

    R_base = r6_to_matrix(base_r6);  t_base = base_trans
    R_k    = r6_to_matrix(abs_r6_k); t_k    = abs_trans_k
    rel_t_k  = R_base^T @ (t_k - t_base)        # base-local translation
    rel_R_k  = R_base^T @ R_k                   # base-local rotation
    rel_r6_k = matrix_to_r6(rel_R_k)

``absolutize_pose`` composes the inverse: ``t_k = R_base @ rel_t_k + t_base``,
``R_k = R_base @ rel_R_k``.

Two implementations share ONE convention, pinned against each other and against
scipy by the unit tests (``tests/real/test_relative_pose.py``):

- **numpy** (:func:`relativize_pose` / :func:`absolutize_pose`) — reuses the
  convention-pinned numpy/scipy helpers in :mod:`mulligan.real.policy.rotation6d`. Used by the
  per-timestep stats pre-pass, the offline action-recon decode, and the tests.
- **torch** (:func:`relativize_pose_torch`) — pure
  on-device tensor ops (no host sync) for the training-time
  :class:`RelativePoseActionProcessorStep`. The Gram-Schmidt algorithm and column
  layout MATCH the numpy path's ``r6_to_rotation_matrix`` / ``rotation_matrix_to_r6``,
  so train-time and offline/stats math agree to floating-point tolerance (the numpy
  path computes in float64, the torch ProcessorStep in the action tensor's float32, so
  they agree to ~1e-6, not bit-for-bit; pinned by the equivalence test).

The 10D layout: ``[0:3]`` translation, ``[3:9]`` r6, ``[9:10]`` gripper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep
from lerobot.types import EnvTransition, TransitionKey

from mulligan.real.policy.rotation6d import (
    R6_DIM,
    r6_to_rotation_matrix,
    rotation_matrix_to_r6,
)

# 10D relative-pose action layout indices.
POSE_DIM = 3 + R6_DIM + 1  # 10
_TRANS = slice(0, 3)
_R6 = slice(3, 9)
_GRIP = slice(9, 10)


def _check_pose_width(arr, name: str) -> None:
    if arr.shape[-1] != POSE_DIM:
        raise ValueError(
            f"{name} must have last dim {POSE_DIM} ([trans(3), r6(6), grip(1)]); got {arr.shape}"
        )


# ---------------------------------------------------------------------------
# numpy / scipy implementation (stats pre-pass, recon decode, tests)
# ---------------------------------------------------------------------------
def relativize_pose(abs_pose, base_pose) -> np.ndarray:
    """Express absolute 10D pose(s) relative to ``base_pose`` (numpy).

    Args:
        abs_pose: ``(..., 10)`` absolute pose ``[trans(3), r6(6), grip(1)]``.
        base_pose: ``(..., 10)`` anchor pose, broadcastable against ``abs_pose``
            over the leading dims (e.g. ``(B, 1, 10)`` against ``(B, T, 10)``).

    Returns:
        ``(..., 10)`` relative pose. The anchor element maps to identity
        (``rel_trans≈0``, ``rel_r6≈[1,0,0,0,1,0]``). Gripper passes through absolute.
    """
    abs_pose = np.asarray(abs_pose, dtype=np.float64)
    base_pose = np.asarray(base_pose, dtype=np.float64)
    _check_pose_width(abs_pose, "abs_pose")
    _check_pose_width(base_pose, "base_pose")

    R_base = r6_to_rotation_matrix(base_pose[..., _R6])  # (..., 3, 3)
    R_base_T = np.swapaxes(R_base, -1, -2)
    R_k = r6_to_rotation_matrix(abs_pose[..., _R6])  # (..., 3, 3)

    dt = abs_pose[..., _TRANS] - base_pose[..., _TRANS]  # (..., 3)
    rel_t = np.einsum("...ij,...j->...i", R_base_T, dt)
    rel_R = np.matmul(R_base_T, R_k)
    rel_r6 = rotation_matrix_to_r6(rel_R)
    grip = np.broadcast_to(abs_pose[..., _GRIP], rel_t.shape[:-1] + (1,))
    return np.concatenate([rel_t, rel_r6, grip], axis=-1).astype(np.float32)


def absolutize_pose(rel_pose, base_pose) -> np.ndarray:
    """Inverse of :func:`relativize_pose` (numpy). Compose ``rel_pose`` back to absolute."""
    rel_pose = np.asarray(rel_pose, dtype=np.float64)
    base_pose = np.asarray(base_pose, dtype=np.float64)
    _check_pose_width(rel_pose, "rel_pose")
    _check_pose_width(base_pose, "base_pose")

    R_base = r6_to_rotation_matrix(base_pose[..., _R6])  # (..., 3, 3)
    rel_R = r6_to_rotation_matrix(rel_pose[..., _R6])

    abs_t = np.einsum("...ij,...j->...i", R_base, rel_pose[..., _TRANS]) + base_pose[..., _TRANS]
    abs_R = np.matmul(R_base, rel_R)
    abs_r6 = rotation_matrix_to_r6(abs_R)
    grip = np.broadcast_to(rel_pose[..., _GRIP], abs_t.shape[:-1] + (1,))
    return np.concatenate([abs_t, abs_r6, grip], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# torch implementation (training-time ProcessorStep — on-device, no host sync)
# ---------------------------------------------------------------------------
def _r6_to_matrix_torch(r6: Tensor) -> Tensor:
    """``(..., 6) -> (..., 3, 3)`` Gram-Schmidt; columns ``[b0, b1, b0 x b1]``.

    Same algorithm as :func:`mulligan.real.policy.rotation6d.r6_to_rotation_matrix`
    (b0 = normalize(a0); b1 = normalize(a1 - <b0,a1> b0); b2 = b0 x b1). No epsilon
    guard: a degenerate r6 propagates NaN loudly rather than silently emitting a
    non-rotation. The training data is built from valid rotations, so this is hot.
    """
    a0 = r6[..., 0:3]
    a1 = r6[..., 3:6]
    b0 = a0 / torch.linalg.norm(a0, dim=-1, keepdim=True)
    dot = (b0 * a1).sum(dim=-1, keepdim=True)
    a1_orth = a1 - dot * b0
    b1 = a1_orth / torch.linalg.norm(a1_orth, dim=-1, keepdim=True)
    b2 = torch.linalg.cross(b0, b1, dim=-1)
    return torch.stack([b0, b1, b2], dim=-1)  # columns


def _matrix_to_r6_torch(m: Tensor) -> Tensor:
    """``(..., 3, 3) -> (..., 6)`` keep the first two columns (Zhou et al. layout)."""
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def _euler_to_matrix_torch(euler: Tensor) -> Tensor:
    """``(..., 3)`` extrinsic-xyz euler (roll, pitch, yaw; rad) ``-> (..., 3, 3)``.

    Matches :func:`mulligan.real.policy.rotation6d.euler_to_r6`'s scipy convention
    (``Rotation.from_euler("xyz", ...)``, lowercase = extrinsic): rotate about the
    FIXED x, then y, then z, i.e. ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``. Built from
    the three axis matrices + matmul (no hand-expanded product) so it is obviously
    the same rotation; pinned bitwise-to-tolerance against the numpy path by
    ``test_relative_pose``.
    """
    a = euler[..., 0]
    b = euler[..., 1]
    c = euler[..., 2]
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)

    def _mat3(r00, r01, r02, r10, r11, r12, r20, r21, r22):
        row0 = torch.stack([r00, r01, r02], dim=-1)
        row1 = torch.stack([r10, r11, r12], dim=-1)
        row2 = torch.stack([r20, r21, r22], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)

    rx = _mat3(one, zero, zero, zero, ca, -sa, zero, sa, ca)
    ry = _mat3(cb, zero, sb, zero, one, zero, -sb, zero, cb)
    rz = _mat3(cc, -sc, zero, sc, cc, zero, zero, zero, one)
    return torch.matmul(rz, torch.matmul(ry, rx))


def _euler_to_r6_torch(euler: Tensor) -> Tensor:
    """``(..., 3)`` extrinsic-xyz euler ``-> (..., 6)`` r6 (torch, on-device)."""
    return _matrix_to_r6_torch(_euler_to_matrix_torch(euler))


def proprio_euler_pose_to_pose10_torch(pose6: Tensor) -> Tensor:
    """``(..., 6)`` proprio EE pose ``[xyz, roll, pitch, yaw]`` ``-> (..., 10)`` pose.

    Builds the 10D ``[trans(3), r6(6), grip(1)]`` layout used as the UMI-relative
    anchor from a measured/commanded cartesian_position pose. The gripper slot is a
    zero placeholder — the anchor's gripper is never used (gripper is absolute
    pass-through, not relativized), so its value is irrelevant to the relativization.
    """
    if pose6.shape[-1] != 6:
        raise ValueError(f"proprio pose must be (..., 6) [xyz, rpy]; got {tuple(pose6.shape)}")
    trans = pose6[..., 0:3]
    r6 = _euler_to_r6_torch(pose6[..., 3:6])
    grip = torch.zeros_like(pose6[..., 0:1])
    return torch.cat([trans, r6, grip], dim=-1)


def relativize_pose_torch(abs_pose: Tensor, base_pose: Tensor) -> Tensor:
    """Express absolute 10D pose(s) relative to ``base_pose`` (torch, on-device).

    Args:
        abs_pose: ``(..., 10)`` absolute pose chunk (e.g. ``(B, T, 10)``).
        base_pose: ``(..., 10)`` anchor pose broadcastable over leading dims (e.g.
            ``(B, 1, 10)``).

    Returns:
        ``(..., 10)`` relative pose, same dtype/device as ``abs_pose``.
    """
    if abs_pose.shape[-1] != POSE_DIM or base_pose.shape[-1] != POSE_DIM:
        raise ValueError(
            f"relativize_pose_torch expects last dim {POSE_DIM}; got {tuple(abs_pose.shape)} / "
            f"{tuple(base_pose.shape)}"
        )
    R_base = _r6_to_matrix_torch(base_pose[..., 3:9])  # (..., 3, 3)
    R_base_T = R_base.transpose(-1, -2)
    R_k = _r6_to_matrix_torch(abs_pose[..., 3:9])

    dt = abs_pose[..., 0:3] - base_pose[..., 0:3]
    rel_t = torch.einsum("...ij,...j->...i", R_base_T, dt)
    rel_R = torch.matmul(R_base_T, R_k)
    rel_r6 = _matrix_to_r6_torch(rel_R)
    grip = abs_pose[..., 9:10].expand(rel_t.shape[:-1] + (1,))
    return torch.cat([rel_t, rel_r6, grip], dim=-1)


def command_pose_chunk_to_relative_torch(action_chunk: Tensor, anchor_state: Tensor) -> Tensor:
    """Convert a windowed absolute COMMAND-pose chunk to the physical UMI-relative rep.

    The IQL-critic-side twin of :class:`RelativePoseActionProcessorStep` (proprio-anchored,
    same anchor convention): the chunk is the commanded EE-pose trajectory sourced
    from ``action.cartesian_position`` + ``action.gripper_position`` and the anchor
    is the current measured PROPRIO pose, so ``rel[0]`` carries the command-vs-proprio
    controller lead. Unlike the DP ProcessorStep this operates on the raw euler
    columns directly (the critic data path never runs the 7→10 r6 remap), producing
    the SAME physical relative values the deployed relative DP emits after its
    per-timestep MIN_MAX un-normalization — so a critic trained on this output scores
    deploy-time candidate chunks in one representation.

    Args:
        action_chunk: ``(B, H, 7)`` absolute command poses per step
            ``[x, y, z (m), roll, pitch, yaw (rad), gripper (absolute 0/1)]``.
        anchor_state: ``(B, D>=6)`` current proprio ``observation.state`` rows whose
            first 6 dims are the euler cartesian_position anchor. A windowed
            ``(B, T_obs, D)`` state is NOT accepted — the caller selects the anchor
            step explicitly so the anchor convention is visible at the call site.

    Returns:
        ``(B, H, 10)`` physical relative chunk ``[rel_trans(3), rel_r6(6), grip(1)]``
        in ``action_chunk``'s dtype/device.
    """
    if action_chunk.ndim != 3 or action_chunk.shape[-1] != 7:
        raise ValueError(
            f"command_pose_chunk_to_relative_torch expects a (B, H, 7) [xyz, rpy, grip] "
            f"chunk; got {tuple(action_chunk.shape)}"
        )
    if anchor_state.ndim != 2 or anchor_state.shape[-1] < 6:
        raise ValueError(
            f"anchor_state must be (B, D>=6) with a euler cartesian_position prefix; "
            f"got {tuple(anchor_state.shape)}"
        )
    if anchor_state.shape[0] != action_chunk.shape[0]:
        raise ValueError(
            f"anchor/chunk batch mismatch: anchor B={anchor_state.shape[0]}, "
            f"chunk B={action_chunk.shape[0]}"
        )
    xyz = action_chunk[..., 0:3]
    r6 = _euler_to_r6_torch(action_chunk[..., 3:6])
    grip = action_chunk[..., 6:7]
    abs_pose10 = torch.cat([xyz, r6, grip], dim=-1)  # (B, H, 10)
    base = proprio_euler_pose_to_pose10_torch(anchor_state[:, :6].to(action_chunk.dtype))
    base = base.unsqueeze(1).to(device=action_chunk.device)  # (B, 1, 10)
    return relativize_pose_torch(abs_pose10, base)


def decode_normalized_relative_chunk(
    chunk_norm_rel, base_pose6, action_min, action_max
) -> np.ndarray:
    """Compose a NORMALIZED relative-pose chunk into ABSOLUTE 7D euler poses (numpy).

    Single source of truth for the deploy-side relative decode, shared by
    ``eval_common.LeRobotRealWorldPolicy.decode_relative_chunk_to_absolute`` (the
    plain relative-DP arm) and ``vision_idql_policy`` (the relative BoN arm): (1)
    per-timestep MIN_MAX un-normalize against the leading ``T'`` rows of the trained
    ``(T, 10)`` bounds, (2) absolutize against the GENERATION-TIME proprio anchor
    (proprio-anchored — held for the whole chunk), (3) decode r6 → euler.

    Args:
        chunk_norm_rel: ``(T', 10)`` normalized relative chunk (the executed window).
        base_pose6: ``(6,)`` anchor pose ``[x, y, z, roll, pitch, yaw]`` (euler rad).
        action_min / action_max: ``(T, 10)`` per-timestep MIN_MAX bounds, ``T >= T'``.

    Returns:
        ``(T', 7)`` absolute poses ``[x, y, z, roll, pitch, yaw, gripper]`` float32.
    """
    from mulligan.real.policy.rotation6d import euler_to_r6, r6_to_euler

    chunk = np.asarray(chunk_norm_rel, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != POSE_DIM:
        raise ValueError(
            f"decode_normalized_relative_chunk expects (T',{POSE_DIM}); got {chunk.shape}"
        )
    tprime = chunk.shape[0]
    a_min = np.asarray(action_min, dtype=np.float64)
    a_max = np.asarray(action_max, dtype=np.float64)
    if a_min.ndim != 2 or a_min.shape[1] != POSE_DIM:
        raise ValueError(
            f"relative decode needs (T,{POSE_DIM}) per-timestep bounds; got {a_min.shape}"
        )
    if a_min.shape[0] < tprime:
        raise ValueError(
            f"per-timestep bounds have {a_min.shape[0]} rows but chunk has {tprime} steps"
        )
    mn = a_min[:tprime]
    mx = a_max[:tprime]
    phys_rel = (chunk + 1.0) / 2.0 * (mx - mn) + mn  # (T',10) physical relative

    base6 = np.asarray(base_pose6, dtype=np.float64).reshape(-1)
    if base6.shape != (6,):
        raise ValueError(f"base_pose6 must be (6,); got {base6.shape}")
    base10 = np.concatenate(
        [base6[:3], np.asarray(euler_to_r6(base6[3:6]), dtype=np.float64).reshape(6), [0.0]]
    )
    abs10 = absolutize_pose(phys_rel, base10[None, :])  # (T',10) absolute [xyz,r6,grip]
    xyz = abs10[:, 0:3]
    euler = np.asarray(r6_to_euler(abs10[:, 3:9]), dtype=np.float64).reshape(tprime, 3)
    grip = abs10[:, 9:10]
    return np.concatenate([xyz, euler, grip], axis=1).astype(np.float32)


@dataclass
class RelativePoseActionProcessorStep(ProcessorStep):
    """Convert a windowed ABSOLUTE-pose action chunk to UMI-relative (training only).

    Inserted into the diffusion preprocessor pipeline immediately BEFORE the
    ``NormalizerProcessorStep`` (spliced in by the DP trainer when
    ``action_mode=="relative"``). At that point the canonical ``action`` column has
    already been remapped to the absolute COMMANDED EE-pose trajectory
    ``[cmd_t … cmd_{t+H-1}]`` (10D) and windowed by LeRobot into ``(B, T, 10)``, and
    ``observation.state`` still carries the raw ``[cartesian_position(6), gripper(1)]``
    proprio pose (this step runs before normalization).

    This step relativizes every element of the command chunk against the anchor =
    the current **proprio** pose (``observation.state`` cartesian_position at the last
    obs step), NOT the command ``chunk[:, 0]`` (proprio-anchored). Because the command leads
    the measured proprio by the soft-controller lag ``L``, this makes ``rel[0] = L``
    (the lead, not identity); composing the decode against live proprio at inference
    then reconstructs the command trajectory with no boundary retreat. The normalizer
    then sees this relative trajectory and the per-timestep ``(T, 10)`` stats fit it.

    It fires ONLY when an action chunk is present, so eval/deploy batches (no
    action) pass through untouched — the eval-side relative→absolute inversion is
    handled separately in ``eval_common`` (gated). Mirrors lerobot's
    ``RelativeActionsProcessorStep`` API: ``__call__`` / ``get_config`` /
    ``transform_features``.

    NOT registered in ``ProcessorStepRegistry``: it serializes by fully-qualified
    class path (``mulligan.real.policy.relative_pose.RelativePoseActionProcessorStep``), which
    the lerobot pipeline loader self-imports on ``from_pretrained`` — so a loaded
    checkpoint's preprocessor reconstructs the step without any global import-time
    registration.
    """

    n_obs_steps: int = 1
    action_dim: int = POSE_DIM

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        # Eval/deploy batches carry no action chunk → no-op (relative inversion is
        # done in eval_common). Also no-op for a bare (B, D) single action (deploy
        # pop): relativizing a single step against itself is meaningless and the
        # deploy path composes the full chunk separately.
        if action is None:
            return transition
        if not isinstance(action, torch.Tensor) or action.ndim != 3:
            return transition
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"RelativePoseActionProcessorStep expects a (B, T, {self.action_dim}) absolute-pose "
                f"action chunk; got {tuple(action.shape)}. Did the proprio-pose-r6 remap run?"
            )
        # Proprio anchor: the current PROPRIO pose (observation.state cartesian_position),
        # NOT the command action[:, 0]. The action horizon is the COMMANDED pose trajectory
        # (which leads proprio by the soft-controller lag); relativizing it against the
        # measured proprio makes rel[0] carry that lead (not identity), so re-grounding the
        # decode on live proprio at inference reconstructs the command trajectory with no
        # boundary retreat. Anchoring on the command frame (chunk[0]) would delete the
        # lead and produce a replan-locked sawtooth.
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or "observation.state" not in observation:
            raise ValueError(
                "action_mode=relative (proprio-anchored) relativizes the command horizon against the "
                "current proprio pose, but the transition has no 'observation.state'. The "
                "relative arm requires proprioception (do not combine with --drop-state)."
            )
        state = observation["observation.state"]
        if not isinstance(state, torch.Tensor):
            raise ValueError(
                f"observation.state must be a tensor for the relative anchor; got {type(state)}"
            )
        # observation.state = [cartesian_position(6 euler), gripper(1), ...]; LeRobot windows
        # it to (B, n_obs, D) — take the LAST obs step (the anchor/current step; n_obs_steps
        # is pinned to 1) — or (B, D) if unwindowed. cartesian_position is the first 6 dims.
        if state.ndim == 3:
            state = state[:, -1, :]
        if state.shape[-1] < 6:
            raise ValueError(
                f"observation.state must carry a 6D cartesian_position prefix; got width "
                f"{state.shape[-1]}"
            )
        base = proprio_euler_pose_to_pose10_torch(state[:, :6]).unsqueeze(1)  # (B, 1, 10)
        base = base.to(dtype=action.dtype, device=action.device)
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = relativize_pose_torch(action, base)
        return new_transition

    def get_config(self) -> dict:
        return {"n_obs_steps": self.n_obs_steps, "action_dim": self.action_dim}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None

    def reset(self) -> None:
        return None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
