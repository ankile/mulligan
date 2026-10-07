"""One raw-image Vision-IQL step Polyak-averages the target encoder and target critics.

``iql_train_step`` on the raw-image path (``use_embedding_cache=False``) calls
``update_targets(update_encoder=True)``, which lerps the ``encoder_target`` parameters
and float BatchNorm buffers and copies the integer ones; the cached-embedding path
leaves ``encoder_target`` untouched. Everything runs on CPU over a tiny fixture.
"""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn

CAMERAS = ("side_1", "wrist_left")
CAMERA_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
IMAGE_HW = (8, 8)
PROPRIO_DIM = 4
ACTION_DIM = 3
K = 2
FEATURE_DIM = 6
BATCH = 4


class _TinyBNEncoder(nn.Module):
    """Conv + BatchNorm + pool -> ``FEATURE_DIM``.

    BatchNorm is the point: ``update_targets(update_encoder=True)`` lerps float
    buffers and copies integer ones, so ``running_mean`` / ``running_var`` /
    ``num_batches_tracked`` are what a leaked Polyak actually corrupts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, FEATURE_DIM, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(FEATURE_DIM)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.bn(self.conv(x)))).flatten(1)


def _build_model():
    from mulligan.networks.q_network import QNetwork
    from mulligan.networks.v_network import VNetwork
    from mulligan.networks.vision_iql import VisionIQL

    torch.manual_seed(0)
    encoder = nn.ModuleDict({name: _TinyBNEncoder() for name in CAMERAS})
    state_dim = FEATURE_DIM * len(CAMERAS) + PROPRIO_DIM
    chunked_action_dim = ACTION_DIM * K

    def _q() -> QNetwork:
        return QNetwork(state_dim, chunked_action_dim, hidden_dims=[8, 8], use_layer_norm=True)

    return VisionIQL(
        encoder=encoder,
        q1=_q(),
        q2=_q(),
        v_net=VNetwork(state_dim, hidden_dims=[8, 8], use_layer_norm=True),
        camera_keys=list(CAMERA_KEYS),
        separate_encoders=True,
        expectile=0.7,
        gamma=0.99,
        # A big tau makes any leaked Polyak unmistakable rather than a rounding wobble.
        tau=0.5,
    )


def _step_inputs(model):
    from mulligan.real.train.critic import IQLStepTensors

    torch.manual_seed(1)
    batch = IQLStepTensors(
        proprio=torch.randn(BATCH, 2, PROPRIO_DIM),
        actions=torch.randn(BATCH, K, ACTION_DIM),
        rewards=torch.rand(BATCH, K),
        dones=torch.zeros(BATCH, K),
        cam_imgs_raw={cam_key: torch.rand(BATCH, 2, 3, *IMAGE_HW) for cam_key in CAMERA_KEYS},
    )
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    return batch, optimizer


def _context():
    from mulligan.real.train.critic import IQLStepContext, TrainingTimer

    args = Namespace(
        proprio_dropout=0.0,
        max_grad_norm=0.0,
    )
    return IQLStepContext(
        args=args,
        timer=TrainingTimer(),
        device=torch.device("cpu"),
        camera_keys=list(CAMERA_KEYS),
        use_embedding_cache=False,
        augment_fn=None,
        proprio_mean=torch.zeros(PROPRIO_DIM),
        proprio_std=torch.ones(PROPRIO_DIM),
        action_mean=torch.zeros(ACTION_DIM),
        action_std=torch.ones(ACTION_DIM),
        discount_powers=torch.tensor([0.99**i for i in range(K)]),
        independent_target_samples=0,
    )


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _identical(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    """Names whose bytes CHANGED (empty == bit-identical)."""
    assert set(before) == set(after)
    return sorted(name for name in before if not torch.equal(before[name], after[name]))


def _run_step():
    from mulligan.real.train.critic import iql_train_step

    model = _build_model()
    batch, optimizer = _step_inputs(model)
    encoder_before = _snapshot(model.encoder_target)
    q1_target_before = _snapshot(model.q1_target)
    losses = iql_train_step(
        model,
        batch,
        optimizer,
        _context(),
        step=0,
        td_lambda_active=False,
        sampling_td_horizon=K,
    )
    return (
        losses,
        _identical(encoder_before, _snapshot(model.encoder_target)),
        _identical(q1_target_before, _snapshot(model.q1_target)),
    )


def test_fixture_encoder_target_actually_has_batchnorm_buffers() -> None:
    """Guard the guard: a buffer-free fixture would make the check vacuous."""
    model = _build_model()
    buffer_names = sorted(name for name, _ in model.encoder_target.named_buffers())
    assert any(name.endswith("running_mean") for name in buffer_names), buffer_names
    assert any(name.endswith("num_batches_tracked") for name in buffer_names), buffer_names


def test_raw_path_step_polyaks_encoder_and_q_targets() -> None:
    losses, encoder_changed, q_target_changed = _run_step()
    assert torch.isfinite(losses["total"]).all()
    assert encoder_changed, "the raw path must Polyak encoder_target toward the encoder"
    assert q_target_changed, "the Q targets must Polyak toward the online critics"
