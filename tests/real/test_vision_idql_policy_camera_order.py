from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn

from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy


class _Feature(nn.Module):
    feature_dim = 1

    def forward(self, x):
        return x.reshape(x.shape[0], -1)[:, :1]


class _IdentityNormalizer:
    _tensor_stats = {}

    def _apply_transform(self, x, key, feature_type, *, inverse):
        return x

    def to(self, *, device):
        return self


class _FakeDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = nn.ModuleDict(
            {
                "wrist_left": _Feature(),
                "side_1": _Feature(),
            }
        )
        self.noise_scheduler = SimpleNamespace()
        self.num_inference_steps = 1

    def conditional_sample(self, n, *, global_cond):
        return torch.zeros(n, 1, 1)


class _FakeDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion = _FakeDiffusion()
        self.config = SimpleNamespace(
            n_obs_steps=1,
            n_action_steps=1,
            horizon=1,
            action_feature=SimpleNamespace(shape=(1,)),
            robot_state_feature=SimpleNamespace(shape=(0,)),
            image_features=OrderedDict(
                [
                    ("observation.images.wrist_left", object()),
                    ("observation.images.side_1", object()),
                ]
            ),
            camera_crop_boxes={},
            dual_side_crop_boxes={},
            action_target="cartesian_velocity",
            cartesian_action_frame="base",
        )


class _Pipeline:
    def __init__(self):
        self.steps = [_IdentityNormalizer()]


class _RecordingQ(nn.Module):
    state_dim = 2
    action_dim = 1

    def __init__(self):
        super().__init__()
        self.last_state = None

    def forward(self, state, action):
        self.last_state = state.detach().clone()
        return torch.zeros(state.shape[0], 1, device=state.device)


class _V(nn.Module):
    state_dim = 2

    def forward(self, state):
        return torch.zeros(state.shape[0], 1, device=state.device)


def test_vision_idql_scores_q_state_in_iql_camera_order():
    q1 = _RecordingQ()
    q2 = _RecordingQ()
    policy = VisionIDQLRealWorldPolicy(
        dp=_FakeDP(),
        dp_preprocessor=_Pipeline(),
        dp_postprocessor=_Pipeline(),
        q1=q1,
        q2=q2,
        v_net=_V(),
        num_action_samples=1,
        camera_keys=[
            "observation.images.side_1",
            "observation.images.wrist_left",
        ],
        iql_camera_keys=[
            "observation.images.side_1",
            "observation.images.wrist_left",
        ],
        image_height=1,
        image_width=1,
        device="cpu",
        iql_encoder=nn.ModuleDict(
            {
                "wrist_left": _Feature(),
                "side_1": _Feature(),
            }
        ),
    )

    policy.select_action(
        {
            "observation.state": torch.empty(0),
            "observation.images.side_1": torch.tensor([[[10.0]]]),
            "observation.images.wrist_left": torch.tensor([[[20.0]]]),
        }
    )

    assert q1.last_state is not None
    assert torch.equal(q1.last_state[0, :2], torch.tensor([10.0, 20.0]))
