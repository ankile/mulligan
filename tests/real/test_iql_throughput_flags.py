"""Tests for the real-world trainer throughput flags.

Covers the semantics contract of the throughput
flags: `encoder_autocast_bf16` and `channels_last` must not change encode()
results on CPU (bf16 is CUDA-gated; channels_last is a layout-only change).
The CLI flags are BooleanOptionalAction and
default ON; the round-trip tests pin
the argparse defaults and the --flag / --no-flag twins for BOTH trainers. The
VisionIQL model kwargs still default off — the trainer passes the parsed args
explicitly.
"""

import pytest
import torch
import torch.nn as nn

from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.networks.vision_iql import VisionIQL

CAM_KEY = "observation.images.cam_left"
IMG_HW = 16
FEAT_DIM = 8
PROPRIO_DIM = 4
ACTION_DIM = 2
K = 2


class _ConvEncoder(nn.Module):
    """Tiny conv encoder over (B, 3, H, W) images -> (B, FEAT_DIM)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
        self.lin = nn.Linear(4 * (IMG_HW // 2) * (IMG_HW // 2), FEAT_DIM)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        return self.lin(h.flatten(1))


def _make_model(**kwargs) -> VisionIQL:
    state_dim = FEAT_DIM + PROPRIO_DIM
    chunked_action_dim = ACTION_DIM * K
    return VisionIQL(
        encoder=_ConvEncoder(),
        q1=QNetwork(state_dim, chunked_action_dim, hidden_dims=[16], use_layer_norm=False),
        q2=QNetwork(state_dim, chunked_action_dim, hidden_dims=[16], use_layer_norm=False),
        v_net=VNetwork(state_dim, hidden_dims=[16]),
        camera_keys=[CAM_KEY],
        separate_encoders=False,
        expectile=0.7,
        gamma=0.99,
        tau=0.005,
        **kwargs,
    )


def _images(batch_size=6):
    return {CAM_KEY: torch.rand(batch_size, 3, IMG_HW, IMG_HW)}


def test_throughput_flags_default_off():
    # Model-level kwargs stay default-off; the CLI defaults (below) drive them.
    model = _make_model()
    assert model.encoder_autocast_bf16 is False
    assert model.channels_last is False


# CLI dest -> (flag, default) for every converted IQL throughput flag.
IQL_FLAG_DEFAULTS = {
    "async_buffer_refresh": ("--async-buffer-refresh", True),
    "enable_tf32": ("--enable-tf32", True),
    "cudnn_benchmark": ("--cudnn-benchmark", True),
    "encoder_autocast_bf16": ("--encoder-autocast-bf16", True),
    "channels_last": ("--channels-last", True),
    "uint8_native_images": ("--uint8-native-images", True),
}

_IQL_BASE_ARGV = [
    "mulligan.real.train.critic",
    "--repo-ids",
    "mulligan/example",
    "--encoder-artifact",
    "entity/project/artifact:v0",
]


def _parse_iql(monkeypatch, extra):
    from mulligan.real.train.iql_args import parse_args

    monkeypatch.setattr("sys.argv", _IQL_BASE_ARGV + extra)
    return parse_args()


def test_iql_throughput_flag_defaults(monkeypatch):
    args = _parse_iql(monkeypatch, [])
    for dest, (_flag, default) in IQL_FLAG_DEFAULTS.items():
        assert getattr(args, dest) is default, dest


@pytest.mark.parametrize("dest", sorted(IQL_FLAG_DEFAULTS))
def test_iql_throughput_flag_round_trip(monkeypatch, dest):
    flag, _default = IQL_FLAG_DEFAULTS[dest]
    assert getattr(_parse_iql(monkeypatch, [flag]), dest) is True
    assert getattr(_parse_iql(monkeypatch, [f"--no-{flag[2:]}"]), dest) is False


_DP_BASE_ARGV = ["mulligan.real.train.policy", "--repo-ids", "mulligan/example"]


def _parse_dp(monkeypatch, extra):
    from mulligan.real.train.policy import parse_args as dp_parse_args

    monkeypatch.setattr("sys.argv", _DP_BASE_ARGV + extra)
    return dp_parse_args()


def test_dp_uint8_native_flag_round_trip(monkeypatch):
    assert _parse_dp(monkeypatch, []).uint8_native_images is True
    assert _parse_dp(monkeypatch, ["--uint8-native-images"]).uint8_native_images is True
    assert _parse_dp(monkeypatch, ["--no-uint8-native-images"]).uint8_native_images is False


# DP throughput flags: channels_last OPT-IN (small L40S-only win);
# decoded_frame_cache DEFAULT-ON (loud fallback when unmet).
DP_PRECISION_FLAG_DEFAULTS = {
    "channels_last": ("--channels-last", False),
    "decoded_frame_cache": ("--decoded-frame-cache", True),
}


@pytest.mark.parametrize("dest", sorted(DP_PRECISION_FLAG_DEFAULTS))
def test_dp_precision_flag_round_trip(monkeypatch, dest):
    flag, default = DP_PRECISION_FLAG_DEFAULTS[dest]
    assert getattr(_parse_dp(monkeypatch, []), dest) is default
    assert getattr(_parse_dp(monkeypatch, [flag]), dest) is True
    assert getattr(_parse_dp(monkeypatch, [f"--no-{flag[2:]}"]), dest) is False


def test_dp_resume_precision_regime_check(monkeypatch):
    from mulligan.real.train.policy import check_resume_precision_regime

    on = _parse_dp(monkeypatch, ["--channels-last"])
    off = _parse_dp(monkeypatch, [])

    # Matching regime resumes fine, in both directions (two-key dicts with
    # autocast_bf16=False included).
    check_resume_precision_regime({"precision_regime": {"channels_last": True}}, on)
    check_resume_precision_regime(
        {"precision_regime": {"autocast_bf16": False, "channels_last": False}}, off
    )
    # A checkpoint without the key is fp32/NCHW: OK with flags off, refused with
    # flags on.
    check_resume_precision_regime({}, off)
    with pytest.raises(ValueError, match="precision-regime mismatch"):
        check_resume_precision_regime({}, on)
    # Cross-regime flips refuse loudly.
    with pytest.raises(ValueError, match="precision-regime mismatch"):
        check_resume_precision_regime({"precision_regime": {"channels_last": True}}, off)
    # A checkpoint that trained under bf16 autocast cannot resume.
    with pytest.raises(ValueError, match="precision-regime mismatch"):
        check_resume_precision_regime(
            {"precision_regime": {"autocast_bf16": True, "channels_last": True}}, on
        )


def test_encode_identical_with_flags_off_vs_bf16_on_cpu():
    # bf16 autocast is CUDA-gated: on CPU the flag must be a strict no-op.
    torch.manual_seed(0)
    base = _make_model()
    flagged = _make_model(encoder_autocast_bf16=True)
    flagged.load_state_dict(base.state_dict())
    imgs = _images()
    torch.testing.assert_close(base.encode(imgs), flagged.encode(imgs))


def test_encode_channels_last_matches_default():
    torch.manual_seed(0)
    base = _make_model()
    flagged = _make_model(channels_last=True)
    flagged.load_state_dict(base.state_dict())
    flagged.encoder.to(memory_format=torch.channels_last)
    flagged.encoder_target.to(memory_format=torch.channels_last)
    imgs = _images()
    torch.testing.assert_close(base.encode(imgs), flagged.encode(imgs), rtol=1e-5, atol=1e-5)


def test_forward_losses_finite_with_flags_on_cpu():
    torch.manual_seed(0)
    model = _make_model(encoder_autocast_bf16=True, channels_last=True)
    model.encoder.to(memory_format=torch.channels_last)
    model.encoder_target.to(memory_format=torch.channels_last)
    bsz = 6
    out = model(
        curr_images=_images(bsz),
        next_images=_images(bsz),
        proprio_curr=torch.randn(bsz, PROPRIO_DIM),
        proprio_next=torch.randn(bsz, PROPRIO_DIM),
        actions=torch.randn(bsz, ACTION_DIM * K),
        rewards=torch.rand(bsz, K),
        dones=torch.zeros(bsz, K),
        discount_powers=torch.ones(bsz, K),
    )
    assert torch.isfinite(out["total"])
    out["total"].backward()


def test_encode_bf16_close_to_fp32_on_cuda():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")
    torch.manual_seed(0)
    base = _make_model().cuda()
    flagged = _make_model(encoder_autocast_bf16=True).cuda()
    flagged.load_state_dict(base.state_dict())
    imgs = {CAM_KEY: torch.rand(6, 3, IMG_HW, IMG_HW, device="cuda")}
    ref = base.encode(imgs)
    got = flagged.encode(imgs)
    assert got.dtype == torch.float32
    torch.testing.assert_close(ref, got, rtol=3e-2, atol=3e-2)


def test_encode_holdout_images_bf16_flag_noop_on_cpu():
    from torch.utils.data import DataLoader as TorchDataLoader

    from mulligan.real.train.iql_eval import encode_holdout_images

    torch.manual_seed(0)
    enc = _ConvEncoder()

    class _Items(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {
                CAM_KEY: torch.rand(1, 3, IMG_HW, IMG_HW, generator=g),
                "observation.state": torch.randn(1, PROPRIO_DIM, generator=g),
            }

    dl = TorchDataLoader(_Items(), batch_size=2)
    ref = encode_holdout_images(
        holdout_dl=dl,
        encoder=enc,
        camera_keys=[CAM_KEY],
        separate_encoders=False,
        device="cpu",
    )
    got = encode_holdout_images(
        holdout_dl=dl,
        encoder=enc,
        camera_keys=[CAM_KEY],
        separate_encoders=False,
        device="cpu",
        encoder_autocast_bf16=True,
        channels_last=True,
    )
    torch.testing.assert_close(ref, got, rtol=1e-5, atol=1e-5)


def test_uint8_refresh_dataset_bit_identical():
    from mulligan.data.vision_replay_buffer import VisionReplayBuffer
    from mulligan.real.train.critic import _Uint8RefreshDataset

    class _Base(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {
                CAM_KEY: torch.rand(2, 3, 8, 8, generator=g),
                "action": torch.randn(4, generator=g),
            }

    base = _Base()
    wrapped = _Uint8RefreshDataset(base, [CAM_KEY])
    for i in range(3):
        raw = base[i]
        packed = wrapped[i]
        assert packed[CAM_KEY].dtype == torch.uint8
        torch.testing.assert_close(
            packed[CAM_KEY], VisionReplayBuffer.pack_images_uint8(raw[CAM_KEY])
        )
        torch.testing.assert_close(packed["action"], raw["action"])
