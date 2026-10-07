"""Tests for the mirrored DP image normalization in the Vision-IQL feature path.

Covers:
- ``_extract_image_normalization`` recovers the exact affine transform a DP
  preprocessor normalizer step applies (ImageNet stats, dataset stats, and the
  identity case), and rejects non-affine transforms and per-camera mismatches;
- ``VisionIQL.encode`` applies the mirrored normalization before the encoder
  (online and target paths) and stays byte-identical when disabled;
- ``encode_holdout_images`` applies the same transform on the holdout path.
"""

import copy

import pytest
import torch
import torch.nn as nn

from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.networks.vision_encoder import _extract_image_normalization
from mulligan.networks.vision_iql import VisionIQL

CAM_KEY = "observation.images.cam_left"
IMG_SHAPE = (3, 8, 8)
FEAT_DIM = 6
PROPRIO_DIM = 4
ACTION_DIM = 2
K = 2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)


class _FakeNormalizerStep:
    """Duck-typed stand-in for LeRobot's normalizer processor step."""

    def __init__(self, stats_by_key: dict):
        self._tensor_stats = stats_by_key

    def _apply_transform(self, tensor, key, feature_type, inverse=False):
        assert not inverse
        stats = self._tensor_stats.get(key)
        if stats is None:
            return tensor
        return (tensor - stats["mean"]) / stats["std"]


class _FakePipeline:
    def __init__(self, step):
        self.steps = [object(), step, object()]


class _ImageEncoder(nn.Module):
    """(B, 3, 8, 8) -> (B, FEAT_DIM); linear so normalization effects are exact."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3 * 8 * 8, FEAT_DIM)

    def forward(self, x):
        return self.lin(x.flatten(start_dim=1))


def _make_model(**kwargs):
    state_dim = FEAT_DIM + PROPRIO_DIM
    q1 = QNetwork(state_dim, ACTION_DIM * K, hidden_dims=[16], use_layer_norm=False)
    q2 = QNetwork(state_dim, ACTION_DIM * K, hidden_dims=[16], use_layer_norm=False)
    v_net = VNetwork(state_dim, hidden_dims=[16])
    return VisionIQL(
        encoder=_ImageEncoder(),
        q1=q1,
        q2=q2,
        v_net=v_net,
        camera_keys=[CAM_KEY],
        separate_encoders=False,
        expectile=0.7,
        gamma=0.99,
        tau=0.005,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extraction_recovers_imagenet_stats():
    step = _FakeNormalizerStep({CAM_KEY: {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}})
    mean, std = _extract_image_normalization(_FakePipeline(step), [CAM_KEY])
    assert torch.allclose(mean, IMAGENET_MEAN, atol=1e-5)
    assert torch.allclose(std, IMAGENET_STD, atol=1e-5)


def test_extraction_identity_when_key_has_no_stats():
    step = _FakeNormalizerStep({})
    mean, std = _extract_image_normalization(_FakePipeline(step), [CAM_KEY])
    assert torch.allclose(mean, torch.zeros(3, 1, 1), atol=1e-6)
    assert torch.allclose(std, torch.ones(3, 1, 1), atol=1e-6)


def test_extraction_rejects_per_camera_mismatch():
    other = "observation.images.cam_right"
    step = _FakeNormalizerStep(
        {
            CAM_KEY: {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            other: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        }
    )
    with pytest.raises(ValueError, match="differ across image keys"):
        _extract_image_normalization(_FakePipeline(step), [CAM_KEY, other])


def test_extraction_rejects_non_affine_transform():
    class _NonAffineStep(_FakeNormalizerStep):
        def _apply_transform(self, tensor, key, feature_type, inverse=False):
            return tensor**2

    with pytest.raises(ValueError, match="not a per-channel affine"):
        _extract_image_normalization(_FakePipeline(_NonAffineStep({})), [CAM_KEY])


def test_extraction_requires_a_normalizer_step():
    class _Bare:
        pass

    with pytest.raises(ValueError, match="Could not locate the normalizer step"):
        _extract_image_normalization(_Bare(), [CAM_KEY])


# ---------------------------------------------------------------------------
# VisionIQL.encode
# ---------------------------------------------------------------------------


def test_encode_applies_mirrored_normalization_online_and_target():
    torch.manual_seed(0)
    model = _make_model(image_norm_mean=IMAGENET_MEAN, image_norm_std=IMAGENET_STD)
    images = {CAM_KEY: torch.rand(5, *IMG_SHAPE)}
    normed = (images[CAM_KEY] - IMAGENET_MEAN.reshape(1, 3, 1, 1)) / IMAGENET_STD.reshape(
        1, 3, 1, 1
    )
    with torch.no_grad():
        got_online = model.encode(images)
        want_online = model.encoder(normed)
        got_target = model.encode(images, model.encoder_target)
        want_target = model.encoder_target(normed)
    assert torch.allclose(got_online, want_online, atol=1e-6)
    assert torch.allclose(got_target, want_target, atol=1e-6)


def test_encode_identity_when_disabled_matches_raw():
    torch.manual_seed(0)
    model = _make_model()
    assert not model.image_normalization_enabled
    images = {CAM_KEY: torch.rand(5, *IMG_SHAPE)}
    with torch.no_grad():
        got = model.encode(images)
        want = model.encoder(images[CAM_KEY])
    assert torch.allclose(got, want, atol=1e-6)


def test_norm_buffers_serialize_and_reject_bad_stats():
    model = _make_model(image_norm_mean=IMAGENET_MEAN, image_norm_std=IMAGENET_STD)
    sd = model.state_dict()
    assert "image_norm_mean" in sd and "image_norm_std" in sd
    with pytest.raises(ValueError, match="must be provided together"):
        _make_model(image_norm_mean=IMAGENET_MEAN)
    with pytest.raises(ValueError, match="std > 0"):
        _make_model(image_norm_mean=IMAGENET_MEAN, image_norm_std=torch.zeros(3, 1, 1))


def test_target_states_route_targets_without_touching_active_qv():
    torch.manual_seed(0)
    model = _make_model(image_norm_mean=IMAGENET_MEAN, image_norm_std=IMAGENET_STD)
    B = 6
    state_dim = FEAT_DIM + PROPRIO_DIM
    curr_state = torch.randn(B, state_dim)
    next_state = torch.randn(B, state_dim)
    target_curr = torch.randn(B, state_dim)
    target_next = torch.randn(B, state_dim)
    actions = torch.randn(B, ACTION_DIM * K)
    rewards = torch.rand(B, K)
    dones = torch.zeros(B, K)
    discount_powers = torch.ones(B, K)

    def run(**kw):
        with torch.no_grad():
            return model.forward_encoded_states(
                curr_state, next_state, actions, rewards, dones, discount_powers, **kw
            )

    matched = run()
    routed = run(target_curr_state=target_curr, bootstrap_next_state=target_next)

    # Active Q reads curr_state in both -> q means unchanged by the target-state swap.
    assert torch.allclose(matched["q1_mean"], routed["q1_mean"])
    assert torch.allclose(matched["q2_mean"], routed["q2_mean"])
    # The bootstrap/target-Q now consume the target states -> TD target must move.
    assert not torch.allclose(matched["td_target_mean"], routed["td_target_mean"])

    # Identity: passing next_state explicitly reproduces the default path.
    explicit = run(bootstrap_next_state=next_state)
    for key in matched:
        assert torch.allclose(matched[key], explicit[key], equal_nan=True), key


def test_independent_target_views_average_qv_outputs_not_features():
    torch.manual_seed(11)
    model = _make_model()
    batch = 5
    target_views = 4
    state_dim = FEAT_DIM + PROPRIO_DIM
    curr_state = torch.randn(batch, state_dim)
    next_state = torch.randn(batch, state_dim)
    target_curr_states = torch.randn(batch, target_views, state_dim)
    target_next_states = torch.randn(batch, target_views, state_dim)
    actions = torch.randn(batch, ACTION_DIM * K)
    rewards = torch.rand(batch, K) * 0.1
    dones = torch.zeros(batch, K)
    discount_powers = torch.tensor([[model.gamma**i for i in range(K)]])

    def run(target_curr, target_next):
        with torch.no_grad():
            return model.forward_encoded_states(
                curr_state,
                next_state,
                actions,
                rewards,
                dones,
                discount_powers,
                target_curr_state=target_curr,
                bootstrap_next_state=target_next,
            )

    averaged = run(target_curr_states, target_next_states)
    per_view = [
        run(target_curr_states[:, view], target_next_states[:, view])
        for view in range(target_views)
    ]
    torch.testing.assert_close(
        averaged["advantage_mean"],
        torch.stack([item["advantage_mean"] for item in per_view]).mean(),
    )
    torch.testing.assert_close(
        averaged["td_target_mean"],
        torch.stack([item["td_target_mean"] for item in per_view]).mean(),
    )

    # Passing the mean feature through nonlinear Q/V is a different estimator.
    feature_mean = run(target_curr_states.mean(dim=1), target_next_states.mean(dim=1))
    assert not torch.allclose(averaged["advantage_mean"], feature_mean["advantage_mean"])
    assert not torch.allclose(averaged["td_target_mean"], feature_mean["td_target_mean"])


def test_multistep_valid_steps_truncation_masks_padding_and_discounts_bootstrap():
    torch.manual_seed(0)
    model = _make_model()
    # Hand-computed targets below assume no Bellman-label clipping.
    assert model.clip_targets_min is None and model.clip_targets_max is None
    state_dim = FEAT_DIM + PROPRIO_DIM
    gamma = model.gamma
    W = 12  # TD window wider than the 6-step action chunk
    curr_state = torch.randn(1, state_dim)
    next_state = torch.randn(1, state_dim)
    actions = torch.randn(1, ACTION_DIM * K)
    discount_powers = torch.tensor([[gamma**i for i in range(W)]])
    with torch.no_grad():
        v_next = model.v_net(next_state)

    # Timeout anchor truncated at 7 of 12 steps: cache canonicalization repeats
    # the last valid row into the padding, so poison the padded rewards to prove
    # they are masked out (unmasked padding would double-count that reward).
    rewards = torch.arange(1.0, W + 1.0).reshape(1, W) / 10
    rewards[0, 7:] = 999.0
    dones = torch.zeros(1, W)
    with torch.no_grad():
        out = model.forward_encoded_states(
            curr_state,
            next_state,
            actions,
            rewards,
            dones,
            discount_powers,
            valid_steps=torch.tensor([7]),
        )
    expected = sum(rewards[0, i].item() * gamma**i for i in range(7)) + gamma**7 * v_next.item()
    assert torch.allclose(out["td_target_mean"], torch.tensor(expected), rtol=1e-5)

    # Full-window valid_steps reproduces the fixed-horizon path.
    rewards_full = torch.arange(1.0, W + 1.0).reshape(1, W) / 10
    with torch.no_grad():
        fixed = model.forward_encoded_states(
            curr_state, next_state, actions, rewards_full, dones, discount_powers
        )
        full = model.forward_encoded_states(
            curr_state,
            next_state,
            actions,
            rewards_full,
            dones,
            discount_powers,
            valid_steps=torch.tensor([W]),
        )
    assert torch.allclose(fixed["td_target_mean"], full["td_target_mean"], rtol=1e-5)

    # Terminal inside the window: padding repeats the first terminal row
    # (done=1, reward repeated). No bootstrap; post-done rewards masked.
    dones_term = torch.zeros(1, W)
    dones_term[0, 3:] = 1.0
    rewards_term = torch.arange(1.0, W + 1.0).reshape(1, W) / 10
    rewards_term[0, 4:] = rewards_term[0, 3]
    with torch.no_grad():
        term = model.forward_encoded_states(
            curr_state,
            next_state,
            actions,
            rewards_term,
            dones_term,
            discount_powers,
            valid_steps=torch.tensor([4]),
        )
    expected_term = sum(rewards_term[0, i].item() * gamma**i for i in range(4))
    assert torch.allclose(term["td_target_mean"], torch.tensor(expected_term), rtol=1e-5)

    # Out-of-range valid_steps fails loud.
    for bad in (0, W + 1):
        with pytest.raises(ValueError, match="valid_steps"):
            model.forward_encoded_states(
                curr_state,
                next_state,
                actions,
                rewards_full,
                dones,
                discount_powers,
                valid_steps=torch.tensor([bad]),
            )


def test_forward_encoded_states_matches_image_forward():
    torch.manual_seed(0)
    model = _make_model(image_norm_mean=IMAGENET_MEAN, image_norm_std=IMAGENET_STD)
    curr_images = {CAM_KEY: torch.rand(5, *IMG_SHAPE)}
    next_images = {CAM_KEY: torch.rand(5, *IMG_SHAPE)}
    proprio_curr = torch.randn(5, PROPRIO_DIM)
    proprio_next = torch.randn(5, PROPRIO_DIM)
    actions = torch.randn(5, ACTION_DIM * K)
    rewards = torch.rand(5, K)
    dones = torch.zeros(5, K)
    discount_powers = torch.ones(5, K)

    image_out = model(
        curr_images,
        next_images,
        proprio_curr,
        proprio_next,
        actions,
        rewards,
        dones,
        discount_powers,
    )
    with torch.no_grad():
        curr_state = torch.cat([model.encode(curr_images), proprio_curr], dim=-1)
        target_curr_state = torch.cat(
            [model.encode(curr_images, model.encoder_target), proprio_curr], dim=-1
        )
        next_state = torch.cat([model.encode(next_images), proprio_next], dim=-1)
    encoded_out = model.forward_encoded_states(
        curr_state,
        next_state,
        actions,
        rewards,
        dones,
        discount_powers,
        target_curr_state=target_curr_state,
    )

    assert image_out.keys() == encoded_out.keys()
    for key in image_out:
        assert torch.allclose(
            image_out[key].detach(),
            encoded_out[key].detach(),
            atol=1e-6,
            equal_nan=True,
        ), key


def test_encoded_path_matches_frozen_image_qv_optimizer_step():
    torch.manual_seed(7)
    image_model = _make_model(image_norm_mean=IMAGENET_MEAN, image_norm_std=IMAGENET_STD)
    encoded_model = copy.deepcopy(image_model)
    for model in (image_model, encoded_model):
        for encoder in (model.encoder, model.encoder_target):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    curr_images = {CAM_KEY: torch.rand(5, *IMG_SHAPE)}
    next_images = {CAM_KEY: torch.rand(5, *IMG_SHAPE)}
    proprio_curr = torch.randn(5, PROPRIO_DIM)
    proprio_next = torch.randn(5, PROPRIO_DIM)
    actions = torch.randn(5, ACTION_DIM * K)
    rewards = torch.rand(5, K)
    dones = torch.zeros(5, K)
    discount_powers = torch.ones(5, K)

    image_parameters = [
        *image_model.q1.parameters(),
        *image_model.q2.parameters(),
        *image_model.v_net.parameters(),
    ]
    encoded_parameters = [
        *encoded_model.q1.parameters(),
        *encoded_model.q2.parameters(),
        *encoded_model.v_net.parameters(),
    ]
    image_optimizer = torch.optim.AdamW(image_parameters, lr=3e-4, weight_decay=1e-4)
    encoded_optimizer = torch.optim.AdamW(encoded_parameters, lr=3e-4, weight_decay=1e-4)

    image_loss = image_model(
        curr_images,
        next_images,
        proprio_curr,
        proprio_next,
        actions,
        rewards,
        dones,
        discount_powers,
    )["total"]
    image_optimizer.zero_grad()
    image_loss.backward()
    image_optimizer.step()

    with torch.no_grad():
        curr_state = torch.cat([encoded_model.encode(curr_images), proprio_curr], dim=-1)
        target_curr_state = torch.cat(
            [encoded_model.encode(curr_images, encoded_model.encoder_target), proprio_curr],
            dim=-1,
        )
        next_state = torch.cat([encoded_model.encode(next_images), proprio_next], dim=-1)
    encoded_loss = encoded_model.forward_encoded_states(
        curr_state,
        next_state,
        actions,
        rewards,
        dones,
        discount_powers,
        target_curr_state=target_curr_state,
    )["total"]
    encoded_optimizer.zero_grad()
    encoded_loss.backward()
    encoded_optimizer.step()

    torch.testing.assert_close(image_loss, encoded_loss, rtol=0, atol=1e-7)
    for image_parameter, encoded_parameter in zip(
        image_parameters,
        encoded_parameters,
        strict=True,
    ):
        torch.testing.assert_close(image_parameter, encoded_parameter, rtol=0, atol=1e-7)


def test_update_targets_can_skip_frozen_encoder():
    torch.manual_seed(0)
    model = _make_model()
    q1_target_before = [p.detach().clone() for p in model.q1_target.parameters()]
    enc_target_before = [p.detach().clone() for p in model.encoder_target.parameters()]

    with torch.no_grad():
        for p in model.q1.parameters():
            p.add_(0.25)
        for p in model.encoder.parameters():
            p.add_(0.25)

    model.update_targets(update_encoder=False)

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(q1_target_before, model.q1_target.parameters(), strict=True)
    )
    assert all(
        torch.allclose(before, after)
        for before, after in zip(enc_target_before, model.encoder_target.parameters(), strict=True)
    )


# ---------------------------------------------------------------------------
# Holdout encode path
# ---------------------------------------------------------------------------


def test_encode_holdout_images_applies_normalization():
    from torch.utils.data import DataLoader

    from mulligan.real.train.iql_eval import encode_holdout_images

    torch.manual_seed(0)
    encoder = _ImageEncoder()
    frames = torch.rand(6, 1, *IMG_SHAPE)  # (N, T, C, H, W): loader takes [:, 0]
    states = torch.rand(6, 1, PROPRIO_DIM)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, i):
            return {CAM_KEY: frames[i], "observation.state": states[i]}

    dl = DataLoader(_DS(), batch_size=3)
    got = encode_holdout_images(
        holdout_dl=dl,
        encoder=encoder,
        camera_keys=[CAM_KEY],
        separate_encoders=False,
        device="cpu",
        image_norm_mean=IMAGENET_MEAN,
        image_norm_std=IMAGENET_STD,
    )
    normed = (frames[:, 0] - IMAGENET_MEAN.reshape(1, 3, 1, 1)) / IMAGENET_STD.reshape(1, 3, 1, 1)
    with torch.no_grad():
        want_visual = encoder(normed)
    assert torch.allclose(got[:, :FEAT_DIM], want_visual, atol=1e-6)
    assert torch.allclose(got[:, FEAT_DIM:], states[:, 0], atol=1e-6)
