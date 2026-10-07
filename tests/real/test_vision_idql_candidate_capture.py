import numpy as np
import pytest

from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy


def _skeleton_policy() -> VisionIDQLRealWorldPolicy:
    policy = object.__new__(VisionIDQLRealWorldPolicy)
    policy._capture_candidate_actions = False
    policy._last_candidate_action_chunks_raw = None
    policy._candidate_action_override_raw = None
    policy._capture_value_distribution = False
    policy._last_value_distribution = None
    policy.n_action_steps = 6
    policy.action_dim = 7
    return policy


def test_candidate_capture_is_opt_in_and_returns_defensive_copy():
    policy = _skeleton_policy()

    with pytest.raises(RuntimeError, match="capture is disabled"):
        policy.captured_candidate_action_chunks()

    policy.set_candidate_action_capture(True)
    with pytest.raises(RuntimeError, match="no candidate action chunk"):
        policy.captured_candidate_action_chunks()

    expected = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    policy._last_candidate_action_chunks_raw = expected.copy()
    captured = policy.captured_candidate_action_chunks()
    np.testing.assert_array_equal(captured, expected)

    captured[0, 0, 0] = -1
    assert policy._last_candidate_action_chunks_raw[0, 0, 0] == 0


def test_disabling_candidate_capture_clears_prior_capture():
    policy = _skeleton_policy()
    policy.set_candidate_action_capture(True)
    policy._last_candidate_action_chunks_raw = np.zeros((1, 1, 1), dtype=np.float32)

    policy.set_candidate_action_capture(False)

    assert policy._last_candidate_action_chunks_raw is None
    with pytest.raises(RuntimeError, match="capture is disabled"):
        policy.captured_candidate_action_chunks()


def test_candidate_action_override_validates_and_copies():
    policy = _skeleton_policy()
    chunks = np.zeros((4, 6, 7), dtype=np.float32)
    chunks[0, 0, 0] = 0.25

    policy.set_candidate_action_override(chunks)
    chunks[0, 0, 0] = 0.75

    assert policy._candidate_action_override_raw.shape == (4, 6, 7)
    assert policy._candidate_action_override_raw[0, 0, 0] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "chunks,match",
    [
        (np.zeros((0, 6, 7), dtype=np.float32), "at least one"),
        (np.zeros((4, 5, 7), dtype=np.float32), "must have shape"),
        (np.full((4, 6, 7), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_candidate_action_override_rejects_invalid_chunks(chunks, match):
    policy = _skeleton_policy()
    with pytest.raises(ValueError, match=match):
        policy.set_candidate_action_override(chunks)


def test_candidate_action_override_can_be_disabled():
    policy = _skeleton_policy()
    policy.set_candidate_action_override(np.zeros((4, 6, 7), dtype=np.float32))
    policy.set_candidate_action_override(None)
    assert policy._candidate_action_override_raw is None


def test_value_distribution_capture_is_opt_in_and_returns_defensive_copies():
    policy = _skeleton_policy()

    with pytest.raises(RuntimeError, match="capture is disabled"):
        policy.captured_value_distribution()

    policy.set_value_distribution_capture(True)
    with pytest.raises(RuntimeError, match="no categorical value distribution"):
        policy.captured_value_distribution()

    atoms = np.linspace(-0.05, 1.05, 5, dtype=np.float32)
    probs = np.asarray([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    policy._last_value_distribution = (atoms.copy(), probs.copy())
    captured_atoms, captured_probs = policy.captured_value_distribution()
    np.testing.assert_array_equal(captured_atoms, atoms)
    np.testing.assert_array_equal(captured_probs, probs)

    captured_atoms[0] = -9
    captured_probs[0] = -9
    assert policy._last_value_distribution[0][0] == atoms[0]
    assert policy._last_value_distribution[1][0] == probs[0]


def test_disabling_value_distribution_capture_clears_prior_capture():
    policy = _skeleton_policy()
    policy.set_value_distribution_capture(True)
    policy._last_value_distribution = (
        np.asarray([0.0, 1.0]),
        np.asarray([0.25, 0.75]),
    )

    policy.set_value_distribution_capture(False)

    assert policy._last_value_distribution is None
    with pytest.raises(RuntimeError, match="capture is disabled"):
        policy.captured_value_distribution()
