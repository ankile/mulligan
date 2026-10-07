import torch

from mulligan.data.critic_sampling import human_chunk_indices


def test_handoff_excludes_autonomous_actions_after_human_anchor():
    human = torch.tensor([True, True, False, True, True])
    padding = torch.tensor(
        [
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [False, True],
        ]
    )
    selected = human_chunk_indices(human, padding)
    assert selected.tolist() == [0, 3, 4]


def test_terminal_padding_keeps_success_supervision_and_excludes_empty_chunks():
    human = torch.tensor([True, True, False, True])
    padding = torch.tensor(
        [
            [False, False, False],
            [False, True, True],
            [True, True, True],
            [False, True, True],
        ]
    )
    assert human_chunk_indices(human, padding).tolist() == [1, 3]


def test_no_human_success_chunks_produces_empty_eligibility():
    human = torch.zeros(3, dtype=torch.bool)
    padding = torch.tensor([[False, False], [False, False], [False, True]])
    assert human_chunk_indices(human, padding).numel() == 0


def test_eligibility_with_real_chunk_preparation():
    from mulligan.training.fast_dataset_loader import prepare_chunked_data

    data = {
        "states": torch.zeros(7, 2),
        "next_states": torch.zeros(7, 2),
        "actions": torch.zeros(7, 1),
        "rewards": torch.zeros(7),
        "dones": torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        "episode_indices": torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        "dataset_indices": torch.zeros(7, dtype=torch.long),
    }
    chunks = prepare_chunked_data(data, chunk_size=2, gamma=0.99)
    human = torch.tensor([True, True, False, True, True, True, True])
    selected = human_chunk_indices(human, chunks["action_is_pad"])
    assert selected.tolist() == [0, 3, 4, 5, 6]
