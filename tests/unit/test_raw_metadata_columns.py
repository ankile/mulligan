"""Metadata reads must preserve row views without touching image transforms."""

from datasets import Dataset

from mulligan.data.transforms import (
    compute_multidataset_valid_boundaries,
    raw_metadata_column,
)


def test_raw_metadata_projection_skips_transform_and_preserves_selected_order():
    hf = Dataset.from_dict(
        {
            "episode_index": [0, 0, 1, 1, 1],
            "is_valid": [1, 0, 1, 1, 0],
            "done": [0, 0, 0, 1, 1],
            "unused_image_payload": [[1] * 100] * 5,
        }
    )
    hf = hf.select([2, 3, 4, 0, 1])

    def fail_transform(batch):
        raise AssertionError("numeric metadata must not invoke the image/torch transform")

    hf.set_transform(fail_transform)
    assert raw_metadata_column(hf, "episode_index") == [1, 1, 1, 0, 0]

    class Sub:
        hf_dataset = hf
        features = hf.features

        def __len__(self):
            return 5

    assert compute_multidataset_valid_boundaries([Sub()]) == ([0, 3], [2, 4], [3, 5], 2)
    # Reading metadata must not remove the original transform from the dataset.
    assert hf.format["type"] == "custom"


def test_raw_metadata_tensor_adapter_keeps_its_column_contract():
    import torch

    columns = {"episode_index": torch.tensor([0, 0, 1])}
    assert raw_metadata_column(columns, "episode_index") is columns["episode_index"]
