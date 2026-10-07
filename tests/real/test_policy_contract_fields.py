"""The config parse load_dp uses keeps the policy I/O contract fields of config.json."""

from __future__ import annotations

import json

from mulligan.real.policy.loader import resolve_policy_action_contract


def _write(d, name, obj):
    (d / name).write_text(json.dumps(obj))


def test_lerobot_config_parse_preserves_contract_fields(tmp_path):
    """The config parse load_dp uses keeps the policy I/O contract fields."""
    import mulligan.real.policy.lerobot_patches  # noqa: F401  (registers the contract fields)
    from lerobot.configs.policies import PreTrainedConfig

    config = {
        "type": "diffusion",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [7]},
            "observation.images.wrist_left": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.images.side_1": {"type": "VISUAL", "shape": [3, 224, 224]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "camera_crop_boxes": {
            "wrist_left": [180, 0, 639, 413],
            "side_1": [138, 0, 580, 447],
        },
        "dual_side_crop_boxes": {},
        "action_target": "cartesian_velocity",
        "cartesian_action_frame": "base",
    }
    _write(tmp_path, "config.json", config)

    parsed = PreTrainedConfig.from_pretrained(tmp_path)

    assert {k: list(v) for k, v in parsed.camera_crop_boxes.items()} == config["camera_crop_boxes"]
    assert parsed.dual_side_crop_boxes == {}
    assert resolve_policy_action_contract(parsed) == ("cartesian_velocity", "base")
