import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mulligan.utils.load_pretrained import _load_lerobot_config_with_compat


def _diffusion_config(use_peft: bool) -> dict:
    return {
        "type": "diffusion",
        "use_peft": use_peft,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [7]},
            "observation.images.left": {"type": "VISUAL", "shape": [3, 224, 224]},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [7]},
        },
        "crop_shape": None,
        "down_dims": [512, 1024],
        "horizon": 16,
    }


def test_load_lerobot_config_ignores_legacy_diffusion_use_peft_false(tmp_path):
    cfg = _load_lerobot_config_with_compat(tmp_path, _diffusion_config(use_peft=False))

    assert cfg.type == "diffusion"
    # use_peft is a real field on lerobot's base PreTrainedConfig (default False), so
    # the attribute always exists; the compat shim drops the legacy checkpoint value
    # and falls back to that safe default (it raises only on use_peft=true).
    assert cfg.use_peft is False
    assert cfg.image_features["observation.images.left"].shape == (3, 224, 224)


def test_load_lerobot_config_rejects_unsupported_diffusion_use_peft_true(tmp_path):
    with pytest.raises(ValueError, match="use_peft=true"):
        _load_lerobot_config_with_compat(Path(tmp_path), _diffusion_config(use_peft=True))


def test_relative_checkpoint_loads_in_a_fresh_process(tmp_path):
    # A fresh process: the processor step must resolve from its saved class path alone.
    script = textwrap.dedent("""
        import json, sys
        from pathlib import Path
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from mulligan.utils.load_pretrained import load_policy_from_checkpoint
        root = Path(sys.argv[1])
        cfg = DiffusionConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
            device="cpu", n_obs_steps=1, horizon=8, n_action_steps=4,
            down_dims=(8, 16), n_groups=8, diffusion_step_embed_dim=16,
            pretrained_backbone_weights=None)
        DiffusionPolicy(cfg).save_pretrained(root)
        device = {"registry_name": "device_processor", "config": {"device": "cpu"}}
        relative = {"class": "mulligan.real.policy.relative_pose.RelativePoseActionProcessorStep",
                    "config": {"n_obs_steps": 1, "action_dim": 10}}
        for name, steps in [("policy_preprocessor", [device, relative]),
                            ("policy_postprocessor", [device])]:
            (root / (name + ".json")).write_text(json.dumps({"name": name, "steps": steps}))
        loaded, pre, post = load_policy_from_checkpoint(root, device="cpu")
        assert next(loaded.parameters()).device.type == "cpu"
        assert type(pre.steps[-1]).__name__ == "RelativePoseActionProcessorStep"
        assert not loaded.training
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
