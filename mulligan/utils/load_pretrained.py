"""Load trained policies from a local directory, an ``hf://`` URI or a W&B artifact.

Two checkpoint formats exist:

* IDQL / DIVL agents (sim): ``policy.pt`` + ``metadata.json`` + ``stats.json``.
  ``policy.pt`` pickles its config classes (``mulligan.configs.policy``), so it is loaded
  with full unpickling: load trusted checkpoints only.
* LeRobot policies (real diffusion policies): ``config.json`` + ``model.safetensors`` +
  processor files.

Use :func:`load_policy` for any source (see :func:`mulligan.release.hub.resolve_checkpoint`)
and :func:`load_policy_from_checkpoint` for a local directory.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

import draccus
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from mulligan.agents.idql import IDQLPolicy
from mulligan.release.hub import resolve_checkpoint
from mulligan.training.normalization import Normalizer

logger = logging.getLogger(__name__)

# Real-policy I/O contract fields (see mulligan.utils.lerobot_patches.apply_policy_io_contract_patch).
# They are re-attached after parsing so a config that predates a field keeps its value.
_POLICY_CONTRACT_FIELDS = (
    "camera_crop_boxes",
    "dual_side_crop_boxes",
    "action_target",
    "cartesian_action_frame",
    "action_mode",
)

_IDQL_CONFIG_CLASSES = ("IDQLPolicyConfig", "IDQLDIVLConfig")


def _load_lerobot_config_with_compat(checkpoint_path: Path, config_data: dict) -> PreTrainedConfig:
    """Parse a LeRobot ``config.json``, tolerating known benign schema drift."""
    sanitized_config = dict(config_data)
    policy_type = sanitized_config.setdefault("type", "diffusion")
    contract = {key: config_data[key] for key in _POLICY_CONTRACT_FIELDS if key in config_data}

    def _reattach_contract(config: PreTrainedConfig) -> PreTrainedConfig:
        for key, value in contract.items():
            setattr(config, key, value)
        return config

    # Diffusion checkpoints may carry use_peft, which this lerobot config does not use.
    # Refuse if PEFT was actually enabled; otherwise drop it.
    if policy_type == "diffusion" and "use_peft" in sanitized_config:
        if sanitized_config.pop("use_peft"):
            raise ValueError(
                f"Checkpoint config has use_peft=true, but this checkout's "
                f"{policy_type} config does not support PEFT. Refusing to load it."
            )
        logger.info(f"Ignoring unused {policy_type} field use_peft=false")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        # draccus rejects fields the config dataclass does not define ("The fields `x`,
        # `y` are not valid for <Cfg>"); drop exactly those and retry.
        for _ in range(32):
            with open(config_path, "w") as f:
                json.dump(sanitized_config, f)
            try:
                with draccus.config_type("json"):
                    return _reattach_contract(draccus.parse(PreTrainedConfig, config_path, args=[]))
            except Exception as e:
                invalid = [k for k in re.findall(r"`([^`]+)`", str(e)) if k in sanitized_config]
                if not invalid:
                    raise ValueError(
                        f"Failed to parse LeRobot config from {checkpoint_path / 'config.json'} "
                        "after applying known compatibility fixes."
                    ) from e
                for k in invalid:
                    sanitized_config.pop(k, None)
                logger.info(f"Dropping config field(s) not in this checkout's schema: {invalid}")
        raise ValueError(
            f"Failed to parse LeRobot config from {checkpoint_path / 'config.json'}: "
            "too many schema-drift fields to strip."
        )


def _normalizer_from_stats(stats: dict, device: str) -> Normalizer:
    missing = [
        key
        for key in ("state_mean", "state_std", "action_min", "action_max", "state_min", "state_max")
        if key not in stats
    ]
    if missing:
        raise KeyError(f"checkpoint stats.json is missing {missing}")
    return Normalizer(
        state_mean=torch.tensor(stats["state_mean"], dtype=torch.float32),
        state_std=torch.tensor(stats["state_std"], dtype=torch.float32),
        action_min=torch.tensor(stats["action_min"], dtype=torch.float32),
        action_max=torch.tensor(stats["action_max"], dtype=torch.float32),
        state_min=torch.tensor(stats["state_min"], dtype=torch.float32),
        state_max=torch.tensor(stats["state_max"], dtype=torch.float32),
        device=device,
    )


def load_lerobot_policy(
    checkpoint_path: str | Path,
    config: PreTrainedConfig,
    *,
    device: str,
    strict: bool,
):
    """Build a LeRobot policy and its saved processor pipelines from a checkpoint directory.

    ``config`` is the caller's parsed and prepared ``config.json`` (schema-drift fixes, policy-contract
    sidecar fields, ``config.device``); the weights load onto ``config.device``. Returns
    ``(policy, preprocessor, postprocessor)`` with the policy in eval mode on ``device`` and
    both pipelines' ``device_processor`` set to ``device``.
    """
    policy = get_policy_class(config.type).from_pretrained(
        str(checkpoint_path), config=config, strict=strict
    )
    policy = policy.to(device)
    policy.eval()
    device_overrides = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides=device_overrides,
        postprocessor_overrides=device_overrides,
    )
    return policy, preprocessor, postprocessor


def load_policy_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    strict: bool = False,
):
    """Load a policy from a local checkpoint directory.

    Returns ``(policy, preprocessor, postprocessor)``. For IDQL/DIVL agents both
    processors are the :class:`~mulligan.training.normalization.Normalizer`; for LeRobot
    policies they are the saved LeRobot pipelines. ``strict`` applies to LeRobot weight loading only.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    logger.info(f"Loading policy from: {checkpoint_path}")

    is_agent_checkpoint = all(
        (checkpoint_path / name).exists() for name in ("metadata.json", "stats.json", "policy.pt")
    )
    if is_agent_checkpoint:
        metadata = json.loads((checkpoint_path / "metadata.json").read_text())
        stats = json.loads((checkpoint_path / "stats.json").read_text())
        policy_type = metadata["policy_type"]

        # metadata.json may name the policy family loosely; the pickled config class decides.
        checkpoint = torch.load(
            checkpoint_path / "policy.pt", map_location="cpu", weights_only=False
        )
        config = checkpoint.get("config")
        if type(config).__name__ not in _IDQL_CONFIG_CLASSES:
            raise ValueError(
                f"Unsupported agent checkpoint at {checkpoint_path}: policy_type={policy_type!r}, "
                f"config class {type(config).__name__}"
            )
        policy = IDQLPolicy.from_checkpoint(checkpoint, device=device)
        policy.eval()
        normalizer = _normalizer_from_stats(stats, device)
        policy.set_normalizer(normalizer)
        return policy, normalizer, normalizer

    if (checkpoint_path / "config.json").exists():
        config_data = json.loads((checkpoint_path / "config.json").read_text())
        policy_config = _load_lerobot_config_with_compat(checkpoint_path, config_data)
        # The policy loads onto the saved training device and is moved afterwards.
        return load_lerobot_policy(checkpoint_path, policy_config, device=device, strict=strict)

    raise ValueError(
        f"Could not determine policy type from checkpoint at {checkpoint_path}. "
        "Expected config.json (LeRobot) or metadata.json + stats.json + policy.pt (agent)."
    )


def load_policy(source: str | Path, device: str = "cpu", strict: bool = False):
    """Resolve ``source`` (local dir, ``hf://`` URI or W&B artifact) and load the policy."""
    return load_policy_from_checkpoint(resolve_checkpoint(source), device=device, strict=strict)


__all__ = ["load_lerobot_policy", "load_policy", "load_policy_from_checkpoint"]
