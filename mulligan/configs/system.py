"""System configuration for device and checkpoint settings."""

from dataclasses import dataclass, field

import torch


def _get_default_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class SystemConfig:
    """
    Configuration for system settings.

    Controls device selection and checkpoint directory.
    """

    # Device
    device: str = field(
        default_factory=_get_default_device,
        metadata={"help": "Device to use (cuda/mps/cpu)"},
    )

    # Checkpoints
    checkpoint_dir: str = field(
        default="./checkpoints",
        metadata={"help": "Base directory for run folders"},
    )

    def __post_init__(self):
        """Validate device availability."""
        valid_devices = ["cuda", "mps", "cpu"]
        # Handle cuda:N format
        base_device = self.device.split(":")[0]
        if base_device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices} (or cuda:N), got {self.device}"
            )

        # Fail before downloading data if the requested device is unusable.
        if base_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"system.device={self.device!r} requires CUDA, but torch.cuda.is_available() is false"
            )
