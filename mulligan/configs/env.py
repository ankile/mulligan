"""Environment configuration for Robosuite environments."""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """
    Configuration for the simulation environment.

    Specifies the Robosuite environment and robot to use for training and evaluation.
    """

    # Environment specification
    name: str = field(
        metadata={
            "help": "Sim task: square_narrow (NutAssemblySquare) or square_broad (Square_D1); "
            "the robosuite env IDs are accepted too"
        },
    )
    robot: str = field(
        metadata={"help": "Robot name(s). Single robot: 'Panda'. Bimanual: 'Panda,Panda'"},
    )

    # Visual aids toggle
    visual_aids: bool = field(
        default=False,
        metadata={"help": "Enable visual indicators in environment"},
    )

    def __post_init__(self):
        """Validate configuration."""
        if not self.name:
            raise ValueError("env.name cannot be empty")
        if not self.robot:
            raise ValueError("env.robot cannot be empty")
