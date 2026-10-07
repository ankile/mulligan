"""The Square tasks of the RLPD and HiL-SERL baselines, one ``TaskSpec`` each. A spec names
the robosuite 1.5.2 env, the object-state width, and the round-0 teleop demos (a
``mulligan/*-c00-teleop-baseline`` dataset at a pinned revision + a content sha256 of the
transitions, ``demos.content_sha256``).

- ``square_narrow`` (Square-Narrow): robosuite ``NutAssemblySquare``, 14-D object
  state, the 100 Square-Narrow demos.
- ``square_broad`` (Square-Broad): MimicGen ``Square_D1`` (nut x +-0.115, y +-0.255,
  full yaw; peg x (-0.1, 0.3), y (-0.2, 0.2)), 17-D object state (+ ``peg_pos``),
  the 200 Square-Broad demos.
"""

from __future__ import annotations

from dataclasses import dataclass

from mulligan.baselines.hilserl.obs import PROPRIO_DIM


@dataclass(frozen=True)
class TaskSpec:
    key: str
    env_name: str
    object_dim: int
    demo_repo_id: str
    demo_revision: str
    num_episodes: int
    num_frames: int
    # sha256 over the six transition arrays (demos.content_sha256)
    demos_sha256: str
    # default session / W&B name stem
    run_stem: str

    @property
    def obs_dim(self) -> int:
        return PROPRIO_DIM + self.object_dim

    @property
    def num_transitions(self) -> int:
        return self.num_frames - self.num_episodes


TASKS = {
    "square_narrow": TaskSpec(
        key="square_narrow",
        env_name="NutAssemblySquare",
        object_dim=14,
        demo_repo_id="mulligan/sim-square-narrow-c00-teleop-baseline",
        demo_revision="c92daf4da49fba126e876d00da07278f7c24e5d4",
        num_episodes=100,
        num_frames=16333,
        demos_sha256="14a5fb6ab166ad608b0e84bc07067f2c14b5d45bed42661d3f6d49667ddebcfc",
        run_stem="hilserl_square_narrow",
    ),
    "square_broad": TaskSpec(
        key="square_broad",
        env_name="Square_D1",
        object_dim=17,
        demo_repo_id="mulligan/sim-square-broad-c00-teleop-baseline",
        demo_revision="d65b70543ccc7d4e607c510e2b317fcc7b5fe1d0",
        num_episodes=200,
        num_frames=35686,
        demos_sha256="715550588ee4fd3c4018bbccc8e50f98dc14f72723ad927db7fab68a5fdec3d5",
        run_stem="hilserl_square_broad",
    ),
}


def get_task(key: str) -> TaskSpec:
    if key not in TASKS:
        raise ValueError(f"unknown task {key!r}; known: {sorted(TASKS)}")
    return TASKS[key]
