"""HiL-SERL run configuration: the RLPD agent and schedule (the defaults of
``mulligan.baselines.rlpd``) plus the HiL additions (``docs/baselines.md``). ``task`` picks the
env and demos (``tasks.py``). The HiL-SERL recipes of ``configs/sim/recipes.json`` hold the
flags of each run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from mulligan.baselines.hilserl.obs import HORIZON
from mulligan.baselines.hilserl.tasks import TaskSpec, get_task

DEFAULT_PORT = 5588
WANDB_PROJECT = "mulligan-hilserl"


@dataclass
class HilSerlConfig:
    task: str = "square_narrow"  # tasks.TASKS key: env + demos
    seed: int = 1
    # --- agent (SACLearner) ---
    hidden_dims: tuple = (256, 256, 256)
    num_qs: int = 10
    num_min_qs: int = 2
    critic_layer_norm: bool = True
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    init_temperature: float = 1.0
    backup_entropy: bool = False
    target_entropy: Optional[float] = None  # None -> -action_dim/2 = -3.5
    # --- schedule ---
    batch_size: int = 256
    utd_ratio: int = 20
    offline_ratio: float = 0.5
    start_training: int = 5000
    max_steps: int = 300_000
    log_interval: int = 1000
    eval_interval: int = 10_000
    eval_episodes: int = 50
    eval_seed_base: int = 10_000  # eval episode i resets with seed eval_seed_base + i
    horizon: int = HORIZON
    # --- HiL additions ---
    demo_gate: str = "success"  # success: intervened steps join the demo half only if the episode succeeds; none: always (HiL-SERL)
    truncation_bootstrap: bool = False  # mask 0 at the 400-step timeout, as in RLPD
    steps_per_publish: int = (
        50  # learner update calls between param broadcasts (hil-serl steps_per_update)
    )
    actor_max_lag: int = (
        400  # actor waits at an episode boundary while the learner owes more update calls than this
    )
    resume_interval_s: float = 300.0
    # --- transport / logging ---
    port: int = DEFAULT_PORT
    wandb_project: str = WANDB_PROJECT
    wandb_entity: Optional[str] = None  # None: your W&B default entity
    wandb_name: Optional[str] = None
    wandb_mode: str = "disabled"  # online | offline | disabled
    eval_workers: int = 8

    # keys whose values a resumed session must not change
    PINNED = (
        "task",
        "seed",
        "hidden_dims",
        "num_qs",
        "num_min_qs",
        "critic_layer_norm",
        "discount",
        "tau",
        "actor_lr",
        "critic_lr",
        "temp_lr",
        "init_temperature",
        "backup_entropy",
        "target_entropy",
        "batch_size",
        "utd_ratio",
        "offline_ratio",
        "start_training",
        "max_steps",
        "eval_interval",
        "eval_episodes",
        "eval_seed_base",
        "horizon",
        "demo_gate",
        "truncation_bootstrap",
    )

    def __post_init__(self):
        get_task(self.task)
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        if self.demo_gate not in ("success", "none"):
            raise ValueError(f"demo_gate must be 'success' or 'none', got {self.demo_gate!r}")
        if self.batch_size % 2 != 0:
            raise ValueError("batch_size must be even (50/50 demo/online split)")
        if self.eval_interval % 1 != 0 or self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(f"wandb_mode {self.wandb_mode!r}")

    @property
    def task_spec(self) -> TaskSpec:
        return get_task(self.task)

    def agent_kwargs(self) -> dict:
        return dict(
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            temp_lr=self.temp_lr,
            hidden_dims=self.hidden_dims,
            discount=self.discount,
            tau=self.tau,
            num_qs=self.num_qs,
            num_min_qs=self.num_min_qs,
            critic_layer_norm=self.critic_layer_norm,
            target_entropy=self.target_entropy,
            init_temperature=self.init_temperature,
            backup_entropy=self.backup_entropy,
        )

    @property
    def offline_batch(self) -> int:
        return int(self.batch_size * self.utd_ratio * self.offline_ratio)

    @property
    def online_batch(self) -> int:
        return int(self.batch_size * self.utd_ratio * (1 - self.offline_ratio))

    def pinned(self) -> dict:
        d = dataclasses.asdict(self)
        return {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items() if k in self.PINNED
        }

    def pinned_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.pinned(), sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d
