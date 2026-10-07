"""Resolve, download and load released real-robot checkpoints.

Model ids accepted by the real-robot loaders:

``hf://<org>/<name>[/<subfolder>][@<revision>]`` (or ``hf://<org>/<name>@<revision>/<subfolder>``)
    A Hugging Face model repo, parsed by :func:`mulligan.release.hub.parse_hf_uri`.
    ``<revision>`` is a commit hash, tag, branch or ``refs/pr/<N>``. Without it, a
    ``mulligan/*`` repo loads at its release pin (``released_checkpoints.json``,
    ``release/revisions.json``) and any other repo at its default branch. Downloads work
    anonymously.
``<path>``
    A local checkpoint directory (a LeRobot DP dir, or a critic dir with
    ``iql_checkpoint.pt`` + ``metadata.json``).
``wandb://<entity>/<project>/<name>:<version>``
    A W&B artifact of your own runs, downloaded with the optional ``wandb`` package.

An IDQL critic names the DP actor it was trained against in ``metadata.json``
(``dp_artifact``): the ``hf://`` id of a released DP, or for the one critic whose DP
version was not released, an ``unreleased/...`` id. :func:`critic_dp_model_id` resolves it
through ``released_checkpoints.json``.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

HF_SCHEME = "hf://"
WANDB_SCHEME = "wandb://"
# Release ids of checkpoints that were not published (see release/models.json `not_released`).
UNRELEASED_PREFIX = "unreleased/"
RELEASED_CHECKPOINTS_PATH = Path(__file__).with_name("released_checkpoints.json")

# Real-robot action-chunk protocol: execute 6 (n_action_steps) before
# re-planning. The prediction horizon is checkpoint/config-specific (for
# example, current marker_d2-era DP runs use horizon=12), so eval code must not
# hardcode it. Defined here, not in the torch-importing loader, so the
# entrypoints' argument parsers can use it before the robot stack loads.
REAL_PROTOCOL_N_ACTION_STEPS = 6


@dataclass(frozen=True)
class HFModelRef:
    repo: str
    subfolder: str | None = None
    revision: str | None = None

    @property
    def model_id(self) -> str:
        model_id = HF_SCHEME + self.repo
        if self.subfolder:
            model_id += f"/{self.subfolder}"
        if self.revision:
            model_id += f"@{self.revision}"
        return model_id


def is_hf_model_id(model_id: str) -> bool:
    return model_id.startswith(HF_SCHEME)


def is_wandb_model_id(model_id: str) -> bool:
    return model_id.startswith(WANDB_SCHEME)


def parse_hf_model_id(model_id: str) -> HFModelRef:
    """Parse an ``hf://`` id with :func:`mulligan.release.hub.parse_hf_uri` (either
    ``@<revision>`` position is accepted)."""
    from mulligan.release.hub import parse_hf_uri

    if not is_hf_model_id(model_id):
        raise ValueError(f"not an hf:// model id: {model_id!r}")
    ref = parse_hf_uri(model_id)
    return HFModelRef(repo=ref.repo_id, subfolder=ref.subfolder, revision=ref.revision)


@functools.cache
def released_checkpoints() -> tuple[dict, ...]:
    """The released real checkpoints (``released_checkpoints.json``), one dict per entry."""
    table = json.loads(RELEASED_CHECKPOINTS_PATH.read_text())
    return tuple(table["checkpoints"])


def released_checkpoint(repo: str, subfolder: str | None = None) -> dict | None:
    for entry in released_checkpoints():
        if entry["repo"] == repo and entry["subfolder"] == subfolder:
            return entry
    return None


def resolve_model_id(model_id: str) -> str:
    """Canonical model id: pin the revision of released ``hf://`` repos. Other ids are returned
    unchanged."""
    if is_hf_model_id(model_id):
        ref = parse_hf_model_id(model_id)
        if ref.revision is None:
            from mulligan.release.hub import default_revision

            entry = released_checkpoint(ref.repo, ref.subfolder)
            revision = entry["revision"] if entry is not None else default_revision(ref.repo)
            ref = HFModelRef(ref.repo, ref.subfolder, revision)
        return ref.model_id
    return model_id


def critic_dp_model_id(dp_artifact: str, *, dp_override: str | None = None) -> str:
    """Model id of the DP actor a critic re-ranks.

    ``dp_artifact`` is the critic's ``metadata.json`` value. ``dp_override`` replaces it
    (``--fixed-policy-dp-override``); it is the only way to load a critic whose trained DP
    was not released (``dp_resolution == "version-mismatch"`` in the table).
    """
    if dp_override is not None:
        return resolve_model_id(dp_override)
    if dp_artifact.startswith(UNRELEASED_PREFIX):
        entry = next(
            (e for e in released_checkpoints() if e.get("dp_checkpoint") == dp_artifact), None
        )
        if entry is None:
            raise ValueError(f"{dp_artifact} is not a released checkpoint; pass a DP override")
        raise ValueError(
            f"Critic {entry['repo']} was trained against a DP version that is not released "
            f"(dp_resolution={entry['dp_resolution']!r}). The closest released DP is "
            f"hf://{entry['dp_repo']}@{entry['dp_revision']}, a different version of the same "
            "run. Pass it (or another DP) explicitly as the DP override "
            "(--fixed-policy-dp-override NAME=<model id>) to load the critic with it."
        )
    return resolve_model_id(dp_artifact)


def _download_wandb_artifact(artifact: str) -> Path:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            f"{artifact} is not a released checkpoint; loading it needs the optional "
            "`wandb` package and W&B credentials."
        ) from exc
    return Path(wandb.Api().artifact(artifact.removeprefix(WANDB_SCHEME)).download())


def checkpoint_dir(
    model_id: str,
    *,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    """Local directory holding the checkpoint files for ``model_id`` (downloads if needed).

    ``cache_dir`` and ``token`` go to ``huggingface_hub.snapshot_download``
    (``token=False`` forces anonymous access).
    """
    model_id = resolve_model_id(model_id)
    if is_hf_model_id(model_id):
        from huggingface_hub import snapshot_download

        ref = parse_hf_model_id(model_id)
        root = snapshot_download(
            ref.repo,
            revision=ref.revision,
            allow_patterns=[f"{ref.subfolder}/*"] if ref.subfolder else None,
            cache_dir=cache_dir,
            token=token,
        )
        path = Path(root) / ref.subfolder if ref.subfolder else Path(root)
    elif is_wandb_model_id(model_id):
        path = _download_wandb_artifact(model_id)
    elif "://" in model_id:
        raise ValueError(f"unsupported model id scheme: {model_id!r}")
    else:
        path = Path(model_id).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found for {model_id!r}: {path}")
    return path


def load_dp(checkpoint_path: str | Path, *, device: str, strict: bool = True):
    """Load a LeRobot diffusion-policy checkpoint dir.

    Returns ``(policy, preprocessor, postprocessor)`` with the policy in eval mode on
    ``device``. The policy I/O contract (``camera_crop_boxes``, ``action_target``, ...) is
    read from ``config.json`` through the fields registered by the Mulligan lerobot patches.
    """
    import mulligan.real.policy.lerobot_patches  # noqa: F401  (policy I/O contract + DP patches)
    from lerobot.configs.policies import PreTrainedConfig

    from mulligan.utils.load_pretrained import load_lerobot_policy

    checkpoint_path = Path(checkpoint_path)
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device  # the saved training device; load straight onto the requested one
    policy, preprocessor, postprocessor = load_lerobot_policy(
        checkpoint_path, config, device=device, strict=strict
    )
    logger.info("Loaded %s policy from %s", config.type, checkpoint_path)
    return policy, preprocessor, postprocessor
