"""Resolve and fetch policy checkpoints: local directories, the HF Hub, and (optionally) W&B.

Released checkpoints live in ``mulligan/*`` model repos on the Hugging Face Hub. A
checkpoint source is one of

* a local checkpoint directory;
* ``hf://<org>/<repo>[@<revision>][/<subfolder>]``, e.g.
  ``hf://mulligan/sim-square-narrow-r01-mulligan-idql@release-1/seed-1`` (the order
  ``hf://<org>/<repo>[/<subfolder>][@<revision>]`` is accepted too). Without
  ``@<revision>`` a ``mulligan/*`` repo loads at its pin in ``release/revisions.json``;
* a W&B artifact (``wandb://entity/project/name:version`` or ``entity/project/name:version``),
  for checkpoints of your own runs. This needs ``wandb`` and a login.

:func:`resolve_checkpoint` returns a local directory for any of these. The W&B helpers
below also upload checkpoints when W&B logging is enabled in a training run.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_REMOVED_POLICY_CONFIG_FILES = ("side_crop.json", "dual_side_crop.json", "action_target.json")
_ARTIFACT_READY_FILE = ".mulligan-artifact-ready.json"


def _wandb_run_can_use_artifacts():
    """Return an active run only when it can resolve remote artifacts.

    W&B installs a truthy disabled-mode run whose ``use_artifact`` method
    returns ``None``. Treat disabled and offline runs as API readers instead;
    loading a pinned parent remains valid even when lineage logging is off.
    """
    run = wandb.run
    if run is None:
        return None
    mode = getattr(getattr(run, "settings", None), "mode", None)
    if mode in {"disabled", "offline"}:
        return None
    return run


def _require_resolved_artifact(artifact, artifact_identifier: str):
    if artifact is None:
        raise RuntimeError(
            f"W&B returned no artifact for '{artifact_identifier}'; "
            "the pinned checkpoint cannot be loaded"
        )
    return artifact


@contextmanager
def _artifact_download_lock(download_dir: Path) -> Iterator[None]:
    """Serialize writers for one artifact cache directory across processes."""
    lock_dir = download_dir.parent / ".mulligan-artifact-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(download_dir.resolve()).encode()).hexdigest()
    lock_path = lock_dir / f"{lock_key}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _ready_artifact_name(download_dir: Path) -> str | None:
    marker = download_dir / _ARTIFACT_READY_FILE
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid W&B artifact readiness marker %s: %s", marker, exc)
        return None
    name = payload.get("artifact_name")
    if not isinstance(name, str) or not name:
        logger.warning("Ignoring malformed W&B artifact readiness marker: %s", marker)
        return None
    return name


def _download_managed_artifact_atomically(artifact, download_dir: Path) -> Path:
    """Populate a managed cache without exposing partial files to readers."""
    download_dir.parent.mkdir(parents=True, exist_ok=True)
    with _artifact_download_lock(download_dir):
        if _ready_artifact_name(download_dir) == artifact.name:
            logger.info("Using complete cached W&B artifact: %s", download_dir)
            return download_dir

        partial_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{download_dir.name}.partial-",
                dir=str(download_dir.parent),
            )
        )
        stale_dir: Path | None = None
        try:
            logger.info("Downloading atomically to temporary cache: %s", partial_dir)
            artifact_dir = Path(artifact.download(root=str(partial_dir)))
            if artifact_dir.resolve() != partial_dir.resolve():
                raise RuntimeError(
                    "W&B artifact download returned an unexpected root: "
                    f"expected {partial_dir}, got {artifact_dir}"
                )
            (partial_dir / _ARTIFACT_READY_FILE).write_text(
                json.dumps({"artifact_name": artifact.name}, sort_keys=True) + "\n"
            )

            if download_dir.exists():
                stale_dir = download_dir.with_name(f".{download_dir.name}.stale-{uuid.uuid4().hex}")
                os.replace(download_dir, stale_dir)
            os.replace(partial_dir, download_dir)
            if stale_dir is not None:
                shutil.rmtree(stale_dir)
            return download_dir
        except BaseException:
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            if stale_dir is not None and stale_dir.exists() and not download_dir.exists():
                os.replace(stale_dir, download_dir)
            raise


def get_artifact_cache_root() -> Path:
    """Directory for cached W&B checkpoint downloads.

    ``MULLIGAN_WANDB_ARTIFACT_CACHE`` wins; otherwise ``$XDG_CACHE_HOME/mulligan/wandb_artifacts``
    (``~/.cache`` when ``XDG_CACHE_HOME`` is unset). HF downloads use the HF Hub cache.
    """
    override = os.environ.get("MULLIGAN_WANDB_ARTIFACT_CACHE")
    if override:
        return Path(override)
    cache_home = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_home) / "mulligan" / "wandb_artifacts"


class _LazyWandbModule:
    def _load(self):
        return importlib.import_module("wandb")

    def __getattr__(self, name):
        return getattr(self._load(), name)


wandb = _LazyWandbModule()


def _normalize_wandb_artifact_identifier(artifact_identifier: str) -> str:
    """Return the SDK artifact path for a W&B artifact URI or bare artifact path."""
    if not artifact_identifier:
        raise ValueError("W&B artifact identifier must be non-empty")
    if artifact_identifier.startswith("wandb://"):
        artifact_identifier = artifact_identifier.removeprefix("wandb://")
    elif "://" in artifact_identifier:
        raise ValueError(
            f"Unsupported artifact URI scheme in {artifact_identifier!r}; expected wandb://"
        )
    if not artifact_identifier:
        raise ValueError("W&B artifact identifier must include a path after wandb://")
    return artifact_identifier


def get_checkpoint_files(checkpoint_path: Path) -> list[Path]:
    """
    Get all relevant checkpoint files that should be uploaded to W&B.

    Supports LeRobot policy checkpoints and agent (IDQL/DIVL) checkpoints.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        List of Path objects for all checkpoint files that exist
    """
    # LeRobot policy files (diffusion policy)
    lerobot_files = [
        "config.json",  # Policy configuration
        "model.safetensors",  # Model weights in SafeTensors format
        "policy_preprocessor.json",  # Input normalization pipeline config
        "policy_postprocessor.json",  # Output denormalization pipeline config
    ]

    # Agent (IDQL/DIVL) checkpoint files
    agent_files = [
        "policy.pt",  # Agent weights and pickled config
        "stats.json",  # Normalization statistics
        "metadata.json",  # Checkpoint metadata (env, robot, etc.)
        "critic_value_summary.json",  # Frozen-actor Q/V provenance and freeze audit
    ]

    # Dynamic processor files (named with step index)
    # These contain the actual normalization statistics (mean/std) for LeRobot policies
    dynamic_files = []
    for file_path in checkpoint_path.glob("policy_*_step_*.safetensors"):
        dynamic_files.append(file_path.name)

    # Collect all files that exist
    all_possible_files = lerobot_files + agent_files + dynamic_files
    existing_files = []

    for filename in all_possible_files:
        file_path = checkpoint_path / filename
        if file_path.exists():
            existing_files.append(file_path)

    if not existing_files:
        logger.warning(f"No checkpoint files found in {checkpoint_path}")

    return existing_files


def upload_checkpoint_to_wandb(
    checkpoint_path: Path,
    artifact_name: str,
    artifact_type: str = "policy",
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[wandb.Artifact]:
    """
    Upload a complete LeRobot policy checkpoint to W&B as an artifact.

    This uploads all necessary files for loading the policy later with:
    `policy, preprocessor, postprocessor = load_checkpoint_from_wandb(...)`

    Args:
        checkpoint_path: Path to the checkpoint directory (containing config.json, model.safetensors, etc.)
        artifact_name: Name for the W&B artifact (will be sanitized)
        artifact_type: Type of artifact (default: "policy")
        description: Optional description for the artifact
        metadata: Optional metadata dict to attach to the artifact

    Returns:
        The uploaded wandb.Artifact object, or None if not logged in to W&B

    Example:
        >>> checkpoint_path = Path("checkpoints/best_model")
        >>> artifact = upload_checkpoint_to_wandb(
        ...     checkpoint_path,
        ...     artifact_name="marker-d2-dp-best",
        ...     description="Best diffusion policy for Marker (success: 95%)",
        ...     metadata={"task": "marker_d2", "success_rate": 0.95}
        ... )
    """
    if not wandb.run:
        logger.warning("W&B run not active. Skipping artifact upload.")
        return None

    # Sanitize artifact name for W&B compatibility
    artifact_name = _sanitize_artifact_name(artifact_name)

    logger.info(f"Creating W&B artifact: {artifact_name}")
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
        metadata=metadata,
    )

    # Get all checkpoint files
    checkpoint_files = get_checkpoint_files(checkpoint_path)

    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {checkpoint_path}")
        return None

    # Add each file to the artifact
    logger.info(f"Adding {len(checkpoint_files)} files to artifact")
    for file_path in checkpoint_files:
        # Use just the filename as the artifact path (preserves directory structure)
        artifact.add_file(str(file_path), name=file_path.name)
        logger.debug(f"  Added: {file_path.name}")

    # Log the artifact
    logger.info("Uploading artifact to W&B...")
    wandb.log_artifact(artifact)
    logger.info(f"✓ Artifact '{artifact_name}' uploaded successfully")

    return artifact


def download_checkpoint_from_wandb(
    artifact_identifier: str,
    download_dir: Optional[Path] = None,
) -> Path:
    """
    Download a policy checkpoint from W&B artifact.

    The checkpoint includes all files needed to reconstruct the policy:
    - model.safetensors (weights)
    - config.json (architecture)
    - policy_preprocessor.json + .safetensors (input normalization)
    - policy_postprocessor.json + .safetensors (output normalization)

    Args:
        artifact_identifier: Artifact identifier in format "entity/project/name:version" or "name:version"
        download_dir: Directory to download to (default: uses cache)

    Returns:
        Path to the downloaded checkpoint directory

    Example:
        >>> # Download best checkpoint for DAgger
        >>> checkpoint_path = download_checkpoint_from_wandb(
        ...     artifact_identifier="my-entity/my-project/idql-best:v0"
        ... )
        >>> # Load for DAgger training
        >>> from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        >>> policy = DiffusionPolicy.from_pretrained(str(checkpoint_path))
    """
    raw_artifact_identifier = artifact_identifier
    artifact_identifier = _normalize_wandb_artifact_identifier(artifact_identifier)
    logger.info(f"Downloading artifact from W&B: {raw_artifact_identifier}")
    if artifact_identifier != raw_artifact_identifier:
        logger.info(f"Resolved W&B artifact path: {artifact_identifier}")

    # Use run.use_artifact() if we have an active run (proper W&B lineage tracking)
    # Otherwise fall back to Api().artifact() (only requires being logged in)
    artifact_run = _wandb_run_can_use_artifacts()
    if artifact_run is not None:
        logger.info("Using run.use_artifact() to track artifact usage in this run")
        artifact = artifact_run.use_artifact(artifact_identifier)
    else:
        logger.info("No active run, using Api().artifact()")
        api = wandb.Api()
        try:
            artifact = api.artifact(artifact_identifier)
        except Exception as e:
            logger.error(f"Failed to find artifact '{artifact_identifier}': {e}")
            raise
    artifact = _require_resolved_artifact(artifact, artifact_identifier)

    # The default cache is managed entirely by this helper: download into a
    # private sibling and atomically publish it under a cross-process lock. W&B
    # deliberately leaves existing files untouched, so two unlocked downloads
    # to one root can expose a partially written checkpoint to a reader.
    managed_cache = download_dir is None
    if managed_cache:
        download_dir = get_artifact_cache_root() / artifact.name
        return _download_managed_artifact_atomically(artifact, download_dir)

    # An explicit directory may contain caller-owned files, so preserve the
    # merge-in-place behavior while still serializing writers.
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # W&B download roots are reused local directories. Delete removed policy-config
    # files before downloading so stale cache files cannot become hidden policy
    # state. If the artifact itself still contains one, artifact.download() will
    # recreate it and eval/rollout will refuse the checkpoint.
    with _artifact_download_lock(download_dir):
        for filename in _REMOVED_POLICY_CONFIG_FILES:
            stale_path = download_dir / filename
            if stale_path.exists():
                stale_path.unlink()
                logger.info(
                    "Removed stale local policy-config file before download: %s", stale_path
                )

        logger.info(f"Downloading to: {download_dir}")
        artifact_dir = artifact.download(root=str(download_dir))

    return Path(artifact_dir)


def _sanitize_artifact_name(name: str, max_length: int = 128) -> str:
    """
    Sanitize artifact name for W&B compatibility.

    W&B has restrictions on artifact names:
    - Must contain only alphanumeric characters, dashes, underscores, and dots
    - No spaces or special characters
    - Maximum length of 128 characters
    """
    import hashlib
    import re

    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # If name is too long, truncate and add hash to preserve uniqueness
    if len(sanitized) > max_length:
        # Create hash of the full name for uniqueness
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        # Keep prefix and suffix, insert hash in middle
        # Format: "prefix...hash...suffix" to maintain readability
        available_length = max_length - len(name_hash) - 2  # -2 for separators
        prefix_length = available_length // 2
        suffix_length = available_length - prefix_length

        prefix = sanitized[:prefix_length].rstrip("_")
        suffix = sanitized[-suffix_length:].lstrip("_")
        sanitized = f"{prefix}-{name_hash}-{suffix}"

        logger.warning(
            f"Artifact name truncated from {len(name)} to {len(sanitized)} chars. "
            f"Original: {name[:50]}... Hash: {name_hash}"
        )

    return sanitized


def create_artifact_metadata(
    repo_id: str,
    success_rate: float,
    avg_reward: float,
    step: int,
    is_best: bool = False,
    dataset_size: int = 0,
    checkpoint_index: Optional[int] = None,
    # Environment configuration
    env: Optional[str] = None,
    robot: Optional[str] = None,
    controller: Optional[str] = None,
    env_config: Optional[str] = None,
    # Camera configuration
    cameras: Optional[list[str]] = None,
    camera_height: Optional[int] = None,
    camera_width: Optional[int] = None,
    # Policy configuration
    policy_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    n_obs_steps: Optional[int] = None,
    action_dim: Optional[int] = None,
    state_dim: Optional[int] = None,
    action_target: Optional[str] = None,
    cartesian_action_frame: Optional[str] = None,
    action_mode: Optional[str] = None,
    camera_crop_boxes: Optional[dict[str, list[int] | tuple[int, int, int, int]]] = None,
    dual_side_crop_boxes: Optional[dict[str, list[int] | tuple[int, int, int, int]]] = None,
) -> dict:
    """
    Create standardized metadata dict for artifact logging.

    This metadata enables evaluation scripts to automatically configure the environment
    and policy without requiring manual parameter specification.

    Args:
        repo_id: Dataset repository ID
        success_rate: Evaluation success rate (0-1)
        avg_reward: Average episode reward
        step: Training step number
        is_best: Whether this is the best checkpoint
        dataset_size: Number of training frames
        checkpoint_index: Index if periodic checkpoint (e.g., 1st, 2nd periodic)

        # Environment configuration (for evaluation)
        env: Environment name (e.g., "NutAssemblySquare")
        robot: Robot type (e.g., "Panda")
        controller: Controller type (optional)
        env_config: Environment configuration (e.g., "bimanual", "default")

        # Camera configuration (for vision-based policies)
        cameras: List of camera names (e.g., ["agentview", "robot0_eye_in_hand"])
        camera_height: Camera image height (e.g., 256)
        camera_width: Camera image width (e.g., 256)

        # Policy configuration (for inference)
        policy_type: Policy class name (e.g., "ACTPolicy")
        chunk_size: Action chunk size (e.g., 100)
        n_obs_steps: Number of observation steps (e.g., 1)
        action_dim: Action space dimension
        state_dim: State observation dimension

    Returns:
        Metadata dictionary suitable for W&B artifact
    """
    metadata = {
        "repo_id": repo_id,
        "success_rate": f"{success_rate:.3f}",
        "avg_reward": f"{avg_reward:.3f}",
        "step": step,
        "is_best": is_best,
        "dataset_size": dataset_size,
    }

    if checkpoint_index is not None:
        metadata["checkpoint_index"] = checkpoint_index

    # Add environment configuration if provided
    if env is not None:
        metadata["env"] = env
    if robot is not None:
        metadata["robot"] = robot
    if controller is not None:
        metadata["controller"] = controller
    if env_config is not None:
        metadata["env_config"] = env_config

    # Add camera configuration if provided
    if cameras is not None:
        metadata["cameras"] = cameras
    if camera_height is not None:
        metadata["camera_height"] = camera_height
    if camera_width is not None:
        metadata["camera_width"] = camera_width

    # Add policy configuration if provided
    if policy_type is not None:
        metadata["policy_type"] = policy_type
    if chunk_size is not None:
        metadata["chunk_size"] = chunk_size
    if n_obs_steps is not None:
        metadata["n_obs_steps"] = n_obs_steps
    if action_dim is not None:
        metadata["action_dim"] = action_dim
    if state_dim is not None:
        metadata["state_dim"] = state_dim
    if action_target is not None:
        # Real-robot action space the policy was trained to predict
        # ("cartesian_velocity" or "cartesian_position"). Lets eval refuse to run
        # a position-target policy through the velocity-only robot path.
        metadata["action_target"] = action_target
    if cartesian_action_frame is not None:
        # Frame for cartesian_velocity targets. Legacy checkpoints omit this and
        # are interpreted as DROID/base frame.
        metadata["cartesian_action_frame"] = cartesian_action_frame
    if action_mode is not None:
        # UMI relative-pose action representation ("relative") vs the older
        # per-frame velocity/absolute-pose target ("absolute"/None). Lets eval
        # auto-detect the relative->absolute inversion.
        metadata["action_mode"] = action_mode
    if camera_crop_boxes:
        metadata["camera_crop_boxes"] = {cam: list(box) for cam, box in camera_crop_boxes.items()}
    if dual_side_crop_boxes:
        metadata["dual_side_crop_boxes"] = {
            cam: list(box) for cam, box in dual_side_crop_boxes.items()
        }

    return metadata


# ---------------------------------------------------------------------------
# Hugging Face Hub checkpoints
# ---------------------------------------------------------------------------

HF_SCHEME = "hf://"
_HF_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# A revision is a commit sha, a tag or branch name, or a PR ref.
_HF_REV_TOKEN = re.compile(r"^[A-Za-z0-9][\w.-]*$")
_HF_PR_REF = re.compile(r"^refs/pr/\d+$")
_HF_GRAMMAR = (
    "hf://<org>/<repo>[@<revision>][/<subfolder>] (or hf://<org>/<repo>[/<subfolder>][@<revision>])"
)


@dataclass(frozen=True)
class HubCheckpoint:
    """A checkpoint directory inside an HF model repo."""

    repo_id: str
    revision: str | None
    subfolder: str | None

    def uri(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        sub = f"/{self.subfolder}" if self.subfolder else ""
        return f"{HF_SCHEME}{self.repo_id}{rev}{sub}"


def _split_revision(text: str, uri: str) -> tuple[str, str | None]:
    """``<revision>[/<subfolder>]`` -> (revision, subfolder)."""
    parts = text.split("/")
    if parts[0] == "refs":
        if len(parts) < 3 or not _HF_PR_REF.match("/".join(parts[:3])):
            raise ValueError(f"{uri!r}: the only ref revision accepted is refs/pr/<N>")
        return "/".join(parts[:3]), "/".join(parts[3:]) or None
    return parts[0], "/".join(parts[1:]) or None


def parse_hf_uri(uri: str) -> HubCheckpoint:
    """Parse an ``hf://`` reference; the one parser for every ``hf://`` id in the release.

    Both orders are accepted: ``hf://<org>/<repo>@<revision>/<subfolder>`` and
    ``hf://<org>/<repo>/<subfolder>@<revision>``. ``<revision>`` is a commit sha, a tag or
    branch name, or ``refs/pr/<N>``. A subfolder on both sides of ``@``, more than one
    ``@`` or an empty part is rejected. The revision stays ``None`` when absent; see
    :func:`default_revision`.
    """
    if not isinstance(uri, str) or not uri.startswith(HF_SCHEME):
        raise ValueError(f"Not an HF checkpoint URI: {uri!r}; expected {_HF_GRAMMAR}")
    body = uri[len(HF_SCHEME) :].rstrip("/")
    path, sep, rest = body.partition("@")
    if sep and (not rest or "@" in rest):
        raise ValueError(f"{uri!r}: expected a single non-empty '@<revision>'")
    parts = path.split("/")
    if len(parts) < 2 or not all(parts) or not _HF_REPO.match("/".join(parts[:2])):
        raise ValueError(f"Not an HF checkpoint URI: {uri!r}; expected {_HF_GRAMMAR}")
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) or None
    revision = None
    if sep:
        revision, after = _split_revision(rest, uri)
        if after is not None:
            if subfolder is not None:
                raise ValueError(
                    f"{uri!r} is ambiguous: a subfolder on both sides of '@<revision>'"
                )
            subfolder = after
        if not (_HF_REV_TOKEN.match(revision) or _HF_PR_REF.match(revision)):
            raise ValueError(f"{uri!r}: malformed revision {revision!r}")
    if subfolder is not None and not all(subfolder.split("/")):
        raise ValueError(f"{uri!r}: empty path component in the subfolder")
    return HubCheckpoint(repo_id, revision, subfolder)


def release_revision(repo_id: str) -> str:
    """The pinned revision of a released repo (``release/revisions.json``)."""
    from mulligan.release.download import pinned_revision

    try:
        return pinned_revision(repo_id)
    except KeyError:
        raise LookupError(
            f"{repo_id} has no release pin in release/revisions.json. "
            f"Pin the revision explicitly: hf://{repo_id}@<revision>/<subfolder>"
        ) from None


def default_revision(repo_id: str) -> str | None:
    """Revision to use when an ``hf://`` id has no ``@<revision>``: the release pin for
    released and ``mulligan/*`` repos (an unpinned ``mulligan/*`` repo raises), the
    default branch (``None``) for any other repo."""
    from mulligan.release.download import load_revisions

    if repo_id.startswith("mulligan/"):
        return release_revision(repo_id)
    try:
        pinned = repo_id in load_revisions()
    except FileNotFoundError:
        # Installed package without a release checkout: other orgs have no pin.
        return None
    return release_revision(repo_id) if pinned else None


def download_hub_checkpoint(ref: HubCheckpoint | str, *, cache_dir: Path | None = None) -> Path:
    """Download one checkpoint directory from the HF Hub and return its local path."""
    from huggingface_hub import snapshot_download

    if isinstance(ref, str):
        ref = parse_hf_uri(ref)
    revision = ref.revision or default_revision(ref.repo_id)
    allow = [f"{ref.subfolder}/*"] if ref.subfolder else None
    logger.info("Downloading %s@%s (%s)", ref.repo_id, revision, ref.subfolder or "repo root")
    local = Path(
        snapshot_download(
            ref.repo_id,
            revision=revision,
            allow_patterns=allow,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    path = local / ref.subfolder if ref.subfolder else local
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(f"{ref.uri()} has no files at revision {revision}")
    return path


def resolve_checkpoint(source: str | Path, *, cache_dir: Path | None = None) -> Path:
    """Return a local checkpoint directory for a local path, ``hf://`` URI or W&B artifact."""
    text = str(source)
    if text.startswith(HF_SCHEME):
        return download_hub_checkpoint(parse_hf_uri(text), cache_dir=cache_dir)
    path = Path(text).expanduser()
    if path.is_dir():
        return path
    if text.startswith("wandb://") or re.match(r"^[\w.-]+/[\w.-]+/[\w.-]+:[\w.-]+$", text):
        # cache_dir is a cache ROOT here too (as for hf://): one subdirectory per
        # artifact, never a shared download target that merges two checkpoints.
        download_dir = (
            None
            if cache_dir is None
            else Path(cache_dir) / text.removeprefix("wandb://").replace("/", "--")
        )
        return download_checkpoint_from_wandb(text, download_dir=download_dir)
    raise FileNotFoundError(
        f"Checkpoint source {text!r} is neither a local directory, an hf:// URI nor a W&B "
        "artifact (entity/project/name:version)."
    )
