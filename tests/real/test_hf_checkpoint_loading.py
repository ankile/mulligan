"""All released real checkpoints load from Hugging Face with no W&B (network).

58 DP actors and 18 IDQL critics, each downloaded anonymously (token=False) at its pinned
release revision into a temporary cache that is deleted after the load. ``wandb`` is made
unimportable for the whole module, so any W&B dependency on the load path fails the test.
"""

import json
import shutil
import sys

import pytest

from mulligan.real.policy.dp import (
    checkpoint_dir,
    critic_dp_model_id,
    released_checkpoints,
)

pytestmark = pytest.mark.network

ENTRIES = released_checkpoints()
DPS = [e for e in ENTRIES if e["kind"] == "dp-actor"]
CRITICS = [e for e in ENTRIES if e["kind"] == "idql-critic"]
MISMATCHED = [c for c in CRITICS if c["dp_resolution"] != "exact"]


def _model_id(entry):
    sub = f"/{entry['subfolder']}" if entry["subfolder"] else ""
    return f"hf://{entry['repo']}{sub}@{entry['revision']}"


@pytest.fixture(autouse=True)
def no_wandb(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", None)  # `import wandb` raises ImportError


@pytest.fixture
def hf_cache(tmp_path):
    cache = tmp_path / "hf"
    yield cache
    shutil.rmtree(cache, ignore_errors=True)


@pytest.mark.parametrize("entry", DPS, ids=[e["repo"] for e in DPS])
def test_dp_loads_from_hf(entry, hf_cache):
    from mulligan.real.policy.loader import LeRobotRealWorldPolicy, load_policy_by_model_id

    path = checkpoint_dir(_model_id(entry), cache_dir=hf_cache, token=False)
    loaded = load_policy_by_model_id(str(path), policy_id=0, device="cpu")
    assert isinstance(loaded.policy, LeRobotRealWorldPolicy)
    assert loaded.policy.config.n_action_steps == 6
    assert (loaded.camera_height, loaded.camera_width) == (224, 224)


@pytest.mark.parametrize("entry", CRITICS, ids=[e["repo"] for e in CRITICS])
def test_critic_loads_from_hf(entry, hf_cache):
    from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

    critic_dir = checkpoint_dir(_model_id(entry), cache_dir=hf_cache, token=False)
    metadata = json.loads((critic_dir / "metadata.json").read_text())
    # The shipped table agrees with the critic's own metadata.
    assert metadata["dp_artifact"] == entry.get("dp_checkpoint", f"hf://{entry['dp_repo']}")
    override = None
    if entry["dp_resolution"] != "exact":
        with pytest.raises(ValueError, match="not released"):
            critic_dp_model_id(metadata["dp_artifact"])
        override = f"hf://{entry['dp_repo']}"
    dp_id = critic_dp_model_id(metadata["dp_artifact"], dp_override=override)
    assert dp_id == f"hf://{entry['dp_repo']}@{entry['dp_revision']}"
    dp_dir = checkpoint_dir(dp_id, cache_dir=hf_cache, token=False)
    policy = VisionIDQLRealWorldPolicy.from_artifact(critic_dir, device="cpu", dp_local_dir=dp_dir)
    assert policy.num_action_samples == metadata["num_action_samples"]
    assert policy.iql_action_dim == metadata["action_dim"]


def test_critic_model_id_end_to_end(monkeypatch, hf_cache):
    """The eval entry point: an ``hf://`` critic id without a revision, default HF cache."""
    import huggingface_hub.constants

    from mulligan.real.policy.loader import load_policy_by_model_id
    from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hf_cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    entry = next(c for c in CRITICS if c["repo"].endswith("square-d2-r05-mulligan-idql-critic"))
    loaded = load_policy_by_model_id(f"hf://{entry['repo']}", policy_id=3, device="cpu")
    assert isinstance(loaded.policy, VisionIDQLRealWorldPolicy)
    assert loaded.model_id == f"hf://{entry['repo']}"
    snapshots = sorted(p.name for p in hf_cache.glob("models--mulligan--*/snapshots/*"))
    assert snapshots == sorted({entry["revision"], entry["dp_revision"]})


@pytest.mark.parametrize("entry", MISMATCHED, ids=[e["repo"] for e in MISMATCHED])
def test_version_mismatch_critic_needs_dp_override(entry, monkeypatch, hf_cache):
    import huggingface_hub.constants

    from mulligan.real.policy.loader import load_policy_by_model_id

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hf_cache))
    with pytest.raises(ValueError, match="not released"):
        load_policy_by_model_id(f"hf://{entry['repo']}", policy_id=0, device="cpu")
    loaded = load_policy_by_model_id(
        f"hf://{entry['repo']}",
        policy_id=0,
        device="cpu",
        dp_artifact_override=f"hf://{entry['dp_repo']}",
    )
    assert loaded.policy.dp.config.n_action_steps == 6
