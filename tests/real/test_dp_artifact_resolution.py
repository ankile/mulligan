"""Every released IDQL critic's dp_artifact resolves to a released DP actor.

The table (mulligan/real/policy/released_checkpoints.json) is checked against itself here;
test_hf_checkpoint_loading.py checks it against the critics' metadata.json on HF.
"""

import re

import pytest

from mulligan.real.policy.dp import (
    HFModelRef,
    critic_dp_model_id,
    parse_hf_model_id,
    released_checkpoint,
    released_checkpoints,
    resolve_model_id,
)

# The one critic whose metadata names a DP version that was not released: its dp_artifact
# is an ``unreleased/...`` id (an auto-resume checkpoint of the run whose canonical final
# checkpoint is released as mulligan/real-routing-d2-velocity-r05-mulligan-dp). It loads only
# with an explicit DP override. Any new mismatch must be added here deliberately.
KNOWN_DP_VERSION_MISMATCH = {"mulligan/real-routing-d2-velocity-r05-mulligan-idql-critic"}

ENTRIES = released_checkpoints()
CRITICS = [e for e in ENTRIES if e["kind"] == "idql-critic"]
DPS = {e["repo"]: e for e in ENTRIES if e["kind"] == "dp-actor"}


def test_table_counts_and_shape():
    assert len(ENTRIES) == 76
    assert len(DPS) == 58
    assert len(CRITICS) == 18
    assert len({(e["repo"], e["subfolder"]) for e in ENTRIES}) == 76
    for e in ENTRIES:
        assert e["repo"].startswith("mulligan/real-"), e["repo"]
        assert re.fullmatch(r"[0-9a-f]{40}", e["revision"]), e
        assert e["kind"] in ("dp-actor", "idql-critic")


def test_table_pins_match_release_revisions():
    """The shipped table (package data, usable without a repo checkout) repeats the
    canonical pins of release/revisions.json; regenerate it when those change."""
    from mulligan.release.download import pinned_revision

    for e in ENTRIES:
        assert e["revision"] == pinned_revision(e["repo"]), e["repo"]
        if e["kind"] == "idql-critic":
            assert e["dp_revision"] == pinned_revision(e["dp_repo"]), e["repo"]


def test_table_names_release_ids_only():
    for e in ENTRIES:
        for key, value in e.items():
            if isinstance(value, str):
                assert "://" not in value, (e["repo"], key)
        if e["kind"] == "idql-critic" and e["dp_resolution"] != "exact":
            assert re.fullmatch(r"unreleased/[\w./-]+", e["dp_checkpoint"]), e


@pytest.mark.parametrize("critic", CRITICS, ids=[c["repo"] for c in CRITICS])
def test_critic_dp_resolves_to_released_dp(critic):
    dp = DPS[critic["dp_repo"]]
    assert critic["dp_revision"] == dp["revision"]
    if critic["dp_resolution"] == "exact":
        assert "dp_checkpoint" not in critic  # the released DP is the trained one
        # The loader resolves the metadata value (the DP's hf:// id) to the pinned release.
        expected = HFModelRef(dp["repo"], dp["subfolder"], dp["revision"]).model_id
        assert critic_dp_model_id(f"hf://{dp['repo']}") == expected
    else:
        assert critic["dp_resolution"] == "version-mismatch"
        assert critic["dp_checkpoint"].startswith("unreleased/")


def test_dp_version_mismatches_are_exactly_the_known_set():
    mismatched = {c["repo"] for c in CRITICS if c["dp_resolution"] != "exact"}
    assert mismatched == KNOWN_DP_VERSION_MISMATCH


def test_version_mismatch_fails_loud_without_override():
    (critic,) = [c for c in CRITICS if c["repo"] in KNOWN_DP_VERSION_MISMATCH]
    metadata_value = critic["dp_checkpoint"]
    with pytest.raises(ValueError, match="not released"):
        critic_dp_model_id(metadata_value)
    override = f"hf://{critic['dp_repo']}"
    assert critic_dp_model_id(metadata_value, dp_override=override) == (
        f"hf://{critic['dp_repo']}@{critic['dp_revision']}"
    )


def test_own_wandb_dp_passes_through_to_optional_wandb_path():
    artifact = "wandb://ent/proj/my-dp-final:v0"
    assert critic_dp_model_id(artifact) == artifact


@pytest.mark.parametrize(
    ("model_id", "ref"),
    [
        ("hf://org/name", HFModelRef("org/name")),
        ("hf://org/name@abc123", HFModelRef("org/name", None, "abc123")),
        ("hf://org/name/sub/dir@v1.0", HFModelRef("org/name", "sub/dir", "v1.0")),
        ("hf://org/name/sub", HFModelRef("org/name", "sub", None)),
    ],
)
def test_parse_hf_model_id_roundtrip(model_id, ref):
    assert parse_hf_model_id(model_id) == ref
    assert ref.model_id == model_id


@pytest.mark.parametrize("bad", ["hf://org", "hf://org/name@", "hf://org/name@a@b", "hf:///name"])
def test_parse_hf_model_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_hf_model_id(bad)


def test_resolve_model_id_pins_release_revision():
    dp = next(iter(DPS.values()))
    assert resolve_model_id(f"hf://{dp['repo']}") == f"hf://{dp['repo']}@{dp['revision']}"
    assert resolve_model_id(f"hf://{dp['repo']}@main") == f"hf://{dp['repo']}@main"
    assert resolve_model_id("hf://someone/else") == "hf://someone/else"
    assert resolve_model_id("/some/local/dir") == "/some/local/dir"


def test_released_checkpoint_lookup():
    critic = CRITICS[0]
    assert released_checkpoint(critic["repo"]) is critic
    assert released_checkpoint("mulligan/not-a-repo") is None
