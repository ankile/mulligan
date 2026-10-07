"""`manifest_eval` repo id resolution: `--no-push` needs no Hugging Face login."""

from __future__ import annotations

import argparse

import pytest

from mulligan.real.eval import manifest_eval


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"hf_repo_id": None, "no_push": False, **kw})


def test_explicit_repo_id_wins():
    assert manifest_eval.resolve_hf_repo_id(_args(hf_repo_id="me/x", no_push=True), "d") == "me/x"


def test_no_push_needs_no_login(monkeypatch):
    import huggingface_hub

    def _fail(*_a, **_k):
        raise AssertionError("whoami must not be called with --no-push")

    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", _fail)
    assert manifest_eval.resolve_hf_repo_id(_args(no_push=True), "w8-eval") == "local/w8-eval"


def test_push_uses_the_logged_in_user(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", lambda self: {"name": "someone"})
    assert manifest_eval.resolve_hf_repo_id(_args(), "w8-eval") == "someone/w8-eval"


@pytest.mark.parametrize("name", ["w8-eval-smoke"])
def test_local_id_is_a_valid_repo_id(name):
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(manifest_eval.resolve_hf_repo_id(_args(no_push=True), name))
