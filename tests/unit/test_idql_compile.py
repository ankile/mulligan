"""Tests for IDQL actor compile configuration helpers."""

import torch
import pytest

from mulligan.agents.idql import IDQLPolicy


def test_configure_actor_compile_disabled_clears_compiled_callable():
    policy = IDQLPolicy.__new__(IDQLPolicy)
    policy._compiled_actor_forward = object()

    policy.configure_actor_compile(enabled=False)

    assert policy._compiled_actor_forward is None


def test_configure_actor_compile_enabled_sets_compiled_callable(monkeypatch):
    def fake_compile(fn, *, mode, fullgraph):
        assert mode == "default"
        assert fullgraph is False
        return ("compiled", fn)

    class Actor:
        _forward_hooks = {}
        _forward_pre_hooks = {}

        def forward(self, batch):
            return batch

    monkeypatch.setattr(torch, "compile", fake_compile)
    policy = IDQLPolicy.__new__(IDQLPolicy)
    policy.actor = Actor()
    policy._compiled_actor_forward = None

    policy.configure_actor_compile(enabled=True)

    assert policy._compiled_actor_forward == ("compiled", policy.actor.forward)


def test_configure_actor_compile_rejects_actor_forward_hooks(monkeypatch):
    class Actor:
        def __init__(self):
            self._forward_hooks = {0: lambda *args: None}

        def forward(self, batch):
            return batch

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("torch.compile should not be called when hooks are present")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    policy = IDQLPolicy.__new__(IDQLPolicy)
    policy.actor = Actor()
    policy._compiled_actor_forward = None

    with pytest.raises(RuntimeError, match="forward hooks are registered"):
        policy.configure_actor_compile(enabled=True)


def test_configure_actor_compile_rejects_actor_forward_pre_hooks(monkeypatch):
    class Actor:
        _forward_hooks = {}

        def __init__(self):
            self._forward_pre_hooks = {0: lambda *args: None}

        def forward(self, batch):
            return batch

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("torch.compile should not be called when pre-hooks are present")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    policy = IDQLPolicy.__new__(IDQLPolicy)
    policy.actor = Actor()
    policy._compiled_actor_forward = None

    with pytest.raises(RuntimeError, match="forward pre-hooks are registered"):
        policy.configure_actor_compile(enabled=True)
