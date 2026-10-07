"""Utility functions for the mulligan package."""

from importlib import import_module

__all__ = [
    "filter_episodes",
    "load_policy",
    "load_policy_from_checkpoint",
]

_EXPORT_MODULES = {
    "filter_episodes": "mulligan.data.transforms",
    "load_policy": "mulligan.utils.load_pretrained",
    "load_policy_from_checkpoint": "mulligan.utils.load_pretrained",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _EXPORT_MODULES[name]
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
