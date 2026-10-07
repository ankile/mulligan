"""Shared test setup for mulligan.real.stage_specs.

``google.genai`` is uninstallable in several environments (GLIBC), so the
stage-labeling tests run against a minimal stub. Installing it ONCE here, at
conftest collection (before any test module imports), gives every test the same
``types.Schema`` / ``types.Part`` behavior — the per-file ``_install_genai_stub``
helpers then no-op via their ``if "google.genai" in sys.modules`` guard, so two
files cannot install stubs that disagree. This stub covers both uses: Schema defaults every attribute the
schema-equivalence reduction reads, and Part/Blob/etc. store their kwargs so the
labeler part-assembly tests can inspect them.
"""

from __future__ import annotations

import sys
import types as _pytypes


def install_genai_stub() -> None:
    if "google.genai" in sys.modules:
        return
    # Reuse the REAL ``google`` namespace package so its ``__path__`` is preserved
    # and ``google.protobuf`` (hence ``wandb``) stays importable for the rest of the
    # session. A bare ``ModuleType("google")`` has no ``__path__``, which shadows the
    # namespace and makes every later ``import wandb`` die with
    # "No module named 'google.protobuf'; 'google' is not a package". We only graft
    # the ``google.genai`` stub on top; we never replace the real package.
    try:
        import google  # real PEP-420 namespace package (binds the module, keeps __path__)
    except ImportError:  # google truly absent: fall back to a path-less stub
        google = _pytypes.ModuleType("google")
    genai = _pytypes.ModuleType("google.genai")
    errors = _pytypes.ModuleType("google.genai.errors")
    types_mod = _pytypes.ModuleType("google.genai.types")

    class APIError(Exception):
        def __init__(self, code: int = 0, *a: object) -> None:
            super().__init__(*a)
            self.code = code

    class _Type:
        INTEGER = "INTEGER"
        NUMBER = "NUMBER"
        BOOLEAN = "BOOLEAN"
        STRING = "STRING"
        OBJECT = "OBJECT"

    class _Schema:
        _ATTRS = (
            "type", "properties", "required", "enum", "minimum", "maximum", "nullable", "description",
        )  # fmt: skip

        def __init__(self, **kw: object) -> None:
            for attr in self._ATTRS:
                setattr(self, attr, kw.get(attr))
            self.__dict__.update(kw)

    class _Stored:
        def __init__(self, **kw: object) -> None:
            self.__dict__.update(kw)

    errors.APIError = APIError
    types_mod.Type = _Type
    types_mod.Schema = _Schema
    for name in (
        "Part",
        "Blob",
        "FileData",
        "VideoMetadata",
        "Content",
        "GenerateContentConfig",
        "HttpOptions",
    ):
        setattr(types_mod, name, _Stored)
    genai.Client = _Stored
    genai.errors = errors
    genai.types = types_mod
    google.genai = genai
    sys.modules.setdefault("google", google)
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.errors"] = errors
    sys.modules["google.genai.types"] = types_mod


# Install at collection, before any stage-labeling test module imports.
install_genai_stub()
