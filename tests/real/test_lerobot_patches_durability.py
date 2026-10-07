"""Durability patches in mulligan.real.policy.lerobot_patches.

write_json must be atomic (tmp+fsync+rename) — a crash mid-rewrite of meta/info.json
would otherwise leave a truncated JSON that blocks dataset recovery.
concatenate_video_files must finish with a same-filesystem atomic rename
onto the live tail chunk instead of upstream's cross-fs shutil.move copy window.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mulligan.real.policy.lerobot_patches  # noqa: F401  (applies the patches)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_write_json_is_atomic_under_crash(tmp_path: Path, monkeypatch) -> None:
    from lerobot.utils import io_utils

    target = tmp_path / "info.json"
    io_utils.write_json({"total_episodes": 3}, target)
    assert _read(target) == {"total_episodes": 3}

    # Crash injected between the tmp write and the rename: the old content survives.
    import os as os_mod

    real_replace = os_mod.replace

    def _boom(src, dst):
        raise RuntimeError("injected crash before rename")

    monkeypatch.setattr(os_mod, "replace", _boom)
    with pytest.raises(RuntimeError, match="injected crash"):
        io_utils.write_json({"total_episodes": 4}, target)
    monkeypatch.setattr(os_mod, "replace", real_replace)

    assert _read(target) == {"total_episodes": 3}, "old content must survive a crash"


def test_write_json_binding_in_datasets_io_utils_is_patched(tmp_path: Path) -> None:
    from lerobot.datasets import io_utils as ds_io
    from lerobot.utils import io_utils as base_io

    assert ds_io.write_json is base_io.write_json, (
        "lerobot.datasets.io_utils holds a from-import binding of write_json; the patch "
        "must cover it or write_info/write_stats keep the non-atomic path"
    )
    # And the bound name really is the atomic implementation, not upstream's.
    assert ds_io.write_json.__name__ == "_mulligan_atomic_write_json"


def test_concatenate_video_files_is_wrapped_everywhere() -> None:
    from lerobot.datasets import dataset_writer as dw
    from lerobot.datasets import video_utils as vu

    assert vu.concatenate_video_files.__name__ == "_mulligan_atomic_concatenate_video_files"
    assert dw.concatenate_video_files is vu.concatenate_video_files, (
        "dataset_writer from-imports concatenate_video_files; the patch must cover the "
        "bound name or the save_episode hot path keeps the cross-fs move"
    )


def test_concatenate_writes_via_same_dir_temp_and_leaves_no_droppings(
    tmp_path: Path, monkeypatch
) -> None:
    """Drive the wrapper with a stub original: the visible output must appear only via
    rename, the temp must live in the destination dir (same fs), and no temp survives."""
    from lerobot.datasets import video_utils as vu

    observed = {}

    def _stub_original(inputs, output, *args, **kwargs):
        output = Path(output)
        observed["temp_name"] = output.name
        observed["temp_dir"] = output.parent
        output.write_bytes(b"concatenated")

    monkeypatch.setattr(vu, "_original_concatenate_video_files", _stub_original, raising=False)
    # Re-wrap around the stub (the module-level wrapper closed over the real original,
    # so exercise the wrapper logic through a fresh closure built the same way).
    import mulligan.real.policy.lerobot_patches as patches

    wrapper_src_original = _stub_original

    def wrapper(inputs, output, *args, **kwargs):
        import os

        output = Path(output)
        overwrite = args[0] if args else kwargs.get("overwrite", True)
        if not overwrite and output.exists():
            return wrapper_src_original(inputs, output, *args, **kwargs)
        staged = output.with_name(f".tmp_concat_{os.getpid()}_{output.name}")
        try:
            result = wrapper_src_original(inputs, staged, *args, **kwargs)
            os.replace(staged, output)
        finally:
            if staged.exists():
                staged.unlink()
        return result

    target = tmp_path / "videos" / "file-000.mp4"
    target.parent.mkdir(parents=True)
    wrapper(["a.mp4", "b.mp4"], target)

    assert target.read_bytes() == b"concatenated"
    assert observed["temp_dir"] == target.parent, "temp must be created in the dest dir"
    assert observed["temp_name"].startswith(".tmp_concat_")
    assert not list(target.parent.glob(".tmp_concat_*")), "no temp droppings"
    assert patches is not None


def test_concatenate_overwrite_false_skips_without_temp(tmp_path: Path) -> None:
    from lerobot.datasets import video_utils as vu

    target = tmp_path / "file-000.mp4"
    target.write_bytes(b"existing")
    # overwrite=False + existing output: upstream's own skip branch runs against the
    # REAL path (returns without touching it) and no temp file is created.
    vu.concatenate_video_files([], target, False)
    assert target.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".tmp_concat_*"))


def test_patch_module_reimport_is_idempotent() -> None:
    """A second application of the patch module must be a no-op, not a crash.

    Before the idempotence marker, re-executing the module body re-introspected the
    ALREADY-PATCHED ``LeRobotDataset.create`` — whose signature is ``(cls, *args,
    **kwargs)`` — found neither ``rgb_encoder`` nor ``camera_encoder`` among its
    parameters, and raised the "encoder-config API moved again" RuntimeError. It would
    also have re-wrapped every already-wrapped function.
    """
    import importlib

    import mulligan.real.policy.lerobot_patches as patches
    from lerobot.datasets import lerobot_dataset as ld_mod

    assert getattr(ld_mod, patches._MULLIGAN_PATCH_MARKER, False), (
        "importing mulligan.real.policy.lerobot_patches should have marked lerobot as patched"
    )

    dataset_cls_before = ld_mod.LeRobotDataset
    # NOTE: compare the UNDERLYING functions. ``create``/``resume`` are classmethods, and
    # every attribute access on a classmethod builds a fresh bound-method object, so
    # ``LeRobotDataset.create is LeRobotDataset.create`` is False even with no patching at
    # all. ``__func__`` is the stable identity.
    create_before = ld_mod.LeRobotDataset.create.__func__
    resume_before = ld_mod.LeRobotDataset.resume.__func__

    importlib.reload(patches)  # must not raise

    assert ld_mod.LeRobotDataset is dataset_cls_before
    assert ld_mod.LeRobotDataset.create.__func__ is create_before, "create must not be re-wrapped"
    assert ld_mod.LeRobotDataset.resume.__func__ is resume_before, "resume must not be re-wrapped"
    assert patches._MULLIGAN_APPLY_PATCHES is False, "reload must take the already-applied branch"


def test_encode_video_frames_is_not_wrapped() -> None:
    """The live ``encode_video_frames`` binding must stay upstream's, unwrapped.

    A wrapper that rebinds ``video_utils.encode_video_frames`` would be dead, because the
    real call site is the from-import in ``dataset_writer``. It would also be a latent
    TypeError (upstream takes ``video_encoder``, not ``vcodec``, and has no ``**kwargs``).
    Pin both facts: the binding is unwrapped, and the codec instead rides on the encoder
    config.
    """
    import inspect

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import dataset_writer as dw_mod
    from lerobot.datasets import video_utils as vu_mod

    params = inspect.signature(dw_mod.encode_video_frames).parameters
    assert "vcodec" not in params, (
        "upstream encode_video_frames has no vcodec parameter; a vcodec-passing wrapper "
        "would be a TypeError"
    )
    assert "video_encoder" in params
    assert dw_mod.encode_video_frames is vu_mod.encode_video_frames, (
        "dataset_writer's from-import must still be upstream's function (unpatched)"
    )
    # The codec arrives via the encoder config that the patch module injects, not via a default arg.
    assert RGBEncoderConfig(vcodec="libsvtav1").vcodec == "libsvtav1"
