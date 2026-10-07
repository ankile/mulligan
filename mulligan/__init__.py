"""
Mulligan: Performance-Guided Data Collection for Efficient On-Robot Learning.

Training (IDQL / DIVL), simulation and real-robot data collection,
evaluation, and the release manifests of the Mulligan paper.
"""

__version__ = "0.1.0"


def _run_import_patch_quietly(fn):
    """Apply import-time patches without polluting CLI stdout/stderr.

    Many repository tools are used as machine-readable CLIs. Robosuite emits
    setup warnings during import on systems without optional assets installed;
    those warnings should not corrupt JSON output before the actual CLI starts.
    Set MULLIGAN_VERBOSE_IMPORT=1 to restore third-party import chatter for
    debugging.
    """
    import os

    if os.environ.get("MULLIGAN_VERBOSE_IMPORT") == "1":
        fn()
        return

    import contextlib
    import io
    import logging

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            fn()
    finally:
        logging.disable(previous_disable_level)


def apply_runtime_patches() -> None:
    """Apply the robosuite and lerobot runtime patches.

    Not run on ``import mulligan``: the patches import the robotics stacks, which makes
    lightweight CLIs slow to start. Entry points that build environments or policies
    call this first.
    """
    from mulligan.sim._patches import apply_all_patches as apply_robosuite_patches
    from mulligan.utils.lerobot_patches import apply_all_patches as apply_lerobot_patches

    _run_import_patch_quietly(apply_robosuite_patches)
    _run_import_patch_quietly(apply_lerobot_patches)
