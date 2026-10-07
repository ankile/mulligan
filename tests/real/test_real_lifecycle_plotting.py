"""Contract test for the shared real-lifecycle plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fixed_probe_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_title("determinism probe")
    return fig


def test_write_svg_deterministic_output(tmp_path, capsys) -> None:
    """Pin the shared SVG-writer contract of the lifecycle figures.

    Contract (current behavior, verified empirically): parent dirs are created;
    ``metadata={"Date": None}`` drops the ``<dc:date>`` timestamp; every line is
    right-stripped and the file ends in exactly one newline; the figure is closed;
    output is byte-deterministic under a fixed ``svg.hashsalt`` (without a salt,
    matplotlib uses uuid-based element ids, so only "deterministic-ish");
    ``heldout_eval._write_svg`` adds only a ``wrote <path>`` print on top of the
    same file content."""
    from mulligan.real.lifecycle import heldout_eval
    from mulligan.real.lifecycle import plotting

    p1 = tmp_path / "nested" / "a.svg"  # nested dir: pins mkdir(parents=True)
    p2 = tmp_path / "b.svg"
    with plt.rc_context({"svg.hashsalt": "mulligan-test"}):
        plotting.write_svg(_fixed_probe_figure(), p1)
        heldout_eval._write_svg(_fixed_probe_figure(), p2)

    text = p1.read_text()
    assert "<dc:date>" not in text
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert all(line == line.rstrip() for line in text.splitlines())
    # Same bytes from both entry points for the same fixed figure.
    assert p1.read_bytes() == p2.read_bytes()
    # heldout's wrapper adds only the print (absolute path when outside the repo);
    # plotting.write_svg prints nothing.
    assert capsys.readouterr().out == f"wrote {p2}\n"
    # Both close their figures.
    assert plt.get_fignums() == []
    assert plotting.display_path(p1) == str(p1)  # outside repo -> absolute
