"""Append-only prompt-variant model for VLM stage labeling.

A task's labeling prompt is a base prompt plus an ordered list of *rulings*, each a
small text appended to its parent that settles one labeling question (e.g. "held at
holder => fully seated" overcall, end-pinch grasps counted as acquisition). This module
stores that chain as data, so a variant is assembled from its rulings instead of being
copied whole.

A :class:`PromptLibrary` is a DAG of :class:`PromptNode` s. Most nodes are a
``parent`` plus an appended ``text`` delta; a node with ``parent=None`` is a
standalone base (the marker chain is the base ``v4d`` plus ordered rulings up to
``v4p16``). :func:`PromptLibrary.assemble` walks parent links to a root and
concatenates the deltas, reproducing each variant string byte-for-byte
(pinned by ``tests/real/test_stage_specs_golden.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptNode:
    """One prompt variant: a base (``parent=None``) or a delta on a parent.

    ``text`` is the *full* prompt for a base, or the appended suffix for a
    delta. ``rationale`` is a one-line summary of what the ruling settles; it never
    affects the assembled prompt.
    """

    variant: str
    parent: str | None
    text: str
    rationale: str = ""


class PromptLibrary:
    """Ordered DAG of prompt variants with byte-exact assembly."""

    def __init__(self, nodes: list[PromptNode]):
        self._nodes: dict[str, PromptNode] = {}
        for node in nodes:
            if node.variant in self._nodes:
                raise ValueError(f"duplicate prompt variant {node.variant!r}")
            self._nodes[node.variant] = node
        # Validate the DAG eagerly: every parent must exist, no cycles.
        for variant in self._nodes:
            self._chain(variant)

    def _chain(self, variant: str) -> list[PromptNode]:
        """Root-first list of nodes from a root down to ``variant``."""
        seen: set[str] = set()
        chain: list[PromptNode] = []
        cursor: str | None = variant
        while cursor is not None:
            if cursor not in self._nodes:
                raise KeyError(
                    f"prompt variant {cursor!r} references unknown parent "
                    f"(from {variant!r}); registered: {sorted(self._nodes)}"
                )
            if cursor in seen:
                raise ValueError(f"cycle in prompt chain at {cursor!r}")
            seen.add(cursor)
            node = self._nodes[cursor]
            chain.append(node)
            cursor = node.parent
        chain.reverse()
        return chain

    def assemble(self, variant: str) -> str:
        """Reproduce the full system prompt for ``variant``."""
        return "".join(node.text for node in self._chain(variant))

    @property
    def variants(self) -> tuple[str, ...]:
        """Variant ids in registration order."""
        return tuple(self._nodes)

    def rationale(self, variant: str) -> str:
        return self._nodes[variant].rationale
