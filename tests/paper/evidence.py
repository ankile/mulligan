"""Decorator for tests that read the pinned paper evidence."""

from __future__ import annotations

import os

import pytest

from paper.appendix.artifacts import EVIDENCE_ENV


def reads_evidence(test):
    """The test reads the pinned paper evidence: from ``$MULLIGAN_PAPER_EVIDENCE`` when it
    points at a local mirror, otherwise from ``mulligan/paper-evidence`` on the Hub, which
    makes it a network test."""
    return test if os.environ.get(EVIDENCE_ENV) else pytest.mark.network(test)
