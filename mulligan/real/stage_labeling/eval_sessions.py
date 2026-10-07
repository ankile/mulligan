"""Released eval datasets that the stage-label cascades treat specially.

The marker_d2 cascade applies human-reviewed calibrations only to these exact
held-out eval datasets (see :mod:`mulligan.real.stage_specs.marker_d2`), and the
square_d2 cascade renders its few-shot anchors from the Nut R1 eval.
"""

from __future__ import annotations

MARKER_D2_R0_HELDOUT_DATASET_REPO_ID = "mulligan/real-marker-d2-r00-eval"
MARKER_D2_R1_HELDOUT_DATASET_REPO_ID = "mulligan/real-marker-d2-r01-eval"
