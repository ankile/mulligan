"""Shared mechanics for task-specific stage-label cascade runners."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mulligan.real.stage_specs.tasks import StageLabelTaskSpec

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STAGE_LABELING_RUNS_DIR = REPO_ROOT / "outputs/real/stage_labeling_runs"
DEFAULT_STAGE_LABELING_BUILD_DIR = Path("/tmp/mulligan_stage_labeling_assets")

DEVELOPER_API_ENV = {
    "MULLIGAN_GEMINI_ROUTE": "developer",
    "GOOGLE_GENAI_USE_VERTEXAI": "false",
}


@dataclass(frozen=True)
class MarkerD2CascadeConfig:
    """Configuration for applying the marker_d2 focused cascade to a backbone run."""

    input_run_name: str
    output_run_name: str
    dataset_repo_id: str
    events_csv: Path
    runs_dir: Path = DEFAULT_STAGE_LABELING_RUNS_DIR
    build_dir: Path = DEFAULT_STAGE_LABELING_BUILD_DIR
    exemplar_dataset_repo_id: str | None = None
    exemplar_build_dir: Path | None = None
    model: str = "gemini-3.5-flash"
    workers: int = 2
    media_resolution: str = "high"
    override_min_frac: float = 0.8
    table_override_min_frac: float = 0.6


@dataclass(frozen=True)
class SquareD2CascadeConfig:
    """Configuration for applying the square_d2 cascade to a backbone run."""

    input_run_name: str
    output_run_name: str
    dataset_repo_id: str
    events_csv: Path
    runs_dir: Path = DEFAULT_STAGE_LABELING_RUNS_DIR
    build_dir: Path = DEFAULT_STAGE_LABELING_BUILD_DIR
    model: str = "gemini-3.5-flash"
    workers: int = 2
    media_resolution: str = "high"
    included_episodes: frozenset[int] = frozenset()
    transport_override_min_frac: float = 0.6
    transport_s4_override_min_frac: float = 1.0
    carry_endpoint_override_min_frac: float = 0.8
    peg_arrival_override_min_frac: float = 0.8


@dataclass(frozen=True)
class CascadeRunOutputs:
    """Paths written by a cascade pipeline run."""

    run_dir: Path
    sample_labels_csv: Path
    labels_joined_csv: Path
    provenance_json: Path
    summary_json: Path


def configure_gemini_developer_api() -> None:
    """Select the API-key Gemini Developer API route and fail if the key is absent."""
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise RuntimeError(
            "Gemini Developer API requires GEMINI_API_KEY; source the private credential "
            "file before launching stage labeling"
        )
    os.environ.update(DEVELOPER_API_ENV)
    print("[gemini] route=developer api_version=v1beta api_key=set")


def configure_gemini() -> str:
    """Configure the selected production route; Developer API is the default."""
    route = os.environ.get("MULLIGAN_GEMINI_ROUTE", "developer").strip().lower()
    if route == "developer":
        configure_gemini_developer_api()
        return route
    if route == "vertex":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        if not project or not location:
            raise RuntimeError(
                "Vertex route requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION"
            )
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        print(f"[gemini] route=vertex api_version=v1 project={project} location={location}")
        return route
    raise ValueError(f"invalid MULLIGAN_GEMINI_ROUTE={route!r}; expected 'developer' or 'vertex'")


def bool_value(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"cannot parse boolean field {field!r}: {value!r}")
    return bool(value)


def consensus_label(row: dict[str, Any], spec: StageLabelTaskSpec) -> dict[str, Any]:
    label = {
        "episode_index": int(row["episode_index"]),
        spec.stage_field: int(row[spec.stage_field]),
        spec.final_state_field: str(row[spec.final_state_field]),
        spec.failure_mode_field: str(row[spec.failure_mode_field]),
        "confidence": str(row["gemini_confidence"]),
        "needs_human_review": bool_value(
            row["model_requested_review"], field="model_requested_review"
        ),
        "notes": "" if row.get("notes") is None else str(row.get("notes", "")),
    }
    for field in spec.bool_fields:
        label[field] = bool_value(row[field], field=field)
    for field in spec.time_fields:
        value = row.get(field)
        label[field] = None if value is None or str(value) == "nan" else float(value)
    return label


def load_raw(input_run_dir: Path) -> list[dict[str, Any]]:
    path = input_run_dir / "raw_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    raw = json.loads(path.read_text())
    if not raw:
        raise ValueError(f"{path} is empty")
    return raw


def load_events(events_csv: Path, episodes: set[int]) -> dict[int, dict[str, Any]]:
    if not events_csv.exists():
        raise FileNotFoundError(f"{events_csv} does not exist")
    rows: dict[int, dict[str, Any]] = {}
    with events_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            episode = int(row["episode_index"])
            if episode not in episodes:
                continue
            if episode in rows:
                raise ValueError(f"duplicate event row for episode {episode} in {events_csv}")
            rows[episode] = row
    missing = sorted(episodes - set(rows))
    if missing:
        raise ValueError(f"{events_csv} missing event rows for episodes {missing}")
    return rows


def raw_samples_per_episode(raw: list[dict[str, Any]]) -> int:
    counts = Counter(int(r["episode_index"]) for r in raw)
    if not counts:
        raise ValueError("raw results are empty")
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError(f"raw results have uneven samples per episode: {dict(counts)}")
    return unique_counts.pop()
