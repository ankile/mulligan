"""Cascade runner for marker_d2 stage-labeling runs."""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mulligan.real.stage_specs import get_label_task_spec
from mulligan.real.stage_labeling.assets import build_items
from mulligan.real.stage_labeling.cascade import (
    MARKER_D2_NODES,
    apply_marker_d2_node_result,
    few_shot_media_paths,
    marker_d2_route,
    run_node,
)
from mulligan.real.stage_labeling.cascade_pipeline.common import (
    CascadeRunOutputs,
    MarkerD2CascadeConfig,
    configure_gemini,
    consensus_label,
    load_events,
    load_raw,
)
from mulligan.real.stage_labeling.consensus import build_consensus, summarize
from mulligan.real.stage_labeling.eval_sessions import (
    MARKER_D2_R0_HELDOUT_DATASET_REPO_ID,
    MARKER_D2_R1_HELDOUT_DATASET_REPO_ID,
)
from mulligan.real.stage_labeling.labeler import (
    LabelerConfig,
    client_route,
    make_client,
    write_outputs,
)
from mulligan.real.stage_specs.marker_d2 import (
    apply_marker_d2_endpoint_prior,
    apply_marker_d2_failure_mode_prior,
    apply_marker_d2_r0_heldout_stage_prior,
    apply_marker_d2_r1_heldout_review_prior,
    has_marker_d2_r0_heldout_stage_calibration,
    has_marker_d2_r1_heldout_review_calibration,
)
from mulligan.real.stage_specs.sensor_constraints import apply_sensor_constraints


def apply_marker_d2_cascade(config: MarkerD2CascadeConfig) -> CascadeRunOutputs:
    """Apply the marker_d2 cascade to a completed backbone run.

    The output run directory is complete: adjusted raw samples, sample labels,
    consensus labels, joined eval labels, review queue, policy summaries, and
    provenance are all written together.
    """
    spec = dataclasses.replace(
        get_label_task_spec("marker_d2"),
        dataset_repo_id=config.dataset_repo_id,
        events_csv=config.events_csv,
    )
    if config.dataset_repo_id == MARKER_D2_R0_HELDOUT_DATASET_REPO_ID:
        exemplar_repo_id = config.dataset_repo_id
        exemplar_build_dir = config.build_dir
    else:
        if config.exemplar_dataset_repo_id != MARKER_D2_R0_HELDOUT_DATASET_REPO_ID:
            raise ValueError(
                "marker_d2 cascade few-shot media must come from the reviewed R0 calibration "
                f"dataset {MARKER_D2_R0_HELDOUT_DATASET_REPO_ID!r}; got "
                f"{config.exemplar_dataset_repo_id!r}"
            )
        if config.exemplar_build_dir is None:
            raise ValueError(
                "marker_d2 cascade on a non-R0 dataset requires exemplar_build_dir; "
                "episode ids are repository-local and must not resolve in the target build dir"
            )
        exemplar_repo_id = config.exemplar_dataset_repo_id
        exemplar_build_dir = config.exemplar_build_dir
    input_run_dir = config.runs_dir / config.input_run_name
    raw = load_raw(input_run_dir)
    episodes = sorted({int(r["episode_index"]) for r in raw})
    event_rows = load_events(config.events_csv, set(episodes))
    build_episodes = set(episodes)
    if config.dataset_repo_id == MARKER_D2_R0_HELDOUT_DATASET_REPO_ID:
        # R0 target: the few-shot banks read their media from this build dir too.
        build_episodes |= {
            int(entry[0])
            for node in MARKER_D2_NODES.values()
            for entry in node.few_shot
            if len(entry) < 4
        }
    built = {
        item["episode_index"]: item for item in build_items(spec, config.build_dir, build_episodes)
    }
    items = {episode: built[episode] for episode in episodes}

    consensus = build_consensus(raw, spec)
    labels: dict[int, dict[str, Any]] = {}
    refinements: dict[int, dict[str, Any]] = {}
    for row in consensus.to_dict(orient="records"):
        episode = int(row["episode_index"])
        labels[episode] = apply_sensor_constraints(
            spec.sensor_rules,
            items[episode],
            consensus_label(row, spec),
        )

    labeler_config = LabelerConfig(
        run_name=config.output_run_name,
        output_dir=config.runs_dir,
        build_dir=config.build_dir,
        few_shot_build_dir=exemplar_build_dir,
        model=config.model,
        media_resolution=config.media_resolution,
        workers=config.workers,
    )

    missing_media = sorted(
        str(path)
        for node in MARKER_D2_NODES.values()
        for path in few_shot_media_paths(spec, labeler_config, node)
        if not path.exists()
    )
    if missing_media:
        raise FileNotFoundError(
            f"marker_d2 cascade few-shot media missing (build the {exemplar_repo_id} "
            f"exemplar episodes into {exemplar_build_dir} first): {missing_media}"
        )

    use_r0_heldout_calibration = config.dataset_repo_id == MARKER_D2_R0_HELDOUT_DATASET_REPO_ID
    use_r1_heldout_calibration = config.dataset_repo_id == MARKER_D2_R1_HELDOUT_DATASET_REPO_ID
    new_jobs: list[tuple[int, str]] = []

    def _finish_refinement(
        episode: int,
        node_key: str,
        result: dict[str, Any],
    ) -> None:
        override_min_frac = (
            config.table_override_min_frac if node_key == "T" else config.override_min_frac
        )
        refined, overrode = apply_marker_d2_node_result(
            labels[episode],
            node_key,
            result,
            override_min_frac=override_min_frac,
        )
        labels[episode] = refined
        refinements[episode] = {
            "refined_by": node_key,
            "overrode": overrode,
            "override_min_frac": override_min_frac,
            "node_result": result,
        }
        print(f"episode {episode:03d}: node {node_key} {result} overrode={overrode}")

    for episode in episodes:
        label = labels[episode]
        node_key = marker_d2_route(label)
        if use_r0_heldout_calibration and has_marker_d2_r0_heldout_stage_calibration(episode):
            refinements[episode] = {
                "refined_by": node_key,
                "overrode": False,
                "skipped": "r0_heldout_stage_calibration",
            }
            continue
        if use_r1_heldout_calibration and has_marker_d2_r1_heldout_review_calibration(episode):
            refinements[episode] = {
                "refined_by": node_key,
                "overrode": False,
                "skipped": "r1_heldout_review_calibration",
            }
            continue
        if node_key is None:
            refinements[episode] = {"refined_by": None, "overrode": False}
            continue
        new_jobs.append((episode, node_key))

    def _run_new(
        job: tuple[int, str],
    ) -> tuple[int, str, dict[str, Any], dict[str, Any]]:
        episode, node_key = job
        client = make_client()
        result = run_node(
            spec,
            labeler_config,
            client,
            MARKER_D2_NODES[node_key],
            items[episode],
        )
        return episode, node_key, result, client_route(client)

    cascade_gemini_route: dict[str, Any] | None = None
    if new_jobs:
        configure_gemini()
        with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
            futures = {pool.submit(_run_new, job): job for job in new_jobs}
            try:
                for future in as_completed(futures):
                    episode, node_key, result, route = future.result()
                    if cascade_gemini_route is None:
                        cascade_gemini_route = route
                    elif cascade_gemini_route != route:
                        raise RuntimeError(
                            "marker_d2 cascade workers constructed clients with different "
                            f"Gemini routes: {cascade_gemini_route} != {route}"
                        )
                    _finish_refinement(episode, node_key, result)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    endpoint_prior_adjustments: dict[int, dict[str, Any]] = {}
    stage_prior_adjustments: dict[int, dict[str, Any]] = {}
    failure_mode_prior_adjustments: dict[int, dict[str, Any]] = {}
    review_prior_adjustments: dict[int, dict[str, Any]] = {}
    for episode in episodes:
        label, adjustment = apply_marker_d2_endpoint_prior(labels[episode], event_rows[episode])
        labels[episode] = label
        if adjustment is not None:
            endpoint_prior_adjustments[episode] = adjustment
            print(
                f"episode {episode:03d}: endpoint prior outcome="
                f"{adjustment['original_outcome']} S{adjustment['before']['max_stage_v2']}"
                f" -> S{adjustment['after_stage']}"
            )
        label, stage_adjustment = apply_marker_d2_r0_heldout_stage_prior(
            labels[episode],
            use_r0_heldout_calibration=use_r0_heldout_calibration,
        )
        labels[episode] = label
        if stage_adjustment is not None:
            stage_prior_adjustments[episode] = stage_adjustment
            print(
                f"episode {episode:03d}: R0 stage prior "
                f"S{stage_adjustment['before']['max_stage_v2']} -> "
                f"S{stage_adjustment['after']['max_stage_v2']}"
            )
        label, fm_adjustment = apply_marker_d2_failure_mode_prior(
            labels[episode],
            use_r0_heldout_calibration=use_r0_heldout_calibration,
        )
        labels[episode] = label
        if fm_adjustment is not None:
            failure_mode_prior_adjustments[episode] = fm_adjustment
            print(
                f"episode {episode:03d}: failure-mode prior "
                f"{fm_adjustment['before']} -> {fm_adjustment['after']}"
            )
        label, review_adjustment = apply_marker_d2_r1_heldout_review_prior(
            labels[episode],
            use_r1_heldout_calibration=use_r1_heldout_calibration,
        )
        labels[episode] = label
        if review_adjustment is not None:
            review_prior_adjustments[episode] = review_adjustment
            print(
                f"episode {episode:03d}: R1 review prior "
                f"S{review_adjustment['before']['max_stage_v2']} -> "
                f"S{review_adjustment['after']['max_stage_v2']}"
            )

    adjusted = []
    for record in raw:
        episode = int(record["episode_index"])
        out = dict(record)
        out["parsed"] = dict(labels[episode])
        out["cascade_refinement"] = refinements[episode]
        adjusted.append(out)

    sample_labels_csv = write_outputs(
        spec,
        labeler_config,
        adjusted,
        gemini_route=cascade_gemini_route,
    )
    run_dir = sample_labels_csv.parent
    summary = summarize(run_dir, spec, events_csv=config.events_csv)

    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    input_provenance_path = input_run_dir / "provenance.json"
    input_provenance = (
        json.loads(input_provenance_path.read_text()) if input_provenance_path.exists() else {}
    )
    provenance.update(
        {
            "source_run_name": config.input_run_name,
            "source_model": input_provenance.get("model"),
            "source_prompt_variant": input_provenance.get("prompt_variant"),
            "cascade": {
                "nodes": sorted(MARKER_D2_NODES),
                "model": config.model,
                "exemplar_dataset_repo_id": exemplar_repo_id,
                "override_min_frac": config.override_min_frac,
                "table_override_min_frac": config.table_override_min_frac,
                "refinements": refinements,
                "endpoint_prior_adjustments": endpoint_prior_adjustments,
                "stage_prior_adjustments": stage_prior_adjustments,
                "failure_mode_prior_adjustments": failure_mode_prior_adjustments,
                "review_prior_adjustments": review_prior_adjustments,
                "r0_heldout_stage_calibration": use_r0_heldout_calibration,
                "r0_heldout_failure_mode_calibration": use_r0_heldout_calibration,
                "r1_heldout_review_calibration": use_r1_heldout_calibration,
            },
            "consensus_summary": summary,
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    return CascadeRunOutputs(
        run_dir=run_dir,
        sample_labels_csv=sample_labels_csv,
        labels_joined_csv=run_dir / "labels_joined.csv",
        provenance_json=provenance_path,
        summary_json=run_dir / "gemini_label_summary.json",
    )
