"""Rebuild the real-world result tables from the hash-locked paper evidence.

``config.json`` is the frozen line-headline configuration of the paper (the suite,
the July stage study, and the collection campaigns), encoded with every path relative
to the ``real_results`` paper evidence. Nothing else is read.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from paper import real_headline as lh
from paper.appendix.artifacts import REAL_DIRS, cache_dir, load_inputs

HERE = Path(__file__).resolve().parent
CACHE = cache_dir("real_results")
TASKS = ("marker_d2", "square_d2", "routing_d2")
INCREMENTS = tuple(f"R{i}" for i in range(1, 6))


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"path"}:
            return CACHE / "raw" / value["path"]
        if set(value) == {"tuple"}:
            return tuple(decode(v) for v in value["tuple"])
        if set(value) == {"type", "fields"}:
            return getattr(lh, value["type"])(**{k: decode(v) for k, v in value["fields"].items()})
        return {k: decode(v) for k, v in value.items()}
    return value


def configs():
    load_inputs(HERE)
    obj = json.loads((HERE / "config.json").read_text())
    return {name: decode(obj[name]) for name in ("suite", "stage", "collection")}


def paper_suite():
    """The paper's three-task headline suite (Marker, Nut, Cable) on the frozen tables."""
    return configs()["suite"]


def headline(task_key: str) -> Path:
    """The frozen headline tables of one task."""
    return CACHE / "raw/real/results" / REAL_DIRS[task_key] / "headline"


def output(cfg):
    path = CACHE / "data" / cfg.task_key
    path.mkdir(parents=True, exist_ok=True)
    return replace(cfg, out_data_dir=path)


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g")


def prepare():
    cfgs = configs()
    suite = tuple(replace(s, cfg=output(s.cfg)) for s in cfgs["suite"])
    for line in suite:
        cfg = line.cfg
        table = lh.build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)
        name = "headline_sr_full_success" if cfg.task_key == "routing_d2" else "headline_sr"
        locked = pd.read_csv(headline(cfg.task_key) / f"{cfg.task_key}_{name}.csv")
        cols = ["round", "arm", "successes", "n"]
        # The locked tables include per-block provenance rows; compare each
        # unique pooled point with the regenerated counts.
        for frame in (table, locked):
            assert frame[frame.is_pooled].duplicated(["round", "arm"]).sum() == 0
        pd.testing.assert_frame_equal(
            table[table.is_pooled][cols].reset_index(drop=True),
            locked[locked.is_pooled][cols].reset_index(drop=True),
            check_dtype=False,
        )
        save(table, lh._data_path(cfg, "headline_sr"))
        save(lh.build_efficiency_table(cfg), lh._data_path(cfg, "headline_efficiency"))
        campaign = lh.build_paired_campaign_table(
            cfg, baseline_arm="baseline", treatment_arm="mulligan_with_cf"
        )
        if cfg.task_key == "routing_d2":
            # Cable's six rounds score their arms on the same 50 starts; the pooled
            # row is the same contrast and test as Marker/Nut (no locked campaign table).
            save(campaign, CACHE / "data" / f"{cfg.task_key}_campaign.csv")
        else:
            expected = pd.read_csv(headline(cfg.task_key) / f"{cfg.task_key}_campaign.csv")
            cols = ["n", "baseline_successes", "treatment_successes"]
            pd.testing.assert_frame_equal(campaign[cols], expected[cols], check_dtype=False)
            save(campaign, CACHE / "data" / f"{cfg.task_key}_campaign.csv")
            critic_arm = "final_marker_iql" if cfg.task_key == "marker_d2" else "final_iql"
            full_rounds = tuple(
                replace(
                    r,
                    policy_arms={
                        **r.policy_arms,
                        critic_arm: r.policy_arms[critic_arm]
                        if critic_arm in r.policy_arms
                        else r.policy_arms["mulligan_with_cf"],
                    },
                )
                for r in cfg.rounds
            )
            full = lh.build_paired_campaign_table(
                replace(cfg, rounds=full_rounds), baseline_arm="baseline", treatment_arm=critic_arm
            )
            save(full, CACHE / "data" / f"{cfg.task_key}_campaign_full_system.csv")
    for task, source_cfg in cfgs["stage"].items():
        cfg = output(source_cfg)
        table = lh.build_substage_table(cfg, arms=cfg.headline_arms, headline_labels=True)
        assert set(table["round"]) == {f"R{i}" for i in range(6)}
        save(table, lh._data_path(cfg, "substage_completion"))
        # Regenerate CF-arm counts from the per-policy records,
        # keeping only rounds where the no-CF arm was actually tested.
        all_arms = lh.build_success_rate_table(cfg, arms=cfg.arm_order, headline_labels=False)
        save(all_arms, lh._data_path(cfg, "headline_sr_all_arms"))
    for task, cfg in cfgs["collection"].items():
        if task == "routing_d2":
            table = pd.read_csv(CACHE / "raw/real/collection/routing/dagger_collection_success.csv")
            assert list(table["round"].unique()) == list(INCREMENTS)
        else:
            table = lh.build_dagger_collection_success_table(cfg)
        save(table, CACHE / "data" / task / "collection.csv")
    burden = []
    for task, cfg in cfgs["collection"].items():
        for round_, directory in cfg.collection_dirs.items():
            split = pd.read_csv(directory / "split_episodes.csv")
            summary = json.loads((directory / "summary.json").read_text())["human_share_burden"]
            for view in ("baseline_no_cf", "mulligan_with_cf"):
                fresh = split[(split.view == view) & ~split.is_counterfactual]
                frames = int(fresh.saved_valid_frames.sum())
                human = int(fresh.saved_human_frames.sum())
                sr = summary[view]["collector_success_rate"]
                raw = 100 * human / frames
                adjusted = raw / (1 - sr)
                np.testing.assert_allclose(
                    [raw, adjusted],
                    [
                        summary[view]["fresh_human_frame_share_pct"],
                        summary[view]["sr_adjusted_fresh_human_frame_share"],
                    ],
                    atol=1e-12,
                )
                burden.append(
                    dict(
                        task=task,
                        round=round_,
                        view=view,
                        human_frames=human,
                        valid_frames=frames,
                        collector_success_rate=sr,
                        human_frame_share_pct=raw,
                        adjusted_share_pct=adjusted,
                        collector_sr_provenance=summary[view]["collector_sr_provenance"],
                    )
                )
    save(pd.DataFrame(burden), CACHE / "data/collection_burden.csv")
    from paper.appendix.real_results.statistics import build_tables

    build_tables(suite)
    return suite, {task: output(cfg) for task, cfg in cfgs["stage"].items()}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
    print(CACHE / "data")


if __name__ == "__main__":
    main()
