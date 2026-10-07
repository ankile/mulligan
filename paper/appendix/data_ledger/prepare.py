"""Final-round training-set ledger for the five reported tasks (tab:appendix-data-ledger).

Every cell is read from pinned inputs: the round-0 teleoperation statistics and the
per-round split ledgers of Insert Marker and Thread Nut (HG-DAgger+Mulligan with-CF
view), the Route Cable actor provenance and 100-episode increment frame counts, and the
paper rollout accounting. The simulation rows read the frame-level parquet of the public training repos of the
HG-DAgger+Mulligan actor (round-0 demonstrations and the R1--R3 with-CF collections) at their
pinned revisions. Critic sets are the recorded episode and frame totals
(``release/datasets.json``) of each deployed final-round critic's training repos
(``release/models.json``). Writes data_ledger.tex. The few counts that no pinned table records
(the Route Cable R5 parent collection, the simulation round-0 demonstrations per arm, and the
Insert Marker and Route Cable critic sets the release totals are checked against) are
documented constants in ``tracker_values.json``.

It also writes the per-round data composition of both real-world collection campaigns
(tab:appendix-data-composition, data_composition.tex): the valid
frames of each round's training view (HG-DAgger no-CF, Mulligan with-CF), split by
source into demonstrations (round 0 and counterfactual replays), human corrections,
and autonomous frames. The figure of the same rows is drawn by ``plot.py``.

It also produces the robot time of the "Ours" rows of tab:appendix-prior-data
(``robot_time.csv``) and checks those cells of the authored table against it. Robot time is
raw recorded wall time at 15 Hz: every stored frame of each episode, including any tail a
later outcome review soft-truncated (reviews shorten num_steps, never the recording).
Collection counts the HG-DAgger+Mulligan arm's 100 round-0 demonstrations (dataset episode
lengths) and its R1--R5 episodes (``saved_frames``, the full stored span). Evaluation counts
the held-out episodes of the three headline methods (``frames`` in the round-dataset lock,
the recorded episode lengths).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from paper.appendix.artifacts import (
    ROOT,
    REAL_DIRS,
    REFERENCE_TABLES,
    cache_dir,
    load_inputs,
    write_json,
    write_table,
)

HERE = Path(__file__).resolve().parent
CACHE = cache_dir("data_ledger")
OUTPUT = "data_ledger.tex"
COMPOSITION_OUTPUT = "data_composition.tex"
TASKS = ("marker_d2", "square_d2", "routing_d2")
ARMS = {"baseline": "baseline_no_cf", "mulligan": "mulligan_with_cf"}
SOURCES = ("demonstration", "correction", "autonomous")
REAL = ("marker_d2", "square_d2")
ROUNDS = ("R1", "R2", "R3", "R4", "R5")
SIM = ("square_narrow", "square_broad")
FPS = 15
HEADLINE_METHODS = ("HG-DAgger", "HG-DAgger+Mulligan", "HiL-IDQL+Mulligan")
LOCK_TASKS = {
    "real-marker-d2": "marker_d2",
    "real-square-d2": "square_d2",
    "real-routing-d2": "routing_d2",
}
EVAL_EPISODES = {"marker_d2": 850, "square_d2": 950, "routing_d2": 750}
ROUND_LOCK = "real/results/round_datasets.json"
# Recorded lengths of the Route Cable Mulligan arm's 100 round-0 demonstrations.
CABLE_R0_EPISODES = "real/collection/routing/r0_episode_lengths.csv"
PRIOR_DATA = REFERENCE_TABLES / "authored/appendix-prior-data.tex"
# The deployed final-round critic of each task (release/models.json).
CRITICS = {
    "marker_d2": "mulligan/real-marker-d2-r05-mulligan-idql-critic",
    "square_d2": "mulligan/real-square-d2-r05-mulligan-idql-critic",
    "routing_d2": "mulligan/real-routing-d2-r05-mulligan-idql-critic",
    "square_narrow": "mulligan/sim-square-narrow-r03-mulligan-divl",
    "square_broad": "mulligan/sim-square-broad-r03-mulligan-divl",
}


def release_records(name: str) -> dict[str, dict]:
    """``release/{datasets,models}.json`` by repo."""
    return {r["repo"]: r for r in json.loads((ROOT / f"release/{name}.json").read_text())[name]}


def training_datasets(model: str) -> tuple[str, ...]:
    """The one training-dataset list shared by every checkpoint (seed) of ``model``."""
    (datasets,) = {
        tuple(c["training_datasets"]) for c in release_records("models")[model]["checkpoints"]
    }
    return datasets


def critic_set(task: str) -> dict:
    """Recorded episode and frame totals of the deployed critic's training repos."""
    datasets = release_records("datasets")
    repos = training_datasets(CRITICS[task])
    return dict(
        critic_episodes=sum(datasets[r]["episodes"] for r in repos),
        critic_frames=sum(datasets[r]["frames"] for r in repos),
    )


def rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def integer(text: str) -> int:
    return int(text.replace(",", ""))


def documented_count(key: str) -> int:
    """A documented count that no pinned table records (see ``tracker_values.json``)."""
    return json.loads((HERE / "tracker_values.json").read_text())[key]["value"]


def real_line(inputs: dict[str, Path], line: str) -> dict:
    """HG-DAgger+Mulligan with-CF view: round-0 demonstrations plus five 50-episode rounds.

    |D_h| counts only episodes with human frames, the actor's filter, as Route Cable's
    retained count does.
    """
    r0 = {
        r["arm"]: r
        for r in rows(inputs[f"real/collection/{REAL_DIRS[line]}/r0_collected_stats.csv"])
    }["mulligan_sobol"]
    assert r0["total_frames"] == r0["total_human_frames"], "round 0 is pure teleoperation"
    episodes, human, valid = (
        int(r0["n_episodes"]),
        int(r0["total_human_frames"]),
        int(r0["total_frames"]),
    )
    for r in ROUNDS:
        split = [
            s
            for s in rows(
                inputs[f"real/collection/{REAL_DIRS[line]}/{r.lower()}/split_episodes.csv"]
            )
            if s["view"] == "mulligan_with_cf"
        ]
        assert len(split) == 50, (line, r, len(split))
        assert all(s["source_success"] == "True" for s in split), (line, r)
        episodes += sum(int(s["saved_human_frames"]) > 0 for s in split)
        human += sum(int(s["saved_human_frames"]) for s in split)
        valid += sum(int(s["saved_valid_frames"]) for s in split)
    share = [
        s
        for s in rows(inputs[f"real/collection/{REAL_DIRS[line]}/human_frame_round_share.csv"])
        if s["training_view"] == "mulligan_with_cf"
    ]
    assert {int(s["total_human_frames"]) for s in share} == {human}, (line, human)
    accounting = {
        r["source_type"]: int(r[line])
        for r in rows(inputs["real/collection/campaign_episodes_by_source.csv"])
    }
    total = accounting.pop("Total recorded active-line episodes")
    assert total == sum(accounting.values()), (line, total, accounting)
    collected = accounting["R0 teleoperation session"] + accounting["DAgger parent collection"]
    return dict(
        episodes=episodes, human_frames=human, all_frames=valid, collected=collected
    ) | critic_set(line)


def cable(inputs: dict[str, Path]) -> dict:
    """R0 demonstrations plus the five 100-episode increments of the Mulligan arm."""
    provenance = {
        (p["round"], p["arm"]): p
        for p in json.loads(inputs["real/training/routing/runs.json"].read_text())
    }
    r0, r5 = provenance[("R0", "mulligan")], provenance[("R5", "mulligan")]
    assert (
        r0["nominal_episodes"] == r0["retained_episodes"] == 100 and r5["nominal_episodes"] == 600
    )
    increments = [
        r
        for r in rows(inputs["real/collection/routing/increment_frame_counts.csv"])
        if r["view"] == "mulligan_with_cf"
    ]
    assert [r["round"] for r in increments] == list(ROUNDS) and all(
        int(r["episodes"]) == 100 for r in increments
    )
    human = r0["training_frames"] + sum(int(r["human_frames"]) for r in increments)
    valid = r0["training_frames"] + sum(int(r["valid_frames"]) for r in increments)
    accounting = {
        r["source_type"]: int(r["routing_d2"])
        for r in rows(inputs["real/collection/campaign_episodes_by_source.csv"])
    }
    collected = (
        accounting["R0 teleoperation session"]
        + accounting["DAgger parent collection"]
        + documented_count("cable_r5_parent_episodes")
    )
    sizes = critic_set("routing_d2")
    assert sizes["critic_episodes"] == documented_count("cable_critic_episodes"), sizes
    return dict(
        episodes=r5["retained_episodes"],
        human_frames=human,
        all_frames=valid,
        collected=collected,
        **sizes,
    )


def sim_task(task: str, demos: int) -> dict:
    """Round-0 demonstrations plus the three 100-episode with-CF rounds the actor trains on.

    Frame-level ``source``/``is_valid`` give the definitions the real-world split ledgers use;
    "collected" adds the uniform-sampling campaign's episodes of the same rounds.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    from mulligan.release.download import pinned_revision

    prefix = f"mulligan/sim-{task.replace('_', '-')}"
    repos = [f"{prefix}-c00-teleop-sobol"] + [f"{prefix}-c0{k}-dagger-mulligan" for k in (1, 2, 3)]
    assert set(repos) < set(training_datasets(CRITICS[task])), task
    episodes = human = valid = collected = 0
    for i, repo in enumerate(repos):
        revision = pinned_revision(repo)
        files = sorted(
            f
            for f in list_repo_files(repo, repo_type="dataset", revision=revision)
            if f.startswith("data/") and f.endswith(".parquet")
        )
        frames = pd.concat(
            pd.read_parquet(
                hf_hub_download(repo, f, repo_type="dataset", revision=revision),
                columns=["episode_index", "source", "success", "is_valid"],
            )
            for f in files
        )
        per = frames.assign(human=(frames.source == 1) & (frames.is_valid == 1)).groupby(
            "episode_index"
        )
        assert per.success.max().all(), repo
        assert per.ngroups == (demos if i == 0 else 100), (repo, per.ngroups)
        if i == 0:
            assert (frames.source == 1).all(), "round 0 is pure teleoperation"
        collected += per.ngroups
        episodes += int((per.human.sum() > 0).sum())
        human += int(per.human.sum().sum())
        valid += int(frames.is_valid.sum())
    datasets = release_records("datasets")
    uniform = [datasets[f"{prefix}-c00-teleop-baseline"]["episodes"]] + [
        datasets[f"{prefix}-c0{k}-dagger-baseline"]["episodes"] for k in (1, 2, 3)
    ]
    assert uniform == [demos, 100, 100, 100], (task, uniform)
    return dict(
        episodes=episodes, human_frames=human, all_frames=valid, collected=collected + sum(uniform)
    ) | critic_set(task)


def analyze(inputs: dict[str, Path]) -> dict:
    marker = real_line(inputs, "marker_d2")
    nut = real_line(inputs, "square_d2")
    assert (marker["critic_episodes"], marker["critic_frames"]) == (
        documented_count("marker_critic_episodes"),
        documented_count("marker_critic_frames"),
    ), marker
    # The deployed Thread Nut critic adds the July R5 rollouts to the July critic's set.
    deployed = set(training_datasets(CRITICS["square_d2"]))
    july = set(training_datasets("mulligan/real-square-d2-r05-mulligan-idql-critic-screen-b02"))
    assert july < deployed and deployed - july == {
        "mulligan/real-square-d2-r05-eval-b02-mulligan-dp-policy-rollouts",
        "mulligan/real-square-d2-r05-eval-b02-mulligan-idql-policy-rollouts",
    }, deployed - july
    return dict(
        marker=marker,
        nut=nut,
        cable=cable(inputs),
        square_narrow=sim_task("square_narrow", documented_count("square_narrow_r0_demos_per_arm")),
        square_broad=sim_task("square_broad", documented_count("square_broad_r0_demos_per_arm")),
    )


def r0_frames(inputs: dict[str, Path], task: str, arm: str) -> int:
    """Round-0 teleoperated training demonstrations (100 per arm, all human frames)."""
    if task == "routing_d2":
        provenance = {
            (p["round"], p["arm"]): p
            for p in json.loads(inputs["real/training/routing/runs.json"].read_text())
        }
        r0 = provenance[("R0", arm)]
        assert r0["nominal_episodes"] == r0["retained_episodes"] == 100
        return r0["training_frames"]
    source = {"baseline": "baseline_uniform", "mulligan": "mulligan_sobol"}[arm]
    r0 = {
        r["arm"]: r
        for r in rows(inputs[f"real/collection/{REAL_DIRS[task]}/r0_collected_stats.csv"])
    }[source]
    assert r0["total_frames"] == r0["total_human_frames"] and int(r0["n_episodes"]) == 100
    return int(r0["total_human_frames"])


def composition(inputs: dict[str, Path]) -> list[dict]:
    """Valid training frames per task, collection arm, round, and source.

    Demonstrations are the human frames of round-0 demonstrations and counterfactual
    replays, corrections the human frames of policy-start episodes, and autonomous
    frames every policy-controlled frame, including those inside mixed-control replays.
    """
    cells = []
    for task in TASKS:
        quota = 100 if task == "routing_d2" else 50
        for arm, view in ARMS.items():
            cells.append(
                dict(
                    task=task,
                    arm=arm,
                    round="R0",
                    episodes=100,
                    demonstration=r0_frames(inputs, task, arm),
                    correction=0,
                    autonomous=0,
                )
            )
            for r in ROUNDS:
                split = [
                    s
                    for s in rows(
                        inputs[f"real/collection/{REAL_DIRS[task]}/{r.lower()}/split_episodes.csv"]
                    )
                    if s["view"] == view
                ]
                assert len(split) == quota, (task, view, r, len(split))
                for s in split:
                    assert s["is_counterfactual"] in ("True", "False")
                    assert int(s["saved_valid_frames"]) == int(s["saved_policy_frames"]) + int(
                        s["saved_human_frames"]
                    )
                replay = [s for s in split if s["is_counterfactual"] == "True"]
                assert arm == "mulligan" or not replay, (task, view, r)
                cells.append(
                    dict(
                        task=task,
                        arm=arm,
                        round=r,
                        episodes=len(split),
                        demonstration=sum(int(s["saved_human_frames"]) for s in replay),
                        correction=sum(
                            int(s["saved_human_frames"])
                            for s in split
                            if s["is_counterfactual"] == "False"
                        ),
                        autonomous=sum(int(s["saved_policy_frames"]) for s in split),
                    )
                )
    # Route Cable increments reproduce the lineage's own frame counts.
    counts = {
        (c["round"], c["view"]): c
        for c in rows(inputs["real/collection/routing/increment_frame_counts.csv"])
    }
    for cell in cells:
        if cell["task"] != "routing_d2" or cell["round"] == "R0":
            continue
        count = counts[(cell["round"], ARMS[cell["arm"]])]
        assert int(count["human_frames"]) == cell["demonstration"] + cell["correction"], cell
        assert (
            int(count["valid_frames"])
            == cell["demonstration"] + cell["correction"] + cell["autonomous"]
        ), cell
    return cells


def check_composition(cells: list[dict], report: dict) -> None:
    """The Mulligan campaign's totals are the final-round ledger's D_h and D_all."""
    for task, key in zip(TASKS, ("marker", "nut", "cable")):
        ours = [c for c in cells if c["task"] == task and c["arm"] == "mulligan"]
        human = sum(c["demonstration"] + c["correction"] for c in ours)
        assert human == report[key]["human_frames"], (task, human)
        assert human + sum(c["autonomous"] for c in ours) == report[key]["all_frames"], task


def cell(value: int) -> str:
    return number(value) if value else "---"


def composition_table(cells: list[dict]) -> str:
    macro = {"marker_d2": r"\marker", "square_d2": r"\nut", "routing_d2": r"\cable"}
    lines = [
        "% Generated by paper.appendix.data_ledger.prepare.",
        r"\begin{tabular}{llrrrrrr}",
        r"  \hline",
        r"  & & \multicolumn{3}{c}{HG-DAgger} & \multicolumn{3}{c}{HiL-IDQL+\ours{}} \\",
        r"  Task & Round & Demos & Corrections & Autonomous & Demos & Corrections & Autonomous \\",
        r"  \hline",
    ]
    for task in TASKS:
        mine = [c for c in cells if c["task"] == task]
        for i, r in enumerate(("R0",) + ROUNDS + ("Total",)):
            values = []
            for arm in ARMS:
                arm_cells = [
                    c for c in mine if c["arm"] == arm and (r == "Total" or c["round"] == r)
                ]
                values += [cell(sum(c[source] for c in arm_cells)) for source in SOURCES]
            name = macro[task] if i == 0 else ""
            if r == "Total":
                lines.append(r"  \cline{2-8}")
            lines.append(f"  {name} & {r} & " + " & ".join(values) + r" \\")
        lines.append(r"  \hline")
    lines += [r"\end{tabular}", ""]
    return "\n".join(lines)


def number(value: int) -> str:
    return "$" + f"{value:,}".replace(",", "{,}") + "$"


def table(report: dict) -> str:
    m, n, c, narrow, broad = (
        report[k] for k in ("marker", "nut", "cable", "square_narrow", "square_broad")
    )
    lines = [
        "% Generated by paper.appendix.data_ledger.prepare.",
        r"\begin{tabular}{llcccccc}",
        r"  \hline",
        r"  Task & Rounds & $M$ & \shortstack{$|\mathcal{D}_h|$\\episodes} & \shortstack{$\mathcal{D}_h$\\frames} & \shortstack{$\mathcal{D}_\text{all}$\\frames} & \shortstack{Critic set\\(episodes / frames)} & \shortstack{Collected\\episodes} \\",
        r"  \hline",
        *(
            rf"  {name} & {rounds} & ${budget}$ & {number(r['episodes'])} & {number(r['human_frames'])} & {number(r['all_frames'])} & {number(r['critic_episodes'])} / {number(r['critic_frames'])} & {number(r['collected'])} \\"
            for name, rounds, budget, r in (
                (r"\marker", "R0--R5", 50, m),
                (r"\nut", "R0--R5", 50, n),
                (r"\cable", "R0--R5", 100, c),
                (r"\sqnarrow", "R0--R3", 100, narrow),
                (r"\sqbroad", "R0--R3", 100, broad),
            )
        ),
        r"  \hline",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def robot_time(inputs: dict[str, Path]) -> list[dict]:
    """Raw recorded collection and evaluation time per real task (see module docstring)."""
    import pandas as pd

    collection = {}
    for line in REAL:
        r0 = {
            r["arm"]: r
            for r in rows(inputs[f"real/collection/{REAL_DIRS[line]}/r0_collected_stats.csv"])
        }["mulligan_sobol"]
        assert r0["length_source"] == "meta/episodes.length" and int(r0["n_episodes"]) == 100
        frames = int(r0["total_frames"])
        for r in ROUNDS:
            split = [
                s
                for s in rows(
                    inputs[f"real/collection/{REAL_DIRS[line]}/{r.lower()}/split_episodes.csv"]
                )
                if s["view"] == "mulligan_with_cf"
            ]
            assert len(split) == 50, (line, r)
            frames += sum(int(s["saved_frames"]) for s in split)
        collection[line] = (350, frames)
    meta = pd.read_csv(inputs[CABLE_R0_EPISODES])
    assert len(meta) == 100 and meta.episode_index.is_unique
    frames, episodes = int(meta["length"].sum()), 100
    for r in ROUNDS:
        split = [
            s
            for s in rows(inputs[f"real/collection/routing/{r.lower()}/split_episodes.csv"])
            if s["view"] == "mulligan_with_cf"
        ]
        assert len(split) == 100, r
        frames += sum(int(s["saved_frames"]) for s in split)
        episodes += len(split)
    collection["routing_d2"] = (episodes, frames)
    lock = json.loads(inputs[ROUND_LOCK].read_text())
    evaluation = {task: [0, 0] for task in EVAL_EPISODES}
    for dataset in lock["datasets"]:
        if dataset["kind"] not in ("round", "cable"):  # candidate screens are not headline
            continue
        task = LOCK_TASKS[dataset["task"]]
        method = {(p["session_id"], p["name"]): p.get("method") for p in dataset["policies"]}
        for e in dataset["episodes"]:
            if (
                e["role"] == "counted"
                and method[(e["session_id"], e["policy"])] in HEADLINE_METHODS
            ):
                # Stored frames are the eval-time step count plus the initial frame.
                assert e["frames"] >= e["num_steps"] + 1, (dataset["id"], e["source_episode_index"])
                evaluation[task][0] += 1
                evaluation[task][1] += e["frames"]
    out = []
    for task in TASKS:
        assert evaluation[task][0] == EVAL_EPISODES[task], (task, evaluation[task])
        out.append(
            dict(
                task=task,
                collection_episodes=collection[task][0],
                collection_frames=collection[task][1],
                collection_min=collection[task][1] / FPS / 60,
                eval_episodes=evaluation[task][0],
                eval_frames=evaluation[task][1],
                eval_min=evaluation[task][1] / FPS / 60,
            )
        )
    return out


def check_prior_data(times: list[dict]) -> None:
    """The authored prior-data table's 'Ours' robot-time cells must equal the producer."""
    text = PRIOR_DATA.read_text()
    macro = {"marker_d2": r"\marker", "square_d2": r"\nut", "routing_d2": r"\cable"}
    for row in times:
        cells = re.search(
            r"Ours\s*& "
            + re.escape(macro[row["task"]])
            + r"\s*&.*?\$(\d+)\$\\,min \$\+\$ \$(\d+)\$\\,min eval",
            text,
        )
        assert cells, row["task"]
        expected = (round(row["collection_min"]), round(row["eval_min"]))
        assert (int(cells[1]), int(cells[2])) == expected, (row["task"], cells.groups(), expected)


def build(check: bool = False) -> list[Path]:
    inputs = load_inputs(HERE)
    times = robot_time(inputs)
    path = CACHE / "data/robot_time.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(times[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(times)
    check_prior_data(times)
    report = analyze(inputs)
    cells = composition(inputs)
    check_composition(cells, report)
    write_json(
        CACHE / "data/analysis.json",
        dict(schema="mulligan.paper.appendix.data_ledger.v1", **report, composition=cells),
    )
    return [
        write_table(OUTPUT, table(report), check=check),
        write_table(COMPOSITION_OUTPUT, composition_table(cells), check=check),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Require the manuscript's table bodies."
    )
    args = parser.parse_args()
    for path in build(check=args.check):
        print(path)


if __name__ == "__main__":
    main()
