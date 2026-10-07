"""HiL-SERL in simulation (app:baselines-hilserl): the per-session operator-cost table.

Reads the pinned HiL-SERL session records (``sim/results/hilserl/<task>/``): the
per-milestone eval CSV and the session totals (``status.json``) of Square-Narrow and
Square-Broad. Writes hilserl_sessions.tex (tab:appendix-hilserl-sessions). The eval curves
of the same sessions are drawn by ``plot.py``.

Totals of a forked session count only its post-fork episodes; its eval columns use
milestones after the fork (``forked_from_step``; the fork-step eval belongs to the source run).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from paper.appendix.artifacts import cache_dir, load_inputs, write_json, write_table

HERE = Path(__file__).resolve().parent
CACHE = cache_dir("hilserl")
OUTPUT = "hilserl_sessions.tex"
# (task, session key, operator protocol as named in app:baselines-hilserl), in the
# order the section presents them. Keys are the session keys of ``status.json``.
SESSIONS: tuple[tuple[str, str, str], ...] = (
    ("square_narrow", "s1", "Step 0, dense"),
    ("square_narrow", "auto5k", "Step 0, sparse bursts"),
    ("square_narrow", "curriculum", "Step 0, long bouts"),
    ("square_narrow", "auto5k_fork130k", "Buffer, then none"),
    ("square_narrow", "fork150k", "After takeoff"),
    ("square_broad", "fork200k", "Fork at 200k"),
)
TASK_MACROS = {"square_narrow": r"\sqnarrow", "square_broad": r"\sqbroad"}


def read_eval(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def session_totals(inputs: dict[str, Path], task: str, key: str) -> dict:
    status = json.loads(inputs[f"sim/results/hilserl/{task}/status.json"].read_text())
    return status["sessions"][key]


def eval_series(inputs: dict[str, Path], task: str, key: str) -> dict[int, float]:
    """Milestone -> eval success rate of one session (only milestones it scored)."""
    return {
        int(row["env_step"]): float(row[f"{key}_sr"])
        for row in read_eval(inputs[f"sim/results/hilserl/{task}/eval.csv"])
        if row[f"{key}_sr"]
    }


def rows(inputs: dict[str, Path]) -> list[dict]:
    out = []
    for task, key, protocol in SESSIONS:
        totals = session_totals(inputs, task, key)
        series = eval_series(inputs, task, key)
        start = totals["forked_from_step"] or 0
        own = {step: sr for step, sr in series.items() if step > start}
        end = max(own)
        assert end == totals["latest_eval_step"], (key, end, totals["latest_eval_step"])
        operator = totals["human_frames"] > 0
        out.append(
            dict(
                task=task,
                key=key,
                protocol=protocol,
                start_step=start,
                end_step=end,
                env_steps=totals["env_steps"],
                episodes=totals["episodes"],
                human_frames=totals["human_frames"],
                takeover_episodes=totals["takeover_episodes"],
                operator_hours=totals["operator_hours"] if operator else None,
                human_frame_frac=totals["human_frame_frac"],
                takeover_episode_frac=totals["takeover_episodes"] / totals["episodes"],
                training_success=totals["successful_episodes"] / totals["episodes"],
                policy_only_successes=totals["policy_only_successes"],
                policy_only_episodes=totals["policy_only_episodes"],
                final_eval=own[end],
                peak_eval=max(own.values()),
            )
        )
    return out


def header(*lines: str) -> str:
    """A two-line column header cell."""
    return r"\begin{tabular}[b]{@{}c@{}}" + r" \\ ".join(lines) + r"\end{tabular}"


def total_row(label: str, cells: list[dict]) -> str | None:
    """Summed cost of the sessions with an operator (None if fewer than two); the outcome
    columns do not pool across protocols and stay blank."""
    operated = [c for c in cells if c["operator_hours"] is not None]
    if len(operated) < 2:
        return None
    steps = sum(c["env_steps"] for c in operated)
    episodes = f"{sum(c['episodes'] for c in operated):,}".replace(",", "{,}")
    hours = sum(c["operator_hours"] for c in operated)
    human = sum(c["human_frames"] for c in operated) / steps
    takeover = sum(c["takeover_episodes"] for c in operated) / sum(c["episodes"] for c in operated)
    return (
        f"  {label} & ${round(steps / 1000)}$k & ${episodes}$ & ${hours:.1f}$ & "
        f"${100 * human:.1f}\\%$ & ${100 * takeover:.0f}\\%$ & & & & \\\\"
    )


def table(cells: list[dict]) -> str:
    columns = (
        header("Operator", "protocol"),
        header("Env", "steps"),
        header("Epi-", "sodes"),
        header("Operator", "hours"),
        header("Human", "frames"),
        header("Takeover", "episodes"),
        header("Training", "success"),
        header("Policy-only", "successes"),
        header("Final", "eval"),
        header("Peak", "eval"),
    )
    lines = [
        "% Generated by paper.appendix.hilserl.prepare.",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"  \hline",
        "  " + " & ".join(columns) + r" \\",
    ]
    task_total = r"\textit{Total}"
    previous = None
    for c in cells:
        if c["task"] != previous:
            if previous is not None:
                lines.append(total_row(task_total, [x for x in cells if x["task"] == previous]))
            lines += [r"  \hline", rf"  \multicolumn{{10}}{{l}}{{{TASK_MACROS[c['task']]}}} \\"]
            previous = c["task"]
        hours = "--" if c["operator_hours"] is None else f"${c['operator_hours']:.1f}$"
        episodes = f"{c['episodes']:,}".replace(",", "{,}")
        lines.append(
            f"  {c['protocol']} & "
            f"${c['start_step'] // 1000}$--${c['end_step'] // 1000}$k & "
            f"${episodes}$ & {hours} & "
            f"${100 * c['human_frame_frac']:.1f}\\%$ & "
            f"${100 * c['takeover_episode_frac']:.0f}\\%$ & "
            f"${c['training_success']:.2f}$ & "
            f"${c['policy_only_successes']}/{c['policy_only_episodes']}$ & "
            f"${c['final_eval']:.2f}$ & ${c['peak_eval']:.2f}$ \\\\"
        )
    lines.append(total_row(task_total, [x for x in cells if x["task"] == previous]))
    lines += [r"  \hline", total_row(r"\textbf{Total, both tasks}", cells)]
    lines += [r"  \hline", r"\end{tabular}", ""]
    return "\n".join(line for line in lines if line is not None)


def write_tables(check: bool = False) -> Path:
    cells = rows(load_inputs(HERE))
    write_json(
        CACHE / "data/analysis.json",
        dict(schema="mulligan.paper.appendix.hilserl.v1", sessions=cells),
    )
    return write_table(OUTPUT, table(cells), check=check)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Require the manuscript's table body.")
    args = parser.parse_args()
    print(write_tables(check=args.check))


if __name__ == "__main__":
    main()
