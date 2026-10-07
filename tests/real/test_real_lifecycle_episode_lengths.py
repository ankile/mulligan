"""Unit tests for the shared episode-length distribution module.

Exercises the round-invariant mechanics on synthetic data (no HF / LeRobot):
length-until-first-terminal (done vs is_valid precedence), per-arm category
assignment + zero-episode loud failure, demo-median computation (with the
DAgger-exclusion invariant on the task registry), and N-arm handling.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulligan.real.lifecycle.episode_lengths import (
    ArmSpec,
    DemoMedianSpec,
    EpisodeLengthConfig,
    build_episode_table,
    demo_median,
    recompute_lengths,
    resolve_demo_medians,
)
from mulligan.real.lifecycle.tasks import get_task_spec, registered_task_specs


class _FakeHF:
    """Minimal stand-in for a LeRobot ``hf_dataset`` (column access + names)."""

    def __init__(self, columns: dict[str, list]):
        self._columns = columns

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __getitem__(self, key: str) -> list:
        return self._columns[key]


def _make_hf(episodes: dict[int, tuple[list[bool], list[bool]]]) -> _FakeHF:
    """Build a fake dataset from ``{episode_index: (done_flags, valid_flags)}``.

    Each episode contributes contiguous 0-based frame indices for the given flags.
    """
    ep_idx: list[int] = []
    frame_idx: list[int] = []
    done: list[bool] = []
    is_valid: list[bool] = []
    for ep, (dones, valids) in episodes.items():
        assert len(dones) == len(valids)
        for f, (d, v) in enumerate(zip(dones, valids, strict=True)):
            ep_idx.append(ep)
            frame_idx.append(f)
            done.append(d)
            is_valid.append(v)
    return _FakeHF(
        {"episode_index": ep_idx, "frame_index": frame_idx, "done": done, "is_valid": is_valid}
    )


# ---------------------------------------------------------------------------
# recompute_lengths: first done is included; first invalid frame is excluded
# ---------------------------------------------------------------------------


def test_recompute_length_done_terminal():
    # done at frame index 4 -> length 5, even though later frames exist.
    hf = _make_hf({0: ([False, False, False, False, True, True], [True] * 6)})
    assert recompute_lengths(hf) == {0: 5}


def test_recompute_length_is_valid_terminal():
    # is_valid goes False at frame index 2 -> two valid frames (no done ever set).
    hf = _make_hf({0: ([False] * 5, [True, True, False, False, False])})
    assert recompute_lengths(hf) == {0: 2}


def test_recompute_length_earliest_terminal_wins():
    # done at frame 3, is_valid=False at frame 1 -> only frame 0 is valid.
    hf = _make_hf({0: ([False, False, False, True], [True, False, False, False])})
    assert recompute_lengths(hf) == {0: 1}


def test_recompute_length_multi_episode():
    hf = _make_hf(
        {
            0: ([False, True], [True, True]),  # length 2
            1: ([False, False, False, True], [True, True, True, True]),  # length 4
        }
    )
    assert recompute_lengths(hf) == {0: 2, 1: 4}


def test_recompute_length_no_terminal_raises():
    hf = _make_hf({0: ([False, False, False], [True, True, True])})
    with pytest.raises(ValueError, match="no done=True or is_valid=False"):
        recompute_lengths(hf)


def test_recompute_length_missing_columns_raises():
    hf = _FakeHF({"episode_index": [0], "frame_index": [0]})
    with pytest.raises(ValueError, match="missing terminal metadata columns"):
        recompute_lengths(hf)


def test_recompute_length_rejects_non_prefix_validity() -> None:
    hf = _make_hf({0: ([False] * 3, [True, False, True])})
    with pytest.raises(RuntimeError, match="not a valid prefix"):
        recompute_lengths(hf)


def test_recompute_length_rejects_done_returning_to_zero() -> None:
    hf = _make_hf({0: ([False, True, False], [True, True, True])})
    with pytest.raises(RuntimeError, match="done returns to zero"):
        recompute_lengths(hf)


# ---------------------------------------------------------------------------
# build_episode_table: category assignment, N-arm handling, zero-episode failure
# ---------------------------------------------------------------------------


def _cfg(arms: tuple[ArmSpec, ...]) -> EpisodeLengthConfig:
    return EpisodeLengthConfig(
        task="routing_d2",
        eval_repo="x",
        outcomes_csv=Path("/nonexistent/paired_round_outcomes.csv"),
        out_csv=Path("/tmp/out.csv"),
        out_svg=Path("/tmp/out.svg"),
        arms=arms,
        plot_title="t",
    )


def _paired_df(arm_prefixes: list[str], n: int) -> tuple[pd.DataFrame, dict[int, int]]:
    """Two-arm-friendly synthetic paired-outcomes frame + recomputed lengths.

    Episode indices are laid out arm-major so each arm gets its own contiguous block.
    """
    rows: dict[str, list] = {"manifest_idx": list(range(n))}
    lengths: dict[int, int] = {}
    ep = 0
    outcomes_cycle = ["success", "timeout", "failure"]
    for arm in arm_prefixes:
        eps, outs, steps = [], [], []
        for i in range(n):
            eps.append(ep)
            outs.append(outcomes_cycle[i % len(outcomes_cycle)])
            steps.append(100 + ep)
            lengths[ep] = 100 + ep + 1  # recomputed is one longer than csv_num_steps
            ep += 1
        rows[f"{arm}_episode_index"] = eps
        rows[f"{arm}_outcome"] = outs
        rows[f"{arm}_num_steps"] = steps
    return pd.DataFrame(rows), lengths


def test_build_table_category_and_length_assignment():
    df, lengths = _paired_df(["baseline", "mulligan"], 3)
    cfg = _cfg((ArmSpec("baseline", "B", "uniform"), ArmSpec("mulligan", "O", "sobol")))
    tbl = build_episode_table(cfg, lengths, df)
    assert len(tbl) == 6  # 2 arms x 3 starts
    base = tbl[tbl["arm"] == "baseline"].reset_index(drop=True)
    assert list(base["outcome"]) == ["success", "timeout", "failure"]
    # recomputed_length = csv_num_steps + 1 for every row -> discrepancy column is all 1
    assert (tbl["length_minus_numsteps"] == 1).all()
    assert list(base["recomputed_length"]) == list(base["csv_num_steps"] + 1)


def test_build_table_n_arm_handling():
    for n_arms in (1, 4):
        prefixes = [f"arm{i}" for i in range(n_arms)]
        df, lengths = _paired_df(prefixes, 5)
        cfg = _cfg(tuple(ArmSpec(p, p.upper(), "uniform") for p in prefixes))
        tbl = build_episode_table(cfg, lengths, df)
        assert len(tbl) == n_arms * 5
        assert set(tbl["arm"]) == set(prefixes)
        for p in prefixes:
            assert int((tbl["arm"] == p).sum()) == 5


def test_build_table_missing_episode_length_raises():
    df, lengths = _paired_df(["baseline"], 3)
    del lengths[1]  # drop one episode's recomputed length
    cfg = _cfg((ArmSpec("baseline", "B", "uniform"),))
    with pytest.raises(ValueError, match="not in recomputed lengths"):
        build_episode_table(cfg, lengths, df)


def test_build_table_cross_repo_resolves_each_arm_in_its_own_namespace():
    arms = (ArmSpec("baseline", "B", "uniform"), ArmSpec("iql", "I", "sobol"))
    cfg = dataclasses.replace(
        _cfg(arms),
        eval_repo_by_arm={"baseline": "org/original", "iql": "org/candidate"},
    )
    df = pd.DataFrame(
        {
            "manifest_idx": [0],
            "baseline_episode_index": [0],
            "baseline_outcome": ["success"],
            "baseline_num_steps": [100],
            "iql_episode_index": [0],
            "iql_outcome": ["success"],
            "iql_num_steps": [200],
        }
    )
    by_arm = {"baseline": {0: 101}, "iql": {0: 201}}
    tbl = build_episode_table(cfg, {0: 999}, df, lengths_by_arm=by_arm)
    assert list(tbl["recomputed_length"]) == [101, 201]
    assert list(tbl["source_repo"]) == ["org/original", "org/candidate"]


def test_episode_length_config_cross_repo_keys_must_match_arms():
    with pytest.raises(ValueError, match="exactly match arm prefixes"):
        dataclasses.replace(
            _cfg((ArmSpec("baseline", "B", "uniform"),)),
            eval_repo_by_arm={"wrong": "org/repo"},
        )


def test_build_table_zero_episode_arm_raises():
    df, lengths = _paired_df(["baseline"], 3)
    cfg = _cfg((ArmSpec("baseline", "B", "uniform"),))
    empty = df.iloc[0:0]  # no starts at all -> the arm has zero episodes
    with pytest.raises(ValueError, match="zero episodes"):
        build_episode_table(cfg, lengths, empty)


# ---------------------------------------------------------------------------
# demo_median: median across repos, DAgger episodes excluded structurally
# ---------------------------------------------------------------------------


def test_demo_median_over_injected_lengths():
    fake = {
        "repoA": {0: 10, 1: 20, 2: 30},  # median 20
        "repoB": {0: 40, 1: 60},  # median 50
    }
    median, n, per_repo = demo_median(["repoA", "repoB"], load_lengths=lambda r: fake[r])
    # pooled lengths [10,20,30,40,60] -> median 30
    assert median == 30.0
    assert n == 5
    assert per_repo == [("repoA", 3, 20), ("repoB", 2, 50)]


def test_demo_median_excludes_dagger_by_repo_selection():
    # The DAgger-exclusion is enforced by which repos the spec lists: only *-r0
    # teleop splits. demo_median computes over exactly the repos it is handed, so a
    # caller that (wrongly) passed an r1 repo would change the median -- proving the
    # exclusion must live in the repo list (the RealTaskSpec field), which the next
    # test pins.
    r0_only = {"routing-r0": {0: 100, 1: 200}}  # median 150
    with_dagger = {**r0_only, "routing-r1": {0: 400, 1: 500}}  # would shift median up
    med_r0, _, _ = demo_median(["routing-r0"], load_lengths=lambda r: r0_only[r])
    med_all, _, _ = demo_median(["routing-r0", "routing-r1"], load_lengths=lambda r: with_dagger[r])
    assert med_r0 == 150.0
    assert med_all > med_r0  # including DAgger corrections biases the demo median up


def test_registered_specs_teleop_repos_are_r0_only():
    """Every registered task's teleop-demo repos are R0 teleop splits only.

    This is the load-bearing DAgger-exclusion invariant: R1+ DAgger corrections
    must never seed the demo-length median. Released R0 teleop splits are named
    ``mulligan/real-<task>-c00-teleop-<set>``.
    """
    any_set = False
    for spec in registered_task_specs():
        if spec.r0_teleop_demo_repos is None:
            continue
        any_set = True
        for demo_set, repos in spec.r0_teleop_demo_repos.items():
            assert repos, f"{spec.name} demo set {demo_set} is empty"
            for repo in repos:
                assert repo.startswith("mulligan/") and "-c00-teleop-" in repo, (
                    f"{spec.name} demo set {demo_set} repo {repo} is not an R0 teleop split"
                )
    assert any_set, "expected at least one registered task to carry r0_teleop_demo_repos"


# ---------------------------------------------------------------------------
# resolve_demo_medians: task-spec resolution + loud skip when unverified
# ---------------------------------------------------------------------------


def test_resolve_demo_medians_from_spec():
    spec = get_task_spec("routing_d2")
    cfg = EpisodeLengthConfig(
        task="routing_d2",
        eval_repo="x",
        outcomes_csv=Path("/x/paired_round_outcomes.csv"),
        out_csv=Path("/tmp/o.csv"),
        out_svg=Path("/tmp/o.svg"),
        arms=(ArmSpec("baseline", "B", "uniform"), ArmSpec("mulligan", "O", "sobol")),
        plot_title="t",
        demo_median_sets=(
            DemoMedianSpec("uniform", "uniform", "baseline demo", "baseline", "baseline set"),
            DemoMedianSpec("sobol", "sobol", "ours demo", "mulligan", "ours set"),
        ),
    )
    # Inject a loader keyed on the actual routing repos so no HF fetch happens.
    fake = {
        "mulligan/real-routing-d2-c00-teleop-baseline": {i: 250 + i for i in range(4)},
        "mulligan/real-routing-d2-c00-teleop-sobol": {i: 260 + i for i in range(4)},
    }
    resolved = resolve_demo_medians(cfg, spec, load_lengths=lambda r: fake[r])
    assert [d.demo_set_key for d, *_ in resolved] == ["uniform", "sobol"]
    assert resolved[0][1] == float(np.median([250, 251, 252, 253]))
    assert resolved[1][1] == float(np.median([260, 261, 262, 263]))


def test_resolve_demo_medians_skips_loud_when_unverified(capsys):
    spec = dataclasses.replace(get_task_spec("square_d2"), r0_teleop_demo_repos=None)
    cfg = EpisodeLengthConfig(
        task="square_d2",
        eval_repo="x",
        outcomes_csv=Path("/x/paired_round_outcomes.csv"),
        out_csv=Path("/tmp/o.csv"),
        out_svg=Path("/tmp/o.svg"),
        arms=(ArmSpec("baseline", "B", "uniform"), ArmSpec("mulligan", "O", "sobol")),
        plot_title="t",
        demo_median_sets=(
            DemoMedianSpec("uniform", "uniform", "baseline demo", "baseline", "baseline set"),
        ),
    )
    resolved = resolve_demo_medians(cfg, spec, load_lengths=lambda r: {0: 1})
    assert resolved == []
    out = capsys.readouterr().out
    assert "SKIPPING demo-median" in out
