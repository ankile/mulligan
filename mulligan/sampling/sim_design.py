"""Build a simulation round's collection starts from a ``configs/sim/<task>/rNN_sampler.yaml``.

A round config lists the collection arms and how each arm's starts are made:

- ``uniform``: i.i.d. uniform starts from a seed (the uniform baseline arm);
- ``file``: starts read from a recorded file (baseline arms that reused the
  starts of the round's baseline policy rollouts);
- ``promote_then_sobol``: the diagnostic failures verbatim, then the next block
  of the task's Sobol stream (the Sobol ablation arm);
- ``select_initial_states``: :func:`mulligan.sampling.select_initial_states.select_initial_states`
  (the Mulligan arm).

The arms are merged into one blind manifest (starts closer than
``match_tolerance`` collapse into one row listing every arm), shuffled with a
fixed seed and checked disjoint from earlier rounds and the evaluation grid. With
``--check`` the result is compared with the locked files named in the config,
which stay authoritative.

Usage::

    python -m mulligan.sampling.sim_design \\
        --config configs/sim/square_narrow/r01_sampler.yaml --out outputs/narrow_r01 --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats.qmc import Sobol

from mulligan.sampling import hardness as H
from mulligan.sampling.select_initial_states import (
    PROMOTED,
    HardnessTerm,
    LocalPerturbation,
    StartSpace,
    check_disjoint,
    coverage_guardrail,
    min_pairwise_distance,
    nearest_state_distance,
    promote_failures,
    sample_uniform,
    select_initial_states,
)
from mulligan.sampling.sobol import SquareD1SobolSampler

REPO_ROOT = Path(__file__).resolve().parents[2]

# Nut-peg clearance of Square-Broad: nut + peg horizontal radius (0.110 + 0.02263).
BROAD_MIN_CLEARANCE = 0.13263

# Square-Narrow Sobol stream: a 2-D scrambled Sobol pool over (nut_y, nut_yaw)
# with nut_x drawn independently uniform from its own seed (App. F.3).
NARROW_SOBOL_SEED = 42
NARROW_SOBOL_X_SEED = 202605234
NARROW_SOBOL_POOL_SIZE = 1024
BROAD_SOBOL_SEED = 42
BROAD_SOBOL_BATCH_SIZE = 2048


def _broad_valid(arr: np.ndarray) -> np.ndarray:
    return np.hypot(arr[:, 0] - arr[:, 3], arr[:, 1] - arr[:, 4]) > BROAD_MIN_CLEARANCE


SQUARE_NARROW = StartSpace(
    keys=("nut_x", "nut_y", "nut_yaw"),
    low=(-0.115, 0.110, -math.pi),
    high=(-0.110, 0.225, math.pi),
    periodic=("nut_yaw",),
)
SQUARE_BROAD = StartSpace(
    keys=("nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"),
    low=(-0.115, -0.255, -math.pi, -0.1, -0.2),
    high=(0.115, 0.255, math.pi, 0.3, 0.2),
    periodic=("nut_yaw",),
    valid=_broad_valid,
)
TASKS = {"square_narrow": SQUARE_NARROW, "square_broad": SQUARE_BROAD}


def sobol_block(task: str, start: int, n: int) -> np.ndarray:
    """States ``start .. start + n - 1`` of the task's collection Sobol stream."""
    if task == "square_narrow":
        if start + n > NARROW_SOBOL_POOL_SIZE:
            raise ValueError(f"Sobol block [{start}, {start + n}) exceeds the locked pool")
        raw = Sobol(d=2, scramble=True, seed=NARROW_SOBOL_SEED).random_base2(
            m=int(np.log2(NARROW_SOBOL_POOL_SIZE))
        )
        xs = np.random.default_rng(NARROW_SOBOL_X_SEED).uniform(
            SQUARE_NARROW.low[0], SQUARE_NARROW.high[0], size=NARROW_SOBOL_POOL_SIZE
        )
        block = raw[start : start + n]
        ys = SQUARE_NARROW.low[1] + block[:, 0] * (SQUARE_NARROW.high[1] - SQUARE_NARROW.low[1])
        yaws = SQUARE_NARROW.low[2] + block[:, 1] * (SQUARE_NARROW.high[2] - SQUARE_NARROW.low[2])
        return np.column_stack([xs[start : start + n], ys, yaws])
    if task == "square_broad":
        sampler = SquareD1SobolSampler(seed=BROAD_SOBOL_SEED, batch_size=BROAD_SOBOL_BATCH_SIZE)
        if len(sampler.planned_points) < start + n:
            raise ValueError(
                f"Sobol stream has {len(sampler.planned_points)} valid points, need {start + n}"
            )
        return np.asarray(
            [[*nut, *peg] for nut, peg in sampler.planned_points[start : start + n]],
            dtype=np.float64,
        )
    raise ValueError(f"unknown task {task!r}")


# ------------------------------------------------------------------ inputs


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def states_sha256(space: StartSpace, states: np.ndarray) -> str:
    """Hash of the state coordinates only (``repr`` of each float, in key order)."""
    text = "\n".join(",".join(repr(float(v)) for v in row) for row in states)
    return hashlib.sha256(f"{','.join(space.keys)}\n{text}".encode()).hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing input {path}")
    return json.loads(path.read_text())


def load_states(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    for key in ("states", "points"):
        if key in payload:
            return payload[key]
    raise ValueError(f"{path}: expected top-level states or points")


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = _read_json(path)["records"]
    if limit is not None:
        if len(records) < limit:
            raise ValueError(f"{path}: {len(records)} records, need {limit}")
        records = records[:limit]
    return records


@dataclass
class RoundInputs:
    """Resolves the repo-relative paths of a round config."""

    root: Path
    task: str
    space: StartSpace
    _eval_grid: np.ndarray | None = field(default=None, repr=False)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def states(self, rel: str, limit: int | None = None) -> np.ndarray:
        states = load_states(self.path(rel))
        if limit is not None:
            states = states[:limit]
        return self.space.array(states)

    def records(self, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        return load_records(self.path(spec["path"]), spec.get("limit"))

    def eval_grid(self, spec: Mapping[str, Any]) -> np.ndarray:
        """Regenerate an evaluation grid and check its state hash."""
        if self._eval_grid is None:
            from mulligan.sim.eval import grid_eval

            if spec["generator"] == "valid_sobol":
                manifest = grid_eval.make_valid_sobol_manifest(
                    task=self.task,
                    num_points=int(spec["num_points"]),
                    seed=int(spec["seed"]),
                    candidate_power=spec.get("candidate_power"),
                )
            elif spec["generator"] == "equal_tile":
                manifest = grid_eval.make_equal_tile_manifest(
                    points_per_cell=int(spec["points_per_cell"]), seed=int(spec["seed"])
                )
            else:
                raise ValueError(f"unknown eval grid generator {spec['generator']!r}")
            points = self.space.array(manifest["points"])
            digest = states_sha256(self.space, points)
            if digest != spec["states_sha256"]:
                raise ValueError(
                    f"regenerated eval grid state hash {digest} != {spec['states_sha256']}"
                )
            self._eval_grid = points
        return self._eval_grid


# ----------------------------------------------------------------- hardness


def _grid_table(inputs: RoundInputs, spec: Mapping[str, Any]) -> H.GridCellTable:
    from mulligan.sim.eval.grid_eval import TASK_SPECS

    axes = [
        H.GridAxis(a.value_key, float(a.range[0]), float(a.range[1]), int(a.n_bins))
        for a in TASK_SPECS[inputs.task].axes
    ]
    return H.GridCellTable.from_csv(inputs.space, axes, inputs.path(spec["table"]))


def build_signal(inputs: RoundInputs, spec: Mapping[str, Any]) -> tuple[H.Signal, bool]:
    """Return ``(signal, uses_eval_grid)`` for one ``signal:`` block of a config."""
    if len(spec) != 1:
        raise ValueError(f"signal spec must have exactly one kind, got {sorted(spec)}")
    kind, args = next(iter(spec.items()))
    if kind == "narrow_shape":
        return H.narrow_shape(inputs.space, args["weights"], float(args["yaw_alpha"])), False
    if kind == "length_model":
        records = [r for src in args["records"] for r in inputs.records(src)]
        model = H.fit_length_model(inputs.space, records, max_steps=float(args["max_steps"]))
        return model, False
    if kind == "grid_sextile_bonus":
        return _grid_table(inputs, args).sextile_bonus(), True
    if kind == "grid_failure_rate":
        return _grid_table(inputs, args).failure_rate(), True
    if kind == "blend":
        parts = [(float(p["weight"]), build_signal(inputs, p["signal"])) for p in args]
        return H.blend([(w, s) for w, (s, _) in parts]), any(g for _, (_, g) in parts)
    raise ValueError(f"unknown hardness signal {kind!r}")


def build_hardness(inputs: RoundInputs, specs: Sequence[Mapping[str, Any]]) -> list[HardnessTerm]:
    terms = []
    for spec in specs:
        signal, uses_grid = build_signal(inputs, spec["signal"])
        normalize = spec.get("normalize", "none")
        if normalize == "minmax":
            signal = H.minmax(signal)
        elif normalize != "none":
            raise ValueError(f"unknown normalize {normalize!r}")
        terms.append(
            HardnessTerm(
                beta=float(spec["beta"]),
                signal=signal,
                uses_eval_grid=uses_grid,
                name=next(iter(spec["signal"])),
            )
        )
    return terms


# --------------------------------------------------------------------- arms


@dataclass
class ArmResult:
    name: str
    policy_source: str
    states: np.ndarray
    components: list[str]


def excluded_states(inputs: RoundInputs, manifest_cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    others = {rel: inputs.states(rel) for rel in manifest_cfg.get("disjoint_from", [])}
    grid = manifest_cfg.get("eval_grid")
    if grid is not None:
        others[f"eval_grid:{grid['generator']}:{grid['seed']}"] = inputs.eval_grid(grid)
    return others


def build_arm(
    inputs: RoundInputs,
    name: str,
    cfg: Mapping[str, Any],
    *,
    budget: int,
    exclude: Mapping[str, np.ndarray],
    exclude_tolerance: float,
) -> ArmResult:
    if len(cfg["starts"]) != 1:
        raise ValueError(f"arm {name}: 'starts' must have exactly one kind")
    kind, spec = next(iter(cfg["starts"].items()))
    space = inputs.space
    if kind == "uniform":
        states = sample_uniform(space, np.random.default_rng(int(spec["seed"])), budget)
        return ArmResult(name, cfg["policy_source"], states, ["uniform"] * budget)
    if kind == "file":
        states = inputs.states(spec["path"], limit=budget)
        if len(states) != budget:
            raise ValueError(f"arm {name}: {spec['path']} has {len(states)} states, need {budget}")
        return ArmResult(name, cfg["policy_source"], states, ["file"] * budget)
    if kind == "promote_then_sobol":
        promoted = promote_failures(space, inputs.records(spec["diagnostics"]))
        fill = sobol_block(inputs.task, int(spec["sobol_start"]), budget - len(promoted))
        return ArmResult(
            name,
            cfg["policy_source"],
            np.concatenate([promoted, fill]),
            [PROMOTED] * len(promoted) + ["sobol_fill"] * len(fill),
        )
    if kind == "select_initial_states":
        perturbation = None
        if "perturbation" in spec:
            p = spec["perturbation"]
            perturbation = LocalPerturbation(
                n=int(p["n"]),
                sigma={k: float(v) for k, v in p.get("sigma", {}).items()},
                seed=int(p["seed"]),
                resample=tuple(p.get("resample", ())),
            )
        prior = [inputs.states(rel) for rel in spec.get("prior", [])]
        candidates = sample_uniform(
            space,
            np.random.default_rng(int(spec["candidates"]["seed"])),
            int(spec["candidates"]["n"]),
        )
        distance = spec.get("distance", "euclidean")
        if distance not in ("euclidean", "squared_euclidean"):
            raise ValueError(f"arm {name}: unknown distance {distance!r}")
        exclude_all = [s for s in exclude.values() if len(s)]
        design = select_initial_states(
            space=space,
            budget=budget,
            records=inputs.records(spec["diagnostics"]),
            prior=np.concatenate(prior) if prior else np.empty((0, space.dim)),
            candidates=candidates,
            hardness=build_hardness(inputs, spec.get("hardness", [])),
            perturbation=perturbation,
            squared_distance=distance == "squared_euclidean",
            prior_distance_periodic=bool(spec.get("prior_distance_periodic", True)),
            exclude=np.concatenate(exclude_all) if exclude_all else None,
            exclude_tolerance=exclude_tolerance,
            allow_eval_grid_feedback=bool(spec.get("eval_grid_feedback", False)),
        )
        return ArmResult(name, cfg["policy_source"], design.states, design.components)
    raise ValueError(f"arm {name}: unknown starts kind {kind!r}")


# ----------------------------------------------------------------- manifest


def merge_arms(
    space: StartSpace, arms: Sequence[ArmResult], *, match_tolerance: float
) -> list[dict[str, Any]]:
    """One row per distinct start; starts within ``match_tolerance`` share a row."""
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for arm in arms:
        for source_index, vec in enumerate(arm.states):
            match = None
            if vectors:
                dist, idx = nearest_state_distance(space, vec[None, :], np.asarray(vectors))
                if dist[0] <= match_tolerance:
                    match = int(idx[0])
            if match is None:
                rows.append(
                    {
                        **dict(zip(space.keys, map(float, vec), strict=True)),
                        "sources": [arm.name],
                        "source_indices": {arm.name: source_index},
                        "policy_source": arm.policy_source,
                    }
                )
                vectors.append(vec)
                continue
            row = rows[match]
            if row["policy_source"] != arm.policy_source:
                raise ValueError(
                    f"shared start maps to two policies: {row['policy_source']} vs {arm.policy_source}"
                )
            if arm.name not in row["sources"]:
                row["sources"] = sorted([*row["sources"], arm.name])
            row["source_indices"][arm.name] = source_index
    return rows


def shuffle_rows(
    rows: list[dict[str, Any]], seed: int, *, row_source: bool = True
) -> list[dict[str, Any]]:
    """Shuffle with a fixed seed and number the rows; ``row_source`` adds the per-row
    ``source`` label (the arm, or ``shared``) that most locked manifests carry."""
    order = np.random.default_rng(seed).permutation(len(rows))
    out = []
    for manifest_index, i in enumerate(order):
        row = dict(rows[int(i)])
        if row_source:
            row["source"] = row["sources"][0] if len(row["sources"]) == 1 else "shared"
        row["manifest_index"] = manifest_index
        out.append(row)
    return out


@dataclass
class RoundBuild:
    task: str
    round: int
    arms: list[ArmResult]
    rows: list[dict[str, Any]]
    match_tolerance: float
    disjoint: list[dict]
    guardrail: dict | None


def build_round(config: Mapping[str, Any], root: Path = REPO_ROOT) -> RoundBuild:
    task = config["task"]
    space = TASKS[task]
    inputs = RoundInputs(root=root, task=task, space=space)
    budget = int(config["budget"])
    manifest_cfg = config["manifest"]
    tolerance = float(manifest_cfg.get("disjoint_tolerance", 1e-6))
    match_tolerance = float(manifest_cfg["match_tolerance"])
    exclude = excluded_states(inputs, manifest_cfg)

    arms = [
        build_arm(inputs, name, cfg, budget=budget, exclude=exclude, exclude_tolerance=tolerance)
        for name, cfg in config["arms"].items()
    ]
    for arm in arms:
        if len(arm.states) != budget:
            raise ValueError(f"arm {arm.name}: {len(arm.states)} starts, budget {budget}")
        gap = min_pairwise_distance(space, arm.states)
        if gap <= 2.0 * match_tolerance:
            raise ValueError(f"arm {arm.name}: two starts only {gap:.3g} apart")

    rows = merge_arms(space, arms, match_tolerance=match_tolerance)
    unique = space.array(rows)
    if min_pairwise_distance(space, unique) <= 2.0 * match_tolerance:
        raise ValueError("merged manifest has two distinct rows closer than 2 x match_tolerance")
    disjoint = check_disjoint(space, unique, exclude, tolerance=tolerance)
    rows = shuffle_rows(
        rows,
        int(manifest_cfg["shuffle_seed"]),
        row_source=bool(manifest_cfg.get("row_source", True)),
    )

    guardrail = None
    if "guardrail" in config:
        g = config["guardrail"]
        arm = next(a for a in arms if a.name == g["arm"])
        prior = [inputs.states(rel) for rel in g.get("prior", [])]
        guardrail = coverage_guardrail(
            space,
            design=arm.states,
            reference=sobol_block(task, int(g["sobol_start"]), budget),
            prior=np.concatenate(prior) if prior else np.empty((0, space.dim)),
            test_points=sample_uniform(
                space,
                np.random.default_rng(int(g["test_points"]["seed"])),
                int(g["test_points"]["n"]),
            ),
            slack=float(g.get("slack", 1.0)),
        )
        if g.get("enforce", False) and not guardrail["passed"]:
            raise ValueError(f"coverage guardrail failed: {guardrail}")
    return RoundBuild(task, int(config["round"]), arms, rows, match_tolerance, disjoint, guardrail)


# ------------------------------------------------------------------ compare


def _match_count(a: np.ndarray, b: np.ndarray, space: StartSpace, tol: float) -> int:
    if len(a) == 0 or len(b) == 0:
        return 0
    dist, _ = nearest_state_distance(space, a, b)
    return int((dist <= tol).sum())


def compare_arm(space: StartSpace, arm: ArmResult, locked: np.ndarray) -> dict[str, Any]:
    """Overlap of a rebuilt arm with its locked starts, per component."""
    report: dict[str, Any] = {
        "n_rebuilt": len(arm.states),
        "n_locked": len(locked),
        "exact": bool(arm.states.shape == locked.shape and np.array_equal(arm.states, locked)),
    }
    offset = 0
    for name in dict.fromkeys(arm.components):
        n = sum(c == name for c in arm.components)
        mine = arm.states[offset : offset + n]
        theirs = locked[offset : offset + n]
        report[name] = {
            "n": n,
            "same_order_exact": bool(mine.shape == theirs.shape and np.array_equal(mine, theirs)),
            "overlap_exact": _match_count(mine, theirs, space, 0.0),
            "overlap_1e-9": _match_count(mine, theirs, space, 1e-9),
            "overlap_1e-3": _match_count(mine, theirs, space, 1e-3),
        }
        offset += n
    return report


MANIFEST_FIELDS = ("sources", "source", "source_indices", "policy_source", "manifest_index")


def compare_manifest(space: StartSpace, rows: list[dict], locked: list[dict]) -> dict[str, Any]:
    mismatched = []
    for i, (mine, theirs) in enumerate(zip(rows, locked, strict=False)):
        same = all(float(mine[k]) == float(theirs[k]) for k in space.keys) and all(
            mine[f] == theirs[f] for f in MANIFEST_FIELDS if f in theirs
        )
        if not same:
            mismatched.append(i)
    return {
        "n_rebuilt": len(rows),
        "n_locked": len(locked),
        "rows_identical": not mismatched and len(rows) == len(locked),
        "n_mismatched_rows": len(mismatched) + abs(len(rows) - len(locked)),
        "state_set_overlap_exact": _match_count(space.array(rows), space.array(locked), space, 0.0),
    }


def compare_locked(build: RoundBuild, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    space = TASKS[build.task]
    locked = config.get("locked", {})
    out: dict[str, Any] = {"arms": {}, "sha256_ok": {}}
    for name, spec in locked.get("arms", {}).items():
        path = root / spec["path"]
        out["sha256_ok"][spec["path"]] = sha256_file(path) == spec["sha256"]
        arm = next(a for a in build.arms if a.name == name)
        out["arms"][name] = compare_arm(space, arm, space.array(load_states(path)))
    if "manifest" in locked:
        path = root / locked["manifest"]["path"]
        out["sha256_ok"][locked["manifest"]["path"]] = (
            sha256_file(path) == locked["manifest"]["sha256"]
        )
        out["manifest"] = compare_manifest(space, build.rows, load_states(path))
    out["reproduced"] = (
        all(out["sha256_ok"].values())
        and all(a["exact"] for a in out["arms"].values())
        and out.get("manifest", {}).get("rows_identical", True)
    )
    return out


# ---------------------------------------------------------------------- CLI


def write_outputs(build: RoundBuild, out_dir: Path, report: Mapping[str, Any]) -> None:
    space = TASKS[build.task]
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in build.arms:
        payload = {
            "states": space.states(arm.states),
            "_meta": {
                "task": build.task,
                "round": build.round,
                "arm": arm.name,
                "policy_source": arm.policy_source,
                "components": dict(Counter(arm.components)),
                "component_per_state": arm.components,
            },
        }
        (out_dir / f"{arm.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    # Top-level fields as in the locked manifests, which the collector, the protocol quota
    # (its manifest hash) and the splitters read.
    manifest = {
        "task": build.task,
        "keys": list(space.keys),
        "match_tolerance": build.match_tolerance,
        "states": build.rows,
        "_meta": {
            "round": build.round,
            "arm_counts": dict(Counter(s for r in build.rows for s in r["sources"])),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="root of config input paths")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 unless the locked files are reproduced exactly"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    build = build_round(config, root=args.root)
    report: dict[str, Any] = {
        "config": str(args.config),
        "counts": {arm.name: dict(Counter(arm.components)) for arm in build.arms},
        "n_manifest_rows": len(build.rows),
        "disjoint": build.disjoint,
        "guardrail": build.guardrail,
    }
    if "locked" in config:
        report["locked"] = compare_locked(build, config, args.root)
    write_outputs(build, args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "disjoint"}, indent=2))
    if args.check and not report.get("locked", {}).get("reproduced", False):
        print("locked files NOT reproduced exactly", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
