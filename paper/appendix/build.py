"""Rebuild and verify every appendix figure and table from the pinned paper evidence.

    python -m paper.appendix.build            # tables into paper/build/tables, figures into paper/build/figs
    python -m paper.appendix.build --check    # verify a finished build against the frozen manuscript

``--check`` requires every generated table body to equal the frozen manuscript's
(``paper/reference/tables``), every restored illustration to equal its archived export,
the catalog to match the manuscript's appendix inventory, and the appendix figures to
match the bundled figure manifest (see :mod:`paper.figures`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mulligan.plotting import paper
from paper import figures
from paper.appendix.artifacts import REFERENCE_TABLES, TABLES_DIR

HERE = Path(__file__).resolve().parent


def catalog() -> dict:
    return json.loads((HERE / "catalog.json").read_text())


def check_coverage() -> None:
    """The catalog must list exactly the frozen manuscript's appendix figures and tables."""
    inventory = catalog()
    manuscript = json.loads(figures.MANUSCRIPT_PATH.read_text())
    figs, tables = set(manuscript["appendix_figures"]), set(manuscript["appendix_tables"])
    expected_figures = {row["file"] for row in inventory["figures"]}
    expected_tables = {row["label"] for row in inventory["tables"]}
    if figs != expected_figures:
        raise AssertionError(f"Figure inventory differs: {sorted(figs ^ expected_figures)}")
    if tables != expected_tables:
        raise AssertionError(f"Table inventory differs: {sorted(tables ^ expected_tables)}")
    print(f"Appendix inventory: {len(figs)} figures/assets and {len(tables)} tables.", flush=True)


def prepare_tables(*, check: bool) -> list[Path]:
    from paper.appendix.critic_data import prepare as critic_data
    from paper.appendix.data_ledger import prepare as ledger
    from paper.appendix.hilserl import prepare as hilserl
    from paper.appendix.paper_side import divl_comparison, training_compute
    from paper.appendix.real_results import statistics
    from paper.appendix.reference_data import prepare as reference
    from paper.appendix.simulation import prepare as simulation
    from paper.appendix.value_learning import prepare as value
    from paper.stats import sim_welch_table

    value.extract()
    paths = list(statistics.write_tables(check=check))
    paths += simulation.write_tables(check=check)
    paths += reference.build(check=check)
    paths += ledger.build(check=check)
    paths.append(hilserl.write_tables(check=check))
    paths.append(critic_data.write_tables(check=check))
    # Tables built by the manuscript-side builders (sim significance, DIVL frozen
    # actor, training compute).
    for builder in (sim_welch_table, divl_comparison, training_compute):
        path = builder.build(TABLES_DIR)
        if check and path.read_text() != (REFERENCE_TABLES / path.name).read_text():
            raise AssertionError(f"generated table differs from the manuscript: {path.name}")
        paths.append(path)
    return paths


def check_tables(paths: list[Path]) -> None:
    """Every generated table of the catalog was written and equals the manuscript's, and
    every authored table has a pinned body (written by ``reference_data``)."""
    written = {path.name for path in paths}
    for row in catalog()["tables"]:
        if row["kind"] == "authored":
            name = row["label"].removeprefix("tab:") + ".tex"
            if not (TABLES_DIR / "authored" / name).is_file():
                raise AssertionError(f"{row['label']}: authored table body has no pinned copy")
            continue
        if row["output"] not in written:
            raise AssertionError(f"{row['label']}: {row['output']} was not generated")
        if (TABLES_DIR / row["output"]).read_text() != (
            REFERENCE_TABLES / row["output"]
        ).read_text():
            raise AssertionError(f"{row['label']}: table body differs from the manuscript")
    print(f"Generated tables equal the manuscript's ({len(written)} files).", flush=True)


def appendix_entries() -> list[str]:
    files = {row["file"] for row in catalog()["figures"]}
    return [entry.name for entry in figures.REGISTRY if set(entry.outputs) & files]


def require_built_figures() -> None:
    """``--check`` validates a finished build; stop before any work if figures are missing."""
    missing = sorted(
        row["file"] for row in catalog()["figures"] if not (paper.FIGS_DIR / row["file"]).is_file()
    )
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise SystemExit(
            f"--check needs a finished build, but {len(missing)} appendix figures/assets are "
            f"missing from {paper.FIGS_DIR} ({shown}). Run `python -m paper.appendix.build` "
            "(or `python -m paper.figures`) first."
        )


def check_assets() -> None:
    from paper.appendix.reference_data.prepare import materialize_assets as reference
    from paper.appendix.simulation.prepare import materialize_assets as simulation

    reference(check=True)
    simulation(check=True)
    print("Authored illustration exports match their archived inputs.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify inputs, generated tables, restored assets and figure hashes.",
    )
    args = parser.parse_args()
    check_coverage()
    if args.check:
        require_built_figures()
    paths = prepare_tables(check=args.check)
    names = appendix_entries()
    if args.check:
        check_tables(paths)
        check_assets()
        if figures.check(only=names) != 0:
            raise SystemExit(1)
        return
    built = figures.run_builds([entry for entry in figures.REGISTRY if entry.name in names])
    print(f"Wrote {len(paths)} tables and {len(built)} appendix figures/assets.", flush=True)


if __name__ == "__main__":
    main()
