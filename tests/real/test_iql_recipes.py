"""Tests for the durable named Vision-IQL recipes (mulligan.real.train.iql_recipes)."""

from __future__ import annotations

import argparse

import pytest

from mulligan.real.train.iql_recipes import (
    DIVL,
    RECIPES,
    apply_recipe,
    recipe_names,
)


def _args(**over):
    """A minimal args namespace with the recipe-managed fields at code defaults
    (mirrors the iql_args argparse defaults for the DIVL block)."""
    ns = argparse.Namespace(
        iql_recipe=None,
        num_atoms=101,
        tau_base=0.7,
        tau_min=0.5,
        tau_max=0.95,
        tau_entropy_alpha=0.0,
        hl_gauss_sigma_ratio=0.75,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


class TestApplyRecipe:
    def test_noop_when_unset(self):
        a = _args()
        before = vars(a).copy()
        assert apply_recipe(a, []) is None
        assert vars(a) == before  # no-op

    def test_applies_divl_v3(self):
        a = _args(iql_recipe="divl")
        name = apply_recipe(a, ["--iql-recipe", "divl"])
        assert name == "divl"
        assert a.tau_entropy_alpha == pytest.approx(0.4)
        assert a.num_atoms == 101
        assert a.tau_base == pytest.approx(0.7)

    def test_explicit_cli_overrides_recipe(self):
        # num_atoms explicitly passed -> recipe must NOT clobber it.
        a = _args(iql_recipe="divl", num_atoms=51, tau_entropy_alpha=0.6)
        apply_recipe(
            a,
            ["--iql-recipe", "divl", "--num-atoms", "51", "--tau-entropy-alpha", "0.6"],
        )
        assert a.num_atoms == 51  # explicit wins
        assert a.tau_entropy_alpha == pytest.approx(0.6)  # explicit wins
        assert a.tau_min == pytest.approx(0.5)  # not overridden -> from recipe

    def test_explicit_equals_form_overrides(self):
        a = _args(iql_recipe="divl", num_atoms=51)
        apply_recipe(a, ["--iql-recipe=divl", "--num-atoms=51"])
        assert a.num_atoms == 51

    def test_unknown_recipe_raises(self):
        a = _args(iql_recipe="does_not_exist")
        with pytest.raises(ValueError, match="unknown --iql-recipe"):
            apply_recipe(a, ["--iql-recipe", "does_not_exist"])

    def test_registry_wellformed(self):
        assert "divl" in recipe_names()
        assert RECIPES["divl"] is DIVL


def test_abbreviated_flag_is_refused_instead_of_clobbered_by_the_recipe(capsys):
    """argparse would accept ``--tau-entropy`` as ``--tau-entropy-alpha``, but the recipe only
    recognizes full flag names and would overwrite the value; the critic parser refuses it."""
    from mulligan.real.train.iql_args import parse_args

    base = ["--repo-ids", "org/r", "--encoder-artifact", "ent/proj/dp:v0"]
    explicit = parse_args([*base, "--tau-entropy-alpha", "0.0", "--iql-recipe", "divl"])
    assert explicit.tau_entropy_alpha == 0.0
    with pytest.raises(SystemExit):
        parse_args([*base, "--tau-entropy", "0.0", "--iql-recipe", "divl"])
    assert "unrecognized arguments: --tau-entropy" in capsys.readouterr().err
