"""Named Vision-IQL value-head recipes.

``--iql-recipe <name>`` applies the recipe's fields as the effective config; a field whose
flag is passed explicitly keeps the passed value (explicit CLI > recipe > argparse
default). Without ``--iql-recipe`` the argparse defaults stand. ``gamma`` is not part of
a recipe: it is set per task in the configs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RealIQLRecipe:
    """A named bundle of Vision-IQL value-head hyperparameters."""

    name: str
    description: str
    num_atoms: int
    tau_base: float
    tau_min: float
    tau_max: float
    tau_entropy_alpha: float
    hl_gauss_sigma_ratio: float


# The value-head settings of the released critics.
DIVL = RealIQLRecipe(
    name="divl",
    description=(
        "Categorical DIVL value head (C51-101 + HL-Gauss) with entropy-adaptive tau; "
        "gamma is set per task."
    ),
    num_atoms=101,
    tau_base=0.7,
    tau_min=0.5,
    tau_max=0.95,
    tau_entropy_alpha=0.4,
    hl_gauss_sigma_ratio=0.75,
)

RECIPES: dict[str, RealIQLRecipe] = {DIVL.name: DIVL}

# recipe field -> the iql_args CLI flag that sets it (for explicit-override
# detection). ``name``/``description`` are metadata, not applied to args.
_FIELD_TO_FLAG: dict[str, str] = {
    "num_atoms": "--num-atoms",
    "tau_base": "--tau-base",
    "tau_min": "--tau-min",
    "tau_max": "--tau-max",
    "tau_entropy_alpha": "--tau-entropy-alpha",
    "hl_gauss_sigma_ratio": "--hl-gauss-sigma-ratio",
}


def recipe_names() -> list[str]:
    return sorted(RECIPES)


def apply_recipe(args, argv: list[str]) -> str | None:
    """Apply ``args.iql_recipe`` to ``args`` in place; explicit CLI flags win.

    Returns the applied recipe name (also stamped onto ``args.iql_recipe`` for
    provenance), or ``None`` when no recipe was requested. Fails loud on an
    unknown recipe name. A no-op when ``args.iql_recipe`` is falsy.
    """
    name = getattr(args, "iql_recipe", None)
    if not name:
        return None
    if name not in RECIPES:
        raise ValueError(f"unknown --iql-recipe {name!r}; known recipes: {recipe_names()}")
    recipe = RECIPES[name]
    # A flag counts as explicitly passed whether written as ``--flag v`` or ``--flag=v``.
    passed = {tok.split("=", 1)[0] for tok in argv if tok.startswith("--")}
    applied: dict[str, object] = {}
    for f in fields(recipe):
        if f.name in ("name", "description"):
            continue
        flag = _FIELD_TO_FLAG[f.name]
        if flag in passed:
            continue  # explicit CLI override wins
        setattr(args, f.name, getattr(recipe, f.name))
        applied[f.name] = getattr(recipe, f.name)
    print(f"[iql-recipe] applied {name!r}: {applied}")
    return name
