"""Run a real trainer from a flattened config in ``configs/real/<task>/``.

    python -m mulligan.real.train.launch configs/real/marker_d2/r05_critic.yaml --output-dir outputs/r05_critic
    python -m mulligan.real.train.launch configs/real/marker_d2/r05_mulligan_dp.yaml --output-dir out -- --training-steps 500

Each config holds the trainer flags of one released checkpoint (``args``), the pinned dataset revisions and
episode selectors (``dataset_revisions``, ``dataset_episodes``), for DP actors the camera crop boxes
(``camera_crops``), and the checkpoint and paper results it belongs to. Flags after ``--`` are
appended and override the config (argparse keeps the last value); ``--dataset-revisions`` /
``--dataset-episodes`` / ``--camera-crop`` after ``--`` replace the config's entry for the repos or cameras
they name and keep the others.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from mulligan.real.train.dataset_selectors import EpisodeSelector

TRAINERS = {
    "dp-actor": "mulligan.real.train.policy",
    "idql-critic": "mulligan.real.train.critic",
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    for key in ("kind", "trainer", "args"):
        if key not in config:
            raise ValueError(f"{path}: missing {key!r}")
    if TRAINERS.get(config["kind"]) != config["trainer"]:
        raise ValueError(
            f"{path}: trainer {config['trainer']!r} does not match kind {config['kind']!r}"
        )
    return config


def _format_value(value) -> str:
    if isinstance(value, bool):
        raise TypeError("booleans are flags, not values")
    return str(value)


def args_to_argv(args: Mapping[str, object]) -> list[str]:
    """``{flag: value}`` -> argv. ``true`` is a bare flag; a list is one comma-joined value (the
    trainers take ``--repo-ids a,b``, ``--hidden-dims 1024,1024``)."""
    argv: list[str] = []
    for flag, value in args.items():
        if value is False or value is None:
            raise ValueError(f"--{flag}: use the flag's --no- form instead of false/null")
        argv.append(f"--{flag}")
        if value is True:
            continue
        if isinstance(value, list):
            argv.append(",".join(_format_value(v) for v in value))
        else:
            argv.append(_format_value(value))
    return argv


# Keyed map flags (``KEY=VALUE`` items): an override replaces the config's entry for its key instead
# of the whole map. ``--side-crop`` is the DP trainer's alias of ``--camera-crop``.
KEY_MAP_FLAGS = {
    "--dataset-revisions": "--dataset-revisions",
    "--dataset-episodes": "--dataset-episodes",
    "--camera-crop": "--camera-crop",
    "--side-crop": "--camera-crop",
}


def split_key_map_overrides(extra: Sequence[str]) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Pull the ``KEY_MAP_FLAGS`` values (``KEY=VALUE``) out of ``extra``."""
    rest: list[str] = []
    overrides: dict[str, dict[str, str]] = {flag: {} for flag in set(KEY_MAP_FLAGS.values())}
    current = None
    for token in extra:
        if token.startswith("--"):
            flag, sep, value = token.partition("=")
            current = KEY_MAP_FLAGS.get(flag)
            if current is None:
                rest.append(token)
                continue
            if not sep:
                continue
            token = value
        elif current is None:
            rest.append(token)
            continue
        key, sep, value = token.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"{current} expects KEY=VALUE, got {token!r}")
        if key in overrides[current]:
            raise ValueError(f"Repeated {current} for {key}")
        overrides[current][key] = value
    return rest, overrides


def config_to_argv(config: Mapping, extra: Sequence[str] = ()) -> list[str]:
    extra, overrides = split_key_map_overrides(extra)
    argv = args_to_argv(config["args"])
    revisions = {**(config.get("dataset_revisions") or {}), **overrides["--dataset-revisions"]}
    if revisions:
        argv += ["--dataset-revisions", *(f"{repo}={rev}" for repo, rev in revisions.items())]
    episodes = {
        repo: EpisodeSelector.from_json(sel).to_cli()
        for repo, sel in (config.get("dataset_episodes") or {}).items()
    }
    episodes.update(overrides["--dataset-episodes"])
    if episodes:
        argv += ["--dataset-episodes", *(f"{repo}={sel}" for repo, sel in episodes.items())]
    crops = {
        camera: ",".join(str(int(v)) for v in box)
        for camera, box in (config.get("camera_crops") or {}).items()
    }
    crops.update(overrides["--camera-crop"])
    for camera, box in crops.items():
        argv += ["--camera-crop", f"{camera}={box}"]
    return argv + list(extra)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-argv", action="store_true", help="print the trainer argv and exit")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    ns = parser.parse_args(argv)
    config = load_config(ns.config)
    trainer_argv = config_to_argv(config, ["--output-dir", ns.output_dir, *extra])
    if ns.print_argv:
        print(" ".join(trainer_argv))
        return
    importlib.import_module(config["trainer"]).main(trainer_argv)


if __name__ == "__main__":
    main()
