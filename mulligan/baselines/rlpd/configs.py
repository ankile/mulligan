# Based on the EXPO source code (https://github.com/pd-perry/EXPO) and the RLPD source code
# (https://github.com/ikostrikov/rlpd), on which EXPO builds.
# Vendored from EXPO's configs/{td,sac,rlpd}_config.py. Both projects are MIT-licensed; their
# notices follow.
#
# ---- EXPO ----
# MIT License
#
# Copyright (c) 2025 pd-perry
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---- RLPD (https://github.com/ikostrikov/rlpd, LICENCE) ----
# MIT License
#
# Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""RLPD agent hyperparameters: the keyword arguments of ``SACLearner.create``.

``AGENT_DEFAULTS`` is the RLPD config (10 critics with 2 sampled for the target, critic
LayerNorm) with three hidden layers of 256 and no entropy term in the critic backup, the agent of
every RLPD and HiL-SERL recipe. Each key is a ``mulligan.baselines.rlpd.train`` flag of the same
name.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

AGENT_DEFAULTS: dict[str, Any] = dict(
    actor_lr=3e-4,
    critic_lr=3e-4,
    temp_lr=3e-4,
    hidden_dims=(256, 256, 256),
    discount=0.99,
    tau=0.005,
    num_qs=10,
    num_min_qs=2,
    critic_dropout_rate=None,
    critic_weight_decay=None,
    critic_layer_norm=True,
    target_entropy=None,
    init_temperature=1.0,
    backup_entropy=False,
)


def parse_bool(text: str) -> bool:
    if text.lower() in ("true", "1", "yes"):
        return True
    if text.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected True/False, got {text!r}")


def parse_dims(text: str) -> tuple[int, ...]:
    """``256,256,256`` -> ``(256, 256, 256)``."""
    return tuple(int(x) for x in text.split(","))


def _optional_float(text: str) -> float | None:
    return None if text.lower() == "none" else float(text)


def _parser(key: str, default: Any) -> Callable[[str], Any]:
    if key == "hidden_dims":
        return parse_dims
    if default is None:
        return _optional_float
    if isinstance(default, bool):
        return parse_bool
    return type(default)


def add_agent_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("agent (SACLearner.create)")
    for key, default in AGENT_DEFAULTS.items():
        shown = ",".join(map(str, default)) if key == "hidden_dims" else default
        group.add_argument(f"--{key}", type=_parser(key, default), default=default, help=f"{shown}")


def agent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in AGENT_DEFAULTS}
