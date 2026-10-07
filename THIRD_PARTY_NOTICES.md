# Third-party notices

Mulligan is released under the MIT License (see `LICENSE`). Section 1 lists code in this repository
that was copied or adapted from other projects, with their notices. Section 2 lists dependencies
that `uv sync` installs but this repository does not contain. Section 3 lists data that the tools
download from third parties. The code, models and datasets of the Mulligan authors are MIT;
third-party code keeps its own terms. Upstream DROID states no license (section 2).

## 1. Code in this repository

### EXPO and RLPD: `mulligan/baselines/rlpd/`

The JAX SAC/RLPD agent (`mulligan/baselines/rlpd/agent/`), its replay buffers
(`mulligan/baselines/rlpd/data/`) and the training loop, evaluation, env wrapper and dataset loader
(`train.py`, `evaluation.py`, `env.py`, `datasets.py`, `configs.py`) are based on the EXPO source code
(<https://github.com/pd-perry/EXPO>) and on the RLPD source code (<https://github.com/ikostrikov/rlpd>),
on which EXPO builds. Every vendored file starts with both notices below and names its EXPO source
file; `mulligan/baselines/rlpd/LICENSE-EXPO` and `LICENSE-RLPD` hold the two license texts. The HiL-SERL replay buffer (`mulligan/baselines/hilserl/replay.py`) is adapted from
EXPO's `Dataset`, `RoboReplayBuffer` and `combine` and carries the same two notices.
`tests/release/test_license_headers.py` checks these headers.

```
MIT License

Copyright (c) 2025 pd-perry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

RLPD is released under the MIT License, Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura
Smith, with the same permission notice as above. RLPD in turn builds on jaxrl
(<https://github.com/ikostrikov/jaxrl>, MIT, Copyright (c) 2021 Ilya Kostrikov). No file was copied
from jaxrl directly; any jaxrl-derived code reaches this repository through RLPD and EXPO.

### LeRobot: `mulligan/utils/lerobot_patches.py`, `mulligan/data/fast_lerobot_reader.py`, `mulligan/real/policy/lerobot_patches.py`

Parts of these three files are adapted from LeRobot (<https://github.com/huggingface/lerobot>, at the
pinned commit `0530dd9b`): `DiffusionConfig.__post_init__` and the `DiffusionPolicy` /
`DiffusionModel` methods that `lerobot_patches.py` replaces at runtime, and
`DatasetReader.get_item`, which `FastDatasetReader.get_item` mirrors, and
`StreamingVideoEncoder.feed_frame`, which the real-robot patch replaces with a blocking version. LeRobot is licensed under the
Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>), Copyright 2024 The
HuggingFace Inc. team (and, for the diffusion policy, Columbia Artificial Intelligence, Robotics
Lab). Each file starts with LeRobot's notice and a statement of the modifications.

### dm_control: `mulligan/sim/render_cgl.py`

The macOS CGL OpenGL context is adapted from the dm_control / MuJoCo Python rendering code and keeps
its header: Copyright 2017 The dm_control Authors, licensed under the Apache License, Version 2.0
(<http://www.apache.org/licenses/LICENSE-2.0>).

### robosuite: `mulligan/teleop/spacemouse.py`, `mulligan/sim/_patches.py`

The SpaceMouse driver is adapted from robosuite's SpaceMouse input device
(<https://github.com/ARISE-Initiative/robosuite>, `robosuite/devices/spacemouse.py`), and the macOS
CGL render-context initializer in `mulligan/sim/_patches.py` from robosuite's
`MjRenderContext.__init__` (`robosuite/utils/binding_utils.py`); both files start with robosuite's
MIT notice. `mulligan/real/operator_ui/cards.py` and
`hardware/square_nut/generate_nut_peg_stl.py` copy the square-nut and peg dimensions from robosuite's
`square-nut.xml` and `pegs_arena.xml` assets. robosuite's license (the MIT part of its `LICENSE` at
the pinned commit `85abee2`):

```
MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Policy Arena: `arena/`

`arena/` is the project's own Policy Arena app, written by the Mulligan authors and released under
the MIT License (`arena/LICENSE`). Its npm dependencies are installed by `bun install` from
`arena/bun.lock` and are not vendored.

## 2. Dependencies (not in this repository)

`uv sync` installs these from exact git pins or PyPI; this repository does not redistribute them.
Licenses were read from each project's license file at the pinned commit.

| Project | Pin | License | Notes |
|---|---|---|---|
| LeRobot (<https://github.com/huggingface/lerobot>) | `0530dd9b97929f017d29640029684644c5313d56` | Apache-2.0, Copyright 2024 The HuggingFace Inc. team | `mulligan/utils/lerobot_patches.py` and `mulligan/real/policy/lerobot_patches.py` patch LeRobot internals at runtime (adapted code: section 1). |
| robosuite (<https://github.com/ARISE-Initiative/robosuite>) | `85abee228d1c43ab1939bce33028099945d453b4` (1.5.2) | MIT, Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab | `mulligan/sim/_patches.py` patches robosuite at runtime (adapted code: section 1). |
| MimicGen (fork of <https://github.com/NVlabs/mimicgen>) | `e0739b2108226af16a03cd499eb344a431e384a3` | NVIDIA Source Code License, non-commercial | See the warning below. |
| agentlace (<https://github.com/youliangtan/agentlace>) | `cf2c337c5e3694cdbfc14831b239bd657bc4894d` | MIT, Copyright (c) 2023 You Liang Tan | HiL-SERL transport. The `LICENSE` file at the pinned commit is the MIT License. |
| DROID (fork of <https://github.com/droid-dataset/droid>) | `c0c8b29e1fcf76e424177529d5440620234918be` | upstream: none stated; the fork's own commits: MIT | `robot/` project only. Neither upstream DROID nor the public fork has a `LICENSE` file or license metadata. The fork's own commits (listed in `docs/hardware/droid_fork.md`) are by the Mulligan authors and are MIT. The upstream DROID code in the fork keeps the terms of its authors, who have not stated any; we cannot relicense it. The release depends on DROID by git pin and ships none of its code. |
| MuJoCo | 3.3.7 (PyPI) | Apache-2.0 | |
| JAX, Flax, Optax, TensorFlow Probability | PyPI | Apache-2.0 | RLPD and HiL-SERL baselines |
| pyzed (Stereolabs ZED SDK Python API) | 4.2, supplied by the station owner | Stereolabs ZED SDK license (proprietary) | `robot/` project only; the wheel is downloaded from Stereolabs and not redistributed. |

### MimicGen license warning

MimicGen is distributed under the NVIDIA Source Code License. Its section 3.3 limits use of the work
and its derivatives to non-commercial purposes, meaning research or evaluation. Mulligan imports it
to register the Square-Broad (`Square_D1`) simulation environment and does not vendor any of its
code, but a default install (`uv sync`) installs it. You are responsible for complying with its
license. Commercial users need to remove the dependency or obtain a separate license from NVIDIA.

## 3. Data downloaded from third parties

The RLPD baseline (`mulligan.baselines.rlpd.datasets`) downloads two public datasets on first use;
neither is redistributed here.

| Dataset | Source | License |
|---|---|---|
| robomimic Square PH (`--offline_data=robomimic_ph`) | robomimic dataset release | robomimic is MIT (Copyright (c) 2021 Stanford Vision and Learning Lab); see the robomimic project for its dataset terms |
| MimicGen core `square_d1` (`--offline_data=mimicgen_core`) | Hugging Face `amandlek/mimicgen_datasets` | CC-BY-4.0 (dataset card) |

The Mulligan datasets on the Hugging Face organization `mulligan` are MIT at their release revisions
(`docs/data.md`). Their simulation data were generated with robosuite and MimicGen environments.
