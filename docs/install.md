# Installation

Mulligan is one [uv](https://docs.astral.sh/uv/) project (Python 3.12) for everything except the robot
workstation: Mulligan itself and the RLPD and HiL-SERL baselines share one lock (`uv.lock`).
The robot workstation uses a second project in `robot/` because the ZED SDK needs numpy 1.x.

Supported platforms: Linux x86_64 and macOS arm64. Only these two are in the lock.

## Quick start

```bash
git clone <repository-url> mulligan && cd mulligan
uv sync --frozen                          # core: sim training/eval and baselines, real training, paper figures
```

Then add what you need. Each line below is a complete `uv sync`. Combine extras in one command
(`uv sync --frozen --extra teleop --extra stage-labeling`), because a later sync without an extra removes it:

```bash
uv sync --frozen --extra teleop           # SpaceMouse collection (sim)
uv sync --frozen --extra stage-labeling   # google-genai; needs GEMINI_API_KEY
uv sync --frozen --all-extras             # everything (what CI installs)
bash scripts/sync_robot_env.sh            # robot workstation only (ZED SDK 4.2 installed)
```

`uv sync` installs the exact versions in the lock, removes anything else, and installs the `mulligan`
and `paper` packages editable. Run commands with `uv run ...` or activate `.venv`.

| Extra | For | Adds |
|---|---|---|
| (core) | sim training and evaluation, the RLPD and HiL-SERL baselines, real-robot training, paper figures and tables | torch, lerobot, robosuite, mimicgen, mujoco, jax 0.6.2 (+ CUDA 12 plugin on Linux), flax, ... |
| `teleop` | SpaceMouse and keyboard teleoperation in simulation | hidapi, easyhid, pynput |
| `stage-labeling` | Gemini stage labeler | google-genai (needs `GEMINI_API_KEY`) |

## What the lock pins

- torch 2.11.0, torchvision 0.26.0, torchcodec 0.11.1. On Linux, torch and torchvision come from the
  PyTorch CUDA 12.8 index (`+cu128` wheels); on macOS from PyPI.
- LeRobot, robosuite and MimicGen at exact git commits (LeRobot `0530dd9b`, robosuite `85abee22` =
  1.5.2, MimicGen fork `e0739b21`). Mulligan patches LeRobot and robosuite internals at runtime, so do
  not upgrade them independently.
- mujoco 3.3.7. mujoco 3.4.0 changes physics behavior.
- numpy 2.2.x (bounded to `>=2.0,<2.3` through `[tool.uv].override-dependencies`).
- One OpenCV distribution: `opencv-python`. LeRobot asks for `opencv-python-headless`, but both builds
  unpack into the same `cv2/` directory, so the headless one is removed by an override.
- jax 0.6.2 resolves alongside torch with a single `nvidia-cudnn-cu12` (9.19). Both
  frameworks can use the same GPU in one process. jax preallocates 75% of GPU
  memory by default; set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if torch shares the GPU.

MimicGen is licensed under the NVIDIA Source Code License (non-commercial). See
`THIRD_PARTY_NOTICES.md`.

## System requirements

On Ubuntu/Debian, the packages CI installs cover the Python side:

```bash
sudo apt-get install -y libgl1 libglib2.0-0 ffmpeg libegl1
export MUJOCO_GL=egl        # headless MuJoCo rendering
```

- glibc >= 2.28 (Linux).
- NVIDIA driver that supports CUDA 12.8 for GPU training on Linux. The CUDA wheels also run on CPU.
  The sim training recipes use bf16 autocast and need an Ampere or newer GPU
  ([compute.md](compute.md#gpu-generation)).
- `libgl1` and `libglib2.0-0` for `opencv-python`.
- FFmpeg 4-7 shared libraries for torchcodec (video decoding): `ffmpeg` on Debian/Ubuntu,
  `brew install ffmpeg` on macOS.
- Headless MuJoCo rendering: EGL (`libegl1`, then `MUJOCO_GL=egl`) or OSMesa (`MUJOCO_GL=osmesa`).
- Paper figures: Chrome or Chromium for the Fig. 1 teaser (`paper/teaser/build_teaser.py`), found
  through `MULLIGAN_CHROME` or `google-chrome` / `chromium` on `PATH`, and Ghostscript (`gs`) to crop
  it. Without Chrome, `python -m paper.figures --exclude teaser` builds everything else. Poppler
  (`pdftoppm`, Debian package `poppler-utils`) is needed only for the visual comparison
  `python -m paper.compare_reference`.
- Arena (`arena/`, optional): [bun](https://bun.sh) 1.3 and Node 20.19+, 22.13+ or 24 for the build,
  and Chrome or Chromium for its browser check (`arena/scripts/verify_release_browser.py`, same lookup
  as above). See `arena/README.md`.
- Disk: the Linux lock pulls about 3 GB of `nvidia-*` wheels plus the jax CUDA plugin. Models, datasets
  and grids download to the Hugging Face cache on first use (`HF_HOME`).

### macOS (arm64)

- torch comes from PyPI (CPU and MPS); jax runs on CPU.
- Interactive MuJoCo viewers (sim teleop, sim DAgger, the HiL-SERL actor) must run under `mjpython`,
  which ships with mujoco: `uv run mjpython -m mulligan.sim.collect.teleop ...`. Offscreen camera
  renders use a CGL context; the sim code sets `MUJOCO_GL=cgl` when it is unset.
- SpaceMouse: `brew install hidapi` and Input Monitoring permission for the terminal
  ([hardware/spacemouse.md](hardware/spacemouse.md)).

Check an install (it registers the simulated Square tasks through the release's robosuite/MimicGen
patches, then imports the rest):

```bash
uv run python -c "from mulligan.sim.envs import register_square_environments; register_square_environments(); import lerobot, torchcodec, torch, cv2; print('ok')"
uv run python -c "from torchcodec.decoders import VideoDecoder"   # fails fast if FFmpeg libs are missing
```

The first command ends with `ok`. Before it, robosuite and MimicGen print warnings that are expected
and harmless: `[robosuite WARNING] No private macro file found!` (and two lines on how to create
one), `Could not import robosuite_models`, `Could not load the mink-based whole-body IK` (robots and
controllers the Square tasks do not use) and `robosuite task zoo environments not imported`.

## Robot workstation (`robot/`)

The Franka/ZED station runs a separate uv project, `robot/`, with its own lock and its own
`robot/.venv`. It installs this repository editable (with the `teleop` extra) plus the DROID hardware
stack (the DROID fork at `c0c8b29`, see [hardware/droid_fork.md](hardware/droid_fork.md); `gym`,
`zerorpc`) and pyzed. pyzed 4.2 is compiled against the numpy 1.x
ABI, so `robot/` overrides numpy to 1.26.4 for its whole lock. uv overrides are lock-global, which is
why this cannot be an extra of the main project.

Requirements: Linux x86_64, ZED SDK 4.2 installed under `/usr/local/zed`, the DROID/polymetis stack
running on the NUC, and a station file at `~/.config/droid/station.env` (see
`configs/real/station.env.example`).

```bash
bash scripts/sync_robot_env.sh
uv run --project robot --frozen python -m mulligan.real.<entrypoint> ...
```

The pyzed wheel is not shipped. The station owner places the wheel matching the installed ZED SDK at
`robot/wheels/pyzed-4.2-cp312-cp312-linux_x86_64.whl` (gitignored). If it is missing,
`scripts/sync_robot_env.sh` downloads it from Stereolabs. `robot/uv.lock` pins its sha256
(`d6b2c0ad...`), so a different wheel makes `uv sync --frozen` fail. For a different ZED SDK version,
change the wheel path in `robot/pyproject.toml` and in the sync script, put the wheel in place, then run
`uv lock --project robot` (re-locking needs the wheel present). A machine without the ZED SDK can skip
pyzed with `uv sync --frozen --project robot --no-group zed`, but only for work that does not import
DROID's robot environment (training, `--help`, offline tests): `droid.robot_env` imports the ZED camera
reader, which fails without pyzed (`NameError: name 'sl' is not defined`), so collection and evaluation
need the `zed` group.

The robot environment takes about 9 GB (`robot/.venv`; uv hardlinks it from its cache when both are on
one filesystem, otherwise count it twice), and each deployed DP actor with its critic adds about 0.5 GB
to the Hugging Face cache. Point `UV_CACHE_DIR` and `HF_HOME` at a disk with room if the home disk is
small.

Always pass `--project robot` for robot commands. A missing `--project` runs in the main `.venv` and
fails with `No module named 'droid'` instead of silently changing numpy.
