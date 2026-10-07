# Robot station

The real-robot experiments ran on a DROID-style station: a Franka arm driven by a NUC
running Polymetis, a Linux workstation that runs the policies, three ZED cameras (a
stereo wrist camera and two side cameras) and a SpaceMouse for human corrections. This
page describes what a new station needs to run collection and blind evaluation with this
repository. It does not reproduce the paper's data: human operators, the physical scene
and the room change every run (see the README).

## Software on the workstation

1. **ZED SDK 4.2** from Stereolabs, installed at `/usr/local/zed`. The robot project pins
   `pyzed` 4.2 (built against the numpy 1.x ABI); `scripts/sync_robot_env.sh` refuses to
   run against another SDK version.
2. **The robot project.** The repo root is the ML environment (numpy 2.x, no hardware
   stack). `robot/` is a second uv project that installs the same `mulligan` package
   (editable) plus DROID, `gym`, `zerorpc` and `pyzed`, with `numpy==1.26.4`:

   ```bash
   bash scripts/sync_robot_env.sh            # build / repair robot/.venv, verify imports
   uv run --project robot --frozen python -m mulligan.real.eval.manifest_eval --help
   ```

   The script downloads the `pyzed` wheel that matches SDK 4.2 into `robot/wheels/`
   (gitignored; the lock pins its sha256), syncs `robot/uv.lock`, makes sure `cv2` has
   HighGUI (the operator windows need it), and imports the eval/collection chain, which
   validates the station file below. It needs about 9 GB of disk
   ([install.md](install.md#robot-workstation-robot)). `--no-group zed` skips pyzed, but
   DROID's robot environment imports it, so collection and evaluation need the `zed` group.
3. **DROID.** The robot project pins the DROID fork at `c0c8b29`; see
   [hardware/droid_fork.md](hardware/droid_fork.md) for what the fork changes and what the
   NUC needs.
4. **SpaceMouse** for teleop and DAgger corrections: [hardware/spacemouse.md](hardware/spacemouse.md).

## Station identity (`~/.config/droid/station.env`)

DROID reads network and robot identity from a per-machine file outside the checkout,
`~/.config/droid/station.env` (override the path with `DROID_STATION_ENV_FILE`; explicit
`DROID_*` environment variables win). One `KEY=VALUE` per line; unknown keys are an
error. The keys are listed in [configs/real/station.env.example](../configs/real/station.env.example).

- The **workstation** needs `DROID_NUC_IP`, `DROID_ROBOT_IP`, `DROID_ROBOT_TYPE`
  (`panda` or `fr3`) and `DROID_ROBOT_SERIAL_NUMBER`. `mulligan.real.robot.station`
  checks this before any robot motion and refuses to run if a NUC-only secret
  (`DROID_SUDO_PASSWORD`, `DROID_UBUNTU_PRO_TOKEN`) is present on the workstation.
- The **NUC** additionally needs `DROID_SUDO_PASSWORD` (see below). A file that holds a
  secret must be `chmod 600`; the loader refuses group/other-readable secret files.

### Recommended: passwordless sudo for the controller launch

The pinned DROID fork launches the Polymetis robot and gripper servers with
`echo <DROID_SUDO_PASSWORD> | sudo -S bash launch_robot.sh`
(`droid/franka/robot.py:26-30`), which puts the password on a command line. On the NUC,
prefer a sudoers rule that allows exactly those two scripts without a password, and leave
`DROID_SUDO_PASSWORD` empty:

```bash
# on the NUC, as root: visudo -f /etc/sudoers.d/droid-controller
<nuc-user> ALL=(root) NOPASSWD: /bin/bash /path/to/droid/droid/franka/launch_robot.sh, /bin/bash /path/to/droid/droid/franka/launch_gripper.sh
```

`sudo -S` still succeeds when no password is required, so the fork works unchanged.

## Camera layout (`configs/real/station.example.yaml`)

Datasets, policies and task specs name cameras by role: `wrist_left`, `wrist_right`
(the two eyes of the ZED wrist camera), `side_1` and `side_2` (left eyes of the side
cameras). The station config maps each role to the ZED `<serial>_<eye>` key of the DROID
observation, lists the cameras that are never stored (the side cameras' right eyes), and
holds the default crop box per role in stored-frame pixels (full frames are stored at
640x480; crops are applied on the fly at train and eval time and are baked into each
trained policy's `camera_crop_boxes`).

`configs/real/station.example.yaml` holds the paper station's roles and crop boxes with
placeholder ZED serials; training and evaluating on the released datasets uses it
unchanged (only the roles and crops matter off the robot). The code reads the identical
copy shipped in the package (`mulligan/real/robot/station.example.yaml`) when
`MULLIGAN_STATION_CONFIG` is unset. A station copies it, replaces the placeholder serials
with its own cameras, and points the code at the copy:

```bash
cp configs/real/station.example.yaml ~/station.yaml     # edit the serials
export MULLIGAN_STATION_CONFIG=~/station.yaml
```

The station boxes are defaults. A task spec can replace the box for individual roles
through `camera_crop_overrides` in `mulligan/real/lifecycle/tasks.py`; the policy trainer
merges them over the station defaults when it runs with `--task`, and an explicit
`--camera-crop` wins over both. Marker and Square use the station defaults. Cable replaces
the `side_1` and `side_2` boxes with boxes fit to its workspace and keeps the
`wrist_left` default. Either way the trained policy stores its boxes in
`camera_crop_boxes`, so evaluation needs no task lookup.

Keep the role names and, if you evaluate the released policies, keep the camera poses
close to the paper's (the crops are fixed in the checkpoints). A dataset you record stores
the role -> serial mapping it was collected with in `meta/camera_role_serials.json`, and
appending to it fails if the live cabling no longer matches. The released datasets omit
this file, so they carry no camera serials.

`python -m mulligan.real.robot.view_cameras` (robot project) shows the live ZED streams
with frame-drop and corruption diagnostics and a preview of each role's default crop.

## Operator display

The operator windows (target card, session panel, optional cropped camera monitors) use
OpenCV HighGUI. Every entry point initializes HighGUI before LeRobot is imported (a Qt/xcb
deadlock otherwise); the order is pinned by `tests/real/test_operator_ui_structure.py`.
`python -m mulligan.real.operator_ui.preview --manifest <manifest>` renders a manifest's
target cards on any machine, without a robot or a display.

The collectors and `manifest_eval` read operator keys from their terminal, so run them in an
interactive terminal (or tmux/screen), not detached with `nohup`.

## Resets

DROID's blocking joint move swallows gRPC errors, so a reset can return without the arm reaching
home. Every collector and `manifest_eval` therefore reset through
`mulligan.real.collect.rollout.verified_reset`, which checks the joint error (threshold 0.15 rad)
and retries up to six times with backoff. Retry warnings
(`Reset attempt k/6: robot didn't reach target`) are normal; if all six fail, the collector stops
after saving what it has. Re-run it (it resumes), which relaunches the controller.

## Remote inference (optional)

`--inference-backend remote` runs policy inference on another GPU machine and keeps
robot control, saving and bookkeeping on the workstation. Either start
`python -m mulligan.real.eval.inference_server` there yourself and pass
`--remote-inference-url`, or let the eval start it over SSH with
`--remote-inference-host <ssh host>` (the checkout path defaults to the local working
directory).

The protocol is pickle over plain HTTP. The server unpickles a request body only after
the request's `X-Mulligan-Inference-Token` header matches its token, but that token is a
shared secret sent in the clear: anyone who holds it, or can read the traffic, can run
arbitrary code as the server's user. The client also unpickles the server's responses.
Run it only between machines you trust:

- **Auto-managed** (`--remote-inference-host`): the eval generates a token (or uses
  `--remote-inference-token` / `MULLIGAN_REMOTE_INFERENCE_TOKEN`), passes it to the
  server on the SSH session's stdin, binds the server to `127.0.0.1` on the GPU machine
  and reaches it through an SSH port forward. Nothing listens on a network interface.
- **Manual** (`--remote-inference-url`): the server needs a token and binds `127.0.0.1`
  by default. Keep that default and forward the port over SSH:

  ```bash
  # GPU machine
  export MULLIGAN_REMOTE_INFERENCE_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
  python -m mulligan.real.eval.inference_server --port 48881 --token-env MULLIGAN_REMOTE_INFERENCE_TOKEN
  # workstation
  ssh -N -L 48881:127.0.0.1:48881 <gpu host> &
  export MULLIGAN_REMOTE_INFERENCE_TOKEN=<same token>
  uv run --project robot --frozen python -m mulligan.real.eval.manifest_eval ... \
      --inference-backend remote --remote-inference-url http://127.0.0.1:48881
  ```

  `--token-env` and the environment variable keep the token out of `ps`; `--token` and
  `--remote-inference-token` also work but put it on a command line. Bind another
  `--host` only on a network you trust; the server prints a warning when it does.

The server refuses to start without a token. `--insecure-no-token` turns the check off
(it prints a warning): then anyone who can reach the port can run arbitrary code as the
server's user, so use it only on a loopback-bound server on a single-user machine.

The server handles one connection at a time and serves a single eval.
