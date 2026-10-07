# DROID fork

The robot project (`robot/pyproject.toml`) installs DROID from a fork pinned at commit
`c0c8b29` (base: upstream DROID `33ae6a6`). The fork is a git dependency, not vendored; the NUC
runs it in Polymetis' Python 3.8 environment. Licensing: [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
This page lists what the fork changes and what a station needs for it.

## Changes relative to upstream DROID

`droid/franka/robot.py` (runs on the NUC, behind the zerorpc server):

1. **Forward kinematics for joint-space actions.** In `create_action_dict()`, joint
   position/velocity commands also get `cartesian_position` and `cartesian_velocity`
   (via FK), so every action representation is present in `action_info` whatever the
   action space. Without it, the dataset writer fails on a missing `cartesian_velocity`.
2. **`gripper_delta` -> `gripper_velocity`.** The position-gripper branch stores the
   computed gripper velocity under the same key as the velocity branch. `action_info`
   always holds `cartesian_velocity`, `cartesian_position`, `joint_velocity`,
   `joint_position`, `gripper_position`, `gripper_velocity` and `robot_state`.
3. **More robot state.** `get_robot_state()` adds the end-effector `cartesian_velocity`
   (Jacobian times joint velocities), `motor_torques_external`, the end-effector wrench
   `ee_wrench` (libfranka `O_F_ext_hat_K`) and the libfranka inertial parameters
   (`m_ee`, `f_x_cee`, `m_load`, `f_x_cload`, `m_total`, `f_x_ctotal`) when Polymetis
   provides them (see "Stock and patched Polymetis" below). The datasets store
   the first two as `observation.state.cartesian_velocity` and
   `telemetry.franka.motor_torques_external`, and the wrench as `telemetry.franka.ee_wrench`.

`droid/robot_env.py` (workstation): `step()` accepts a per-call action space and returns
the observation with `action_info` attached; `reset()` returns the observation.

`droid/misc/parameters.py` + new `droid/misc/station_env.py`: station identity (IPs,
robot type and serial, the NUC sudo password) is read from `~/.config/droid/station.env`
or `DROID_*` environment variables instead of being edited into `parameters.py` on every
machine. See [configs/real/station.env.example](../../configs/real/station.env.example).

`pyproject.toml`: the upstream hard pins (mujoco, opencv, protobuf, dm-control, ...) move
to a `full` extra so uv can resolve `droid` next to LeRobot and robosuite.

## Stock and patched Polymetis

The wrench and the inertial parameters of item 3 exist only in an unpublished patched Polymetis
`RobotState` (proto + server). `get_robot_state()` reads each of them with
`getattr(robot_state, name, None)` (`droid/franka/polymetis_state.py`) and leaves its key out when the
`RobotState` lacks it:

- With stock Polymetis the state dict has no `ee_wrench`, `m_ee`, `f_x_cee`, `m_load`,
  `f_x_cload`, `m_total` or `f_x_ctotal`. The collectors then record
  `telemetry.franka.ee_wrench` as NaN (as for any telemetry the server does not expose).
  Nothing the released policies or critics read depends on these fields: they are
  telemetry, and the released critics use `critic_extra_state: none`.
- With the patched Polymetis the dict also has these keys.

To record the wrench on a new station, add the fields to Polymetis' `RobotState` and fill
them from libfranka's `franka::RobotState` (`O_F_ext_hat_K`, `m_ee`, `F_x_Cee`, `m_load`,
`F_x_Cload`, `m_total`, `F_x_Ctotal`). `scripts/tests/check_polymetis_state_fields.py` in
the fork checks both cases without a robot.

## Packaging

The fork drops upstream DROID's git submodules `droid/fairo` and `droid/oculus_reader`
(fork commit `c0c8b29`) and packages only `droid*`, so uv and pip install it from git
without GitHub SSH keys. Neither is needed by the workstation: the NUC builds Polymetis
from its own fairo checkout, and Mulligan's teleop uses a SpaceMouse, not the Oculus
reader (clone `rail-berkeley/oculus_reader` next to the checkout if you need it).

## Deploying to the NUC

The NUC runs the Polymetis server and DROID's `FrankaRobot`. Install the fork there at the
same commit as the workstation pin, into the conda environment that the Polymetis build
creates (`polymetis-local`: Python 3.8 and torch 1.13.1 per Polymetis' `environment.yml`;
the fork needs Python >= 3.7). Following upstream DROID's NUC recipe:

```bash
conda activate polymetis-local
git clone https://github.com/ankile/droid && cd droid
git checkout c0c8b29e1fcf76e424177529d5440620234918be
pip install -e '.[full]'
# DROID's IK solver, without dependencies as upstream DROID installs it (its dependencies
# would replace the mujoco pin of the `full` extra)
pip install --no-deps dm-robotics-moma==0.5.0 dm-robotics-transformations==0.5.0 \
    dm-robotics-agentflow==0.5.0 dm-robotics-geometry==0.5.0 \
    dm-robotics-manipulation==0.5.0 dm-robotics-controllers==0.5.0
```

Install the `full` extra on the NUC. A plain `pip install -e .` in a fresh environment pulls
the newest `opencv-python`, whose `cv2.aruco` no longer has the `Dictionary_get` that
`droid/misc/parameters.py` calls, so importing `droid.franka.robot` fails with
`AttributeError: module 'cv2.aruco' has no attribute 'Dictionary_get'` (the workstation
patches this in `mulligan.real.robot.droid_compat`; the NUC has no such shim). Under
Python 3.6, pip's setuptools cannot read the fork's `pyproject.toml` and installs a package
named `UNKNOWN` with no dependencies.

Check the install without the robot. `DROID_STATION_ENV_FILE` points at a test file, so the
NUC's real station file is not read:

```bash
DROID_STATION_ENV_FILE=/tmp/station-test.env python -c "import droid.misc.parameters as p, droid.robot_ik.robot_ik_solver; print(repr(p.robot_type))"
```

Clear stale bytecode after updates (`find droid -name __pycache__ -exec rm -rf {} +`).
Create the NUC's own `~/.config/droid/station.env` (mode 600), and prefer the passwordless
sudo rule in [docs/station.md](../station.md) over storing `DROID_SUDO_PASSWORD`.

If the NUC runs older DROID code, the first `env.step()` on the workstation fails loudly
with a `KeyError` (for example a missing `gripper_velocity` or `cartesian_velocity`)
instead of recording zero-filled columns.

