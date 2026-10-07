# Task objects and scenes

The three real tasks share one table layout and one way of placing objects. This page describes the
objects of each task, where they go on the table, what the start-state sampler varies, and what counts as
success. Sources: the task registry [`mulligan/real/lifecycle/tasks.py`](../../mulligan/real/lifecycle/tasks.py),
the stage-label specs in [`mulligan/real/stage_specs/`](../../mulligan/real/stage_specs/), the locked
manifests in [`data/real/manifests/`](../../data/real/manifests/) and the paper's task appendix. The paper
gives centimeters; the code and the operator cards use inches (positions) and meters (manifests).

| Task | Name | Objects | Sampled start (paper) | Success | Eval cap |
|---|---|---|---|---|---|
| Insert Marker | `marker_d2` | whiteboard marker, red holder | pen 15.2 x 30.5 cm and 360° yaw; holder 5.1 x 10.2 cm | marker released and fully seated (S7) | 800 steps (53 s) |
| Thread Nut | `square_d2` | square nut, square peg | nut 25.4 x 45.7 cm and 360° yaw; peg 5.1 x 10.2 cm | nut released and fully seated (S7) | 600 steps (40 s) |
| Route Cable | `routing_d2` | rope, two clips | rope 40.6 cm in x; each clip 40.6 x 22.9 cm | both clips seated (S10) | 850 steps (57 s) |

Caps are in environment steps at 15 Hz and apply to evaluation episodes; collection episodes end when the
operator stops them.

## Table and coordinates

- The tabletop is a breadboard with a 1-inch hole lattice. Printed placement dots sit halfway between
  holes, at half-inch offsets in both axes. Tape marks on the table fix the origin. The side-camera
  calibration in `paper/data/real/side1_operator_frame_calibration.json` is fit to this lattice.
- Positions are in the operator frame: +x forward from the operator, +y to the operator's left. Yaw is the
  in-plane rotation; 0 means the object's own +x axis points forward.
- Holes are at integer-inch coordinates and dots at half-integer coordinates. The Marker holder grid is at
  integer inches (on holes); the Nut peg and the Cable clips are on dots.
- Before each episode the operator places the objects from a per-episode card that shows the sampled
  poses on the allowed grid. Render the cards of any manifest without a robot:

  ```bash
  uv run python -m mulligan.real.operator_ui.preview \
    --manifest data/real/manifests/square_d2/r05/square_d2_r5_eval_heldout_independent_sobol.json \
    --idx 0 --out-dir /tmp/cards
  ```

- Cameras: the Marker and Nut policies read `side_1` and `wrist_left`; the Cable policy reads `side_1`,
  `side_2` and `wrist_left`, because the rope spans the whole board and the two opposing side views
  cover it end to end. Camera roles are set up as in [station.md](../station.md).

## Insert Marker (`marker_d2`)

Objects: a whiteboard marker (the "pen" in the code) and a red holder with holes. The robot grasps the
marker from the table and inserts it into a hole of the holder. The clearance at the holder mouth is about
1 mm (paper, task appendix).

Placement and randomization:

- Marker: lies free on the table. `pen_x` in [-3, 3] in, `pen_y` in [-6, 6] in, `pen_yaw` over the full
  circle, all continuous.
- Holder: moved every episode to one of 15 grid points, `holder_x` in {6, 7, 8} in and `holder_y` in
  {-4, -3, -2, -1, 0} in. It is sampled jointly with the marker and snapped to the grid.
- The sampler treats this as a 5-D space (marker x, y, yaw and holder x, y). The hardest starts put the
  marker at the low end of its x range, closest to the robot base, pointing toward the robot.

Success (S7): the gripper opens, the arm moves away, and the marker stays fully seated in the holder,
inserted to full depth and flush with it. A marker seated but still held at the end is S6; a released
partial insertion the holder retains is S5. The full S0-S7 ladder is in
[`mulligan/real/stage_specs/marker_d2.py`](../../mulligan/real/stage_specs/marker_d2.py).

Parts: [`hardware/marker_holder/marker_holder.stl`](../../hardware/marker_holder/marker_holder.stl) is
the printed holder, in millimeters. The body is 100.6 x 30.9 x 26.0 mm, with four 18.9 mm blind holes
25 mm deep at a 23.9 mm pitch. Three 5.08 mm (0.2 in) pegs, 12.7 mm long and 1 in apart, stand under the
body and seat in the breadboard holes. The release has no part number for the marker; the 18.9 mm holes
leave about 1 mm of clearance around it at the mouth. The stage-labeling prompts tell the model to look
for a red holder, so a holder of another color needs a matching prompt change before using the labeler.

## Thread Nut (`square_d2`)

Objects: a white square nut with a handle and a vertical square peg. In the stage-label prompts the peg
stands on a white base with an orange support. The robot grasps the nut and lowers it over the peg until it
rests at the base.

Placement and randomization:

- Nut: lies free on the table. `nut_x` in [-5, 5] in, `nut_y` in [-9, 9] in, `nut_yaw` over the full
  circle, all continuous.
- Peg: moved every episode to one of 15 dots, `peg_x` in {9.5, 10.5, 11.5} in and `peg_y` in {-4.5, -3.5,
  -2.5, -1.5, -0.5} in. It is sampled jointly with the nut and snapped to the dot grid. The peg range
  starts at 9.5 in so that the nut, whose rotated footprint reaches about 3.2 in from its center at the
  handle tip, cannot overlap the peg at setup.
- The sampler treats this as a 5-D space (nut x, y, yaw and peg x, y). The hardest starts put the nut at
  the low end of its x range, closest to the robot base, pointing toward the robot; from there the nut has
  to be repositioned without a grasp before it can be grasped.

Success (S7): the gripper releases and the nut stays fully seated at the base of the peg. A visible peg or
base alone is not success; the nut body has to be on the peg at the base after release. A nut seated but
still held is S6; a released nut hanging on the peg above the base is S5. The full ladder is in
[`mulligan/real/stage_specs/square_d2.py`](../../mulligan/real/stage_specs/square_d2.py).

Parts: [`hardware/square_nut/generate_nut_peg_stl.py`](../../hardware/square_nut/generate_nut_peg_stl.py)
writes STL files for 3D printing the nut and the peg, using the dimensions of robosuite's simulated
square nut and peg at 1:1 scale:

```bash
uv run --with numpy-stl --with triangle python hardware/square_nut/generate_nut_peg_stl.py
```

| File | Geometry (from the script) |
|---|---|
| `square_nut.stl` | 123.0 x 87.5 x 20.0 mm overall (a square frame plus a handle tab), 45.5 x 45.5 mm square hole |
| `square_peg.stl` | 32 x 32 x 200 mm |
| `square_peg_with_mount.stl` | the peg on a 3 x 3 in, 5 mm plate with two 6.8 mm counterbored bolt holes 2 in apart |
| `square_peg_with_center_pilot_hole.stl` | the peg with a 5.3 mm blind pilot hole, 1 in deep, for a 1/4-20 bolt from below |

The peg-to-hole clearance is 6.75 mm per side. The script prints these values when it runs.

## Route Cable (`routing_d2`)

Objects: one flexible rope and two clips mounted on the breadboard. The robot presses the rope into the
mouth of the first clip, then the second, threading it left to right across the board. It may let go and
re-grasp the rope between the clips.

Placement and randomization:

- Rope: lies across the board along y. Its free end is placed at `rope_x` in [-6.5, 9.5] in, continuous.
- Clips: one on the left half (`clip_left_y` in [0.5, 9.5] in) and one on the right half (`clip_right_y`
  in [-9.5, -0.5] in), each with x in [-6.5, 9.5] in, snapped to the dots (17 x 10 positions per clip).
- Each clip also takes one of three orientations: -45°, -90° or -135°, where 0° is the clip's long axis
  along +x. All three turn the clip mouth toward -y, the direction the rope is threaded.
- Clip centers are at least 2 in apart. Each clip is fastened through the holes around it, and two closer
  clips would need the same holes; the manifest builder rejects such draws and the collection loader
  checks it.
- The sampler covers 7 dimensions: `rope_x` plus x, y and orientation of each clip.

Success: a clip is seated when the rope is pressed under or into its mouth and stays there after the
gripper moves away or opens. The episode score is the number of seated clips (0-2), and the binary success
is both clips seated (S10). The operator marks the first seat during the episode with the sub-goal key
(`g` or numpad `3`). The paper's headline metric for this task is clip success rate, the mean score
divided by two. The ladder is in [`mulligan/real/stage_specs/`](../../mulligan/real/stage_specs/).

Parts: the release has no part number for the rope. Each clip is printed in two pieces that join with a
sliding dovetail: a 41.4 x 41.4 mm base plate with four 5.08 mm (0.2 in) pins, 1 in long on a 1 in
square, that seat in the breadboard holes around a dot, and a top that holds the rope. The straight clip
gives the -90° orientation; the 45° clip, turned either way on the same pins, gives -45° and -135°. The
STLs in [`hardware/cable_clip/`](../../hardware/cable_clip/) are in millimeters, in the orientation they
were printed (the bases on their side, so the pins print horizontally):

| File | Piece |
|---|---|
| `cable_clip_base_straight.stl` | base for the straight clip |
| `cable_clip_top_straight.stl` | straight clip top |
| `cable_clip_base_45deg.stl` | base for the 45° clip |
| `cable_clip_top_45deg.stl` | 45° clip top |

They were printed in PLA on a Bambu Lab P1S with 0.2 mm layers, 8 walls, 80% infill and supports.
