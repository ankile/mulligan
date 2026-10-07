"""Generate STL files for the square nut and peg from robosuite's NutAssemblySquare task.

The dimensions below are copied from the MuJoCo XML primitive definitions in the
robosuite package and written as 3D-printable STL files at 1:1 scale in millimeters.

Source files, in robosuite at the git commit pinned in pyproject.toml (85abee2, which
reports version 1.5.2):
  - Square nut: robosuite/models/assets/objects/square-nut.xml (the five box geoms of body "object")
  - Square peg: robosuite/models/assets/arenas/pegs_arena.xml (the box geom of body "peg1")

Usage, from the repository root:
    uv run --with numpy-stl --with triangle python hardware/square_nut/generate_nut_peg_stl.py

``numpy-stl`` and ``triangle`` are not project dependencies (not in ``uv.lock``);
``uv run --with`` adds them for this one run. Outside the project environment,
``uv run --no-project --with numpy --with numpy-stl --with triangle python ...``
works too.

Writes square_nut.stl, square_peg.stl, square_peg_with_mount.stl and
square_peg_with_center_pilot_hole.stl to the current directory and prints their
dimensions.
"""

import numpy as np
import triangle as triangle_lib
from stl import mesh

# All dimensions in mm (converted from MuJoCo meters, half-extents → full extents)

# Square nut: 5 overlapping boxes
# Each entry: (position_xyz_mm, full_size_xyz_mm)
NUT_BOXES = [
    # Left arm
    ((-33.25, 0.0, 0.0), (21.0, 87.5, 20.0)),
    # Top arm
    ((0.0, 33.25, 0.0), (62.5, 21.0, 20.0)),
    # Bottom arm
    ((0.0, -33.25, 0.0), (62.5, 21.0, 20.0)),
    # Right arm
    ((33.25, 0.0, 0.0), (21.0, 87.5, 20.0)),
    # Handle tab
    ((54.0, 0.0, 0.0), (50.5, 31.75, 20.0)),
]

# Square peg: single box
PEG_SIZE = (32.0, 32.0, 200.0)

# Square peg with mounting base.
INCH_MM = 25.4
MOUNT_PLATE_SIZE = (3.0 * INCH_MM, 3.0 * INCH_MM, 5.0)
MOUNT_HOLE_SPACING = 2.0 * INCH_MM
BOLT_CLEARANCE_DIAMETER = 6.8
BOLT_HEAD_COUNTERBORE_DIAMETER = 13.0
COUNTERBORE_DEPTH = 3.0
HOLE_SEGMENTS = 96

# Centered bottom pilot hole for a 1/4-20 bolt. A standard 1/4-20 tap drill is
# #7 / 0.201 in / 5.1 mm; 5.3 mm gives a little extra clearance for printed
# plastic tolerances while still being undersized relative to a 6.35 mm bolt.
CENTER_PILOT_HOLE_DIAMETER = 5.3
CENTER_PILOT_HOLE_DEPTH = 1.0 * INCH_MM
CENTER_PILOT_COUNTERSINK_DEPTH = 2.0
CENTER_PILOT_COUNTERSINK_DIAMETER = CENTER_PILOT_HOLE_DIAMETER + 4.0


def box_triangles(center, size):
    """Generate 12 triangles (2 per face) for an axis-aligned box.

    Args:
        center: (cx, cy, cz) center position in mm
        size: (sx, sy, sz) full dimensions in mm

    Returns:
        numpy array of shape (12, 3, 3) — 12 triangles, 3 vertices each, 3 coords each
    """
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2

    # 8 corners of the box
    corners = np.array(
        [
            [cx - hx, cy - hy, cz - hz],  # 0: left  bottom back
            [cx + hx, cy - hy, cz - hz],  # 1: right bottom back
            [cx + hx, cy + hy, cz - hz],  # 2: right top    back
            [cx - hx, cy + hy, cz - hz],  # 3: left  top    back
            [cx - hx, cy - hy, cz + hz],  # 4: left  bottom front
            [cx + hx, cy - hy, cz + hz],  # 5: right bottom front
            [cx + hx, cy + hy, cz + hz],  # 6: right top    front
            [cx - hx, cy + hy, cz + hz],  # 7: left  top    front
        ]
    )

    # 12 triangles (2 per face), vertex indices with outward-facing normals
    faces = np.array(
        [
            # Back face (z-)
            [0, 2, 1],
            [0, 3, 2],
            # Front face (z+)
            [4, 5, 6],
            [4, 6, 7],
            # Bottom face (y-)
            [0, 1, 5],
            [0, 5, 4],
            # Top face (y+)
            [3, 6, 2],
            [3, 7, 6],
            # Left face (x-)
            [0, 4, 7],
            [0, 7, 3],
            # Right face (x+)
            [1, 2, 6],
            [1, 6, 5],
        ]
    )

    triangles = corners[faces]  # (12, 3, 3)
    return triangles


def create_stl_mesh(triangles_list):
    """Create an STL mesh from a list of triangle arrays.

    Args:
        triangles_list: list of arrays, each shape (N, 3, 3)

    Returns:
        stl.mesh.Mesh object
    """
    all_triangles = np.concatenate(triangles_list, axis=0)
    n = len(all_triangles)

    stl_mesh = mesh.Mesh(np.zeros(n, dtype=mesh.Mesh.dtype))
    for i, tri in enumerate(all_triangles):
        stl_mesh.vectors[i] = tri

    return stl_mesh


def rectangle_loop(width, height):
    hx, hy = width / 2.0, height / 2.0
    return np.array(
        [
            [-hx, -hy],
            [hx, -hy],
            [hx, hy],
            [-hx, hy],
        ],
        dtype=float,
    )


def circle_loop(center, radius, segments=HOLE_SEGMENTS):
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cx, cy = center
    return np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))


def triangulate_region_2d(outer_loop, hole_loops):
    """Triangulate a planar region with constrained hole boundaries."""
    vertices = []
    segments = []
    holes = []

    def add_loop(loop):
        start = len(vertices)
        vertices.extend(loop.tolist())
        n = len(loop)
        for idx in range(n):
            segments.append((start + idx, start + ((idx + 1) % n)))
        return start

    add_loop(outer_loop)
    for loop in hole_loops:
        add_loop(loop)
        holes.append(np.mean(loop, axis=0))

    triangulation_input = {
        "vertices": np.asarray(vertices, dtype=float),
        "segments": np.asarray(segments, dtype=int),
    }
    if holes:
        triangulation_input["holes"] = np.asarray(holes, dtype=float)

    result = triangle_lib.triangulate(triangulation_input, "p")
    if "triangles" not in result:
        raise RuntimeError("Failed to triangulate mounting plate")
    return result["vertices"], result["triangles"]


def surface_triangles(vertices_2d, faces, z, upward=True):
    triangles = []
    for face in faces:
        ordered = face if upward else face[::-1]
        triangles.append(np.column_stack((vertices_2d[ordered], np.full(3, z))))
    return np.asarray(triangles)


def loop_side_triangles(loop, z_min, z_max):
    """Create vertical side-wall triangles along a 2D loop."""
    triangles = []
    for idx in range(len(loop)):
        p0 = loop[idx]
        p1 = loop[(idx + 1) % len(loop)]
        bottom0 = np.array([p0[0], p0[1], z_min])
        bottom1 = np.array([p1[0], p1[1], z_min])
        top0 = np.array([p0[0], p0[1], z_max])
        top1 = np.array([p1[0], p1[1], z_max])
        triangles.append(np.array([bottom0, bottom1, top1]))
        triangles.append(np.array([bottom0, top1, top0]))

    return np.asarray(triangles)


def loop_pair_side_triangles(lower_loop, upper_loop, z_min, z_max):
    """Create side-wall triangles between two equal-length 2D loops."""
    if len(lower_loop) != len(upper_loop):
        raise ValueError("lower_loop and upper_loop must have the same number of points")

    triangles = []
    for idx in range(len(lower_loop)):
        lower0 = lower_loop[idx]
        lower1 = lower_loop[(idx + 1) % len(lower_loop)]
        upper0 = upper_loop[idx]
        upper1 = upper_loop[(idx + 1) % len(upper_loop)]
        bottom0 = np.array([lower0[0], lower0[1], z_min])
        bottom1 = np.array([lower1[0], lower1[1], z_min])
        top0 = np.array([upper0[0], upper0[1], z_max])
        top1 = np.array([upper1[0], upper1[1], z_max])
        triangles.append(np.array([bottom0, bottom1, top1]))
        triangles.append(np.array([bottom0, top1, top0]))

    return np.asarray(triangles)


def mounted_peg_triangles():
    """Generate the square peg with a counterbored mounting base."""
    plate_x, plate_y, plate_z = MOUNT_PLATE_SIZE
    outer_loop = rectangle_loop(plate_x, plate_y)
    peg_loop = rectangle_loop(PEG_SIZE[0], PEG_SIZE[1])
    hole_centers = [
        (-MOUNT_HOLE_SPACING / 2.0, 0.0),
        (MOUNT_HOLE_SPACING / 2.0, 0.0),
    ]
    shaft_radius = BOLT_CLEARANCE_DIAMETER / 2.0
    head_radius = BOLT_HEAD_COUNTERBORE_DIAMETER / 2.0
    counterbore_floor_z = plate_z - COUNTERBORE_DEPTH

    shaft_holes = [circle_loop(center, shaft_radius) for center in hole_centers]
    head_holes = [circle_loop(center, head_radius) for center in hole_centers]

    bottom_vertices, bottom_faces = triangulate_region_2d(outer_loop, shaft_holes)
    top_vertices, top_faces = triangulate_region_2d(outer_loop, [*head_holes, peg_loop])
    peg_top_vertices, peg_top_faces = triangulate_region_2d(peg_loop, [])

    triangles = [
        surface_triangles(bottom_vertices, bottom_faces, 0.0, upward=False),
        surface_triangles(top_vertices, top_faces, plate_z, upward=True),
        loop_side_triangles(outer_loop, 0.0, plate_z),
        loop_side_triangles(peg_loop, plate_z, plate_z + PEG_SIZE[2]),
        surface_triangles(peg_top_vertices, peg_top_faces, plate_z + PEG_SIZE[2], upward=True),
    ]

    for shaft_hole, head_hole in zip(shaft_holes, head_holes):
        triangles.append(loop_side_triangles(shaft_hole, 0.0, counterbore_floor_z))
        triangles.append(loop_side_triangles(head_hole, counterbore_floor_z, plate_z))

        floor_triangles = []
        for idx in range(HOLE_SEGMENTS):
            inner0 = shaft_hole[idx]
            inner1 = shaft_hole[(idx + 1) % HOLE_SEGMENTS]
            outer0 = head_hole[idx]
            outer1 = head_hole[(idx + 1) % HOLE_SEGMENTS]
            floor_triangles.append(
                np.array(
                    [
                        [inner0[0], inner0[1], counterbore_floor_z],
                        [outer1[0], outer1[1], counterbore_floor_z],
                        [outer0[0], outer0[1], counterbore_floor_z],
                    ]
                )
            )
            floor_triangles.append(
                np.array(
                    [
                        [inner0[0], inner0[1], counterbore_floor_z],
                        [inner1[0], inner1[1], counterbore_floor_z],
                        [outer1[0], outer1[1], counterbore_floor_z],
                    ]
                )
            )
        triangles.append(np.asarray(floor_triangles))

    return triangles


def center_pilot_hole_peg_triangles():
    """Generate the square peg with a centered blind pilot hole from the bottom."""
    peg_loop = rectangle_loop(PEG_SIZE[0], PEG_SIZE[1])
    hole_loop = circle_loop((0.0, 0.0), CENTER_PILOT_HOLE_DIAMETER / 2.0)
    countersink_loop = circle_loop((0.0, 0.0), CENTER_PILOT_COUNTERSINK_DIAMETER / 2.0)
    z_min = -PEG_SIZE[2] / 2.0
    z_max = PEG_SIZE[2] / 2.0
    countersink_top_z = z_min + CENTER_PILOT_COUNTERSINK_DEPTH
    hole_top_z = countersink_top_z + CENTER_PILOT_HOLE_DEPTH

    bottom_vertices, bottom_faces = triangulate_region_2d(peg_loop, [countersink_loop])
    top_vertices, top_faces = triangulate_region_2d(peg_loop, [])
    hole_cap_vertices, hole_cap_faces = triangulate_region_2d(hole_loop, [])

    return [
        surface_triangles(bottom_vertices, bottom_faces, z_min, upward=False),
        surface_triangles(top_vertices, top_faces, z_max, upward=True),
        loop_side_triangles(peg_loop, z_min, z_max),
        loop_pair_side_triangles(countersink_loop, hole_loop, z_min, countersink_top_z),
        loop_side_triangles(hole_loop, countersink_top_z, hole_top_z),
        surface_triangles(hole_cap_vertices, hole_cap_faces, hole_top_z, upward=False),
    ]


def main():
    # Generate square nut mesh (5 boxes combined)
    nut_triangles = [box_triangles(pos, size) for pos, size in NUT_BOXES]
    nut_mesh = create_stl_mesh(nut_triangles)
    nut_mesh.save("square_nut.stl")

    # Generate square peg mesh (single box)
    peg_triangles = [box_triangles((0, 0, 0), PEG_SIZE)]
    peg_mesh = create_stl_mesh(peg_triangles)
    peg_mesh.save("square_peg.stl")

    mounted_peg_mesh = create_stl_mesh(mounted_peg_triangles())
    mounted_peg_mesh.save("square_peg_with_mount.stl")

    center_pilot_hole_peg_mesh = create_stl_mesh(center_pilot_hole_peg_triangles())
    center_pilot_hole_peg_mesh.save("square_peg_with_center_pilot_hole.stl")

    # Print summary
    print("Generated STL files (dimensions in mm):")
    print()

    print("square_nut.stl")
    nut_min = nut_mesh.vectors.reshape(-1, 3).min(axis=0)
    nut_max = nut_mesh.vectors.reshape(-1, 3).max(axis=0)
    nut_extent = nut_max - nut_min
    print(f"  Bounding box: {nut_extent[0]:.1f} x {nut_extent[1]:.1f} x {nut_extent[2]:.1f} mm")
    print(f"  Min corner:   ({nut_min[0]:.2f}, {nut_min[1]:.2f}, {nut_min[2]:.2f})")
    print(f"  Max corner:   ({nut_max[0]:.2f}, {nut_max[1]:.2f}, {nut_max[2]:.2f})")

    # Inner hole: gap between left/right arms in X, top/bottom arms in Y
    inner_x = (33.25 - 21.0 / 2) - (-33.25 + 21.0 / 2)
    inner_y = (33.25 - 21.0 / 2) - (-33.25 + 21.0 / 2)
    print(f"  Square hole:  {inner_x:.1f} x {inner_y:.1f} mm")
    print(f"  Triangles:    {len(nut_mesh.vectors)}")
    print()

    print("square_peg.stl")
    print(f"  Size:         {PEG_SIZE[0]:.1f} x {PEG_SIZE[1]:.1f} x {PEG_SIZE[2]:.1f} mm")
    print(f"  Triangles:    {len(peg_mesh.vectors)}")
    print()

    print("square_peg_with_mount.stl")
    mounted_min = mounted_peg_mesh.vectors.reshape(-1, 3).min(axis=0)
    mounted_max = mounted_peg_mesh.vectors.reshape(-1, 3).max(axis=0)
    mounted_extent = mounted_max - mounted_min
    print(
        f"  Bounding box: {mounted_extent[0]:.1f} x {mounted_extent[1]:.1f} x "
        f"{mounted_extent[2]:.1f} mm"
    )
    print(
        f"  Plate:        {MOUNT_PLATE_SIZE[0]:.1f} x {MOUNT_PLATE_SIZE[1]:.1f} x "
        f"{MOUNT_PLATE_SIZE[2]:.1f} mm"
    )
    print(f"  Hole spacing: {MOUNT_HOLE_SPACING:.1f} mm")
    print(f"  Bolt holes:   {BOLT_CLEARANCE_DIAMETER:.1f} mm through")
    print(
        f"  Counterbore:  {BOLT_HEAD_COUNTERBORE_DIAMETER:.1f} mm diameter x "
        f"{COUNTERBORE_DEPTH:.1f} mm deep"
    )
    print(f"  Triangles:    {len(mounted_peg_mesh.vectors)}")
    print()

    print("square_peg_with_center_pilot_hole.stl")
    pilot_min = center_pilot_hole_peg_mesh.vectors.reshape(-1, 3).min(axis=0)
    pilot_max = center_pilot_hole_peg_mesh.vectors.reshape(-1, 3).max(axis=0)
    pilot_extent = pilot_max - pilot_min
    print(
        f"  Bounding box: {pilot_extent[0]:.1f} x {pilot_extent[1]:.1f} x {pilot_extent[2]:.1f} mm"
    )
    print(f"  Pilot hole:   {CENTER_PILOT_HOLE_DIAMETER:.1f} mm diameter")
    print(f"  Pilot depth:  {CENTER_PILOT_HOLE_DEPTH:.1f} mm after countersink")
    print(
        f"  Countersink:  {CENTER_PILOT_COUNTERSINK_DIAMETER:.1f} mm diameter x "
        f"{CENTER_PILOT_COUNTERSINK_DEPTH:.1f} mm deep"
    )
    print(f"  Triangles:    {len(center_pilot_hole_peg_mesh.vectors)}")
    print()

    clearance = (inner_x - PEG_SIZE[0]) / 2
    print(f"Peg-to-hole clearance: {clearance:.2f} mm per side")


if __name__ == "__main__":
    main()
