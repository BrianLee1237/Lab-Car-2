"""
mujoco_map_gen.py

Generates randomized MuJoCo scenes for RL training: a bounded arena with
procedurally placed wall segments (maze-like corridors), a simple car body
(matching the bicycle-model car used elsewhere in this repo), and a
configurable number of randomly-placed "pedestrian" capsule bodies.

This is meant as a starting point for RL training environments (e.g. via
gymnasium + mujoco), separate from the Gazebo setup used for the classical
controllers (PID / Stanley / Pure Pursuit).

Usage:
    python3 mujoco_map_gen.py --out scene.xml --seed 0 --n-peds 8

Then load it with:
    import mujoco
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
"""

import argparse
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom


def make_arena(size=10.0, wall_height=1.0, wall_thickness=0.1):
    """Return the outer boundary walls as a list of (x, y, w, h) rects
    in local ground-plane coordinates (half-extents used later)."""
    s = size
    t = wall_thickness
    return [
        # (cx, cy, half_x, half_y)  -- N, S, E, W walls
        (0, s, s + t, t),
        (0, -s, s + t, t),
        (s, 0, t, s + t),
        (-s, 0, t, s + t),
    ]


def make_maze_walls(size, n_walls, rng, wall_height=1.0, wall_thickness=0.1,
                     min_len=1.5, max_len=4.0):
    """Randomly place interior wall segments to create maze-like corridors.
    Each wall is either horizontal or vertical, placed away from the arena
    edges so paths remain traversable."""
    walls = []
    margin = 1.5
    for _ in range(n_walls):
        horizontal = rng.random() < 0.5
        length = rng.uniform(min_len, max_len)
        cx = rng.uniform(-size + margin, size - margin)
        cy = rng.uniform(-size + margin, size - margin)
        if horizontal:
            walls.append((cx, cy, length / 2, wall_thickness))
        else:
            walls.append((cx, cy, wall_thickness, length / 2))
    return walls


def make_pedestrians(n_peds, size, rng, exclude_radius=1.0, exclude_points=None):
    """Randomly place pedestrian capsule bodies, avoiding the robot's
    spawn point and each other (coarse rejection sampling)."""
    if exclude_points is None:
        exclude_points = [(0, 0)]
    margin = 1.0
    peds = []
    attempts = 0
    while len(peds) < n_peds and attempts < n_peds * 50:
        attempts += 1
        x = rng.uniform(-size + margin, size - margin)
        y = rng.uniform(-size + margin, size - margin)
        too_close = any(
            (x - ex) ** 2 + (y - ey) ** 2 < exclude_radius ** 2
            for ex, ey in exclude_points + [(p[0], p[1]) for p in peds]
        )
        if not too_close:
            peds.append((x, y))
    return peds


def build_xml(seed, n_peds, n_walls, size, out_path):
    rng = random.Random(seed)

    mujoco_el = ET.Element("mujoco", model="randomized_arena")
    ET.SubElement(mujoco_el, "option", timestep="0.002", gravity="0 0 -9.81")

    default = ET.SubElement(mujoco_el, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", friction="0.3 0.05 0.05")

    asset = ET.SubElement(mujoco_el, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.2 0.2 0.2", rgb2="0.3 0.3 0.3", width="512", height="512")
    ET.SubElement(asset, "material", name="grid_mat", texture="grid",
                  texrepeat="10 10", reflectance="0.1")

    worldbody = ET.SubElement(mujoco_el, "worldbody")
    ET.SubElement(worldbody, "light", diffuse=".8 .8 .8", pos="0 0 6", dir="0 0 -1")
    ET.SubElement(worldbody, "geom", name="floor", type="plane",
                  size=f"{size} {size} 0.1", material="grid_mat")

    # --- outer boundary walls ---
    for i, (cx, cy, hx, hy) in enumerate(make_arena(size)):
        ET.SubElement(worldbody, "geom", name=f"boundary_{i}", type="box",
                      pos=f"{cx} {cy} 0.5", size=f"{hx} {hy} 0.5",
                      rgba="0.5 0.5 0.55 1")

    # --- randomized interior maze walls ---
    for i, (cx, cy, hx, hy) in enumerate(make_maze_walls(size, n_walls, rng)):
        ET.SubElement(worldbody, "geom", name=f"wall_{i}", type="box",
                      pos=f"{cx} {cy} 0.5", size=f"{hx} {hy} 0.5",
                      rgba="0.6 0.55 0.5 1")

    # --- car body (simple bicycle-model box + free joint, spawned at origin) ---
    car = ET.SubElement(worldbody, "body", name="car", pos="0 0 0.15")
    ET.SubElement(car, "freejoint", name="car_free")
    ET.SubElement(car, "geom", name="car_body", type="box", size="0.25 0.13 0.08",
                  rgba="0.8 0.1 0.1 1", mass="3.5")
    ET.SubElement(car, "geom", name="car_nose", type="box", pos="0.25 0 0",
                  size="0.03 0.1 0.06", rgba="0.2 0.2 0.2 1", mass="0.1")

    # --- randomized pedestrians (simple capsule bodies with free joints) ---
    pedestrian_positions = make_pedestrians(n_peds, size, rng, exclude_points=[(0, 0)])
    for i, (x, y) in enumerate(pedestrian_positions):
        ped = ET.SubElement(worldbody, "body", name=f"ped_{i}", pos=f"{x} {y} 0.9")
        ET.SubElement(ped, "freejoint", name=f"ped_{i}_free")
        ET.SubElement(ped, "geom", name=f"ped_{i}_geom", type="capsule",
                      fromto="0 0 -0.9 0 0 0.2", size="0.2",
                      rgba=f"{rng.uniform(0.3,0.9):.2f} {rng.uniform(0.3,0.9):.2f} {rng.uniform(0.3,0.9):.2f} 1",
                      mass="60")

    # pretty-print and write
    rough = ET.tostring(mujoco_el, "utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    with open(out_path, "w") as f:
        f.write(pretty)

    return out_path, len(pedestrian_positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="scene.xml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-peds", type=int, default=8)
    parser.add_argument("--n-walls", type=int, default=12)
    parser.add_argument("--size", type=float, default=10.0)
    args = parser.parse_args()

    out_path, n_placed = build_xml(args.seed, args.n_peds, args.n_walls, args.size, args.out)
    print(f"Wrote {out_path} (seed={args.seed}, walls={args.n_walls}, pedestrians placed={n_placed}/{args.n_peds})")


if __name__ == "__main__":
    main()
