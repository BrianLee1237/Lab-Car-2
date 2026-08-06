"""
generate_map_batch.py

Generates a batch of randomized MuJoCo scene files (using mujoco_map_gen.py's
building blocks) and validates that each one actually loads and steps
correctly in MuJoCo before saving it -- so you don't end up with a folder
of broken XML files partway through a long RL training run.

Usage:
    python3 generate_map_batch.py --out-dir maps --n-maps 200 --n-peds 8 --n-walls 12

This will produce maps/map_0000.xml ... maps/map_0199.xml, each with a
different random layout (different seed = different maze walls + pedestrian
placement), and print a summary of how many passed validation.
"""

import argparse
import os
import sys

import mujoco

# reuse the scene-building logic from mujoco_map_gen.py
from mujoco_map_gen import build_xml


def validate_scene(path, n_steps=50):
    """Load the scene and step it a few times to catch broken XML or
    unstable physics (e.g. bodies overlapping at spawn) before we trust it."""
    try:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
        return True, model.nbody, model.ngeom
    except Exception as e:
        return False, str(e), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="maps")
    parser.add_argument("--n-maps", type=int, default=100)
    parser.add_argument("--n-peds", type=int, default=8)
    parser.add_argument("--n-walls", type=int, default=12)
    parser.add_argument("--size", type=float, default=10.0)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_ok = 0
    n_failed = 0
    failed_seeds = []

    for i in range(args.n_maps):
        seed = args.start_seed + i
        out_path = os.path.join(args.out_dir, f"map_{i:04d}.xml")
        build_xml(seed, args.n_peds, args.n_walls, args.size, out_path)

        ok, info, ngeom = validate_scene(out_path)
        if ok:
            n_ok += 1
        else:
            n_failed += 1
            failed_seeds.append((seed, info))
            print(f"  [FAILED] seed={seed} -> {info}")

        if (i + 1) % 25 == 0 or (i + 1) == args.n_maps:
            print(f"Progress: {i + 1}/{args.n_maps} generated ({n_ok} valid, {n_failed} failed)")

    print()
    print(f"Done. {n_ok}/{args.n_maps} maps valid, saved in '{args.out_dir}/'")
    if failed_seeds:
        print(f"{n_failed} maps failed validation (seeds: {[s for s, _ in failed_seeds]})")
        sys.exit(1)


if __name__ == "__main__":
    main()
