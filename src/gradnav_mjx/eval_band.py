"""Evaluate a saved SAC checkpoint on a specific goal-distance band.

Training/eval normally samples the whole curriculum range, so a single
headline number mixes easy near goals with hard far ones. This reports
per-band success so "86% at 1-6m" can be broken down.
"""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

import mjx_solver_patch
mjx_solver_patch.apply()

from mjx_car_scene import build_car_scene_xml
from mjx_random_maps import generate_map_set
from train_sac import evaluate_sac

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="sac_policy_final.npz")
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--action-repeat", type=int, default=20)
parser.add_argument("--n-walls", type=int, default=4)
parser.add_argument("--eval-n", type=int, default=100)
parser.add_argument("--goal-cone", type=float, default=1.05)
parser.add_argument("--bands", default="1,2;2,4;4,6;1,6")
parser.add_argument("--map-seed", type=int, default=0,
                     help="MUST match the --seed the checkpoint trained with "
                          "to score on its training map; use another value for "
                          "a held-out map.")
args = parser.parse_args()

walls_list = generate_map_set(n_maps=1, n_walls=args.n_walls, base_seed=args.map_seed)[0]
walls = jnp.array(walls_list)
model = mujoco.MjModel.from_xml_path(build_car_scene_xml(walls_list, out_path="band_map.xml"))
mjx_model = mjx.put_model(model)

ckpt = np.load(args.checkpoint)
n_layers = len([k for k in ckpt.files if k.endswith("_W")])
policy_params = [(jnp.array(ckpt[f"p{i}_W"]), jnp.array(ckpt[f"p{i}_b"])) for i in range(n_layers)]

eval_jit = jax.jit(
    lambda pp, lo, hi: evaluate_sac(pp, mjx_model, walls, args.horizon, args.eval_n,
                                     lo, hi, args.action_repeat, args.goal_cone)
)

print(f"checkpoint={args.checkpoint}  map_seed={args.map_seed}  n={args.eval_n} goals/band  "
      f"horizon={args.horizon}x{args.action_repeat} steps "
      f"({args.horizon*args.action_repeat*0.002:.0f}s)\n")
print(f"{'band (m)':>12}  {'success':>8}  {'mean closest':>13}  {'collisions':>11}")
for band in args.bands.split(";"):
    lo, hi = (float(v) for v in band.split(","))
    d, c, s = eval_jit(policy_params, lo, hi)
    print(f"{lo:5.1f}-{hi:<5.1f}  {float(s)*100:7.1f}%  {float(d):12.3f}  "
          f"{int(c):5d}/{args.eval_n}")
