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

from train_sac import evaluate_sac, make_env

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
parser.add_argument("--room", action="store_true")
parser.add_argument("--room-size", type=float, default=8.0)
parser.add_argument("--n-inner", type=int, default=6)
parser.add_argument("--n-humans", type=int, default=6)
parser.add_argument("--spawn-half", type=float, default=6.5)
parser.add_argument("--max-dist", type=float, default=10.0)
args = parser.parse_args()

# built by the same function training uses, so an evaluation can never
# silently score a policy on a different world than it was trained in
env = make_env(args.map_seed, room=args.room, room_size=args.room_size,
                n_inner=args.n_inner, n_humans=args.n_humans,
                n_walls=args.n_walls, spawn_half=args.spawn_half,
                max_dist=args.max_dist)
mjx_model, walls = env["mjx_model"], env["walls"]

ckpt = np.load(args.checkpoint)
n_layers = len([k for k in ckpt.files if k.endswith("_W")])
policy_params = [(jnp.array(ckpt[f"p{i}_W"]), jnp.array(ckpt[f"p{i}_b"])) for i in range(n_layers)]

eval_jit = jax.jit(
    lambda pp, lo, hi: evaluate_sac(pp, mjx_model, walls, args.horizon, args.eval_n,
                                     lo, hi, args.action_repeat, args.goal_cone,
                                     env["goal_bound"], env["ped_params"],
                                     env["ped_z"])
)

print(f"checkpoint={args.checkpoint}  map_seed={args.map_seed}  "
      f"{len(env['walls_list'])} walls / {len(env['humans_list'])} people  "
      f"n={args.eval_n} goals/band  "
      f"horizon={args.horizon}x{args.action_repeat} steps "
      f"({args.horizon*args.action_repeat*0.002:.0f}s)\n")
print(f"{'band (m)':>12}  {'success':>8}  {'mean closest':>13}  "
      f"{'wall hits':>10}  {'ped hits':>9}")
for band in args.bands.split(";"):
    lo, hi = (float(v) for v in band.split(","))
    d, wh, ph, s = eval_jit(policy_params, lo, hi)
    print(f"{lo:5.1f}-{hi:<5.1f}  {float(s)*100:7.1f}%  {float(d):12.3f}  "
          f"{int(wh):4d}/{args.eval_n}  {int(ph):4d}/{args.eval_n}")
