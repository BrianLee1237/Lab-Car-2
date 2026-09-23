"""Quick trajectory trace for a trained SAC policy checkpoint."""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

import mjx_solver_patch
mjx_solver_patch.apply()

from mjx_car_scene import STEER_RANGE
from jax_sac_networks import deterministic_action
from train_diffmjx_final import get_car_xy_heading, QVEL_CLAMP, wall_distances
from train_sac import (build_obs_sac, make_env, N_LIDAR, _peds_at, _move_peds,
                       reset_data_at, CAR_RADIUS, SUCCESS_DIST)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="sac_policy_best.npz")
parser.add_argument("--horizon", type=int, default=150)
parser.add_argument("--action-repeat", type=int, default=20)
parser.add_argument("--n-walls", type=int, default=4)
parser.add_argument("--map-seed", type=int, default=0)
parser.add_argument("--room", action="store_true")
parser.add_argument("--room-size", type=float, default=8.0)
parser.add_argument("--n-inner", type=int, default=6)
parser.add_argument("--n-humans", type=int, default=6)
parser.add_argument("--max-dist", type=float, default=10.0)
# Fixed start and goal. Training deliberately randomises both (see
# sample_spawn_and_goal in train_sac.py) so the policy cannot memorise
# one layout, but a demo or a specific "does it handle THIS situation"
# test needs to pin them down.
parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"),
                     default=None, help="start position; default 0 0")
parser.add_argument("--start-yaw", type=float, default=0.0,
                     help="start heading in radians; 0 faces +x")
parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"),
                     default=None,
                     help="goal position. Omit to run the built-in sweep.")
args = parser.parse_args()

env = make_env(args.map_seed, room=args.room, room_size=args.room_size,
                n_inner=args.n_inner, n_humans=args.n_humans,
                n_walls=args.n_walls, max_dist=args.max_dist)
mjx_model, walls = env["mjx_model"], env["walls"]
ped_params, ped_z = env["ped_params"], env["ped_z"]

ckpt = np.load(args.checkpoint)
n_layers = len([k for k in ckpt.files if k.endswith("_W")])
policy_params = [(jnp.array(ckpt[f"p{i}_W"]), jnp.array(ckpt[f"p{i}_b"])) for i in range(n_layers)]

@jax.jit
def step_fn(data, goal, prev_action, prev_prev_action, prev_scan, i):
    t0 = i * args.action_repeat * 0.002
    obs, x, y, theta, obstacle_d, scan = build_obs_sac(
        data, goal, walls, _peds_at(t0, ped_params), prev_action,
        prev_prev_action, prev_scan)
    action = deterministic_action(policy_params, obs)
    ctrl = jnp.array([STEER_RANGE * action[0], action[1], action[1]])

    def repeat_body(d, k):
        d = _move_peds(d, t0 + k * 0.002, ped_params, ped_z)
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mjx_model, d)
        d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
        return d, None

    data, _ = jax.lax.scan(repeat_body, data,
                            jnp.arange(args.action_repeat, dtype=jnp.float32))
    x2, y2, _ = get_car_xy_heading(data)
    clear = jnp.min(wall_distances(jnp.array([x2, y2]), walls))
    return data, x, y, theta, action, scan, clear

start = jnp.array(args.start if args.start else [0.0, 0.0])
fresh = mjx.make_data(mjx_model)

if args.goal:
    goals = {f"({args.goal[0]:g},{args.goal[1]:g})": tuple(args.goal)}
else:
    goals = {"1m": (0.9, 0.3), "2m": (1.8, 0.6), "3m": (2.7, 0.9),
             "4m": (3.6, 1.2), "6m": (5.4, 1.8)}

for name, (gx, gy) in goals.items():
    goal = jnp.array([gx, gy])
    # reset_data_at places the chassis and heading, matching exactly how
    # training resets an episode -- so a fixed-start trace is comparable
    # to what the policy saw while learning.
    data = reset_data_at(fresh, start, jnp.array(args.start_yaw))
    prev_action = jnp.zeros(2)
    prev_prev_action = jnp.zeros(2)
    prev_scan = jnp.ones(N_LIDAR)
    crash_t = None
    sx, sy = float(start[0]), float(start[1])
    closest = float(np.hypot(gx - sx, gy - sy))
    print(f"\n=== start=({sx:g},{sy:g}) yaw={args.start_yaw:g}  "
          f"goal {name}=({gx:g},{gy:g})  dist={closest:.2f}m ===")
    for t in range(args.horizon):
        data, x, y, theta, action, prev_scan, clear = step_fn(
            data, goal, prev_action, prev_prev_action, prev_scan, float(t))
        # Training and eval END the episode on contact. Without this the
        # trace grinds against a wall at full throttle for the remaining
        # decisions and reports only a poor closest-approach, which reads
        # as "navigated badly" rather than "crashed at t=97".
        if crash_t is None and float(clear) < CAR_RADIUS:
            crash_t = t
        prev_prev_action, prev_action = prev_action, action
        gd = float(jnp.sqrt((gx - x) ** 2 + (gy - y) ** 2))
        closest = min(closest, gd)
        if t % 15 == 0 or t == args.horizon - 1:
            v = float(jnp.linalg.norm(data.qvel[:2]))
            print(f"  t={t:4d}  pos=({float(x):+.3f},{float(y):+.3f})  theta={float(theta):+.2f}  "
                  f"v={v:.3f}  steer={float(action[0]):+.2f}  throttle={float(action[1]):+.2f}  goal_dist={gd:.3f}")
    if closest < SUCCESS_DIST:
        verdict = "SUCCESS"
    elif crash_t is not None:
        verdict = f"CRASHED into a wall at t={crash_t}"
    else:
        verdict = "FAIL (no crash -- ran out of time or stalled)"
    print(f"  -> closest approach {closest:.3f}  ({verdict})")
