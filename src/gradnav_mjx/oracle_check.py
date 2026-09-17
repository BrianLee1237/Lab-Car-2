"""Sanity check: can a hand-written controller reach the goals at all?

If a simple proportional steer + forward throttle cannot reach goals
within the configured horizon/action_repeat budget, then the task is
unsolvable as configured and no amount of RL tuning will fix it. This
separates "the learner is bad" from "the environment/budget is wrong".
"""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

import mjx_solver_patch
mjx_solver_patch.apply()

from mjx_car_scene import STEER_RANGE
from train_diffmjx_final import get_car_xy_heading, QVEL_CLAMP
from mjx_obstacle_dist import obstacle_distances
import train_sac as T

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=150)
parser.add_argument("--action-repeat", type=int, default=20)
parser.add_argument("--n-walls", type=int, default=4)
parser.add_argument("--map-size", type=float, default=6.0)
parser.add_argument("--map-seed", type=int, default=0)
parser.add_argument("--room", action="store_true")
parser.add_argument("--room-size", type=float, default=8.0)
parser.add_argument("--n-inner", type=int, default=6)
parser.add_argument("--n-humans", type=int, default=6)
parser.add_argument("--spawn-half", type=float, default=3.0)
parser.add_argument("--n-goals", type=int, default=20)
parser.add_argument("--max-dist", type=float, default=2.0)
parser.add_argument("--min-dist", type=float, default=0.3)
parser.add_argument("--cone", type=float, default=3.1416,
                     help="half-angle (rad) of the forward cone goals are sampled in; "
                          "pi = all directions (current training setup)")
parser.add_argument("--throttle", type=float, default=1.0)
parser.add_argument("--steer-gain", type=float, default=1.0)
args = parser.parse_args()

env = T.make_env(args.map_seed, room=args.room, room_size=args.room_size,
                  n_inner=args.n_inner, n_humans=args.n_humans,
                  n_walls=args.n_walls, map_size=args.map_size,
                  spawn_half=args.spawn_half, max_dist=args.max_dist)
mjx_model, walls = env["mjx_model"], env["walls"]
ped_params, ped_z = env["ped_params"], env["ped_z"]


@jax.jit
def step_fn(data, goal, i):
    t0 = i * args.action_repeat * 0.002
    x, y, theta = get_car_xy_heading(data)
    dx, dy = goal[0] - x, goal[1] - y
    c, s = jnp.cos(theta), jnp.sin(theta)
    # goal angle in body frame -> proportional steering
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    angle = jnp.arctan2(by, bx)
    # Slow down to turn tightly: the car understeers badly with speed
    # (measured turn radius ~0.6m at throttle 0.5 but ~1.9m at throttle
    # 1.0), so charging at full throttle leaves most off-axis goals
    # unreachable. Scale throttle by how well we're already aimed.
    aim = jnp.cos(angle)
    # If the goal is mostly behind us, back up toward it (steering
    # inverted) rather than sweeping a wide forward arc -- the car's
    # minimum forward turn radius makes many rear goals unreachable
    # otherwise, and reversing is what a real driver does.
    reverse = jnp.abs(angle) > 1.9
    rev_angle = jnp.arctan2(-by, -bx)
    fwd_throttle = args.throttle * jnp.clip(0.25 + 0.75 * aim, 0.25, 1.0)
    throttle = jnp.where(reverse, -0.5 * args.throttle, fwd_throttle)
    raw_steer = jnp.where(reverse, -args.steer_gain * rev_angle, args.steer_gain * angle)
    steer = jnp.clip(raw_steer, -STEER_RANGE, STEER_RANGE)
    ctrl = jnp.array([steer, throttle, throttle])

    def repeat_body(d, k):
        d = T._move_peds(d, t0 + k * 0.002, ped_params, ped_z)
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mjx_model, d)
        d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
        return d, None

    data, _ = jax.lax.scan(repeat_body, data,
                            jnp.arange(args.action_repeat, dtype=jnp.float32))
    peds = T._peds_at(t0 + args.action_repeat*0.002, ped_params)
    clear = jnp.min(obstacle_distances(jnp.array([x, y]), walls, peds))
    return data, x, y, theta, clear


spawns, yaws, goals = jax.vmap(
    lambda k: T.sample_spawn_and_goal(k, walls, args.min_dist, args.max_dist,
                                       args.cone, env["goal_bound"])
)(jax.random.split(jax.random.PRNGKey(999), args.n_goals))
fresh = mjx.make_data(mjx_model)
n_coll = 0
n_success = 0
for i in range(args.n_goals):
    gx, gy = float(goals[i, 0]), float(goals[i, 1])
    goal = jnp.array([gx, gy])
    data = T.reset_data_at(fresh, spawns[i], yaws[i])
    closest = float(np.hypot(gx - float(spawns[i,0]), gy - float(spawns[i,1])))
    init = closest
    min_clear = 9e9
    for t in range(args.horizon):
        data, x, y, theta, clear = step_fn(data, goal, float(t))
        gd = float(jnp.sqrt((gx - x) ** 2 + (gy - y) ** 2))
        closest = min(closest, gd)
        min_clear = min(min_clear, float(clear))
    ok = closest < 0.5
    hit = min_clear < 0.24
    n_success += ok; n_coll += hit
    final_v = float(jnp.linalg.norm(data.qvel[:2]))
    print(f"goal {i:2d}  init_dist={init:.2f}  closest={closest:.3f}  "
          f"min_clear={min_clear:.2f}  {'SUCCESS' if ok else 'FAIL'}"
          f"{'  COLLIDED' if hit else ''}")

print(f"\nORACLE success: {n_success}/{args.n_goals} = {100*n_success/args.n_goals:.0f}%"
      f"   collisions: {n_coll}/{args.n_goals}")
print(f"(horizon={args.horizon} decisions x {args.action_repeat} steps x 0.002s = "
      f"{args.horizon*args.action_repeat*0.002:.1f}s of sim time)")
