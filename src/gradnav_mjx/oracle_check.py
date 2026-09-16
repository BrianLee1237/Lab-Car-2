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

from mjx_car_scene import build_car_scene_xml, STEER_RANGE
from mjx_random_maps import generate_map_set
from train_diffmjx_final import get_car_xy_heading, QVEL_CLAMP, sample_goals

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=150)
parser.add_argument("--action-repeat", type=int, default=20)
parser.add_argument("--n-walls", type=int, default=4)
parser.add_argument("--n-goals", type=int, default=20)
parser.add_argument("--max-dist", type=float, default=2.0)
parser.add_argument("--min-dist", type=float, default=0.3)
parser.add_argument("--cone", type=float, default=3.1416,
                     help="half-angle (rad) of the forward cone goals are sampled in; "
                          "pi = all directions (current training setup)")
parser.add_argument("--throttle", type=float, default=1.0)
parser.add_argument("--steer-gain", type=float, default=1.0)
args = parser.parse_args()

walls_list = generate_map_set(n_maps=1, n_walls=args.n_walls, base_seed=0)[0]
walls = jnp.array(walls_list)
model = mujoco.MjModel.from_xml_path(build_car_scene_xml(walls_list, out_path="oracle_map.xml"))
mjx_model = mjx.put_model(model)


@jax.jit
def step_fn(data, goal):
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

    def repeat_body(d, _):
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mjx_model, d)
        d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
        return d, None

    data, _ = jax.lax.scan(repeat_body, data, None, length=args.action_repeat)
    return data, x, y, theta


_k1, _k2 = jax.random.split(jax.random.PRNGKey(999))
_ang = jax.random.uniform(_k1, (args.n_goals,), minval=-args.cone, maxval=args.cone)
_d = jax.random.uniform(_k2, (args.n_goals,), minval=args.min_dist, maxval=args.max_dist)
goals = jnp.stack([_d * jnp.cos(_ang), _d * jnp.sin(_ang)], axis=-1)
n_success = 0
for i in range(args.n_goals):
    gx, gy = float(goals[i, 0]), float(goals[i, 1])
    goal = jnp.array([gx, gy])
    data = mjx.make_data(mjx_model)
    closest = float(np.hypot(gx, gy))
    init = closest
    for t in range(args.horizon):
        data, x, y, theta = step_fn(data, goal)
        gd = float(jnp.sqrt((gx - x) ** 2 + (gy - y) ** 2))
        closest = min(closest, gd)
    ok = closest < 0.5
    n_success += ok
    final_v = float(jnp.linalg.norm(data.qvel[:2]))
    print(f"goal {i:2d}  init_dist={init:.2f}  closest={closest:.3f}  final_v={final_v:.2f}  "
          f"{'SUCCESS' if ok else 'FAIL'}")

print(f"\nORACLE success: {n_success}/{args.n_goals} = {100*n_success/args.n_goals:.0f}%")
print(f"(horizon={args.horizon} decisions x {args.action_repeat} steps x 0.002s = "
      f"{args.horizon*args.action_repeat*0.002:.1f}s of sim time)")
