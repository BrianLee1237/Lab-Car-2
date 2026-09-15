"""Quick trajectory trace for a trained SAC policy checkpoint."""
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
from jax_sac_networks import deterministic_action
from train_diffmjx_final import get_car_xy_heading, build_obs, QVEL_CLAMP

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="sac_policy_best.npz")
parser.add_argument("--horizon", type=int, default=2500)
parser.add_argument("--n-walls", type=int, default=4)
args = parser.parse_args()

walls_list = generate_map_set(n_maps=1, n_walls=args.n_walls, base_seed=0)[0]
walls = jnp.array(walls_list)
scene_path = build_car_scene_xml(walls_list, out_path="sac_trace_map.xml")
model = mujoco.MjModel.from_xml_path(scene_path)
mjx_model = mjx.put_model(model)

ckpt = np.load(args.checkpoint)
n_layers = len([k for k in ckpt.files if k.endswith("_W")])
policy_params = [(jnp.array(ckpt[f"p{i}_W"]), jnp.array(ckpt[f"p{i}_b"])) for i in range(n_layers)]

@jax.jit
def step_fn(data, goal):
    obs, x, y, theta, obstacle_d = build_obs(data, goal, walls)
    action = deterministic_action(policy_params, obs)
    ctrl = jnp.array([action[0], action[1], action[1]])
    data = data.replace(ctrl=ctrl)
    data = mjx.step(mjx_model, data)
    data = data.replace(qvel=jnp.clip(data.qvel, -QVEL_CLAMP, QVEL_CLAMP))
    return data, x, y, theta, action

goals = {"1m": (0.9, 0.3), "2m": (1.8, 0.6), "3m": (2.7, 0.9), "4m": (3.6, 1.2), "6m": (5.4, 1.8)}

for name, (gx, gy) in goals.items():
    goal = jnp.array([gx, gy])
    data = mjx.make_data(mjx_model)
    print(f"\n=== Goal: {name}  xy=({gx}, {gy}) ===")
    for t in range(args.horizon):
        data, x, y, theta, action = step_fn(data, goal)
        if t % 300 == 0 or t == args.horizon - 1:
            gd = float(jnp.sqrt((gx - x) ** 2 + (gy - y) ** 2))
            v = float(jnp.linalg.norm(data.qvel[:2]))
            print(f"  t={t:5d}  pos=({float(x):+.3f},{float(y):+.3f})  theta={float(theta):+.2f}  "
                  f"v={v:.3f}  steer={float(action[0]):+.2f}  throttle={float(action[1]):+.2f}  goal_dist={gd:.3f}")
