"""Measure what the steering actuator actually does: is ctrl[0] a
steering ANGLE (position servo) or a TORQUE that runs the joint to its
mechanical limit? Prints the steer joint angle over time for a few
constant steer commands."""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

import mjx_solver_patch
mjx_solver_patch.apply()

from mjx_car_scene import build_car_scene_xml, STEER_RANGE
from mjx_random_maps import generate_map_set
from train_diffmjx_final import get_car_xy_heading, QVEL_CLAMP

walls_list = generate_map_set(n_maps=1, n_walls=4, base_seed=0)[0]
model = mujoco.MjModel.from_xml_path(build_car_scene_xml(walls_list, out_path="steer_map.xml"))
mjx_model = mjx.put_model(model)

steer_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fl")
steer_qadr = model.jnt_qposadr[steer_jid]
print(f"steer_fl qpos address = {steer_qadr}, joint range = {model.jnt_range[steer_jid]}")


@partial(jax.jit, static_argnums=(2,))
def run(steer_cmd, throttle, n_steps):
    def body(d, _):
        d = d.replace(ctrl=jnp.array([steer_cmd, throttle, throttle]))
        d = mjx.step(mjx_model, d)
        d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
        return d, d.qpos[steer_qadr]

    data = mjx.make_data(mjx_model)
    data, angles = jax.lax.scan(body, data, None, length=n_steps)
    return data, angles


print("\nsteer_cmd -> steering joint angle (rad) over 2s at throttle=0.5")
print("  cmd     t=0.1s   t=0.5s   t=1.0s   t=2.0s   turn_radius(m)")
for cmd in [0.05, 0.15, 0.3, 0.45, 0.6]:
    data, angles = run(cmd, 0.5, 1000)
    a = np.array(angles)
    x, y, theta = get_car_xy_heading(data)
    # radius from final yaw rate and speed
    v = float(jnp.linalg.norm(data.qvel[:2]))
    yaw_rate = float(data.qvel[5])
    radius = v / abs(yaw_rate) if abs(yaw_rate) > 1e-4 else float("inf")
    print(f"  {cmd:4.2f}   {a[50]:+.4f}  {a[250]:+.4f}  {a[500]:+.4f}  {a[999]:+.4f}   {radius:6.2f}")
