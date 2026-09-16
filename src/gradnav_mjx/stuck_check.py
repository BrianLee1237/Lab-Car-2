"""Why does the car stop dead at some steering commands? Log pose,
height and tilt over time at a few constant steer angles."""
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
print("walls:", np.round(np.array(walls_list), 2).tolist())
model = mujoco.MjModel.from_xml_path(build_car_scene_xml(walls_list, out_path="stuck_map.xml"))
mjx_model = mjx.put_model(model)


@partial(jax.jit, static_argnums=(2,))
def run(steer, throttle, n):
    def body(d, _):
        d = d.replace(ctrl=jnp.array([steer, throttle, throttle]))
        d = mjx.step(mjx_model, d)
        d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
        # x, y, z, speed, |tilt| of body z-axis away from world z
        qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        # body z-axis in world coords, z-component
        upz = 1 - 2 * (qx ** 2 + qy ** 2)
        return d, jnp.array([d.qpos[0], d.qpos[1], d.qpos[2],
                             jnp.linalg.norm(d.qvel[:2]), upz])

    data = mjx.make_data(mjx_model)
    data, log = jax.lax.scan(body, data, None, length=n)
    return log


for steer in [0.0, 0.3, 0.6]:
    log = np.array(run(steer, 1.0, 3000))
    print(f"\n--- steer={steer}  throttle=1.0 ---")
    print("   t(s)      x       y       z     speed   up_z(1=level)")
    for i in range(0, 3000, 300):
        x, y, z, v, upz = log[i]
        print(f"  {i*0.002:5.2f}  {x:+6.2f}  {y:+6.2f}  {z:+5.3f}  {v:6.3f}   {upz:+.3f}")
