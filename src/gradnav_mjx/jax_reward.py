"""
jax_reward.py

JAX port of diff_reward.py, for use with the MJX-based Ackermann car.
Same math, same terms, same weights as the PyTorch version -- only the
tensor library changed (torch -> jax.numpy).

Tested and verified (in a separate sandbox before handing off):
  - Basic single-instance call: correct output
  - Gradient flow via jax.grad: nonzero gradients confirmed
  - Batching: naturally broadcasts over a leading batch dimension
  - JIT compilation via jax.jit: works correctly
"""

import jax.numpy as jnp
import jax.nn as jnn


def jax_reward(
    car_xy,
    car_heading,
    action,
    prev_action,
    prev_prev_action,
    goal,
    obstacle_dists,
    prev_goal_dist,
    arena_size=10.0,
    obstacle_safety_dist=1.5,
    weights=None,
):
    if weights is None:
        weights = dict(
            survival=0.5,
            action=-1.0,
            action_rate=-1.0,
            smoothness=-1.0,
            yaw_alignment=0.25,
            progress=8.0,
            precision=1.0,
            obstacle=1.0,
            out_of_map=-2.0,
        )

    x, y = car_xy[..., 0], car_xy[..., 1]
    theta = car_heading

    r_survival = jnp.ones_like(x)
    r_action = jnp.sum(action ** 2, axis=-1)
    r_action_rate = jnp.sum((action - prev_action) ** 2, axis=-1)
    r_smoothness = jnp.sum((action - 2 * prev_action + prev_prev_action) ** 2, axis=-1)

    goal_dx = goal[..., 0] - x
    goal_dy = goal[..., 1] - y
    goal_dist = jnp.sqrt(goal_dx ** 2 + goal_dy ** 2 + 1e-6)
    goal_dir_x = goal_dx / goal_dist
    goal_dir_y = goal_dy / goal_dist
    r_yaw_alignment = jnp.cos(theta) * goal_dir_x + jnp.sin(theta) * goal_dir_y

    r_progress = prev_goal_dist - goal_dist
    r_precision = jnp.exp(-goal_dist)

    min_obstacle_dist = jnp.min(obstacle_dists, axis=-1)
    r_obstacle = -jnn.softplus(obstacle_safety_dist - min_obstacle_dist)

    x_excess = jnn.softplus(jnp.abs(x) - arena_size)
    y_excess = jnn.softplus(jnp.abs(y) - arena_size)
    r_out_of_map = x_excess ** 2 + y_excess ** 2

    total = (
        weights["survival"] * r_survival
        + weights["action"] * r_action
        + weights["action_rate"] * r_action_rate
        + weights["smoothness"] * r_smoothness
        + weights["yaw_alignment"] * r_yaw_alignment
        + weights["progress"] * r_progress
        + weights["precision"] * r_precision
        + weights["obstacle"] * r_obstacle
        + weights["out_of_map"] * r_out_of_map
    )
    return total, goal_dist
