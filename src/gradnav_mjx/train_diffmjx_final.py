"""
train_diffmjx_final.py (v2)

Added the two things that made the PyTorch version actually show real
learning: curriculum (start with close goals, expand over training)
and a FIXED evaluation benchmark (same seed every check, decoupled
from the curriculum/random training goals) so we can actually see if
mean_final_dist is improving, not just bouncing around noisily.
"""

import argparse

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from mjx_car_scene import build_car_scene_xml
from mjx_random_maps import generate_map_set
from mjx_obstacle_dist import wall_distances
from jax_networks import init_policy_params, policy_forward, init_value_params, value_forward
from jax_reward import jax_reward


DIFFMJX_OPTS = dict(
    scan_loop=True,
    col_soft_enable=True,
    softjax_mode="c2",
    col_softness=0.1,
)

CAR_RADIUS = 0.24


def adam_init(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": m, "v": v, "t": 0}


def adam_update(params, grads, state, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * (g ** 2), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return new_params, {"m": m, "v": v, "t": t}


def get_car_xy_heading(data):
    x, y = data.qpos[0], data.qpos[1]
    qw, qz = data.qpos[3], data.qpos[6]
    theta = 2 * jnp.arctan2(qz, qw)
    return x, y, theta


def build_obs(data, goal, walls):
    x, y, theta = get_car_xy_heading(data)
    v = jnp.sqrt(data.qvel[0] ** 2 + data.qvel[1] ** 2)
    goal_dx = goal[0] - x
    goal_dy = goal[1] - y
    obstacle_d = wall_distances(jnp.array([x, y]), walls)
    extra = jnp.stack([v, theta, goal_dx, goal_dy])
    obs = jnp.concatenate([obstacle_d, extra])
    return obs, x, y, theta, obstacle_d


def rollout(mjx_model, policy_params, value_params, walls, goal, horizon, gamma=0.99):
    data = mjx.make_data(mjx_model)
    prev_action = jnp.zeros(2)
    prev_prev_action = jnp.zeros(2)
    x0, y0, _ = get_car_xy_heading(data)
    prev_goal_dist = jnp.sqrt((goal[0] - x0) ** 2 + (goal[1] - y0) ** 2)
    discounted_reward = jnp.array(0.0)
    discount = 1.0
    min_obstacle_dist_seen = jnp.array(jnp.inf)

    for t in range(horizon):
        obs, x, y, theta, obstacle_d = build_obs(data, goal, walls)
        action = policy_forward(policy_params, obs)
        ctrl = jnp.array([action[0], action[1], action[1]])
        data = data.replace(ctrl=ctrl)
        data = mjx.step(mjx_model, data)

        car_xy = jnp.array([x, y])
        r, goal_dist = jax_reward(
            car_xy, theta, action, prev_action, prev_prev_action,
            goal, obstacle_d, prev_goal_dist,
        )
        discounted_reward = discounted_reward + discount * r
        discount *= gamma
        prev_prev_action = prev_action
        prev_action = action
        prev_goal_dist = goal_dist
        min_obstacle_dist_seen = jnp.minimum(min_obstacle_dist_seen, jnp.min(obstacle_d))

    final_obs, x, y, theta, _ = build_obs(data, goal, walls)
    bootstrap = discount * value_forward(value_params, final_obs)
    total_return = discounted_reward + bootstrap
    final_dist = jnp.sqrt((goal[0] - x) ** 2 + (goal[1] - y) ** 2)
    return total_return, final_dist, min_obstacle_dist_seen


def batched_loss(mjx_model, policy_params, value_params, walls, goals, horizon):
    def single(goal):
        return rollout(mjx_model, policy_params, value_params, walls, goal, horizon)
    returns, final_dists, min_obstacle_dists = jax.vmap(single)(goals)
    loss = -jnp.mean(returns)
    return loss, final_dists, min_obstacle_dists


def sample_goals(key, batch_size, min_dist, max_dist):
    angle_key, dist_key = jax.random.split(key)
    angles = jax.random.uniform(angle_key, (batch_size,), minval=0, maxval=2 * jnp.pi)
    dists = jax.random.uniform(dist_key, (batch_size,), minval=min_dist, maxval=max_dist)
    return jnp.stack([dists * jnp.cos(angles), dists * jnp.sin(angles)], axis=-1)


def clip_tree(grads, max_norm=5.0):
    leaves = jax.tree_util.tree_leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def sanitize_grads(grads):
    return jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)), grads
    )


def curriculum_goal_range(progress, max_dist, min_start=0.5):
    current_max = min_start + progress * (max_dist - min_start)
    return 0.3, current_max


def evaluate_fixed_benchmark(mjx_model, policy_params, value_params, walls, horizon, n_eval=16):
    """Always full difficulty, ALWAYS the same seed -- decoupled from
    training curriculum, so this is a trustworthy, comparable metric
    across the whole run."""
    g = jax.random.PRNGKey(999)
    goals = sample_goals(g, n_eval, 0.5, 2.5)
    _, final_dists, min_obstacle_dists = batched_loss(mjx_model, policy_params, value_params, walls, goals, horizon)
    valid = jnp.isfinite(final_dists)
    mean_dist = jnp.mean(jnp.where(valid, final_dists, 0.0)) / jnp.maximum(jnp.mean(valid), 1e-6)
    n_collisions = int(jnp.sum(min_obstacle_dists < CAR_RADIUS))
    return float(mean_dist), n_collisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-dist", type=float, default=2.5)
    parser.add_argument("--n-maps", type=int, default=4)
    parser.add_argument("--n-walls", type=int, default=4)
    parser.add_argument("--switch-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    map_wall_lists = generate_map_set(n_maps=args.n_maps, n_walls=args.n_walls, base_seed=args.seed)
    map_walls_arrays = [jnp.array(w) for w in map_wall_lists]
    mjx_models = []
    for i, wall_list in enumerate(map_wall_lists):
        scene_path = build_car_scene_xml(wall_list, out_path=f"diffmjx_map_{i}.xml")
        model = mujoco.MjModel.from_xml_path(scene_path)
        m = mjx.put_model(model)
        m = m.replace(opt=m.opt.replace(**DIFFMJX_OPTS))
        mjx_models.append(m)
    print(f"Built {args.n_maps} randomized maps using DiffMJX.")

    key = jax.random.PRNGKey(args.seed)
    pkey, vkey = jax.random.split(key)
    policy_params = init_policy_params(pkey, lidar_dim=args.n_walls)
    value_params = init_value_params(vkey, lidar_dim=args.n_walls)
    policy_opt_state = adam_init(policy_params)
    value_opt_state = adam_init(value_params)

    grad_fn = jax.value_and_grad(
        lambda pp, vp, mjx_model, walls, goals: batched_loss(
            mjx_model, pp, vp, walls, goals, args.horizon
        )[0],
        argnums=(0, 1),
    )

    for it in range(args.iterations):
        map_idx = (it // args.switch_every) % args.n_maps
        mjx_model = mjx_models[map_idx]
        walls = map_walls_arrays[map_idx]

        progress = min(1.0, it / (args.iterations * 0.7))
        min_dist, max_dist = curriculum_goal_range(progress, args.max_dist)

        key, gkey = jax.random.split(key)
        goals = sample_goals(gkey, args.batch_size, min_dist, max_dist)

        loss, (pgrads, vgrads) = grad_fn(policy_params, value_params, mjx_model, walls, goals)
        pgrads = sanitize_grads(pgrads)
        vgrads = sanitize_grads(vgrads)
        pgrads = clip_tree(pgrads)
        vgrads = clip_tree(vgrads)

        policy_params, policy_opt_state = adam_update(policy_params, pgrads, policy_opt_state, lr=args.lr)
        value_params, value_opt_state = adam_update(value_params, vgrads, value_opt_state, lr=args.lr)

        if (it + 1) % args.eval_every == 0:
            eval_map = mjx_models[0]
            eval_walls = map_walls_arrays[0]
            eval_dist, eval_collisions = evaluate_fixed_benchmark(eval_map, policy_params, value_params, eval_walls, args.horizon)
            loss_display = float(loss) if jnp.isfinite(loss) else float("nan")
            print(f"iter {it+1:4d}  map={map_idx}  progress={progress:.2f}  goal_range=[{min_dist:.1f},{max_dist:.1f}]  "
                  f"loss={loss_display:.3f}  EVAL_dist(fixed)={eval_dist:.3f}  EVAL_collisions={eval_collisions}/16")

    print("\nDone.")
    import numpy as np
    np.savez("diffmjx_policy_final.npz",
             **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(policy_params)},
             **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(policy_params)})
    print("Policy saved to diffmjx_policy_final.npz")


if __name__ == "__main__":
    main()
