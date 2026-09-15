"""
train_sac.py

Soft Actor-Critic (Haarnoja et al. 2018, arXiv:1801.01290) for the same
Ackermann car / MJX environment used by train_diffmjx_final.py.

Model-free: unlike the DiffRL pipeline, this never backpropagates
through the physics simulator. MJX is used purely as a forward
simulator to collect (obs, action, reward, next_obs, done) transitions
into a replay buffer; the policy and twin Q-networks are trained from
sampled minibatches the usual off-policy actor-critic way. This
sidesteps every contact-gradient/stiff-dynamics fragility the DiffRL
side has been fighting (exploding gradients through the contact
solver, the car getting stuck spiraling because reward shaping
interacts badly with truncated-BPTT credit assignment, etc.) at the
cost of needing many more environment steps to learn from, since it
doesn't get "free" analytic gradient information the way DiffRL does.

Reuses the same car physics (mjx_car_scene, mjx_solver_patch), the
same reward function (jax_reward), and the same observation
construction (build_obs) as the DiffRL pipeline, so results are
directly comparable -- only the learning algorithm differs.
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
from jax_reward import jax_reward
from jax_sac_networks import (
    ACTION_DIM, init_sac_policy_params, init_q_params,
    sample_action, deterministic_action, q_apply,
)
from train_diffmjx_final import (
    get_car_xy_heading, build_obs, QVEL_CLAMP, CAR_RADIUS,
    sample_goals, curriculum_goal_range, adam_init, adam_update,
)

SUCCESS_DIST = 0.5
REWARD_SCALE = 0.05


def make_fresh_data(mjx_model):
    return mjx.make_data(mjx_model)


def env_step_batched(policy_params, mjx_model, walls, states, key, horizon, min_dist, max_dist, gamma, fresh_data):
    """states: dict of batched arrays (leading dim N_ENVS):
       data (mjx.Data pytree), goal (N,2), prev_action (N,2),
       prev_prev_action (N,2), step_count (N,)
    Returns (new_states, transitions) where transitions is a dict of
    (obs, action, reward, next_obs, done) each with leading dim N_ENVS.
    """
    data = states["data"]
    goal = states["goal"]
    prev_action = states["prev_action"]
    prev_prev_action = states["prev_prev_action"]
    step_count = states["step_count"]
    n_envs = goal.shape[0]

    act_key, goal_key = jax.random.split(key)
    act_keys = jax.random.split(act_key, n_envs)
    goal_keys = jax.random.split(goal_key, n_envs)

    def single(data_i, goal_i, prev_action_i, prev_prev_action_i, step_count_i, act_key_i, goal_key_i):
        obs, x, y, theta, obstacle_d = build_obs(data_i, goal_i, walls)
        prev_goal_dist = jnp.sqrt((goal_i[0] - x) ** 2 + (goal_i[1] - y) ** 2 + 1e-9)

        action, _ = sample_action(policy_params, obs, act_key_i)
        ctrl = jnp.array([action[0], action[1], action[1]])
        new_data = data_i.replace(ctrl=ctrl)
        new_data = mjx.step(mjx_model, new_data)
        new_data = new_data.replace(qvel=jnp.clip(new_data.qvel, -QVEL_CLAMP, QVEL_CLAMP))

        next_obs, x2, y2, theta2, obstacle_d2 = build_obs(new_data, goal_i, walls)
        goal_dist2 = jnp.sqrt((goal_i[0] - x2) ** 2 + (goal_i[1] - y2) ** 2 + 1e-9)

        reward, _ = jax_reward(
            jnp.array([x2, y2]), theta2, action, prev_action_i, prev_prev_action_i,
            goal_i, obstacle_d2, prev_goal_dist,
        )
        # jax_reward's weights (progress=150.0 especially) were tuned for
        # DiffRL's short truncated-BPTT window (32 steps), where returns
        # never accumulate far. SAC bootstraps the full discounted return
        # (gamma=0.99 -> ~100-step effective horizon), so unscaled this
        # blows Q-targets up to a range the small critic MLP can't fit,
        # and the actor collapses to saturated, input-independent actions
        # chasing the miscalibrated critic (confirmed via trajectory
        # trace: steer/throttle pinned near +-0.95 regardless of goal).
        # SAC's own paper (Haarnoja et al. 2018, Table 1) calls this out
        # as "reward scale", a per-environment hyperparameter -- scaling
        # down here keeps Q-magnitudes in a range the critic can track.
        reward = reward * REWARD_SCALE

        success = goal_dist2 < SUCCESS_DIST
        timeout = (step_count_i + 1) >= horizon
        reset = success | timeout
        # bootstrap mask: only true success zeroes future value; timeout is
        # an artificial cutoff, not a real terminal state, so we still
        # bootstrap through it (standard truncation-vs-termination handling).
        done_for_bootstrap = success

        dist_key, angle_key = jax.random.split(goal_key_i)
        angle = jax.random.uniform(angle_key, (), minval=0, maxval=2 * jnp.pi)
        dist = jax.random.uniform(dist_key, (), minval=min_dist, maxval=max_dist)
        new_goal_sample = jnp.array([dist * jnp.cos(angle), dist * jnp.sin(angle)])

        out_data = jax.tree_util.tree_map(
            lambda f, n: jnp.where(reset, f, n), fresh_data, new_data
        )
        out_goal = jnp.where(reset, new_goal_sample, goal_i)
        out_prev_action = jnp.where(reset, jnp.zeros(2), action)
        out_prev_prev_action = jnp.where(reset, jnp.zeros(2), prev_action_i)
        out_step_count = jnp.where(reset, 0, step_count_i + 1)

        return (out_data, out_goal, out_prev_action, out_prev_prev_action, out_step_count), \
               (obs, action, reward, next_obs, done_for_bootstrap.astype(jnp.float32))

    new_carry, transitions = jax.vmap(single)(
        data, goal, prev_action, prev_prev_action, step_count, act_keys, goal_keys
    )
    new_data, new_goal, new_prev_action, new_prev_prev_action, new_step_count = new_carry
    new_states = {
        "data": new_data, "goal": new_goal, "prev_action": new_prev_action,
        "prev_prev_action": new_prev_prev_action, "step_count": new_step_count,
    }
    return new_states, transitions


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, action, reward, next_obs, done):
        n = obs.shape[0]
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            jnp.array(self.obs[idx]), jnp.array(self.action[idx]), jnp.array(self.reward[idx]),
            jnp.array(self.next_obs[idx]), jnp.array(self.done[idx]),
        )


def sac_update(policy_params, q1_params, q2_params, q1_target, q2_target, log_alpha,
                policy_opt, q1_opt, q2_opt, alpha_opt,
                obs, action, reward, next_obs, done, key, gamma, tau, target_entropy, lr):
    alpha = jnp.exp(log_alpha)
    key1, key2 = jax.random.split(key)

    def critic_loss_fn(q1_params, q2_params):
        next_action, next_log_prob = sample_action(policy_params, next_obs, key1)
        next_q1 = q_apply(q1_target, next_obs, next_action)
        next_q2 = q_apply(q2_target, next_obs, next_action)
        next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_prob
        target = jax.lax.stop_gradient(reward + gamma * (1 - done) * next_q)

        q1_pred = q_apply(q1_params, obs, action)
        q2_pred = q_apply(q2_params, obs, action)
        return jnp.mean((q1_pred - target) ** 2) + jnp.mean((q2_pred - target) ** 2)

    critic_loss, (q1_grads, q2_grads) = jax.value_and_grad(critic_loss_fn, argnums=(0, 1))(q1_params, q2_params)
    q1_params, q1_opt = adam_update(q1_params, q1_grads, q1_opt, lr=lr)
    q2_params, q2_opt = adam_update(q2_params, q2_grads, q2_opt, lr=lr)

    def actor_loss_fn(policy_params):
        new_action, log_prob = sample_action(policy_params, obs, key2)
        q1_val = q_apply(q1_params, obs, new_action)
        q2_val = q_apply(q2_params, obs, new_action)
        q_val = jnp.minimum(q1_val, q2_val)
        loss = jnp.mean(alpha * log_prob - q_val)
        return loss, log_prob

    (actor_loss, log_prob), policy_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(policy_params)
    policy_params, policy_opt = adam_update(policy_params, policy_grads, policy_opt, lr=lr)

    def alpha_loss_fn(log_alpha):
        return -jnp.mean(log_alpha * jax.lax.stop_gradient(log_prob + target_entropy))

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    log_alpha, alpha_opt = adam_update(log_alpha, alpha_grad, alpha_opt, lr=lr)

    q1_target = jax.tree_util.tree_map(lambda t, s: tau * s + (1 - tau) * t, q1_target, q1_params)
    q2_target = jax.tree_util.tree_map(lambda t, s: tau * s + (1 - tau) * t, q2_target, q2_params)

    return (policy_params, q1_params, q2_params, q1_target, q2_target, log_alpha,
            policy_opt, q1_opt, q2_opt, alpha_opt, critic_loss, actor_loss, alpha_loss)


def evaluate_sac(policy_params, mjx_model, walls, horizon, n_eval, min_dist, max_dist, seed=999):
    g = jax.random.PRNGKey(seed)
    goals = sample_goals(g, n_eval, min_dist, max_dist)

    def rollout_one(goal):
        data = mjx.make_data(mjx_model)

        def step(carry, _):
            data, min_dist_seen = carry
            obs, x, y, theta, obstacle_d = build_obs(data, goal, walls)
            action = deterministic_action(policy_params, obs)
            ctrl = jnp.array([action[0], action[1], action[1]])
            data = data.replace(ctrl=ctrl)
            data = mjx.step(mjx_model, data)
            data = data.replace(qvel=jnp.clip(data.qvel, -QVEL_CLAMP, QVEL_CLAMP))
            min_dist_seen = jnp.minimum(min_dist_seen, jnp.min(obstacle_d))
            return (data, min_dist_seen), None

        init_dist = jnp.sqrt(goal[0] ** 2 + goal[1] ** 2)
        (data, min_obs_dist), _ = jax.lax.scan(step, (data, jnp.array(jnp.inf)), None, length=horizon)
        _, x, y, _, _ = build_obs(data, goal, walls)
        final_dist = jnp.sqrt((goal[0] - x) ** 2 + (goal[1] - y) ** 2)
        return final_dist, min_obs_dist

    final_dists, min_obs_dists = jax.vmap(rollout_one)(goals)
    success = jnp.mean(final_dists < SUCCESS_DIST)
    mean_dist = jnp.mean(final_dists)
    collisions = jnp.sum(min_obs_dists < CAR_RADIUS)
    return mean_dist, collisions, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--max-dist", type=float, default=2.0)
    parser.add_argument("--n-walls", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    walls_list = generate_map_set(n_maps=1, n_walls=args.n_walls, base_seed=args.seed)[0]
    walls = jnp.array(walls_list)
    scene_path = build_car_scene_xml(walls_list, out_path="sac_map_0.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    mjx_model = mjx.put_model(model)
    print(f"Built SAC training map (n_walls={args.n_walls}).")

    key = jax.random.PRNGKey(args.seed)
    pkey, q1key, q2key = jax.random.split(key, 3)
    obs_dim = args.n_walls + 4
    policy_params = init_sac_policy_params(pkey, obs_dim)
    q1_params = init_q_params(q1key, obs_dim)
    q2_params = init_q_params(q2key, obs_dim)
    q1_target = q1_params
    q2_target = q2_params
    log_alpha = jnp.array(0.0)
    target_entropy = -float(ACTION_DIM)

    policy_opt = adam_init(policy_params)
    q1_opt = adam_init(q1_params)
    q2_opt = adam_init(q2_params)
    alpha_opt = adam_init(log_alpha)

    fresh_data = make_fresh_data(mjx_model)

    env_step_jit = jax.jit(
        lambda policy_params, states, key, min_dist, max_dist: env_step_batched(
            policy_params, mjx_model, walls, states, key, args.horizon, min_dist, max_dist, args.gamma, fresh_data
        )
    )
    sac_update_jit = jax.jit(
        lambda *a, **kw: sac_update(*a, gamma=args.gamma, tau=args.tau, target_entropy=target_entropy, lr=args.lr, **kw)
    )
    eval_jit = jax.jit(
        lambda pp, min_dist, max_dist: evaluate_sac(pp, mjx_model, walls, args.horizon, args.eval_n, min_dist, max_dist)
    )

    n_envs = args.n_envs
    states = {
        "data": jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_envs,) + x.shape), fresh_data),
        "goal": sample_goals(jax.random.PRNGKey(args.seed + 1), n_envs, 0.3, 0.5),
        "prev_action": jnp.zeros((n_envs, 2)),
        "prev_prev_action": jnp.zeros((n_envs, 2)),
        "step_count": jnp.zeros((n_envs,), dtype=jnp.int32),
    }

    buffer = ReplayBuffer(args.buffer_size, obs_dim)
    rng = np.random.default_rng(args.seed)

    best_success = -1.0
    total_env_steps = 0
    it = 0
    while total_env_steps < args.total_steps:
        progress = min(1.0, total_env_steps / (args.total_steps * 0.7))
        min_dist, max_dist = curriculum_goal_range(progress, args.max_dist)

        key, step_key = jax.random.split(key)
        states, transitions = env_step_jit(policy_params, states, step_key, min_dist, max_dist)
        obs_b, action_b, reward_b, next_obs_b, done_b = jax.device_get(transitions)
        buffer.add_batch(obs_b, action_b, reward_b, next_obs_b, done_b)
        total_env_steps += n_envs
        it += 1

        if buffer.size >= max(args.warmup_steps, args.batch_size):
            for _ in range(args.updates_per_step):
                obs_s, action_s, reward_s, next_obs_s, done_s = buffer.sample(args.batch_size, rng)
                key, upd_key = jax.random.split(key)
                (policy_params, q1_params, q2_params, q1_target, q2_target, log_alpha,
                 policy_opt, q1_opt, q2_opt, alpha_opt,
                 critic_loss, actor_loss, alpha_loss) = sac_update_jit(
                    policy_params, q1_params, q2_params, q1_target, q2_target, log_alpha,
                    policy_opt, q1_opt, q2_opt, alpha_opt,
                    obs=obs_s, action=action_s, reward=reward_s, next_obs=next_obs_s, done=done_s, key=upd_key,
                )

        if total_env_steps % args.eval_every < n_envs:
            eval_dist, eval_collisions, eval_success = eval_jit(policy_params, 0.3, args.max_dist)
            eval_success_f = float(eval_success)
            alpha_display = float(jnp.exp(log_alpha))
            print(f"steps {total_env_steps:7d}  progress={progress:.2f}  goal_range=[0.3,{max_dist:.1f}]  "
                  f"alpha={alpha_display:.3f}  buffer={buffer.size}  "
                  f"EVAL_dist={float(eval_dist):.3f}  EVAL_success={eval_success_f*100:.0f}%  "
                  f"EVAL_collisions={int(eval_collisions)}/{args.eval_n}")
            if eval_success_f > best_success:
                best_success = eval_success_f
                np.savez("sac_policy_best.npz",
                         **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(policy_params)},
                         **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(policy_params)})
                print(f"         -> new best checkpoint (success={best_success*100:.0f}%) saved")

    print(f"\nDone. Best checkpoint: success={best_success*100:.0f}%")
    eval_dist, eval_collisions, eval_success = eval_jit(policy_params, 0.3, args.max_dist)
    print(f"LARGE-SAMPLE final eval: success={float(eval_success)*100:.1f}%  mean_dist={float(eval_dist):.3f}")
    np.savez("sac_policy_final.npz",
             **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(policy_params)},
             **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(policy_params)})


if __name__ == "__main__":
    main()
