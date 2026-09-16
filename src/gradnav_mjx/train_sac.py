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

Reuses the same car physics (mjx_car_scene, mjx_solver_patch) and the
same reward *function* (jax_reward) as the DiffRL pipeline, but NOT
its reward weights or its observation encoding: both were tuned for
DiffRL's fixed-horizon, freeze-on-success, backprop-through-time setup
and are actively wrong for an off-policy bootstrapped value learner.
See SAC_REWARD_WEIGHTS and build_obs_sac for what differs and why.
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
from mjx_obstacle_dist import wall_distances
from mjx_lidar import lidar_scan
from jax_reward import jax_reward
from jax_sac_networks import (
    ACTION_DIM, init_sac_policy_params, init_q_params,
    sample_action, deterministic_action, q_apply,
)
from train_diffmjx_final import (
    get_car_xy_heading, QVEL_CLAMP, CAR_RADIUS,
    sample_goals, curriculum_goal_range, adam_init, adam_update,
    clip_tree, sanitize_grads,
)

SUCCESS_DIST = 0.5

# --- Reward shaping, SAC-specific -------------------------------------
# jax_reward's default weights are DiffRL's and must NOT be reused here.
# DiffRL accumulates a fixed-horizon return and freezes the state on
# success, so an unconditional positive per-step term is harmless. SAC
# bootstraps an infinite-horizon value and masks the bootstrap on
# success (target = r + gamma*(1-done)*Q'), so any standing positive
# reward makes *reaching the goal* strictly worse than loitering next
# to it: with survival=+0.5 and yaw_alignment up to +2.0, cruising was
# worth ~2.6/step forever, i.e. V ~ 2.6/(1-0.99) ~ 260, while touching
# the goal collapsed the target to the single-step reward ~2.6. The
# car was correctly optimising a reward that told it never to finish
# (observed directly: creeping at 0.02-0.04 m/s, goal-facing, for
# thousands of steps).
#
# Correct goal-reaching formulation: no standing bonus, a small per-step
# time cost so finishing sooner wins, dense progress shaping, and a
# real terminal bonus on success (DiffRL has TERMINAL_BONUS=50.0; SAC
# had none at all). progress is a telescoping sum, so its weight is
# just "reward per metre of distance closed".
SAC_REWARD_WEIGHTS = dict(
    survival=0.0,        # was 0.5  -- the "never finish" attractor
    action=-0.005,
    action_rate=-0.02,
    smoothness=-0.02,
    yaw_alignment=0.02,  # was 2.0  -- keep as faint shaping, not a standing wage
    progress=10.0,       # was 150.0 (DiffRL) -- +10.0 total per metre closed
    precision=0.0,       # was 1.0  -- another standing reward for loitering
    obstacle=0.5,
    out_of_map=-1.0,
)
TERMINAL_BONUS = 20.0
TIME_COST = -0.02        # per decision
# Magnitudes matter relative to SAC's entropy bonus alpha*H, not just
# in absolute terms. An earlier pass set progress=2.0/terminal=5.0 AND
# slowed alpha's decay 16x; together the entropy term swamped the tiny
# Q-values, so the policy optimised entropy instead of reward and its
# MEAN action went to ~0 -- i.e. zero throttle, car motionless, eval
# mean-closest-distance frozen at exactly the mean initial distance.

# --- Observation, SAC-specific ----------------------------------------
OBS_WALL_SCALE = 5.0
OBS_V_SCALE = 3.0
OBS_GOAL_DIST_SCALE = 8.0


def sample_goals_cone(key, n, min_dist, max_dist, cone):
    """Goals in a forward cone of half-angle `cone`, not uniformly over
    all 360 degrees like train_diffmjx_final.sample_goals.

    The car is Ackermann-steered with a measured minimum turning radius
    of ~0.6m slow and ~1.9m at speed. Sampling goals uniformly in every
    direction at 0.3-2.0m therefore places a large share of them INSIDE
    the car's own turning circle -- a goal 0.7m directly to the side
    cannot be reached by any single arc and needs a multi-point turn.
    A hand-written controller with speed modulation and reversing
    (oracle_check.py) tops out at 53% under that sampling, so the
    learner was being scored against a ~50% ceiling for reasons that
    had nothing to do with learning. The same controller reaches 97%
    on a +-60 degree forward cone at 1.0-3.0m -- which is also the
    realistic hardware scenario, with the goal ahead of the car.
    """
    ang_key, dist_key = jax.random.split(key)
    ang = jax.random.uniform(ang_key, (n,), minval=-cone, maxval=cone)
    dist = jax.random.uniform(dist_key, (n,), minval=min_dist, maxval=max_dist)
    return jnp.stack([dist * jnp.cos(ang), dist * jnp.sin(ang)], axis=-1)


SPAWN_HALF = 3.0        # spawn box half-extent (m)
SPAWN_CLEAR = 0.8       # required clearance from any wall at spawn (m)
SPAWN_CANDIDATES = 8


def sample_spawn_and_goal(key, walls, min_dist, max_dist, cone):
    """Random start pose + a goal in a cone ahead of THAT pose.

    The car used to spawn at the origin facing +x on every episode, on a
    single fixed map, with eval on that same map -- so there was no
    generalization signal at all and the policy could simply memorise one
    layout. Multi-map training is the obvious fix but MJX bakes each
    map's geometry into static, hashed model metadata, so every distinct
    map forces a separate compile, and compilation is the dominant cost
    here. Randomising the START POSE instead gives the same variety of
    wall configurations *relative to the car* on one compiled model: a
    given map looks like a different obstacle course from every pose.
    It is also closer to the hardware case, where the car does not begin
    each run at a surveyed origin.

    Spawn is rejection-sampled against the walls (first of
    SPAWN_CANDIDATES draws with SPAWN_CLEAR clearance; falls back to the
    first draw if none qualify, which the clearance margin makes rare).
    """
    k_pos, k_yaw, k_ang, k_dist = jax.random.split(key, 4)

    cand = jax.random.uniform(k_pos, (SPAWN_CANDIDATES, 2),
                              minval=-SPAWN_HALF, maxval=SPAWN_HALF)
    clearance = jnp.min(wall_distances(cand, walls), axis=-1)
    ok = clearance > SPAWN_CLEAR
    spawn = cand[jnp.argmax(ok)]

    yaw = jax.random.uniform(k_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
    ang = jax.random.uniform(k_ang, (), minval=-cone, maxval=cone)
    dist = jax.random.uniform(k_dist, (), minval=min_dist, maxval=max_dist)

    # goal in the car's frame, then rotated into world coords
    local = jnp.array([dist * jnp.cos(ang), dist * jnp.sin(ang)])
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    goal = spawn + jnp.array([c * local[0] - s * local[1],
                              s * local[0] + c * local[1]])
    return spawn, yaw, goal


def reset_data_at(fresh_data, spawn, yaw):
    """fresh_data respawned at (spawn, yaw). qpos layout for the free
    joint is [x, y, z, qw, qx, qy, qz, ...]; a yaw-only rotation is
    (cos(yaw/2), 0, 0, sin(yaw/2))."""
    qpos = fresh_data.qpos
    qpos = qpos.at[0].set(spawn[0]).at[1].set(spawn[1])
    qpos = (qpos.at[3].set(jnp.cos(yaw / 2))
                .at[4].set(0.0).at[5].set(0.0)
                .at[6].set(jnp.sin(yaw / 2)))
    return fresh_data.replace(qpos=qpos)


def make_fresh_data(mjx_model):
    return mjx.make_data(mjx_model)


# obs = N_LIDAR ranges + [vx_b, vy_b, sin, cos, dir_x, dir_y, dist]
#       + prev_action(2) + prev_prev_action(2)
N_LIDAR = 16
LIDAR_MAX_RANGE = 8.0
OBS_EXTRA_DIM_SAC = 11


def build_obs_sac(data, goal, walls, prev_action, prev_prev_action):
    """SAC observation. Deliberately not train_diffmjx_final.build_obs:

    - Goal in BODY frame (unit direction + separate normalized distance)
      instead of world-frame goal_dx/goal_dy paired with raw heading.
      World frame forces the network to learn the rotation itself before
      it can tell left from right; body frame makes "goal is to my left"
      directly readable, which is the standard goal-conditioned nav
      encoding. Splitting direction (unit norm) from distance also keeps
      the direction signal well-scaled at every distance, instead of
      shrinking toward 0 up close and saturating against the clip far
      away.
    - sin/cos of heading instead of theta/pi, which jumps discontinuously
      from +1 to -1 as theta wraps through pi.
    - SIGNED body-frame velocity (forward, lateral) instead of the speed
      magnitude |v|. With only |v| the policy literally cannot tell
      whether it is driving forwards or backwards -- and driving backwards
      away from the goal was one of the dominant observed failures.
    - Obstacles sensed by a body-frame LIDAR (mjx_lidar) rather than
      wall_distances' one scalar per wall. Those scalars are
      directionless and indexed by arbitrary wall ID -- "wall #2 is
      1.3m away" says nothing about whether it is ahead or behind, so
      avoidance was effectively impossible (collisions ran 17-40 per 50
      episodes at 6m goals, vs 0-2 per 50 at 1-2m where the straight
      path rarely meets a wall). jax_networks.py still defaults to
      lidar_dim=16, i.e. this is what the original design sensed with.
    - prev_action and prev_prev_action appended. jax_reward's action_rate
      and smoothness terms are functions of both (worth up to ~-2.4), but
      they appeared nowhere in the old observation, so the reward was not
      a function of (obs, action) alone. DiffRL gets away with it by
      carrying them in its scan carry and backpropagating the true
      trajectory; SAC's Q(s,a) structurally cannot fit a reward that
      depends on hidden state, so that part of every TD target was
      unlearnable noise.
    """
    x, y, theta = get_car_xy_heading(data)
    c, s = jnp.cos(theta), jnp.sin(theta)

    vx_w, vy_w = data.qvel[0], data.qvel[1]
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w

    # Lidar for the OBSERVATION (directional); true wall distances are
    # still returned for the reward's safety term and collision checks.
    obstacle_d = wall_distances(jnp.array([x, y]), walls)
    ranges = lidar_scan(jnp.array([x, y]), theta, walls,
                        n_rays=N_LIDAR, max_range=LIDAR_MAX_RANGE)
    ranges_n = ranges / LIDAR_MAX_RANGE

    dx = goal[0] - x
    dy = goal[1] - y
    dist = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-9)
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    dir_x = bx / dist
    dir_y = by / dist
    dist_n = jnp.clip(dist, 0.0, OBS_GOAL_DIST_SCALE) / OBS_GOAL_DIST_SCALE

    obs = jnp.concatenate([
        ranges_n,                                                      # N_LIDAR
        jnp.stack([
            jnp.clip(vx_b, -OBS_V_SCALE, OBS_V_SCALE) / OBS_V_SCALE,   # 1
            jnp.clip(vy_b, -OBS_V_SCALE, OBS_V_SCALE) / OBS_V_SCALE,   # 1
            s, c,                                                      # 2
            dir_x, dir_y,                                              # 2
            dist_n,                                                    # 1
        ]),
        prev_action,                                                   # 2
        prev_prev_action,                                              # 2
    ])
    return obs, x, y, theta, obstacle_d


def env_step_batched(policy_params, mjx_model, walls, states, key, horizon, min_dist, max_dist,
                      gamma, fresh_data, action_repeat, goal_cone):
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
        obs, x, y, theta, obstacle_d = build_obs_sac(
            data_i, goal_i, walls, prev_action_i, prev_prev_action_i
        )
        prev_goal_dist = jnp.sqrt((goal_i[0] - x) ** 2 + (goal_i[1] - y) ** 2 + 1e-9)

        action, _ = sample_action(policy_params, obs, act_key_i)
        ctrl = jnp.array([STEER_RANGE * action[0], action[1], action[1]])

        # Action repeat. The physics timestep is 0.002s, so one action per
        # physics step meant a decision rate of 500Hz -- two separate
        # problems. (1) gamma=0.99 per decision is an effective horizon of
        # ~100 decisions = 0.2s, but reaching a goal metres away takes
        # 2-4s, so the discount could not even represent the task. (2) SAC
        # explores by sampling an independent Gaussian action every
        # decision; resampled every 2ms those draws just average out and
        # the car jitters in place instead of committing to a direction --
        # consistent with the observed 0.02-0.04 m/s creep. Repeating each
        # action gives a decision rate near what the real MuSHR controller
        # runs at (~25Hz at repeat=20), makes gamma=0.99 cover the whole
        # episode, and cuts transitions-per-second of sim by the same
        # factor.
        def repeat_body(d, _):
            d = d.replace(ctrl=ctrl)
            d = mjx.step(mjx_model, d)
            d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
            return d, None

        new_data, _ = jax.lax.scan(repeat_body, data_i, None, length=action_repeat)

        next_obs, x2, y2, theta2, obstacle_d2 = build_obs_sac(
            new_data, goal_i, walls, action, prev_action_i
        )
        goal_dist2 = jnp.sqrt((goal_i[0] - x2) ** 2 + (goal_i[1] - y2) ** 2 + 1e-9)

        reward, _ = jax_reward(
            jnp.array([x2, y2]), theta2, action, prev_action_i, prev_prev_action_i,
            goal_i, obstacle_d2, prev_goal_dist,
            weights=SAC_REWARD_WEIGHTS,
        )

        success = goal_dist2 < SUCCESS_DIST
        timeout = (step_count_i + 1) >= horizon
        reset = success | timeout
        # bootstrap mask: only true success zeroes future value; timeout is
        # an artificial cutoff, not a real terminal state, so we still
        # bootstrap through it (standard truncation-vs-termination handling).
        done_for_bootstrap = success

        # Terminal bonus + per-decision time cost. Without the bonus,
        # zeroing the bootstrap on success makes reaching the goal a pure
        # loss of future value, so the optimal policy is to approach and
        # then never touch it. See SAC_REWARD_WEIGHTS.
        reward = reward + TIME_COST + jnp.where(success, TERMINAL_BONUS, 0.0)

        spawn, yaw, new_goal_sample = sample_spawn_and_goal(
            goal_key_i, walls, min_dist, max_dist, goal_cone
        )
        respawned = reset_data_at(fresh_data, spawn, yaw)

        out_data = jax.tree_util.tree_map(
            lambda f, n: jnp.where(reset, f, n), respawned, new_data
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
                obs, action, reward, next_obs, done, key, gamma, tau, target_entropy, lr, alpha_lr):
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

    # sac_update had no gradient sanitization/clipping at all, unlike
    # DiffRL's rollout (which needs it because BPTT-through-contact
    # gradients can blow up to ~1e25). SAC gradients are normally much
    # better-behaved, but the Q-value miscalibration/actor-saturation
    # issues already found here show real numerical instability can still
    # happen (e.g. a bad bootstrap target from an undertrained critic) --
    # this is cheap insurance against a single bad batch corrupting params.
    critic_loss, (q1_grads, q2_grads) = jax.value_and_grad(critic_loss_fn, argnums=(0, 1))(q1_params, q2_params)
    q1_grads = clip_tree(sanitize_grads(q1_grads))
    q2_grads = clip_tree(sanitize_grads(q2_grads))
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
    policy_grads = clip_tree(sanitize_grads(policy_grads))
    policy_params, policy_opt = adam_update(policy_params, policy_grads, policy_opt, lr=lr)

    def alpha_loss_fn(log_alpha):
        return -jnp.mean(log_alpha * jax.lax.stop_gradient(log_prob + target_entropy))

    # alpha_lr is independent of the critic/actor lr: when updates_per_step
    # scales with n_envs (fixing the earlier 16x-too-low UTD ratio), alpha
    # gets the same 16x more gradient steps per unit of real env experience
    # as everything else, so it converges to near-zero (killing exploration)
    # within the first ~10% of a run instead of decaying over the full
    # curriculum -- confirmed via a run that flatlined at 10% success for
    # its last 130k/150k steps once alpha collapsed by step 20k. Scaling
    # alpha's own lr down by the same factor keeps its decay paced to real
    # env-steps instead of gradient-step count.
    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_grad = sanitize_grads(alpha_grad)
    log_alpha, alpha_opt = adam_update(log_alpha, alpha_grad, alpha_opt, lr=alpha_lr)

    q1_target = jax.tree_util.tree_map(lambda t, s: tau * s + (1 - tau) * t, q1_target, q1_params)
    q2_target = jax.tree_util.tree_map(lambda t, s: tau * s + (1 - tau) * t, q2_target, q2_params)

    return (policy_params, q1_params, q2_params, q1_target, q2_target, log_alpha,
            policy_opt, q1_opt, q2_opt, alpha_opt, critic_loss, actor_loss, alpha_loss)


def evaluate_sac(policy_params, mjx_model, walls, horizon, n_eval, min_dist, max_dist,
                  action_repeat, goal_cone, seed=999):
    g = jax.random.PRNGKey(seed)
    # Evaluate from randomized start poses too, matching training. A
    # fixed-origin eval would only ever score one pose on one map.
    spawns, yaws, goals = jax.vmap(
        lambda k: sample_spawn_and_goal(k, walls, min_dist, max_dist, goal_cone)
    )(jax.random.split(g, n_eval))

    fresh = mjx.make_data(mjx_model)

    def rollout_one(spawn, yaw, goal):
        data = reset_data_at(fresh, spawn, yaw)

        def step(carry, _):
            data, prev_action, prev_prev_action, min_dist_seen, min_obs_seen = carry
            obs, x, y, theta, obstacle_d = build_obs_sac(
                data, goal, walls, prev_action, prev_prev_action
            )
            action = deterministic_action(policy_params, obs)
            ctrl = jnp.array([STEER_RANGE * action[0], action[1], action[1]])

            def repeat_body(d, _):
                d = d.replace(ctrl=ctrl)
                d = mjx.step(mjx_model, d)
                d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
                return d, None

            data, _ = jax.lax.scan(repeat_body, data, None, length=action_repeat)

            _, x2, y2, _, obstacle_d2 = build_obs_sac(data, goal, walls, action, prev_action)
            goal_dist = jnp.sqrt((goal[0] - x2) ** 2 + (goal[1] - y2) ** 2)
            min_dist_seen = jnp.minimum(min_dist_seen, goal_dist)
            min_obs_seen = jnp.minimum(min_obs_seen, jnp.min(obstacle_d2))
            return (data, action, prev_action, min_dist_seen, min_obs_seen), None

        init_dist = jnp.sqrt((goal[0] - spawn[0]) ** 2 + (goal[1] - spawn[1]) ** 2)
        carry0 = (data, jnp.zeros(2), jnp.zeros(2), init_dist, jnp.array(jnp.inf))
        (data, _, _, min_dist_seen, min_obs_dist), _ = jax.lax.scan(
            step, carry0, None, length=horizon
        )
        return min_dist_seen, min_obs_dist

    min_dists, min_obs_dists = jax.vmap(rollout_one)(spawns, yaws, goals)
    # match training's actual success semantics: did the car ever get
    # within SUCCESS_DIST, not just where it happened to be at the very
    # last timestep (the old final-position-only check was undercounting
    # real successes whenever the car reached the goal then drove past it
    # with no stopping/holding behavior, since nothing in training
    # rewards *staying* at the goal once reached).
    success = jnp.mean(min_dists < SUCCESS_DIST)
    mean_dist = jnp.mean(min_dists)
    collisions = jnp.sum(min_obs_dists < CAR_RADIUS)
    return mean_dist, collisions, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=50000,
                         help="Total environment DECISIONS (= replay transitions), "
                              "not physics steps; each decision advances the sim by "
                              "action_repeat physics steps.")
    parser.add_argument("--horizon", type=int, default=150,
                         help="Episode length in DECISIONS. At action_repeat=20 and a "
                              "0.002s physics timestep, 150 decisions = 6 seconds of "
                              "sim time.")
    parser.add_argument("--action-repeat", type=int, default=20,
                         help="Physics steps per policy decision. 20 gives a ~25Hz "
                              "decision rate, close to the real MuSHR controller, and "
                              "makes gamma=0.99 span the whole episode instead of 0.2s.")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1,
                         help="Gradient updates per env transition PER ENV, i.e. total "
                              "updates each outer iteration = updates_per_step * n_envs. "
                              "Standard SAC practice is ~1 update per single transition "
                              "(UTD ratio ~1) -- with n_envs parallel envs collecting "
                              "n_envs transitions per iteration, that means n_envs updates "
                              "per iteration, not 1.")
    parser.add_argument("--max-dist", type=float, default=3.0)
    parser.add_argument("--min-dist", type=float, default=1.0)
    parser.add_argument("--alpha-lr-scale", type=float, default=1.0,
                         help="entropy-temperature lr as a multiple of --lr. "
                              "1.0 is standard SAC.")
    parser.add_argument("--goal-cone", type=float, default=1.05,
                         help="half-angle (rad) of the forward cone goals are drawn "
                              "from. ~1.05 = +-60 deg. pi would be all directions, "
                              "which puts many goals inside the car's minimum turning "
                              "circle (see sample_goals_cone).")
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
    obs_dim = N_LIDAR + OBS_EXTRA_DIM_SAC
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
            policy_params, mjx_model, walls, states, key, args.horizon, min_dist, max_dist,
            args.gamma, fresh_data, args.action_repeat, args.goal_cone
        )
    )
    sac_update_jit = jax.jit(
        lambda *a, **kw: sac_update(*a, gamma=args.gamma, tau=args.tau, target_entropy=target_entropy, lr=args.lr,
                                     alpha_lr=args.lr * args.alpha_lr_scale, **kw)
    )
    eval_jit = jax.jit(
        lambda pp, min_dist, max_dist: evaluate_sac(
            pp, mjx_model, walls, args.horizon, args.eval_n, min_dist, max_dist,
            args.action_repeat, args.goal_cone
        )
    )

    n_envs = args.n_envs
    _init_spawns, _init_yaws, _init_goals = jax.vmap(
        lambda k: sample_spawn_and_goal(k, walls, args.min_dist,
                                        args.min_dist + 0.3, args.goal_cone)
    )(jax.random.split(jax.random.PRNGKey(args.seed + 1), n_envs))
    states = {
        "data": jax.vmap(lambda sp, yw: reset_data_at(fresh_data, sp, yw))(
            _init_spawns, _init_yaws),
        "goal": _init_goals,
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
        max_dist = args.min_dist + progress * (args.max_dist - args.min_dist)
        min_dist = args.min_dist

        key, step_key = jax.random.split(key)
        states, transitions = env_step_jit(policy_params, states, step_key, min_dist, max_dist)
        obs_b, action_b, reward_b, next_obs_b, done_b = jax.device_get(transitions)
        buffer.add_batch(obs_b, action_b, reward_b, next_obs_b, done_b)
        total_env_steps += n_envs
        it += 1

        if buffer.size >= max(args.warmup_steps, args.batch_size):
            for _ in range(args.updates_per_step * n_envs):
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
            eval_dist, eval_collisions, eval_success = eval_jit(policy_params, args.min_dist, args.max_dist)
            eval_success_f = float(eval_success)
            alpha_display = float(jnp.exp(log_alpha))
            print(f"steps {total_env_steps:7d}  progress={progress:.2f}  goal_range=[{min_dist:.1f},{max_dist:.1f}]  "
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
    eval_dist, eval_collisions, eval_success = eval_jit(policy_params, args.min_dist, args.max_dist)
    print(f"LARGE-SAMPLE final eval: success={float(eval_success)*100:.1f}%  mean_dist={float(eval_dist):.3f}")
    np.savez("sac_policy_final.npz",
             **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(policy_params)},
             **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(policy_params)})


if __name__ == "__main__":
    main()
