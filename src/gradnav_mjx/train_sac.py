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
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

import mjx_solver_patch
mjx_solver_patch.apply()

from mjx_car_scene import build_car_scene_xml, STEER_RANGE
from mjx_random_maps import (generate_map_set, generate_room_set,
                              generate_room_layout)
from mjx_room_scene import (build_room_scene_xml, HUMAN_HEIGHT,
                             HUMAN_RADIUS)
from mjx_pedestrians import (make_patrol_params, pedestrian_circles,
                              pedestrian_mocap_pos)
from mjx_obstacle_dist import wall_distances, obstacle_distances
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
    obstacle=2.5,
    out_of_map=-1.0,
)
# obstacle 0.5 -> 2.5: with wall and pedestrian hits finally reported
# apart, the first room runs read wall_hits=10, ped_hits=0 -- the car
# avoids people but drives into walls, and the safety term was too weak
# to matter next to progress=10.0 per metre.
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
# Set from --max-dist in main(): a goal beyond this saturates the
# distance input to exactly 1.0 and the policy can no longer tell 8m
# from 14m -- the same clipping bug that silently broke the first 6m
# attempts, just at a different threshold.
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


SPAWN_HALF = 3.0        # spawn box half-extent (m); overridden in main()
SPAWN_CLEAR = 0.8       # required clearance from any wall at spawn (m)
GOAL_CLEAR = 0.6        # required clearance from any wall at the goal (m)
SPAWN_CANDIDATES = 8
CONE_FULL_DIST = 3.0    # beyond this, goals may sit in ANY direction
CURRICULUM_START_MIN = 1.0   # goal range at the start of training
CURRICULUM_START_MAX = 2.5


def cone_for_distance(dist, cone_min):
    """Allowed goal bearing half-angle, widening with distance.

    A fixed narrow cone means the car only ever drives roughly straight.
    A fixed wide one re-creates the unreachability problem: the car's
    minimum turning radius is ~0.6m slow and ~1.9m at speed, so a NEAR
    goal off to the side sits inside the turning circle and needs a
    multi-point turn. That constraint only binds at short range -- a
    goal several metres away is reachable at any bearing, since the car
    can simply turn toward it first. So scale the cone with distance:
    +-cone_min at 1m, all directions by CONE_FULL_DIST. Long goals then
    genuinely require turning, without making short ones impossible.
    """
    return jnp.clip(jnp.pi * dist / CONE_FULL_DIST, cone_min, jnp.pi)


def sample_spawn_and_goal(key, walls, min_dist, max_dist, cone, bound=1e9):
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
    spawn = cand[jnp.argmax(clearance > SPAWN_CLEAR)]

    yaw = jax.random.uniform(k_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
    dist = jax.random.uniform(k_dist, (SPAWN_CANDIDATES,),
                              minval=min_dist, maxval=max_dist)
    half = cone_for_distance(dist, cone)
    u = jax.random.uniform(k_ang, (SPAWN_CANDIDATES,), minval=-1.0, maxval=1.0)
    ang = u * half

    # goals in the car's frame, rotated into world coords
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    lx, ly = dist * jnp.cos(ang), dist * jnp.sin(ang)
    cands = spawn[None, :] + jnp.stack([c * lx - s * ly, s * lx + c * ly], axis=-1)
    # Reject goals buried inside a wall -- otherwise a share of episodes
    # are unreachable no matter how good the policy is, which is exactly
    # how the goal-geometry ceiling went unnoticed before.
    goal_clear = jnp.min(wall_distances(cands, walls), axis=-1)
    # ...and inside the room: an enclosed arena means a goal sampled
    # past the perimeter is unreachable by construction.
    inside = (jnp.abs(cands[:, 0]) < bound) & (jnp.abs(cands[:, 1]) < bound)
    goal = cands[jnp.argmax((goal_clear > GOAL_CLEAR) & inside)]
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


def _peds_at(t, ped_params, car_xy=None):
    """(H,3) pedestrian footprints at time t, or None if the room has
    no people (so the static-room path costs nothing)."""
    if ped_params is None:
        return None
    return pedestrian_circles(t, ped_params, car_xy)


def _move_peds(d, t, ped_params, ped_z):
    """Write the pedestrians' current pose into mocap_pos so the physics
    agrees with what the lidar sensed. People give way to the car, so
    their pose depends on where the car currently is."""
    if ped_params is None:
        return d
    car_xy = d.qpos[0:2]
    return d.replace(
        mocap_pos=pedestrian_mocap_pos(t, ped_params, ped_z, car_xy))


def make_fresh_data(mjx_model):
    return mjx.make_data(mjx_model)


# obs = N_LIDAR ranges + [vx_b, vy_b, sin, cos, dir_x, dir_y, dist]
#       + prev_action(2) + prev_prev_action(2)
N_LIDAR = 16
LIDAR_MAX_RANGE = 10.0
# Two stacked scans: current and previous. A single scan says WHERE
# obstacles are but not which way they are moving, so with walking
# people the policy cannot distinguish someone stepping into its path
# from someone clearing it, and reacts late. Stacking makes range-rate
# (hence pedestrian motion) inferable from the observation.
OBS_EXTRA_DIM_SAC = 11
OBS_DIM_SAC = 2 * N_LIDAR + OBS_EXTRA_DIM_SAC


def build_obs_sac(data, goal, walls, peds, prev_action, prev_prev_action,
                   prev_scan):
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
    obstacle_d = obstacle_distances(jnp.array([x, y]), walls, peds)
    ranges = lidar_scan(jnp.array([x, y]), theta, walls, peds,
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
        prev_scan,                                                     # N_LIDAR
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
    return obs, x, y, theta, obstacle_d, ranges_n


def env_step_batched(policy_params, mjx_model, walls, states, key, horizon, min_dist, max_dist,
                      gamma, fresh_data, action_repeat, goal_cone, arena_size, goal_bound,
                      ped_params=None, ped_z=0.0, phys_dt=0.002):
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
    prev_scan = states["prev_scan"]
    n_envs = goal.shape[0]
    decision_dt = action_repeat * phys_dt

    act_key, goal_key = jax.random.split(key)
    act_keys = jax.random.split(act_key, n_envs)
    goal_keys = jax.random.split(goal_key, n_envs)

    def single(data_i, goal_i, prev_action_i, prev_prev_action_i, step_count_i,
                prev_scan_i, act_key_i, goal_key_i):
        # episode clock drives the pedestrians; they are a pure
        # function of time, so nothing extra rides in the buffer
        t0 = step_count_i * decision_dt
        peds0 = _peds_at(t0, ped_params, data_i.qpos[0:2])
        obs, x, y, theta, obstacle_d, scan = build_obs_sac(
            data_i, goal_i, walls, peds0, prev_action_i, prev_prev_action_i,
            prev_scan_i
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
        def repeat_body(d, k):
            # walk the pedestrians every physics step so they move
            # smoothly through the action-repeat block
            d = _move_peds(d, t0 + k * phys_dt, ped_params, ped_z)
            d = d.replace(ctrl=ctrl)
            d = mjx.step(mjx_model, d)
            d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
            return d, None

        new_data, _ = jax.lax.scan(repeat_body, data_i,
                                    jnp.arange(action_repeat, dtype=jnp.float32))

        t1 = t0 + decision_dt
        next_obs, x2, y2, theta2, obstacle_d2, scan2 = build_obs_sac(
            new_data, goal_i, walls, _peds_at(t1, ped_params, new_data.qpos[0:2]), action,
            prev_action_i, scan
        )
        goal_dist2 = jnp.sqrt((goal_i[0] - x2) ** 2 + (goal_i[1] - y2) ** 2 + 1e-9)

        reward, _ = jax_reward(
            jnp.array([x2, y2]), theta2, action, prev_action_i, prev_prev_action_i,
            goal_i, obstacle_d2, prev_goal_dist,
            arena_size=arena_size,
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
            goal_key_i, walls, min_dist, max_dist, goal_cone, goal_bound
        )
        respawned = reset_data_at(fresh_data, spawn, yaw)

        out_data = jax.tree_util.tree_map(
            lambda f, n: jnp.where(reset, f, n), respawned, new_data
        )
        out_goal = jnp.where(reset, new_goal_sample, goal_i)
        out_prev_action = jnp.where(reset, jnp.zeros(2), action)
        out_prev_prev_action = jnp.where(reset, jnp.zeros(2), prev_action_i)
        out_step_count = jnp.where(reset, 0, step_count_i + 1)
        out_scan = jnp.where(reset, jnp.ones_like(scan2), scan2)

        return (out_data, out_goal, out_prev_action, out_prev_prev_action,
                out_step_count, out_scan), \
               (obs, action, reward, next_obs, done_for_bootstrap.astype(jnp.float32))

    new_carry, transitions = jax.vmap(single)(
        data, goal, prev_action, prev_prev_action, step_count, prev_scan,
        act_keys, goal_keys
    )
    (new_data, new_goal, new_prev_action, new_prev_prev_action,
     new_step_count, new_scan) = new_carry
    new_states = {
        "data": new_data, "goal": new_goal, "prev_action": new_prev_action,
        "prev_prev_action": new_prev_prev_action, "step_count": new_step_count,
        "prev_scan": new_scan,
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
                  action_repeat, goal_cone, goal_bound=1e9, ped_params=None,
                  ped_z=0.0, phys_dt=0.002, seed=999):
    g = jax.random.PRNGKey(seed)
    # Evaluate from randomized start poses too, matching training. A
    # fixed-origin eval would only ever score one pose on one map.
    spawns, yaws, goals = jax.vmap(
        lambda k: sample_spawn_and_goal(k, walls, min_dist, max_dist, goal_cone, goal_bound)
    )(jax.random.split(g, n_eval))

    fresh = mjx.make_data(mjx_model)

    def rollout_one(spawn, yaw, goal):
        data = reset_data_at(fresh, spawn, yaw)

        decision_dt = action_repeat * phys_dt

        def step(carry, i):
            (data, prev_action, prev_prev_action, prev_scan,
             min_dist_seen, min_wall_seen, min_ped_seen) = carry
            t0 = i * decision_dt
            obs, x, y, theta, obstacle_d, scan = build_obs_sac(
                data, goal, walls, _peds_at(t0, ped_params, data.qpos[0:2]), prev_action,
                prev_prev_action, prev_scan
            )
            action = deterministic_action(policy_params, obs)
            ctrl = jnp.array([STEER_RANGE * action[0], action[1], action[1]])

            def repeat_body(d, k):
                d = _move_peds(d, t0 + k * phys_dt, ped_params, ped_z)
                d = d.replace(ctrl=ctrl)
                d = mjx.step(mjx_model, d)
                d = d.replace(qvel=jnp.clip(d.qvel, -QVEL_CLAMP, QVEL_CLAMP))
                return d, None

            data, _ = jax.lax.scan(repeat_body, data,
                                    jnp.arange(action_repeat, dtype=jnp.float32))

            _, x2, y2, _, obstacle_d2, scan2 = build_obs_sac(
                data, goal, walls,
                _peds_at(t0 + decision_dt, ped_params, data.qpos[0:2]),
                action, prev_action, scan)
            goal_dist = jnp.sqrt((goal[0] - x2) ** 2 + (goal[1] - y2) ** 2)
            min_dist_seen = jnp.minimum(min_dist_seen, goal_dist)
            # Walls and people are scored apart. A wall collision is
            # unambiguously the car's doing; a person who walks into the
            # car is not the same event, and lumping them together made
            # the metric read 20/20 for a hand-written controller and a
            # trained policy alike -- i.e. measure nothing.
            car_xy2 = jnp.array([x2, y2])
            min_wall_seen = jnp.minimum(
                min_wall_seen, jnp.min(wall_distances(car_xy2, walls)))
            peds2 = _peds_at(t0 + decision_dt, ped_params, car_xy2)
            if peds2 is None:
                ped_clear = jnp.array(jnp.inf)
            else:
                ped_clear = jnp.min(
                    jnp.linalg.norm(car_xy2[None, :] - peds2[:, :2], axis=-1)
                    - peds2[:, 2])
            min_ped_seen = jnp.minimum(min_ped_seen, ped_clear)
            return (data, action, prev_action, scan2,
                    min_dist_seen, min_wall_seen, min_ped_seen), None

        init_dist = jnp.sqrt((goal[0] - spawn[0]) ** 2 + (goal[1] - spawn[1]) ** 2)
        carry0 = (data, jnp.zeros(2), jnp.zeros(2), jnp.ones(N_LIDAR),
                  init_dist, jnp.array(jnp.inf), jnp.array(jnp.inf))
        (data, _, _, _, min_dist_seen, min_wall, min_ped), _ = jax.lax.scan(
            step, carry0, jnp.arange(horizon, dtype=jnp.float32)
        )
        return min_dist_seen, min_wall, min_ped

    min_dists, min_walls, min_peds = jax.vmap(rollout_one)(spawns, yaws, goals)
    # match training's actual success semantics: did the car ever get
    # within SUCCESS_DIST, not just where it happened to be at the very
    # last timestep (the old final-position-only check was undercounting
    # real successes whenever the car reached the goal then drove past it
    # with no stopping/holding behavior, since nothing in training
    # rewards *staying* at the goal once reached).
    success = jnp.mean(min_dists < SUCCESS_DIST)
    mean_dist = jnp.mean(min_dists)
    wall_hits = jnp.sum(min_walls < CAR_RADIUS)
    ped_hits = jnp.sum(min_peds < CAR_RADIUS)
    return mean_dist, wall_hits, ped_hits, success


def save_train_state(path, state):
    """Full training state, so a killed run can be resumed rather than
    restarted. This sandbox has repeatedly killed long runs partway
    through; without this, every restart threw away all progress.

    The replay buffer is deliberately not saved (it is ~100MB and
    refills quickly); the networks, their target copies, the optimiser
    moments and the entropy temperature are what actually carry the
    learning.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(state), f)
    os.replace(tmp, path)     # atomic: a kill mid-write cannot corrupt it


def load_train_state(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, state)


def make_env(seed, room=True, room_size=8.0, n_inner=6, n_humans=6,
              n_walls=4, map_size=6.0, spawn_half=6.5, max_dist=10.0):
    """Build the environment. Single source of truth, used by training,
    eval and tracing alike.

    Previously each script built its own scene, and they drifted: an
    evaluator was still constructing the old open-field map while the
    checkpoint had been trained in a room, which silently scores a
    policy on a world it never saw. Anything that loads a policy should
    call this with the same arguments the run used.
    """
    global SPAWN_HALF, OBS_GOAL_DIST_SCALE
    SPAWN_HALF = spawn_half
    OBS_GOAL_DIST_SCALE = max_dist * 1.6
    arena_size = spawn_half + max_dist + 4.0
    ped_params, ped_z, humans_list = None, 0.0, []

    if room:
        walls_list, humans_list = generate_room_layout(
            room_half=room_size, n_inner=n_inner, n_humans=n_humans, seed=seed)
        SPAWN_HALF = min(spawn_half, room_size - 1.5)
        goal_bound = room_size - 1.0
        arena_size = room_size + 2.0
        if humans_list:
            import random as _random
            ped_params = make_patrol_params(humans_list, _random.Random(seed),
                                             room_half=room_size)
            ped_z = HUMAN_HEIGHT * 0.39 + HUMAN_RADIUS
        scene_path = build_room_scene_xml(walls_list, humans_list,
                                           out_path=f"sac_room_{seed}.xml",
                                           room_half=room_size)
    else:
        walls_list = generate_map_set(n_maps=1, n_walls=n_walls,
                                       size=map_size, base_seed=seed)[0]
        goal_bound = 1e9
        scene_path = build_car_scene_xml(walls_list, out_path=f"sac_map_{seed}.xml",
                                          arena_size=arena_size + 4.0)

    model = mujoco.MjModel.from_xml_path(scene_path)
    return dict(
        mjx_model=mjx.put_model(model),
        walls=jnp.array(walls_list),
        walls_list=walls_list,
        humans_list=humans_list,
        ped_params=ped_params,
        ped_z=ped_z,
        goal_bound=goal_bound,
        arena_size=arena_size,
        spawn_half=SPAWN_HALF,
    )


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
    parser.add_argument("--map-size", type=float, default=6.0,
                         help="half-extent (m) the wall field is scattered over "
                              "(legacy open-field maps; ignored when --room is set)")
    parser.add_argument("--room", action="store_true",
                         help="use an enclosed room (perimeter walls + interior "
                              "obstacles) instead of free-floating sticks on an "
                              "unbounded floor")
    parser.add_argument("--room-size", type=float, default=8.0,
                         help="room half-extent (m); 8.0 = a 16x16m room")
    parser.add_argument("--n-inner", type=int, default=4,
                         help="interior walls inside the room")
    parser.add_argument("--n-humans", type=int, default=3,
                         help="walking people inside the room (0 = none)")
    parser.add_argument("--spawn-half", type=float, default=3.0,
                         help="half-extent (m) of the box the car spawns in")
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", default="sac_train_state.pkl",
                         help="rolling full-state checkpoint for --resume")
    parser.add_argument("--resume", action="store_true",
                         help="continue from --ckpt if it exists")
    args = parser.parse_args()

    env = make_env(args.seed, room=args.room, room_size=args.room_size,
                    n_inner=args.n_inner, n_humans=args.n_humans,
                    n_walls=args.n_walls, map_size=args.map_size,
                    spawn_half=args.spawn_half, max_dist=args.max_dist)
    mjx_model = env["mjx_model"]
    walls = env["walls"]
    walls_list, humans_list = env["walls_list"], env["humans_list"]
    ped_params, ped_z = env["ped_params"], env["ped_z"]
    goal_bound, arena_size = env["goal_bound"], env["arena_size"]
    print(f"Built SAC map: {len(walls_list)} walls, {len(humans_list)} people "
          f"({'room ' + str(2*args.room_size) + 'm sq' if args.room else 'open field'}), "
          f"spawn +-{env['spawn_half']:.1f}m, "
          f"goals {args.min_dist}-{args.max_dist}m, "
          f"cone +-{args.goal_cone:.2f}rad at 1m widening to all directions "
          f"by {CONE_FULL_DIST}m")

    key = jax.random.PRNGKey(args.seed)
    pkey, q1key, q2key = jax.random.split(key, 3)
    obs_dim = OBS_DIM_SAC
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

    start_steps = 0
    if args.resume and os.path.exists(args.ckpt):
        st = load_train_state(args.ckpt)
        (policy_params, q1_params, q2_params, q1_target, q2_target, log_alpha,
         policy_opt, q1_opt, q2_opt, alpha_opt) = (
            st["policy"], st["q1"], st["q2"], st["q1t"], st["q2t"], st["log_alpha"],
            st["policy_opt"], st["q1_opt"], st["q2_opt"], st["alpha_opt"])
        start_steps = int(st["steps"])
        print(f"resumed from {args.ckpt} at {start_steps} steps "
              f"(alpha={float(jnp.exp(log_alpha)):.3f})")

    fresh_data = make_fresh_data(mjx_model)

    env_step_jit = jax.jit(
        lambda policy_params, states, key, min_dist, max_dist: env_step_batched(
            policy_params, mjx_model, walls, states, key, args.horizon, min_dist, max_dist,
            args.gamma, fresh_data, args.action_repeat, args.goal_cone, arena_size,
            goal_bound, ped_params, ped_z
        )
    )
    sac_update_jit = jax.jit(
        lambda *a, **kw: sac_update(*a, gamma=args.gamma, tau=args.tau, target_entropy=target_entropy, lr=args.lr,
                                     alpha_lr=args.lr * args.alpha_lr_scale, **kw)
    )
    eval_jit = jax.jit(
        lambda pp, min_dist, max_dist: evaluate_sac(
            pp, mjx_model, walls, args.horizon, args.eval_n, min_dist, max_dist,
            args.action_repeat, args.goal_cone, goal_bound, ped_params, ped_z
        )
    )

    n_envs = args.n_envs
    _init_spawns, _init_yaws, _init_goals = jax.vmap(
        lambda k: sample_spawn_and_goal(k, walls, args.min_dist,
                                        args.min_dist + 0.3, args.goal_cone, goal_bound)
    )(jax.random.split(jax.random.PRNGKey(args.seed + 1), n_envs))
    states = {
        "data": jax.vmap(lambda sp, yw: reset_data_at(fresh_data, sp, yw))(
            _init_spawns, _init_yaws),
        "goal": _init_goals,
        "prev_action": jnp.zeros((n_envs, 2)),
        "prev_prev_action": jnp.zeros((n_envs, 2)),
        "step_count": jnp.zeros((n_envs,), dtype=jnp.int32),
        # 1.0 = "nothing within lidar range", the right prior for the
        # first step of an episode when there is no previous scan yet
        "prev_scan": jnp.ones((n_envs, N_LIDAR)),
    }

    buffer = ReplayBuffer(args.buffer_size, obs_dim)
    rng = np.random.default_rng(args.seed)

    best_success = -1.0
    total_env_steps = start_steps
    it = 0
    while total_env_steps < args.total_steps:
        progress = min(1.0, total_env_steps / (args.total_steps * 0.7))
        # Ramp BOTH ends of the goal range, not just the far end. With a
        # fixed --min-dist of 5m every episode was maximally hard from
        # step one, so the policy never got the easy early wins that let
        # it discover "drive at the goal" before having to also solve
        # routing around walls. The runs that actually converged (90% at
        # 1-6m) all started near 1m and grew. Evaluation is unaffected --
        # it always scores the full --min-dist..--max-dist target range.
        min_dist = CURRICULUM_START_MIN + progress * (args.min_dist - CURRICULUM_START_MIN)
        max_dist = CURRICULUM_START_MAX + progress * (args.max_dist - CURRICULUM_START_MAX)

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
            eval_dist, eval_wall, eval_ped, eval_success = eval_jit(
                policy_params, args.min_dist, args.max_dist)
            eval_success_f = float(eval_success)
            alpha_display = float(jnp.exp(log_alpha))
            print(f"steps {total_env_steps:7d}  progress={progress:.2f}  goal_range=[{min_dist:.1f},{max_dist:.1f}]  "
                  f"alpha={alpha_display:.3f}  buffer={buffer.size}  "
                  f"EVAL_dist={float(eval_dist):.3f}  EVAL_success={eval_success_f*100:.0f}%  "
                  f"wall_hits={int(eval_wall)}/{args.eval_n}  "
                  f"ped_hits={int(eval_ped)}/{args.eval_n}")
            if eval_success_f > best_success:
                best_success = eval_success_f
                np.savez("sac_policy_best.npz",
                         **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(policy_params)},
                         **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(policy_params)})
                print(f"         -> new best checkpoint (success={best_success*100:.0f}%) saved")
            save_train_state(args.ckpt, dict(
                policy=policy_params, q1=q1_params, q2=q2_params,
                q1t=q1_target, q2t=q2_target, log_alpha=log_alpha,
                policy_opt=policy_opt, q1_opt=q1_opt, q2_opt=q2_opt,
                alpha_opt=alpha_opt, steps=total_env_steps))

    print(f"\nDone. Best checkpoint: success={best_success*100:.0f}%")
    eval_dist, eval_wall, eval_ped, eval_success = eval_jit(
        policy_params, args.min_dist, args.max_dist)
    print(f"LARGE-SAMPLE final eval: success={float(eval_success)*100:.1f}%  "
          f"mean_dist={float(eval_dist):.3f}  wall_hits={int(eval_wall)}  "
          f"ped_hits={int(eval_ped)}")
    np.savez("sac_policy_final.npz",
             **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(policy_params)},
             **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(policy_params)})


if __name__ == "__main__":
    main()
