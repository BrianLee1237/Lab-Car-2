"""
car_env.py (v2)

Same purpose as before -- a Gymnasium environment wrapping the randomized
MuJoCo maps -- but now with:

  1. Lidar-style sensing: N raycasts around the car, distance to the
     nearest obstacle (wall or pedestrian) per ray. This replaces the
     placeholder and mirrors the *idea* of GRaD-Nav++'s min-pooled depth
     observation (d_t), adapted from a camera to a lidar ring since we
     don't have a rendered camera feed here.

  2. A real reward function, adapted from GRaD-Nav++ Table I:
       - survival bonus (alive each step)
       - action penalty (discourage large throttle/steering)
       - action-rate penalty (discourage jerky control)
       - waypoint/goal reward: exp(-distance to goal)
       - obstacle-avoidance penalty: penalized when nearest lidar hit
         (wall OR pedestrian) is closer than a safety threshold
       - out-of-bounds penalty
     Dropped from the original: height/pose terms (drone-specific, not
     applicable to a ground vehicle).

  3. A single goal point per episode (opposite corner of the arena) is
     used in place of GRaD-Nav++'s multi-waypoint reference trajectory,
     since real path planning is being handled by a separate package
     (per Tarun) -- this reward just needs *a* target to shape learning
     while that integration happens later.

NOTE: this still does not have real actuators (steering-constrained
bicycle dynamics) -- throttle/steering map to a simplified force/torque.
That's the next realism upgrade after the reward function itself is
confirmed to shape sensible behavior.
"""

import glob
import os
import random

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

N_LIDAR_RAYS = 16
LIDAR_MAX_RANGE = 8.0
COLLISION_THRESHOLD = 0.4   # meters -- closer than this to any obstacle = collision
OBSTACLE_SAFETY_DIST = 1.5  # meters -- start penalizing closeness within this range


class RandomMapCarEnv(gym.Env):
    def __init__(self, maps_dir="maps", max_steps=500, arena_size=10.0):
        super().__init__()
        self.map_paths = sorted(glob.glob(os.path.join(maps_dir, "*.xml")))
        if not self.map_paths:
            raise FileNotFoundError(f"No .xml maps found in '{maps_dir}/' -- run generate_map_batch.py first")

        self.max_steps = max_steps
        self.arena_size = arena_size

        # obs: [yaw, vx, vy, yaw_rate, goal_dx, goal_dy, lidar(N_LIDAR_RAYS), prev_action(2)]
        obs_dim = 4 + 2 + N_LIDAR_RAYS + 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.model = None
        self.data = None
        self.steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.car_body_id = None
        self.prev_dist_to_goal = None
        self.curriculum_progress = 1.0  # 0.0 = easy (close goal), 1.0 = full difficulty (far corner)

    def _load_random_map(self):
        path = random.choice(self.map_paths)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "car")
        margin = 1.5
        full_goal = self.arena_size - margin
        self.goal = np.array([full_goal, full_goal], dtype=np.float32) * self.curriculum_progress

    def _get_lidar(self, car_pos, car_yaw):
        readings = np.full(N_LIDAR_RAYS, LIDAR_MAX_RANGE, dtype=np.float32)
        pnt = np.array([car_pos[0], car_pos[1], 0.3], dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        for i in range(N_LIDAR_RAYS):
            angle = car_yaw + (i / N_LIDAR_RAYS) * 2 * np.pi
            vec = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
            dist = mujoco.mj_ray(self.model, self.data, pnt, vec, None, 1, self.car_body_id, geomid)
            if dist >= 0:
                readings[i] = min(dist, LIDAR_MAX_RANGE)
        return readings

    def _get_obs(self):
        qpos = self.data.qpos[:7]
        qvel = self.data.qvel[:6]
        x, y = qpos[0], qpos[1]
        qw, qz = qpos[3], qpos[6]
        yaw = 2 * np.arctan2(qz, qw)
        vx, vy = qvel[0], qvel[1]
        yaw_rate = qvel[5]

        raw_lidar = self._get_lidar((x, y), yaw)
        lidar_norm = raw_lidar / LIDAR_MAX_RANGE
        goal_delta = self.goal - np.array([x, y], dtype=np.float32)

        obs = np.concatenate([
            [yaw, vx, vy, yaw_rate],
            goal_delta,
            lidar_norm,
            self.prev_action,
        ]).astype(np.float32)
        return obs, (x, y), raw_lidar

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_random_map()
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        obs, _, _ = self._get_obs()
        self.prev_dist_to_goal = float(np.linalg.norm(self.goal - self.data.qpos[:2]))
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        throttle, steering = float(action[0]), float(action[1])

        force_scale = 15.0
        self.data.xfrc_applied[self.car_body_id, 0] = throttle * force_scale
        self.data.xfrc_applied[self.car_body_id, 5] = steering * force_scale * 0.3

        # frame-skip: hold the action for several physics substeps so each
        # policy decision covers a meaningful amount of simulated time.
        # model timestep is 0.002s -- 20 substeps = 0.04s of sim time per
        # env.step(), so max_steps=300 now covers 12 simulated seconds
        # instead of 0.6.
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs, (x, y), raw_lidar = self._get_obs()
        min_obstacle_dist = float(np.min(raw_lidar))
        dist_to_goal = float(np.linalg.norm(self.goal - np.array([x, y])))
        out_of_bounds = abs(x) > self.arena_size or abs(y) > self.arena_size
        collided = min_obstacle_dist < COLLISION_THRESHOLD
        flipped = self.data.qpos[2] < 0.02

        # --- reward, adapted from GRaD-Nav++ Table I ---
        reward = 0.0
        reward += 0.05                                                  # survival bonus
        reward += 8.0 * (self.prev_dist_to_goal - dist_to_goal)         # progress reward: closer = positive, farther = negative (boosted weight)
        reward -= 0.02 * float(np.sum(action ** 2))                      # action penalty (small)
        reward -= 0.02 * float(np.sum((action - self.prev_action) ** 2)) # action-rate penalty (small)
        if min_obstacle_dist < OBSTACLE_SAFETY_DIST:
            reward -= 0.3 * (OBSTACLE_SAFETY_DIST - min_obstacle_dist)  # obstacle-avoidance penalty (reduced -- was too harsh for maze corridors)
        if out_of_bounds:
            reward -= 2.0
        if collided:
            reward -= 5.0
        if dist_to_goal < 0.5:
            reward += 20.0  # goal-reached bonus

        self.prev_dist_to_goal = dist_to_goal
        self.prev_action = action.copy()

        terminated = bool(flipped or collided or out_of_bounds or dist_to_goal < 0.5)
        truncated = self.steps >= self.max_steps

        if dist_to_goal < 0.5:
            end_reason = "goal_reached"
        elif collided:
            end_reason = "collision"
        elif out_of_bounds:
            end_reason = "out_of_bounds"
        elif flipped:
            end_reason = "flipped"
        elif truncated:
            end_reason = "timeout"
        else:
            end_reason = "ongoing"

        info = {"dist_to_goal": dist_to_goal, "min_obstacle_dist": min_obstacle_dist, "end_reason": end_reason}
        return obs, reward, terminated, truncated, info


def make_env(maps_dir="maps", max_steps=500):
    return RandomMapCarEnv(maps_dir=maps_dir, max_steps=max_steps)
