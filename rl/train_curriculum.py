"""
train_curriculum.py

Trains with a curriculum: starts with an easy, close goal, then gradually
increases goal distance to full difficulty. This tends to be far more
sample-efficient than throwing the full 12m goal at an untrained policy
from scratch -- the policy learns basic navigation on easy goals first,
then extends that skill to harder ones.

Usage:
    python3 train_curriculum.py --maps-dir maps --steps-per-stage 200000
"""

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from car_env import make_env

# curriculum stages: (curriculum_progress, description)
STAGES = [
    (0.2, "very easy -- goal close by"),
    (0.4, "easy"),
    (0.6, "medium"),
    (0.8, "hard"),
    (1.0, "full difficulty -- far corner"),
]


def make_curriculum_env(maps_dir, max_steps, progress):
    def _make():
        e = make_env(maps_dir=maps_dir, max_steps=max_steps)
        e.curriculum_progress = progress
        return e
    return _make


def evaluate(model, maps_dir, max_steps, n_episodes=20):
    env = make_env(maps_dir=maps_dir, max_steps=max_steps)
    env.curriculum_progress = 1.0  # always evaluate at full difficulty
    dists = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        dists.append(info["dist_to_goal"])
    return np.mean(dists), np.min(dists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--steps-per-stage", type=int, default=200000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--out", default="curriculum_model.zip")
    args = parser.parse_args()

    model = None
    for progress, desc in STAGES:
        print(f"\n=== Stage: progress={progress} ({desc}) ===")
        env = make_vec_env(
            make_curriculum_env(args.maps_dir, args.max_steps, progress),
            n_envs=args.n_envs,
        )
        if model is None:
            model = PPO("MlpPolicy", env, verbose=1)
        else:
            model.set_env(env)
        model.learn(total_timesteps=args.steps_per_stage, reset_num_timesteps=False)

        mean_dist, min_dist = evaluate(model, args.maps_dir, args.max_steps)
        print(f"  [eval @ full difficulty] mean_dist_to_goal={mean_dist:.2f} min={min_dist:.2f}")

    model.save(args.out)
    print(f"\nDone. Final model saved to {args.out}")


if __name__ == "__main__":
    main()
