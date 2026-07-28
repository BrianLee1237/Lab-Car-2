"""
train_diagnostic.py

Trains PPO on the randomized maps and, critically, tracks *distance to
goal* at episode end over time -- not just the blended reward, which can
look flat/declining even while real progress (or regression) is happening
underneath it.

Usage:
    python3 train_diagnostic.py --timesteps 200000 --n-envs 4
"""

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from car_env import make_env


class DistToGoalTracker(BaseCallback):
    """Logs mean dist_to_goal at episode end, every `log_every` steps."""
    def __init__(self, log_every=10000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.episode_dists = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "dist_to_goal" in info and (
                self.locals.get("dones") is not None
            ):
                pass  # captured below via episode-end detection
        # collect dist_to_goal whenever an episode just ended
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done and "dist_to_goal" in info:
                    self.episode_dists.append(info["dist_to_goal"])

        if self.num_timesteps - self.last_log_step >= self.log_every:
            self.last_log_step = self.num_timesteps
            if self.episode_dists:
                recent = self.episode_dists[-50:]
                print(f"  [diagnostic] step={self.num_timesteps:>7} "
                      f"mean_dist_to_goal(last {len(recent)} eps)={np.mean(recent):.2f}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--out", default="diagnostic_model.zip")
    args = parser.parse_args()

    env = make_vec_env(lambda: make_env(maps_dir=args.maps_dir, max_steps=args.max_steps), n_envs=args.n_envs)
    model = PPO("MlpPolicy", env, verbose=1)

    tracker = DistToGoalTracker(log_every=10000)
    model.learn(total_timesteps=args.timesteps, callback=tracker)
    model.save(args.out)

    print(f"\nDone. Model saved to {args.out}")
    if len(tracker.episode_dists) >= 100:
        first_50 = np.mean(tracker.episode_dists[:50])
        last_50 = np.mean(tracker.episode_dists[-50:])
        print(f"First 50 episodes mean dist_to_goal: {first_50:.2f}")
        print(f"Last 50 episodes mean dist_to_goal:  {last_50:.2f}")
        if last_50 < first_50:
            print("-> Improving: getting closer to goal over training.")
        else:
            print("-> NOT improving: not getting closer to goal. Reward shaping likely needs tuning.")


if __name__ == "__main__":
    main()
