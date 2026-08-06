"""
train_diffrl.py (v8)

Now uses REAL wall geometry parsed from an actual PPO map (map_walls.py
+ diff_lidar.py's rectangular ray-casting), instead of the earlier
hand-made circle test environment. Pedestrians excluded (dynamic,
not simulated in this pure-PyTorch pipeline) -- known gap.

Uses a single fixed map for training (maps/map_0000.xml by default) so
results are reproducible and comparable across runs; --map lets you
point at a different one.
"""

import argparse

import torch
import torch.optim as optim

from diff_ackermann_dynamics import DiffAckermannDynamics
from diff_reward import diff_reward
from diff_lidar import cast_lidar
from map_walls import load_walls
from policy_network import PolicyNetwork, build_observation
from value_network import ValueNetwork


def batched_random_start_goal(batch_size, min_dist, max_dist, generator=None):
    start = torch.zeros(batch_size, 5)
    start[:, 2] = (torch.rand(batch_size, generator=generator) * 2 - 1) * 0.25

    goal_angle = torch.rand(batch_size, generator=generator) * 2 * 3.14159
    goal_dist = min_dist + torch.rand(batch_size, generator=generator) * (max_dist - min_dist)
    goal = torch.stack([
        goal_dist * torch.cos(goal_angle),
        goal_dist * torch.sin(goal_angle),
    ], dim=-1)
    return start, goal


def batched_build_observation(state, goal, lidar):
    x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    goal_dx = goal[:, 0] - x
    goal_dy = goal[:, 1] - y
    extra = torch.stack([v, theta, goal_dx, goal_dy], dim=-1)
    obs = torch.cat([lidar, extra], dim=-1)
    return obs


def batched_rollout(dyn, policy, value_fn, horizon, batch_size, min_dist, max_dist,
                     walls, gamma=0.99, generator=None, train=True):
    state, goal = batched_random_start_goal(batch_size, min_dist, max_dist, generator)

    prev_action = torch.zeros(batch_size, 2)
    prev_prev_action = torch.zeros(batch_size, 2)
    prev_goal_dist = torch.norm(state[:, :2] - goal, dim=-1)

    discounted_reward = torch.zeros(batch_size)
    discount = 1.0
    min_dist_seen = torch.full((batch_size,), float('inf'))

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for t in range(horizon):
            lidar = cast_lidar(state[:, :2], state[:, 2], walls)
            obs = batched_build_observation(state, goal, lidar)
            action = policy(obs)
            state = dyn.step(state, action)
            r, goal_dist = diff_reward(
                state, action, prev_action, prev_prev_action, goal, lidar, prev_goal_dist,
            )
            discounted_reward = discounted_reward + discount * r
            discount *= gamma
            prev_prev_action = prev_action
            prev_action = action
            prev_goal_dist = goal_dist
            min_dist_seen = torch.minimum(min_dist_seen, lidar.min(dim=-1).values)

        final_lidar = cast_lidar(state[:, :2], state[:, 2], walls)
        final_obs = batched_build_observation(state, goal, final_lidar)
        bootstrap = discount * value_fn(final_obs)
        total_return = discounted_reward + bootstrap
        loss = -total_return.mean()

    final_dists = torch.norm(state[:, :2] - goal, dim=-1)
    collided = min_dist_seen < 0.3
    return loss, final_dists, total_return, collided


def evaluate_fixed_benchmark(dyn, policy, value_fn, horizon, walls, n_eval=64):
    g = torch.Generator().manual_seed(999)
    _, final_dists, _, collided = batched_rollout(
        dyn, policy, value_fn, horizon, n_eval, min_dist=0.5, max_dist=8.0,
        walls=walls, generator=g, train=False,
    )
    return final_dists.mean().item(), collided.float().mean().item()


def curriculum_goal_range(progress, max_dist=8.0, min_start=1.0):
    current_max = min_start + progress * (max_dist - min_start)
    return 0.5, current_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--map", default="maps/map_0000.xml")
    parser.add_argument("--out", default="diffrl_policy.pt")
    args = parser.parse_args()

    walls = load_walls(args.map)
    print(f"Loaded {walls.shape[0]} wall segments from {args.map}")

    dyn = DiffAckermannDynamics()
    policy = PolicyNetwork()
    value_fn = ValueNetwork()

    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=args.lr,
    )

    for it in range(args.iterations):
        progress = min(1.0, it / (args.iterations * 0.7))
        min_dist, max_dist = curriculum_goal_range(progress)

        optimizer.zero_grad()
        loss, final_dists, total_return, collided = batched_rollout(
            dyn, policy, value_fn, args.horizon, args.batch_size, min_dist, max_dist, walls
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_fn.parameters()), max_norm=5.0
        )
        optimizer.step()

        if (it + 1) % args.eval_every == 0:
            eval_dist, eval_collision_rate = evaluate_fixed_benchmark(dyn, policy, value_fn, args.horizon, walls)
            print(f"iter {it + 1:5d}  progress={progress:.2f}  goal_range=[{min_dist:.1f},{max_dist:.1f}]  "
                  f"train_dist={final_dists.mean().item():.3f}  "
                  f"EVAL_dist={eval_dist:.3f}  EVAL_collision_rate={eval_collision_rate*100:.1f}%  "
                  f"train_collision_rate={collided.float().mean().item()*100:.1f}%")

    torch.save(policy.state_dict(), args.out)
    print(f"\nDone. Policy saved to {args.out}")


if __name__ == "__main__":
    main()
