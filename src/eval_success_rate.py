import argparse

import torch

from diff_ackermann_dynamics import DiffAckermannDynamics
from map_walls import load_walls
from policy_network import PolicyNetwork
from value_network import ValueNetwork
from train_diffrl import batched_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="diffrl_policy.pt")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--map", default="maps/map_0000.xml")
    args = parser.parse_args()

    walls = load_walls(args.map)

    dyn = DiffAckermannDynamics()
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(args.policy))
    value_fn = ValueNetwork()

    g = torch.Generator().manual_seed(999)
    with torch.no_grad():
        _, final_dists, _, collided = batched_rollout(
            dyn, policy, value_fn, args.horizon, args.n_eval,
            min_dist=0.5, max_dist=8.0, walls=walls, generator=g, train=False,
        )

    reached_goal = final_dists < args.success_threshold
    true_success = reached_goal & (~collided)

    print(f"Evaluated {args.n_eval} fixed benchmark goals (0.5-8.0m, seed=999) on {args.map}")
    print(f"Reached goal (dist < {args.success_threshold}m): {reached_goal.float().mean().item()*100:.1f}%")
    print(f"Collision rate: {collided.float().mean().item()*100:.1f}%")
    print(f"TRUE success (reached goal AND no collision): {true_success.float().mean().item()*100:.1f}%  "
          f"({int(true_success.sum().item())}/{args.n_eval})")
    print(f"Mean final distance: {final_dists.mean().item():.3f}m")


if __name__ == "__main__":
    main()
