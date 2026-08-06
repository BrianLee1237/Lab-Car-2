import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, lidar_dim=16, obs_extra_dim=4, hidden_sizes=(128, 128, 64)):
        super().__init__()
        obs_dim = lidar_dim + obs_extra_dim

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h

        self.body = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, 2)

    def forward(self, obs):
        x = self.body(obs)
        action = torch.tanh(self.action_head(x))
        return action


def build_observation(state, goal, lidar):
    # state order matches DiffAckermannDynamics: [x, y, theta, v, delta]
    x, y, theta, v, delta = state[0], state[1], state[2], state[3], state[4]
    goal_dx = goal[0] - x
    goal_dy = goal[1] - y
    extra = torch.stack([v, theta, goal_dx, goal_dy])
    obs = torch.cat([lidar, extra])
    return obs
