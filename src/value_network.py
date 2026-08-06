import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Differentiable value network V_phi(s_t) for DiffRL horizon bootstrapping.
    """

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
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, obs):
        x = self.body(obs)
        return self.value_head(x).squeeze(-1)
