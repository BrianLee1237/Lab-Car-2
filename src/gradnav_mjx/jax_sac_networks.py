"""
jax_sac_networks.py

Networks for Soft Actor-Critic (Haarnoja et al. 2018, arXiv:1801.01290):
  - Squashed Gaussian policy: outputs a per-action-dim mean/log_std,
    samples via the reparameterization trick, squashes through tanh to
    keep actions in [-1, 1], with the standard tanh log-prob correction
    (paper's Appendix C) so the resulting log_prob is exact.
  - Twin Q-networks: Q(s, a) -> scalar, same MLP shape as the DiffRL
    value network but taking the action as extra input.

Reuses init_mlp_params/mlp_forward from jax_networks.py so the hidden
architecture ([128, 128, 64], tanh) matches the rest of this project.
"""

import jax
import jax.numpy as jnp

from jax_networks import init_mlp_params, mlp_forward

ACTION_DIM = 2
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def init_sac_policy_params(key, obs_dim, hidden_sizes=(128, 128, 64)):
    return init_mlp_params(key, obs_dim, hidden_sizes, out_dim=2 * ACTION_DIM)


def policy_apply(params, obs):
    """Returns (mean, log_std), each shape (..., ACTION_DIM)."""
    out = mlp_forward(params, obs, final_activation=None)
    mean, log_std = jnp.split(out, 2, axis=-1)
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mean, log_std


def sample_action(params, obs, key):
    """Reparameterized sample: returns (action, log_prob). log_prob is
    summed over action dims (shape (...,))."""
    mean, log_std = policy_apply(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    pre_tanh = mean + std * noise
    action = jnp.tanh(pre_tanh)

    # log N(pre_tanh; mean, std)
    gaussian_log_prob = (
        -0.5 * ((pre_tanh - mean) / (std + 1e-6)) ** 2
        - log_std
        - 0.5 * jnp.log(2 * jnp.pi)
    )
    # tanh squash correction (SAC paper appendix C): log|d tanh/dx| = log(1 - tanh(x)^2)
    correction = jnp.log(1 - action ** 2 + 1e-6)
    log_prob = jnp.sum(gaussian_log_prob - correction, axis=-1)
    return action, log_prob


def deterministic_action(params, obs):
    """Mean action (no sampling), for evaluation."""
    mean, _ = policy_apply(params, obs)
    return jnp.tanh(mean)


def init_q_params(key, obs_dim, hidden_sizes=(128, 128, 64)):
    return init_mlp_params(key, obs_dim + ACTION_DIM, hidden_sizes, out_dim=1)


def q_apply(params, obs, action):
    x = jnp.concatenate([obs, action], axis=-1)
    return mlp_forward(params, x, final_activation=None)[..., 0]
