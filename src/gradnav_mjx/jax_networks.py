"""
jax_networks.py

Pure-JAX policy and value networks (no Flax/Haiku dependency).
Mirrors policy_network.py / value_network.py's architecture exactly:
  - Policy: MLP [128, 128, 64] -> tanh -> 2D action (steer, throttle)
  - Value:  MLP [128, 128, 64] -> scalar value estimate

Tested and verified (in a separate sandbox before handing off):
  - Init + single forward pass: correct shapes, action bounded to [-1,1]
  - Batched forward pass: correct shapes over a batch dimension
  - Gradient flow via jax.grad: nonzero gradients through every layer,
    for both policy and value networks
  - JIT compilation via jax.jit: works correctly
"""

import jax
import jax.numpy as jnp


def init_mlp_params(key, in_dim, hidden_sizes, out_dim):
    sizes = [in_dim] + list(hidden_sizes) + [out_dim]
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i in range(len(sizes) - 1):
        w_key, b_key = jax.random.split(keys[i])
        scale = jnp.sqrt(2.0 / sizes[i])
        W = jax.random.normal(w_key, (sizes[i], sizes[i + 1])) * scale
        b = jnp.zeros((sizes[i + 1],))
        params.append((W, b))
    return params


def mlp_forward(params, x, final_activation=None):
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        is_last = i == len(params) - 1
        if not is_last:
            x = jnp.tanh(x)
        elif final_activation is not None:
            x = final_activation(x)
    return x


def init_policy_params(key, lidar_dim=16, obs_extra_dim=4, hidden_sizes=(128, 128, 64)):
    obs_dim = lidar_dim + obs_extra_dim
    return init_mlp_params(key, obs_dim, hidden_sizes, out_dim=2)


def policy_forward(params, obs):
    return mlp_forward(params, obs, final_activation=jnp.tanh)


def init_value_params(key, lidar_dim=16, obs_extra_dim=4, hidden_sizes=(128, 128, 64)):
    obs_dim = lidar_dim + obs_extra_dim
    return init_mlp_params(key, obs_dim, hidden_sizes, out_dim=1)


def value_forward(params, obs):
    v = mlp_forward(params, obs, final_activation=None)
    return v[..., 0]
