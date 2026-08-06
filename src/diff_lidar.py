"""
diff_lidar.py (v2)

Real, differentiable lidar sensing against RECTANGULAR wall obstacles
(matching the actual PPO map style from mujoco_map_gen.py), replacing
both the earlier fake constant placeholder AND the earlier circle-only
version. Uses the standard "slab method" for ray vs. axis-aligned-box
intersection, since every wall in the real maps is axis-aligned.

Pedestrians are NOT included -- they're dynamic bodies in the real
maps (MuJoCo physics-driven), which this pure-PyTorch pipeline doesn't
simulate. Only static walls are sensed. Known gap, not silently ignored.
"""

import torch


def _safe_div(num, den, eps=1e-6):
    sign = torch.sign(den)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    den_safe = torch.where(den.abs() < eps, sign * eps, den)
    return num / den_safe


def cast_lidar(pos, theta, walls, n_rays=16, max_range=8.0):
    """
    pos:    (B, 2) car x,y
    theta:  (B,)   car heading
    walls:  (K, 4) [cx, cy, half_x, half_y] -- axis-aligned rectangles
    returns (B, n_rays) distance to nearest wall along each ray, clipped
            to max_range
    """
    B = pos.shape[0]
    K = walls.shape[0]

    ray_offsets = torch.linspace(0, 2 * torch.pi, n_rays + 1)[:-1]
    angles = theta.unsqueeze(-1) + ray_offsets.unsqueeze(0)  # (B, n_rays)
    dx = torch.cos(angles).unsqueeze(1)  # (B, 1, n_rays)
    dy = torch.sin(angles).unsqueeze(1)

    px = pos[:, 0].view(B, 1, 1)
    py = pos[:, 1].view(B, 1, 1)

    cx = walls[:, 0].view(1, K, 1)
    cy = walls[:, 1].view(1, K, 1)
    hx = walls[:, 2].view(1, K, 1)
    hy = walls[:, 3].view(1, K, 1)

    # slab method: intersection interval with the box's x-slab and y-slab
    tx1 = _safe_div((cx - hx) - px, dx)
    tx2 = _safe_div((cx + hx) - px, dx)
    tx_min = torch.minimum(tx1, tx2)
    tx_max = torch.maximum(tx1, tx2)

    ty1 = _safe_div((cy - hy) - py, dy)
    ty2 = _safe_div((cy + hy) - py, dy)
    ty_min = torch.minimum(ty1, ty2)
    ty_max = torch.maximum(ty1, ty2)

    t_enter = torch.maximum(tx_min, ty_min)
    t_exit = torch.minimum(tx_max, ty_max)

    hit = (t_exit >= t_enter) & (t_exit > 0)
    t_hit = torch.where(t_enter > 0, t_enter, t_exit)  # if starting inside, use exit point
    t_or_max = torch.where(hit, t_hit, torch.full_like(t_hit, max_range))

    dists = t_or_max.min(dim=1).values  # nearest wall per ray, (B, n_rays)
    return torch.clamp(dists, min=0.0, max=max_range)
