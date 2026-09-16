"""Body-frame 2D lidar against capsule walls.

mjx_obstacle_dist.wall_distances returns one scalar per wall -- the
distance from the car's centre to that wall -- indexed by wall ID. That
is directionless: the policy is told "wall #2 is 1.3m away" with no
indication of whether it is ahead, behind, left or right, and the
wall's index carries no spatial meaning. Obstacle avoidance is
essentially impossible from that, which is why collisions stayed at
17-40 per 50 episodes at 6m goals while being 0-2 per 50 at 1-2m,
where the straight path rarely meets a wall.

jax_networks.py still defaults to lidar_dim=16, i.e. the original
GRaD-Nav design sensed obstacles with a 16-ray lidar; the real MuSHR
car likewise carries a planar laser scanner. This restores that: rays
cast at evenly spaced angles in the car's BODY frame, each returning
the range to the nearest wall along it.

Ray/capsule intersection is exact (not ray-marched): for each wall it
solves the ray against the infinite slab around the segment's line,
keeps hits whose projection lands within the segment, and unions that
with the two endpoint circles.
"""

import jax.numpy as jnp


def _cross2(v, u):
    return v[..., 0] * u[..., 1] - v[..., 1] * u[..., 0]


def _ray_circle(o, d, centre, r):
    """Smallest t >= 0 with |o + t d - centre| = r, else inf. |d| == 1."""
    oc = o - centre
    b = jnp.sum(oc * d, axis=-1)
    c = jnp.sum(oc * oc, axis=-1) - r ** 2
    disc = b ** 2 - c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    t = jnp.where(t0 >= 0.0, t0, t1)
    return jnp.where((disc > 0.0) & (t >= 0.0), t, jnp.inf)


def lidar_scan(car_xy, theta, walls, n_rays=16, max_range=8.0, fov=jnp.pi):
    """Ranges along n_rays rays spread over +-fov about the car's heading.

    car_xy: (2,), theta: scalar, walls: (W, 5) as (x1, y1, x2, y2, r).
    Returns (n_rays,) ranges, clipped to max_range.
    """
    angles = theta + jnp.linspace(-fov, fov, n_rays)
    d = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)      # (R, 2)
    o = car_xy[None, :]                                             # (1, 2)

    a = walls[:, 0:2][None, :, :]                                   # (1, W, 2)
    b = walls[:, 2:4][None, :, :]
    r = walls[:, 4][None, :]                                        # (1, W)

    ba = b - a
    seg_len = jnp.linalg.norm(ba, axis=-1) + 1e-9                   # (1, W)
    u = ba / seg_len[..., None]                                     # unit along segment

    dr = d[:, None, :]                                              # (R, 1, 2)
    oa = o[:, None, :] - a                                          # (1, W, 2)

    # Perpendicular offset from the segment's line, as a function of t:
    #   cross(oa + t*dr, u) = c0 + t*cd ; hits where |c0 + t*cd| = r
    c0 = _cross2(oa, u)                                             # (1, W)
    cd = _cross2(dr, u)                                             # (R, W)
    safe_cd = jnp.where(jnp.abs(cd) < 1e-9, 1e-9, cd)

    best = jnp.full((d.shape[0], walls.shape[0]), jnp.inf)
    for sign in (1.0, -1.0):
        t = (sign * r - c0) / safe_cd
        hit = o[:, None, :] + t[..., None] * dr                     # (R, W, 2)
        proj = jnp.sum((hit - a) * u, axis=-1)                       # along-segment coord
        valid = (
            (jnp.abs(cd) > 1e-9)
            & (t >= 0.0)
            & (proj >= 0.0)
            & (proj <= seg_len)
        )
        best = jnp.minimum(best, jnp.where(valid, t, jnp.inf))

    # Rounded ends
    for centre in (a, b):
        t = _ray_circle(o[:, None, :], dr, centre, r)
        best = jnp.minimum(best, t)

    return jnp.clip(jnp.min(best, axis=-1), 0.0, max_range)
