import jax.numpy as jnp


def wall_distances(car_xy, walls):
    a = walls[:, 0:2]
    b = walls[:, 2:4]
    r = walls[:, 4]

    ab = b - a
    ab_len_sq = jnp.sum(ab ** 2, axis=-1) + 1e-8

    if car_xy.ndim == 1:
        ap = car_xy[None, :] - a
        t = jnp.clip(jnp.sum(ap * ab, axis=-1) / ab_len_sq, 0.0, 1.0)
        closest = a + t[:, None] * ab
        dist_to_line = jnp.linalg.norm(car_xy[None, :] - closest, axis=-1)
        return dist_to_line - r
    else:
        ap = car_xy[:, None, :] - a[None, :, :]
        t = jnp.clip(
            jnp.sum(ap * ab[None, :, :], axis=-1) / ab_len_sq[None, :], 0.0, 1.0
        )
        closest = a[None, :, :] + t[..., None] * ab[None, :, :]
        dist_to_line = jnp.linalg.norm(car_xy[:, None, :] - closest, axis=-1)
        return dist_to_line - r[None, :]


def human_distances(car_xy, humans):
    """Distance from the car to each human's circular footprint."""
    c = humans[:, 0:2]
    r = humans[:, 2]
    if car_xy.ndim == 1:
        return jnp.linalg.norm(car_xy[None, :] - c, axis=-1) - r
    return jnp.linalg.norm(car_xy[:, None, :] - c[None, :, :], axis=-1) - r[None, :]


def obstacle_distances(car_xy, walls, humans=None):
    """Distances to every obstacle: walls and, if given, humans.

    Used for the reward's safety term and for collision counting, so
    that people count as things you must not hit, exactly like walls.
    """
    d = wall_distances(car_xy, walls)
    if humans is None or humans.shape[0] == 0:
        return d
    return jnp.concatenate([d, human_distances(car_xy, humans)], axis=-1)
