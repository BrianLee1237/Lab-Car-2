"""
mjx_pedestrians.py

Walking humans for the room scene.

Each pedestrian patrols a straight line: it walks from one end to the
other at a fixed speed, then turns around -- a triangle wave in time,
so motion is smooth, bounded, and fully determined by the clock. That
matters for two reasons. It is stateless (position is a pure function
of t, so nothing extra has to be carried through the replay buffer or
the episode scan), and it is exactly reproducible, so an evaluation is
comparable across policies.

The same positions drive both the physics (written into mocap_pos) and
the sensing (lidar and the reward's safety term), so what the car is
scored against is exactly where the people actually are.
"""

import jax.numpy as jnp


def make_patrol_params(humans, rng, room_half, amplitude=2.5, speed_lo=0.6,
                        speed_hi=1.4):
    """Build patrol parameters from static human placements.

    humans: list of (x, y, r). Returns a dict of arrays. Walking speed
    is drawn in 0.6-1.4 m/s, roughly the range of real human gait.
    """
    import math
    import random as _r
    cx, cy, rad, dx, dy, amp, spd, phase = [], [], [], [], [], [], [], []
    for (hx, hy, hr) in humans:
        ang = rng.uniform(0.0, 2 * math.pi)
        ux, uy = math.cos(ang), math.sin(ang)
        # shorten the patrol so it stays inside the room
        room_lim = room_half - 1.0
        a = amplitude
        for _ in range(40):
            if (abs(hx + ux * a) < room_lim and abs(hy + uy * a) < room_lim and
                    abs(hx - ux * a) < room_lim and abs(hy - uy * a) < room_lim):
                break
            a *= 0.8
        cx.append(hx); cy.append(hy); rad.append(hr)
        dx.append(ux); dy.append(uy); amp.append(a)
        spd.append(rng.uniform(speed_lo, speed_hi))
        phase.append(rng.uniform(0.0, 1.0))
    return dict(
        centre=jnp.array(list(zip(cx, cy))),
        direction=jnp.array(list(zip(dx, dy))),
        radius=jnp.array(rad),
        amplitude=jnp.array(amp),
        speed=jnp.array(spd),
        phase=jnp.array(phase),
    )


AVOID_RADIUS = 1.3      # m: a person starts giving way inside this
MAX_SIDESTEP = 0.35     # m: but only ever steps this far aside


def pedestrian_xy(t, p, car_xy=None):
    """(H, 2) positions at time t seconds. Triangle wave: walk to one
    end, turn around, walk back, at constant speed.

    If car_xy is given, people give way: each steps directly away from
    the car, by up to MAX_SIDESTEP, once the car is inside AVOID_RADIUS.

    This matters because the people are mocap bodies -- kinematic, and
    so effectively infinitely massive. Walking blind, they push the car
    around and even overlap it, and with several crossing a room the
    car gets touched no matter what it does: a hand-written controller
    and a trained policy both collided in 20 of 20 episodes, so the
    collision metric measured nothing. Giving way makes contact
    avoidable. The cap is the point though -- a person sidesteps, they
    do not teleport, so driving straight at someone still hits them and
    the policy still has to learn to avoid people.

    Deliberately a function of the CURRENT car position only, never of
    history, so pedestrian state stays a pure function of (t, car) and
    nothing extra has to ride in the replay buffer.
    """
    amp = p["amplitude"]
    # period for a full there-and-back at `speed`
    period = jnp.maximum(4.0 * amp / jnp.maximum(p["speed"], 1e-6), 1e-6)
    u = jnp.mod(t / period + p["phase"], 1.0)
    # triangle wave in [-1, 1]
    tri = 4.0 * jnp.abs(u - 0.5) - 1.0
    offset = (amp * tri)[:, None] * p["direction"]
    base = p["centre"] + offset
    if car_xy is None:
        return base
    delta = base - car_xy[None, :]
    dist = jnp.linalg.norm(delta, axis=-1, keepdims=True)
    step = jnp.clip(AVOID_RADIUS - dist, 0.0, MAX_SIDESTEP)
    return base + delta / (dist + 1e-6) * step


def pedestrian_circles(t, p, car_xy=None):
    """(H, 3) as (x, y, radius) at time t -- the footprint the lidar and
    the reward's safety term see."""
    xy = pedestrian_xy(t, p, car_xy)
    return jnp.concatenate([xy, p["radius"][:, None]], axis=-1)


def pedestrian_mocap_pos(t, p, z, car_xy=None):
    """(H, 3) world positions to write into data.mocap_pos."""
    xy = pedestrian_xy(t, p, car_xy)
    return jnp.concatenate([xy, jnp.full((xy.shape[0], 1), z)], axis=-1)
