import math
import random


def _dist_to_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy)


def make_capsule_maze_walls(size, n_walls, seed, radius=0.1, min_len=1.5, max_len=4.0,
                             spawn_clear_radius=1.3, max_resample=50):
    """spawn_clear_radius: reject-and-resample any wall placement whose
    segment passes within this distance of the origin (the car's spawn
    point). Without this, nothing stops a wall from landing right next
    to spawn -- e.g. one training/eval map (seed 0) had a wall only
    0.757m from the origin, well inside the 0.3-1.0m goal-sampling
    range, silently blocking some goals regardless of policy quality.
    1.3m gives clearance past the max training/eval goal distance
    (1.0m) plus the car's own footprint."""
    rng = random.Random(seed)
    walls = []
    margin = 1.5
    for _ in range(n_walls):
        for _ in range(max_resample):
            horizontal = rng.random() < 0.5
            length = rng.uniform(min_len, max_len)
            cx = rng.uniform(-size + margin, size - margin)
            cy = rng.uniform(-size + margin, size - margin)
            if horizontal:
                x1, y1 = cx - length / 2, cy
                x2, y2 = cx + length / 2, cy
            else:
                x1, y1 = cx, cy - length / 2
                x2, y2 = cx, cy + length / 2
            if _dist_to_segment(0.0, 0.0, x1, y1, x2, y2) - radius >= spawn_clear_radius:
                break
        walls.append((x1, y1, x2, y2, radius))
    return walls


def generate_map_set(n_maps=4, size=6.0, n_walls=4, base_seed=0):
    return [
        make_capsule_maze_walls(size, n_walls, seed=base_seed + i)
        for i in range(n_maps)
    ]


def make_room_walls(size, n_inner, seed, radius=0.1, perimeter_radius=0.15,
                     min_len=1.5, max_len=5.0, inner_margin=1.2,
                     min_gap=1.2, max_resample=120):
    """An enclosed rectangular room: 4 perimeter walls plus n_inner
    interior obstacles.

    make_capsule_maze_walls produces free-floating, disconnected sticks
    on an unbounded floor -- no perimeter, so the car can simply drive
    out of the obstacle field into open space, and most straight lines
    across the map hit nothing at all. That makes long-range runs mostly
    empty driving and gives the lidar almost nothing to do (collisions
    were 0-2 per 50 at 1-2m purely because the short path rarely met a
    wall).

    Here the room is closed, so every episode happens inside a bounded
    space the lidar can actually perceive, and interior walls have to be
    driven around rather than escaped past.

    min_gap: interior walls are kept at least this far from each other,
    so the room never closes into a sealed pocket that traps the car or
    walls off a goal. Interior walls also stay inner_margin from the
    perimeter, leaving a driveable lane around the outside.
    """
    half = size
    walls = [
        (-half, -half,  half, -half, perimeter_radius),   # bottom
        (-half,  half,  half,  half, perimeter_radius),   # top
        (-half, -half, -half,  half, perimeter_radius),   # left
        ( half, -half,  half,  half, perimeter_radius),   # right
    ]

    rng = random.Random(seed)
    lo, hi = -half + inner_margin, half - inner_margin
    inner = []
    for _ in range(n_inner):
        for _ in range(max_resample):
            horizontal = rng.random() < 0.5
            length = rng.uniform(min_len, min(max_len, 2 * (hi - lo) / 3))
            cx = rng.uniform(lo, hi)
            cy = rng.uniform(lo, hi)
            if horizontal:
                x1, y1, x2, y2 = cx - length / 2, cy, cx + length / 2, cy
            else:
                x1, y1, x2, y2 = cx, cy - length / 2, cx, cy + length / 2
            # keep the whole segment inside the driveable interior
            x1, x2 = max(x1, lo), min(x2, hi)
            y1, y2 = max(y1, lo), min(y2, hi)
            if math.hypot(x2 - x1, y2 - y1) < min_len * 0.6:
                continue
            # keep a driveable gap to every wall placed so far
            too_close = False
            for (ax1, ay1, ax2, ay2, ar) in inner:
                if _segment_gap(x1, y1, x2, y2, ax1, ay1, ax2, ay2) < min_gap:
                    too_close = True
                    break
            if not too_close:
                break
        inner.append((x1, y1, x2, y2, radius))

    return walls + inner


def _segment_gap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """Approximate min distance between two segments (endpoint probes)."""
    return min(
        _dist_to_segment(ax1, ay1, bx1, by1, bx2, by2),
        _dist_to_segment(ax2, ay2, bx1, by1, bx2, by2),
        _dist_to_segment(bx1, by1, ax1, ay1, ax2, ay2),
        _dist_to_segment(bx2, by2, ax1, ay1, ax2, ay2),
    )


def generate_room_set(n_maps=1, size=8.0, n_inner=8, base_seed=0):
    return [
        make_room_walls(size, n_inner, seed=base_seed + i)
        for i in range(n_maps)
    ]


def generate_room_layout(room_half=8.0, n_inner=6, n_humans=6, seed=0,
                          wall_half_thickness=0.075, human_radius=0.25,
                          inner_margin=1.5, min_gap=1.6, human_clear=1.0,
                          min_len=2.0, max_len=6.0, max_resample=200):
    """An enclosed room: perimeter walls, interior walls, and humans.

    Returns (walls, humans) where walls are axis-aligned segments
    (x1, y1, x2, y2, half_thickness) and humans are circles (x, y, r).
    Both are the 2D footprints matching the 3D geoms built by
    mjx_room_scene, so lidar/collision math stays exact.

    Everything is placed with clearance so the room stays driveable:
    interior walls keep `min_gap` from each other and `inner_margin`
    from the perimeter, and humans keep `human_clear` from walls and
    from each other. Without that a room can seal into a pocket that
    traps the car or walls off a goal, which would cap success for
    reasons that have nothing to do with the policy.
    """
    rng = random.Random(seed)
    t = wall_half_thickness
    h = room_half
    walls = [
        (-h, -h,  h, -h, t),
        (-h,  h,  h,  h, t),
        (-h, -h, -h,  h, t),
        ( h, -h,  h,  h, t),
    ]

    lo, hi = -h + inner_margin, h - inner_margin
    inner = []
    for _ in range(n_inner):
        for _ in range(max_resample):
            horizontal = rng.random() < 0.5
            length = rng.uniform(min_len, max_len)
            cx, cy = rng.uniform(lo, hi), rng.uniform(lo, hi)
            if horizontal:
                x1, y1, x2, y2 = cx - length / 2, cy, cx + length / 2, cy
            else:
                x1, y1, x2, y2 = cx, cy - length / 2, cx, cy + length / 2
            x1, x2 = max(x1, lo), min(x2, hi)
            y1, y2 = max(y1, lo), min(y2, hi)
            if math.hypot(x2 - x1, y2 - y1) < min_len * 0.5:
                continue
            if all(_segment_gap(x1, y1, x2, y2, *w[:4]) >= min_gap for w in inner):
                break
        inner.append((x1, y1, x2, y2, t))

    all_walls = walls + inner

    humans = []
    for _ in range(n_humans):
        for _ in range(max_resample):
            hx = rng.uniform(-h + inner_margin, h - inner_margin)
            hy = rng.uniform(-h + inner_margin, h - inner_margin)
            wall_ok = all(
                _dist_to_segment(hx, hy, *w[:4]) - w[4] >= human_clear
                for w in all_walls
            )
            human_ok = all(
                math.hypot(hx - ox, hy - oy) >= human_clear + human_radius + orad
                for (ox, oy, orad) in humans
            )
            if wall_ok and human_ok:
                break
        humans.append((hx, hy, human_radius))

    return all_walls, humans
