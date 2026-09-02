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
