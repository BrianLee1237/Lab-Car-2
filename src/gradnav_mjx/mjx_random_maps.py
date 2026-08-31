import random


def make_capsule_maze_walls(size, n_walls, seed, radius=0.1, min_len=1.5, max_len=4.0):
    rng = random.Random(seed)
    walls = []
    margin = 1.5
    for _ in range(n_walls):
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
        walls.append((x1, y1, x2, y2, radius))
    return walls


def generate_map_set(n_maps=4, size=6.0, n_walls=4, base_seed=0):
    return [
        make_capsule_maze_walls(size, n_walls, seed=base_seed + i)
        for i in range(n_maps)
    ]
