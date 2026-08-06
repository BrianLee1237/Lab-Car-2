"""
diff_reward.py (v4)

Added a PRECISION term back in: a small exp(-distance) bonus stacked
on top of the progress reward.

Diagnosis: after training with progress-only reward, success rate
(final dist < 0.5m) was only 0.5% (1/200) despite mean distance
improving substantially during training (3.79 -> 2.88m). progress
reward gives good gradient for closing distance generally, but has no
special incentive to STOP precisely at the goal -- it's equally happy
moving from 5m to 4m as from 0.6m to 0.4m. exp(-distance) is weak at
long range (which is why it was replaced originally) but is exactly
the right shape for close-range precision: it grows sharply as
distance approaches 0, creating a strong pull to actually reach and
stop at the goal rather than just get generally closer.

Combining both: progress handles the far-range "which way do I go"
problem, exp(-distance) handles the close-range "actually arrive and
stop here" problem. Kept weight small (1.0) relative to progress (8.0)
so it doesn't reintroduce the original vanishing-gradient-at-range
issue as the dominant signal.
"""

import torch


def diff_reward(
    state: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    prev_prev_action: torch.Tensor,
    goal: torch.Tensor,
    obstacle_dists: torch.Tensor,
    prev_goal_dist: torch.Tensor,
    arena_size: float = 10.0,
    obstacle_safety_dist: float = 1.5,
    weights: dict = None,
):
    if weights is None:
        weights = dict(
            survival=0.5,
            action=-1.0,
            action_rate=-1.0,
            smoothness=-1.0,
            yaw_alignment=0.25,
            progress=8.0,
            precision=1.0,   # NEW: small exp(-distance) bonus for close-range precision
            obstacle=1.0,
            out_of_map=-2.0,
        )

    x, y, theta, v, delta = state.unbind(-1)

    r_survival = torch.ones_like(x)
    r_action = (action ** 2).sum(-1)
    r_action_rate = ((action - prev_action) ** 2).sum(-1)
    r_smoothness = ((action - 2 * prev_action + prev_prev_action) ** 2).sum(-1)

    goal_dx = goal[..., 0] - x
    goal_dy = goal[..., 1] - y
    goal_dist = torch.sqrt(goal_dx ** 2 + goal_dy ** 2 + 1e-6)
    goal_dir_x = goal_dx / goal_dist
    goal_dir_y = goal_dy / goal_dist
    r_yaw_alignment = torch.cos(theta) * goal_dir_x + torch.sin(theta) * goal_dir_y

    r_progress = prev_goal_dist - goal_dist
    r_precision = torch.exp(-goal_dist)  # sharp reward for actually being close/at the goal

    min_obstacle_dist = obstacle_dists.min(dim=-1).values
    r_obstacle = -torch.nn.functional.softplus(obstacle_safety_dist - min_obstacle_dist)

    x_excess = torch.nn.functional.softplus(torch.abs(x) - arena_size)
    y_excess = torch.nn.functional.softplus(torch.abs(y) - arena_size)
    r_out_of_map = x_excess ** 2 + y_excess ** 2

    total = (
        weights["survival"] * r_survival
        + weights["action"] * r_action
        + weights["action_rate"] * r_action_rate
        + weights["smoothness"] * r_smoothness
        + weights["yaw_alignment"] * r_yaw_alignment
        + weights["progress"] * r_progress
        + weights["precision"] * r_precision
        + weights["obstacle"] * r_obstacle
        + weights["out_of_map"] * r_out_of_map
    )
    return total, goal_dist
