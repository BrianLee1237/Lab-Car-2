"""
train_diffmjx_final.py (v3)

Added the two things that made the PyTorch version actually show real
learning: curriculum (start with close goals, expand over training)
and a FIXED evaluation benchmark (same seed every check, decoupled
from the curriculum/random training goals) so we can actually see if
mean_final_dist is improving, not just bouncing around noisily.

v3: dropped the DiffMJX fork dependency (scan_loop/col_soft_enable/
softjax_mode/col_softness are custom fields on martius-lab's forked
`mjx.Option` -- they don't exist on stock mujoco-mjx and raise on
`m.opt.replace(**DIFFMJX_OPTS)`). Root-caused the NaN separately:
with `m.opt.iterations` left at its default, mjx.step's contact
solver runs `jax.lax.while_loop`, which is not reverse-mode
differentiable, so `jax.grad` through a multi-step rollout never
finishes. The XML already sets `iterations="1"` to force the
single-shot, differentiable solver body (see solver.py:599's
`if m.opt.iterations == 1: ctx = body(ctx)` special case) -- but a
single CG/linesearch iteration isn't always enough to resolve a
stiff, high-speed contact (car hitting a wall), so qvel could blow
up over a handful of steps and go NaN. Clamping qvel after every
`mjx.step` call bounds the state and keeps the whole rollout (and
its gradient) finite; confirmed on this scene with a 60-step
rollout (forward value and `jax.grad` both finite, no NaNs).

mjx_solver_patch.py (applied below) fixes a further, separate issue:
`iterations=1` also means the contact solve isn't very *accurate*
(one CG step), not just non-differentiable at higher iteration counts.
An open-loop full-throttle calibration test showed the car's
distance-covered bouncing erratically step to step (wobble/jitter)
instead of increasing smoothly. mjx already ships a differentiable
scan-based while-loop internally for its linesearch
(`_while_loop_scan`, docstring: "reverse-mode autodiff ok") -- the
patch reuses that same construct for the outer CG solve too, so
`iterations` can be raised above 1 (mjx_car_scene.py sets 4) while
staying differentiable. Combined with the real MuSHR hardware's
actual tire friction and drivetrain torque (mu=2.0, gear=0.78,
derived from the traction limit implied by the real racecar.urdf's
`mu1=2` wheel friction spec -- github.com/dhruvdotc/Lab-Car-2), the
calibration test then shows smooth, monotonic progress with no wobble.

Caution: the multi-iteration solve is significantly more memory-
hungry during backprop (it retains state across all `iterations`
solver steps per physics step, not just one). Training at the long
horizon multi-meter goals need (roughly 3500-4000 steps per meter at
this car's real, friction-limited cruise speed of ~0.11-0.21 m/s) can
need far more device memory than a typical sandboxed/cloud dev
environment provides -- confirmed OOM in exactly that kind of
environment even at batch_size=2, horizon=3500 (~1m target). If you
hit OOM, reduce --batch-size and/or --horizon, or run on a machine
with more GPU/TPU memory.
"""

import argparse

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

import mjx_solver_patch
mjx_solver_patch.apply()

from mjx_car_scene import build_car_scene_xml
from mjx_random_maps import generate_map_set
from mjx_obstacle_dist import wall_distances
from jax_networks import init_policy_params, policy_forward, init_value_params, value_forward
from jax_reward import jax_reward


CAR_RADIUS = 0.24
QVEL_CLAMP = 15.0


def adam_init(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": m, "v": v, "t": 0}


def adam_update(params, grads, state, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * (g ** 2), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return new_params, {"m": m, "v": v, "t": t}


def get_car_xy_heading(data):
    x, y = data.qpos[0], data.qpos[1]
    qw, qz = data.qpos[3], data.qpos[6]
    theta = 2 * jnp.arctan2(qz, qw)
    return x, y, theta


OBS_DIST_SCALE = 5.0
OBS_V_SCALE = 3.0


def build_obs(data, goal, walls):
    """Returns a *normalized* obs vector for the policy/value networks
    (unnormalized inputs of magnitude 5-20 were saturating the tanh
    MLPs at init -- pre-activations up to +-19, so gradients through
    the first layer were ~0 and the policy never moved off its
    initial output regardless of training) plus the raw quantities
    (x, y, theta, obstacle_d) used by the reward function, which are
    unaffected by this normalization."""
    x, y, theta = get_car_xy_heading(data)
    v = jnp.sqrt(data.qvel[0] ** 2 + data.qvel[1] ** 2)
    goal_dx = goal[0] - x
    goal_dy = goal[1] - y
    obstacle_d = wall_distances(jnp.array([x, y]), walls)

    obstacle_d_n = jnp.clip(obstacle_d, -OBS_DIST_SCALE, OBS_DIST_SCALE) / OBS_DIST_SCALE
    v_n = jnp.clip(v, 0.0, OBS_V_SCALE) / OBS_V_SCALE
    theta_n = theta / jnp.pi
    goal_dx_n = jnp.clip(goal_dx, -OBS_DIST_SCALE, OBS_DIST_SCALE) / OBS_DIST_SCALE
    goal_dy_n = jnp.clip(goal_dy, -OBS_DIST_SCALE, OBS_DIST_SCALE) / OBS_DIST_SCALE

    extra = jnp.stack([v_n, theta_n, goal_dx_n, goal_dy_n])
    obs = jnp.concatenate([obstacle_d_n, extra])
    return obs, x, y, theta, obstacle_d


BPTT_WINDOW = 32
SUCCESS_DIST = 0.5
TERMINAL_BONUS = 50.0


def rollout(mjx_model, policy_params, value_params, target_value_params, walls, goal, horizon, gamma=0.99):
    """Uses jax.lax.scan instead of a Python for-loop over `horizon` --
    unrolling in Python builds one XLA op per step per rollout, and once
    that's vmapped over a batch and differentiated it blows up compile
    time/memory (observed OOM at horizon=30, batch=4 on a memory-limited
    box). scan compiles the step body once and loops it, and is fully
    supported by reverse-mode autodiff.

    Full backprop-through-time over the whole horizon is not usable here:
    differentiating through mjx's contact solver compounds every step
    (measured: policy gradient norm ~-3.7e7 at 200 steps, ~1e25 at 600,
    NaN by 800), because iterations=1 (needed for jax.grad to work at
    all -- see the note by QVEL_CLAMP) makes each step's contact solve
    a single, none-too-accurate linesearch iteration whose local error
    compounds multiplicatively through the chain rule over long chains.
    Reaching a goal several meters away needs ~1000 sim steps (see
    horizon calibration in the corresponding write-up), so full BPTT
    over the episode isn't an option.

    Fix: truncated BPTT with periodic value-function bootstrapping
    (the short-horizon actor-critic / SHAC trick, standard for DiffRL
    over stiff/contact-rich dynamics). The physics state is detached
    with stop_gradient every BPTT_WINDOW steps, so gradients only ever
    flow back through a short, numerically stable chain; the value
    network estimates the return-to-go at each window boundary so the
    policy still gets full-episode credit assignment, just not a full
    differentiable chain for it.
    """
    data = mjx.make_data(mjx_model)
    x0, y0, _ = get_car_xy_heading(data)
    prev_goal_dist0 = jnp.sqrt((goal[0] - x0) ** 2 + (goal[1] - y0) ** 2)

    n_windows = max(1, horizon // BPTT_WINDOW)

    def inner_step(carry, _):
        (data, prev_action, prev_prev_action, prev_goal_dist, window_reward,
         window_discount, min_obstacle_dist_seen, done) = carry

        obs, x, y, theta, obstacle_d = build_obs(data, goal, walls)
        action = policy_forward(policy_params, obs)
        ctrl = jnp.array([action[0], action[1], action[1]])
        ctrl = jnp.where(done, jnp.zeros_like(ctrl), ctrl)
        prev_data = data
        data = data.replace(ctrl=ctrl)
        data = mjx.step(mjx_model, data)
        data = data.replace(qvel=jnp.clip(data.qvel, -QVEL_CLAMP, QVEL_CLAMP))
        # Once the goal has been reached, freeze the physics state entirely
        # (straight-through select, same trick as elsewhere in this file):
        # otherwise the car keeps driving with zeroed ctrl and can still
        # coast/drift past the goal under momentum, which is exactly the
        # "reached it, then wandered off before the episode ended" failure
        # mode a trajectory trace found -- final-position success was
        # undercounting real navigation success by ~23 points (26% vs. 49%
        # on an "ever reached the goal" metric) purely because there was no
        # incentive or mechanism to stop once there.
        data = jax.tree_util.tree_map(lambda old, new: jnp.where(done, old, new), prev_data, data)

        car_xy = jnp.array([x, y])
        r, goal_dist = jax_reward(
            car_xy, theta, action, prev_action, prev_prev_action,
            goal, obstacle_d, prev_goal_dist,
        )
        just_succeeded = (goal_dist < SUCCESS_DIST) & (~done)
        r = jnp.where(done, 0.0, r) + jnp.where(just_succeeded, TERMINAL_BONUS, 0.0)
        done = done | just_succeeded

        window_reward = window_reward + window_discount * r
        window_discount = window_discount * gamma
        min_obstacle_dist_seen = jnp.minimum(min_obstacle_dist_seen, jnp.min(obstacle_d))

        new_carry = (
            data, action, prev_action, goal_dist,
            window_reward, window_discount, min_obstacle_dist_seen, done,
        )
        return new_carry, None

    @jax.checkpoint
    def outer_step(carry, _):
        # jax.checkpoint (remat): jax.lax.scan's reverse-mode autodiff
        # normally retains every window's intermediate values for the
        # whole outer scan, even for windows whose gradient contribution
        # is cut below by stop_gradient -- so memory still scales with the
        # full horizon (n_windows), not just BPTT_WINDOW, even though the
        # differentiable chain itself is short. remat instead recomputes
        # each window's forward pass during the backward pass rather than
        # storing it, trading extra compute for peak memory that no
        # longer scales with horizon length. Necessary to fit the longer
        # horizons multi-meter goals need (and the extra memory the
        # multi-iteration solver patch itself uses) in limited device
        # memory (confirmed OOM without this at batch=4, horizon=7000
        # trying to allocate ~30GB; with it, the same horizon should need
        # a small, roughly constant amount of memory per window instead).
        (data, prev_action, prev_prev_action, prev_goal_dist, total_return,
         global_discount, min_obstacle_dist_seen, value_loss, done) = carry

        # truncate the differentiable chain: nothing before this point in
        # the episode contributes gradient to what happens in this window.
        data = jax.lax.stop_gradient(data)
        prev_action = jax.lax.stop_gradient(prev_action)
        prev_prev_action = jax.lax.stop_gradient(prev_prev_action)
        prev_goal_dist = jax.lax.stop_gradient(prev_goal_dist)

        window_start_obs, _, _, _, _ = build_obs(data, goal, walls)

        window_carry0 = (data, prev_action, prev_prev_action, prev_goal_dist,
                          jnp.array(0.0), jnp.array(1.0), min_obstacle_dist_seen, done)
        window_final, _ = jax.lax.scan(inner_step, window_carry0, None, length=BPTT_WINDOW)
        (data, action, prev_action_out, goal_dist, window_reward, window_discount,
         min_obstacle_dist_seen, done) = window_final

        final_obs, _, _, _, _ = build_obs(data, goal, walls)
        # Bootstrap with the *target* critics (slow, Polyak-averaged copies
        # of value_params, updated outside this rollout/gradient) rather
        # than the live ones. SHAC's own writeup notes the live critic's
        # value landscape is noisy enough on harder tasks that bootstrapping
        # with it directly destabilizes training; a delayed target fixes it
        # the same way it does in DDPG/TD3/SAC-style target networks.
        #
        # Double critic (TD3-style, per AHAC -- Georgiev et al. 2024, the
        # direct successor to SHAC built for exactly this contact-rich
        # setting): value_params/target_value_params are each a pair of
        # independently-initialized critics; bootstrapping with their min
        # counters the single-critic overestimation bias that otherwise
        # compounds through the return, and gives each critic a slightly
        # different TD target so their errors decorrelate over training.
        vt1, vt2 = target_value_params
        bootstrap_val = jnp.minimum(
            value_forward(vt1, final_obs), value_forward(vt2, final_obs)
        )
        bootstrap = window_discount * bootstrap_val
        total_return = total_return + global_discount * (window_reward + bootstrap)
        global_discount = global_discount * window_discount

        # SHAC-style critic loss: regress V(obs at window start) toward the
        # actual window return + bootstrap (a semi-gradient TD target --
        # stop_gradient so the target doesn't get optimized, only the
        # prediction does). window_start_obs is already detached from
        # everything before this window by the stop_gradient above, so this
        # term contributes zero gradient to policy_params, only to
        # value_params -- proper actor/critic separation falls out of the
        # truncation scheme for free. Both critics regress to the same
        # (shared, min-based) target -- only their initialization differs.
        td_target = jax.lax.stop_gradient(window_reward + bootstrap)
        v1, v2 = value_params
        pred_value_1 = value_forward(v1, window_start_obs)
        pred_value_2 = value_forward(v2, window_start_obs)
        value_loss = value_loss + (pred_value_1 - td_target) ** 2 + (pred_value_2 - td_target) ** 2

        new_carry = (
            data, action, prev_action_out, goal_dist,
            total_return, global_discount, min_obstacle_dist_seen, value_loss, done,
        )
        return new_carry, None

    carry0 = (
        data,
        jnp.zeros(2),          # prev_action
        jnp.zeros(2),          # prev_prev_action
        prev_goal_dist0,
        jnp.array(0.0),        # total_return
        jnp.array(1.0),        # global_discount
        jnp.array(jnp.inf),    # min_obstacle_dist_seen
        jnp.array(0.0),        # value_loss
        jnp.array(False),      # done (goal reached -> freeze state, per-goal terminal bonus)
    )
    final_carry, _ = jax.lax.scan(outer_step, carry0, None, length=n_windows)
    data, _, _, _, total_return, _, min_obstacle_dist_seen, value_loss, _ = final_carry

    final_obs, x, y, theta, _ = build_obs(data, goal, walls)
    final_dist = jnp.sqrt((goal[0] - x) ** 2 + (goal[1] - y) ** 2)
    return total_return, final_dist, min_obstacle_dist_seen, value_loss / n_windows


VALUE_LOSS_WEIGHT = 0.5


def batched_loss(mjx_model, policy_params, value_params, target_value_params, walls, goals, horizon):
    def single(goal):
        return rollout(mjx_model, policy_params, value_params, target_value_params, walls, goal, horizon)
    returns, final_dists, min_obstacle_dists, value_losses = jax.vmap(single)(goals)
    actor_loss = -jnp.mean(returns)
    critic_loss = jnp.mean(value_losses)
    loss = actor_loss + VALUE_LOSS_WEIGHT * critic_loss
    return loss, final_dists, min_obstacle_dists, actor_loss, critic_loss


def sample_goals(key, batch_size, min_dist, max_dist):
    angle_key, dist_key = jax.random.split(key)
    angles = jax.random.uniform(angle_key, (batch_size,), minval=0, maxval=2 * jnp.pi)
    dists = jax.random.uniform(dist_key, (batch_size,), minval=min_dist, maxval=max_dist)
    return jnp.stack([dists * jnp.cos(angles), dists * jnp.sin(angles)], axis=-1)


def clip_tree(grads, max_norm=5.0):
    leaves = jax.tree_util.tree_leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def sanitize_grads(grads):
    return jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)), grads
    )


def curriculum_goal_range(progress, max_dist, min_start=0.5):
    current_max = min_start + progress * (max_dist - min_start)
    return 0.3, current_max


def evaluate_fixed_benchmark(mjx_model, policy_params, value_params, walls, horizon,
                              n_eval=48, min_dist=0.3, max_dist=1.0, eval_chunk=None):
    """Always full difficulty, ALWAYS the same seed -- decoupled from
    training curriculum, so this is a trustworthy, comparable metric
    across the whole run. n_eval=48 (up from an earlier 16): with only 16
    goals a single flipped success/failure swings the reported rate by
    ~6 percentage points, which was making run-to-run comparisons mostly
    noise -- 48 cuts that swing to ~2 points.

    min_dist/max_dist should match whatever --max-dist the run trained
    with -- calibrate the car's actual travel envelope under your horizon
    first (an open-loop full-throttle rollout; see mjx_solver_patch.py's
    module docstring) rather than guessing, otherwise this silently tests
    goals the car cannot physically reach in the episode regardless of
    policy quality, which caps success rate on reachability, not skill
    (this exact bug previously existed with a hardcoded 0.5-2.5m range
    against a ~1.0-1.1m actual envelope).

    eval_chunk: if set, evaluates in batches of this size sequentially
    instead of one big vmap -- necessary at long horizons where a full
    n_eval-sized batch can exceed available device memory during the
    forward pass (observed OOM at n_eval=300, horizon=14000)."""
    g = jax.random.PRNGKey(999)
    goals = sample_goals(g, n_eval, min_dist, max_dist)
    if eval_chunk is None:
        eval_chunk = n_eval
    final_dists_chunks, min_obs_chunks = [], []
    for i in range(0, n_eval, eval_chunk):
        chunk_goals = goals[i:i + eval_chunk]
        _, cfd, cmo, _, _ = batched_loss(
            mjx_model, policy_params, value_params, value_params, walls, chunk_goals, horizon
        )
        final_dists_chunks.append(cfd)
        min_obs_chunks.append(cmo)
    final_dists = jnp.concatenate(final_dists_chunks)
    min_obstacle_dists = jnp.concatenate(min_obs_chunks)
    valid = jnp.isfinite(final_dists)
    mean_dist = jnp.mean(jnp.where(valid, final_dists, 0.0)) / jnp.maximum(jnp.mean(valid), 1e-6)
    n_collisions = jnp.sum(min_obstacle_dists < CAR_RADIUS)
    success_rate = jnp.mean((final_dists < 0.5) & valid)
    return mean_dist, n_collisions, success_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-final-frac", type=float, default=0.1,
                         help="final LR as a fraction of --lr, linearly decayed over training")
    parser.add_argument("--target-tau", type=float, default=0.005,
                         help="Polyak averaging rate for the target critic")
    parser.add_argument("--max-dist", type=float, default=2.5)
    parser.add_argument("--n-maps", type=int, default=4)
    parser.add_argument("--n-walls", type=int, default=4)
    parser.add_argument("--switch-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-n", type=int, default=48)
    parser.add_argument("--eval-min-dist", type=float, default=0.3)
    parser.add_argument("--eval-chunk", type=int, default=None,
                         help="evaluate in batches of this size instead of one big vmap -- "
                              "use at long horizons to avoid OOM during eval (try e.g. 10-20)")
    args = parser.parse_args()

    map_wall_lists = generate_map_set(n_maps=args.n_maps, n_walls=args.n_walls, base_seed=args.seed)
    map_walls_arrays = [jnp.array(w) for w in map_wall_lists]
    mjx_models = []
    for i, wall_list in enumerate(map_wall_lists):
        scene_path = build_car_scene_xml(wall_list, out_path=f"diffmjx_map_{i}.xml")
        model = mujoco.MjModel.from_xml_path(scene_path)
        m = mjx.put_model(model)
        mjx_models.append(m)
    print(f"Built {args.n_maps} randomized maps (stock mujoco-mjx, patched multi-iteration "
          f"differentiable solver + qvel clamp).")

    key = jax.random.PRNGKey(args.seed)
    pkey, vkey1, vkey2 = jax.random.split(key, 3)
    policy_params = init_policy_params(pkey, lidar_dim=args.n_walls)
    # double critic (AHAC/TD3-style): two independently-initialized value
    # networks, bootstrapped via their min -- see the note in rollout().
    value_params = (
        init_value_params(vkey1, lidar_dim=args.n_walls),
        init_value_params(vkey2, lidar_dim=args.n_walls),
    )
    target_value_params = value_params  # starts identical to the live critics
    policy_opt_state = adam_init(policy_params)
    value_opt_state = adam_init(value_params)

    grad_fn = jax.value_and_grad(
        lambda pp, vp, tvp, mjx_model, walls, goals: batched_loss(
            mjx_model, pp, vp, tvp, walls, goals, args.horizon
        )[0],
        argnums=(0, 1),
    )

    @jax.jit
    def train_step(policy_params, value_params, target_value_params, policy_opt_state, value_opt_state,
                    mjx_model, walls, goals, lr):
        loss, (pgrads, vgrads) = grad_fn(policy_params, value_params, target_value_params, mjx_model, walls, goals)
        pgrads = clip_tree(sanitize_grads(pgrads))
        vgrads = clip_tree(sanitize_grads(vgrads))
        policy_params, policy_opt_state = adam_update(policy_params, pgrads, policy_opt_state, lr=lr)
        value_params, value_opt_state = adam_update(value_params, vgrads, value_opt_state, lr=lr)
        return policy_params, value_params, policy_opt_state, value_opt_state, loss

    @jax.jit
    def update_target(value_params, target_value_params, tau):
        return jax.tree_util.tree_map(
            lambda v, t: tau * v + (1 - tau) * t, value_params, target_value_params
        )

    eval_fn = jax.jit(
        lambda mjx_model, pp, vp, walls: evaluate_fixed_benchmark(
            mjx_model, pp, vp, walls, args.horizon,
            n_eval=args.eval_n, min_dist=args.eval_min_dist, max_dist=args.max_dist,
            eval_chunk=args.eval_chunk,
        )
    )

    best_success = -1.0
    best_policy_params = policy_params
    best_value_params = value_params

    for it in range(args.iterations):
        map_idx = (it // args.switch_every) % args.n_maps
        mjx_model = mjx_models[map_idx]
        walls = map_walls_arrays[map_idx]

        progress = min(1.0, it / (args.iterations * 0.7))
        min_dist, max_dist = curriculum_goal_range(progress, args.max_dist)

        # linear LR decay: args.lr -> args.lr * args.lr_final_frac over training
        lr_progress = it / max(1, args.iterations - 1)
        lr = args.lr * (1.0 - lr_progress) + (args.lr * args.lr_final_frac) * lr_progress

        key, gkey = jax.random.split(key)
        goals = sample_goals(gkey, args.batch_size, min_dist, max_dist)

        policy_params, value_params, policy_opt_state, value_opt_state, loss = train_step(
            policy_params, value_params, target_value_params, policy_opt_state, value_opt_state,
            mjx_model, walls, goals, lr,
        )
        target_value_params = update_target(value_params, target_value_params, args.target_tau)

        if (it + 1) % args.eval_every == 0:
            eval_map = mjx_models[0]
            eval_walls = map_walls_arrays[0]
            eval_dist, eval_collisions, eval_success = eval_fn(eval_map, policy_params, value_params, eval_walls)
            loss_display = float(loss) if jnp.isfinite(loss) else float("nan")
            print(f"iter {it+1:4d}  map={map_idx}  progress={progress:.2f}  goal_range=[{min_dist:.1f},{max_dist:.1f}]  lr={lr:.2e}  "
                  f"loss={loss_display:.3f}  EVAL_dist(fixed)={float(eval_dist):.3f}  "
                  f"EVAL_success={float(eval_success)*100:.0f}%  EVAL_collisions={int(eval_collisions)}/{args.eval_n}")

            if float(eval_success) > best_success:
                best_success = float(eval_success)
                best_policy_params = policy_params
                best_value_params = value_params
                print(f"         -> new best checkpoint (success={best_success*100:.0f}%) saved in memory")

    print(f"\nDone. Best checkpoint during training: success={best_success*100:.0f}% (on the standard {args.eval_n}-goal eval).")

    # Standard ML practice: report/ship the best-validation checkpoint, not
    # blindly the final iteration's -- and re-score it on a much larger,
    # lower-variance goal sample (48 goals means one flipped success/failure
    # swings the reported rate by ~2 points; 300 goals cuts that to <0.3).
    print("Re-evaluating best checkpoint on a larger (n=300) goal sample for a low-variance estimate...")
    eval_map = mjx_models[0]
    eval_walls = map_walls_arrays[0]
    large_dist, large_collisions, large_success = evaluate_fixed_benchmark(
        eval_map, best_policy_params, best_value_params, eval_walls, args.horizon, n_eval=300,
        min_dist=args.eval_min_dist, max_dist=args.max_dist, eval_chunk=args.eval_chunk,
    )
    print(f"LARGE-SAMPLE EVAL (best checkpoint, n=300): "
          f"success={float(large_success)*100:.1f}%  mean_dist={float(large_dist):.3f}  "
          f"collisions={int(large_collisions)}/300")

    import numpy as np
    np.savez("diffmjx_policy_final.npz",
             **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(policy_params)},
             **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(policy_params)})
    np.savez("diffmjx_policy_best.npz",
             **{f"p{i}_W": np.array(w) for i, (w, b) in enumerate(best_policy_params)},
             **{f"p{i}_b": np.array(b) for i, (w, b) in enumerate(best_policy_params)})
    print("Final-iteration policy saved to diffmjx_policy_final.npz")
    print("Best-checkpoint policy saved to diffmjx_policy_best.npz")


if __name__ == "__main__":
    main()
