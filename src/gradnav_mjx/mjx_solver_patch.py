"""
mjx_solver_patch.py

Stock mujoco.mjx._src.solver.solve() has two code paths (solver.py:598-602):
  - m.opt.iterations == 1: a single un-looped body(ctx) call (differentiable,
    but only ever does ONE CG iteration -- too coarse for stiff/high-speed
    contact).
  - m.opt.iterations > 1: jax.lax.while_loop(cond, body, ctx) -- NOT
    reverse-mode differentiable.

mjx's own linesearch (solver.py:542) already solves this exact problem with
`_while_loop_scan`: a jax.lax.scan over a *fixed* max_iter that applies
jax.lax.cond per step to no-op once `cond_fun` is satisfied -- same runtime
behavior as a while_loop, fully differentiable ("Scan-based implementation
(jit ok, reverse-mode autodiff ok)").

This patch swaps solve()'s outer CG loop to use that same scan-based
construct for any m.opt.iterations > 1, so a fixed multi-iteration contact
solve can run every physics step while staying differentiable end to end.
"""

import jax.numpy as jp
import mujoco
from mujoco.mjx._src import solver as _solver_mod
from mujoco.mjx._src.types import DisableBit


def _patched_solve(m, d):
    if not isinstance(m.opt._impl, _solver_mod.OptionJAX):
        raise ValueError('solve requires JAX backend implementation.')

    def cond(ctx):
        improvement = _solver_mod._rescale(m, ctx.prev_cost - ctx.cost)
        gradient = _solver_mod._rescale(m, _solver_mod.math.norm(ctx.grad))
        done = ctx.solver_niter >= m.opt.iterations
        done |= improvement < m.opt.tolerance
        done |= gradient < m.opt.tolerance
        return ~done

    def body(ctx):
        ctx = _solver_mod._linesearch(m, d, ctx)
        prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad
        ctx = _solver_mod._update_constraint(m, d, ctx)
        ctx = _solver_mod._update_gradient(m, d, ctx)

        if m.opt.solver == _solver_mod.SolverType.NEWTON:
            search = -ctx.Mgrad
        else:
            beta = jp.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
            beta = beta / jp.maximum(mujoco.mjMINVAL, jp.dot(prev_grad, prev_Mgrad))
            beta = jp.maximum(0, beta)
            search = -ctx.Mgrad + beta * ctx.search
        ctx = ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)
        return ctx

    qacc = d.qacc_smooth
    if not m.opt.disableflags & DisableBit.WARMSTART:
        warm = _solver_mod.Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
        smth = _solver_mod.Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
        qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
    d = d.replace(qacc=qacc)

    ctx = _solver_mod.Context.create(m, d)
    if m.opt.iterations == 1:
        ctx = body(ctx)
    else:
        ctx = _solver_mod._while_loop_scan(cond, body, ctx, m.opt.iterations)

    d = d.tree_replace({
        'qfrc_constraint': ctx.qfrc_constraint,
        'qacc': ctx.qacc,
        '_impl.efc_force': ctx.efc_force,
    })
    return d


_ORIGINAL_SOLVE = _solver_mod.solve


def apply():
    """Install the patch. ONLY for DiffRL, which needs to differentiate
    through the contact solve.

    Do not apply this for a model-free learner. The scan-based solve
    trades solver behaviour for differentiability, and in contact-rich
    scenes it goes unstable: in an enclosed 16x16m room the car reached
    19.2 m/s (its calibrated top speed is ~3.5-3.9) and was launched
    clean through the perimeter to y=+94, versus 3.25 m/s and no escape
    on the stock solver. That corrupted every room result -- constant
    wall contacts meant constant instability -- while the earlier
    open-field maps hid it, since a car that rarely touches anything
    rarely invokes the contact solver at all.
    """
    _solver_mod.solve = _patched_solve
    import mujoco.mjx as mjx
    mjx.solve = _patched_solve


def restore():
    """Put the stock MJX solver back."""
    _solver_mod.solve = _ORIGINAL_SOLVE
    import mujoco.mjx as mjx
    mjx.solve = _ORIGINAL_SOLVE
