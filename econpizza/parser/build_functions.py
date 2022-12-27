
"""Dynamically build functions."""

import jax
import jax.numpy as jnp
from grgrlib.jaxed import *
from .het_agent_funcs import _backwards_sweep, _forwards_sweep, _final_step, _second_sweep, _stacked_func_dist


def get_func_stst_raw(func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, exog_grid_vars_init, tol_backw, maxit_backw, tol_forw, maxit_forw):
    """Get a function that evaluates the steady state
    """

    zshock = jnp.zeros(len(shocks))

    def cond_func(cont):
        (vf, _, _), (vf_old, cnt), _ = cont
        cond0 = jnp.abs(vf - vf_old).max() > tol_backw
        cond1 = cnt < maxit_backw
        return jnp.logical_and(cond0, cond1)

    def body_func(cont):
        (vf, _, _), (_, cnt), (x, par) = cont
        return func_backw(x, x, x, x, vf, zshock, par), (vf, cnt + 1), (x, par)

    def find_stat_vf(x, par):

        (vf, decisions_output, exog_grid_vars), (_, cnt), _ = jax.lax.while_loop(
            cond_func, body_func, ((init_vf, decisions_output_init, exog_grid_vars_init), (init_vf+1, 0), (x, par)))

        return vf, decisions_output, exog_grid_vars, cnt

    def func_stst_raw(y):

        x, par = func_pre_stst(y)
        x = x[..., None]

        if not func_stst_dist:
            return func_eqns(x, x, x, x, zshock, par), None

        vf, decisions_output, exog_grid_vars, cnt_backw = find_stat_vf(
            x, par)
        dist, cnt_forw = func_stst_dist(decisions_output, tol_forw, maxit_forw)

        # TODO: for more than one dist this should be a loop...
        decisions_output_array = decisions_output[..., None]
        dist_array = dist[..., None]

        aux = (vf, decisions_output, exog_grid_vars,
               cnt_backw), (dist, cnt_forw)
        out = func_eqns(x, x, x, x, zshock, par,
                        dist_array, decisions_output_array)

        return out, aux

    return func_stst_raw


def get_stacked_func_dist(pars, func_backw, func_dist, func_eqns, stst, vfSS, distSS, horizon, nvars):
    """Get a function that returns the (flattend) value and Jacobian of the stacked aggregate model equations.
    """

    nshpe = (nvars, horizon-1)
    # build partials of input functions
    func_backw = jax.tree_util.Partial(func_backw, XSS=stst, pars=pars)
    func_dist = jax.tree_util.Partial(func_dist)

    # build partials of output functions
    backwards_sweep = jax.tree_util.Partial(
        _backwards_sweep, stst=stst, vfSS=vfSS, horizon=horizon, func_backw=func_backw)
    forwards_sweep = jax.tree_util.Partial(
        _forwards_sweep, distSS=distSS, horizon=horizon, func_dist=func_dist)
    final_step = jax.tree_util.Partial(
        _final_step, stst=stst, horizon=horizon, nshpe=nshpe, pars=pars, func_eqns=func_eqns)
    second_sweep = jax.tree_util.Partial(
        _second_sweep, forwards_sweep=forwards_sweep, final_step=final_step)
    stacked_func_dist = jax.tree_util.Partial(
        _stacked_func_dist, backwards_sweep=backwards_sweep, second_sweep=second_sweep)

    return stacked_func_dist, backwards_sweep, forwards_sweep, second_sweep


def get_derivatives(model, nvars, pars, stst, x_stst, zshocks, horizon, verbose):

    st = time.time()

    shocks = model.get("shocks") or ()
    # get functions
    func_eqns = model['context']["func_eqns"]
    func_backw = model['context'].get('func_backw')
    func_dist = model['context'].get('func_dist')

    # get stuff for het-agent models
    vfSS = model['steady_state'].get('decisions')
    distSS = jnp.array(model['steady_state']['distributions'])[..., None]
    decisions_outputSS = jnp.array(
        model['steady_state']['decisions_output'])[..., None]

    # get actual functions
    func_raw, backwards_sweep, forwards_sweep, second_sweep = get_stacked_func_dist(
        pars, func_backw, func_dist, func_eqns, stst, vfSS, distSS[..., 0], horizon, nvars)

    # basis for steady state jacobian construction
    basis = jnp.zeros((nvars*(horizon-1), nvars))
    basis = basis.at[-nvars:, -nvars:].set(jnp.eye(nvars))

    # get steady state jacobians for dist transition
    doSS, do2x = jvp_vmap(backwards_sweep)(
        (x_stst[1:-1].flatten(), stst, zshocks), (basis,))
    _, (f2do,) = vjp_vmap(second_sweep, argnums=1)(
        (x_stst[1:-1].flatten(), doSS, stst, zshocks), basis.T)
    f2do = jnp.moveaxis(f2do, -1, 1)

    # get steady state jacobians for direct effects x on f
    jacrev_func_eqns = jax.jacrev(func_eqns, argnums=(0, 1, 2))
    f2X = jacrev_func_eqns(stst[:, None], stst[:, None], stst[:, None],
                           stst, zshocks[:, 0], pars, distSS, decisions_outputSS)

    # store everything
    model['context']['func_raw'] = func_raw
    model['context']['backwards_sweep'] = backwards_sweep
    model['context']['forwards_sweep'] = forwards_sweep
    model['jvp_func'] = lambda primals, tangens, x0, shocks: jax.jvp(
        func_raw, (primals, x0, shocks), (tangens, jnp.zeros(nvars), zshocks))

    if verbose:
        duration = time.time() - st
        print(
            f"(get_derivatives:) Derivatives calculation done ({duration:1.3f}s).")

    return f2X, f2do, do2x
