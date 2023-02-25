
"""Dynamically build functions."""

import jax
import time
import jax.numpy as jnp
from grgrjax import jvp_vmap, vjp_vmap
from .het_agent_funcs import *


def func_stst_rep_agent(y, func_pre_stst, func_eqns):
    x, par = func_pre_stst(y)
    x = x[..., None]
    return func_eqns(x, x, x, x, pars=par), None


def func_stst_het_agent(y, func_pre_stst, find_stat_vf, func_stst_dist, func_eqns):

    x, par = func_pre_stst(y)
    x = x[..., None]

    vf, decisions_output, cnt_backw = find_stat_vf(
        x, par)
    dist, cnt_forw = func_stst_dist(decisions_output)

    # TODO: for more than one dist this should be a loop...
    decisions_output_array = decisions_output[..., None]
    dist_array = dist[..., None]

    aux = (vf, decisions_output, cnt_backw), (dist, cnt_forw)
    out = func_eqns(x, x, x, x, pars=par, distributions=dist_array,
                    decisions_outputs=decisions_output_array)

    return out, aux


def get_func_stst_raw(func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, tol_backw, maxit_backw, tol_forw, maxit_forw):
    """Get a function that evaluates the steady state
    """

    zshock = jnp.zeros(len(shocks))
    partial_pre_stst = jax.tree_util.Partial(func_pre_stst)
    partial_eqns = jax.tree_util.Partial(func_eqns, shocks=zshock)

    if not func_stst_dist:
        return jax.tree_util.Partial(func_stst_rep_agent, func_pre_stst=partial_pre_stst, func_eqns=partial_eqns)

    partial_backw = jax.tree_util.Partial(func_backw, shocks=zshock)
    carry = (init_vf, decisions_output_init), (init_vf +
                                               1, 0), (partial_backw, tol_backw, maxit_backw)
    backwards_stst = jax.tree_util.Partial(backwards_sweep_stst, carry=carry)
    forwards_stst = jax.tree_util.Partial(
        func_stst_dist, tol=tol_forw, maxit=maxit_forw)

    return jax.tree_util.Partial(func_stst_het_agent, func_pre_stst=partial_pre_stst, find_stat_vf=backwards_stst, func_stst_dist=forwards_stst, func_eqns=partial_eqns)


def get_stst_derivatives(model, nvars, pars, stst, x_stst, zshocks, horizon, verbose):

    st = time.time()

    func_eqns = model['context']["func_eqns"]
    backwards_sweep = model['context']['backwards_sweep']
    second_sweep = model['context']['second_sweep']

    distSS = jnp.array(model['steady_state']['distributions'])
    decisions_outputSS = jnp.array(
        list(model['steady_state']['decisions'].values()))[..., None]

    # basis for steady state jacobian construction
    basis = jnp.zeros((nvars*(horizon-1), nvars))
    basis = basis.at[-nvars:].set(jnp.eye(nvars))

    # get steady state jacobians for dist transition
    doSS, do2x = jvp_vmap(backwards_sweep)(
        (x_stst[1:-1].flatten(), stst, zshocks), (basis,))
    _, (f2do,) = vjp_vmap(second_sweep, argnums=1)(
        (x_stst[1:-1].flatten(), doSS, stst, distSS, zshocks), basis.T)
    f2do = jnp.moveaxis(f2do, -1, 1)

    # get steady state jacobians for direct effects x on f
    jacrev_func_eqns = jax.jacrev(func_eqns, argnums=(0, 1, 2))
    f2X = jacrev_func_eqns(stst[:, None], stst[:, None], stst[:, None],
                           stst, zshocks[:, 0], pars, distSS[..., None], decisions_outputSS)

    if verbose:
        duration = time.time() - st
        print(
            f"(get_derivatives:) Derivatives calculation done ({duration:1.3f}s).")

    return f2X, f2do, do2x


def get_stacked_func_het_agents(pars, func_backw, func_dist, func_eqns, stst, vfSS, horizon, nvars):
    """Get a function that returns the (flattend) value and Jacobian of the stacked aggregate model equations.
    """

    nshpe = (nvars, horizon-1)
    # build partials of input functions
    partial_backw = jax.tree_util.Partial(func_backw, XSS=stst, pars=pars)
    partial_dist = jax.tree_util.Partial(func_dist)

    # build partials of output functions
    backwards_sweep_local = jax.tree_util.Partial(
        backwards_sweep, stst=stst, vfSS=vfSS, horizon=horizon, func_backw=partial_backw)
    forwards_sweep_local = jax.tree_util.Partial(
        forwards_sweep, horizon=horizon, func_dist=partial_dist)
    final_step_local = jax.tree_util.Partial(
        final_step, stst=stst, horizon=horizon, nshpe=nshpe, pars=pars, func_eqns=func_eqns)
    second_sweep_local = jax.tree_util.Partial(
        second_sweep, forwards_sweep=forwards_sweep_local, final_step=final_step_local)
    stacked_func_dist_local = jax.tree_util.Partial(
        stacked_func_het_agents, backwards_sweep=backwards_sweep_local, second_sweep=second_sweep_local)

    return stacked_func_dist_local, backwards_sweep_local, forwards_sweep_local, second_sweep_local


def build_aggr_het_agent_funcs(model, nvars, pars, stst, zshocks, horizon):

    shocks = model.get("shocks") or ()
    # get functions
    func_eqns = model['context']["func_eqns"]
    func_backw = model['context'].get('func_backw')
    func_dist = model['context'].get('func_dist')

    # get stuff for het-agent models
    vfSS = model['steady_state'].get('value_functions')
    distSS = jnp.array(model['steady_state']['distributions'])

    # get actual functions
    func_raw, backwards_sweep, forwards_sweep, second_sweep = get_stacked_func_het_agents(
        pars, func_backw, func_dist, func_eqns, stst, vfSS, horizon, nvars)

    # store everything
    model['context']['func_raw'] = func_raw
    model['context']['backwards_sweep'] = backwards_sweep
    model['context']['forwards_sweep'] = forwards_sweep
    model['context']['second_sweep'] = second_sweep
    model['context']['jvp_func'] = lambda primals, tangens, x0, dist0, shocks: jax.jvp(
        func_raw, (primals, x0, dist0, shocks), (tangens, jnp.zeros(nvars), jnp.zeros_like(distSS), zshocks))
    model['context']['vjp_func'] = lambda primals, tangens, x0, dist0, shocks: jax.vjp(
        lambda x: func_raw(x, x0, dist0, shocks), primals)[1](tangens)[0]
