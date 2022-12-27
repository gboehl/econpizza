
"""Dynamically build functions."""

import jax
import jax.numpy as jnp
from grgrlib.jaxed import *


def get_func_stst_raw(func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, exog_grid_vars_init, tol_backw, maxit_backw, tol_forw, maxit_forw):
    """Get a function that evaluates the steady state
    """

    zshock = jnp.zeros(len(shocks))

    def cond_func(cont):
        (vf, _, _), vf_old, cnt = cont
        cond0 = jnp.abs(vf - vf_old).max() > tol_backw
        cond1 = cnt < maxit_backw
        return cond0 & cond1

    def body_func_raw(cont, x, par):
        (vf, _, _), _, cnt = cont
        return func_backw(x, x, x, x, vf, zshock, par), vf, cnt + 1

    def func_backw_ext(x, par):

        def body_func(cont): return body_func_raw(cont, x, par)

        (vf, decisions_output, exog_grid_vars), _, cnt = jax.lax.while_loop(
            cond_func, body_func, ((init_vf, decisions_output_init, exog_grid_vars_init), init_vf+1, 0))

        return vf, decisions_output, exog_grid_vars, cnt

    def func_stst_raw(y, full_output=False):

        x, par = func_pre_stst(y)
        x = x[..., None]

        if not func_stst_dist:
            return func_eqns(x, x, x, x, zshock, par)

        vf, decisions_output, exog_grid_vars, cnt_backw = func_backw_ext(
            x, par)
        dist, cnt_forw = func_stst_dist(decisions_output, tol_forw, maxit_forw)

        # TODO: for more than one dist this should be a loop...
        decisions_output_array = decisions_output[..., None]
        dist_array = dist[..., None]

        if full_output:
            return (vf, decisions_output, exog_grid_vars, cnt_backw), (dist, cnt_forw)

        return func_eqns(x, x, x, x, zshock, par, dist_array, decisions_output_array)

    return func_stst_raw


def get_stacked_func_dist(pars, func_backw, func_dist, func_eqns, stst, vfSS, distSS, horizon, nvars):
    """Get a function that returns the (flattend) value and Jacobian of the stacked aggregate model equations.
    """

    nshpe = (nvars, horizon-1)

    def backwards_step(carry, i):

        vf, X, shocks = carry
        vf, decisions_output, exog_grid_vars = func_backw(
            X[:, i], X[:, i+1], X[:, i+2], stst, vf, shocks[:, i], pars)

        return (vf, X, shocks), decisions_output

    def backwards_sweep(x, x0, shocks):

        X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T

        _, decisions_output_storage = jax.lax.scan(
            backwards_step, (vfSS, X, shocks), jnp.arange(horizon-1), reverse=True)
        decisions_output_storage = jnp.moveaxis(
            decisions_output_storage, 0, -1)

        return decisions_output_storage

    def forwards_step(carry, i):

        dist_old, decisions_output_storage = carry
        dist = func_dist(dist_old, decisions_output_storage[..., i])

        return (dist, decisions_output_storage), dist_old

    def forwards_sweep(decisions_output_storage):

        _, dists_storage = jax.lax.scan(
            forwards_step, (distSS, decisions_output_storage), jnp.arange(horizon-1))
        dists_storage = jnp.moveaxis(dists_storage, 0, -1)

        return dists_storage

    def final_step(x, dists_storage, decisions_output_storage, x0, shocks):

        X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T
        out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
            nshpe), stst, shocks, pars, dists_storage, decisions_output_storage)

        return out

    def second_sweep(x, decisions_output_storage, x0, shocks):

        # forwards step
        dists_storage = forwards_sweep(decisions_output_storage)
        # final step
        out = final_step(x, dists_storage,
                         decisions_output_storage, x0, shocks)

        return out

    def stacked_func_dist(x, x0, shocks):

        # backwards step
        decisions_output_storage = backwards_sweep(x, x0, shocks)
        # combined step
        out = second_sweep(x, decisions_output_storage, x0, shocks)

        return out

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
    model['jvp'] = lambda primals, tangens, x0, shocks: jax.jvp(
        func_raw, (primals, x0, shocks), (tangens, jnp.zeros(nvars), zshocks))

    if verbose:
        duration = time.time() - st
        print(
            f"(get_derivatives:) Derivatives calculation done ({duration:1.3f}s).")

    return f2X, f2do, do2x
