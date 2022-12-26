
"""Build dynamic functions."""

import jax
import jax.numpy as jnp
import scipy.sparse as ssp
from grgrlib.jaxed import *


def get_func_stst_raw(func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, exog_grid_vars_init, tol_backw, maxit_backw, tol_forw, maxit_forw):
    """Get a function that evaluates the steady state
    """

    def cond_func(cont):
        (vf, _, _), vf_old, cnt = cont
        cond0 = jnp.abs(vf - vf_old).max() > tol_backw
        cond1 = cnt < maxit_backw
        return cond0 & cond1

    def body_func_raw(cont, x, par):
        (vf, _, _), _, cnt = cont
        return func_backw(x, x, x, x, vf, [], par), vf, cnt + 1

    def func_backw_ext(x, par):

        def body_func(cont): return body_func_raw(cont, x, par)

        (vf, decisions_output, exog_grid_vars), _, cnt = jax.lax.while_loop(
            cond_func, body_func, ((init_vf, decisions_output_init, exog_grid_vars_init), init_vf+1, 0))

        return vf, decisions_output, exog_grid_vars, cnt

    def func_stst_raw(y, full_output=False):

        x, par = func_pre_stst(y)
        x = x[..., None]

        if not func_stst_dist:
            return func_eqns(x, x, x, x, jnp.zeros(len(shocks)), par)

        vf, decisions_output, exog_grid_vars, cnt_backw = func_backw_ext(
            x, par)
        dist, cnt_forw = func_stst_dist(decisions_output, tol_forw, maxit_forw)

        # TODO: for more than one dist this should be a loop...
        decisions_output_array = decisions_output[..., None]
        dist_array = dist[..., None]

        if full_output:
            return (vf, decisions_output, exog_grid_vars, cnt_backw), (dist, cnt_forw)

        return func_eqns(x, x, x, x, [], par, dist_array, decisions_output_array)

    return func_stst_raw


def get_stacked_func_rep_agent(pars, func_eqns, stst, zshock, horizon, nvars):
    """Get a function that returns the (flattend) value and Jacobian of the stacked aggregate model equations.
    """

    nshpe = (nvars, horizon-1)

    def final_step(x, x0, xT):

        X = jnp.hstack((x0, x, xT)).reshape(horizon+1, -1).T
        out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe),
                        X[:, 2:].reshape(nshpe), stst, zshock, pars, [], [])

        return out

    return final_step


def get_stacked_func_dist(pars, func_backw, func_dist, func_eqns, stst, vfSS, distSS, zshock, horizon, nvars):
    """Get a function that returns the (flattend) value and Jacobian of the stacked aggregate model equations.
    """

    nshpe = (nvars, horizon-1)

    def backwards_step(carry, i):

        vf, X = carry
        vf, decisions_output, exog_grid_vars = func_backw(
            X[:, i], X[:, i+1], X[:, i+2], stst, vf, [], pars)

        return (vf, X), decisions_output

    def backwards_sweep(x, x0, xT):

        X = jnp.hstack((x0, x, xT)).reshape(horizon+1, -1).T

        _, decisions_output_storage = jax.lax.scan(
            backwards_step, (vfSS, X), jnp.arange(horizon-1), reverse=True)
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

    def final_step(x, dists_storage, decisions_output_storage, x0, xT):

        X = jnp.hstack((x0, x, xT)).reshape(horizon+1, -1).T
        out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
            nshpe), stst, zshock, pars, dists_storage, decisions_output_storage)

        return out

    def second_sweep(x, decisions_output_storage, x0, xT):

        # forwards step
        dists_storage = forwards_sweep(decisions_output_storage)
        # final step
        out = final_step(x, dists_storage, decisions_output_storage, x0, xT)

        return out

    def stacked_func_dist(x, x0, xT, full_output=False):

        # backwards step
        decisions_output_storage = backwards_sweep(x, x0, xT)
        # combined step
        out = second_sweep(x, decisions_output_storage, x0, xT)

        return out

    return stacked_func_dist, backwards_sweep, second_sweep


def compile_functions(model, zshock, horizon, nvars, pars, stst, xstst):

    print('starting')
    ts = time.time()

    # get functions
    func_eqns = model['context']["func_eqns"]
    func_backw = model['context'].get('func_backw')
    func_dist = model['context'].get('func_dist')

    jac_eqns = jax.jacrev(func_eqns, argnums=(0, 1, 2))

    if model.get('distributions'):
        # get stuff for het-agent models
        vfSS = model['steady_state'].get('decisions')
        distSS = jnp.array(model['steady_state']['distributions'])[..., None]
        decisions_outputSS = jnp.array(
            model['steady_state']['decisions_output'])[..., None]
    else:
        distSS = []
        decisions_outputSS = []

    if model.get('distributions'):
        func_raw, backwards_sweep, second_sweep = get_stacked_func_dist(
            pars, func_backw, func_dist, func_eqns, stst, vfSS, distSS[..., 0], zshock, horizon, nvars)

        # should be a nicer way. Maybe I can even rewrite func to use non-flattend input
        basis = jnp.zeros((nvars*(horizon-1), nvars))
        basis = basis.at[-nvars:, -nvars:].set(jnp.eye(nvars))

        doSS, do2x = jvp_vmap(backwards_sweep)(
            (xstst[1:-1].flatten(), stst, stst), (basis,))
        _, (f2do,) = vjp_vmap(second_sweep, argnums=1)(
            (xstst[1:-1].flatten(), doSS, stst, stst), basis.T)
        f2do_re = jnp.moveaxis(f2do, -1, 1)
    else:
        func_raw = get_stacked_func_rep_agent(
            pars, func_eqns, stst, zshock, horizon, nvars)
        f2do_re = None
        do2x = None

    f2X = jac_eqns(stst[:, None], stst[:, None], stst[:, None],
                   stst, zshock, pars, distSS, decisions_outputSS)

    model['jvp'] = lambda primals, tangens, x0, xT: jax.jvp(
        func_raw, (primals, x0, xT), (tangens, jnp.zeros(nvars), jnp.zeros(nvars)))
    model['func_raw'] = func_raw

    print('derivatives')
    print(-ts + time.time())

    return f2X, f2do_re, do2x
