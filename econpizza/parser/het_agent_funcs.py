"""Subfunctions of stacked_func_dist
"""

import jax
import jax.numpy as jnp


def backwards_step(carry, i):

    vf, X, shocks, func_backw, stst = carry
    vf, decisions_output, exog_grid_vars = func_backw(
        X[:, i], X[:, i+1], X[:, i+2], VFPrime=vf, shocks=shocks[:, i])

    return (vf, X, shocks, func_backw, stst), decisions_output


def _backwards_sweep(x, x0, shocks, stst, vfSS, horizon, func_backw):

    X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T

    _, decisions_output_storage = jax.lax.scan(
        backwards_step, (vfSS, X, shocks, func_backw, stst), jnp.arange(horizon-1), reverse=True)
    decisions_output_storage = jnp.moveaxis(
        decisions_output_storage, 0, -1)

    return decisions_output_storage


def forwards_step(carry, i):

    dist_old, decisions_output_storage, func_dist = carry
    dist = func_dist(dist_old, decisions_output_storage[..., i])

    return (dist, decisions_output_storage, func_dist), dist_old


def _forwards_sweep(decisions_output_storage, distSS, horizon, func_dist):

    _, dists_storage = jax.lax.scan(
        forwards_step, (distSS, decisions_output_storage, func_dist), jnp.arange(horizon-1))
    dists_storage = jnp.moveaxis(dists_storage, 0, -1)

    return dists_storage


def _final_step(x, dists_storage, decisions_output_storage, x0, shocks, stst, horizon, nshpe, pars, func_eqns):

    X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T
    out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
        nshpe), stst, shocks, pars, dists_storage, decisions_output_storage)

    return out


def _second_sweep(x, decisions_output_storage, x0, shocks, forwards_sweep, final_step):

    # forwards step
    dists_storage = forwards_sweep(decisions_output_storage)
    # final step
    out = final_step(x, dists_storage,
                     decisions_output_storage, x0, shocks)

    return out


def _stacked_func_dist(x, x0, shocks, backwards_sweep, second_sweep):

    # backwards step
    decisions_output_storage = backwards_sweep(x, x0, shocks)
    # combined step
    out = second_sweep(x, decisions_output_storage, x0, shocks)

    return out
