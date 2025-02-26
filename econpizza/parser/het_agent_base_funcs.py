"""Internal subfunctions for heterogeneous agent models
"""

import jax
import jax.numpy as jnp
from jax._src.typing import Array
from typing import Callable


def _backwards_stst_cond(carry):
    _, (wf, _), (wf_old, cnt), (_, tol, maxit) = carry
    cond0 = jnp.abs(wf - wf_old).max() > tol
    cond1 = cnt < maxit
    return jnp.logical_and(cond0, cond1)


def _backwards_stst_body(carry):
    (x, par), (wf, _), (_, cnt), (func, tol, maxit) = carry
    return (x, par), func(x, x, x, x, pars=par, WFPrime=wf), (wf, cnt + 1), (func, tol, maxit)


def backwards_sweep_stst(x, par, carry):
    _, (wf, decisions_output), (_, cnt), _ = jax.lax.while_loop(
        _backwards_stst_cond, _backwards_stst_body, ((x, par), *carry))
    return wf, decisions_output, cnt


def _backwards_step(carry, i):

    wf, X, shocks, func_backw, stst, pars = carry
    wf, decisions_output = func_backw(
        X[:, i], X[:, i+1], X[:, i+2], pars=pars, WFPrime=wf, shocks=shocks[:, i])

    return (wf, X, shocks, func_backw, stst, pars), (wf, decisions_output)


def backwards_sweep(x: Array, x0: Array, shocks: Array, pars: Array, stst: Array, wfSS: Array, func_backw: Callable, return_wf=False) -> Array:

    # get horizon from input size
    horizon = len(x)//len(stst) + 1
    X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T

    _, (wf_storage, decisions_output_storage) = jax.lax.scan(
        _backwards_step, (wfSS, X, shocks, func_backw, stst, pars), jnp.arange(horizon-1), reverse=True)
    decisions_output_storage = [jnp.moveaxis(
        dos, 0, -1) for dos in decisions_output_storage]
    wf_storage = jnp.moveaxis(wf_storage, 0, -1)

    if return_wf:
        return wf_storage, decisions_output_storage
    return decisions_output_storage


def _forwards_step(carry, i):

    dist_old, decisions_output_storage, func_forw = carry
    dist = func_forw(dist_old, [dos[..., i]
                     for dos in decisions_output_storage])

    return (dist, decisions_output_storage, func_forw), dist_old


def forwards_sweep(decisions_output_storage: Array, dist0: Array, horizon: int, func_forw: callable) -> Array:

    _, dists_storage = jax.lax.scan(
        _forwards_step, (dist0, decisions_output_storage, func_forw), jnp.arange(horizon-1))
    dists_storage = jnp.moveaxis(dists_storage, 0, -1)

    return dists_storage


def final_step(x: Array, dists_storage: Array, decisions_output_storage: Array, x0: Array, shocks: Array, pars: Array, stst: Array, horizon: int, nshpe, func_eqns: Callable) -> Array:

    X = jnp.hstack((x0, x, stst)).reshape(horizon+1, -1).T
    out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
        nshpe), stst, pars, shocks, dists_storage, decisions_output_storage)

    return out


def combined_sweep(x: Array, decisions_output_storage: Array, x0: Array, dist0: Array, shocks: Array, pars: Array, forwards_sweep: Callable, final_step: Callable) -> Array:

    # forwards step
    dists_storage = forwards_sweep(decisions_output_storage, dist0)
    # final step
    out = final_step(x, dists_storage,
                     decisions_output_storage, x0, shocks, pars)

    return out


def stacked_func_het_agents(x: Array, x0: Array, dist0: Array, shocks: Array, pars: Array, backwards_sweep: Callable, combined_sweep: Callable):

    # backwards step
    decisions_output_storage = backwards_sweep(x, x0, shocks, pars)
    # combined step
    out = combined_sweep(x, decisions_output_storage, x0, dist0, shocks, pars)

    return out
