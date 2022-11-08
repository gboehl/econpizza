
"""Build dynamic functions."""

import jax
import jax.numpy as jnp
from scipy import sparse
from grgrlib.jaxed import jacfwd_and_val, jacrev_and_val


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


def get_stacked_func_dist(pars, func_backw, func_dist, func_eqns, x0, stst, vfSS, distSS, zshock, tshock, horizon, nvars, endpoint, has_distributions, shock):
    """Get a function that returns the (flattend) value and Jacobian of the stacked aggregate model equations.
    """

    nshpe = (nvars, horizon-1)

    def backwards_step(carry, i):

        vf, X = carry
        vf, decisions_output, exog_grid_vars = func_backw(
            X[:, i], X[:, i+1], X[:, i+2], stst, vf, [], pars)

        return (vf, X), decisions_output

    def forwards_step(carry, i):

        dist_old, decisions_output_storage = carry
        dist = func_dist(dist_old, decisions_output_storage[..., i])

        return (dist, decisions_output_storage), dist_old

    def stacked_func_dist(x, full_output=False):

        X = jnp.hstack((x0, x, endpoint)).reshape(horizon+1, -1).T

        if has_distributions:
            # backwards step
            _, decisions_output_storage = jax.lax.scan(
                backwards_step, (vfSS, X), jnp.arange(horizon-1), reverse=True)
            decisions_output_storage = jnp.moveaxis(
                decisions_output_storage, 0, -1)
            # forwards step
            _, dists_storage = jax.lax.scan(
                forwards_step, (distSS, decisions_output_storage), jnp.arange(horizon-1))
            dists_storage = jnp.moveaxis(dists_storage, 0, -1)
        else:
            decisions_output_storage, dists_storage = [], []

        if full_output:
            return decisions_output_storage, dists_storage

        out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
            nshpe), stst, zshock, pars, dists_storage, decisions_output_storage)

        return out

    return stacked_func_dist


def get_stacked_func(pars, func_eqns, stst, x0, horizon, nvars, endpoint, zshock, tshock, shock, dist_dummy=[], decisions_dummy=[]):
    """Get a function that returns the (flattend) value and Jacobian of the stacked aggregate model equations IGNORING the effects of distributions.
    """

    jac_vmap = jax.vmap(jacfwd_and_val(lambda x: func_eqns(
        x[:nvars, None], x[nvars:-nvars, None], x[-nvars:, None], stst, zshock, pars, dist_dummy, decisions_dummy)))
    jac_shock = jacfwd_and_val(lambda x: func_eqns(
        x[:nvars], x[nvars:-nvars], x[-nvars:], stst, tshock, pars, dist_dummy, decisions_dummy))

    def stacked_func(x):

        X = jnp.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint))
        Y = jnp.hstack((X[:-2], X[1:-1], X[2:]))
        out, jac_parts = jac_vmap(Y)

        # the ordering is ((period1(f=1,...,N), ..., periodT(f=1,...,N)) x (period1(var=1,...,N), ..., periodT(var=1,...,N)))
        J = sparse.lil_array(((horizon-1)*nvars, (horizon-1)*nvars))

        if shock is None:
            J[:nvars, :nvars * 2] = jac_parts[0, :, nvars:]
        else:
            out_shocked, jac_part_shocked = jac_shock(X[:3].ravel())
            J[:nvars, :nvars * 2] = jac_part_shocked[:, nvars:]
            out = out.at[0].set(out_shocked)
        J[-nvars:, (horizon-3) * nvars:horizon *
          nvars] = jac_parts[horizon-2, :, :-nvars]

        for t in range(1, horizon-2):
            J[nvars*t:nvars*(t+1), (t-1)*nvars:(t+2)*nvars] = jac_parts[t]

        return out.ravel(), J

    return stacked_func


def get_combined_funcs(stacked_func_dist, stacked_func, mask_out, use_jacrev):
    """Combine Jacobians and values from stacked_func and stacked_func_dist
    """

    if use_jacrev:
        jav_stacked_func_dist = jacrev_and_val(
            jax.jit(lambda x: stacked_func_dist(x)[mask_out]))
    else:
        jav_stacked_func_dist = jacfwd_and_val(
            jax.jit(lambda x: stacked_func_dist(x)[mask_out]))

    def combined_funcs(x):

        fval, jacval = stacked_func(x)
        fval_dists, jacval_dists = jav_stacked_func_dist(x)
        fval = fval.at[mask_out].set(fval_dists)
        jacval[mask_out] = jacval_dists

        return fval, jacval

    return combined_funcs
