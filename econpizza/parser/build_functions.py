
"""Build dynamic functions."""

import jax
import jax.numpy as jnp
from scipy import sparse


def get_func_stst_raw(func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, exog_grid_vars_init, tol_backw, maxit_backw, tol_forw, maxit_forw):

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
        x = x[..., jnp.newaxis]

        if not func_stst_dist:
            return func_eqns(x, x, x, x, jnp.zeros(len(shocks)), par)

        vf, decisions_output, exog_grid_vars, cnt_backw = func_backw_ext(
            x, par)
        dist, cnt_forw = func_stst_dist(decisions_output, tol_forw, maxit_forw)

        # TODO: for more than one dist this should be a loop...
        decisions_output_array = decisions_output[..., jnp.newaxis]
        dist_array = dist[..., jnp.newaxis]

        if full_output:
            return (vf, decisions_output, exog_grid_vars, cnt_backw), (dist, cnt_forw)

        return func_eqns(x, x, x, x, [], par, dist_array, decisions_output_array)

    return func_stst_raw


def get_stacked_func(pars, func_backw, func_dist, func_eqns, x0, stst, vfSS, distSS, zshock, tshock, horizon, nvars, endpoint, has_distributions, shock):

    nshpe = (nvars, horizon-1)

    def backwards_step(carry, i):

        vf_old, X = carry
        vf, decisions_output, exog_grid_vars = func_backw(
            X[:, i], X[:, i+1], X[:, i+2], stst, vf_old, [], pars)

        return (vf, X), decisions_output

    def forwards_step(carry, i):

        dist_old, decisions_output_storage = carry
        dist = func_dist(dist_old, decisions_output_storage[..., i])

        return (dist, decisions_output_storage), dist
    from grgrlib.jaxed import jax_print

    def stacked_func(x, full_output=False):

        X = jnp.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint)).T

        if has_distributions:
            # backwards step
            _, decisions_output_storage = jax.lax.scan(
                backwards_step, (vfSS, X), jnp.arange(horizon-2, -1, -1))
            decisions_output_storage = jnp.flip(decisions_output_storage, 0)
            decisions_output_storage = jnp.moveaxis(
                decisions_output_storage, 0, -1)
            # forwards step
            _, dists_storage = jax.lax.scan(
                forwards_step, (distSS, decisions_output_storage), jnp.arange(horizon-1))
            dists_storage = jnp.vstack((distSS[jnp.newaxis], dists_storage))
            dists_storage = jnp.moveaxis(dists_storage, 0, -1)
        else:
            decisions_output_storage, dists_storage = [], []

        if full_output:
            return decisions_output_storage, dists_storage

        elif has_distributions:
            # important: use the time-{t-1} distribution in time t. This is correct because the output from EGM corresponds to the time-(t-1) distribution
            dists_storage = dists_storage[..., :-1]

        out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
            nshpe), stst, zshock, pars, dists_storage, decisions_output_storage)

        if shock is not None:
            out = out.at[jnp.arange(nvars)*(horizon-1)].set(
                func_eqns(X[:, 0], X[:, 1], X[:, 2], stst, tshock, pars))

        return out

    return stacked_func


def get_jac(pars, func_eqns, stst, x0, horizon, nvars, endpoint, zshock, tshock, shock):

    jac_vmap = jax.vmap(jax.jacfwd(lambda x: func_eqns(
        x[:nvars], x[nvars:-nvars], x[-nvars:], stst, zshock, pars)))
    jac_shock = jax.jacfwd(lambda x: func_eqns(
        x[:nvars], x[nvars:-nvars], x[-nvars:], stst, tshock, pars))
    hrange = jnp.arange(nvars)*(horizon-1)

    # the ordering is ((equation1(t=1,...,T), ..., equationsN(t=1,...,T)) x (period1(var=1,...,N), ..., periodT(var=1,...,N)))
    # TODO: an ordering ((equation1(t=1,...,T), ..., equationsN(t=1,...,T)) x (variable1(t=1,...,T), ..., variableN(t=1,...,T))) would actually be clearer
    # this is simply done by adjusting the way the funcition output is flattened
    # TODO: also, this function can be sourced out
    def jac(x):

        X = jnp.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint))
        Y = jnp.hstack((X[:-2], X[1:-1], X[2:]))
        jac_parts = jac_vmap(Y)

        J = sparse.lil_array(((horizon-1)*nvars, (horizon-1)*nvars))
        if shock is None:
            J[jnp.arange(nvars)*(horizon-1), :nvars *
              2] = jac_parts[0, :, nvars:]
        else:
            J[jnp.arange(nvars)*(horizon-1), :nvars *
              2] = jac_shock(X[:3].flatten())[:, nvars:]
        J[jnp.arange(nvars)*(horizon-1)+horizon-2, (horizon-3) *
          nvars:horizon*nvars] = jac_parts[horizon-2, :, :-nvars]

        for t in range(1, horizon-2):
            J[hrange+t, (t-1)*nvars:(t-1+3)*nvars] = jac_parts[t]

        return sparse.csr_array(J)

    return jac
