#!/bin/python
# -*- coding: utf-8 -*-

import os
import jax
import time
import jax.numpy as jnp
from scipy import sparse
from grgrlib.jaxed import newton_jax
from .shooting import find_path_linear


def find_stack(
    model,
    x0=None,
    shock=None,
    init_path=None,
    horizon=250,
    tol=None,
    maxit=None,
    use_linear_guess=True,
    use_linear_endpoint=None,
    verbose=True,
):

    st = time.time()

    stst = jnp.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    pars = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()
    func_eqns = model['context']["func_eqns"]

    if tol is None:
        tol = 1e-8
    if maxit is None:
        maxit = 30

    x0 = jnp.array(list(x0)) if x0 is not None else stst
    x = jnp.ones((horizon + 1, nvars)) * stst
    x = x.at[0].set(x0)

    x_init, x_lin = find_path_linear(
        model, shock, horizon, x, use_linear_guess)

    if use_linear_endpoint is None:
        use_linear_endpoint = False if x_lin is None else True
    elif use_linear_endpoint and x_lin is None:
        print("(find_path_stacked:) Linear solution for the endpoint not available")
        use_linear_endpoint = False

    if init_path is not None:
        x_init[1: len(init_path)] = init_path[1:]

    zshock = jnp.zeros(len(shocks))
    tshock = jnp.copy(zshock)
    if shock is not None:
        tshock = tshock.at[shocks.index(shock[0])].set(shock[1])
        if model.get('distributions'):
            print("(find_path_stacked:) Warning: shocks for heterogenous agent models are not yet fully supported.")

    endpoint = x_lin[-1] if use_linear_endpoint else stst

    if model.get('distributions'):
        vfSS = model['decisions']['stst']
        distSS = jnp.array(model['distributions']['stst'])

        # define shapes
        decisions_output_shape = jnp.shape(
            model['init_run']['decisions_output'])
        dist_shape = jnp.shape(distSS)

        # load functions
        func_backw = model['context'].get('func_backw')
        func_dist = model['context'].get('func_dist')

    nshpe = (nvars, horizon-1)

    def backwards_step(i, cont):

        i = horizon-2-i  # reversed
        decisions_output_storage, vf_old, X = cont
        vf, decisions_output = func_backw(
            X[:, i], X[:, i+1], X[:, i+2], stst, vf_old, [], pars)
        decisions_output_storage = decisions_output_storage.at[..., i].set(
            decisions_output)

        return decisions_output_storage, vf, X

    def forwards_step(i, cont):

        dists_storage, dist_old, decisions_output_storage = cont
        dist = func_dist(dist_old, decisions_output_storage[..., i])
        dists_storage = dists_storage.at[..., i].set(dist)

        return dists_storage, jnp.array(dist), decisions_output_storage

    def stacked_func(x):

        X = jax.numpy.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint)).T

        if model.get('distributions'):
            decisions_output_storage = jnp.empty(
                (*decisions_output_shape, horizon-1))  # last storage is the stst
            dists_storage = jnp.empty((*dist_shape, horizon-1))

            # backwards step
            decisions_output_storage, _, _ = jax.lax.fori_loop(
                0, horizon-1, backwards_step, (decisions_output_storage, vfSS, X))
            # forwards step
            dists_storage, _, _ = jax.lax.fori_loop(
                0, horizon-1, forwards_step, (dists_storage, distSS, decisions_output_storage))
        else:
            decisions_output_storage, dists_storage = [], []

        out = func_eqns(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
            nshpe), stst, zshock, pars, dists_storage, decisions_output_storage)

        if shock is not None:
            out = out.at[jnp.arange(nvars)*(horizon-1)].set(
                func_eqns(X[:, 0], X[:, 1], X[:, 2], stst, tshock, pars))

        return out

    stacked_func = jax.jit(stacked_func)

    if verbose:
        print("(find_path_stacked:) Solving stack (size: %s)..." %
              (horizon*nvars))

    jac_vmap = jax.vmap(jax.jacfwd(lambda x: func_eqns(
        x[:nvars], x[nvars:-nvars], x[-nvars:], stst, zshock, pars)))
    jac_shock = jax.jacfwd(lambda x: func_eqns(
        x[:nvars], x[nvars:-nvars], x[-nvars:], stst, tshock, pars))
    hrange = jnp.arange(nvars)*(horizon-1)

    # the ordering is ((equation1(t=1,...,T), ..., equationsN(t=1,...,T)) x (period1(var=1,...,N), ..., periodT(var=1,...,N)))
    # TODO: an ordering ((equation1(t=1,...,T), ..., equationsN(t=1,...,T)) x (variable1(t=1,...,T), ..., variableN(t=1,...,T))) would actually be clearer
    # this is simply done by adjusting the way the funcition output is flattened
    # TODO: also, this function can be sourced out
    def jac_func(x):

        X = jax.numpy.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint))
        Y = jax.numpy.hstack((X[:-2], X[1:-1], X[2:]))
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

        return sparse.csc_matrix(J)

    jac = None if model.get('distributions') else jac_func
    res = newton_jax(
        stacked_func, x_init[1:-1].flatten(), jac, maxit, tol, True, verbose=verbose)

    err = jnp.abs(res['fun']).max()
    x = x.at[1:-1].set(res['x'].reshape((horizon - 1, nvars)))

    mess = res['message']
    if err > tol:
        mess += " Max error is %1.2e." % jnp.abs(stacked_func(res['x'])).max()

    if verbose:
        duration = time.time() - st
        print(
            f"(find_path_stacked:) Stacking done after {duration:1.3f} seconds. " + mess)

    return x, x_lin, not res['success']
