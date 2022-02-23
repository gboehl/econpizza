#!/bin/python
# -*- coding: utf-8 -*-

import os
import jax
import time
import numpy as np
from scipy import sparse
from grgrlib.jaxed import newton_jax
from .shooting import find_path_linear


def find_stack(
    model,
    x0=None,
    shock=None,
    init_path=None,
    horizon=50,
    tol=None,
    maxit=None,
    use_linear_guess=True,
    use_linear_endpoint=None,
    parallel=False,
    verbose=True,
):

    st = time.time()

    stst = np.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    func = model["func"]
    pars = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    if tol is None:
        tol = 1e-8
    if maxit is None:
        maxit = 30

    ncores = os.cpu_count()
    if parallel and (horizon - 1) % ncores:
        horizon += ncores - (horizon - 1) % ncores

    x0 = np.array(list(x0)) if x0 is not None else stst
    x = np.ones((horizon + 1, nvars)) * stst
    x[0] = x0

    x_init, x_lin = find_path_linear(
        model, shock, horizon, x, use_linear_guess)

    if use_linear_endpoint is None:
        use_linear_endpoint = False if x_lin is None else True
    elif use_linear_endpoint and x_lin is None:
        print("(find_path_stacked:) Linear solution for the endpoint not available")
        use_linear_endpoint = False

    if init_path is not None:
        x_init[1: len(init_path)] = init_path[1:]

    zshock = np.zeros(len(shocks))
    tshock = np.copy(zshock)
    if shock is not None:
        tshock[shocks.index(shock[0])] = shock[1]

    endpoint = x_lin[-1] if use_linear_endpoint else stst

    if parallel:
        pfunc = jax.pmap(lambda x0, x1, x2: func(
            x0, x1, x2, stst, zshock, pars), in_axes=2, out_axes=2)
        nshpe = (nvars, int((horizon-1)/ncores), ncores)
    else:
        def pfunc(x0, x1, x2): return func(x0, x1, x2, stst, zshock, pars)
        nshpe = (nvars, horizon-1)

    def stacked_func(x):

        X = jax.numpy.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint)).T
        out = pfunc(X[:, :-2].reshape(nshpe),
                    X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(nshpe))

        if shock is not None:
            out = out.at[:, 0].set(
                func(X[:, 0], X[:, 1], X[:, 2], stst, tshock, pars))

        return out.flatten()

    if parallel:
        stacked_func = stacked_func
    else:
        stacked_func = jax.jit(stacked_func)

    if verbose:
        print("(find_path_stacked:) Solving stack (size: %s)..." %
              (horizon*nvars))

    jac_vmap = jax.vmap(jax.jacfwd(lambda x: func(
        x[:nvars], x[nvars:-nvars], x[-nvars:], stst, zshock, pars)))
    jac_shock = jax.jacfwd(lambda x: func(
        x[:nvars], x[nvars:-nvars], x[-nvars:], stst, tshock, pars))
    hrange = np.arange(nvars)*(horizon-1)

    def jac(x):

        X = jax.numpy.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint))
        Y = jax.numpy.hstack((X[:-2], X[1:-1], X[2:]))
        jac_parts = jac_vmap(Y)

        J = sparse.lil_array(((horizon-1)*nvars, (horizon-1)*nvars))
        if shock is None:
            J[np.arange(nvars)*(horizon-1), :nvars*2] = jac_parts[0, :, nvars:]
        else:
            J[np.arange(nvars)*(horizon-1), :nvars *
              2] = jac_shock(X[:3].flatten())[:, nvars:]
        J[np.arange(nvars)*(horizon-1)+horizon-2, (horizon-3) *
          nvars:horizon*nvars] = jac_parts[horizon-2, :, :-nvars]

        for t in range(1, horizon-2):
            J[hrange+t, (t-1)*nvars:(t-1+3)*nvars] = jac_parts[t]

        return sparse.csc_matrix(J)

    res = newton_jax(
        stacked_func, x_init[1:-1].flatten(), jac, maxit, tol, True, verbose)

    err = np.abs(res['fun']).max()
    x[1:-1] = res['x'].reshape((horizon - 1, nvars))

    mess = res['message']
    if err > tol:
        mess += " Max error is %1.2e." % np.abs(stacked_func(res['x'])).max()

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("(find_path_stacked:) Stacking done after %s seconds. " %
              duration + mess)

    return x, x_lin, not res['success']
