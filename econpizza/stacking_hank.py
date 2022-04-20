#!/bin/python
# -*- coding: utf-8 -*-

import os
import jax
import time
import numpy as np
from scipy import sparse
from grgrlib.jaxed import newton_jax


def find_stack_hank(
    model,
    x0=None,
    horizon=250,
    tol=None,
    maxit=None,
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

    x0 = np.array(list(x0))
    x = np.ones((horizon + 1, nvars)) * stst
    x[0] = x0

    x_init, x_lin = x.copy(), None

    zshock = np.zeros(len(shocks))
    tshock = np.copy(zshock)

    endpoint = stst

    def pfunc(x0, x1, x2): return func(x0, x1, x2, stst, zshock, pars)
    nshpe = (nvars, horizon-1)

    def stacked_func(x):

        X = jax.numpy.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint)).T
        out = pfunc(X[:, :-2].reshape(nshpe),
                    X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(nshpe))

        return out.flatten()

    stacked_func = jax.jit(stacked_func)

    if verbose:
        print("(find_path_stacked:) Solving stack (size: %s)..." %
              (horizon*nvars))

    res = newton_jax(
        stacked_func, x_init[1:-1].flatten(), None, maxit, tol, True, verbose=verbose)

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
