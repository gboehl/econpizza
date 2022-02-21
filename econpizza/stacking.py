#!/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import scipy.optimize as so
from .shooting import find_path_linear
import jax
from jaxopt import ScipyRootFinding


def find_stack(
    model,
    x0=None,
    shock=None,
    init_path=None,
    horizon=50,
    tol=None,
    use_linear_guess=True,
    use_linear_endpoint=None,
    root_options={},
    verbose=True,
):

    st = time.time()

    stst = np.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    func = model["func"]
    pars = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    model["root_options"] = root_options

    if tol is None:
        tol = 1e-8

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

    if shock is None:
        @jax.jit
        def stacked_func(x):

            X = jax.numpy.vstack(
                (x0, x.reshape((horizon - 1, nvars)), endpoint))
            out = func(X[:-2].T, X[1:-1].T, X[2:].T,
                       stst, zshock, pars).flatten()

            return out
    else:
        @jax.jit
        def stacked_func(x):

            X = jax.numpy.vstack((x.reshape((horizon - 1, nvars)), endpoint))
            out_1st = func(x0, X[0], X[1], stst, tshock, pars)
            out_rst = func(X[:-2].T, X[1:-1].T, X[2:].T,
                           stst, zshock, pars).flatten()

            return jax.numpy.hstack((out_1st, out_rst))

    sproot = ScipyRootFinding(
        optimality_fun=stacked_func,
        method="hybr",
        use_jacrev=False,
        options=root_options,
    )

    res = sproot.run(x_init[1:-1].flatten())

    err = np.abs(res[1][0]).max()
    x[1:-1] = res[0].reshape((horizon - 1, nvars))

    mess = ''
    if err > tol:
        mess += "Max error is %1.2e." % np.abs(stacked_func(res[0])).max()

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("(find_path_stacked:) Stacking done after %s seconds (size: %.2e). " % (
            duration, horizon*nvars) + mess)

    return x, x_lin, not res[1][1]
