#!/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import scipy.optimize as so
from numba import njit, prange
from .shooting import find_path_linear

# experimental
try:
    import jax.numpy as jnp

    def stacked_func_jax(
        x, x0, endpoint, func, horizon, nvars, stst, tshock, zshock, pars
    ):

        # preallocation is bullshit with jax, so lets fix this soonish
        out = jnp.empty((horizon - 1) * nvars)
        X = x.reshape((horizon - 1, nvars))

        out = out.at[:nvars].set(func(x0, X[0], X[1], stst, tshock, pars))
        out = out.at[-nvars:].set(func(X[-2], X[-1], endpoint, stst, zshock, pars))

        # this should be vectorized instead
        for t in range(1, horizon - 2):
            out = out.at[t * nvars : (t + 1) * nvars].set(
                func(X[t - 1], X[t], X[t + 1], stst, zshock, pars)
            )

        return out

except:
    pass


def stacked_func_plain(
    x, x0, endpoint, func, horizon, nvars, stst, tshock, zshock, pars
):

    out = np.empty((horizon - 1) * nvars)
    X = x.reshape((horizon - 1, nvars))

    out[:nvars] = func(x0, X[0], X[1], stst, tshock, pars)
    out[-nvars:] = func(X[-2], X[-1], endpoint, stst, zshock, pars)

    for t in prange(1, horizon - 2):
        out[t * nvars : (t + 1) * nvars] = func(
            X[t - 1], X[t], X[t + 1], stst, zshock, pars
        )

    return out


stacked_func_njit = njit(stacked_func_plain)
stacked_func_njit_parallel = njit(stacked_func_plain, parallel=True)


def find_stack(
    model,
    x0=None,
    shock=None,
    init_path=None,
    horizon=50,
    tol=None,
    use_numba=False,
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

    if root_options:
        model["root_options"] = root_options

    if "xtol" not in model["root_options"]:
        if tol is None:
            tol = 1e-5
        elif "xtol" in root_options:
            print(
                "(find_path:) Specification of xtol in `root_options` overwrites `tol`"
            )
        model["root_options"]["xtol"] = tol

    x0 = np.array(list(x0)) if x0 is not None else stst
    x = np.ones((horizon + 1, nvars)) * stst
    x[0] = x0

    x_init, x_lin = find_path_linear(model, shock, horizon, x, use_linear_guess)

    if use_linear_endpoint is None:
        use_linear_endpoint = False if x_lin is None else True
    elif use_linear_endpoint and x_lin is None:
        print("(find_path_stacked:) Linear solution for the endpoint not available")
        use_linear_endpoint = False

    if init_path is not None:
        x_init[1 : len(init_path)] = init_path[1:]

    zshock = np.zeros(len(shocks))
    tshock = np.copy(zshock)
    if shock is not None:
        tshock[shocks.index(shock[0])] = shock[1]

    endpoint = x_lin[-1] if use_linear_endpoint else stst
    out = np.empty((horizon - 1, nvars))

    if use_numba in ("p", "parallel"):
        stacked_func = lambda x: stacked_func_njit_parallel(
            x, x0, endpoint, func, horizon, nvars, stst, tshock, zshock, pars
        )
    elif use_numba:
        stacked_func = lambda x: stacked_func_njit(
            x, x0, endpoint, func, horizon, nvars, stst, tshock, zshock, pars
        )
    else:
        stacked_func = lambda x: stacked_func_plain(
            x, x0, endpoint, func, horizon, nvars, stst, tshock, zshock, pars
        )

    # experimental!
    if model["use_jax"]:
        from jax import jacfwd

        stacked_func = lambda x: stacked_func_jax(
            x, x0, endpoint, func, horizon, nvars, stst, tshock, zshock, pars
        )

        jacobian = jacfwd(stacked_func)
        res = so.root(stacked_func, x_init[1:-1].flatten(), jac=jacobian)
        res_x, res_fun, res_succ, res_mess = (
            res["x"],
            res["fun"],
            res["success"],
            res["message"],
        )

        # TODO: to use jaxopt. So far this is VERY slow
        # from jaxopt import ScipyRootFinding
        # rf = ScipyRootFinding(optimality_fun=stacked_func, method='hybr', use_jacrev=False)

        # res = rf.run(x_init[1:-1].flatten())
        # res_x, res_fun, res_succ, res_mess = res[0], res[1][0], res[1][1], '<no message>'

    else:
        res = so.root(stacked_func, x_init[1:-1].flatten())
        res_x, res_fun, res_succ, res_mess = (
            res["x"],
            res["fun"],
            res["success"],
            res["message"],
        )

    err = np.abs(res_fun).max()
    x[1:-1] = res_x.reshape((horizon - 1, nvars))

    mess = " ".join(res_mess.replace("\n", " ").split())
    if err > tol:
        mess += " Max error is %1.2e." % np.abs(stacked_func(res["x"])).max()

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("(find_path_stacked:) Stacking done after %s seconds. " % duration + mess)

    return x, x_lin, not res_succ
