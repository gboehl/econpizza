#!/bin/python
# -*- coding: utf-8 -*-

from jax.experimental.host_callback import id_print
from jaxopt import ScipyRootFinding
import jax
import sys
import time
import numpy as np
import scipy.optimize as so
from .shooting import find_path_linear

import os
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
# this must go where we first imprt jax!


def find_stack(
    model,
    x0=None,
    shock=None,
    init_path=None,
    horizon=50,
    xtol=None,
    use_linear_guess=True,
    use_linear_endpoint=None,
    root_options={},
    parallel=False,
    live_dangerously=False,
    verbose=True,
):

    st = time.time()

    stst = np.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    func = model["func"]
    pars = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    model["root_options"] = root_options

    if xtol is not None:
        root_options['xtol'] = xtol

    if not 'xtol' in root_options:
        root_options['xtol'] = 1e-8

    ncores = os.cpu_count()
    # if parallel and (horizon - 1) % ncores:
    if shock is None:
        horizon += ncores - (horizon - 1) % ncores
    else:
        horizon += ncores - (horizon - 2) % ncores

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
            x0, x1, x2, stst, zshock, pars), in_axes=2)
        nshpe = (nvars, int((horizon-1-bool(shock))/ncores), ncores)
    else:
        def pfunc(x0, x1, x2): return func(x0, x1, x2, stst, zshock, pars)
        nshpe = (nvars, horizon-1-bool(shock))

    def stacked_func(x):

        if shock is None:
            X = jax.numpy.vstack(
                (x0, x.reshape((horizon - 1, nvars)), endpoint)).T

            out = pfunc(X[:, :-2].reshape(nshpe), X[:, 1:-
                        1].reshape(nshpe), X[:, 2:].reshape(nshpe)).flatten()
            out = pfunc(X[:, :-2].reshape(nshpe),
                        X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(nshpe))
        else:
            X = jax.numpy.vstack((x.reshape((horizon - 1, nvars)), endpoint)).T

            out_1st = func(x0, X[:, 0], X[:, 1], stst, tshock, pars)
            out_rst = pfunc(X[:, :-2].reshape(nshpe),
                            X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(nshpe))
            out = jax.numpy.hstack((out_1st, out_rst.flatten()))

        # works, but slows down and does not support strings
        if verbose and live_dangerously:
            id_print(jax.numpy.abs(out).max())

        return out.flatten()

    res = so.root(stacked_func, x0=x_init[1:-1].flatten(),
                  jac=jax.jacfwd(stacked_func), options=root_options)

    err = np.abs(res['fun']).max()
    x[1:-1] = res['x'].reshape((horizon - 1, nvars))

    mess = res['message']
    if err > root_options['xtol']:
        mess += " Max error is %1.2e." % np.abs(stacked_func(res['x'])).max()

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("(find_path_stacked:) Stacking done after %s seconds. " %
              duration + mess)

    return x, x_lin, not res['success']
