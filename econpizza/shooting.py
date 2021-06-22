#!/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import scipy.optimize as so
from numba import njit


def solve_current(model, XLag, XPrime, tol):

    func = model["func"]
    pars = np.array(list(model["parameters"].values()))
    stst = np.array(list(model["stst"].values()))
    shocks = np.zeros(len(model.get("shocks") or ()))

    func_current = lambda x: func(XLag, x, XPrime, stst, shocks, pars)

    res = so.root(func_current, XPrime, options=model["root_options"])
    err = np.max(np.abs(func_current(res["x"])))

    return res["x"], not res["success"], err > tol


def find_path_linear(model, T, x, use_linear_guess):

    stst = np.array(list(model["stst"].values()))
    sel = stst.astype(bool)

    if model.get("lin_pol") is not None:
        x_lin = np.empty_like(x)
        x_lin[0][sel] = (x[0] / stst - 1)[sel]
        x_lin[0][~sel] = x[0][~sel]

        for i in range(T):
            x_lin[i + 1] = -model["lin_pol"] @ x_lin[i]

        x_lin[:, sel] = ((1 + x_lin) * stst)[:, sel]

        if use_linear_guess:
            return x_lin.copy(), x_lin
        else:
            return x, x_lin
    else:
        return x, None


def find_path(
    model,
    x0,
    T=30,
    init_path=None,
    max_horizon=200,
    max_loops=100,
    max_iter=None,
    tol=1e-5,
    use_linear_guess=False,
    root_options={},
    verbose=True,
):
    """Find the expected trajectory given an initial state. A good strategy is to first set `tol` to a low value (e.g. 1e-3) and check for a good max_horizon. Then, set max_horizon to a reasonable value and let max_loops be high.

    Parameters
    ----------
    model : dict
        model dict as defined/parsed above
    x0 : array
        initial state
    T : int, optional
        number of periods to simulate
    init_path : array, optional
        a first guess on the trajectory. Normally not necessary
    max_horizon : int, optional
        number of periods until the system is assumed to be back in the steady state. A good idea to set this corresponding to the respective problem. Note that a horizon too far away may cause the accumulation of numerical errors.
    max_loops : int, optional
        number of repetitions to iterate over the whole trajectory. Should eventually be high.
    max_iterations : int, optional
        number of iterations. Default is `max_horizon`. It should not be lower than that (and will raise an error). Normally it should not be higher, better use `max_loops` instead.
    tol : float, optional
        convergence criterion
    root_options : dict, optional
        dictionary with solver-specific options to be passed on to `scipy.optimize.root`
    verbose : bool, optional
        degree of verbosity. 0/`False` is silent

    Returns
    -------
    array
        array of the trajectory
    flag
        integer of error flag
    """

    st = time.time()

    if max_iter is None:
        max_iter = max_horizon
    elif max_iter < max_horizon:
        Exception(
            "max_iter should be higher or equal max_horizon, but is %s and %s."
            % (max_iter, max_horizon)
        )

    stst = np.array(list(model["stst"].values()))
    nvars = len(model["variables"])

    if root_options:
        model["root_options"] = root_options

    # precision of root finding should be some magnitudes higher than of solver
    if "xtol" not in model["root_options"]:
        model["root_options"]["xtol"] = max(tol * 1e-3, 1e-8)

    x_fin = np.empty((T + 1, nvars))
    x_fin[0] = list(x0)

    x = np.ones((T + max_horizon + 1, nvars)) * stst
    x[0] = list(x0)

    x, x_lin = find_path_linear(model, T + max_horizon, x, use_linear_guess)
    x_lin = x_lin[: T + 1] if x_lin is not None else None

    if init_path is not None:
        x[1 : len(init_path)] = init_path[1:]

    fin_flag = np.zeros(5, dtype=bool)
    old_clock = time.time()

    try:
        for i in range(T):

            loop = 1
            cnt = 2
            flag = np.zeros(5, dtype=bool)

            while True:

                x_old = x[1].copy()
                imax = min(cnt, max_horizon)

                for t in range(imax):

                    x[t + 1], flag_root, flag_ftol = solve_current(
                        model, x[t], x[t + 2], tol
                    )

                flag[0] |= flag_root
                flag[1] |= not flag_root and flag_ftol
                flag[2] |= np.any(np.isnan(x))
                flag[3] |= np.any(np.isinf(x))

                if cnt == max_iter:
                    if loop < max_loops:
                        loop += 1
                        cnt = 2
                    else:
                        flag[4] |= True

                fin_flag |= flag
                err = np.abs(x_old - x[1]).max()

                clock = time.time()
                if verbose and clock - old_clock > 0.5:
                    old_clock = clock
                    print(
                        "Period{:>4d} | loop{:>5d} | iter.{:>5d} | flag{:>2d} | error: {:>1.8e}".format(
                            i, loop, cnt, 2 ** np.arange(5) @ fin_flag, err
                        )
                    )

                if (err < tol and cnt > 2) or flag.any():
                    break

                cnt += 1

            x_fin[i + 1] = x[1].copy()
            x = x[1:].copy()

    except Exception as error:
        raise type(error)(
            str(error)
            + " (raised in t=%s at iteration no. %s for forecast %s steps ahead)"
            % (i, cnt, t)
        ).with_traceback(sys.exc_info()[2])

    msgs = (
        ", non-convergence in root finding",
        ", ftol not reached in root finding",
        ", contains NaNs",
        ", contains infs",
        ", max_iter reached",
    )
    mess = [i * bool(j) for i, j in zip(msgs, fin_flag)]
    fin_flag = 2 ** np.arange(5) @ fin_flag

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("Pizza done after %s seconds%s." % (duration, "".join(mess)))

    return x_fin, x_lin, fin_flag


def find_path_stacked(
    model,
    x0,
    init_path=None,
    horizon=50,
    tol=None,
    use_numba=False,
    use_linear_guess=False,
    root_options={},
    verbose=True,
):

    st = time.time()

    stst = np.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    func = model["func"]
    pars = np.array(list(model["parameters"].values()))
    shocks = np.zeros(len(model.get("shocks") or ()))

    if root_options:
        model["root_options"] = root_options

    if "xtol" not in model["root_options"]:
        if tol is None:
            tol = 1e-5
        elif "xtol" in root_options:
            print("Specification of xtol in `root_options` overwrites `tol`")
        model["root_options"]["xtol"] = tol

    x0 = np.array(list(x0))
    x = np.ones((horizon + 1, nvars)) * stst
    x[0] = x0

    x, x_lin = find_path_linear(model, horizon - 1, x, use_linear_guess)

    if init_path is not None:
        x[1 : len(init_path)] = init_path[1:]

    def stacked_func(x):

        X = x.reshape((horizon - 1, nvars))
        out = np.zeros_like(X)

        out[0] = func(x0, X[0], X[1], stst, shocks, pars)
        out[-1] = func(X[-2], X[-1], stst, stst, shocks, pars)

        for t in range(1, horizon - 2):
            out[t] = func(X[t - 1], X[t], X[t + 1], stst, shocks, pars)

        return out.flatten()

    if use_numba:
        stacked_func = njit(stacked_func)

    res = so.root(stacked_func, x[1:-1])

    x[1:-1] = res["x"].reshape((horizon - 1, nvars))

    mess = res["message"]

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("Stacking done after %s seconds. " % duration + mess)

    return x, x_lin, res["success"]
