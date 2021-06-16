#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy.optimize as so


def solve_current(model, XLag, XPrime, tol):

    func = model["func"]
    par = model["parameters"]
    stst = model.get("stst")
    shocks = model.get("shocks") or ()

    func_current = lambda x: func(
        XLag,
        x,
        XPrime,
        np.array(list(stst.values())),
        np.zeros(len(shocks)),
        np.array(list(par.values())),
    )

    res = so.root(func_current, XPrime, options=model["root_options"])
    err = np.max(np.abs(func_current(res["x"])))

    return res["x"], not res["success"], err > tol


def find_path(
    model,
    x0,
    T=30,
    init_path=None,
    max_horizon=200,
    max_loops=100,
    max_iter=None,
    tol=1e-5,
    reverse=False,
    root_options=None,
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
        number of periods until the system is assumed to be back in the steady state. A good idea to set this corresponding to the respective problem
    max_loops : int, optional
        number of repetitions to iterate over the whole trajectory. Should eventually be high.
    max_iterations : int, optional
        number of iterations. Default is `max_horizon`. It should not be lower than that (and will raise an error). Normally it should not be higher, better use `max_loops` instead.
    tol : float, optional
        convergence criterion
    reverse : bool, optional
        whether to start each iteration with the values most far in the future. Normally not a good idea.
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

    stst = list(model["stst"].values())
    evars = model["variables"]

    if root_options:
        model["root_options"] = root_options

    # precision of root finding should be some magnitudes higher than of solver
    if "xtol" not in model["root_options"]:
        model["root_options"]["xtol"] = max(tol * 1e-3, 1e-8)

    x_fin = np.empty((T + 1, len(evars)))
    x_fin[0] = list(x0)

    x = np.ones((T + max_horizon + 1, len(evars))) * np.array(stst)
    x[0] = list(x0)

    if init_path is not None:
        x[1 : len(init_path)] = init_path[1:]

    fin_flag = np.zeros(5, dtype=bool)

    try:
        for i in range(T):

            loop = 1
            cnt = 2 - reverse
            old_clock = time.time()
            flag = np.zeros(5, dtype=bool)

            while True:

                x_old = x[1].copy()
                imax = min(cnt, max_horizon)

                for t in range(imax):

                    if reverse:
                        t = imax - t - 1

                    x[t + 1], flag_root, flag_ftol = solve_current(
                        model, x[t], x[t + 2], tol
                    )

                flag[0] |= cnt == flag_root
                flag[1] |= np.any(np.isnan(x))
                flag[2] |= np.any(np.isinf(x))
                flag[3] |= cnt == flag_ftol

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
                        "Period{:>4d} | loop{:>5d} | iter.{:>5d} | error: {:>1.8e}".format(
                            i, loop, cnt, err
                        )
                    )

                if (err < tol and cnt > 2) or flag.any():
                    break

                cnt += 1

            x_fin[i + 1] = x[1].copy()
            x = x[1:]

    except Exception as error:
        raise type(error)(
            "The following error was raised in t=%s at iteration no. %s for forecast %s steps ahead:\n\n"
            % (i, cnt, t)
            + str(error)
        )

    msgs = (
        ", non-convergence in root finding",
        ", contains NaNs",
        ", contains infs",
        ", ftol not reached in root finding",
        ", max_iter reached",
    )
    mess = [i * bool(j) for i, j in zip(msgs, fin_flag)]
    fin_flag = 2 ** np.arange(5) @ fin_flag

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("Pizza done after %s seconds%s." % (duration, "".join(mess)))

    return x_fin, fin_flag
