#!/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import scipy.optimize as so
from numba import njit, prange


def solve_current(model, shock, XLag, XLastGuess, XPrime, tol):

    func = model["func"]
    pars = np.array(list(model["parameters"].values()))
    stst = np.array(list(model["stst"].values()))

    func_current = lambda x: func(XLag, x, XPrime, stst, shock, pars)

    res = so.root(func_current, XLastGuess, options=model["root_options"])
    err = np.max(np.abs(func_current(res["x"])))

    return res["x"], not res["success"], err > tol


def find_path_linear(model, shock, T, x, use_linear_guess):

    if model.get("lin_pol") is not None:

        stst = np.array(list(model["stst"].values()))
        sel = np.isclose(stst, 0)

        shocks = model.get("shocks") or ()
        tshock = np.zeros(len(shocks))
        if shock is not None:
            tshock[shocks.index(shock[0])] = shock[1]

        x_lin = np.empty_like(x)
        x_lin[0][~sel] = (x[0] / stst - 1)[~sel]
        x_lin[0][sel] = 0

        for t in range(T):
            x_lin[t + 1] = model["lin_pol"][0] @ x_lin[t]

            if not t:
                x_lin[t + 1] += model["lin_pol"][1] @ tshock

        x_lin[:, ~sel] = ((1 + x_lin) * stst)[:, ~sel]

        if use_linear_guess:
            return x_lin.copy(), x_lin
        else:
            return x, x_lin
    else:
        return x, None


def find_pizza(
    model,
    x0=None,
    shock=None,
    T=30,
    init_path=None,
    max_horizon=200,
    max_loops=100,
    max_iter=None,
    tol=1e-5,
    use_linear_guess=False,
    root_options={},
    raise_error=False,
    verbose=True,
):
    """Find the expected trajectory given an initial state. A good strategy is to first set `tol` to a low value (e.g. 1e-3) and check for a good max_horizon. Then, set max_horizon to a reasonable value and let max_loops be high.

    Parameters
    ----------
    model : dict
        model dict as defined/parsed above
    x0 : array
        initial state
    shock : tuple
        shock in period 0 as in `(shock_name_as_str, shock_size)`
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
    shocks = model.get("shocks") or ()

    if root_options:
        model["root_options"] = root_options

    # precision of root finding should be some magnitudes higher than of solver
    if "xtol" not in model["root_options"]:
        model["root_options"]["xtol"] = min(tol / max_horizon, 1e-8)

    x_fin = np.empty((T + 1, nvars))
    x_fin[0] = list(x0) if x0 is not None else stst

    x = np.ones((T + max_horizon + 1, nvars)) * stst
    x[0] = x_fin[0]

    x, x_lin = find_path_linear(model, shock, T + max_horizon, x, use_linear_guess)
    x_lin = x_lin[: T + 1] if x_lin is not None else None

    if init_path is not None:
        x[1 : len(init_path)] = init_path[1:]

    tshock = np.zeros(len(shocks))

    fin_flag = np.zeros(5, dtype=bool)
    old_clock = time.time()

    msgs = (
        ", root finding did not converge",
        ", ftol not reached in root finding",
        ", contains NaNs",
        ", contains infs",
        ", max_iter reached",
    )

    try:
        for i in range(T):

            loop = 1
            cnt = 2
            flag = np.zeros_like(fin_flag)

            while True:

                x_old = x[1].copy()
                imax = min(cnt, max_horizon)

                flag_loc = np.zeros(2, dtype=bool)

                for t in range(imax):

                    if not t and not i and shock is not None:
                        tshock[shocks.index(shock[0])] = shock[1]
                    else:
                        tshock[:] = 0

                    x[t + 1], flag_root, flag_ftol = solve_current(
                        model, tshock, x[t], x[t + 1], x[t + 2], tol
                    )

                    flag_loc[0] |= flag_root
                    flag_loc[1] |= flag_ftol and not flag_root

                flag[2] |= np.any(np.isnan(x))
                flag[3] |= np.any(np.isinf(x))

                if cnt == max_iter:
                    if loop < max_loops:
                        loop += 1
                        cnt = 2
                    else:
                        flag[4] |= True

                err = np.abs(x_old - x[1]).max()

                clock = time.time()
                if verbose and clock - old_clock > 0.5:
                    old_clock = clock
                    print(
                        "   Period{:>4d} | loop{:>5d} | iter.{:>5d} | flag{:>3d} | error: {:>1.8e}".format(
                            i, loop, cnt, 2 ** np.arange(5) @ fin_flag, err
                        )
                    )

                if (err < tol and cnt > 2) or flag.any():
                    flag[:2] |= flag_loc
                    if raise_error and flag.any():
                        mess = [i * bool(j) for i, j in zip(msgs, flag)]
                        raise Exception("Aborting%s" % "".join(mess))
                    fin_flag |= flag
                    break

                cnt += 1

            x_fin[i + 1] = x[1].copy()
            x = x[1:].copy()

    except Exception as error:
        raise type(error)(
            str(error)
            + " (raised in period %s during loop %s for forecast %s steps ahead)"
            % (i, loop, t)
        ).with_traceback(sys.exc_info()[2])

    fin_flag[1] &= not fin_flag[0]
    mess = [i * bool(j) for i, j in zip(msgs, fin_flag)]
    fin_flag = 2 ** np.arange(5) @ fin_flag

    if verbose:
        duration = np.round(time.time() - st, 3)
        print("(find_path:) Pizza done after %s seconds%s." % (duration, "".join(mess)))

    return x_fin, x_lin, fin_flag
