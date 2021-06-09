#!/bin/python
# -*- coding: utf-8 -*-

import yaml
import re
import warnings
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from numba import njit


def parse(mfile, verbose=True):

    f = open(mfile)
    mtxt = f.read()
    f.close()

    mtxt = mtxt.replace("^", "**")
    mtxt = re.sub(r"@ ?\n", " ", mtxt)
    model = yaml.safe_load(mtxt)

    evars = model["variables"]
    shocks = model.get("shocks") or ()
    par = model["parameters"]
    eqns = model["equations"]

    if len(evars) != len(eqns):
        raise Exception(
            "Model has %s variables but %s equations." % (len(evars), len(eqns))
        )

    for i, eqn in enumerate(eqns):
        if "=" in eqn:
            lhs, rhs = eqn.split("=")
            eqns[i] = "F[%s] = " % i + lhs + "- (" + rhs + ")"
        else:
            eqns[i] = "F[%s] = " % i + eqn

    eqns_aux = model.get("aux_equations")

    if not shocks:
        shock_str = ""
    elif len(shocks) > 1:
        shock_str = ", ".join(shocks) + " = shocks"
    else:
        shock_str = shocks[0] + " = shocks[0]"

    func_str = """def func_raw(XLag, X, XPrime, XSS, shocks, pars):\n %s\n %s\n %s\n %s\n %s\n %s\n F=np.empty(%s)\n %s\n %s\n return F""" % (
        ", ".join(v + "Lag" for v in evars) + " = XLag",
        ", ".join(evars) + " = X",
        ", ".join(v + "Prime" for v in evars) + " = XPrime",
        ", ".join(v + "SS" for v in evars) + " = XSS",
        shock_str,
        ", ".join(par.keys()) + " = pars",
        str(len(evars)),
        "\n ".join(eqns_aux) if eqns_aux else "",
        "\n ".join(eqns),
    )

    try:
        exec(func_str, globals())
        func = njit(func_raw)
    except Exception as error:
        raise type(error)(
            str(error)
            + "\n\n This is the transition function as I tried to compile it:\n\n"
            + func_str
        )

    model["func"] = func
    model["func_str"] = func_str
    solve_stst(model, raise_error=False, verbose=verbose)

    if verbose:
        print("Parsing done.")

    return model


def solve_stst(model, raise_error=True, verbose=True):

    evars = model["variables"]
    func = model["func"]
    par = model["parameters"]
    inits = model["steady_state"].get("init_guesses")
    shocks = model.get("shocks") or ()

    stst = model["steady_state"].get("fixed_values")
    for k in stst:
        if isinstance(stst[k], str):
            stst[k] = eval(stst[k])

    model["stst"] = stst

    def func_stst(x):

        xss = ()
        for i, v in enumerate(evars):
            if v in stst:
                xss += (stst[v],)
            else:
                xss += (x[i],)

        XSS = np.array(xss)

        return func(
            XSS, XSS, XSS, XSS, np.zeros(len(shocks)), np.array(list(par.values()))
        )

    init = ()
    for v in evars:

        ininit = False
        if isinstance(inits, dict):
            if v in inits.keys():
                ininit = True

        if v in stst.keys():
            init += (stst[v],)
        elif ininit:
            init += (inits[v],)
        else:
            init += (1.0,)

    res = so.root(func_stst, init)

    if not res["success"] or np.any(np.abs(func_stst(res["x"])) > 1e-8):
        if raise_error:
            raise Exception(
                "Steady state not found. Root finding reports:\n\n" + str(res)
            )
        else:
            warnings.warn("Steady state not found", RuntimeWarning)
    elif verbose:
        print("Steady state found.")

    for v in stst:
        res["x"][evars.index(v)] = stst[v]

    rdict = dict(zip(evars, res["x"]))
    model["stst"] = rdict
    model["stst_vals"] = np.array(list(rdict.values()))

    return rdict


def solve_current(model, XLag, XPrime, tol):

    func = model["func"]
    par = model["parameters"]
    stst = model.get("stst")
    shocks = model.get("shocks") or ()

    def func_current(x):
        return func(
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
    max_horizon=500,
    max_iter=None,
    tol=1e-5,
    root_options=None,
    debug=False,
    verbose=True,
):

    if max_iter is None:
        max_iter = max_horizon

    stst = list(model["stst"].values())
    evars = model["variables"]

    if root_options is None:
        root_options = {}

    # precision of root finding should be some magnitudes higher than of solver
    if "xtol" not in root_options:
        root_options["xtol"] = max(tol * 1e-3, 1e-8)

    model["root_options"] = root_options

    x_fin = np.empty((T + 1, len(evars)))
    x_fin[0] = list(x0)

    x = np.ones((T + max_horizon + 1, len(evars))) * np.array(stst)
    x[0] = list(x0)

    if init_path is not None:
        x[1 : len(init_path)] = init_path[1:]

    fin_flag = np.zeros(5, dtype=bool)

    try:
        for i in range(T):

            cnt = 2
            flag = np.zeros(5, dtype=bool)

            while True:

                x_old = x[1].copy()
                imax = min(cnt, max_horizon)

                for t in range(imax):
                    x[t + 1], flag_root, flag_ftol = solve_current(
                        model, x[t], x[t + 2], tol
                    )

                flag[0] |= cnt == flag_root
                flag[1] |= np.any(np.isnan(x))
                flag[2] |= np.any(np.isinf(x))
                flag[3] |= cnt == flag_ftol
                flag[4] |= cnt == max_iter

                fin_flag |= flag
                err = np.abs(x_old - x[1]).max()

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
    mess = [i * bool(j) for i, j in zip(msgs, flag)]
    fin_flag = 2 ** np.arange(5) @ fin_flag

    if verbose:
        print("Pizza done%s." % "".join(mess))

    return x_fin, fin_flag
