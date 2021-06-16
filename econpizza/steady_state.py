#!/bin/python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import scipy.optimize as so
from .shooting import solve_current


def solve_stst(model, raise_error=True, tol=1e-8, verbose=True):

    evars = model["variables"]
    func = model["func"]
    par = np.array(list(model["parameters"].values()))
    inits = model["steady_state"].get("init_guesses")
    shocks = model.get("shocks") or ()

    stst = model["steady_state"].get("fixed_values")
    for k in stst:
        if isinstance(stst[k], str):
            stst[k] = eval(stst[k])

    model["stst"] = stst

    # draw a random sequence and hope that its columns are linearly independent
    np.random.seed(0)
    shifter = np.random.normal(size=(len(evars), len(stst)))

    def func_stst(x):

        # use the random sequence to force root finding to set st.st values
        corr = [x[evars.index(v)] - stst[v] for i, v in enumerate(stst)]

        return func(x, x, x, x, np.zeros(len(shocks)), par) + shifter @ corr

    init = np.ones(len(evars)) * 1.1

    if isinstance(inits, dict):
        for v in inits:
            init[evars.index(v)] = inits[v]

    for v in stst:
        init[evars.index(v)] = stst[v]

    res = so.root(func_stst, init)
    err = np.abs(func_stst(res["x"])).max()

    if err > tol:
        if raise_error:
            print(res)
            raise Exception(
                "Steady state not found (error is %1.2e). See the root finding result above. Be aware that the root finding report might be missleading because fixed st.st. values are overwriting the guess."
                % err
            )
        else:
            warnings.warn("Steady state not found", RuntimeWarning)
    elif verbose:
        print("Steady state found.")

    rdict = dict(zip(evars, res["x"]))
    model["stst"] = rdict
    model["stst_vals"] = np.array(list(rdict.values()))

    return rdict


def check_evs(model, x=None, eps=1e-5, tol=1e-20, raise_error=True, verbose=True):

    stst = list(model["stst"].values())

    if x is None:
        x = np.array(stst)

    AA = np.empty((len(stst), len(stst)))
    CC = AA.copy()

    for i in range(len(stst)):
        XLag = x.copy()
        XPrime = x.copy()
        XLag[i] += eps
        XPrime[i] += eps
        # AA[:, i] = (solve_current(model, x, XPrime, tol)[0] - x) / eps
        # CC[:, i] = (solve_current(model, XLag, x, tol)[0] - x) / eps
        AA[i, :] = (solve_current(model, x, XPrime, tol)[0] - x) / eps
        CC[i, :] = (solve_current(model, XLag, x, tol)[0] - x) / eps

    I = np.eye(len(stst))
    Z = np.zeros_like(I)
    # A = np.block([[AA, I], [I, Z]])
    # B = np.block([[Z, CC], [-I, Z]])
    A = np.block([[I, AA], [Z, Z]])
    B = np.block([[Z, Z], [CC, I]])

    Aev = np.abs(np.linalg.eig(A)[0])
    Aev_err = Aev > 1
    Bev = np.abs(np.linalg.eig(B)[0])
    Bev_err = Bev > 1

    mess = ""
    if Aev_err.any():
        mess += "%s forward looking EV%s larger than unity (%s)" % (
            Aev_err.sum(),
            "s are" if Aev_err.sum() > 1 else " is",
            Aev[Aev_err],
        )
    if Aev_err.any() and Bev_err.any():
        mess += " and "
    if Bev_err.any():
        mess += "%s backward looking EV%s larger than unity (%s)" % (
            Bev_err.sum(),
            "s are" if Bev_err.sum() > 1 else " is",
            *Bev[Bev_err],
        )
    if mess:
        mess += "."
        if raise_error:
            raise Exception(mess)
        warnings.warn(mess)
        return False

    elif verbose:
        print("All eigenvalues are good.")

    return True
