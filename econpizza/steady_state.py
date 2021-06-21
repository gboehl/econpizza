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


def check_evs(
    model,
    x=None,
    eps=1e-5,
    tol=1e-8,
    raise_error=True,
    check_contraction=False,
    lti_max_iter=1000,
    verbose=True,
):
    """Does half-way SGU and uses LTI to check for determinancy"""

    evars = model["variables"]
    func = model["func"]
    par = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()
    stst = list(model["stst"].values())

    if x is None:
        x = np.array(stst)

    AA = np.empty((len(stst), len(stst)))
    BB = AA.copy()
    CC = AA.copy()

    fx = func(x, x, x, x, np.zeros(len(shocks)), par)

    for i in range(len(evars)):
        X = x.copy()
        X[i] += eps

        CC[:, i] = (func(X, x, x, x, np.zeros(len(shocks)), par) - fx) / eps
        BB[:, i] = (func(x, X, x, x, np.zeros(len(shocks)), par) - fx) / eps
        AA[:, i] = (func(x, x, X, x, np.zeros(len(shocks)), par) - fx) / eps

    model["ABC"] = AA, BB, CC

    I = np.eye(len(stst))
    Z = np.zeros_like(I)
    A0 = np.block([[BB, AA], [I, Z]])
    B0 = np.block([[CC, Z], [Z, -I]])

    mess = ""
    success = True

    try:
        from grgrlib import klein

        _, model["lin_pol"] = klein(A0, B0, len(stst))
        mess = "All eigenvalues are good."

    except ImportError:
        mess = "'grgrlib' not found, could not check eigenvalues."

    except Exception as error:
        success = False
        if raise_error:
            raise error
        else:
            mess = str(error)

    if check_contraction:
        A = np.linalg.inv(BB) @ AA
        B = np.linalg.inv(BB) @ CC

        Aev = np.abs(np.linalg.eig(A)[0])
        Aev_err = Aev > 1
        Bev = np.abs(np.linalg.eig(B)[0])
        Bev_err = Bev > 1
        flag += Aev_err.any() or Bev_err.any()
        mess += ", but " if success else ""
        if Aev_err.any():
            mess += "%s forward looking EV%s larger than unity (%s)" % (
                Aev_err.sum(),
                "s are" if Aev_err.sum() > 1 else " is",
                *Aev[Aev_err],
            )
        if Aev_err.any() and Bev_err.any():
            mess += " and "
        if Bev_err.any():
            mess += "%s backward looking EV%s larger than unity (%s)" % (
                Bev_err.sum(),
                "s are" if Bev_err.sum() > 1 else " is",
                *Bev[Bev_err],
            )

    if mess and verbose:
        print(mess + ".")

    return success
