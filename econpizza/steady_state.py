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
    """Does half-way SGU and uses Klein's method to check for determinancy"""

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

    model['ABC'] = AA, BB, CC

    I = np.eye(len(stst))
    Z = np.zeros_like(I)
    A = np.block([[AA, BB], [Z, I]])
    B = np.block([[Z, CC], [-I, Z]])

    try:
        from grgrlib import klein

        model['omg'], model['lam'] = klein(A, B, len(stst))
        if verbose:
            print("All eigenvalues are good.")
        return True

    except ImportError:
        if verbose:
            print("'grgrlib' not found, could not check eigenvalues.")
        return False

    except Exception as error:
        if raise_error:
            raise error
        else:
            print(str(error))
            return False
