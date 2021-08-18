#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as so
from scipy.linalg import block_diag
from .shooting import solve_current


def solve_stst(model, raise_error=True, method=None, tol=1e-8, verbose=True):

    evars = model["variables"]
    func = model["func"]
    par = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()
    stst = model["stst"]
    stst_np = np.array(list(stst.values()))
    method = method or model["steady_state"].get("method")

    if method not in ("reduction", "aux_function", None):
        raise NotImplementedError(
            "Steady state method must either be 'aux_function' or 'reduction' or `None`."
        )

    if method in ("aux_function", None):

        # create a sequence and ensure that its columns are linearly independent
        shifter_rand = np.arange(len(evars) * len(stst)).reshape(len(evars), len(stst))
        svd_u, _, svd_v = np.linalg.svd(shifter_rand, full_matrices=False)
        shifter = svd_u @ svd_v
        x_ind = np.array([evars.index(v) for v in stst.keys()])

        stst_no_zero = stst_np.copy()
        stst_no_zero[np.isclose(stst_np, 0)] = 1

        def func_stst(x):

            # use the sequence to force root finding to set fixed st.st values
            corr = (x[x_ind] - stst_np) / stst_no_zero

            return func(x, x, x, x, np.zeros(len(shocks)), par, True) + shifter @ corr

        # find stst
        res = so.root(func_stst, model["init"])
        err = np.abs(func_stst(res["x"])).max()
        mess = " ".join(res["message"].replace("\n", " ").split())

    if method == "reduction":

        # create a sequence and ensure that its columns are linearly independent
        shifter_rand = np.ones((len(evars) - len(stst), len(evars)))
        svd_u, _, svd_v = np.linalg.svd(shifter_rand, full_matrices=False)
        shifter = svd_u @ svd_v

        x_ind = np.array([stst.get(v) is not None for v in evars])
        x = np.empty(len(evars))

        def func_stst(x_guess):

            x[x_ind] = stst_np
            x[~x_ind] = x_guess

            return shifter @ func(x, x, x, x, np.zeros(len(shocks)), par, True)

        # find stst
        res = so.root(func_stst, model["init"][~x_ind])
        err = np.abs(func_stst(res["x"])).max()
        mess = " ".join(res["message"].replace("\n", " ").split())

    if err > tol:
        if raise_error and not res["success"]:
            print(res)
            raise Exception(
                "Steady state not found (error is %1.2e). %s See the root finding result above."
                % (err, mess)
            )
        else:
            print(
                "(solve_stst:) Steady state not found (error is %1.2e). %s"
                % (err, mess)
            )
    elif verbose:
        print("(solve_stst:) Steady state found.")

    rdict = dict(zip(evars, res["x"]))
    model["stst"] = rdict
    model["init"] = np.array(list(rdict.values()))
    model["stst_vals"] = np.array(list(rdict.values()))

    return rdict


def solve_linear(
    model,
    x=None,
    eps=1e-5,
    tol=1e-8,
    raise_error=True,
    check_contraction=False,
    lti_max_iter=1000,
    verbose=True,
):
    """Does half-way SGU and uses Klein's method to check for determinancy and solve the system"""

    evars = model["variables"]
    func = model["func"]
    par = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()
    stst = list(model["stst"].values())
    nshc = len(shocks)

    if x is None:
        x = np.array(stst)

    AA = np.empty((len(stst), len(stst)))
    BB = AA.copy()
    CC = AA.copy()
    DD = np.empty((len(stst), nshc))

    zshock = np.zeros(len(shocks))
    fx = func(x, x, x, x, np.zeros(nshc), par)

    for i in range(len(evars)):

        xerr = x.copy()
        xerr[i] -= eps

        AA[:, i] = (func(x, x, xerr, x, zshock, par) - fx) * x[i] / eps
        BB[:, i] = (func(x, xerr, x, x, zshock, par) - fx) * x[i] / eps
        CC[:, i] = (func(xerr, x, x, x, zshock, par) - fx) * x[i] / eps

    for i in range(len(shocks)):
        cshock = zshock.copy()
        cshock[i] += eps

        DD[:, i] = (func(x, x, x, x, cshock, par) - fx) / eps

    A = np.pad(AA, ((0, nshc), (0, nshc)))
    B = block_diag(BB, np.eye(nshc))
    C = np.block([[CC, DD], [np.zeros((nshc, A.shape[1]))]])

    model["ABC"] = A, B, C

    I = np.eye(len(stst) + nshc)
    Z = np.zeros_like(I)
    P = np.block([[B, A], [I, Z]])
    M = np.block([[C, Z], [Z, -I]])

    mess = ""
    success = True

    try:
        from grgrlib import klein

        _, lam = klein(P, M, len(stst) + nshc, verbose=verbose - 1)
        model["lin_pol"] = -lam[:-nshc, :-nshc], -lam[:-nshc, -nshc:]
        mess = "All eigenvalues are good"

    except ImportError:
        mess = "'grgrlib' not found, could not check eigenvalues"

    except Exception as error:
        success = False
        if raise_error:
            raise error
        else:
            mess = str(error).strip()
            if mess[-1] == ".":
                mess = mess[:-1]

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
        print("(solve_linear:) " + mess + ("" if mess[-1] in ".?!" else "."))

    return success
