#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import numpy as np
from scipy.linalg import block_diag
from grgrlib import klein, speed_kills
from grgrlib.jaxed import newton_jax
from .shooting import solve_current


def solve_stst(model, raise_error=True, tol=1e-8, maxit=30, verbose=True):
    """Solves for the steady state.
    """

    st = time.time()

    evars = model["variables"]
    func = model["func"]
    par = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    # find stst
    func_stst = jax.jit(
        lambda x: func(x, x, x, x, jax.numpy.zeros(len(shocks)), par, True)
    )

    res = newton_jax(func_stst, model['init'], None,
                     maxit, tol, sparse=False, verbose=False)

    # exchange those values that are identified via stst_equations
    stst_vals = func(res['x'], res['x'], res['x'], res['x'],
                     np.zeros(len(shocks)), par, True, True)
    # calculate error
    err = np.abs(func_stst(stst_vals)).max()

    if err > tol:
        grad = jax.jacfwd(func_stst)(model["init"])
        rank = np.linalg.matrix_rank(grad)
        df0 = sum(np.all(np.isclose(grad, 0), 0))
        df1 = sum(np.all(np.isclose(grad, 0), 1))
        mess = "Function has rank %s (%s variables) and %s vs %s degrees of freedom." % (
            rank,
            grad.shape[0],
            df0,
            df1,
        )
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
        duration = np.round(time.time() - st, 3)
        print(f"(solve_stst:) Steady state found in {duration} seconds.")

    rdict = dict(zip(evars, stst_vals))
    model["stst"] = rdict
    model["init"] = np.array(list(rdict.values()))
    model["stst_vals"] = np.array(list(rdict.values()))

    return model['stst']


def solve_linear(
    model,
    x=None,
    eps=1e-5,
    raise_error=True,
    check_contraction=False,
    lti_max_iter=1000,
    verbose=True,
):
    """Does half-way SGU, solves the model using linear time iteration and uses Klein's method to check for determinancy if the solution fails"""

    evars = model["variables"]
    func = model["func"]
    par = np.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()
    stst = list(model["stst"].values())
    nshc = len(shocks)
    nsts = len(stst)

    if x is None:
        x = np.array(stst)

    xmult = x.copy()
    xmult[np.isclose(x, 0)] = 1

    AA = np.empty((nsts, nsts))
    BB = AA.copy()
    CC = AA.copy()
    DD = np.empty((nsts, nshc))

    zshock = np.zeros(len(shocks))
    fx = func(x, x, x, x, np.zeros(nshc), par)

    # use numdifftools if possible
    AA = jax.jacfwd(lambda err: func(x, x, err, x, zshock, par))(x) * xmult
    BB = jax.jacfwd(lambda err: func(x, err, x, x, zshock, par))(x) * xmult
    CC = jax.jacfwd(lambda err: func(err, x, x, x, zshock, par))(x) * xmult
    DD = jax.jacfwd(lambda err: func(x, x, x, x, err, par))(zshock)
    DD = DD.reshape((nsts, len(shocks)))

    A = np.pad(AA, ((0, nshc), (0, nshc)))
    B = block_diag(BB, np.eye(nshc))
    C = np.block([[CC, DD], [np.zeros((nshc, A.shape[1]))]])

    model["ABC"] = A, B, C

    I = np.eye(nsts + nshc)
    Z = np.zeros_like(I)
    P = np.block([[B, A], [I, Z]])
    M = np.block([[C, Z], [Z, -I]])

    mess = ""
    success = True

    try:
        try:
            lam = -speed_kills(
                P, M, nsts + nshc, max_iter=lti_max_iter, verbose=verbose - 1
            )[1]

        except:
            _, lam = klein(P, M, nsts + nshc, verbose=verbose - 1)

        model["lin_pol"] = -lam[:nsts, :nsts], -lam[:nsts, nsts:]
        mess = "All eigenvalues are good"

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

    return model["ABC"]
