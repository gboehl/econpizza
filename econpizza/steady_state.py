#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import numpy as np
import jax.numpy as jnp
from scipy.linalg import block_diag
from grgrlib import klein, speed_kills
from grgrlib.jaxed import newton_jax, value_and_jac
from .shooting import solve_current
from .utilities.function_builders import get_func_stst_raw


# use a solver that can deal with ill-conditioned jacobians
def solver(jval, fval):
    return jax.numpy.linalg.pinv(jval) @ fval


def solve_stst(model, raise_error=True, tol=1e-8, maxit_newton=30, tol_backwards=None, maxit_backwards=1000, tol_forwards=None, maxit_forwards=1000, force=False, verbose=True):
    """Solves for the steady state.
    """

    st = time.time()

    evars = model["variables"]
    func_pre_stst = model['context']["func_pre_stst"]
    par = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    tol_backwards = tol if tol_backwards is None else tol_backwards
    tol_forwards = tol if tol_forwards is None else tol_forwards

    # check if steady state was already calculated
    try:
        cond0 = np.allclose(model["stst_used_pars"], par)
        cond1 = model["stst_used_setup"] == (
            model.get('functions_file_plain'), tol, maxit_backwards)
        if cond0 and cond1 and not force:
            if verbose:
                print("(solve_stst:) Steady state already known.")

            return model['stst_used_res']
    except KeyError:
        pass

    func_eqns = model['context']['func_eqns']
    func_backw = model['context'].get('func_backw')
    func_stst_dist = model['context'].get('func_stst_dist')

    decisions_output_init = model['init_run'].get('decisions_output')
    init_vf = model.get('init_vf')

    func_stst_raw, func_backw_ext = get_func_stst_raw(
        par, func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, tol_backw=tol_backwards, maxit_backw=maxit_backwards, tol_forw=tol_forwards, maxit_forw=maxit_forwards)

    # define jitted stst function that returns jacobian and func. value
    func_stst = value_and_jac(jax.jit(func_stst_raw))

    # actual root finding
    res = newton_jax(func_stst, model['init'], None, maxit_newton, tol,
                     sparse=False, func_returns_jac=True, solver=solver, verbose=verbose)

    # exchange those values that are identified via stst_equations
    stst_vals = func_pre_stst(res['x'][:len(evars)], par)

    rdict = dict(zip(evars, stst_vals))
    model["stst"] = rdict
    model["init"] = stst_vals
    model["stst_used_pars"] = par
    model["stst_used_setup"] = model.get(
        'functions_file_plain'), tol, maxit_backwards
    model["stst_used_res"] = res

    mess = ''

    if func_stst_dist:
        vfSS, decisions_output, cnt_backwards = func_backw_ext(stst_vals)
        distSS, cnt_forwards = func_stst_dist(
            decisions_output, tol_forwards, maxit_forwards)
        if cnt_backwards == maxit_backwards:
            mess += f'Maximum of {maxit_backwards} backwards calls reached. '
        if cnt_forwards == maxit_forwards:
            mess += f'Maximum of {maxit_forwards} forwards calls reached. '
        # TODO: this should loop over the objects in distSS/vfSS and store under the name of the distribution/decisions (i.e. 'D' or 'Va')
        model["distributions"]['stst'] = distSS
        model['decisions']['stst'] = vfSS

    # calculate error
    err = jnp.abs(func_stst(jnp.array(stst_vals))[0]).max()

    if err > tol:
        _, jac = func_stst(stst_vals)
        rank = jnp.linalg.matrix_rank(jac)
        df0 = sum(jnp.all(jnp.isclose(jac, 0), 0))
        df1 = sum(jnp.all(jnp.isclose(jac, 0), 1))
        mess += f"Function has rank {rank} ({jac.shape[0]} variables) and {df0} vs {df1} degrees of freedom. "
        if raise_error and not res["success"]:
            print(res)
            raise Exception(
                f"Steady state not found (error is {err:1.2e}). {mess}The root finding result is given above."
            )
        else:
            print(
                f"(solve_stst:) Steady state error is {err:1.2e}. {mess}"
            )
    elif verbose:
        duration = time.time() - st
        print(
            f"(solve_stst:) {mess}Steady state found in {duration:1.5g} seconds.")

    return res


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
    func = model['context']["func_eqns"]
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
        print(f"(solve_linear:) {mess} {'' if mess[-1] in '.?!' else '.'}")

    return model["ABC"]
