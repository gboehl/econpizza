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


def solve_stst(model, raise_error=True, tol=1e-8, maxit=30, verbose=True):
    """Solves for the steady state.
    """

    st = time.time()

    evars = model["variables"]
    func = model["func"]
    func_pre_stst = model['context']["func_pre_stst"]
    par = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    func_backw_raw = model['context'].get('func_backw_raw')
    func_stst_dist = model['context'].get('func_stst_dist')
    if func_stst_dist:
        init_vf = model['steady_state']['init_guesses'][model['decisions']['inputs'][0]]

    # TODO: these two functions could be sourced out

    def func_backw_ext(x):

        def cond_func(cont):
            return jnp.abs(cont[0]-cont[1]).max() > tol

        def body_func(cont):
            vf, _ = cont
            return func_backw_raw(x, x, x, x, vf, [], par)[0], vf

        vf = jax.lax.while_loop(cond_func, body_func, (init_vf, init_vf+1))[0]
        vf, decisions_output = func_backw_raw(x, x, x, x, vf, [], par)

        return vf, decisions_output

    def func_stst_raw(x, return_vf_and_dist=False):

        x = func_pre_stst(x, par)[..., jnp.newaxis]

        if not func_stst_dist:
            return func(x, x, x, x, jax.numpy.zeros(len(shocks)), par)

        vf, decisions_output = func_backw_ext(x)
        dist = func_stst_dist(decisions_output)

        if return_vf_and_dist:
            return x, vf, dist

        # TODO: for more than one dist this should be a loop...
        decisions_output_array = jnp.array(decisions_output)[..., jnp.newaxis]
        dist_array = jnp.array(dist)[..., jnp.newaxis]
        return func(x, x, x, x, [], par, dist_array, decisions_output_array)

    # define jitted stst function that returns jacobian and func. value
    def func_stst(x): return value_and_jac(
        jax.jit(func_stst_raw, static_argnames='return_vf_and_dist'), x)

    # use a solver that can deal with ill-conditioned jacobians
    def solver(jval, fval): return jax.numpy.linalg.pinv(jval) @ fval

    # actual root finding
    res = newton_jax(func_stst, model['init'], None, maxit, tol,
                     sparse=False, func_returns_jac=True, solver=solver, verbose=verbose)

    # exchange those values that are identified via stst_equations
    stst_vals = func_pre_stst(res['x'][:len(evars)], par)

    rdict = dict(zip(evars, stst_vals))
    model["stst"] = rdict
    model["init"] = stst_vals

    if func_stst_dist:
        xSS, vfSS, distSS = func_stst_raw(stst_vals, return_vf_and_dist=True)
        # TODO: this should loop over the objects in distSS/vfSS and store under the name of the distribution/decisions (i.e. 'D' or 'Va')
        model["distributions"]['stst'] = distSS
        model['decisions']['stst'] = vfSS

    # calculate error
    err = jnp.abs(func_stst(jnp.array(stst_vals))[0]).max()

    if err > tol:
        grad = jax.jacfwd(func_stst)(model["init"])
        rank = jnp.linalg.matrix_rank(grad)
        df0 = sum(jnp.all(jnp.isclose(grad, 0), 0))
        df1 = sum(jnp.all(jnp.isclose(grad, 0), 1))
        mess = f"Function has rank {rank} ({grad.shape[0]} variables) and {df0} vs {df1} degrees of freedom."
        if raise_error and not res["success"]:
            print(res)
            raise Exception(
                f"Steady state not found (error is {err:1.2e}). {mess} The root finding result is given above."
            )
        else:
            print(
                f"(solve_stst:) Steady state error is {err:1.2e}. {mess}"
            )
    elif verbose:
        duration = time.time() - st
        print(f"(solve_stst:) Steady state found in {duration:1.5g} seconds.")

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
        print("(solve_linear:) {mess} {'' if mess[-1] in '.?!' else '.'}")

    return model["ABC"]
