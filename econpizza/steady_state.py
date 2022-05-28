#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
from scipy.linalg import block_diag
from grgrlib import klein, speed_kills
from grgrlib.jaxed import newton_jax, value_and_jac
from .parser.build_functions import get_func_stst_raw


# use a solver that can deal with ill-conditioned jacobians
def solver(jval, fval):
    return jnp.linalg.pinv(jval) @ fval


def solve_stst(model, tol_newton=1e-8, maxit_newton=30, tol_backwards=None, maxit_backwards=2000, tol_forwards=None, maxit_forwards=5000, force=False, verbose=True, **newton_kwargs):
    """Solves for the steady state.
    """

    st = time.time()

    evars = model["variables"]
    func_pre_stst = model['context']["func_pre_stst"]
    par = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    tol_backwards = tol_newton if tol_backwards is None else tol_backwards
    tol_forwards = 1e-2*tol_newton if tol_forwards is None else tol_forwards

    # check if steady state was already calculated
    try:
        cond0 = jnp.allclose(model["stst_used_pars"], par)
        cond1 = model["stst_used_setup"] == (
            model.get('functions_file_plain'), tol_newton, maxit_newton, tol_backwards, maxit_backwards, tol_forwards, maxit_forwards)
        if cond0 and cond1 and not force:
            if verbose:
                print("(solve_stst:) Steady state already known.")

            return model['stst_used_res']
    except KeyError:
        pass

    # get all necessary functions
    func_eqns = model['context']['func_eqns']
    func_backw = model['context'].get('func_backw')
    func_stst_dist = model['context'].get('func_stst_dist')

    # get initial values for heterogenous agents
    decisions_output_init = model['init_run'].get('decisions_output')
    exog_grid_vars_init = model['init_run'].get('exog_grid_vars')
    init_vf = model.get('init_vf')

    # get the actual steady state function
    func_stst_raw, func_backw_ext = get_func_stst_raw(
        par, func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, exog_grid_vars_init, tol_backw=tol_backwards, maxit_backw=maxit_backwards, tol_forw=tol_forwards, maxit_forw=maxit_forwards)

    # define jitted stst function that returns jacobian and func. value
    func_stst = value_and_jac(jax.jit(func_stst_raw))

    # actual root finding
    res = newton_jax(func_stst, model['init'], None, maxit_newton, tol_newton, sparse=False,
                     func_returns_jac=True, solver=solver, verbose=verbose, **newton_kwargs)

    # exchange those values that are identified via stst_equations
    stst_vals = func_pre_stst(res['x'][:len(evars)], par)

    rdict = dict(zip(evars, stst_vals))
    model["stst"] = rdict
    model["stst_used_pars"] = par
    model["stst_used_setup"] = model.get(
        'functions_file_plain'), tol_newton, maxit_newton, tol_backwards, maxit_backwards, tol_forwards, maxit_forwards
    model["stst_used_res"] = res

    mess = ''

    if func_stst_dist:
        vfSS, decisions_output, exog_grid_vars, cnt_backwards = func_backw_ext(
            stst_vals)
        distSS, cnt_forwards = func_stst_dist(
            decisions_output, tol_forwards, maxit_forwards)
        if jnp.isnan(jnp.array(vfSS)).any() or jnp.isnan(jnp.array(decisions_output)).any():
            mess += f"Backward iteration returns 'NaN's. "
        elif jnp.isnan(distSS).any():
            mess += f"Forward iteration returns 'NaN's. "
        if cnt_backwards == maxit_backwards:
            mess += f'Maximum of {maxit_backwards} backwards calls reached. '
        if cnt_forwards == maxit_forwards:
            mess += f'Maximum of {maxit_forwards} forwards calls reached. '
        # TODO: this should loop over the objects in distSS/vfSS and store under the name of the distribution/decisions (i.e. 'D' or 'Va')
        model['steady_state']["distributions"] = distSS
        model['steady_state']['decisions'] = vfSS

    # calculate error
    err = jnp.abs(res['fun']).max()

    if err > tol_newton or not res['success']:
        jac = res['jac']
        rank = jnp.linalg.matrix_rank(jac)
        if rank:
            mess += f"Jacobian has rank {rank} for {jac.shape[0]} variables ({len(model['steady_state']['fixed_evalued'])} fixed). "
        if not res["success"]:
            mess = f"Steady state not found (error is {err:1.2e}). {res['message']} {mess}"
        else:
            mess = f"Steady state error is {err:1.2e}. {res['message']} {mess}"
    elif verbose:
        duration = time.time() - st
        mess += f"Steady state found in {duration:1.5g} seconds. {res['message']}"

    if mess:
        print(f"(solve_stst:) {mess}")

    return res
