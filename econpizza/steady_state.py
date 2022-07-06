#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
from grgrlib.jaxed import newton_jax, jacfwd_and_val
from .parser.build_functions import get_func_stst_raw


# use a solver that can deal with ill-conditioned jacobians
def solver(jval, fval):
    """A default solver to solve indetermined problems.
    """
    return jnp.linalg.pinv(jval) @ fval


def solve_stst(model, tol=1e-8, tol_newton=None, maxit_newton=30, tol_backwards=None, maxit_backwards=2000, tol_forwards=None, maxit_forwards=5000, force=False, verbose=True, **newton_kwargs):
    """Solves for the steady state.

    Parameters
    ----------
    tol_newton : float, optional
        tolerance of the Newton method, defaults to 1e-8
    maxit_newton : int, optional
        maximum of iterations for the Newton method, defaults to 30
    tol_backwards : float, optional
        tolerance required for backward iteration. Defaults to tol_newton
    maxit_backwards : int, optional
        maximum of iterations for the backward iteration. Defaults to maxit_newton
    tol_forwards : float, optional
        tolerance required for forward iteration. Defaults to tol_newton
    maxit_forwards : int, optional
        maximum of iterations for the forward iteration. Defaults to maxit_newton
    force : bool, optional
        force recalculation of steady state, even if it is already evaluated. Defaults to False
    verbose : bool, optional
        level of verbosity. Defaults to True
    newton_kwargs : keyword arguments
        keyword arguments passed on to the Newton method

    Returns
    -------
    res : dict
        results dictionary from the Newton method
    """

    st = time.time()

    evars = model["variables"]
    func_pre_stst = model['context']["func_pre_stst"]
    par = model["parameters"]
    shocks = model.get("shocks") or ()

    tol_newton = tol if tol_newton is None else tol_newton
    tol_backwards = tol if tol_backwards is None else tol_backwards
    tol_forwards = 1e-2*tol if tol_forwards is None else tol_forwards

    # check if steady state was already calculated
    try:
        cond0 = jnp.allclose(model["stst_used_pars"], jnp.array(
            list(model['steady_state']['fixed_evalued'].values())))
        cond1 = model["stst_used_setup"] == (
            model.get('functions_file_plain'), tol_newton, maxit_newton, tol_backwards, maxit_backwards, tol_forwards, maxit_forwards)
        if cond0 and cond1 and not force:
            if verbose:
                print(
                    f"(solve_stst:) Steady state already {'known' if model['stst_used_success'] else 'failed'}.")

            return model['stst_used_res']
    except KeyError:
        pass

    # reset for recalculation
    model['stst_jacobian'] = None
    model['stacked_func'] = None

    # get all necessary functions
    func_eqns = model['context']['func_eqns']
    func_backw = model['context'].get('func_backw')
    func_stst_dist = model['context'].get('func_stst_dist')

    # get initial values for heterogenous agents
    decisions_output_init = model['init_run'].get('decisions_output')
    exog_grid_vars_init = model['init_run'].get('exog_grid_vars')
    init_vf = model.get('init_vf')

    # get the actual steady state function
    func_stst_raw = get_func_stst_raw(func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init,
                                      exog_grid_vars_init, tol_backw=tol_backwards, maxit_backw=maxit_backwards, tol_forw=tol_forwards, maxit_forw=maxit_forwards)

    # define jitted stst function that returns jacobian and func. value
    func_stst = jacfwd_and_val(jax.jit(func_stst_raw))
    # store functions
    model["context"]['func_stst_raw'] = func_stst_raw
    model["context"]['func_stst'] = func_stst

    # actual root finding
    res = newton_jax(func_stst, jnp.array(list(model['init'].values())), None, maxit_newton, tol_newton, sparse=False,
                     func_returns_jac=True, solver=solver, verbose=verbose, **newton_kwargs)

    # exchange those values that are identified via stst_equations
    stst_vals, par_vals = func_pre_stst(res['x'])

    model["stst"] = dict(zip(evars, stst_vals))
    model["parameters"] = dict(zip(par, par_vals))
    model["stst_used_pars"] = jnp.array(
        list(model['steady_state']['fixed_evalued'].values()))
    model["stst_used_setup"] = model.get(
        'functions_file_plain'), tol_newton, maxit_newton, tol_backwards, maxit_backwards, tol_forwards, maxit_forwards
    model["stst_used_res"] = res
    model["stst_used_success"] = res['success']

    mess = ''

    if model.get('distributions'):
        # TODO: loosing some time here
        res_backw, res_forw = func_stst_raw(res['x'], True)
        vfSS, decisions_output, exog_grid_vars, cnt_backwards = res_backw
        distSS, cnt_forwards = res_forw
        if jnp.isnan(jnp.array(vfSS)).any() or jnp.isnan(jnp.array(decisions_output)).any():
            mess += f"Backward iteration returns 'NaN's. "
        elif jnp.isnan(distSS).any():
            mess += f"Forward iteration returns 'NaN's. "
        elif distSS.min() < 0:
            mess += f"Distribution contains negative values. "
        if cnt_backwards == maxit_backwards:
            mess += f'Maximum of {maxit_backwards} backwards calls reached. '
        if cnt_forwards == maxit_forwards:
            mess += f'Maximum of {maxit_forwards} forwards calls reached. '
        # TODO: this should loop over the objects in distSS/vfSS and store under the name of the distribution/decisions (i.e. 'D' or 'Va')
        model['steady_state']["distributions"] = distSS
        model['steady_state']['decisions'] = vfSS
        model['steady_state']['decisions_output'] = decisions_output

    # calculate error
    err = jnp.abs(res['fun']).max()

    if err > tol_newton or not res['success']:
        jac = res['jac']
        rank = jnp.linalg.matrix_rank(jac)
        if rank:
            nvars = len(evars)+len(par)
            nfixed = len(model['steady_state']['fixed_evalued'])
            mess += f"Jacobian has rank {rank} for {nvars - nfixed} degrees of freedom ({nvars} variables/parameters, {nfixed} fixed). "
        if not res["success"]:
            mess = f"Steady state FAILED (error is {err:1.2e}). {res['message']} {mess}"
        else:
            mess = f"{res['message']} WARNING: Steady state error is {err:1.2e}. {mess}"
    elif verbose:
        duration = time.time() - st
        mess = f"Steady state found in {duration:1.5g} seconds. {res['message']}" + (
            ' WARNING: ' + mess if mess else '')

    if mess:
        print(f"(solve_stst:) {mess}")

    return res
