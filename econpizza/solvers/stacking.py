#!/bin/python
# -*- coding: utf-8 -*-

import os
import jax
import time
import jax.numpy as jnp
import scipy.sparse as ssp
from grgrlib.jaxed import newton_jax, jacfwd_and_val, jacrev_and_val
from ..parser.build_functions import *


def find_path_stacking(
    model,
    x0=None,
    shock=None,
    horizon=250,
    init_guess=None,
    endpoint=None,
    tol=None,
    maxit=None,
    verbose=True,
    raise_errors=True,
    **solver_kwargs,
):
    """Find the expected trajectory given an initial state.

    Parameters
    ----------
    model : dict
        model dict or PizzaModel instance
    x0 : array
        initial state
    shock : tuple, optional
        shock in period 0 as in `(shock_name_as_str, shock_size)`
    horizon : int, optional
        number of periods until the system is assumed to be back in the steady state. A good idea to set this corresponding to the respective problem. A too large value may be computationally expensive. A too small value may generate inaccurate results
    init_guess : array, optional
        a first guess on the trajectory. Not necessary in general
    endpoint : array, optional
        the final state at `horizon`. Defaults to the steay state if `None`
    tol : float, optional
        convergence criterion. Defaults to 1e-8
    maxit : int, optional
        number of iterations. Default is 30.
    use_jacrev : bool, optional
        whether to use reverse mode or forward mode automatic differentiation. By construction, reverse AD is faster, but does not work for all types of functions. Defaults to True
    verbose : bool, optional
        degree of verbosity. 0/`False` is silent
    raise_errors : bool, optional
        whether to raise errors as exceptions, or just inform about them. Defaults to `True`
    solver_kwargs : optional
        any additional keyword arguments will be passed on to the solver

    Returns
    -------
    x : array
        array of the trajectory
    flag : bool
        returns True if the solver was successful, else False
    """

    st = time.time()

    # set defaults
    tol = 1e-8 if tol is None else tol
    maxit = 30 if maxit is None else maxit

    # get variables
    stst = jnp.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    pars = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    # get functions
    func_eqns = model['context']["func_eqns"]
    func_backw = model['context'].get('func_backw')
    func_dist = model['context'].get('func_dist')

    # get initial guess
    x0 = jnp.array(list(x0)) if x0 is not None else stst
    x_init = jnp.ones((horizon + 1, nvars)) * stst
    x_init = x_init.at[0].set(x0)
    if init_guess is not None:
        x_init = x_init.at[1: len(init_guess)].set(init_guess[1:])

    # set terminal condition
    endpoint = endpoint if endpoint is not None else stst

    # deal with shocks if any
    zshock = jnp.zeros(len(shocks))
    tshock = jnp.copy(zshock)
    if shock is not None:
        tshock = tshock.at[shocks.index(shock[0])].set(shock[1])
        if model.get('distributions'):
            print("(find_stack:) Warning: shocks for heterogenous agent models are not yet fully supported. Use adjusted steady state values as x0 instead.")

    if model.get('distributions'):
        # get stuff for het-agent models
        vfSS = model['steady_state']['decisions']
        distSS = jnp.array(model['steady_state']['distributions'])
        decisions_outputSS = jnp.array(
            model['steady_state']['decisions_output'])

        # compile stacked function if distribution matters
        stacked_func_dist_raw = get_stacked_func_dist(
            pars, func_backw, func_dist, func_eqns, x0, stst, vfSS, distSS, zshock, horizon, nvars, endpoint)
        stacked_func = jax.jit(stacked_func_dist_raw,
                               static_argnames='full_output')
    else:
        # compile stacked function if independent of the distribution
        stacked_func = get_stacked_func_simple(
            pars, func_eqns, stst, x0, horizon, nvars, endpoint, zshock, tshock, shock, [], [])

    model['context']['stacked_func_dist'] = stacked_func
    if verbose:
        print("(find_stack:) Solving stack (size: %s)..." %
              (horizon*nvars))

    # find path
    res = newton_jax(stacked_func, x_init[1:-1].flatten(), None, maxit, tol, sparse=not model.get(
        'distributions'), func_returns_jac=True, verbose=verbose, **solver_kwargs)

    # calculate error
    err = jnp.abs(res['fun']).max()
    x = x_init.at[1:-1].set(res['x'].reshape((horizon - 1, nvars)))
    mess = res['message']

    # compile error/report message
    if err > tol or not res['success']:
        mess += f" Max. error is {err:1.2e}."
        verbose = True

    if verbose:
        duration = time.time() - st
        sucess = 'done' if res['success'] else 'FAILED'
        if not res['success'] and raise_errors:
            raise Exception(
                f"(find_path:) Stacking {sucess} after {duration:1.3f} seconds. " + mess)

        print(
            f"(find_path:) Stacking {sucess} after {duration:1.3f} seconds. " + mess)

    return x, not res['success']
