#!/bin/python
# -*- coding: utf-8 -*-

import os
import jax
import time
import jax.numpy as jnp
import scipy.sparse as ssp
from grgrlib.jaxed import *
from ..parser.build_functions import *
from ..utilities.jacobian import get_jacobian
from ..utilities.newton import newton_for_jvp, newton_for_banded_jac


def find_path_stacking(
    model,
    x0=None,
    shock=None,
    horizon=300,
    endpoint=None,
    verbose=True,
    raise_errors=True,
    **newton_args
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
    endpoint : array, optional
        the final state at `horizon`. Defaults to the steay state if `None`
    verbose : bool, optional
        degree of verbosity. 0/`False` is silent
    raise_errors : bool, optional
        whether to raise errors as exceptions, or just inform about them. Defaults to `True`
    newton_args : optional
        any additional arguments to be passed on to the solver

    Returns
    -------
    x : array
        array of the trajectory
    flag : bool
        returns True if the solver was successful, else False
    """

    st = time.time()

    # get variables
    stst = jnp.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    pars = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()

    # get initial guess
    x_stst = jnp.ones((horizon + 1, nvars)) * stst
    x0 = jnp.array(list(x0)) if x0 is not None else stst
    x_init = x_stst.at[0].set(x0)

    # set terminal condition
    xT = endpoint if endpoint is not None else stst

    # deal with shocks if any
    zero_shocks = jnp.zeros((horizon-1, len(shocks)))
    if shock is not None:
        try:
            shock_series = zero_shocks.at[0,
                                          shocks.index(shock[0])].set(shock[1])
        except ValueError:
            raise ValueError(f"Shock '{shock[0]}' is not defined.")
    else:
        shock_series = zero_shocks

    if not model.get('distributions'):

        # get transition function
        func_eqns = model['context']["func_eqns"]

        def func_eqns_partial(xLag, x, xPrime, e_shock): return func_eqns(
            xLag, x, xPrime, stst, e_shock, pars, [], [])
        jav_func = jax.tree_util.Partial(
            jacrev_and_val(func_eqns_partial, (0, 1, 2)))

        # actual newton iterations
        x_out, flag, mess = newton_for_banded_jac(
            jav_func, nvars, horizon, x_init, shock_series, verbose, **newton_args)

    else:
        if model['new_model_horizon'] != horizon:
            # get derivatives via AD and compile functions
            derivatives = get_derivatives(
                model, nvars, pars, stst, x_stst, zero_shocks.T, horizon, verbose)

            # accumulate steady stat jacobian
            get_jacobian(model, derivatives, horizon, nvars, verbose)
            model['new_model_horizon'] = horizon

        # get jvp function and steady state jacobian
        jvp_partial = jax.tree_util.Partial(
            model['jvp'], x0=x0, xT=xT, shocks=shock_series.T)
        jacobian = model['jac']

        # actual newton iterations
        x, flag, mess = newton_for_jvp(
            jvp_partial, jacobian, x_init, verbose, **newton_args)
        x_out = x_init.at[1:-1].set(x.reshape((horizon - 1, nvars)))

    # some informative print messages
    if verbose:
        duration = time.time() - st
        result = 'done' if not flag else 'FAILED'
        mess = f"(find_path:) Stacking {result} ({duration:1.3f}s). " + mess
        if flag and raise_errors:
            raise Exception(mess)
        else:
            print(mess)

    return x_out, flag
