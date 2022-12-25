#!/bin/python
# -*- coding: utf-8 -*-

import os
import jax
import time
import jax.numpy as jnp
import scipy.sparse as ssp
from grgrlib.jaxed import *
from ..parser.build_functions import *
from ..utilities.jacobian import get_jac
from ..utilities.newton import newton


def find_path_stacking(
    model,
    x0=None,
    shock=None,
    horizon=250,
    endpoint=None,
    tol=None,
    maxit=None,
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

    # get initial guess
    x_stst = jnp.ones((horizon + 1, nvars)) * stst
    x0 = jnp.array(list(x0)) if x0 is not None else stst
    x_init = x_stst.at[0].set(x0)

    # set terminal condition
    xT = endpoint if endpoint is not None else stst

    # deal with shocks if any
    zshock = jnp.zeros(len(shocks))
    tshock = jnp.copy(zshock)
    if shock is not None:
        tshock = tshock.at[shocks.index(shock[0])].set(shock[1])
        if model.get('distributions'):
            print("(find_stack:) Warning: shocks for heterogenous agent models are not yet fully supported. Use adjusted steady state values as x0 instead.")

    if model['new_model_flag']:
        derivatives = compile_functions(
            model, zshock, horizon, nvars, pars, stst, x_stst)

        get_jac(model, derivatives, model.get('distributions'), horizon, nvars)
        model['new_model_flag'] = False

    # TODO: jvp should also alow for the terminal point
    jac, jvp = model['jac'], model['jvp']
    jvp_partial = jax.tree_util.Partial(jvp, x0=x0, xT=xT)
    x, err = newton(jvp_partial, jac, x_init, **newton_args)

    # calculate error
    x_out = x_init.at[1:-1].set(x.reshape((horizon - 1, nvars)))
    # mess = res['message']
    # TODO: create messages
    mess = ''

    # compile error/report message
    # TODO: create success criterion
    success = True
    if err > tol or not success:
        mess += f" Max. error is {err:1.2e}."
        verbose = True

    if verbose:
        duration = time.time() - st
        sucess = 'done' if success else 'FAILED'
        if not success and raise_errors:
            raise Exception(
                f"(find_path:) Stacking {sucess} after {duration:1.3f} seconds. " + mess)

        print(
            f"(find_path:) Stacking {sucess} after {duration:1.3f} seconds. " + mess)

    return x_out, not success
