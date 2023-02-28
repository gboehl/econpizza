# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
from ..utilities.jacobian import get_stst_jacobian
from ..parser.checks import check_if_compiled, write_cache
from ..parser.build_functions import build_aggr_het_agent_funcs, get_stst_derivatives


def find_path_linear(model, shock=None, init_state=None, pars=None, horizon=200, verbose=True):
    """Find the linear expected trajectory given an initial state.

    Parameters
    ----------
    model : PizzaModel
        PizzaModel instance
    init_state : array
        initial state
    pars : dict, optional
        alternative parameters. Warning: do only change those parameters that are invariant to the steady state.
    shock : tuple, optional
        shock in period 0 as in `(shock_name_as_str, shock_size)`. NOTE: Not (yet) implemented.
    horizon : int, optional
        number of periods to simulate
    verbose : bool, optional
        degree of verbosity. 0/`False` is silent

    Returns
    -------
    x : array
        array of the trajectory
    flag : bool
        for consistency. Always returns `True`
    """

    if shock is not None:
        raise NotImplementedError(
            "Shocks are not (yet) implemented for the linear solution.")

    if not model.get('distributions'):
        raise NotImplementedError(
            "Models without heterogeneous agents are not (yet) implemented for the linear solution.")

    st = time.time()

    # get model variables
    stst = jnp.array(list(model['stst'].values()))
    nvars = len(model["variables"])
    pars = jnp.array(
        list((pars if pars is not None else model["pars"]).values()))
    shocks = model.get("shocks") or ()
    x_stst = jnp.ones((horizon + 1, nvars)) * stst

    # deal with shocks
    zero_shocks = jnp.zeros((horizon-1, len(shocks))).T

    x0 = jnp.array(list(init_state)) if init_state is not None else stst

    if not check_if_compiled(model, horizon, pars):
        # get derivatives via AD and compile functions
        build_aggr_het_agent_funcs(
            model, nvars, pars, stst, zero_shocks, horizon)
        derivatives = get_stst_derivatives(
            model, nvars, pars, stst, x_stst, zero_shocks, horizon, verbose)

        # accumulate steady stat jacobian
        get_stst_jacobian(model, derivatives, horizon, nvars, verbose)
        write_cache(model, horizon, pars)

    jacobian = model['jac']

    x0 -= stst
    x = - \
        jax.scipy.linalg.solve(
            jacobian[nvars:, nvars:], jacobian[nvars:, :nvars] @ x0)
    x = jnp.hstack((x0, x)).reshape(-1, nvars) + stst

    if verbose:
        duration = time.time() - st
        print(
            f"(find_path_linear:) Linear solution done ({duration:1.3f}s).")

    return x, True
